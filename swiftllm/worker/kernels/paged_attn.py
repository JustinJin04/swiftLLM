import torch
import triton
import triton.language as tl

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.infer_state import LlamaInferState

@triton.jit
def _fwd_paged_attention_phase1(
    mid_o: torch.Tensor,	# [num_decoding_seqs*num_lookahead_tokens, num_q_heads, num_seq_chunks, head_dim], contiguous. num_seq_chunks = ceil(max_seq_len / seq_chunk_size)
    mid_o_logexpsum: torch.Tensor,	# [num_decoding_seqs*num_lookahead_tokens, num_q_heads, num_seq_chunks], contiguous
    q: torch.Tensor,    	# [num_decoding_seqs*num_lookahead_tokens, num_q_heads, head_dim], contiguous
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    softmax_scale: tl.float16,
    decoding_seq_lens: torch.Tensor,	# [num_decoding_seqs], contiguous
    seq_ids: torch.Tensor,  # [num_decoding_seqs], contiguous
    num_seq_chunks: int,
    cur_layer: int,

    num_layers: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_my_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    seq_chunk_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    num_lookahead_tokens: tl.constexpr,
):
    # grid shape: [num_decoding_seqs, num_lookahead_tokens, num_q_heads, num_seq_chunks]
    my_batch_id = tl.program_id(0).to(tl.int64)
    my_q_head_id = tl.program_id(1).to(tl.int64)
    my_seq_chunk_id = tl.program_id(2)
    my_lookahead_token_id = tl.program_id(3).to(tl.int64)
    my_kv_head_id = my_q_head_id // num_my_heads

    my_seq_id = tl.load(seq_ids + my_batch_id)
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id) - num_lookahead_tokens + my_lookahead_token_id + 1
    my_start_token_idx = my_seq_chunk_id * seq_chunk_size

    if my_start_token_idx >= my_seq_len:
        return

    offs_q = (my_batch_id*num_lookahead_tokens+my_lookahead_token_id)*num_q_heads*head_dim + my_q_head_id*head_dim + tl.arange(0, head_dim)
    my_q = tl.load(q + offs_q) # [head_dim]    

    start_block_idx = my_seq_chunk_id*(seq_chunk_size//block_size)
    k_ptrs = k_cache + (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]
    v_ptrs = v_cache + (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]

    max_score = float("-1e20")
    sum_exp = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    # In the following code we deal with the case where the sequence block is
    # the last one in the sequence separately, because:
    #   - The last sequence block may not be a full block, therefore maskings
    #     are needed.
    #   - We can use tl.arange() when the sequence block is not the last one,
    #     leading to better performance.
    if my_start_token_idx + seq_chunk_size > my_seq_len:
        # The seq block I am processing is the last one in the sequence
        my_num_blocks = tl.cdiv(
            my_seq_len - my_start_token_idx,
            block_size
        )
        for block_i in range(0, my_num_blocks):
            block_idx = start_block_idx + block_i
            block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + block_idx).to(tl.int64)
            k_block = tl.load(k_ptrs + block_index*num_layers*num_kv_heads*block_size*head_dim) # [block_size, head_dim]
            attn_score = tl.sum(my_q[None, :] * k_block, axis=1) # [block_size]
            attn_score = attn_score * softmax_scale
            offs_token = block_i*block_size + my_start_token_idx + tl.arange(0, block_size)
            attn_score = tl.where(offs_token < my_seq_len, attn_score, float('-1e20'))
            v_block = tl.load(v_ptrs + block_index*num_layers*num_kv_heads*block_size*head_dim) # [block_size, head_dim]
            
            cur_max_score = tl.max(attn_score, axis=0)
            new_max_score = tl.maximum(max_score, cur_max_score)
            exp_attn_score = tl.math.exp2(attn_score - new_max_score)
            old_acc_scale = tl.math.exp2(max_score - new_max_score)

            acc = acc*old_acc_scale + tl.sum(exp_attn_score[:, None]*v_block, axis=0)
            sum_exp = sum_exp*old_acc_scale + tl.sum(exp_attn_score, axis=0)
            max_score = new_max_score
    else:
        # The seq block I am processing is NOT the last one in the sequence
        for block_i in tl.static_range(0, seq_chunk_size // block_size):
            block_idx = start_block_idx + block_i
            block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + block_idx).to(tl.int64)
            k_block = tl.load(k_ptrs + block_index*num_layers*num_kv_heads*block_size*head_dim) # [block_size, head_dim]
            attn_score = tl.sum(my_q[None, :] * k_block, axis=1) # [block_size]
            attn_score = attn_score * softmax_scale
            v_block = tl.load(v_ptrs + block_index*num_layers*num_kv_heads*block_size*head_dim) # [block_size, head_dim]
            
            cur_max_score = tl.max(attn_score, axis=0)
            new_max_score = tl.maximum(max_score, cur_max_score)
            exp_attn_score = tl.math.exp2(attn_score - new_max_score)
            old_acc_scale = tl.math.exp2(max_score - new_max_score)

            acc = acc*old_acc_scale + tl.sum(exp_attn_score[:, None]*v_block, axis=0)
            sum_exp = sum_exp*old_acc_scale + tl.sum(exp_attn_score, axis=0)
            max_score = new_max_score

    offs_mid_o = (my_batch_id*num_lookahead_tokens+my_lookahead_token_id)*num_q_heads*num_seq_chunks*head_dim + my_seq_chunk_id*head_dim + (my_q_head_id*num_seq_chunks*head_dim) + tl.arange(0, head_dim)
    tl.store(mid_o + offs_mid_o, acc / sum_exp)
    offs_mid_o_logexpsum = (my_batch_id*num_lookahead_tokens+my_lookahead_token_id)*num_q_heads*num_seq_chunks + my_seq_chunk_id + my_q_head_id*num_seq_chunks
    tl.store(mid_o_logexpsum + offs_mid_o_logexpsum, tl.math.log2(sum_exp) + max_score)   # Here tl.log(sum_exp) + max_score = log(sum(e^{a_i}))


@triton.jit
def _fwd_paged_attention_phase2(
    mid_o: torch.Tensor,	# [num_decoding_seqs*num_lookahead_tokens, num_q_heads, num_seq_chunks, head_dim], contiguous
    mid_o_logexpsum: torch.Tensor,	# [num_decoding_seqs*num_lookahead_tokens, num_q_heads, num_seq_chunks], contiguous
    o: torch.Tensor,		# [num_decoding_seqs*num_lookahead_tokens, num_q_heads, head_dim], contiguous

    decoding_seq_lens: torch.Tensor,	# [num_decoding_seqs], contiguous

    num_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_seq_chunks: tl.constexpr,
    seq_chunk_size: tl.constexpr,
    num_lookahead_tokens: tl.constexpr,
):
    # grid shape: [num_decoding_seqs, num_lookahead_tokens, num_q_heads]
    my_batch_id = tl.program_id(0)
    my_q_head_id = tl.program_id(1)
    my_lookahead_token_id = tl.program_id(2)

    my_seq_len = tl.load(decoding_seq_lens + my_batch_id) - num_lookahead_tokens + my_lookahead_token_id + 1
    my_num_seq_chunks = tl.cdiv(my_seq_len, seq_chunk_size)

    sum_exp = 0.0
    max_score = float("-1e20")
    acc = tl.zeros([head_dim], dtype=tl.float32)

    for seq_chunk_id in range(my_num_seq_chunks):
        offs_mid_o = ((my_batch_id*num_q_heads+my_q_head_id)*num_seq_chunks+seq_chunk_id)*head_dim + tl.arange(0, head_dim)
        offs_mid_o_logexpsum = (my_batch_id*num_q_heads+my_q_head_id)*num_seq_chunks+seq_chunk_id
        cur_mid_o = tl.load(mid_o + offs_mid_o)   # [head_dim]
        cur_mid_o_logexpsum = tl.load(mid_o_logexpsum + offs_mid_o_logexpsum)

        new_max_score = tl.maximum(max_score, cur_mid_o_logexpsum)
        old_scale = tl.math.exp2(max_score - new_max_score)
        exp_score = tl.math.exp2(cur_mid_o_logexpsum - new_max_score)
        acc = acc * old_scale + exp_score * cur_mid_o
        sum_exp = sum_exp * old_scale + exp_score
        max_score = new_max_score

    offs_o = ((my_batch_id*num_lookahead_tokens+my_lookahead_token_id)*num_q_heads+my_q_head_id)*head_dim + tl.arange(0, head_dim)
    tl.store(o + offs_o, (acc / sum_exp).to(tl.float16))


def paged_attention(
    q: torch.Tensor,                    # [num_decoding_seqs*num_lookahead_tokens, num_q_heads, head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    infer_state: LlamaInferState,
    cur_layer: int,
    o: torch.Tensor     # [num_decoding_seqs*num_lookahead_tokens, num_q_heads, head_dim]
):
    assert q.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert infer_state.seq_chunk_size % engine_config.block_size == 0
    assert o.is_contiguous()
    assert q.shape[0] == infer_state.num_decoding_seqs * engine_config.num_lookahead_tokens
    assert o.shape[0] == infer_state.num_decoding_seqs * engine_config.num_lookahead_tokens

    mid_o = torch.empty((
        infer_state.num_decoding_seqs*engine_config.num_lookahead_tokens,
        model_config.num_q_heads,
        infer_state.num_seq_chunks,
        model_config.head_dim
    ), device=q.device, dtype=torch.float32)
    mid_o_logexpsum = torch.empty((
        infer_state.num_decoding_seqs*engine_config.num_lookahead_tokens,
        model_config.num_q_heads,
        infer_state.num_seq_chunks
    ), device=q.device, dtype=torch.float32)

    grid = (infer_state.num_decoding_seqs, engine_config.num_lookahead_tokens, model_config.num_q_heads, infer_state.num_seq_chunks)
    _fwd_paged_attention_phase1[grid](
        mid_o, mid_o_logexpsum,
        q, k_cache, v_cache,
        block_table,

        # Here we multiply softmax_scale by log2(e) and use `exp2` instead of
        # `exp` because of two reasons:
        # 1. Up to 12 Jun 2024, all NVIDIA GPUs does not have a `exp` instruction
        #    in PTX. When calculating `exp`, they multiply the input by log2(e)
        #    and use `exp2` instead.
        # 2. Some optimizations are disabled while using `exp` in a loop, see
        #    https://github.com/triton-lang/triton/issues/2961
        infer_state.softmax_scale * 1.442695040888963,
        infer_state.decoding_seq_lens,
        infer_state.seq_ids[infer_state.num_prefill_seqs:],
        infer_state.num_seq_chunks,
        cur_layer,

        model_config.num_layers,
        model_config.num_q_heads,
        model_config.num_kv_heads,
        model_config.num_q_heads // model_config.num_kv_heads,
        engine_config.block_size,
        model_config.head_dim,
        infer_state.seq_chunk_size,
        engine_config.max_blocks_per_seq,
        num_warps = 1,
        num_stages = 4
    )

    grid = (infer_state.num_decoding_seqs, engine_config.num_lookahead_tokens, model_config.num_q_heads)
    _fwd_paged_attention_phase2[grid](
        mid_o, mid_o_logexpsum,
        o,
        infer_state.decoding_seq_lens,
        model_config.num_q_heads,
        model_config.head_dim,
        infer_state.num_seq_chunks,
        infer_state.seq_chunk_size,
    )
