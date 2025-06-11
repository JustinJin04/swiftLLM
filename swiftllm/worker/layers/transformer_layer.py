import torch
import vllm_flash_attn

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.infer_state import LlamaInferState

from swiftllm.worker.kernels.linear import linear
from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace
from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
from swiftllm.worker.kernels.paged_attn import paged_attention
from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace

class LlamaTransformerLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        engine_config: EngineConfig,
        weight: LlamaTransformerLayerWeight,
        decoding_piggyback_stream: torch.cuda.Stream,
        layer_id: int
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.decoding_piggyback_stream = decoding_piggyback_stream
        self.layer_id = layer_id
    
    def forward(
        self,
        input_embds: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf: torch.Tensor, # [num_tokens, hidden_size]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        infer_state: LlamaInferState,
    ) -> torch.Tensor:
        # (fused) Add last layer's residual, and perform RMSNorm
        # Before: input_embds is the output of the last FFN block, and residual_buf
        #         is the residual to be added to input_embds
        # After: input_embds will be RMSNorm(input_embds + residual_buf), and
        #        residual_buf will be input_embds + residual_buf (which will be
        #        used as the residual after the attention block)
        fused_add_rmsnorm_inplace(
            input_embds,
            residual_buf,
            self.weight.attn_norm,
            self.model_config.rms_norm_eps
        )

        # Calculate QKV
        q = self.weight.q_proj(input_embds)
        k = self.weight.k_proj(input_embds)
        v = self.weight.v_proj(input_embds)
        q = q.view(-1, self.model_config.num_q_heads,  self.model_config.head_dim)	# [num_total_tokens, num_q_heads, head_dim]
        k = k.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]
        v = v.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]

        # print(f"q: {q}")
        # print(f"q: {q.shape}\nk: {k.shape}")
        # print(f"infer_state.position_cos: {infer_state.position_cos.shape}\ninfer_state.position_sin: {infer_state.position_sin.shape}")

        # Rotary emb
        rotary_embedding_inplace(
            q,
            k,
            infer_state
        )

        if not infer_state.ignore_kvcache:
            store_kvcache(
                k, v,
                k_cache, v_cache,
                block_table,
                self.model_config,
                self.engine_config,
                infer_state,
                self.layer_id
            )
        store_kvcache_event = torch.cuda.Event()
        store_kvcache_event.record()

        # Attention
        o = input_embds    # [num_total_tokens, hidden_size]
        if infer_state.num_prefill_seqs > 0:
            # Here the performance of vLLM's flash attention is better than us,
            # so use vllm_flash_attn
            o[:infer_state.num_prefill_tokens, :] = vllm_flash_attn.flash_attn_varlen_func(
                q[:infer_state.num_prefill_tokens, :, :],
                k[:infer_state.num_prefill_tokens, :, :],
                v[:infer_state.num_prefill_tokens, :, :],
                infer_state.prefill_seq_start_locs_with_end,
                infer_state.prefill_seq_start_locs_with_end,
                infer_state.max_prefill_len,
                infer_state.max_prefill_len,
                softmax_scale=infer_state.softmax_scale,
                causal=True
            ).reshape(-1, self.model_config.hidden_size)
            # prefill_attention(
            #     q, k, v, o[:infer_state.num_prefill_tokens, :],
            #     self.model_config, self.engine_config, infer_state
            # )
        if infer_state.num_decoding_seqs > 0:
            assert not infer_state.ignore_kvcache
            with torch.cuda.stream(self.decoding_piggyback_stream):
                torch.cuda.current_stream().wait_event(store_kvcache_event)
                paged_attention(
                    q[infer_state.num_prefill_tokens:, :, :],
                    k_cache, v_cache, block_table,
                    self.model_config, self.engine_config, infer_state,
                    self.layer_id,
                    o[infer_state.num_prefill_tokens:, :],
                )
                event = torch.cuda.Event()
                event.record()
            torch.cuda.default_stream().wait_event(event)
        
        # Output GEMM
        o = self.weight.o_proj(o)	# [num_total_tokens, hidden_size]

        # residual & FFN norm
        fused_add_rmsnorm_inplace(o, residual_buf, self.weight.ffn_norm, self.model_config.rms_norm_eps)
        q = None
        k = None
        v = None

        # FFN
        # TODO: Currently marlin does not support fused version of up_proj and gate_proj 
        # how to support fused version??
        up_gate_proj = self.weight.up_gate_proj(o)
        silu_and_mul_inplace(up_gate_proj)
        up_gate_proj = up_gate_proj[:, :self.model_config.ffn_inter_dim].contiguous()
        ffn_out = self.weight.down_proj(up_gate_proj[:, :self.model_config.ffn_inter_dim])

        return ffn_out
