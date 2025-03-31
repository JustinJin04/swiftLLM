import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight
from swiftllm.worker.kernels.rmsnorm import rmsnorm_inplace
from swiftllm.worker.infer_state import LlamaInferState
from swiftllm.worker.kernels.linear import linear
from swiftllm.worker.output import ModelOutput

class LlamaPostLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
    
    def forward(
        self,
        input_embds: torch.Tensor,	# [num_total_tokens, hidden_size]
                                    # num_total_tokens = num_prefill_tokens + num_decoding_seqs*num_lookahead_tokens
        infer_state: LlamaInferState
    ) -> torch.Tensor:
        # Slice to get the last token embedding for each request
        last_token_indices = torch.cat(
            (
                infer_state.prefill_seq_start_locs + infer_state.prefill_seq_lens - 1,
                torch.arange(infer_state.num_prefill_tokens, infer_state.num_tokens, device=input_embds.device, dtype=torch.int32)
            ), dim=0
        ) # num_prefill_seqs + num_decoding_seqs*num_lookahead_tokens
        last_input = torch.empty((infer_state.num_tokens-infer_state.num_prefill_tokens+infer_state.num_prefill_seqs, self.model_config.hidden_size), device=input_embds.device, dtype=input_embds.dtype)
        last_input[:, :] = input_embds[last_token_indices, :] #[num_prefill_seqs+num_decoding_seqs*num_lookahead_tokens, hidden_size]
        # Apply RMS-norm
        rmsnorm_inplace(
            last_input,
            self.weights.final_norm,
            self.model_config.rms_norm_eps
        )
        logits = linear(last_input, self.weights.lm_head)    # [num_prefill_seqs+num_lookahead_tokens*num_decoding_seqs, vocab_size]
        output_tokens = torch.argmax(logits, dim=1).tolist()

        # return output_tokens
        return ModelOutput(
            num_prefill_requests=infer_state.num_prefill_seqs,
            prefill_logits=logits[:infer_state.num_prefill_seqs, :],
            prefill_tokens=output_tokens[:infer_state.num_prefill_seqs],

            num_decoding_requests=infer_state.num_decoding_seqs,
            decoding_logits=logits[infer_state.num_prefill_seqs:, :],
            decoding_tokens=output_tokens[infer_state.num_prefill_seqs:],
        )
    