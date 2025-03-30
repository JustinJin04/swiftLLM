import dataclasses
from torch import Tensor

@dataclasses.dataclass
class ModelOutput:
    num_prefill_requests: int
    prefill_logits: Tensor # [num_prefill_tokens, vocab_size]
    prefill_tokens: list[int] # [num_prefill_requests]

    num_decoding_requests: int
    decoding_logits: Tensor # [num_decoding_requests*num_lookahead_tokens, vocab_size]
    decoding_tokens: list[int] # [num_decoding_requests*num_lookahead_tokens]

