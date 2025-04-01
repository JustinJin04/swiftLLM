import time

from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig
from swiftllm.worker.model import LlamaModel
from swiftllm.worker.output import ModelOutput, SpecWorkerOutput

class SpecDecWorker:
    def __init__(self, target_engine_config: EngineConfig, drafter_engine_config: EngineConfig):
        start_time = time.perf_counter()
        self.target = LlamaModel(target_engine_config)
        self.target.load_weight_and_init_kvcache()
        model_creation_time = time.perf_counter() - start_time
        print(f"Target Model creation time: {model_creation_time:.2f} seconds")

        self.drafter = LlamaModel(drafter_engine_config)
        start_time = time.perf_counter()
        self.drafter.load_weight_and_init_kvcache()
        model_creation_time = time.perf_counter() - start_time
        print(f"Draft Model creation time: {model_creation_time:.2f} seconds")

        self.num_lookahead_tokens = target_engine_config.num_lookahead_tokens

        # used for debugging
        self.tokenizer = AutoTokenizer.from_pretrained(target_engine_config.model_path)

    def _verify(self, verify_input_ids: list[int], verify_output: ModelOutput):
        '''
        If rejection occurs at the middle, return (next_draft_token_id, num_accepted_tokens)
        If all accepted, return (bonus_token_id, num_accepted_tokens)
        '''
        assert len(verify_input_ids) == len(verify_output.decoding_tokens)
        assert len(verify_input_ids) == self.num_lookahead_tokens
        for i in range(self.num_lookahead_tokens-1):
            if verify_output.decoding_tokens[i] != verify_input_ids[i+1]:
                return verify_output.decoding_tokens[i], i
        # all accepted
        return verify_output.decoding_tokens[-1], self.num_lookahead_tokens-1


    def forward(
        self,
        input_ids: list[int],
    )->SpecWorkerOutput:
        target_prefill_output = self.target.prefill(
            input_ids,
        )
        draft_prefill_output = self.drafter.prefill(
            input_ids,
        )

        # statistics collector
        final_output_ids = [target_prefill_output.prefill_tokens[0]]
        num_accepted_tokens_list = []

        verify_input_ids = [target_prefill_output.prefill_tokens[0]]
        draft_input_id = draft_prefill_output.prefill_tokens[0]
        draft_seq_len = len(input_ids)
        for _ in range(10):
            for i in range(self.num_lookahead_tokens-1):
                draft_seq_len += 1
                draft_output = self.drafter.decode(
                    [draft_input_id],
                    draft_seq_len,
                )
                draft_input_id = draft_output.decoding_tokens[0]
                verify_input_ids.append(draft_input_id)
            
            print(f"> draft tokens: {self.tokenizer.decode(final_output_ids+verify_input_ids[1:], skip_special_tokens=True)}")
            
            # verify phase
            verify_output = self.target.decode(
                verify_input_ids,
                draft_seq_len+1,
            )
            next_token_id, num_accepted_tokens = self._verify(verify_input_ids, verify_output)
            draft_seq_len = draft_seq_len - (self.num_lookahead_tokens-2)+num_accepted_tokens
            if num_accepted_tokens == self.num_lookahead_tokens-1:
                _ = self.drafter.decode(
                    [draft_input_id],
                    draft_seq_len,
                )
            draft_input_id = next_token_id
            verify_input_ids = [next_token_id]

            final_output_ids.extend(verify_output.decoding_tokens[:num_accepted_tokens+1])
            num_accepted_tokens_list.append(num_accepted_tokens)
            print(f"> after verify: {self.tokenizer.decode(final_output_ids, skip_special_tokens=True)}")
                
        
        return SpecWorkerOutput(
            final_output_ids=final_output_ids,
            num_accepted_tokens_list=num_accepted_tokens_list,
        )




