import time
import json
import argparse
import numpy as np
from transformers import AutoTokenizer

import swiftllm
import swiftllm.worker

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--drafter_path",
        type=str,
        required=True,
    )
    return parser.parse_args()

class DatasetIter:
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path
        self.questions = []
        with open(jsonl_path, "r") as f:
            for line in f:
                question = json.loads(line)
                self.questions.append(question['question'])
        self.current = 0
        self.end = len(self.questions)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        ret = self.questions[self.current]
        self.current += 1
        return ret



def main():
    args = get_args()

    target_engine_config = swiftllm.EngineConfig(
        model_path=args.target_path,
        use_dummy=False,

        block_size=16,
        gpu_mem_utilization=0.3,
        num_cpu_blocks=0,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=2048,

        # The following are not used in the offline example
        max_batch_size=1,
        max_tokens_in_batch=2048*1,

        # spec decoding
        num_lookahead_tokens=10,
    )
    drafter_engine_config = swiftllm.EngineConfig(
        model_path=args.drafter_path,
        use_dummy=False,

        block_size=16,
        gpu_mem_utilization=0.3,
        num_cpu_blocks=0,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=2048,

        # The following are not used in the offline example
        max_batch_size=1,
        max_tokens_in_batch=2048*1,

        # spec decoding
        num_lookahead_tokens=1,
    )

    spec_worker = swiftllm.SpecDecWorker(
        target_engine_config=target_engine_config,
        drafter_engine_config=drafter_engine_config
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_path)
    
    dataset_iter = DatasetIter("/zhan/dataset/questions.jsonl")
    for question in dataset_iter:
        input_ids = tokenizer(question)['input_ids']
        spec_output = spec_worker.forward(
            input_ids,
            num_max_tokens_to_generate=100,
        )
        output_text = tokenizer.decode(spec_output.final_output_ids, skip_special_tokens=True)
        print(f"{question}|{output_text}")
        print(f"accepted tokens list: {spec_output.num_accepted_tokens_list}")

if __name__ == "__main__":
    main()
