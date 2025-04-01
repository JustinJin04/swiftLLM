import time
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

def main():
    args = get_args()

    target_engine_config = swiftllm.EngineConfig(
        model_path=args.target_path,
        use_dummy=False,

        block_size=16,
        gpu_mem_utilization=0.7,
        num_cpu_blocks=0,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=2048,

        # The following are not used in the offline example
        max_batch_size=1,
        max_tokens_in_batch=2048,

        # spec decoding
        num_lookahead_tokens=10,
    )
    drafter_engine_config = swiftllm.EngineConfig(
        model_path=args.drafter_path,
        use_dummy=False,

        block_size=16,
        gpu_mem_utilization=0.2,
        num_cpu_blocks=0,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=2048,

        # The following are not used in the offline example
        max_batch_size=1,
        max_tokens_in_batch=2048,

        # spec decoding
        num_lookahead_tokens=1,
    )

    spec_worker = swiftllm.SpecDecWorker(
        target_engine_config=target_engine_config,
        drafter_engine_config=drafter_engine_config
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_path)
    
    prompt = str(np.arange(1000).tolist())
    input_ids = tokenizer(prompt)['input_ids']

    # warmup
    for _ in range(2):
        spec_output = spec_worker.forward(
            input_ids,
            num_max_tokens_to_generate=1000,
        )
    
    # timing
    start_time = time.perf_counter()
    rounds = 5
    for _ in range(rounds):
        spec_worker.forward(
            input_ids,
            num_max_tokens_to_generate=1000,
        )
    end_time = time.perf_counter()
    print(f"SpecDecoding time: {(end_time-start_time)/rounds:.2f} seconds")
    print(f"Accepted tokens list: {spec_output.num_accepted_tokens_list}")

if __name__ == "__main__":
    main()

        
    