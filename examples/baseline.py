import time
import argparse
import numpy as np
from transformers import AutoTokenizer

import swiftllm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    return parser.parse_args()

def main():
    args = get_args()

    engine_config = swiftllm.EngineConfig(
        model_path=args.model_path,
        use_dummy=False,

        block_size=16,
        gpu_mem_utilization=0.7,
        num_cpu_blocks=0,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=2048,

        # The following are not used in the offline example
        max_batch_size=1,
        max_tokens_in_batch=2048*1,

        # spec decoding
        num_lookahead_tokens=1,
    )
    start_time = time.perf_counter()
    model = swiftllm.LlamaModel(engine_config)
    model.load_weight_and_init_kvcache()
    model_creation_time = time.perf_counter() - start_time
    print(f"Model creation time: {model_creation_time:.2f} seconds")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    prompt = str(np.arange(1000).tolist())
    input_ids = tokenizer(prompt)['input_ids']

    # warmup
    for _ in range(2):
        model.prefill_decode(
            input_ids,
            num_max_tokens_to_generate=1000,
        )
    print(f"warmup done")
    
    # timing
    start_time = time.perf_counter()
    rounds = 5
    for _ in range(rounds):
        model.prefill_decode(
            input_ids,
            num_max_tokens_to_generate=1000,
        )
    end_time = time.perf_counter()
    print(f"Baseline Time: {(end_time - start_time)/rounds:.2f} seconds")

if __name__ == "__main__":
    main()

        
    