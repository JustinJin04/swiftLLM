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
        gpu_mem_utilization=0.5,
        num_cpu_blocks=0,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=2048,

        # The following are not used in the offline example
        max_batch_size=1,
        max_tokens_in_batch=2048,

        # spec decoding
        num_lookahead_tokens=1,
    )
    start_time = time.perf_counter()
    model = swiftllm.LlamaModel(engine_config)
    model.load_weight_and_init_kvcache()
    model_creation_time = time.perf_counter() - start_time
    print(f"Model creation time: {model_creation_time:.2f} seconds")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    prompt = "Write a short story about a robot learning to love."
    input_ids = tokenizer.encode(prompt)

    model_output = model.prefill_decode(
        input_ids,
        num_max_tokens_to_generate=50,
    )
    output_text = tokenizer.decode(model_output, skip_special_tokens=True)
    print(f"[Prompt]: {prompt}")
    print(f"[Answer]: {output_text}")


if __name__ == "__main__":
    main()

        
    