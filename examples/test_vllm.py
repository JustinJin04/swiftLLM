import time
import argparse
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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

    # Initialize vLLM engine
    start_time = time.perf_counter()
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=1,  # Adjust based on your GPU setup
        gpu_memory_utilization=0.8,
        block_size=16,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
    )
    model_creation_time = time.perf_counter() - start_time
    print(f"Model creation time: {model_creation_time:.2f} seconds")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    prompt = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40"
    
    # Create sampling params
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic output
        top_p=1.0,
        max_tokens=100,
    )
    
    # Generate output
    start_time = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    generation_time = time.perf_counter() - start_time
    
    output_text = outputs[0].outputs[0].text
    print(f"{prompt}|{output_text}")
    print(f"Generation time: {generation_time:.2f} seconds")

if __name__ == "__main__":
    main()