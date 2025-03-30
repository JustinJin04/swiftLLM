import time
import argparse
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
    model_path = args.model_path

    engine_config = swiftllm.EngineConfig(
        model_path=model_path,
        use_dummy=False,

        block_size=16,
        gpu_mem_utilization=0.99,
        num_cpu_blocks=0,
        max_seqs_in_block_table=128,
        max_blocks_per_seq=2048,

        # The following are not used in the offline example
        max_batch_size=4,
        max_tokens_in_batch=2048*4,

        # spec decoding
        num_lookahead_tokens=1,
    )

    start_time = time.perf_counter()

    # Initialize the model
    # For instructions on how to initialize the model, see comments in swiftllm/worker/model.py
    model = swiftllm.LlamaModel(engine_config)
    model.load_weights()
    num_blocks = model.profile_num_blocks()
    print("Number of blocks:", num_blocks)
    model.init_kvcache_and_swap(num_blocks)
    model_creation_time = time.perf_counter() - start_time
    print(f"Model creation time: {model_creation_time:.2f} seconds")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    prompts = [
        "Life blooms like a flower, far away",
        "one two three four five",
        "A B C D E F G H I J K L M N O P Q R S T U V",
        "To be or not to be,",
    ]
    outputs = []

    # prefill
    input_ids = tokenizer(prompts)['input_ids']
    seq_ids_list = list(range(len(prompts)))
    prefill_outputs = model.forward(
        input_ids,
        seq_ids_list,
        [],
    )
    outputs.append(prefill_outputs.prefill_tokens)

    # decode
    seq_lens = [len(x) for x in input_ids]
    last_round_output = prefill_outputs.prefill_tokens
    for _ in range(20):
        for i, _ in enumerate(prompts):
            seq_lens[i] += 1
        decoding_outputs = model.forward(
            [[x] for x in last_round_output],
            seq_ids_list,
            seq_lens,
        )
        last_round_output = decoding_outputs.decoding_tokens
        outputs.append(last_round_output)
    
    for i, prompt in enumerate(prompts):
        output_tokens = [x[i] for x in outputs]
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"{prompt}|{output_text}")


if __name__ == "__main__":
    main()

        
    