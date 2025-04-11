import time
import argparse
import numpy as np
from datasets import load_dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
    )
    return parser.parse_args()

def main():
    args = get_args()
    ds = load_dataset(args.dataset_path)
    for i in range(5):
        example = ds['train'][i]
        print(f"Question {i+1}: {example['Problem']}")

if __name__ == "__main__":
    main()