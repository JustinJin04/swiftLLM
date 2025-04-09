import safetensors
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Parse command line arguments.")
    parser.add_argument(
        "--tensor_path",
        type=str,
        required=True,
        help="Path to the model file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
    )
    args = parser.parse_args()
    return args


def dump_tensor(tensor):
    path = get_args().output_path
    with open(path, "w") as f:
        for v in tensor.flatten():
            f.write(f"{v.item()}\n")

if __name__ == "__main__":
    args = get_args()
    with safetensors.safe_open(args.tensor_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            print(f"{k}: {f.get_tensor(k).shape}, {f.get_tensor(k).dtype}")
            # if(k == "model.layers.0.mlp.up_proj.g_idx"):
            #     dump_tensor(f.get_tensor(k))
