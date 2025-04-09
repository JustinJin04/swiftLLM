import safetensors
import argparse
import os
import torch
import marlin

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
    A = torch.randn(1, 4096, dtype=torch.float16, device="cuda")
    B = None
    scale = None
    with safetensors.safe_open(args.tensor_path, framework="pt", device="cuda") as f:
        for k in f.keys():
            # print(f"{k}: {f.get_tensor(k).shape}, {f.get_tensor(k).dtype}")
            if k == "model.layers.0.self_attn.k_proj.B":
                B = f.get_tensor(k)
            elif k == "model.layers.0.self_attn.k_proj.s":
                scale = f.get_tensor(k)
    assert A.device == B.device and A.device == scale.device
    infeatures = scale.shape[0] * 128
    outfeatures = scale.shape[1]
    # layer = marlin.Layer(infeatures, outfeatures,128).to(A.device)
    # layer.B.copy_(B)
    # layer.s.copy_(scale)
    # C = layer(A)
    # print(C.shape)
    # print(C.device)
    C = torch.zeros((infeatures, outfeatures), dtype=torch.float16, device=A.device)
    workspace = torch.zeros(outfeatures // 128 * 16, device=A.device)
    marlin.mul(A, B, C, scale, workspace, 64, 256, -1)
    print(C.shape)
    


            
