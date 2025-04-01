import safetensors
import torch

MODEL_PATH="/home/gehao/jz/deepseek/model/llama2-1b/model.safetensors"

with safetensors.safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
    for k in f.keys():
        print(f"{k}: {f.get_tensor(k).shape}, {f.get_tensor(k).dtype}")
