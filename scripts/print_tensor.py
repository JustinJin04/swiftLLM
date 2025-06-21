import safetensors

def print_tensor_info(file_path):
    with safetensors.safe_open(file_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"Key: {key}, Shape: {tensor.shape}, Dtype: {tensor.dtype}")

print_tensor_info("/zhan/model/llama-2-7b-chat-marlin/model.safetensors")