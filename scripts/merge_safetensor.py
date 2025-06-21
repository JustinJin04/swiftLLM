import os
from safetensors.torch import load_file, save_file

# List your files to merge
files = [
    "/zhan/swiftLLM/scripts/w4a16/model-00001-of-00002.safetensors",
    "/zhan/swiftLLM/scripts/w4a16/model-00002-of-00002.safetensors",
    # add more if needed
]

merged_state_dict = {}
for fname in files:
    state_dict = load_file(fname)
    merged_state_dict.update(state_dict)

# Save the combined result
os.remove("/zhan/swiftLLM/scripts/w4a16/model-00001-of-00002.safetensors")
os.remove("/zhan/swiftLLM/scripts/w4a16/model-00002-of-00002.safetensors")
os.remove("/zhan/swiftLLM/scripts/w4a16/model.safetensors.index.json")
save_file(merged_state_dict, "model.safetensors")
