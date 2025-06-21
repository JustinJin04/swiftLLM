import argparse

from transformers import AutoTokenizer, AutoConfig
from auto_gptq import AutoGPTQForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str)
parser.add_argument("--save-dir", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    # print("Validating quantization is compatible with Marlin...")
    # config = AutoConfig.from_pretrained(args.model_id)
    # if not hasattr(config, "quantization_config"):
    #     raise ValueError(
    #         f"Config for {args.model_id} does not have quantization_config attribute"
    #     )
    # quantization_config = config.quantization_config
    # required_attrs = ["group_size"]
    # for attr in required_attrs:
    #     if attr not in quantization_config:
    #         raise ValueError(
    #             f"Attribute '{attr}' not found in quantization config"
    #         )

    # if quantization_config["bits"] != 4:
    #     raise ValueError(
    #         f"This model is not compatible with Marlin. Marlin requires 4 bit gptq, but "
    #         f"quantization_config['bits'] = {quantization_config['bits']}"
    #     )
    # if quantization_config["group_size"] != 128 and quantization_config["group_size"] != -1:
    #     raise ValueError(
    #         f"This model is not compatible with Marlin. Marlin requires group size to be 128 or -1 (channelwise), but "
    #         f"quantization_config['group_size'] = {quantization_config['group_size']}"
    #     )
    # elif quantization_config["desc_act"]:
    #     raise ValueError(
    #         f"This model is not compatible with Marlin. Marlin requires no activation ordering, but "
    #         f"quantization_config['desc_act'] = {quantization_config['desc_act']}"
    #     )
    # elif not quantization_config["sym"]:
    #     raise ValueError(
    #         f"This model is not compatible with Marlin. Marlin requires symmetric quantization, but "
    #         f"quantization_config['sym'] = {quantization_config['sym']}"
    #     )

    print("Converting model to Marlin...")
    model = AutoGPTQForCausalLM.from_quantized(
        args.model_id, 
        device_map="auto", 
        use_marlin=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id
    )

    print(f"Saving model to {args.save_dir}")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)