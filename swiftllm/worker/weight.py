import json
import os
import dataclasses
from typing import Union

import torch
import safetensors
import marlin

from swiftllm.model_config import LlamaModelConfig

@dataclasses.dataclass
class RegisteredWeightItem:
    attr_name: str
    key: str
    shape: tuple
    dtype: torch.dtype

class WeightBase:
    """
    The base class of all weight classes (i.e. LlamaTransformerLayerWeight or LlamaWeight)

    During weight initialization, each concrete weight class should first register
    all weight items. Each weight item has its own attribute name, key, shape, and dtype.

    During weight loading, RegisterWeightItem will be passed to the weight getter
    function, which should return the corresponding weight value (real/dummy).
    """

    def __init__(self):
        self.registered_weights = []

    def register_weight(self, item: RegisteredWeightItem):
        self.registered_weights.append(item)

    def _post_process_after_load(self, getter: callable):
        """
        This function is called after loading weights (real/dummy).
        Defined in each concrete weight class, called by load_weights().
        """
        raise NotImplementedError()
    
    def load_weights(self, getter: callable):
        """
        Load weights
        """
        for item in self.registered_weights:
            weight_value = getter(item)
            assert weight_value is not None, f"getter() returned None for {item.key} ({item})"
            assert isinstance(weight_value, torch.Tensor), f"Weight {item.key} is not a tensor"
            assert weight_value.shape == item.shape, f"Shape of weight {item.key} does not match"
            assert weight_value.device.type == "cuda", f"Weight {item.key} is not on GPU"
            setattr(self, item.attr_name, weight_value.to(item.dtype))
        self._post_process_after_load(getter)


class LlamaTransformerLayerWeight(WeightBase):
    """
    Class stores the weights of one transformer layer (transformer block) in Llama model.
    """

    def __init__(
        self,
        layer_id: int,
        model_config: LlamaModelConfig,
        dtype: torch.dtype
    ):
        super().__init__()

        self.layer_id = layer_id
        self.model_config = model_config
        self.dtype = dtype

        self.register_weight(RegisteredWeightItem(
            "attn_norm",
            f"model.layers.{self.layer_id}.input_layernorm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "q_proj",
            f"model.layers.{self.layer_id}.self_attn.q_proj.weight",
            (self.model_config.hidden_size, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "k_proj",
            f"model.layers.{self.layer_id}.self_attn.k_proj.weight",
            (self.model_config.num_kv_heads*self.model_config.head_dim, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "v_proj",
            f"model.layers.{self.layer_id}.self_attn.v_proj.weight",
            (self.model_config.num_kv_heads*self.model_config.head_dim, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "o_proj",
            f"model.layers.{self.layer_id}.self_attn.o_proj.weight",
            (self.model_config.hidden_size, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "ffn_norm",
            f"model.layers.{self.layer_id}.post_attention_layernorm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "up_proj",
            f"model.layers.{self.layer_id}.mlp.up_proj.weight",
            (self.model_config.ffn_inter_dim, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "gate_proj",
            f"model.layers.{self.layer_id}.mlp.gate_proj.weight",
            (self.model_config.ffn_inter_dim, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "down_proj",
            f"model.layers.{self.layer_id}.mlp.down_proj.weight",
            (self.model_config.hidden_size, self.model_config.ffn_inter_dim),
            self.dtype
        ))

    def _post_process_after_load(self, getter: callable):
        # wrap with torch.nn.Linear
        for attr_name in ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"]:
            weight = getattr(self, attr_name)
            setattr(self, attr_name, torch.nn.Linear(
                in_features=weight.shape[1],
                out_features=weight.shape[0],
                bias=False,
                device=weight.device,
                dtype=weight.dtype)
            )
            getattr(self, attr_name).weight.data.copy_(weight)

        # pylint: disable=no-member
        # self.qkv_proj = torch.cat((self.q_proj, self.k_proj, self.v_proj), dim=0).contiguous()
        # del self.q_proj, self.k_proj, self.v_proj
        up_gate_proj = torch.cat((self.up_proj, self.gate_proj), dim=0).contiguous()
        self.up_gate_proj = torch.nn.Linear(
            in_features=up_gate_proj.shape[1],
            out_features=up_gate_proj.shape[0],
            bias=False,
            device=up_gate_proj.device,
            dtype=up_gate_proj.dtype
        )
        self.up_gate_proj.weight.data.copy_(up_gate_proj)
        del self.up_proj, self.gate_proj


class LlamaWeight(WeightBase):
    def __init__(
        self,
        model_config: LlamaModelConfig,
        dtype: torch.dtype
    ):
        super().__init__()

        self.model_config = model_config
        self.dtype = dtype

        self.register_weight(RegisteredWeightItem(
            "wte",
            "model.embed_tokens.weight",
            (self.model_config.vocab_size, self.model_config.hidden_size),
            self.dtype
        ))
        if self.model_config.tie_word_embeddings:
            self.register_weight(RegisteredWeightItem(
                "lm_head",
                "model.embed_tokens.weight",
                (self.model_config.vocab_size, self.model_config.hidden_size),
                self.dtype
            ))
        else:
            self.register_weight(RegisteredWeightItem(
                "lm_head",
                "lm_head.weight",
                (self.model_config.vocab_size, self.model_config.hidden_size),
                self.dtype
            ))
        self.register_weight(RegisteredWeightItem(
            "final_norm",
            "model.norm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))

        self.layers: list[LlamaTransformerLayerWeight] = []
        for i in range(self.model_config.num_layers):
            layer = LlamaTransformerLayerWeight(i, self.model_config, self.dtype)
            self.layers.append(layer)

    def _post_process_after_load(self, getter: callable):
        for layer in self.layers:
            layer.load_weights(getter)


@dataclasses.dataclass
class MarlinWeights:
    groupsize: int
    B: torch.Tensor # [infeatures // 16, outfeatures*2]
    s: torch.Tensor # [infeatures // groupsize, outfeatures]

    @property
    def infeatures(self) -> int:
        return self.s.shape[0] * self.groupsize
    @property
    def outfeatures(self) -> int:
        return self.s.shape[1]

    def convert_to_marlin_layer(self) -> marlin.Layer:
        """Convert to optimized Marlin format"""
        layer = marlin.Layer(
            self.infeatures, 
            self.outfeatures,
            groupsize=self.groupsize
        ).to(self.B.device)  # <-- move the entire layer to the correct device
        
        # layer.B.copy_(self.B)
        setattr(layer, "B", torch.nn.Parameter(self.B, requires_grad=False))
        # layer.s.copy_(self.s)
        setattr(layer, "s", torch.nn.Parameter(self.s, requires_grad=False))
        assert layer.B.device == self.B.device
        return layer


class LlamaTransformerLayerWeightMarlin(WeightBase):

    def _register_quant_weight(
        self, 
        attr_name: str, 
        key: str, 
        infeatures: int,
        outfeatures: int,
        dtype: torch.dtype,
    ):
        """
        register all tensors that are related to each weight. For examples:
        B: torch.Tensor # [infeatures // 16, outfeatures*2], int32
        s: torch.Tensor # [infeatures // groupsize, outfeatures], float16
        """
        groupsize = self.model_config.quantization_config["group_size"]
        assert groupsize == 128
        
        self.register_weight(RegisteredWeightItem(
            f"{attr_name}_B",
            # f"{key}.qweight",
            f"{key}.B",
            (infeatures // 16, outfeatures*2),
            torch.int32
        ))
        self.register_weight(RegisteredWeightItem(
            f"{attr_name}_s",
            # f"{key}.scales",
            f"{key}.s",
            (infeatures // groupsize, outfeatures),
            self.model_config.dtype
        ))
        self.quant_weight_name_list.append(attr_name)


    def __init__(
        self,
        layer_id: int,
        model_config: LlamaModelConfig,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.layer_id = layer_id
        self.model_config = model_config
        self.dtype = dtype

        self.quant_weight_name_list: list[str] = []

        self.register_weight(RegisteredWeightItem(
            "attn_norm",
            f"model.layers.{self.layer_id}.input_layernorm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "ffn_norm",
            f"model.layers.{self.layer_id}.post_attention_layernorm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))
        self._register_quant_weight(
            "q_proj",
            f"model.layers.{self.layer_id}.self_attn.q_proj",
            self.model_config.hidden_size,
            self.model_config.hidden_size,
            self.dtype
        )
        self._register_quant_weight(
            "k_proj",
            f"model.layers.{self.layer_id}.self_attn.k_proj",
            self.model_config.hidden_size,
            self.model_config.num_kv_heads*self.model_config.head_dim,
            self.dtype
        )
        self._register_quant_weight(
            "v_proj",
            f"model.layers.{self.layer_id}.self_attn.v_proj",
            self.model_config.hidden_size,
            self.model_config.num_kv_heads*self.model_config.head_dim,
            self.dtype
        )
        self._register_quant_weight(
            "o_proj",
            f"model.layers.{self.layer_id}.self_attn.o_proj",
            self.model_config.hidden_size,
            self.model_config.hidden_size,
            self.dtype
        )
        self._register_quant_weight(
            "up_proj",
            f"model.layers.{self.layer_id}.mlp.up_proj",
            self.model_config.hidden_size,
            self.model_config.ffn_inter_dim,
            self.dtype
        )
        self._register_quant_weight(
            "gate_proj",
            f"model.layers.{self.layer_id}.mlp.gate_proj",
            self.model_config.hidden_size,
            self.model_config.ffn_inter_dim,
            self.dtype
        )
        self._register_quant_weight(
            "down_proj",
            f"model.layers.{self.layer_id}.mlp.down_proj",
            self.model_config.ffn_inter_dim,
            self.model_config.hidden_size,
            self.dtype
        )


    def _post_process_after_load(self, getter):
        """
        1. convert quant weight to fp16
        2. pack to marlin layer and store the B and s
        3. delete useless weights
        4. fuse up_proj and gate (TODO)
        """
        for attr_name in self.quant_weight_name_list:
            w4a16_weight = MarlinWeights(
                self.model_config.quantization_config["group_size"],
                getattr(self, f"{attr_name}_B"),
                getattr(self, f"{attr_name}_s"),
            )
            setattr(self, attr_name, w4a16_weight.convert_to_marlin_layer())
            # delete useless weights
            delattr(self, f"{attr_name}_B")
            delattr(self, f"{attr_name}_s")


        # TODO: how to fuse up_gate_proj ????
        up_unpack = marlin.unpack_unswizzle_untile(self.up_proj.B)
        gate_unpack = marlin.unpack_unswizzle_untile(self.gate_proj.B)
        up_gate = torch.cat((up_unpack, gate_unpack), dim=1).contiguous()
        up_gate_B = marlin.tile_swizzle_pack(up_gate)
        up_gate_s = torch.cat((self.up_proj.s, self.gate_proj.s), dim=1).contiguous()
        up_gate_weight = MarlinWeights(
            self.model_config.quantization_config["group_size"],
            up_gate_B,
            up_gate_s
        )
        self.up_gate_proj = up_gate_weight.convert_to_marlin_layer()
        del self.up_proj, self.gate_proj
        # self.up_gate_proj = torch.cat((self.up_proj, self.gate_proj), dim=0).contiguous()
        # del self.up_proj, self.gate_proj


class LLamaWeightMarlin(WeightBase):
    def __init__(
        self,
        model_config: LlamaModelConfig,
        dtype: torch.dtype
    ):
        super().__init__()

        self.model_config = model_config
        self.dtype = dtype
        self.register_weight(RegisteredWeightItem(
            "wte",
            "model.embed_tokens.weight",
            (self.model_config.vocab_size, self.model_config.hidden_size),
            self.dtype
        ))
        if self.model_config.tie_word_embeddings:
            self.register_weight(RegisteredWeightItem(
                "lm_head",
                "model.embed_tokens.weight",
                (self.model_config.vocab_size, self.model_config.hidden_size),
                self.dtype
            ))
        else:
            self.register_weight(RegisteredWeightItem(
                "lm_head",
                "lm_head.weight",
                (self.model_config.vocab_size, self.model_config.hidden_size),
                self.dtype
            ))
        self.register_weight(RegisteredWeightItem(
            "final_norm",
            "model.norm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))

        self.layers: list[LlamaTransformerLayerWeightMarlin] = []
        for i in range(self.model_config.num_layers):
            self.layers.append(
                LlamaTransformerLayerWeightMarlin(i, model_config, dtype)
            )

    def _post_process_after_load(self, getter: callable):
        for layer in self.layers:
            layer.load_weights(getter)



def load_weights(
    model_config: LlamaModelConfig,
    dtype: torch.dtype,
    model_path: str,
    use_dummy: bool = False
) -> Union[LlamaWeight, LLamaWeightMarlin]:
    """
    Load weights from a given path
    """
    if use_dummy:
        def weight_getter_dummy(item: RegisteredWeightItem):
            return torch.empty(item.shape, dtype=item.dtype, device="cuda").uniform_(-0.001, 0.001)
        getter = weight_getter_dummy
    else:
        safetensor_files = [name for name in os.listdir(model_path) if name.endswith(".safetensors")]
        if len(safetensor_files) > 0:
            # Use Safetensors
            safetensor_index_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(safetensor_index_path):
                # The weight is stored in multiple files
                f = open(safetensor_index_path, "r", encoding="utf-8")
                safetensor_index = json.load(f)["weight_map"]
                safetensor_filename = None
            else:
                # The weight is stored in a single file
                assert len(safetensor_files) == 1, "model.safetensors.index.json not found, but there are multiple .safetensors files"
                safetensor_index = None
                safetensor_filename = safetensor_files[0]

            def weight_getter_real(item: RegisteredWeightItem):
                file_name = safetensor_index[item.key] if safetensor_index is not None else safetensor_filename
                file_path = os.path.join(model_path, file_name)
                # For safetensor files, since "opening" it is cheap, we open it every time
                with safetensors.safe_open(file_path, framework="pt", device="cuda") as f:
                    tensor = f.get_tensor(item.key)
                return tensor.to(item.dtype)
            getter = weight_getter_real

        else:
            # Use PyTorch
            pytorch_index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
            if os.path.exists(pytorch_index_path):
                # The weight is stored in multiple files
                f = open(pytorch_index_path, "r", encoding="utf-8")
                pytorch_index = json.load(f)["weight_map"]
                pytorch_filename = None
            else:
                # The weight is stored in a single file
                pytorch_index = None
                pytorch_filename = "pytorch_model.bin"
            
            # For PyTorch files, since "opening" it is slow (due to deserialization),
            # we open it only once and then store the opened files in a dictionary.
            # We add `mmap=True` to avoid loading the entire file into memory.
            opened_files = {}
            def weight_getter_real(item: RegisteredWeightItem):
                file_name = pytorch_index[item.key] if pytorch_index is not None else pytorch_filename
                file_path = os.path.join(model_path, file_name)
                if file_path not in opened_files:
                    opened_files[file_path] = torch.load(file_path, map_location="cuda", mmap=True)
                file = opened_files[file_path]
                return file[item.key].to(item.dtype)
            getter = weight_getter_real

    def get_weight_type():
        if model_config.quantization_config is None:
            return LlamaWeight
        elif model_config.quantization_config["quant_method"] == "marlin":
            return LLamaWeightMarlin
        else:
            raise ValueError(f"Unsupported quantization method: {model_config.quantization_config['quant_method']}")

    weight = get_weight_type()(
        model_config,
        model_config.dtype
    )
    weight.load_weights(getter)
    return weight


