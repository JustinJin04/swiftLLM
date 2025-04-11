import argparse
import safetensors
import torch

from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types


class MacheteLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        self.register_buffer('B', torch.empty((self.in_features // 8, self.out_features), dtype=torch.int32))
        self.register_buffer('s', torch.empty((self.in_features // self.group_size, self.out_features), dtype=torch.bfloat16))

    def transform_w_q(self, weight: torch.Tensor):
        # weight: [out_features, in_features // 8] torch.int32
        assert weight.shape == (self.out_features, self.in_features // 8)
        assert weight.dtype == torch.int32
        weight = weight.t()

        weight = ops.machete_prepack_B(weight.t().contiguous().t(),
                                        a_type=torch.bfloat16,
                                        b_type=scalar_types.uint4b8,
                                        group_scales_type=torch.bfloat16)

        assert weight.shape == self.B.shape
        assert weight.is_contiguous()
        self.B.copy_(weight)
    
    def transform_w_s(self, scale: torch.Tensor):
        # scale: [out_features, in_features // self.group_size], torch.bfloat16
        assert scale.shape == (self.out_features, self.in_features // self.group_size)
        assert scale.dtype == torch.bfloat16
        assert scale.is_contiguous()

        self.s.copy_(scale.t().contiguous())
    
    def forward(self, x):
        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (self.out_features, )

        out = ops.machete_mm(a=x_2d,
                             b_q=self.B,
                             b_type=scalar_types.uint4b8,
                             b_group_zeros=None,
                             b_group_scales=self.s,
                             b_group_size=self.group_size)
        out = out.reshape(out_shape)
        assert out.is_contiguous()
        return out


def get_tensor_packed(tensor_path: str)->torch.Tensor:
    with safetensors.safe_open(tensor_path, framework="pt", device="cuda") as f:
        for k in f.keys():
            if(k == "model.layers.0.mlp.down_proj.weight_packed"):
                return f.get_tensor(k)

def get_tensor_scale(tensor_path: str)->torch.Tensor:
    with safetensors.safe_open(tensor_path, framework="pt", device="cuda") as f:
        for k in f.keys():
            if(k == "model.layers.0.mlp.down_proj.weight_scale"):
                return f.get_tensor(k)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensor_path",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main():
    args = get_args()

    weight_packed = get_tensor_packed(args.tensor_path)
    weight_scale = get_tensor_scale(args.tensor_path)

    infeatures = weight_packed.shape[1] * 8
    outfeatures = weight_packed.shape[0]
    group_size = 128

    layer = MacheteLayer(infeatures, outfeatures, group_size)
    layer.transform_w_q(weight_packed)
    layer.transform_w_s(weight_scale)

    x = torch.randn(1, infeatures, dtype=torch.bfloat16).cuda()
    out = layer(x)
    print(f"{out.sum()}")


if __name__ == "__main__":
    main()