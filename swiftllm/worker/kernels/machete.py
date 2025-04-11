import torch

from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types


class MacheteLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, dtype, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dtype = dtype
        
        self.register_buffer('B', torch.empty((self.in_features // 8, self.out_features), dtype=torch.int32, device="cuda"))
        self.register_buffer('s', torch.empty((self.in_features // self.group_size, self.out_features), dtype=self.dtype, device="cuda"))

    def transform_w_q(self, weight: torch.Tensor):
        # weight: [out_features, in_features // 8] torch.int32
        assert weight.shape == (self.out_features, self.in_features // 8)
        assert weight.dtype == torch.int32
        weight = weight.t()

        weight = ops.machete_prepack_B(weight.t().contiguous().t(),
                                        a_type=self.dtype,
                                        b_type=scalar_types.uint4b8,
                                        group_scales_type=self.dtype)

        assert weight.shape == self.B.shape
        assert weight.is_contiguous()
        assert self.B.device == weight.device
        self.B.copy_(weight)
    
    def transform_w_s(self, scale: torch.Tensor):
        # scale: [out_features, in_features // self.group_size], self.dtype
        assert scale.shape == (self.out_features, self.in_features // self.group_size)
        assert scale.dtype == self.dtype
        assert scale.is_contiguous()
        assert self.s.device == scale.device

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