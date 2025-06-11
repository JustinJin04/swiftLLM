import torch
import torch.nn as nn
import numpy as np

from marlin import marlin_gemm, Layer as MarlinLayer


torch.manual_seed(0)
np.random.seed(0)
DEV=torch.device('cuda:0')


def gen_quant4(m, n, groupsize=-1):
    maxq = 2 ** 4 - 1
    w = torch.randn((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = MarlinLayer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return ref, q, s


def test_speedup(batch_size, infeatures, outfeatures, rounds):

    m=1
    k=infeatures
    n=outfeatures
    groupsize=128

    A = torch.randn((m, k), dtype=torch.half, device=DEV)
    B_ref, B, s = gen_quant4(k, n, groupsize=groupsize)

    for _ in range(rounds):
        C_ref = torch.matmul(A, B_ref)
    
    for _ in range(rounds):
        workspace = torch.zeros(n // 64 * 16, device=DEV)
        C = marlin_gemm(A, B, s, workspace)

    print(f"mismatch: {torch.max(torch.abs(C - C_ref))}")


def main():
    test_speedup(1, 14336*2, 4096, 10000)


if __name__ == "__main__":
    main()