"""Polar Express matrix sign function.

Vendored from thinking-machines-lab/manifolds, src/msign.py, commit
89dcae50f01af59f1e0570289474da3a2ecaa60b (MIT license; see
third_party/manifolds/). Identical algorithm and operation order, with one
deliberate change approved in PLAN.md ("Numerical precision": fp32 in our
code, bf16 only in faithful reruns): the compute dtype is parameterized,
default fp32, where the original hardcodes bfloat16. ``dtype=torch.bfloat16``
reproduces the vendored numerics exactly. ``G.to(dtype, copy=True)`` forces a
copy so the in-place normalization never mutates the caller's tensor (the
original's ``G.bfloat16()`` always copied).
"""

import torch

ABC_LIST: list[tuple[float, float, float]] = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

# safety factor for numerical stability (but exclude last polynomial)
ABC_LIST_STABLE: list[tuple[float, float, float]] = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in ABC_LIST[:-1]
] + [ABC_LIST[-1]]

POLAR_EXPRESS_STEPS: int = 10


@torch.no_grad()
def msign(G: torch.Tensor, steps: int = 10, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Polar Express algorithm for the matrix sign function:
    https://arxiv.org/abs/2505.16932
    """
    assert G.ndim >= 2
    should_transpose: bool = G.size(-2) > G.size(-1)

    x = G.to(dtype, copy=True)
    if should_transpose:
        x = x.mT

    x /= x.norm(dim=(-2, -1), keepdim=True) * 1.01
    for step in range(steps):
        a, b, c = ABC_LIST_STABLE[step] if step < len(ABC_LIST_STABLE) else ABC_LIST_STABLE[-1]
        s = x @ x.mT
        # goal is to compute x = a x + b S x + c S^2 x
        # we can break this up into: x = (a I + (b I + c S) S) x
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x

    if should_transpose:
        x = x.mT
    x = torch.nan_to_num(x)
    return x.float()
