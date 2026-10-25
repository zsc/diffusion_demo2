"""Optional pointwise or full-model compilation; parameters/checkpoints stay intact.

The GPU backend lowers these small PyTorch graphs to its installed Triton. GEMM,
convolution and SDPA remain vendor library calls. Enable explicitly after validation.
"""
import torch
from torch.nn import functional as F


def prepare_training_model(model, backend="eager", emulate_precision_casts=True):
    """Return the training callable while keeping the original checkpoint namespace.

    Call before DDP construction. Full compilation disables Dynamo's DDP graph
    partitioning; DDP still synchronizes gradients, with less communication overlap.
    Keep the original model for optimizer, EMA and checkpoint operations.
    """
    if backend != "compile_full":
        return configure_execution(model, backend, emulate_precision_casts)
    configure_execution(model, "eager", emulate_precision_casts)
    torch._dynamo.config.optimize_ddp = False
    return torch.compile(model, fullgraph=True, dynamic=False,
                         options={"emulate_precision_casts": emulate_precision_casts})


def rms(x, weight):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    return y.to(x.dtype) * weight.to(x.dtype)


def rope(x, cos, sin):
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = cos.to(x.dtype), sin.to(x.dtype)
    return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)


def swiglu(gate, up):
    return F.silu(gate) * up


def modulate(x, scale, shift):
    return x * (1 + scale) + shift


def residual(x, gate, branch):
    return x + gate * branch


def configure_execution(model, backend="eager", emulate_precision_casts=True):
    if backend not in ["eager", "contiguous", "compile_norm", "compile_pointwise"]:
        raise ValueError(f"Unknown execution backend: {backend}")
    model._contiguous_tokens = backend == "contiguous"
    from diffusion_demo.core.models.pixel_jit import RMSNorm, Attention, SwiGLU, Block
    compiled = {}
    def choose(fn):
        if fn not in compiled:
            compiled[fn] = torch.compile(fn, fullgraph=True, dynamic=False,
                                         options={"emulate_precision_casts": emulate_precision_casts})
        return compiled[fn]
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module._rms_impl = rms if backend in ["eager", "contiguous"] else choose(rms)
        if isinstance(module, Attention):
            module._contiguous_qk = backend == "contiguous"
            module._rope_impl = choose(rope) if backend == "compile_pointwise" else rope
        if isinstance(module, SwiGLU):
            module._activation_impl = choose(swiglu) if backend == "compile_pointwise" else swiglu
        if isinstance(module, Block):
            module._modulate_impl = choose(modulate) if backend == "compile_pointwise" else modulate
            module._residual_impl = choose(residual) if backend == "compile_pointwise" else residual
    model._modulate_impl = choose(modulate) if backend == "compile_pointwise" else modulate
    return model
