"""Exact ctypes ABI declarations vendored from BCIR aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d."""

import ctypes as ct

_FP = ct.POINTER(ct.c_float)

_UP = ct.POINTER(ct.c_uint32)

_SZ = ct.c_size_t

_MM = ct.CFUNCTYPE(
    None, ct.c_void_p, ct.c_int, ct.c_int, _SZ, _SZ, _SZ, ct.c_float, _FP, _FP, ct.c_float, _FP
)

_SILU = ct.CFUNCTYPE(None, ct.c_void_p, _SZ, _FP, _FP)

class _Spec(ct.Structure):
    _fields_ = [
        (n, ct.c_uint32)
        for n in ("vocab", "width", "heads", "kvheads", "layers", "ff", "batch", "time", "tied")
    ] + [("rope_base", ct.c_double), ("rms_eps", ct.c_double)]

class _Plan(ct.Structure):
    _fields_ = [("spec", _Spec)] + [(n, _SZ) for n in ("parameters", "activations", "scratch")]

class _Provider(ct.Structure):
    _fields_ = [("ctx", ct.c_void_p), ("mm", _MM), ("silu", _SILU)]

class _State(ct.Structure):
    _fields_ = (
        [("plan", _Plan)]
        + [
            (n, _FP)
            for n in ("weights", "gradients", "moment1", "moment2", "activations", "scratch")
        ]
        + [
            (n + "_capacity", _SZ)
            for n in ("weight", "gradient", "moment1", "moment2", "activation", "scratch")
        ]
        + [
            ("step", ct.c_uint64),
            ("beta1_power", ct.c_double),
            ("beta2_power", ct.c_double),
            ("provider", _Provider),
            ("pool", ct.c_void_p),  # ABI 2 (NDT-THREADS): the pool the kernels run on
            ("contract", ct.c_uint32),  # ABI 3 (NDT-FMA): the GEMM contract, _CONTRACTS
        ]
    )

class _Optimizer(ct.Structure):
    _fields_ = [
        (n, ct.c_double) for n in ("lr", "beta1", "beta2", "epsilon", "weight_decay", "grad_clip")
    ]

class _Event(ct.Structure):
    _fields_ = [("loss", ct.c_double), ("grad_norm", ct.c_double), ("step", ct.c_uint64)]
