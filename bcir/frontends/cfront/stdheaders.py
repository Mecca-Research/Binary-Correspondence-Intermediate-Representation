"""The standard headers the frontend supplies itself (CSUB-1): the text of `<stdint.h>`, `<stddef.h>`, `<stdbool.h>`,
`<limits.h>`, `<float.h>`, `<iso646.h>`, `<stdalign.h>` and `<math.h>`'s constants, for one target.

A `#include <...>` no search path holds had been read as nothing, the frontend modeling the standard *types*
intrinsically (`uint32_t`, `size_t`, `wchar_t`) and none of the *macros*: `UINT64_C(1)` was a call of an undefined
function, `INT_MAX` an undeclared name, `offsetof` unknown (the C subset audit's CSUB-1). Each header below is now
read where one is named, as the text of a header file would be, so a unit sees exactly the macros and typedefs the
standard puts in it -- and `#ifndef UINT64_C` before a unit's own fallback definition reads what a real header gives.

The values are the target's, as Clang lays the target out (`abi.TargetABI`), and each macro's type is the one C
gives it: a limit has the type of an object of its type after the integer promotions (C11 7.20.2p2, 5.2.4.2.1p1), so
`UINT8_MAX` and `UINT8_C(1)` are `int`s and `UINT32_MAX` an `unsigned int`; `SIZE_MAX` and `PTRDIFF_MAX` are typed
as the target's `size_t` and `ptrdiff_t` are (`unsigned long` on LP64 Linux, `unsigned long long` on Windows x64,
`unsigned int` on i386). Clang's freestanding headers agree everywhere except where they give an `unsigned` type C
gives `int` (`UINT8_C`, `UINT16_C`, and the `wchar_t` and `wint_t` limits of Windows); the C library's own headers
agree with C there, and `MB_LEN_MAX` is the hosted C library's (16 in glibc, 5 in the MSVC runtime), as the rails
declare a hosted implementation (`__STDC_HOSTED__ 1`). The floating limits are spelled as Clang predefines them, so
the resident backend reads the same constants.

What is left out is left out on purpose, and a unit that names it is refused for that name, never miscompiled:
`<stdint.h>`'s `int_fastN_t` family, whose width the C library chooses (glibc and Clang's freestanding header disagree
on x86-64); `FLT_ROUNDS`, the current rounding mode, a runtime query; `<float.h>`'s C23 additions; `<stddef.h>`'s
`nullptr_t` and `unreachable()`; and `<stdnoreturn.h>`, whose `_Noreturn` neither rail reads yet. Of `<math.h>` only
its constants (`NAN`, `INFINITY`, `HUGE_VAL`, `HUGE_VALF`, `HUGE_VALL`) and its classification macros (`isfinite`,
`isnan`, `isinf`, `signbit`, `isnormal`, as glibc spells them) are here; its functions are the calls the lowering
already knows, and `fpclassify` and the comparison macros (`isgreater`, ...) are not yet modeled.

A typedef here is marked `__bcir_builtin`: the parser defines it and the linkable emit does not write it out again,
including the real header instead (`emit._STDDEF_NAME`). The twin builds the same texts (`bcir_cpp.c`,
`builtin_header`), and `bcir/tests/test_c_cfront.py` holds each rail's expansion of every macro to the other's and
to Clang's on every target."""

from __future__ import annotations

from .abi import TargetABI

#: The headers this module supplies.
NAMES = (
    "stdint.h",
    "stddef.h",
    "stdbool.h",
    "limits.h",
    "float.h",
    "iso646.h",
    "stdalign.h",
    "math.h",
)

_I8, _I16, _I32 = 127, 32767, 2147483647
_I64 = 9223372036854775807
_U64 = 18446744073709551615


def _facts(abi: TargetABI) -> dict:
    """The parts of the target the headers read: the C type of each typedef'd integer and its constant suffixes."""
    i64 = "long" if abi.long_size == 8 else "long long"
    s64 = "L" if abi.long_size == 8 else "LL"
    ptr = i64 if abi.pointer_size == 8 else "int"
    sptr = s64 if abi.pointer_size == 8 else ""
    msvc = abi.triple.endswith("msvc")
    if abi.long_double_size == 8:
        ldbl = "double"
    elif abi.triple.startswith(("x86_64", "i386")):
        ldbl = "x87"
    else:
        ldbl = "quad"
    return {
        "i64": i64,
        "s64": s64,
        "ptr": ptr,
        "sptr": sptr,
        "msvc": msvc,
        "ldbl": ldbl,
        "eval": 2 if abi.triple.startswith("i386") else 0,
    }


def _signed(name: str, bits: int, suffix: str) -> list[str]:
    top = {8: _I8, 16: _I16, 32: _I32, 64: _I64}[bits]
    return [f"#define {name}_MIN (-{top}{suffix}-1)", f"#define {name}_MAX {top}{suffix}"]


def _unsigned_max(name: str, bits: int, suffix: str) -> str:
    """`name_MAX` of an unsigned type of `bits`: a 8- or 16-bit one is promoted to `int` (no suffix)."""
    top = {8: 255, 16: 65535, 32: 4294967295, 64: _U64}[bits]
    return f"#define {name}_MAX {top}{suffix}"


def _stdint(abi: TargetABI) -> list[str]:
    f = _facts(abi)
    s64 = f["s64"]
    out = []
    for bits in (8, 16, 32, 64):
        out.append(f"__bcir_builtin typedef int{bits}_t int_least{bits}_t;")
        out.append(f"__bcir_builtin typedef uint{bits}_t uint_least{bits}_t;")
    out.append(f"__bcir_builtin typedef {f['i64']} intmax_t;")
    out.append(f"__bcir_builtin typedef unsigned {f['i64']} uintmax_t;")
    for bits, sfx, usfx in ((8, "", ""), (16, "", ""), (32, "", "U"), (64, s64, "U" + s64)):
        for kind in (f"INT{bits}", f"INT_LEAST{bits}"):
            out += _signed(kind, bits, sfx)
        for kind in (f"UINT{bits}", f"UINT_LEAST{bits}"):
            out.append(_unsigned_max(kind, bits, usfx))
    if abi.pointer_size == 8:
        out += _signed("INTPTR", 64, f["sptr"]) + [_unsigned_max("UINTPTR", 64, "U" + f["sptr"])]
        out += _signed("PTRDIFF", 64, f["sptr"]) + [_unsigned_max("SIZE", 64, "U" + f["sptr"])]
    else:
        out += _signed("INTPTR", 32, "") + [_unsigned_max("UINTPTR", 32, "U")]
        out += _signed("PTRDIFF", 32, "") + [_unsigned_max("SIZE", 32, "U")]
    out += _signed("INTMAX", 64, s64) + [_unsigned_max("UINTMAX", 64, "U" + s64)]
    out += _signed("SIG_ATOMIC", 32, "")
    if abi.wchar_size == 2:  # an `unsigned short`: promoted to `int`
        out += ["#define WCHAR_MIN 0", "#define WCHAR_MAX 65535"]
        out += ["#define WINT_MIN 0", "#define WINT_MAX 65535"]
    else:
        if abi.wchar_signed:
            out += _signed("WCHAR", 32, "")
        else:
            out += ["#define WCHAR_MIN 0U", "#define WCHAR_MAX 4294967295U"]
        out += ["#define WINT_MIN 0U", "#define WINT_MAX 4294967295U"]
    for bits in (8, 16):
        out += [f"#define INT{bits}_C(c) c", f"#define UINT{bits}_C(c) c"]
    out += ["#define INT32_C(c) c", "#define UINT32_C(c) c ## U"]
    out += [f"#define INT64_C(c) c ## {s64}", f"#define UINT64_C(c) c ## U{s64}"]
    out += [f"#define INTMAX_C(c) c ## {s64}", f"#define UINTMAX_C(c) c ## U{s64}"]
    wbits = 8 * abi.pointer_size
    out.append("#if __STDC_VERSION__ >= 202311L")
    for bits in (8, 16, 32, 64):
        for kind in (f"INT{bits}", f"UINT{bits}", f"INT_LEAST{bits}", f"UINT_LEAST{bits}"):
            out.append(f"#define {kind}_WIDTH {bits}")
    for kind in ("INTPTR", "UINTPTR", "PTRDIFF", "SIZE"):
        out.append(f"#define {kind}_WIDTH {wbits}")
    out += ["#define INTMAX_WIDTH 64", "#define UINTMAX_WIDTH 64", "#define SIG_ATOMIC_WIDTH 32"]
    out += [f"#define WCHAR_WIDTH {8 * abi.wchar_size}", f"#define WINT_WIDTH {8 * abi.wchar_size}"]
    out.append("#endif")
    return out


def _stddef(abi: TargetABI) -> list[str]:
    f = _facts(abi)
    out = [f"__bcir_builtin typedef {f['ptr']} ptrdiff_t;"]
    if f["msvc"]:
        out.append("__bcir_builtin typedef double max_align_t;")
    else:  # Clang's: a `long long` aligned as `__alignof__` reads it (8, on i386 too), then a `long double`
        out.append(
            "__bcir_builtin typedef struct { _Alignas(8) long long __bcir_max_align_ll; "
            "long double __bcir_max_align_ld; } max_align_t;"
        )
    out += ["#define NULL ((void*)0)", "#define offsetof(t, d) __builtin_offsetof(t, d)"]
    return out


def _stdbool(abi: TargetABI) -> list[str]:
    # C23 makes `bool`, `true` and `false` keywords: the header then defines only its obsolescent flag (7.18)
    return [
        "#if __STDC_VERSION__ < 202311L",
        "#define bool _Bool",
        "#define true 1",
        "#define false 0",
        "#endif",
        "#define __bool_true_false_are_defined 1",
    ]


def _limits(abi: TargetABI) -> list[str]:
    f = _facts(abi)
    long_bits = 8 * abi.long_size
    lsfx = "L"
    out = ["#define CHAR_BIT 8", f"#define MB_LEN_MAX {5 if f['msvc'] else 16}"]
    out += ["#define SCHAR_MIN (-127-1)", "#define SCHAR_MAX 127", "#define UCHAR_MAX 255"]
    if abi.char_signed:
        out += ["#define CHAR_MIN (-127-1)", "#define CHAR_MAX 127"]
    else:
        out += ["#define CHAR_MIN 0", "#define CHAR_MAX 255"]
    out += ["#define SHRT_MIN (-32767-1)", "#define SHRT_MAX 32767", "#define USHRT_MAX 65535"]
    out += [
        "#define INT_MIN (-2147483647-1)",
        "#define INT_MAX 2147483647",
        "#define UINT_MAX 4294967295U",
    ]
    if long_bits == 64:
        out += _signed("LONG", 64, lsfx) + ["#define ULONG_MAX 18446744073709551615UL"]
    else:
        out += _signed("LONG", 32, lsfx) + ["#define ULONG_MAX 4294967295UL"]
    out += _signed("LLONG", 64, "LL") + ["#define ULLONG_MAX 18446744073709551615ULL"]
    out.append("#if __STDC_VERSION__ >= 202311L")
    out += ["#define BOOL_WIDTH 1"]
    for kind, bits in (
        ("CHAR", 8),
        ("SCHAR", 8),
        ("UCHAR", 8),
        ("SHRT", 16),
        ("USHRT", 16),
        ("INT", 32),
        ("UINT", 32),
        ("LONG", long_bits),
        ("ULONG", long_bits),
        ("LLONG", 64),
        ("ULLONG", 64),
    ):
        out.append(f"#define {kind}_WIDTH {bits}")
    out.append("#endif")
    return out


#: <float.h>'s limits of `float` and `double`, as Clang predefines them on every target here.
_FLT = (
    ("MANT_DIG", "24"),
    ("DIG", "6"),
    ("MIN_EXP", "(-125)"),
    ("MIN_10_EXP", "(-37)"),
    ("MAX_EXP", "128"),
    ("MAX_10_EXP", "38"),
    ("MAX", "3.40282347e+38F"),
    ("EPSILON", "1.19209290e-7F"),
    ("MIN", "1.17549435e-38F"),
    ("TRUE_MIN", "1.40129846e-45F"),
    ("DECIMAL_DIG", "9"),
    ("HAS_SUBNORM", "1"),
)
_DBL = (
    ("MANT_DIG", "53"),
    ("DIG", "15"),
    ("MIN_EXP", "(-1021)"),
    ("MIN_10_EXP", "(-307)"),
    ("MAX_EXP", "1024"),
    ("MAX_10_EXP", "308"),
    ("MAX", "1.7976931348623157e+308"),
    ("EPSILON", "2.2204460492503131e-16"),
    ("MIN", "2.2250738585072014e-308"),
    ("TRUE_MIN", "4.9406564584124654e-324"),
    ("DECIMAL_DIG", "17"),
    ("HAS_SUBNORM", "1"),
)
#: ... and of `long double`, by its format: the x87 80-bit one (x86-64 and i386 Linux), IEEE binary128 (AArch64 and
#: RISC-V Linux) and `double` (Windows).
_LDBL = {
    "x87": (
        ("MANT_DIG", "64"),
        ("DIG", "18"),
        ("MIN_EXP", "(-16381)"),
        ("MIN_10_EXP", "(-4931)"),
        ("MAX_EXP", "16384"),
        ("MAX_10_EXP", "4932"),
        ("MAX", "1.18973149535723176502e+4932L"),
        ("EPSILON", "1.08420217248550443401e-19L"),
        ("MIN", "3.36210314311209350626e-4932L"),
        ("TRUE_MIN", "3.64519953188247460253e-4951L"),
        ("DECIMAL_DIG", "21"),
        ("HAS_SUBNORM", "1"),
    ),
    "quad": (
        ("MANT_DIG", "113"),
        ("DIG", "33"),
        ("MIN_EXP", "(-16381)"),
        ("MIN_10_EXP", "(-4931)"),
        ("MAX_EXP", "16384"),
        ("MAX_10_EXP", "4932"),
        ("MAX", "1.18973149535723176508575932662800702e+4932L"),
        ("EPSILON", "1.92592994438723585305597794258492732e-34L"),
        ("MIN", "3.36210314311209350626267781732175260e-4932L"),
        ("TRUE_MIN", "6.47517511943802511092443895822764655e-4966L"),
        ("DECIMAL_DIG", "36"),
        ("HAS_SUBNORM", "1"),
    ),
    "double": tuple((k, v + "L" if "e" in v else v) for k, v in _DBL),
}


def _float(abi: TargetABI) -> list[str]:
    f = _facts(abi)
    ldbl = _LDBL[f["ldbl"]]
    out = [f"#define FLT_EVAL_METHOD {f['eval']}", "#define FLT_RADIX 2"]
    out.append(f"#define DECIMAL_DIG {dict(ldbl)['DECIMAL_DIG']}")
    for prefix, table in (("FLT", _FLT), ("DBL", _DBL), ("LDBL", ldbl)):
        out += [f"#define {prefix}_{k} {v}" for k, v in table]
    return out


def _iso646(abi: TargetABI) -> list[str]:
    pairs = (
        ("and", "&&"),
        ("and_eq", "&="),
        ("bitand", "&"),
        ("bitor", "|"),
        ("compl", "~"),
        ("not", "!"),
        ("not_eq", "!="),
        ("or", "||"),
        ("or_eq", "|="),
        ("xor", "^"),
        ("xor_eq", "^="),
    )
    return [f"#define {k} {v}" for k, v in pairs]


def _stdalign(abi: TargetABI) -> list[str]:
    # C23 makes `alignas` and `alignof` keywords (6.4.1), and the header then defines nothing -- not its old flags
    # either, as GCC 13 and Clang 18 and 23 read C23 (7.15)
    return [
        "#if __STDC_VERSION__ < 202311L",
        "#define alignas _Alignas",
        "#define alignof _Alignof",
        "#define __alignas_is_defined 1",
        "#define __alignof_is_defined 1",
        "#endif",
    ]


def _math(abi: TargetABI) -> list[str]:
    # the constants and the classification macros, spelled as glibc spells them: `isinf` its sign-returning builtin
    return [
        '#define NAN (__builtin_nanf(""))',
        "#define INFINITY (__builtin_inff())",
        "#define HUGE_VAL (__builtin_huge_val())",
        "#define HUGE_VALF (__builtin_huge_valf())",
        "#define HUGE_VALL (__builtin_huge_vall())",
        "#define isfinite(x) __builtin_isfinite(x)",
        "#define isnan(x) __builtin_isnan(x)",
        "#define isinf(x) __builtin_isinf_sign(x)",
        "#define signbit(x) __builtin_signbit(x)",
        "#define isnormal(x) __builtin_isnormal(x)",
    ]


_BUILDERS = {
    "stdint.h": _stdint,
    "stddef.h": _stddef,
    "stdbool.h": _stdbool,
    "limits.h": _limits,
    "float.h": _float,
    "iso646.h": _iso646,
    "stdalign.h": _stdalign,
    "math.h": _math,
}


def clang_source(text: str) -> str:
    """The preprocessed text `text` as a C compiler reads it beside the real headers: each `__bcir_builtin` typedef
    line blank, its line kept so every other line keeps its number. The frontend reads those typedefs; a compiler
    that includes the real header defines them itself, `max_align_t` as a type no other definition is compatible
    with -- so the text Clang compiles in a behaviour-equivalence harness (`CompileResult.source`) is this one."""
    return "\n".join("" if ln.startswith("__bcir_builtin ") else ln for ln in text.split("\n"))


def header_text(name: str, abi: TargetABI) -> str | None:
    """The text of the standard header `name` on `abi`'s target, guarded so a second include reads nothing; None for
    a header this module does not supply."""
    build = _BUILDERS.get(name)
    if build is None:
        return None
    guard = "__BCIR_" + name.replace(".", "_").upper()
    return "\n".join([f"#ifndef {guard}", f"#define {guard} 1", *build(abi), "#endif"]) + "\n"
