"""Python<->C parity for the multi-channel lowering decision (`runtime/c/bcir_channel.c`).

The C router consumes the same `channel.json` manifests and, for each claim, derives the same
required capabilities, the same per-channel eligibility, and the same routed backend as the oracle
(`bcir/channels`: `claim_required_caps` / `channel_suits` / `route_claim`). Toolchain-gated: the
quick tier still checks the oracle routing is computable; the C byte-for-byte parity runs under
c-runtime/thorough.
"""

import os
import shutil
import subprocess
import tempfile

from bcir.channel_plugin import load_manifest
from bcir.channels import channel_suits, claim_required_caps, route_claim
from bcir.model import Claim, Opcode, StrideClass

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_CHAN = os.path.join(_ROOT, "channels")
_CC = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")

# cpu (universal plugin), tpu (a capability plugin), pim (no caps -> the legacy per-kind rule).
_CHANNEL_FILES = ["example_cpu.channel.json", "example_tpu.channel.json", "example_pim.channel.json"]
# claims spanning the routing space: op-derived caps (reduce/gather/matmul) + every stride class.
_CLAIMS = [
    ("reduce.sum", StrideClass.UNIT),
    ("matmul.acc", StrideClass.TILE),
    ("gather.load", StrideClass.RANDOM),
    ("vector.add", StrideClass.UNIT),
    ("scalar.mov", StrideClass.SCALAR),
    ("c.bin.add", StrideClass.SCALAR),
    ("strided.load", StrideClass.STRIDED),
    ("tile.mac", StrideClass.TILE),
    ("cacheline.load", StrideClass.CACHELINE),
]


def _claim(op: str, sc: StrideClass) -> Claim:
    return Claim(id=1, opcode=Opcode.LOAD, op=op, stride_class=sc)


def _py_line(op: str, sc: StrideClass, mans) -> str:
    cl = _claim(op, sc)
    caps = ",".join(sorted(claim_required_caps(cl)))
    chosen = route_claim(cl, mans)
    route = chosen.name if chosen else "NONE"
    suits = ";".join(f"{m.name}={1 if channel_suits(cl, m) else 0}" for m in mans)
    return f"{op}|{int(sc)}|caps={caps}|route={route}|suits={suits}"


def _build(d: str) -> str:
    exe = os.path.join(d, "tch")
    for std in ("c23", "c11"):
        b = subprocess.run([_CC, f"-std={std}", "-O2", "-Wall", "-Wextra", "-I", _C,
                            os.path.join(_C, "bcir_channel.c"), os.path.join(_C, "test_channel.c"),
                            "-o", exe], capture_output=True, text=True)
        if b.returncode == 0:
            return exe
    raise AssertionError(f"channel router build failed:\n{b.stderr}")


def test_python_c_channel_routing_parity():
    mans = [load_manifest(os.path.join(_CHAN, f)) for f in _CHANNEL_FILES]
    if not _CC:                                  # quick tier: the oracle routing is deterministic.
        for op, sc in _CLAIMS:
            assert "route=" in _py_line(op, sc, mans)
        return
    paths = [os.path.join(_CHAN, f) for f in _CHANNEL_FILES]
    stdin = "".join(f"{op} {int(sc)}\n" for op, sc in _CLAIMS)
    with tempfile.TemporaryDirectory() as d:
        exe = _build(d)
        out = subprocess.run([exe, *paths], input=stdin, capture_output=True, text=True).stdout
    c_lines = [ln for ln in out.strip().splitlines() if ln]
    assert len(c_lines) == len(_CLAIMS), out
    for (op, sc), c_line in zip(_CLAIMS, c_lines):
        assert c_line == _py_line(op, sc, mans), f"{op}: parity diverged\n C: {c_line}\nPY: {_py_line(op, sc, mans)}"


def test_c_channel_parser_rejects_malformed():
    """The C channel.json reader fails cleanly on a missing name/kind and a non-object."""
    if not _CC:
        return
    with tempfile.TemporaryDirectory() as d:
        drv = os.path.join(d, "drv.c")
        open(drv, "w").write(
            '#include <stdio.h>\n#include <string.h>\n#include "bcir_channel.h"\n'
            'int main(int c,char**v){(void)c;bcir_channel ch;char dg[128];'
            'if(!bcir_channel_parse("{}",BCIR_CHANNEL_JSON_MAX_BYTES+1u,&ch,dg,sizeof dg))return 4;'
            'const char*j=v[1];memset(&ch,0xa5,sizeof ch);'
            'int r=bcir_channel_parse(j,strlen(j),&ch,dg,sizeof dg);'
            'if(!r)return 1;for(size_t i=0;i<sizeof ch;i++)if(((unsigned char*)&ch)[i])return 2;'
            'if(bcir_claim_required_caps(NULL)!=0||bcir_channel_suits(NULL,NULL)!=0||'
            'bcir_channel_route(NULL,NULL,1)!=-1)return 3;return 0;}\n')
        exe = os.path.join(d, "drv")
        b = subprocess.run([_CC, "-std=c11", "-O1", "-I", _C, os.path.join(_C, "bcir_channel.c"),
                            drv, "-o", exe], capture_output=True, text=True)
        assert b.returncode == 0, b.stderr
        for bad in ('not json', '{"kind":"cpu"}', '{"name":"x"}',
                    '{"name":"x" "kind":"cpu"}',                      # missing comma
                    '{"name":"x","kind":"cpu",}',                    # trailing comma
                    '{"name":"x","kind":"cpu","modeled":"true"}', # wrong value type
                    '{"name":"x","name":"y","kind":"cpu"}',       # duplicate key
                    '{"name":"x","kind":"cpu","capabilities":["telepathy"]}',
                    '{"name":"x","kind":"cpu","capabilities":["reduce","reduce"]}',
                    '{"name":"x","kind":"quantum"}',
                    '{"name":"x","kind":"cpu","provenance":"real","modeled":true}',
                    '{"name":"x\\nroot","kind":"cpu"}',             # escaped log/control injection
                    '{"name":"x","kind":"cpu","profile":{"x":[1,{bad}]}}',
                    '{"name":"x","kind":"cpu"} trailing',
                    '{"name":"' + ('x' * 80) + '","kind":"cpu"}'):
            assert subprocess.run([exe, bad]).returncode == 0, f"should reject: {bad}"
        # a well-formed minimal channel parses (returns non-zero from the probe == parse ok).
        assert subprocess.run([exe, '{"name":"x","kind":"cpu","capabilities":["universal"]}']).returncode == 1
