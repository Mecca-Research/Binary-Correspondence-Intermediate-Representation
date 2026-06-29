#!/usr/bin/env bash
# Standalone build+run gate for the SYCL backend DIFFERENTIAL ORACLE (a SAXPY parallel_for).
#
# SYCL is a backend CHANNEL + a differential oracle, NEVER on the legality path
# (docs/SYCL_INTEROP.md). This gate proves the emitted single-source C++ SAXPY kernel
# (bcir.lower.c_kernel.emit_sycl_saxpy_c) genuinely COMPILES + RUNS and reproduces
# BCIR's own deterministic reference (out[i] = a*x[i] + y[i]) to float round-off:
#   * the PORTABLE scalar C++ fallback (no -fsycl) -- the real reference-verification work
#     (CI has no SYCL compiler), and
#   * the SYCL DEVICE path (-DBCIR_USE_SYCL -fsycl) IF a real SYCL compiler is detected
#     (icpx / acpp / clang++ -fsycl that actually compiles+links a SYCL probe).
#
# CRITICAL: SYCL is a C++ COMPILER MODE (-fsycl), NOT a c.call.libm: -l<lib> symbol edge --
# there is no link-flag rule for it (see linkflags.py, untouched). It is STANDALONE: plain
# C++17, no MLIR/LLVM cmake. It self-skips (exit 0) if no C++ compiler is present, exactly
# like check_handoff.sh.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX="${CXX:-$(command -v g++ || command -v clang++ || command -v c++ || true)}"

if [ -z "${CXX}" ]; then
  echo "[sycl-diff] no C++ compiler (g++/clang++/c++); skipping SYCL differential check." >&2
  exit 0
fi

tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Emit the kernel via the oracle (the same emitter the test pins) + a fixed-data main that prints results.
N=8
A=2.0
python3 - "${tmp}/saxpy.cpp" "${N}" "${A}" <<'PY' || { echo "[sycl-diff] FAIL: could not emit the SYCL SAXPY kernel"; exit 1; }
import sys
from bcir.lower.c_kernel import emit_sycl_saxpy_c
from bcir.kbcir.sycl_saxpy import saxpy_reference
out_path, n, a = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
x = [1.0, 2.0, 3.0, 0.5, -1.0, 4.0, -2.5, 0.0][:n]
y = [10.0, 20.0, 30.0, -0.5, 1.0, -4.0, 2.5, 7.0][:n]
ref = saxpy_reference(a, x, y)                 # BCIR's own deterministic reference
xb = ", ".join(f"{v:.8f}f" for v in x)
yb = ", ".join(f"{v:.8f}f" for v in y)
main = (
    "\n#include <cstdio>\nint main(void){\n"
    f"  float a = {a:.8f}f;\n"
    f"  float x[{n}] = {{{xb}}};\n"
    f"  float y[{n}] = {{{yb}}};\n"
    f"  float out[{n}];\n  bcir_saxpy(a, x, y, out);\n"
    f"  for (int i = 0; i < {n}; ++i) printf(\"%.6f\\n\", (double)out[i]);\n"
    "  return 0;\n}\n"
)
open(out_path, "w").write(emit_sycl_saxpy_c(n, "bcir_saxpy") + main)
open(out_path + ".ref", "w").write("\n".join(f"{v:.6f}" for v in ref) + "\n")
PY

# Compare emitted-kernel output against BCIR's reference to float round-off (1e-3 relative tolerance).
check_against_ref() {  # <produced-file>
  python3 - "$1" "${tmp}/saxpy.cpp.ref" <<'PY'
import sys
got = [float(v) for v in open(sys.argv[1]).read().split()]
ref = [float(v) for v in open(sys.argv[2]).read().split()]
if len(got) != len(ref):
    print(f"length mismatch got={len(got)} ref={len(ref)}"); sys.exit(1)
for g, r in zip(got, ref):
    if abs(g - r) > 1e-3 * (1.0 + abs(r)):
        print(f"diff too large: got={g} ref={r}"); sys.exit(1)
sys.exit(0)
PY
}

echo "[sycl-diff] compile the PORTABLE scalar C++17 fallback (no -fsycl) + run vs BCIR's SAXPY reference"
if "${CXX}" -std=c++17 -O2 "${tmp}/saxpy.cpp" -o "${tmp}/saxpy_fallback" 2>"${tmp}/fb.err"; then
  fb_out="$("${tmp}/saxpy_fallback")" || { echo "  FAIL: fallback run"; echo "${fb_out}"; exit 1; }
  echo "${fb_out}" > "${tmp}/fb.out"
  if check_against_ref "${tmp}/fb.out"; then
    echo "  PASS #sycl-fallback (scalar C++ a*x+y == BCIR reference to float round-off; no SYCL needed)"
  else
    echo "  FAIL: scalar fallback diverged from the BCIR SAXPY reference"; exit 1
  fi
else
  echo "  FAIL: portable C++ fallback did not build"; cat "${tmp}/fb.err"; exit 1
fi

# Detect a REAL SYCL compiler: a candidate that actually compiles+links a SYCL probe TU with -fsycl
# (so a plain clang++ without the SYCL runtime is correctly rejected). icpx / acpp / clang++ -fsycl.
SYCL_CXX=""; SYCL_FLAGS=""
printf '#include <sycl/sycl.hpp>\nint main(){ sycl::queue q; (void)q; return 0; }\n' > "${tmp}/probe.cpp"
for cand in "icpx -fsycl" "acpp " "clang++ -fsycl"; do
  set -- ${cand}
  tool="$1"; shift; flags="$*"
  command -v "${tool}" >/dev/null 2>&1 || continue
  # shellcheck disable=SC2086
  if "${tool}" -std=c++17 ${flags} "${tmp}/probe.cpp" -o "${tmp}/probe" 2>/dev/null; then
    SYCL_CXX="${tool}"; SYCL_FLAGS="${flags}"; break
  fi
done

if [ -n "${SYCL_CXX}" ]; then
  echo "[sycl-diff] a SYCL compiler is present (${SYCL_CXX} ${SYCL_FLAGS}); compile the DEVICE path (-DBCIR_USE_SYCL -fsycl)"
  # shellcheck disable=SC2086
  if "${SYCL_CXX}" -std=c++17 -O2 -DBCIR_USE_SYCL ${SYCL_FLAGS} "${tmp}/saxpy.cpp" -o "${tmp}/saxpy_device" 2>"${tmp}/dev.err"; then
    dev_out="$("${tmp}/saxpy_device")" || { echo "  FAIL: SYCL device run"; echo "${dev_out}"; exit 1; }
    echo "${dev_out}" > "${tmp}/dev.out"
    if check_against_ref "${tmp}/dev.out"; then
      echo "  PASS #sycl-device (parallel_for a*x+y == BCIR reference; the heterogeneous backend is measured)"
    else
      echo "  FAIL: SYCL device path diverged from the BCIR SAXPY reference"; exit 1
    fi
  else
    echo "  FAIL: SYCL device path did not build with -fsycl"; cat "${tmp}/dev.err"; exit 1
  fi
else
  echo "[sycl-diff] no SYCL compiler (icpx/acpp/clang++ -fsycl); #sycl-device path self-skipped (expected on CI)"
fi

# --- S2: the RESIDENT DISPATCH round-trip (the channel EXECUTES routed work) ----------------------------
# The probes above prove the emitted SAXPY kernel compiles + diffs STANDALONE. This drives the resident
# DISPATCHER (bcir.lower.sycl_dispatch.SyclDispatcher) end-to-end on fixed data and checks the EXECUTED
# output == BCIR's saxpy_reference -- the channel now has a dispatch path that RUNS routed work. The
# dispatcher self-selects the portable fallback (CI) or the -fsycl device path (when a SYCL compiler probes
# OK); it reports its `mode` so we PASS #sycl-dispatch honestly. The dispatcher uses a C++ compiler itself,
# so this stays self-skipping when none is present.
echo "[sycl-dispatch] resident DISPATCH round-trip (SyclDispatcher executes a*x+y == BCIR reference)"
disp_out="$(python3 - <<'PY'
import sys
from bcir.kbcir.sycl_saxpy import saxpy_reference
from bcir.lower.sycl_dispatch import SyclDispatcher, DispatchUnavailable
a = 2.0
x = [1.0, 2.0, 3.0, 0.5, -1.0, 4.0, -2.5, 0.0]
y = [10.0, 20.0, 30.0, -0.5, 1.0, -4.0, 2.5, 7.0]
disp = SyclDispatcher()
try:
    got = disp.run_saxpy(a, x, y)
except DispatchUnavailable as e:
    print("SKIP " + str(e).splitlines()[0]); sys.exit(0)
finally:
    disp.close()
ref = saxpy_reference(a, x, y)
for g, r in zip(got, ref):
    if abs(g - r) > 1e-3 * (1.0 + abs(r)):
        print(f"DIVERGE got={g} ref={r}"); sys.exit(1)
print("OK " + disp.mode)
PY
)" || { echo "  FAIL: the resident SYCL dispatch round-trip diverged from the BCIR reference"; echo "${disp_out}"; exit 1; }
case "${disp_out}" in
  "OK fallback")
    echo "  PASS #sycl-dispatch (portable fallback: SyclDispatcher executed a*x+y == BCIR reference)" ;;
  "OK sycl-device")
    echo "  PASS #sycl-dispatch + #sycl-device (the -fsycl dispatch executed a*x+y == BCIR reference)" ;;
  SKIP\ *)
    echo "  ${disp_out} (no C++ compiler; the dispatch round-trip self-skipped, expected when toolchain absent)" ;;
  *)
    echo "  FAIL: unexpected dispatch probe result: ${disp_out}"; exit 1 ;;
esac

echo "[sycl-dispatch] ok"
echo "[sycl-diff] ok"
