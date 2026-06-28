#!/usr/bin/env bash
# Build the bcir_cfront C twin (the `test_cfront` driver) under AddressSanitizer + UBSan and run
# EVERY runtime/c/cfront_*.c fixture plus a bounded differential fuzz campaign through it -- so a
# memory bug (buffer overflow, use-after-free, UB) INSIDE the compiler's own lexer/parser/lowering/
# emit is caught (the existing fuzzers only compared the two rails' OUTPUTS, never sanitized the twin).
# A bounded Valgrind pass over a fixture subset (plain -O0 -g twin) catches definite leaks / invalid
# accesses the compiler-rt sanitizers might miss. Mirrors the style + tool-resolution of check_runtime.sh
# and fuzz_streampack.sh: every stage guards on `command -v`/a real link probe and prints `SKIP <stage>
# (tool absent)` rather than failing when a tool is missing; when a tool IS present the stage must pass.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
# The EXACT source list + include path check_runtime.sh builds the `test_cfront` driver from (the C main
# that wraps the twin and prints the oracle summary). Keep this in lockstep with check_runtime.sh.
CFRONT_SRCS="${C}/bcir_cfront.c ${C}/bcir_cpp.c ${C}/bcir_verify.c ${C}/bcir_runtime.c ${C}/test_cfront.c"
# Bounded campaign sizes (tuned so the whole script stays well under ~60s on one core).
N_VALID="${SANITIZE_N_VALID:-300}"        # random VALID programs (the rich cfront subset generator)
M_MALFORMED="${SANITIZE_M_MALFORMED:-400}" # random MALFORMED inputs (the totality contract: never crash)
FUZZ_SEED="${SANITIZE_SEED:-20240628}"     # deterministic: no unseeded randomness
VG_SUBSET="${SANITIZE_VG_SUBSET:-10}"      # how many fixtures the (slow ~20x) Valgrind pass covers

tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT
fail=0
FIXTURES="$(cd "${C}" && ls cfront_*.c)"
SAN="-fsanitize=address,undefined -fno-sanitize-recover=all -g -O1"
export ASAN_OPTIONS="halt_on_error=1:detect_leaks=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"

# Build the sanitized driver with $1, trying c23 -> c2x -> c11 (gcc rejects -std=c23; clang accepts it).
# Echoes the produced binary path on success; nonzero (no echo) on failure.
build_san() {  # <compiler> <out-path>
  local cc="$1" out="$2" std
  for std in c23 c2x c11; do
    if "${cc}" -std=${std} ${SAN} -I "${C}" ${CFRONT_SRCS} -o "${out}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

# Probe whether a compiler can actually LINK the sanitizer runtime (clang here often lacks compiler-rt,
# exactly as fuzz_streampack.sh guards: a missing libclang_rt.asan makes the link fail). Only a compiler
# that links a trivial ASan/UBSan program is usable as a sanitizer engine.
can_sanitize() {  # <compiler>
  printf 'int main(void){return 0;}' \
    | "$1" -fsanitize=address,undefined -fno-sanitize-recover=all -x c - -o /dev/null 2>/dev/null
}

# Run every fixture + the fuzz campaign through one sanitized driver. Sets fail=1 on any diagnostic.
run_san_engine() {  # <label> <san-binary>
  local label="$1" bin="$2" fx errf rc
  local nfix; nfix=$(printf '%s\n' ${FIXTURES} | wc -l)
  echo "[sanitize] ${label}: all ${nfix} cfront_*.c fixtures through the sanitized twin"
  local nfx=0 ffail=0
  for fx in ${FIXTURES}; do
    nfx=$((nfx+1)); errf="${tmp}/fxerr.txt"
    "${bin}" "${C}/${fx}" >/dev/null 2>"${errf}"; rc=$?
    if [ "${rc}" -ne 0 ] || grep -qiE "AddressSanitizer|UndefinedBehaviorSanitizer|runtime error|heap-buffer-overflow|use-after-free|stack-buffer" "${errf}"; then
      echo "  FAIL: ${label} sanitizer diagnostic on fixture ${fx} (rc=${rc}):"; sed 's/^/    /' "${errf}" | head -25; ffail=1; fail=1
    fi
  done
  [ "${ffail}" -eq 0 ] && echo "  PASS ${label} fixtures (${nfx} cfront_*.c, no ASan/UBSan diagnostic)"

  echo "[sanitize] ${label}: bounded fuzz campaign (${N_VALID} valid + ${M_MALFORMED} malformed, seed ${FUZZ_SEED})"
  # Drive the EXISTING generators (no new generator): the rich valid-program generator in
  # tools/c/fuzz_cfront.py (Gen.program) + the malformed corruptor in cfront.cfuzz (gen_program+mutate).
  # Each program is written to a file and fed to the sanitized twin; any sanitizer error / crash fails.
  SAN_BIN="${bin}" SAN_N="${N_VALID}" SAN_M="${M_MALFORMED}" SAN_SEED="${FUZZ_SEED}" \
  python3 - "${ROOT}" <<'PY'
import os, sys, random, subprocess, tempfile
root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "tools", "c"))
sys.path.insert(0, root)
import fuzz_cfront                                  # Gen.program() -- the rich VALID-program generator
from bcir.frontends.cfront import cfuzz            # gen_program() + mutate() -- the MALFORMED corruptor
exe = os.environ["SAN_BIN"]
N, M, seed = int(os.environ["SAN_N"]), int(os.environ["SAN_M"]), int(os.environ["SAN_SEED"])
d = tempfile.mkdtemp(prefix="san_cfront_")
env = dict(os.environ, ASAN_OPTIONS="halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
def feed(src):
    p = os.path.join(d, "f.c")
    with open(p, "w") as fh: fh.write(src)
    r = subprocess.run([exe, p], capture_output=True, text=True, env=env)
    bad = (r.returncode < 0  # killed by a signal (e.g. ASan SIGABRT escaping the exit code)
           or "Sanitizer" in r.stderr or "runtime error" in r.stderr)
    return bad, r.returncode, r.stderr
bad = 0
rng = random.Random(seed)                          # deterministic VALID campaign
for i in range(N):
    src = fuzz_cfront.Gen(rng).program().source
    b, rc, err = feed(src)
    if b:
        bad += 1
        sys.stderr.write(f"  FAIL: sanitizer error on VALID program #{i} (rc={rc}):\n{src}\n{err[:1200]}\n")
        break
rng = random.Random(seed)                          # deterministic MALFORMED campaign (totality contract)
for i in range(M):
    src = cfuzz.gen_program(rng)
    for _ in range(rng.randint(1, 5)):
        src = cfuzz.mutate(src, rng)
    b, rc, err = feed(src)
    if b:
        bad += 1
        sys.stderr.write(f"  FAIL: sanitizer error / crash on MALFORMED input #{i} (rc={rc}):\n{src!r}\n{err[:1200]}\n")
        break
sys.exit(1 if bad else 0)
PY
  if [ $? -ne 0 ]; then
    echo "  FAIL: ${label} fuzz campaign tripped the sanitizer"; fail=1
  else
    echo "  PASS ${label} fuzz (${N_VALID} valid + ${M_MALFORMED} malformed clean under ASan/UBSan)"
  fi
}

# ----- stage 1: clang ASan/UBSan (primary) ----------------------------------------------------------
if command -v clang >/dev/null 2>&1; then
  if can_sanitize clang; then
    if build_san clang "${tmp}/cfront_clang_san"; then
      run_san_engine "clang ASan/UBSan" "${tmp}/cfront_clang_san"
    else
      echo "[sanitize] FAIL: clang sanitizer build of the twin failed"; fail=1
    fi
  else
    echo "[sanitize] SKIP clang ASan/UBSan (tool absent: clang has no compiler-rt / libclang_rt.asan to link)"
  fi
else
  echo "[sanitize] SKIP clang ASan/UBSan (tool absent: no clang)"
fi

# ----- stage 2: gcc ASan/UBSan (bonus -- and the real engine when clang lacks compiler-rt) -----------
if command -v gcc >/dev/null 2>&1; then
  if can_sanitize gcc; then
    if build_san gcc "${tmp}/cfront_gcc_san"; then
      run_san_engine "gcc ASan/UBSan" "${tmp}/cfront_gcc_san"
    else
      echo "[sanitize] FAIL: gcc sanitizer build of the twin failed"; fail=1
    fi
  else
    echo "[sanitize] SKIP gcc ASan/UBSan (tool absent: gcc has no libasan to link)"
  fi
else
  echo "[sanitize] SKIP gcc ASan/UBSan (tool absent: no gcc)"
fi

# Note when no compiler could link a sanitizer runtime (only happens in an environment with neither a
# compiler-rt clang nor a libasan gcc -- not the CI runners, which always run at least one engine above).
if ! can_sanitize clang && ! can_sanitize gcc; then
  echo "[sanitize] SKIP all ASan/UBSan stages (no compiler can link the sanitizer runtime)"
fi

# ----- stage 3: bounded Valgrind pass (plain -O0 -g twin, subset of fixtures) ------------------------
# Valgrind catches definite leaks / invalid accesses on a NON-sanitized build (~20x slowdown), so cap
# the subset. SANITIZE_SKIP_VALGRIND=1 lets CI drop the heaviest stage while keeping ASan/UBSan always-on.
if [ "${SANITIZE_SKIP_VALGRIND:-0}" = "1" ]; then
  echo "[sanitize] SKIP valgrind (SANITIZE_SKIP_VALGRIND=1)"
elif ! command -v valgrind >/dev/null 2>&1; then
  echo "[sanitize] SKIP valgrind (tool absent: no valgrind)"
else
  # Build a plain (no-sanitizer) -O0 -g twin -- prefer clang, fall back to gcc; try c23 -> c2x -> c11.
  PLAIN=""
  for cc in clang gcc; do
    command -v "${cc}" >/dev/null 2>&1 || continue
    for std in c23 c2x c11; do
      if "${cc}" -std=${std} -O0 -g -I "${C}" ${CFRONT_SRCS} -o "${tmp}/cfront_plain" 2>/dev/null; then
        PLAIN="${tmp}/cfront_plain"; break 2
      fi
    done
  done
  if [ -z "${PLAIN}" ]; then
    echo "[sanitize] SKIP valgrind (could not build a plain twin)"
  else
    # A representative subset: the first VG_SUBSET fixtures (the foundational ones -- regmap/array/deref/
    # callgraph/branch/loops -- exercise lex/parse/lower/emit broadly).
    VG_FIX="$(printf '%s\n' ${FIXTURES} | head -n "${VG_SUBSET}")"
    nvg=0; vgfail=0
    echo "[sanitize] valgrind (--leak-check=full, definite leaks) over ${VG_SUBSET} fixtures (plain -O0 -g twin)"
    for fx in ${VG_FIX}; do
      nvg=$((nvg+1))
      valgrind --leak-check=full --error-exitcode=1 --errors-for-leak-kinds=definite -q \
        "${PLAIN}" "${C}/${fx}" >/dev/null 2>"${tmp}/vg.txt"
      if [ $? -ne 0 ]; then
        echo "  FAIL: valgrind found a definite leak / invalid access on ${fx}:"; sed 's/^/    /' "${tmp}/vg.txt" | head -30; vgfail=1; fail=1
      fi
    done
    [ "${vgfail}" -eq 0 ] && echo "  PASS valgrind (${nvg} fixtures, no definite leak / invalid access)"
  fi
fi

# ----- summary --------------------------------------------------------------------------------------
if [ "${fail}" -eq 0 ]; then
  echo "[sanitize] all sanitizer/valgrind checks passed"
  exit 0
else
  echo "[sanitize] FAILURES above"
  exit 1
fi
