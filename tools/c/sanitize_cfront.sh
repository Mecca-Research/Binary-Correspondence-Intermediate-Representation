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
CLANG="${CLANG:-$(command -v clang || true)}"
if [ -n "${CLANG}" ]; then
  CLANG="$(command -v "${CLANG}" || true)"
fi
# The EXACT source list + include path check_runtime.sh builds the `test_cfront` driver from (the C main
# that wraps the twin and prints the oracle summary). Keep this in lockstep with check_runtime.sh.
CFRONT_SRCS="${C}/bcir_cfront.c ${C}/bcir_cpp.c ${C}/bcir_verify.c ${C}/bcir_runtime.c ${C}/test_cfront.c"
# Bounded campaign sizes (tuned so the whole script stays well under ~60s on one core).
N_VALID="${SANITIZE_N_VALID:-300}"        # random VALID programs (the rich cfront subset generator)
M_MALFORMED="${SANITIZE_M_MALFORMED:-400}" # random MALFORMED inputs (the totality contract: never crash)
FUZZ_SEED="${SANITIZE_SEED:-20240628}"     # deterministic: no unseeded randomness
VG_SUBSET="${SANITIZE_VG_SUBSET:-10}"      # how many fixtures the (slow ~20x) Valgrind pass covers
# SANITIZE_ENGINES_PARALLEL=1 overlaps the three independent engine stages (clang ASan/UBSan, gcc ASan/UBSan,
# clang UBSan-trap) instead of running them one after another -- same coverage, wall time max() not sum().
# Off by default so an interactive run streams its output live; CI sets it. See the scheduler near the end.

tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT
fail=0
engines_ran=0   # how many sanitizer engines actually built + ran the twin (clang and/or gcc); a green
                # gate with zero engines tested NOTHING -- the all-absent case is a LOUD failure below.
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
# <key> namespaces this engine's scratch files: with SANITIZE_ENGINES_PARALLEL=1 the clang and gcc engines
# run at the same time, and a SHARED fxerr.txt/secerr.txt would let one engine's stderr overwrite the
# other's between the write and the grep -- silently losing a real diagnostic (or attributing it to the
# wrong compiler). Every path below is therefore per-engine.
run_san_engine() {  # <key> <label> <san-binary>
  local key="$1" label="$2" bin="$3" fx errf rc
  engines_ran=$((engines_ran+1))   # an engine actually built + is running the twin -- the gate has teeth
  local nfix; nfix=$(printf '%s\n' ${FIXTURES} | wc -l)
  echo "[sanitize] ${label}: all ${nfix} cfront_*.c fixtures through the sanitized twin"
  local nfx=0 ffail=0
  for fx in ${FIXTURES}; do
    nfx=$((nfx+1)); errf="${tmp}/fxerr_${key}.txt"
    "${bin}" "${C}/${fx}" >/dev/null 2>"${errf}"; rc=$?
    # A clean parse error is rc 1 with no sanitizer text (the adversarial deep-nest / lexer-tail fixtures
    # deliberately PARSE-ERR -- "nesting too deep" / "input too large" -- which is the CORRECT, crash-free
    # behavior, not a failure). Only a raw crash (rc>=128: SIGSEGV/SIGABRT not caught by ASan) OR any
    # sanitizer diagnostic (incl. ASan's rc-1 `stack-overflow` DEADLYSIGNAL, caught by the grep) is a FAIL.
    if [ "${rc}" -ge 128 ] || grep -qiE "AddressSanitizer|UndefinedBehaviorSanitizer|runtime error|heap-buffer-overflow|global-buffer-overflow|stack-overflow|use-after-free|stack-buffer|out of bounds" "${errf}"; then
      echo "  FAIL: ${label} sanitizer diagnostic on fixture ${fx} (rc=${rc}):"; sed 's/^/    /' "${errf}" | head -25; ffail=1; fail=1
    fi
  done
  [ "${ffail}" -eq 0 ] && echo "  PASS ${label} fixtures (${nfx} cfront_*.c, no ASan/UBSan diagnostic)"

  echo "[sanitize] ${label}: bounded fuzz campaign (${N_VALID} valid + ${M_MALFORMED} malformed, seed ${FUZZ_SEED})"
  # Drive the EXISTING generators (no new generator): the rich valid-program generator in
  # tools/c/fuzz_cfront.py (Gen.program) + the malformed corruptor in cfront.cfuzz (gen_program+mutate).
  # Each program is written to a file and fed to the sanitized twin; any sanitizer error / crash fails.
  # SAN_LEAKS=1 additionally runs the twin with ASAN_OPTIONS=detect_leaks=1 over the SAME bounded campaign,
  # so error-path leaks (Bug 3 -- a compile-error / parse-failure that never frees the in-progress unit)
  # are caught going forward, not just memory-safety violations.
  run_fuzz_campaign() {  # <leaks:0|1>  -- echoes nothing; returns nonzero on a sanitizer/leak diagnostic
    SAN_BIN="${bin}" SAN_N="${N_VALID}" SAN_M="${M_MALFORMED}" SAN_SEED="${FUZZ_SEED}" SAN_LEAKS="$1" \
    python3 - "${ROOT}" <<'PY'
import os, sys, random, subprocess, tempfile
root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "tools", "c"))
sys.path.insert(0, root)
import fuzz_cfront                                  # Gen.program() -- the rich VALID-program generator
from bcir.frontends.cfront import cfuzz            # gen_program() + mutate() -- the MALFORMED corruptor
exe = os.environ["SAN_BIN"]
N, M, seed = int(os.environ["SAN_N"]), int(os.environ["SAN_M"]), int(os.environ["SAN_SEED"])
leaks = os.environ.get("SAN_LEAKS", "0") == "1"
d = tempfile.mkdtemp(prefix="san_cfront_")
# The leak pass turns LeakSanitizer ON (detect_leaks=1) and treats a leak report as a failure; the plain
# pass keeps it OFF so a transient process-exit reachable-block is not misread as a leak.
env = dict(os.environ,
           ASAN_OPTIONS="halt_on_error=1:detect_leaks=" + ("1" if leaks else "0"),
           UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
def feed(src):
    p = os.path.join(d, "f.c")
    with open(p, "w") as fh: fh.write(src)
    r = subprocess.run([exe, p], capture_output=True, text=True, env=env)
    bad = (r.returncode < 0  # killed by a signal (e.g. ASan SIGABRT escaping the exit code)
           or "Sanitizer" in r.stderr or "runtime error" in r.stderr
           or (leaks and "detected memory leaks" in r.stderr))
    return bad, r.returncode, r.stderr
bad = 0
rng = random.Random(seed)                          # deterministic VALID campaign
for i in range(N):
    src = fuzz_cfront.Gen(rng).program().source
    b, rc, err = feed(src)
    if b:
        bad += 1
        sys.stderr.write(f"  FAIL: sanitizer/leak error on VALID program #{i} (rc={rc}):\n{src}\n{err[:1200]}\n")
        break
rng = random.Random(seed)                          # deterministic MALFORMED campaign (totality contract)
for i in range(M):
    src = cfuzz.gen_program(rng)
    for _ in range(rng.randint(1, 5)):
        src = cfuzz.mutate(src, rng)
    b, rc, err = feed(src)
    if b:
        bad += 1
        sys.stderr.write(f"  FAIL: sanitizer/leak error / crash on MALFORMED input #{i} (rc={rc}):\n{src!r}\n{err[:1200]}\n")
        break
sys.exit(1 if bad else 0)
PY
  }
  if run_fuzz_campaign 0; then
    echo "  PASS ${label} fuzz (${N_VALID} valid + ${M_MALFORMED} malformed clean under ASan/UBSan)"
  else
    echo "  FAIL: ${label} fuzz campaign tripped the sanitizer"; fail=1
  fi
  # ----- LSan/leak pass: the SAME bounded campaign with detect_leaks=1 (error-path leaks -- Bug 3) -------
  echo "[sanitize] ${label}: leak pass (detect_leaks=1) over the same ${N_VALID}+${M_MALFORMED} campaign"
  if run_fuzz_campaign 1; then
    echo "  PASS ${label} leak pass (${N_VALID} valid + ${M_MALFORMED} malformed, no LeakSanitizer report)"
  else
    echo "  FAIL: ${label} leak pass tripped LeakSanitizer (an error-path leak)"; fail=1
  fi

  # ----- dedicated adversarial fixtures: pin env-realloc UAF, signature accumulation, parser/lexer
  # bounds, and overlong preprocessor macro parameters. These are SANITIZER-only fixtures (not in the dual-rail
  # parity FIXTURES list); each must compile through the twin with NO ASan/UBSan/leak diagnostic. They are
  # also swept by the cfront_*.c fixture loop above, but a named stage makes the regression intent explicit
  # (a FAIL here names the exact bug class) and runs them with detect_leaks=1 too. -----------------------
  # cfront_sec_envrealloc/sigoverflow must compile clean (rc 0); deep-nest / lexer-tail / cppmacro
  # deliberately return a clean error (rc 1: a bounded capacity diagnostic) --
  # the CORRECT crash-free behavior. For both, only a raw crash (rc>=128) or a sanitizer/leak diagnostic
  # is a FAIL; a clean rc-1 parse error for the deep-nest/lexer-tail pins is expected, NOT a regression.
  local sec adv_fail=0
  for sec in cfront_sec_envrealloc.c cfront_sec_sigoverflow.c cfront_sec_deepnest.c cfront_sec_lextail.c cfront_sec_cppmacro.c; do
    if [ ! -f "${C}/${sec}" ]; then
      echo "  FAIL: ${label} adversarial fixture ${sec} is MISSING (a memory-safety regression pin was removed)"; adv_fail=1; fail=1; continue
    fi
    errf="${tmp}/secerr_${key}.txt"
    ASAN_OPTIONS="halt_on_error=1:detect_leaks=1" "${bin}" "${C}/${sec}" >/dev/null 2>"${errf}"; rc=$?
    if [ "${rc}" -ge 128 ] || grep -qiE "AddressSanitizer|UndefinedBehaviorSanitizer|runtime error|stack-buffer|heap-buffer-overflow|global-buffer-overflow|stack-overflow|use-after-free|out of bounds|detected memory leaks" "${errf}"; then
      echo "  FAIL: ${label} adversarial fixture ${sec} tripped the sanitizer (rc=${rc}):"; sed 's/^/    /' "${errf}" | head -25; adv_fail=1; fail=1
    fi
  done
  [ "${adv_fail}" -eq 0 ] && echo "  PASS ${label} adversarial fixtures (env realloc, signature overflow, deep nesting, lexer tail, preprocessor macro overflow; ASan+leak clean)"
}

# ----- the three engine stages, each a self-contained function ---------------------------------------
# Every stage ends with `stage_result <key>`, which is the ONLY way a stage reports back. With
# SANITIZE_ENGINES_PARALLEL=1 a stage runs backgrounded, i.e. in a SUBSHELL, where an assignment to `fail`
# or `engines_ran` is invisible to the parent -- so the parent must not read those variables at all and
# instead aggregates the per-stage files below. This is the same discipline fuzz_streampack.sh uses for its
# concurrent campaigns (per-campaign rc FILES, never `wait`-ing on a PID: `wait -n` reaps an ARBITRARY
# child, so a later `wait "$pid"` finds the status already consumed and a CRASHING engine reports PASS).
# Serial mode calls the stages in a subshell too, so the reporting path is identical either way.
stage_result() {  # <key>
  echo "${fail}" >"${tmp}/rc_$1"
  echo "${engines_ran}" >"${tmp}/ran_$1"
}

# ----- stage 1: clang ASan/UBSan (primary) ----------------------------------------------------------
stage_clang() {
  fail=0; engines_ran=0
  if [ -n "${CLANG}" ] && [ -x "${CLANG}" ]; then
    if can_sanitize "${CLANG}"; then
      if build_san "${CLANG}" "${tmp}/cfront_clang_san"; then
        run_san_engine clang "clang ASan/UBSan" "${tmp}/cfront_clang_san"
      else
        echo "[sanitize] FAIL: clang sanitizer build of the twin failed"; fail=1
      fi
    else
      echo "[sanitize] SKIP clang ASan/UBSan (tool absent: clang has no compiler-rt / libclang_rt.asan to link)"
    fi
  else
    echo "[sanitize] SKIP clang ASan/UBSan (tool absent: no clang)"
  fi
  stage_result clang
}

# ----- stage 2: gcc ASan/UBSan (bonus -- and the real engine when clang lacks compiler-rt) -----------
stage_gcc() {
  fail=0; engines_ran=0
  if command -v gcc >/dev/null 2>&1; then
    if can_sanitize gcc; then
      if build_san gcc "${tmp}/cfront_gcc_san"; then
        run_san_engine gcc "gcc ASan/UBSan" "${tmp}/cfront_gcc_san"
      else
        echo "[sanitize] FAIL: gcc sanitizer build of the twin failed"; fail=1
      fi
    else
      echo "[sanitize] SKIP gcc ASan/UBSan (tool absent: gcc has no libasan to link)"
    fi
  else
    echo "[sanitize] SKIP gcc ASan/UBSan (tool absent: no gcc)"
  fi
  stage_result gcc
}

# ----- stage 2b: clang UBSan in TRAP mode (no compiler-rt / libclang_rt needed) ---------------------
# clang's -fsanitize=undefined is STRICTER than gcc's: it flags out-of-bounds POINTER ARITHMETIC (forming
# `a + i` past an array, even with no dereference) which gcc's UBSan does NOT -- exactly the class that
# passed a gcc-only local run but failed clang-UBSan on CI (the sig[512] `sig+sw` formation). With
# -fsanitize-trap=undefined each UB becomes an inline trap (ud2 -> SIGILL), needing NO sanitizer runtime to
# link, so this stage runs even where clang lacks compiler-rt -- closing the "local gcc green, CI clang red"
# gap. UB-only (no ASan/leak), so it does NOT count as an ASan engine for the vacuous-gate below.
stage_ctrap() {
  fail=0; engines_ran=0
  if [ -n "${CLANG}" ] && [ -x "${CLANG}" ]; then
    CTRAP=""
    for std in c23 c2x c11; do
      if "${CLANG}" -std=${std} -fsanitize=undefined -fsanitize-trap=undefined -fno-sanitize-recover=all -g -O1 \
           -I "${C}" ${CFRONT_SRCS} -o "${tmp}/cfront_clang_trap" 2>/dev/null; then CTRAP="${tmp}/cfront_clang_trap"; break; fi
    done
    if [ -z "${CTRAP}" ]; then
      echo "[sanitize] SKIP clang UBSan-trap (clang could not build the trap twin)"
    else
      nct=$(printf '%s\n' ${FIXTURES} | wc -l)
      echo "[sanitize] clang UBSan-trap (-fsanitize=undefined -fsanitize-trap, no compiler-rt): all ${nct} cfront_*.c fixtures"
      ctfail=0
      for fx in ${FIXTURES}; do
        "${CTRAP}" "${C}/${fx}" >/dev/null 2>"${tmp}/ct_ctrap.txt"; rc=$?
        # A UB trap dies by signal (rc>=128: SIGILL=132). A normal parse error is rc 1 (fine, not UB).
        if [ "${rc}" -ge 128 ]; then
          echo "  FAIL: clang UBSan-trap caught undefined behavior on ${fx} (rc=${rc}, signal $((rc-128))) -- a clang-strict UB gcc's UBSan misses"; ctfail=1; fail=1
        fi
      done
      [ "${ctfail}" -eq 0 ] && echo "  PASS clang UBSan-trap (${nct} cfront_*.c fixtures incl. adversarial, no clang-strict UB)"
    fi
  else
    echo "[sanitize] SKIP clang UBSan-trap (no clang)"
  fi
  stage_result ctrap
}

# ----- engine scheduler ------------------------------------------------------------------------------
# The three stages are independent: each builds its OWN binary from the same read-only sources, keeps its
# own scratch files (the <key> namespacing above), and the fuzz driver mkdtemp's its own program directory.
# Nothing is written that another stage reads, so SANITIZE_ENGINES_PARALLEL=1 can overlap them and turn the
# gate's wall time from sum(stages) into max(stages). Logs are buffered per stage and replayed in a FIXED
# order afterwards, so a parallel run's output is byte-identical to a serial one instead of interleaved.
STAGE_KEYS="clang gcc ctrap"
if [ "${SANITIZE_ENGINES_PARALLEL:-0}" = "1" ]; then
  echo "[sanitize] SANITIZE_ENGINES_PARALLEL=1: clang, gcc and clang-trap engines run concurrently"
  echo "[sanitize] (per-engine binaries + scratch files; logs replayed below in serial order)"
  for key in ${STAGE_KEYS}; do
    ( "stage_${key}" ) >"${tmp}/log_${key}" 2>&1 &
  done
  wait
  for key in ${STAGE_KEYS}; do
    [ -f "${tmp}/log_${key}" ] && cat "${tmp}/log_${key}"
  done
else
  for key in ${STAGE_KEYS}; do ( "stage_${key}" ); done
fi

# Aggregate the stages. `fail`/`engines_ran` in THIS shell are meaningless until now (each stage zeroed its
# own copy inside a subshell), so they are recomputed here purely from the per-stage files.
fail=0
engines_ran=0
for key in ${STAGE_KEYS}; do
  if [ ! -f "${tmp}/rc_${key}" ] || [ ! -f "${tmp}/ran_${key}" ]; then
    # A stage that never wrote its result file died before finishing (killed, or the shell aborted mid-stage).
    # Silence there must NOT read as success -- that is exactly the vacuous-green shape this gate exists to
    # prevent, so a missing result is a hard failure.
    echo "[sanitize] FAIL: engine stage '${key}' produced no result file -- it died before reporting"
    fail=1
    continue
  fi
  [ "$(cat "${tmp}/rc_${key}")" = "0" ] || fail=1
  engines_ran=$(( engines_ran + $(cat "${tmp}/ran_${key}") ))
done

# The vacuous-gate guard: if NO sanitizer engine actually ran (neither a compiler-rt clang nor a libasan
# gcc could link the runtime), the per-tool SKIPs above would otherwise leave `fail=0` and exit 0 -- a
# green gate that tested NOTHING. Make that case a LOUD, distinct non-zero failure so a CI step cannot pass
# a vacuous run. (The per-tool SKIP-when-absent behaviour is unchanged; only the ALL-absent case fails.)
# SANITIZE_ALLOW_NO_ENGINE=1 lets a deliberately tool-free environment opt out (still printed prominently).
if [ "${engines_ran}" -eq 0 ]; then
  if [ "${SANITIZE_ALLOW_NO_ENGINE:-0}" = "1" ]; then
    echo "[sanitize] WARNING: sanitizer gate ran NOTHING (no clang/gcc could link a sanitizer runtime); SANITIZE_ALLOW_NO_ENGINE=1 set -- not failing"
  else
    echo "[sanitize] FAIL: WARNING: sanitizer gate ran NOTHING -- no compiler (clang/gcc) could link a"
    echo "[sanitize]       sanitizer runtime, so ZERO ASan/UBSan/leak coverage executed. This is a hard"
    echo "[sanitize]       failure (a green gate that tested nothing). Install a compiler-rt clang or a"
    echo "[sanitize]       libasan gcc, or set SANITIZE_ALLOW_NO_ENGINE=1 to opt out deliberately."
    fail=1
  fi
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
  for cc in "${CLANG}" "$(command -v gcc || true)"; do
    [ -n "${cc}" ] && [ -x "${cc}" ] || continue
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
