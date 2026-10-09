#!/usr/bin/env bash
# The C runtime gate. BCIR Make builds the C rails from runtime/manifest.json and runs every
# section of the gate (tools/c/sections/*.sh) as a task; this script shows each section's verdict
# in turn and runs the gates it delegates to (the manifest's `delegated` entries).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CC_WAS_SET="${CC+x}"
CC="${CC:-$(command -v clang || command -v cc || true)}"
if [ -z "${CC}" ]; then
  echo "no C compiler (clang/cc); skipping runtime check." >&2
  exit 0
fi

# Older distributions can expose an unversioned clang that predates the final
# `-std=c23` spelling while also installing a newer versioned clang.  Prefer an
# explicitly supplied CC, but otherwise upgrade the default to the highest
# versioned clang that accepts the spelling used throughout this gate.  This
# keeps the C23 checks meaningful instead of failing on tool-name discovery.
if [ "${CC_WAS_SET}" != x ] &&
   ! printf 'int main(void){return 0;}\n' | "${CC}" -std=c23 -x c -c -o /dev/null - >/dev/null 2>&1; then
  best=""; best_major=-1
  old_ifs="${IFS}"; IFS=:
  for directory in ${PATH}; do
    IFS="${old_ifs}"
    for candidate in "${directory}"/clang-[0-9]*; do
      [ -x "${candidate}" ] || continue
      major="${candidate##*/clang-}"
      [[ "${major}" =~ ^[0-9]+$ ]] || continue
      [ "${major}" -gt "${best_major}" ] || continue
      if printf 'int main(void){return 0;}\n' | "${candidate}" -std=c23 -x c -c -o /dev/null - >/dev/null 2>&1; then
        best="${candidate}"; best_major="${major}"
      fi
    done
    IFS=:
  done
  IFS="${old_ifs}"
  [ -z "${best}" ] || CC="${best}"
fi

# The gates this script delegates to -- the manifest's `delegated` entries, each a script that builds
# and judges binaries of its own -- run here in turn. BCIR_SKIP_DELEGATED_GATES=1 skips them for a
# caller that runs each one itself: the CTest `c-runtime` entry, beside which CMake registers every
# delegated gate as an entry of its own, so `ctest` runs each once (BUILD-2j). M16 of
# tools/build/manifest.py holds these calls, the manifest and the switch in step. Anyone invoking
# check_runtime.sh directly still gets every gate.
echo "[c-runtime] memory classes + allocator/context/channel fault sweep"
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP memory-discipline gate (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as c-memory-discipline)"
elif CC="${CC}" bash "${ROOT}/tools/c/check_memory_discipline.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "  FAIL: memory-discipline gate"; exit 1
fi

tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# BCIR Make builds everything the sections take and runs every section as a task (BUILD-8,
# docs/BCIR_BUILD_ROADMAP.md §8): tools/build/bcirfile.py writes the rails' BCIRfile from
# runtime/manifest.json -- every library, tool, harness, variant and kernel, and every section a
# task over them -- and bcir-make judges it (MK0-MK5, the tools checked by identity) and runs it:
# each target up to date, restored from the artifact cache, or run, two at a time. The compile
# lines this gate used to carry are gone; build-section-parity holds BCIR Make's binaries to the
# CMake build's, section by section. The blocks below show each section's recorded verdict where
# the section's block stands. The C twin of bcir-make, which the same run builds, must then plan
# the finished tree as the oracle does, byte for byte (tools/build/make_parity.py holds the two to
# a generated corpus).
echo "[c-runtime] BCIR Make: the rails from runtime/manifest.json, every section a task"
make_out="build/check-runtime/$(basename "${CC}")"
make_meta="${make_out}/.bcir-make"
if ! python3 "${ROOT}/tools/build/bcirfile.py" --cc "${CC}" --sections --out-dir "${make_out}" \
     -o "${ROOT}/${make_out}/BCIRfile" > "${tmp}/bcirfile.log" 2>&1; then
  echo "  FAIL: the rails' BCIRfile could not be written"; sed 's/^/    /' "${tmp}/bcirfile.log"; exit 1
fi
(cd "${ROOT}" && python3 -m bcir.make -f "${make_out}/BCIRfile" --root "${ROOT}" --check-tools \
   --state "${make_meta}/state.json" --cache "${make_meta}/cache" --logs "${make_meta}/logs" \
   --keep-going) > "${tmp}/make.log" 2>&1
make_rc=$?
grep -E '^(bcir-make|laws|plan|run):' "${tmp}/make.log" | sed 's/^/  /'
grep -E '^  MK[0-9]' "${tmp}/make.log" | head -20
[ "${make_rc}" -le 1 ] || { echo "  FAIL: bcir-make could not judge the rails' BCIRfile"; sed 's/^/    /' "${tmp}/make.log"; exit 1; }
# Nothing ran unless the laws passed, and then every target ran, was restored or was up to date --
# except a section that failed, whose verdict is shown where its block stands. Anything else fails
# the gate here: a BCIRfile that breaks a law, a build target that failed (its command log's tail is
# shown) and whatever it stopped. A verdict on disk from an earlier run is never shown in their place.
if [ "$(grep -cx 'laws: PASS (MK0-MK5)' "${tmp}/make.log")" != 1 ]; then
  echo "  FAIL: the rails' BCIRfile breaks a law, so BCIR Make ran nothing"; exit 1
fi
stopped="$(grep -E '^(failed|skipped) ' "${tmp}/make.log" | grep -v '^failed section\.')"
if [ -n "${stopped}" ]; then
  printf '%s\n' "${stopped}" | grep -E '^failed ' | head -3 | while read -r _ name _; do
    log="${ROOT}/${make_meta}/logs/${name}.log"
    [ -f "${log}" ] && { echo "  --- ${name}: ${log#"${ROOT}"/} ---"; tail -30 "${log}" | sed 's/^/    /'; }
  done
  echo "  FAIL: BCIR Make did not build the rails ($(printf '%s\n' "${stopped}" | grep -c '^failed ') failed, $(printf '%s\n' "${stopped}" | grep -c '^skipped ') skipped):"
  printf '%s\n' "${stopped}" | head -10 | sed 's/^/    /'
  exit 1
fi
twin="${ROOT}/${make_out}/bin/bcir-make"
if [ -x "${twin}" ]; then
  (cd "${ROOT}" && python3 -m bcir.make --dry-run -f "${make_out}/BCIRfile" --root "${ROOT}" \
     --state "${make_meta}/state.json") > "${tmp}/oracle.plan" 2>&1
  (cd "${ROOT}" && "${twin}" --dry-run -f "${make_out}/BCIRfile" --root "${ROOT}" \
     --state "${make_meta}/state.json") > "${tmp}/twin.plan" 2>&1
  if cmp -s "${tmp}/oracle.plan" "${tmp}/twin.plan"; then
    echo "  PASS the C twin plans the finished tree as the oracle does ($(wc -l < "${tmp}/oracle.plan") lines, byte-identical)"
  else
    echo "  FAIL: the C twin's plan of the finished tree is not the oracle's"
    diff "${tmp}/oracle.plan" "${tmp}/twin.plan" | head -20 | sed 's/^/    /'; exit 1
  fi
else
  echo "  FAIL: BCIR Make did not build its C twin (${twin})"
  grep -E "^(failed|skipped) bcir-make" "${tmp}/make.log" | sed 's/^/    /'; exit 1
fi
show_section() {  # <name>: the verdict BCIR Make recorded for the section -- stdout, stderr, status
  local verdict="${ROOT}/${make_out}/verdicts/$1"
  # a section this run's BCIRfile does not plan has no verdict of this run, whatever is on disk
  if ! grep -Fxq "target section.$1" "${ROOT}/${make_out}/BCIRfile"; then
    echo "  FAIL: section $1 is not planned in the rails' BCIRfile"
    return 1
  fi
  if [ ! -f "${verdict}/status" ]; then
    echo "  FAIL: section $1 has no verdict: BCIR Make did not run it"
    grep -E "^(failed|skipped) " "${tmp}/make.log" | head -5 | sed 's/^/    /'
    return 1
  fi
  cat "${verdict}/stdout"
  cat "${verdict}/stderr" >&2
  [ "$(cat "${verdict}/status")" = "0" ] || { echo "  FAIL: section $1 exited $(cat "${verdict}/status")"; return 1; }
}

echo "[c-runtime] freestanding compile (-ffreestanding -nostdlib), C11 + C23: the freestanding core"
show_section freestanding || exit 1

echo "[c-runtime] x86 interrupt-frame ABI header (C11 + C23)"
show_section x86_interrupt || exit 1

echo "[c-runtime] build harness (C23) + Python->C ABI parity"
show_section runtime || exit 1

echo "[c-runtime] BCAB artifact bundle: freestanding reader + Python/C selection parity"
show_section artifact_bundle || exit 1

echo "[c-runtime] StreamPack executor: Python->C parity"
show_section executor || exit 1

echo "[c-runtime] StreamPack encoder: byte-identical re-encode"
show_section encoder || exit 1

echo "[c-runtime] ExecutionPlanV1 (G11): freestanding decode + Python->C->Python byte-identical round trip + plan/pack binding"
show_section execution_plan || exit 1

echo "[c-runtime] PlanStatementV1 + TrustStoreV1 (G21): freestanding Ed25519 (RFC 8032) + SHA-512, signatures byte-identical to the oracle's, every forgery the same verdict on both rails"
show_section plan_sign || exit 1

echo "[c-runtime] ControlRecordV1 (G14): freestanding plane + Python->C->Python round trip + declared refusal statuses + identical two-rail plane traces"
show_section control_plane || exit 1

echo "[c-runtime] live SPSC ring + TelemetryEnvelopeV0 (G15): freestanding ring/envelope + generated signal table + two-rail scenario traces + concurrency under ThreadSanitizer"
show_section ring || exit 1
# The ThreadSanitizer leg: the ring harness built with -fsanitize=thread, and the same build of a
# ring whose relaxed atomic stores are made plain (the manifest's sanitizer variants), judged by
# tools/c/sections/ring_tsan.sh -- the stress runs must report no data race, and the plain-store
# ring must be REPORTED as racing. TSan needs the compiler-rt runtime. The x86 C runtime job
# installs it and sets BCIR_REQUIRE_TSAN=1, so there an unavailable TSan is a failure (L2: the job
# that installed the tool owns its absence); on a host or runner without it (the aarch64 job
# installs clang/lld/llvm only) this leg is an explicit, reported skip -- the concurrent rows above
# still ran there, natively. Available means a trivial TSan program builds AND runs:
# tools/build/sanitizer.py, the one predicate the CMake configure and the section-parity gate ask.
if python3 "${ROOT}/tools/build/sanitizer.py" --cc "${CC}" thread >/dev/null 2>&1; then
  show_section ring_tsan || exit 1
elif [ "${BCIR_REQUIRE_TSAN:-0}" = "1" ]; then
  echo "  FAIL: ThreadSanitizer is required here (BCIR_REQUIRE_TSAN=1), but a trivial TSan program does not build and run"; exit 1
else
  echo "  SKIP live ring under ThreadSanitizer: UNAVAILABLE on this host (no TSan runtime); CI's x86 C runtime job requires it"
fi

echo "[c-runtime] data-plane hand-off (G16): freestanding pack table + shard manifest + per-step freeze + oracle/C traces"
show_section handoff || exit 1

echo "[c-runtime] native K_BCIR planner (G17): byte parity with the oracle"
show_section kplan || exit 1

echo "[c-runtime] UART telemetry frame (#telemetry-frame): byte-identical re-encode"
show_section telemetry_frame || exit 1

echo "[c-runtime] frozen Q8 table (#embed / fallback): build + self-check (C11 + C23)"
show_section q8_tables || exit 1

echo "[c-runtime] plug-in C frontend (bcir_cfront): Python<->C parity"
# bcir_cfront.c is the host compiler tool, the C twin of bcir/frontends/cfront: it lowers
# driver/register-map C to the claim-graph IR of bcir_cir.h (freestanding: the freestanding
# section compiles it with no libc) and verifies what it lowers (bcir_verify.c, a host tool).
show_section cfront || exit 1

echo "[c-runtime] target-ABI matrix (bcir_cfront --target): sizeof data model == oracle (#abi)"
show_section cfront_abi || exit 1

echo "[c-runtime] full C compile->execute loop (cfront -> plan -> hydrate -> exec, no Python)"
show_section cfront_loop || exit 1

echo "[c-runtime] multi-channel lowering decision (bcir_channel): channel.json -> backend pick"
show_section channel || exit 1

echo "[c-runtime] bcir-cc compiler driver: compile a driver (sibling header) + emit artifacts"
# bcir_cc.c is the cc-like driver over the full C pipeline (bcir_cpp_run_ex -I/-D -> bcir_cfront ->
# plan -> hydrate). It must compile a driver with sibling headers via a normal compile command.
show_section bcir_cc || exit 1

echo "[c-runtime] bcir-cc --emit-c links the runtime: self-contained masked unit (#emitlink)"
show_section emitlink || exit 1

echo "[c-runtime] bcir-cc masked emit + recovery override: the recorded two-truth crossing (#recover)"
show_section recover || exit 1

echo "[c-runtime] bcir-cc WRITE-guard: an OOB store fails-fast, never clamps (#writeguard)"
show_section writeguard || exit 1

echo "[c-runtime] OOB ring counter is atomic under concurrent events (#atomicring)"
show_section atomicring || exit 1

echo "[c-runtime] rid->extent tamper-evidence: BCIR_EXTENT_ASSERT (freestanding lightweight check) (#extentassert)"
show_section extentassert || exit 1

echo "[c-runtime] Clang-style diagnostic renderer (bcir_diag): caret layout == oracle (#diag)"
show_section diag || exit 1

echo "[c-runtime] fallback contract (bcir-cc --fallback): route-to-LLVM decision == oracle (#fallback)"
show_section fallback || exit 1

echo "[c-runtime] constant-expression evaluator: enum/case/static folds == reference (#cexpr)"
show_section cexpr || exit 1

echo "[c-runtime] R21 lifetime policy (bcir-cc --r21): verdict + exit-code parity vs the oracle (#r21policy)"
show_section r21policy || exit 1

echo "[c-runtime] project mode (bcir-cc multi-file): verdict line + exit code == oracle (#project)"
show_section project || exit 1

echo "[c-runtime] Phase 3 linking (bcir-cc prototypes): emitted caller + host linker == reference (#link)"
show_section link || exit 1

echo "[c-runtime] effect footprint + escape analysis (bcir-cc --emit-effects/--emit-escape) == oracle over the corpus, generated units and forms (#effects, G10)"
show_section effects || exit 1

echo "[c-runtime] automatic link-flag emission (bcir-cc --emit-link-flags): derived flags == oracle (#linkflags)"
show_section linkflags || exit 1
echo "[c-runtime] B2 FFTW link-flag rule (bcir_cfront_link_flags twin): fftwf_* -> -lfftw3 (#linkflags-fftw)"
show_section linkflags_fftw || exit 1

echo "[c-runtime] LAPACK link-flag rule (bcir_cfront_link_flags twin): LAPACKE_*/sgesv_ -> -llapack (#linkflags-lapack)"
show_section linkflags_lapack || exit 1

echo "[c-runtime] E1 OLS portable fallback (emit_lapack_ols_c): normal-equations recovers a known x (#ols)"
# Emit the fallback OLS kernel from the oracle, append a main that fits y = 2x + 1 (m=8, n=2 -> [1, x] rows,
# recovers [c0=1, c1=2]), and assert the recovered coefficients match to float round-off.
show_section ols || exit 1

echo "[c-runtime] E2 PCA portable fallback (emit_lapack_eigh_c): Jacobi recovers a known spectrum (#pca)"
show_section pca || exit 1

echo "[c-runtime] E3 Transformer layernorm (emit_layernorm_c): per-row normalize to mean~0/var~1 (#layernorm)"
show_section layernorm || exit 1

echo "[c-runtime] E4 recurrent LSTM cell (emit_lstm_cell_c): 1x1 cell matches hand-computed forward (#lstm)"
show_section lstm || exit 1

echo "[c-runtime] E5 classical-ML RBF-SVM predict (emit_svm_rbf_predict_c): decision function matches reference (#classical #svm)"
show_section svm || exit 1

echo "[c-runtime] E5 classical-ML decision-tree predict (emit_tree_predict_c): exact threshold traversal, NO libm (#classical #tree)"
# the tree kernel is EXACT -- no libm needed (we still link -lm only for the harness fabsf-free main; not required).
show_section tree || exit 1

echo "[c-runtime] E6 unsupervised K-means assign (emit_kmeans_assign_c): nearest-centroid argmin, NO libm (#kmeans)"
# the K-means assign kernel is EXACT -- no libm needed (no -lm; the EXACT half of the Area-B pattern).
show_section kmeans || exit 1

echo "[c-runtime] GSL link-flag rule (bcir_cfront_link_flags twin): gsl_* -> -lgsl (#linkflags-gsl)"
show_section linkflags_gsl || exit 1

echo "[c-runtime] SLEEF link-flag rule (bcir_cfront_link_flags twin): Sleef_* -> -lsleef (#linkflags-sleef)"
show_section linkflags_sleef || exit 1

echo "[c-runtime] libcerf link-flag rule (bcir_cfront_link_flags twin): erfcx* -> -lcerf (#linkflags-cerf)"
show_section linkflags_cerf || exit 1

echo "[c-runtime] scalable IR (bcir-cc, no fixed BCIR_MAX_*): cap-busting unit compiles + matches oracle (#scale)"
show_section scale || exit 1

echo "[c-runtime] integer promotions + UAC (bcir-cc): signed/mixed-width emit == Clang full-range (#intpromote)"
show_section intpromote || exit 1

echo "[c-runtime] designated initializers (bcir-cc): dispatch-table emit == Clang (#designated)"
show_section designated || exit 1

echo "[c-runtime] local aggregate init (bcir-cc): struct/union {.field=v} emit == Clang (#aggregate)"
show_section aggregate || exit 1

echo "[c-runtime] scalable parser state (bcir-cc, no fixed gv/s/td/ec/env caps): cap-busting unit == oracle (#pscale)"
show_section pscale || exit 1

echo "[c-runtime] array element stores (bcir-cc): a[i]=v emit == Clang (#astore)"
show_section astore || exit 1

echo "[c-runtime] extern linkage (bcir-cc): extern global emit == Clang (#extern)"
show_section extern || exit 1

echo "[c-runtime] thread-local storage (bcir-cc): _Thread_local emit == Clang (#threadlocal)"
show_section threadlocal || exit 1

echo "[c-runtime] multi-declarator locals (bcir-cc): emit == Clang (#multidecl)"
show_section multidecl || exit 1

echo "[c-runtime] comma-operator for-step (bcir-cc): emit == Clang (#commastep)"
show_section commastep || exit 1

echo "[c-runtime] multi-declarator struct members (bcir-cc): layout + access == Clang (#structmulti)"
show_section structmulti || exit 1

echo "[c-runtime] empty statements (bcir-cc): emit == Clang (#emptystmt)"
show_section emptystmt || exit 1

echo "[c-runtime] store through a pointer (bcir-cc): emit == Clang (#ptrstore)"
show_section ptrstore || exit 1

echo "[c-runtime] nested struct member access (bcir-cc): layout + access == Clang (#nestmember)"
show_section nestmember || exit 1

echo "[c-runtime] reused loop-counter names (bcir-cc): unique emit == Clang (#loopreuse)"
show_section loopreuse || exit 1

echo "[c-runtime] for-loop variable scope (bcir-cc): emit == Clang (#loopscope)"
show_section loopscope || exit 1

echo "[c-runtime] bare-block variable scope (bcir-cc): emit == Clang (#blockscope)"
show_section blockscope || exit 1

echo "[c-runtime] struct member arrays (bcir-cc): layout + access == Clang (#memberarray)"
show_section memberarray || exit 1

echo "[c-runtime] multi-dimensional local arrays (bcir-cc): emit == Clang (#localmd)"
show_section localmd || exit 1

echo "[c-runtime] the 'signed' type specifier (bcir-cc): emit == Clang (#signedty)"
show_section signedty || exit 1

echo "[c-runtime] signed comparison vs literal (bcir-cc): emit == Clang (#signedcmp)"
show_section signedcmp || exit 1

echo "[c-runtime] unary on a wide operand (bcir-cc): emit == Clang (#longunary)"
show_section longunary || exit 1

echo "[c-runtime] store into a _Bool normalizes to 0/1 (bcir-cc): emit == Clang (#boolnorm)"
show_section boolnorm || exit 1

echo "[c-runtime] integer-promotion + float unary -/~ (bcir-cc): emit == Clang (#unarypromote)"
show_section unarypromote || exit 1

echo "[c-runtime] float -> signed-int cast (bcir-cc): emit == Clang (#floatsigncast)"
show_section floatsigncast || exit 1

echo "[c-runtime] integer cast to a signed type (bcir-cc): emit == Clang (#intsigncast)"
show_section intsigncast || exit 1

echo "[c-runtime] cast to _Bool normalizes the full value (bcir-cc): emit == Clang (#boolcast)"
show_section boolcast || exit 1

echo "[c-runtime] signed bitfield read sign-extends (bcir-cc): emit == Clang (#signedbf)"
show_section signedbf || exit 1

echo "[c-runtime] signed sub-int storage read sign-extends (bcir-cc): emit == Clang (#signedload)"
show_section signedload || exit 1

echo "[c-runtime] enum used as a type (bcir-cc): emit == Clang (#enumtype)"
show_section enumtype || exit 1

echo "[c-runtime] pointer locals -- T *p = &x (bcir-cc): emit == Clang (#ptrlocal)"
show_section ptrlocal || exit 1

echo "[c-runtime] pointer values -- return p + i (bcir-cc): emit == Clang (#ptrvalue)"
show_section ptrvalue || exit 1

echo "[c-runtime] pointer struct fields -- s->p = q (bcir-cc): emit == Clang (#ptrfield)"
show_section ptrfield || exit 1

echo "[c-runtime] pointer-to-pointer -- int **pp, **pp (bcir-cc): emit == Clang (#ptr2ptr)"
show_section ptr2ptr || exit 1

echo "[c-runtime] deref-through a loaded pointer field -- *(s->p) / s->mid->leaf->x / s->p[i] (#fieldderef)"
show_section fieldderef || exit 1

echo "[c-runtime] pointer-element signedness -- signed vs unsigned pointee load/store (#ptrsign)"
show_section ptrsign || exit 1

echo "[c-runtime] funcptr dispatch through a loaded pointer -- d->ops->fn(args) (#fnptrchain)"
show_section fnptrchain || exit 1

echo "[c-runtime] per-declarator pointer/array in a multi-declarator decl -- int *p, q; (#multiptr)"
show_section multiptr || exit 1

echo "[c-runtime] faithful char types -- char / signed char / unsigned char (#chartypes)"
show_section chartypes || exit 1

echo "[c-runtime] compound literals -- (type){init} by value / scalar / &(literal) / .field (#complit)"
show_section complit || exit 1

echo "[c-runtime] typeof -- typeof(type-name) / typeof(variable) / typeof(expr) (#typeof)"
show_section typeof || exit 1

echo "[c-runtime] variadic functions -- va_list / va_start / va_arg / va_end / va_copy (#variadic)"
show_section variadic || exit 1

echo "[c-runtime] wide/float compound assignment -- OP= / ++ / -- keeps width (#compoundwide)"
show_section compoundwide || exit 1

echo "[c-runtime] external variadic calls -- snprintf / vsnprintf passthrough (#extvariadic)"
show_section extvariadic || exit 1

echo "[c-runtime] long double -- 80/128-bit extended float, +l libm variants (#longdouble)"
show_section longdouble || exit 1

echo "[c-runtime] _Generic -- C11 type-generic selection (#generic)"
show_section generic || exit 1

echo "[c-runtime] nested designated initializers -- .a.b / .v[i] / .m[i][j] (#designate)"
show_section designate || exit 1

echo "[c-runtime] nested member access at a non-first offset (#nestoffset)"
show_section nestoffset || exit 1

echo "[c-runtime] address-of a (nested) struct member -- &s.f / &t.q.a / &t.q (#addrmember)"
show_section addrmember || exit 1

echo "[c-runtime] _Atomic local objects -- _Atomic int / _Atomic(int) (#atomiclocal)"
show_section atomiclocal || exit 1

echo "[c-runtime] GCC/Clang integer builtins -- popcount/clz/ctz/bswap/abs (#builtins)"
show_section builtins || exit 1

echo "[c-runtime] GCC statement expressions -- ({ ...; e; }) (#stmtexpr)"
show_section stmtexpr || exit 1

echo "[c-runtime] inline assembly: ISA-neutral trusted opaque edge, verbatim emit + memory barrier (#inlineasm)"
show_section inlineasm || exit 1

echo "[c-runtime] port-mapped I/O intrinsics: typed barriered edge, x86 in/out emit, assemble-only (#portio)"
show_section portio || exit 1

echo "[c-runtime] memory-fence intrinsics: typed kinded barriered edge, per-ISA emit, native assemble (#barrier)"
show_section barrier || exit 1

echo "[c-runtime] X.691 PER primitives: strict-warning build (#per)"
# A decoder whose answers depend on the optimiser is not a decoder: each ASN.1 twin's section runs
# its -O0 and -O3 builds (manifest variants) over the same campaign and requires identical output --
# a signed-overflow or shift-past-width bug typically diverges at only one of the two. A twin that
# builds at -O2 but not at -O0 or -O3 fails the BCIR Make build, and so the gate (laws.md L12, L21);
# these sweeps once SKIPped on a failed build.
show_section per || exit 1

echo "[c-runtime] X.693 XER lexical layer: strict-warning build (#xer)"
show_section xer || exit 1

echo "[c-runtime] X.697 bounded JER reader: strict-warning build (#jer)"
show_section jer || exit 1

echo "[c-runtime] plan-driven ASN.1 encoder: strict-warning build (#emit)"
show_section asn1_emit || exit 1

echo "[c-runtime] X.691 plan-driven PER decoder: strict-warning build (#per)"
show_section per_plan || exit 1

echo "[c-runtime] X.696 OER decoder: strict-warning build (#oer)"
show_section oer || exit 1

echo "[c-runtime] native ASN.1 decode microbench: strict-warning build (#asn1bench)"
show_section asn1bench || exit 1

echo "[c-runtime] DER -> native StreamPack fast path: byte-identical reconstruction (#asn1fast)"
show_section asn1fast || exit 1

# Sanitizer + memory-stress harness for the cfront C twin (#sanitize): the dual-rail parity gates above
# compare the twin's OUTPUT against the oracle, but never build bcir_cfront.c under AddressSanitizer/UBSan
# and never run Valgrind -- so a memory bug (buffer overflow, use-after-free, UB) INSIDE the compiler's own
# lexer/parser/lowering/emit would go uncaught. sanitize_cfront.sh closes that: it builds the SAME
# `test_cfront` driver (its sources the manifest's, held by M9) under ASan/UBSan with clang and/or gcc, runs every
# cfront_*.c fixture + a bounded seeded fuzz campaign (valid + malformed) through it, and runs a bounded
# Valgrind pass over a fixture subset. Each stage self-skips when its tool is absent (SKIP, not FAIL). The
# heavy Valgrind stage can be dropped with SANITIZE_SKIP_VALGRIND=1 while ASan/UBSan stay always-on.
#
# BCIR_SKIP_CFRONT_SANITIZE=1 suppresses this nested call for a CALLER THAT RUNS THE SAME
# HARNESS ITSELF. The CI c-runtime job does exactly that (a dedicated step, so a sanitizer
# diagnostic is attributed to the sanitizer rather than buried in this script's output), and
# without the opt-out the whole 300-valid/400-malformed campaign ran twice in one job. Anyone
# invoking check_runtime.sh directly still gets the harness, valgrind included.
if [ "${BCIR_SKIP_CFRONT_SANITIZE:-0}" = "1" ]; then
  echo "[c-runtime] SKIP cfront sanitizer harness (BCIR_SKIP_CFRONT_SANITIZE=1; the caller runs it)"
else
echo "[c-runtime] cfront twin under ASan/UBSan + Valgrind (sanitize_cfront.sh)"
if CLANG="${CC}" bash "${ROOT}/tools/c/sanitize_cfront.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS cfront sanitizer/valgrind harness"
else
  echo "  FAIL: cfront sanitizer/valgrind harness reported a diagnostic"; exit 1
fi
fi

# StreamPack SEMANTIC trust boundary (R10/R11 + lane/width/dispatch range, ported into the
# freestanding C decoder/executor): the CRC + bounds decode is memory-safe, but a CRC-VALID pack
# can still be semantically corrupt (a dangling/redirected claim_id, a swapped/undeclared RID, an
# out-of-range lane/width, an unresolved prefetch, a stale generation, a tampered dispatch). Each
# used to execute SILENTLY in C. check_streampack_semantic.sh crafts each as a CRC-FIXED pack and
# asserts the C rail (bcir_sp_verify_semantic / bcir_sp_execute_checked) now REJECTS it, plus a
# C-decode == Python-decode differential (the lane-asymmetry class) over v1/v2/v3 packs.
echo "[c-runtime] StreamPack semantic trust boundary (check_streampack_semantic.sh)"
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP StreamPack semantic trust boundary (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as c-streampack-semantic)"
elif CC="${CC}" bash "${ROOT}/tools/c/check_streampack_semantic.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS StreamPack semantic-corruption rejection + C/Python decode differential"
else
  echo "  FAIL: a CRC-valid semantically-corrupt pack was not rejected on the C rail"; exit 1
fi

# The hosted SIMD rail (#jersimd, J5): an optional C++17 UTF-8 accept-scanner behind the
# SCALAR C ABI, with SSE2/AVX2/NEON tiers, runtime feature detection and scalar fallback.
# 4.1 is the constraint: "the scalar rail is authoritative for native parser correctness.
# SIMD is an optimization candidate, not a separate semantic implementation." So the vector
# pass answers only "is this block entirely ASCII?" and hands everything else to
# bcir_jer_validate_utf8 ITSELF -- there is no second UTF-8 implementation to keep in step.
# check_jer_simd.sh proves every tier returns an identical status AND byte offset to the
# scalar rail over 489 documents, and that an unavailable tier degrades rather than faults.
# The two-host advantage clause of J5's gate is deliberately NOT checked on a shared runner;
# 8 refuses "noisy timing thresholds" there. Self-skips if no C++ compiler is present.
# The hosted structural index (#jerindex, J5's second half): `bcir_jer_index_scan` rebuilds
# `bcir_jer_scan`'s DISPATCH on the exported cursor and reuses its token scanners verbatim, so
# it is a second dispatch loop rather than a second scanner. Checked here as a build gate --
# warning-clean under C++17, and the C core still freestanding with the cursor exported. The
# equivalence differential itself lives in test_cpp_jer_index.py, where it can sweep 4.3's
# work ceiling across every failure position.
# Target ABI portability (#targetabi). The freestanding core is compiled for triples this
# host cannot run -- Android's among them -- and no source may hand-declare a libc function.
# #699 shipped a bench that did not build under Termux because it did exactly that, and the
# existing cross-compile gates target aarch64-linux-GNU, which is the right architecture and
# the wrong libc.
echo "[c-runtime] target ABI: the freestanding core builds for Android and 32-bit (#targetabi)"
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP target ABI sweep (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as c-target-abi)"
elif bash "${ROOT}/tools/c/check_target_abi.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS target ABI sweep (freestanding core portable; no hand-declared libc prototypes)"
else
  echo "  FAIL: the freestanding core is not portable, or a source declares a libc function"; exit 1
fi

echo "[c-runtime] hosted structural index: the cursor seam and its vector pass (#jerindex)"
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP hosted structural index (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as cpp-jer-index)"
elif bash "${ROOT}/tools/cpp/check_jer_index.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS hosted structural index (freestanding core preserved, tier is real not scalar)"
else
  echo "  FAIL: the hosted structural index did not build, or a tier degraded to scalar"; exit 1
fi

echo "[c-runtime] hosted SIMD rail: scalar-identical status/offset at every tier (#jersimd)"
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP hosted SIMD rail (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as cpp-jer-simd)"
elif bash "${ROOT}/tools/cpp/check_jer_simd.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS hosted SIMD rail (same corpus + same trace at every tier, no unsupported-CPU fault)"
else
  echo "  FAIL: the hosted SIMD rail diverged from the scalar rail or did not build"; exit 1
fi

# The C<->C++ hand-off seam scaffold (#cpphandoff, docs/languages/CPP_HANDOFF_BOUNDARY.md): the boundary
# between the deterministic single-node C/IR rail and the C++ layer ABOVE it (dynamic graph
# topology + distributed MPI/NCCL orchestration). check_handoff.sh compiles the STANDALONE C++17
# scaffold (runtime/cpp/, NOT part of the MLIR build), hands a StreamPack the C/IR path produces
# through the single-node Orchestrator, and asserts the seam ROUND-TRIPS (the C++ dispatch order ==
# the direct C/IR decode of the same artifact) -- plus that a corrupted artifact is REJECTED at the
# boundary (admit() carries the C verifier's verdict; the two-truth quarantine holds across the seam).
# The dynamic-graph + distributed backends are documented STUBS behind the same interface (no real
# MPI/NCCL dependency). Self-skips (exit 0) if no C++ compiler is present, like the gates above.
echo "[c-runtime] C<->C++ hand-off seam scaffold round-trip (tools/cpp/check_handoff.sh)"
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP C<->C++ hand-off seam (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as cpp-handoff)"
elif bash "${ROOT}/tools/cpp/check_handoff.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS C<->C++ hand-off seam compiles + round-trips (single-node Orchestrator == direct C/IR)"
else
  echo "  FAIL: the C++ hand-off seam scaffold did not compile or round-trip"; exit 1
fi

# The SYCL backend differential oracle (#sycldiff, docs/kernel/SYCL_INTEROP.md): SYCL is a backend CHANNEL +
# a differential oracle, NEVER on the legality path. check_sycl.sh emits the single-source C++ SAXPY
# kernel (emit_sycl_saxpy_c) and proves it reproduces BCIR's own deterministic reference (a*x+y) to
# float round-off: the PORTABLE scalar C++ fallback always (the real reference-verification work, no
# SYCL needed -- the #sycl-fallback marker), and the SYCL DEVICE parallel_for (-DBCIR_USE_SYCL -fsycl,
# the #sycl-device marker) IF a real SYCL compiler is detected (icpx/acpp/clang++ -fsycl that compiles+
# links a probe). SYCL is a compiler MODE, NOT a c.call.libm: -l<lib> edge (no link-flag rule). Lives
# above the G8 C++ boundary; self-skips (exit 0) if no C++ compiler, like check_handoff.sh.
echo "[c-runtime] SYCL backend differential oracle (tools/cpp/check_sycl.sh)"
if [ "${BCIR_SKIP_DELEGATED_GATES:-0}" = "1" ]; then
  echo "  SKIP SYCL backend differential oracle (BCIR_SKIP_DELEGATED_GATES=1; CTest runs it as cpp-sycl)"
elif bash "${ROOT}/tools/cpp/check_sycl.sh" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "  PASS SYCL SAXPY differential (portable C++ fallback == BCIR reference; device path when -fsycl present)"
else
  echo "  FAIL: the SYCL backend differential oracle did not compile or agree"; exit 1
fi

echo "[c-runtime] ok"
