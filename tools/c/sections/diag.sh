#!/usr/bin/env bash
# diag: Clang-style diagnostic renderer (bcir_diag): caret layout == oracle (#diag)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-diag` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The build of
# test_diag stays in the gate with its other compile lines.
#
#   usage: diag.sh <test_diag>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: diag.sh <test_diag>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Clang-grade diagnostics (#diag): the C source-location model + caret renderer (bcir_diag.c, the C
# twin of cfront/diagnostics.py). Fed the SAME synthetic diagnostic (severity / message / byte span)
# over the same source, the C renderer's Clang-layout output (banner + source line + ^~~~ underline)
# is byte-identical to diagnostics.render() -- so the two rails share one diagnostic format,
# independent of which parser produced the error (messages aren't shared; the LAYOUT is).
printf 'unsigned f(unsigned x){ return x + ; }\n'   > "${tmp}/da.c"
printf 'int main(void)\n{\n\treturn foo(1, 2);\n}\n' > "${tmp}/db.c"   # a tab-indented line
run_diag() {  # <src-file> <filename> <severity> <start> <end> <message>
  local src="$1" fn="$2" sev="$3" st="$4" en="$5" msg="$6"
  local c_out py_out
  c_out="$(printf '%s\t%s\t%s\t%s\n' "${sev}" "${st}" "${en}" "${msg}" | "${HARNESS}" "${src}" "${fn}")"
  py_out="$(SRCF="${src}" FN="${fn}" SEV="${sev}" ST="${st}" EN="${en}" MSG="${msg}" "${PYTHON}" -c "
import os, sys
from bcir.frontends.cfront.diagnostics import SourceDiagnostic, Span, render
src=open(os.environ['SRCF']).read()
s,e=int(os.environ['ST']),int(os.environ['EN'])
span=None if (s==-1 and e==-1) else Span(s,e)
d=SourceDiagnostic(os.environ['SEV'], os.environ['MSG'], span=span)
sys.stdout.write(render(d, src, os.environ['FN']))")" || { echo "  FAIL: oracle render ${fn}"; exit 1; }
  [ "${c_out}" = "${py_out}" ] \
    && echo "  PASS diag ${fn} ${sev}@${st}:${en}" \
    || { echo "  FAIL: diag ${fn} (C != PY)"; printf '   C : %s\n   PY: %s\n' "${c_out}" "${py_out}"; exit 1; }
}
run_diag "${tmp}/da.c" u.c error    34 35 "expected ';'"          # a spanned single-caret error
run_diag "${tmp}/da.c" u.c error    -1 -1 "file-level problem"    # a spanless (no-caret) banner
run_diag "${tmp}/da.c" u.c warning   9 10 "odd parameter name"    # a warning severity
run_diag "${tmp}/db.c" m.c error    19 22 "implicit declaration of 'foo'"  # tab-indented line: caret aligns
run_diag "${tmp}/db.c" m.c error    34 34 "zero-width insertion point"     # zero-width span -> one caret
echo "  PASS diagnostic renderer is byte-identical to diagnostics.render()"
# the machine-readable (JSON) feed: bcir_diag_to_json == DiagnosticReport.to_json (json.dumps indent=2).
run_diag_json() {  # <src-file> <filename> <severity> <start> <end> <message>
  local src="$1" fn="$2" sev="$3" st="$4" en="$5" msg="$6"
  local c_out py_out
  c_out="$(printf '%s\t%s\t%s\t%s\n' "${sev}" "${st}" "${en}" "${msg}" | "${HARNESS}" --json "${src}" "${fn}")"
  py_out="$(SRCF="${src}" FN="${fn}" SEV="${sev}" ST="${st}" EN="${en}" MSG="${msg}" "${PYTHON}" -c "
import os, sys
from bcir.frontends.cfront.diagnostics import SourceDiagnostic, Span, DiagnosticReport
src=open(os.environ['SRCF']).read()
s,e=int(os.environ['ST']),int(os.environ['EN'])
span=None if (s==-1 and e==-1) else Span(s,e)
d=SourceDiagnostic(os.environ['SEV'], os.environ['MSG'], span=span, phase='parse')
sys.stdout.write(DiagnosticReport([d], src, os.environ['FN']).to_json())")" || { echo "  FAIL: oracle json ${fn}"; exit 1; }
  [ "${c_out}" = "${py_out}" ] \
    && echo "  PASS diag-json ${fn} ${sev}@${st}:${en}" \
    || { echo "  FAIL: diag-json ${fn} (C != PY)"; printf '   C : %s\n   PY: %s\n' "${c_out}" "${py_out}"; exit 1; }
}
run_diag_json "${tmp}/da.c" u.c error    34 35 "expected ';'"               # spanned -> full location object
run_diag_json "${tmp}/da.c" u.c warning  -1 -1 "file-level problem"         # spanless -> just "file"
run_diag_json "${tmp}/db.c" m.c error    19 22 "implicit declaration of 'foo'"
echo "  PASS diagnostic JSON feed is byte-identical to DiagnosticReport.to_json()"
# fix-it hints: the verb (replace with / insert / remove) is derived from the fix-it's span +
# replacement, the replacement printed with Python repr() in text and JSON-escaped in the feed. A
# primary + one fix-it must match diagnostics.render() / to_json() byte-for-byte on both rails.
run_diag_fixit() {  # <verb-label> <mode:text|json> <fx-start> <fx-end> <replacement>
  local label="$1" mode="$2" fs="$3" fe="$4" repl="$5" c_out py_out jflag=""
  [ "${mode}" = json ] && jflag="--json"
  c_out="$(printf 'error\t34\t35\texpected token\n+\t%s\t%s\t%s\n' "${fs}" "${fe}" "${repl}" \
           | "${HARNESS}" ${jflag} "${tmp}/da.c" u.c)"
  py_out="$(SRCF="${tmp}/da.c" MODE="${mode}" FS="${fs}" FE="${fe}" REPL="${repl}" "${PYTHON}" -c "
import os, sys
from bcir.frontends.cfront.diagnostics import SourceDiagnostic, Span, FixIt, DiagnosticReport, render
src=open(os.environ['SRCF']).read()
d=SourceDiagnostic('error','expected token', span=Span(34,35),
                   fixits=[FixIt(Span(int(os.environ['FS']),int(os.environ['FE'])), os.environ['REPL'])], phase='parse')
sys.stdout.write(DiagnosticReport([d], src, 'u.c').to_json() if os.environ['MODE']=='json' else render(d, src, 'u.c'))")" \
    || { echo "  FAIL: oracle fix-it ${label}/${mode}"; exit 1; }
  [ "${c_out}" = "${py_out}" ] \
    && echo "  PASS fix-it ${label}/${mode} (oracle == C)" \
    || { echo "  FAIL: fix-it ${label}/${mode}"; printf '   C : %s\n   PY: %s\n' "${c_out}" "${py_out}"; exit 1; }
}
for m in text json; do
  run_diag_fixit replace-with "${m}" 34 35 ";"     # span -> "replace with ';'"
  run_diag_fixit insert       "${m}" 34 34 ")"     # zero-width span -> "insert ')'"
  run_diag_fixit remove       "${m}" 34 36 ""      # empty replacement -> "remove ''"
done
echo "  PASS diagnostic fix-it hints are byte-identical across both rails"
# include / line-map origin: a diagnostic relocated to its origin file:line, with the #include chain
# printed as Clang "In file included from ...:" frames (text) / an "includedFrom" array (JSON). The
# primary error sits at offset 34 (column 35) of da.c; the origin relocates it to inc/b.h:42.
run_diag_origin() {  # <mode:text|json>
  local mode="$1" c_out py_out jflag=""
  [ "${mode}" = json ] && jflag="--json"
  c_out="$(printf 'error\t34\t35\tundeclared token\n@\t42\t0\tinc/b.h\n^\t10\t0\tmain.c\n^\t3\t0\tinc/a.h\n' \
           | "${HARNESS}" ${jflag} "${tmp}/da.c" u.c)"
  py_out="$(SRCF="${tmp}/da.c" MODE="${mode}" "${PYTHON}" -c "
import os, sys, json
from bcir.frontends.cfront.diagnostics import SourceDiagnostic, Span, render, diagnostic_to_dict
src=open(os.environ['SRCF']).read()
d=SourceDiagnostic('error','undeclared token', span=Span(34,35), phase='parse')
origin=('inc/b.h', 42, [('main.c',10), ('inc/a.h',3)])
if os.environ['MODE']=='json':
    sys.stdout.write(json.dumps([diagnostic_to_dict(d, src, 'u.c', origin=origin)], indent=2))
else:
    sys.stdout.write(render(d, src, 'u.c', origin=origin))")" || { echo "  FAIL: oracle origin/${mode}"; exit 1; }
  [ "${c_out}" = "${py_out}" ] \
    && echo "  PASS origin/${mode} (#include chain oracle == C)" \
    || { echo "  FAIL: origin/${mode}"; printf '   C : %s\n   PY: %s\n' "${c_out}" "${py_out}"; exit 1; }
}
run_diag_origin text
run_diag_origin json
echo "  PASS diagnostic include-stack origin is byte-identical across both rails"
# parser error recovery: a panic-mode run reports EVERY error it resynchronizes past (not just the
# first). The oracle's diagnose() produces that multi-diagnostic DiagnosticReport; the C report
# renderer (bcir_diag_report_render / bcir_diag_to_json over the array) formats the identical report.
printf 'unsigned f(unsigned x) { return x + ; }\nunsigned g(unsigned y) { return y 7; }\n' > "${tmp}/rec_src.c"
RSRC="${tmp}/rec_src.c" RPP="${tmp}/rec_pp.c" RSPEC="${tmp}/rec.spec" RTXT="${tmp}/rec.txt" RJSON="${tmp}/rec.json" \
"${PYTHON}" -c "
import os
from bcir.frontends.cfront.pipeline import diagnose
rep=diagnose(open(os.environ['RSRC']).read(), filename='multi.c')
open(os.environ['RPP'],'w').write(rep.source)
lines=[]
for d in rep.diagnostics:
    s,e=(d.span.start,d.span.end) if d.span else (-1,-1)
    lines.append(f'{d.severity}\t{s}\t{e}\t{d.message}')
    for fx in d.fixits: lines.append(f'+\t{fx.span.start}\t{fx.span.end}\t{fx.replacement}')
    for nt in d.notes:
        ns,ne=(nt.span.start,nt.span.end) if nt.span else (-1,-1)
        lines.append(f'-\t{ns}\t{ne}\t{nt.message}')
open(os.environ['RSPEC'],'w').write('\n'.join(lines)+('\n' if lines else ''))
open(os.environ['RTXT'],'w').write(rep.render())
open(os.environ['RJSON'],'w').write(rep.to_json())
assert len(rep.diagnostics) >= 2, f'expected >=2 recovered diagnostics, got {len(rep.diagnostics)}'
" || { echo "  FAIL: oracle recovery report (expected >=2 diagnostics)"; exit 1; }
[ "$("${HARNESS}" "${tmp}/rec_pp.c" multi.c < "${tmp}/rec.spec")" = "$(cat "${tmp}/rec.txt")" ] \
  && echo "  PASS recovery report text (multi-diagnostic oracle == C)" \
  || { echo "  FAIL: recovery report text"; exit 1; }
[ "$("${HARNESS}" --json "${tmp}/rec_pp.c" multi.c < "${tmp}/rec.spec")" = "$(cat "${tmp}/rec.json")" ] \
  && echo "  PASS recovery report JSON (multi-diagnostic oracle == C)" \
  || { echo "  FAIL: recovery report JSON"; exit 1; }
