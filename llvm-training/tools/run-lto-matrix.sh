#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
EXPERIMENT_DIR="$TRAINING_ROOT/07-optimization/experiments"
MANIFEST="$EXPERIMENT_DIR/matrix.json"
OUTPUT_DIR="$TRAINING_ROOT/.artifacts/lto-matrix"
LTO_FILTER=""
OPT_FILTER=""
TARGET_FILTER=""
INCLUDE_BOLT=0
BOLT_PROFILE=""
REQUIRE_SUPPORTED=0

usage() {
  cat <<USAGE
usage: $0 [options]
  --output DIR             artifact/report directory (default: llvm-training/.artifacts/lto-matrix)
  --lto none|thin|full     run only one LTO dimension
  --optimization O0|O2    run only one optimization dimension
  --target host|TRIPLE     run only one target dimension
  --include-bolt           run BOLT configurations (otherwise record explicit skips)
  --bolt-profile FILE      profile consumed by the optional BOLT rewrite
  --require-supported      fail if any selected non-BOLT configuration is unsupported
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --output) OUTPUT_DIR=$2; shift 2 ;;
    --lto) LTO_FILTER=$2; shift 2 ;;
    --optimization) OPT_FILTER=$2; shift 2 ;;
    --target) TARGET_FILTER=$2; shift 2 ;;
    --include-bolt) INCLUDE_BOLT=1; shift ;;
    --bolt-profile) BOLT_PROFILE=$2; shift 2 ;;
    --require-supported) REQUIRE_SUPPORTED=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "$LTO_FILTER"; do
  [ -z "$value" ] || [[ "$value" =~ ^(none|thin|full)$ ]] || { printf 'error: invalid --lto value: %s\n' "$value" >&2; exit 2; }
done
for value in "$OPT_FILTER"; do
  [ -z "$value" ] || [[ "$value" =~ ^O(0|2)$ ]] || { printf 'error: invalid --optimization value: %s\n' "$value" >&2; exit 2; }
done

find_versioned_tool() {
  local base=$1 suffix candidate
  if [ -n "${LLVM_SUFFIX:-}" ]; then
    candidate="${base}${LLVM_SUFFIX}"
    if path=$(command -v "$candidate" 2>/dev/null) && "$path" --version >/dev/null 2>&1; then printf '%s\n' "$path"; return 0; fi
    return 1
  fi
  for suffix in $(seq 30 -1 10); do
    candidate="${base}-${suffix}"
    if path=$(command -v "$candidate" 2>/dev/null) && "$path" --version >/dev/null 2>&1; then printf '%s\n' "$path"; return 0; fi
  done
  if path=$(command -v "$base" 2>/dev/null) && "$path" --version >/dev/null 2>&1; then printf '%s\n' "$path"; return 0; fi
  return 1
}

major_version() {
  "$1" --version 2>&1 | python3 -c 'import re,sys; text=sys.stdin.read(); m=re.search(r"(?:version |LLVM |LLD )([0-9]+)", text, re.I); print(m.group(1) if m else "unknown")'
}

declare -A TOOL
missing=()
for name in clang ld.lld llvm-nm llvm-objdump llvm-readobj llvm-size; do
  if path=$(find_versioned_tool "$name"); then
    TOOL[$name]=$path
  else
    missing+=("$name")
  fi
done
if ! command -v python3 >/dev/null 2>&1; then missing+=(python3); fi
if ! command -v sha256sum >/dev/null 2>&1; then missing+=(sha256sum); fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/cases"
RESULTS_JSONL="$OUTPUT_DIR/results.jsonl"
: > "$RESULTS_JSONL"

TOOL_STATUS=ok
TOOL_REASON=""
if [ "${#missing[@]}" -gt 0 ]; then
  TOOL_STATUS=unsupported
  TOOL_REASON="missing required tools: ${missing[*]}"
else
  CLANG_MAJOR=$(major_version "${TOOL[clang]}")
  mismatched=()
  for name in ld.lld llvm-nm llvm-objdump llvm-readobj llvm-size; do
    version=$(major_version "${TOOL[$name]}")
    if [ "$version" != "$CLANG_MAJOR" ]; then mismatched+=("$name=$version"); fi
  done
  if [ "${#mismatched[@]}" -gt 0 ]; then
    TOOL_STATUS=unsupported
    TOOL_REASON="LLVM major-version mismatch: clang=$CLANG_MAJOR ${mismatched[*]}"
  fi
fi

write_result() {
  CASE_ID=$1 STATUS=$2 REASON=$3 LTO=$4 OPT=$5 TARGET=$6 BOLT=$7 CASE_DIR=$8 \
  python3 - "$RESULTS_JSONL" <<'PY'
import json, os, pathlib, sys
case_dir = pathlib.Path(os.environ["CASE_DIR"])
def text(name):
    path = case_dir / name
    return path.read_text(encoding="utf-8").strip() if path.exists() else None
def integer(name):
    value = text(name)
    return int(value) if value not in (None, "") else None
result = {
    "id": os.environ["CASE_ID"],
    "dimensions": {
        "lto": os.environ["LTO"],
        "bolt_rewrite": os.environ["BOLT"] == "true",
        "target": os.environ["TARGET"],
        "optimization": os.environ["OPT"],
    },
    "status": os.environ["STATUS"],
    "reason": os.environ["REASON"] or None,
    "artifacts": {},
    "measurements": {},
}
for key, filename in {
    "binary": "program", "main_object": "main.o", "worker_object": "worker.o",
    "symbols": "symbols.txt", "symbol_presence": "symbol-presence.json", "sections": "sections.txt",
    "size_by_section": "size-by-section.txt", "relocations": "relocations.txt",
    "disassembly_summary": "disassembly.normalized.txt", "lto_resolution": "program.resolution.txt",
    "lto_ir_summary": "lto-ir-summary.txt", "bolt_report": "report.json",
    "rewritten_binary": "program.bolt", "rewritten_summary": "rewritten-summary.txt",
    "profile_input": "profile-input", "rewrite_command": "rewrite-command.txt",
}.items():
    if (case_dir / filename).exists(): result["artifacts"][key] = str((case_dir / filename).resolve())
for key, filename in {
    "file_size_bytes": "file-size.txt", "main_object_size_bytes": "main-object-size.txt",
    "worker_object_size_bytes": "worker-object-size.txt", "relocation_count": "relocation-count.txt",
    "defined_symbol_count": "defined-symbol-count.txt",
}.items():
    value = integer(filename)
    if value is not None: result["measurements"][key] = value
hash_value = text("disassembly.sha256")
if hash_value: result["measurements"]["disassembly_sha256"] = hash_value.split()[0]
with open(sys.argv[1], "a", encoding="utf-8") as stream:
    json.dump(result, stream, sort_keys=True)
    stream.write("\n")
PY
}

mapfile -t CONFIGS < <(python3 - "$MANIFEST" <<'PY'
import json, sys
for c in json.load(open(sys.argv[1], encoding="utf-8"))["configurations"]:
    print("\t".join([c["id"], c["lto"], c["optimization"], c["target"], str(c["bolt_rewrite"]).lower()]))
PY
)

selected=0
failed=0
unsupported=0
for row in "${CONFIGS[@]}"; do
  IFS=$'\t' read -r id lto opt target bolt <<< "$row"
  [ -z "$LTO_FILTER" ] || [ "$lto" = "$LTO_FILTER" ] || continue
  [ -z "$OPT_FILTER" ] || [ "$opt" = "$OPT_FILTER" ] || continue
  [ -z "$TARGET_FILTER" ] || [ "$target" = "$TARGET_FILTER" ] || continue
  selected=$((selected + 1))
  case_dir="$OUTPUT_DIR/cases/$id"
  mkdir -p "$case_dir"

  if [ "$bolt" = true ]; then
    if [ "$INCLUDE_BOLT" -ne 1 ]; then
      write_result "$id" skipped "optional BOLT leg not requested (pass --include-bolt)" "$lto" "$opt" "$target" "$bolt" "$case_dir"
      printf '[skip] %s: optional BOLT leg not requested\n' "$id"
      continue
    fi
    bolt_args=(--output "$case_dir")
    [ -z "$BOLT_PROFILE" ] || bolt_args+=(--profile "$BOLT_PROFILE")
    if "$SCRIPT_DIR/run-bolt-experiment.sh" "${bolt_args[@]}"; then
      status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$case_dir/report.json")
      reason=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("reason") or "")' "$case_dir/report.json")
      write_result "$id" "$status" "$reason" "$lto" "$opt" "$target" "$bolt" "$case_dir"
      [ "$status" != unsupported ] || unsupported=$((unsupported + 1))
    else
      write_result "$id" failed "BOLT experiment runner failed" "$lto" "$opt" "$target" "$bolt" "$case_dir"
      failed=$((failed + 1))
    fi
    continue
  fi

  if [ "$TOOL_STATUS" != ok ]; then
    write_result "$id" unsupported "$TOOL_REASON" "$lto" "$opt" "$target" "$bolt" "$case_dir"
    printf '[unsupported] %s: %s\n' "$id" "$TOOL_REASON"
    unsupported=$((unsupported + 1))
    continue
  fi

  actual_target=$target
  [ "$target" != host ] || actual_target=$("${TOOL[clang]}" -dumpmachine)
  target_flag=(--target="$actual_target")
  lto_flag=()
  case "$lto" in thin) lto_flag=(-flto=thin) ;; full) lto_flag=(-flto) ;; esac
  common=("-${opt}" "${target_flag[@]}" "${lto_flag[@]}" -ffunction-sections -fdata-sections)
  {
    printf '%q ' "${TOOL[clang]}" "${common[@]}" -c "$EXPERIMENT_DIR/main.c" -o "$case_dir/main.o"; printf '\n'
    printf '%q ' "${TOOL[clang]}" "${common[@]}" -c "$EXPERIMENT_DIR/worker.c" -o "$case_dir/worker.o"; printf '\n'
    printf '%q ' "${TOOL[clang]}" "-${opt}" "${target_flag[@]}" "${lto_flag[@]}" -fuse-ld=lld -Wl,--save-temps -Wl,--emit-relocs "$case_dir/main.o" "$case_dir/worker.o" -o "$case_dir/program"; printf '\n'
  } > "$case_dir/build-command.txt"

  printf '[run] %s ... ' "$id"
  if ! "${TOOL[clang]}" "${common[@]}" -c "$EXPERIMENT_DIR/main.c" -o "$case_dir/main.o" >"$case_dir/build.log" 2>&1 ||
     ! "${TOOL[clang]}" "${common[@]}" -c "$EXPERIMENT_DIR/worker.c" -o "$case_dir/worker.o" >>"$case_dir/build.log" 2>&1 ||
     ! "${TOOL[clang]}" "-${opt}" "${target_flag[@]}" "${lto_flag[@]}" -fuse-ld=lld -Wl,--save-temps -Wl,--emit-relocs "$case_dir/main.o" "$case_dir/worker.o" -o "$case_dir/program" >>"$case_dir/build.log" 2>&1; then
    write_result "$id" unsupported "compile or link rejected this configuration; see build.log" "$lto" "$opt" "$actual_target" "$bolt" "$case_dir"
    printf 'unsupported\n'
    unsupported=$((unsupported + 1))
    continue
  fi
  if ! "$case_dir/program"; then
    write_result "$id" failed "fixture executable returned failure" "$lto" "$opt" "$actual_target" "$bolt" "$case_dir"
    printf 'FAILED\n'
    failed=$((failed + 1))
    continue
  fi

  "${TOOL[llvm-nm]}" --defined-only --print-size --size-sort "$case_dir/program" | sed -E 's#^[0-9a-fA-F]+ ##' > "$case_dir/symbols.txt"
  python3 - "$MANIFEST" "$case_dir/symbols.txt" "$case_dir/symbol-presence.json" <<'PY'
import json, sys
manifest=json.load(open(sys.argv[1], encoding="utf-8"))
symbols={line.split()[-1] for line in open(sys.argv[2], encoding="utf-8") if line.split()}
tracked=manifest["fixture"]["tracked_symbols"]
expected=manifest["fixture"]["expected_symbols"]
result={"tracked": {name: name in symbols for name in tracked}, "missing_expected": [name for name in expected if name not in symbols]}
with open(sys.argv[3], "w", encoding="utf-8") as stream: json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
PY
  if python3 -c 'import json,sys; raise SystemExit(bool(json.load(open(sys.argv[1]))["missing_expected"]))' "$case_dir/symbol-presence.json"; then :; else
    write_result "$id" failed "expected symbol missing from linked binary" "$lto" "$opt" "$actual_target" "$bolt" "$case_dir"
    printf 'FAILED (symbol check)\n'; failed=$((failed + 1)); continue
  fi
  "${TOOL[llvm-objdump]}" --section-headers "$case_dir/program" | sed -E 's/^[[:space:]]*[0-9]+[[:space:]]+/ /' > "$case_dir/sections.txt"
  "${TOOL[llvm-objdump]}" --reloc "$case_dir/program" > "$case_dir/relocations.txt"
  "${TOOL[llvm-objdump]}" --disassemble --no-show-raw-insn --symbolize-operands "$case_dir/program" |
    sed -E 's/^[[:space:]]*[0-9a-fA-F]+:/ADDR:/; s/[[:space:]]+#.*$//' > "$case_dir/disassembly.normalized.txt"
  sha256sum "$case_dir/disassembly.normalized.txt" > "$case_dir/disassembly.sha256"
  wc -c < "$case_dir/program" | tr -d ' ' > "$case_dir/file-size.txt"
  wc -c < "$case_dir/main.o" | tr -d ' ' > "$case_dir/main-object-size.txt"
  wc -c < "$case_dir/worker.o" | tr -d ' ' > "$case_dir/worker-object-size.txt"
  awk '/^[0-9a-fA-F]+[[:space:]]/ {count++} END {print count+0}' "$case_dir/relocations.txt" > "$case_dir/relocation-count.txt"
  wc -l < "$case_dir/symbols.txt" | tr -d ' ' > "$case_dir/defined-symbol-count.txt"
  "${TOOL[llvm-size]}" -A "$case_dir/program" > "$case_dir/size-by-section.txt"
  : > "$case_dir/lto-ir-summary.txt"
  for bc in "$case_dir"/*.import.bc "$case_dir"/*.internalize.bc "$case_dir"/*.opt.bc; do
    [ -e "$bc" ] || continue
    ll="${bc%.bc}.ll"
    "${TOOL[clang]}" -S -emit-llvm "$bc" -o "$ll"
    printf '## %s\n' "$(basename "$ll")" >> "$case_dir/lto-ir-summary.txt"
    sed -n -E '/^(define|declare) /p' "$ll" >> "$case_dir/lto-ir-summary.txt"
  done
  write_result "$id" passed "" "$lto" "$opt" "$actual_target" "$bolt" "$case_dir"
  printf 'passed\n'
done

if [ "$selected" -eq 0 ]; then
  printf 'error: filters selected no manifest configurations\n' >&2
  exit 2
fi

export TOOL_clang=${TOOL[clang]:-} TOOL_ld_lld=${TOOL[ld.lld]:-} TOOL_llvm_nm=${TOOL[llvm-nm]:-}
export TOOL_llvm_objdump=${TOOL[llvm-objdump]:-} TOOL_llvm_readobj=${TOOL[llvm-readobj]:-} TOOL_llvm_size=${TOOL[llvm-size]:-}
MANIFEST_PATH=$MANIFEST TOOL_STATUS=$TOOL_STATUS TOOL_REASON=$TOOL_REASON \
python3 - "$RESULTS_JSONL" "$OUTPUT_DIR/report.json" <<'PY'
import datetime, json, os, pathlib, subprocess, sys
results = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
tools = {}
for name in ("clang", "ld.lld", "llvm-nm", "llvm-objdump", "llvm-readobj", "llvm-size"):
    path = os.environ.get("TOOL_" + name.replace(".", "_").replace("-", "_"))
    if path:
        version = subprocess.run([path, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.splitlines()[0]
        tools[name] = {"path": path, "version": version}
report = {
    "schema_version": 1,
    "kind": "llvm-training-lto-experiment-matrix",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "manifest": str(pathlib.Path(os.environ["MANIFEST_PATH"]).resolve()),
    "toolchain_status": os.environ["TOOL_STATUS"],
    "toolchain_reason": os.environ["TOOL_REASON"] or None,
    "tools": tools,
    "summary": {status: sum(r["status"] == status for r in results) for status in ("passed", "skipped", "unsupported", "failed")},
    "results": results,
}
with open(sys.argv[2], "w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
printf '[report] %s\n' "$OUTPUT_DIR/report.json"
[ "$failed" -eq 0 ] || exit 1
if [ "$REQUIRE_SUPPORTED" -eq 1 ] && [ "$unsupported" -ne 0 ]; then exit 1; fi
