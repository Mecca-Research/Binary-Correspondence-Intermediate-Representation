#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
EXPERIMENT_DIR="$TRAINING_ROOT/07-optimization/experiments"
OUTPUT_DIR="$TRAINING_ROOT/.artifacts/bolt-experiment"
PROFILE=""

usage() {
  cat <<USAGE
usage: $0 [--output DIR] [--profile FILE]

Builds and preserves a FullLTO baseline. A profile is deliberately required for
the rewrite: pass an llvm-bolt-readable .fdata or YAML profile with --profile.
Missing or incompatible BOLT support is reported as "unsupported", not passed.
USAGE
}
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output) OUTPUT_DIR=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

find_tool() {
  local base=$1 candidate suffix
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
major() {
  "$1" --version 2>&1 | python3 -c 'import re,sys; m=re.search(r"(?:version |LLVM |LLD )([0-9]+)",sys.stdin.read(),re.I); print(m.group(1) if m else "unknown")'
}

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
REPORT="$OUTPUT_DIR/report.json"
status=unsupported
reason=""
CLANG=$(find_tool clang || true)
LLD=$(find_tool ld.lld || true)
BOLT=$(find_tool llvm-bolt || true)
OBJDUMP=$(find_tool llvm-objdump || true)
NM=$(find_tool llvm-nm || true)

write_report() {
  STATUS=$status REASON=$reason OUTPUT_DIR=$OUTPUT_DIR CLANG=${CLANG:-} LLD=${LLD:-} BOLT=${BOLT:-} OBJDUMP=${OBJDUMP:-} NM=${NM:-} \
  python3 - "$REPORT" <<'PY'
import datetime, json, os, pathlib, subprocess, sys
root=pathlib.Path(os.environ["OUTPUT_DIR"])
tools={}
for name, env in (("clang","CLANG"),("ld.lld","LLD"),("llvm-bolt","BOLT"),("llvm-objdump","OBJDUMP"),("llvm-nm","NM")):
    path=os.environ.get(env)
    if path:
        version=subprocess.run([path,"--version"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.splitlines()[0]
        tools[name]={"path":path,"version":version}
artifacts={}
for key,name in (("baseline_binary","program"),("profile_input","profile-input"),("rewrite_command","rewrite-command.txt"),("rewritten_binary","program.bolt"),("baseline_summary","baseline-summary.txt"),("rewritten_summary","rewritten-summary.txt"),("tool_versions","tool-versions.txt")):
    if (root/name).exists(): artifacts[key]=str((root/name).resolve())
report={"schema_version":1,"kind":"llvm-training-bolt-experiment","generated_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":os.environ["STATUS"],"reason":os.environ["REASON"] or None,"tools":tools,"artifacts":artifacts}
with open(sys.argv[1],"w",encoding="utf-8") as stream: json.dump(report,stream,indent=2,sort_keys=True); stream.write("\n")
PY
}

if [ -z "$CLANG" ] || [ -z "$LLD" ] || [ -z "$OBJDUMP" ] || [ -z "$NM" ]; then
  reason="missing baseline tools (clang, ld.lld, llvm-objdump, or llvm-nm)"
  write_report; printf '[unsupported] %s\n' "$reason"; exit 0
fi
clang_major=$(major "$CLANG")
for pair in "$LLD" "$OBJDUMP" "$NM"; do
  if [ "$(major "$pair")" != "$clang_major" ]; then
    reason="baseline LLVM tools do not match clang major version $clang_major"
    write_report; printf '[unsupported] %s\n' "$reason"; exit 0
  fi
done

{
  for tool in "$CLANG" "$LLD" "$OBJDUMP" "$NM" ${BOLT:+"$BOLT"}; do
    printf '%s: ' "$tool"; "$tool" --version 2>&1 | head -n 1
  done
} > "$OUTPUT_DIR/tool-versions.txt"

build=("$CLANG" -O2 -flto -fuse-ld=lld -fno-pie -no-pie -fno-omit-frame-pointer -ffunction-sections -Wl,--emit-relocs "$EXPERIMENT_DIR/main.c" "$EXPERIMENT_DIR/worker.c" -o "$OUTPUT_DIR/program")
printf '%q ' "${build[@]}" > "$OUTPUT_DIR/build-command.txt"; printf '\n' >> "$OUTPUT_DIR/build-command.txt"
if ! "${build[@]}" > "$OUTPUT_DIR/build.log" 2>&1 || ! "$OUTPUT_DIR/program"; then
  status=failed; reason="baseline build or execution failed"
  write_report; printf '[failed] %s\n' "$reason"; exit 1
fi
{
  "$NM" --defined-only --print-size --size-sort "$OUTPUT_DIR/program"
  "$OBJDUMP" --section-headers "$OUTPUT_DIR/program"
  "$OBJDUMP" --reloc "$OUTPUT_DIR/program"
} > "$OUTPUT_DIR/baseline-summary.txt"

if [ -z "$BOLT" ]; then
  reason="llvm-bolt is not installed; baseline retained"
  write_report; printf '[unsupported] %s\n' "$reason"; exit 0
fi
if [ "$(major "$BOLT")" != "$clang_major" ]; then
  reason="llvm-bolt major version does not match baseline clang $clang_major; baseline retained"
  write_report; printf '[unsupported] %s\n' "$reason"; exit 0
fi
if [ -z "$PROFILE" ]; then
  reason="no --profile input supplied; baseline retained and rewrite not attempted"
  write_report; printf '[unsupported] %s\n' "$reason"; exit 0
fi
if [ ! -f "$PROFILE" ]; then
  status=failed; reason="profile input does not exist: $PROFILE"
  write_report; printf '[failed] %s\n' "$reason"; exit 1
fi
cp "$PROFILE" "$OUTPUT_DIR/profile-input"
rewrite=("$BOLT" "$OUTPUT_DIR/program" -data="$OUTPUT_DIR/profile-input" -reorder-blocks=ext-tsp -reorder-functions=hfsort+ -split-functions -o "$OUTPUT_DIR/program.bolt")
printf '%q ' "${rewrite[@]}" > "$OUTPUT_DIR/rewrite-command.txt"; printf '\n' >> "$OUTPUT_DIR/rewrite-command.txt"
if ! "${rewrite[@]}" > "$OUTPUT_DIR/rewrite.log" 2>&1 || ! "$OUTPUT_DIR/program.bolt"; then
  status=failed; reason="llvm-bolt rewrite failed; see rewrite.log"
  write_report; printf '[failed] %s\n' "$reason"; exit 1
fi
if ! "$OUTPUT_DIR/program.bolt"; then
  status=failed; reason="rewritten binary failed fixture execution"
  write_report; printf '[failed] %s\n' "$reason"; exit 1
fi
{
  "$NM" --defined-only --print-size --size-sort "$OUTPUT_DIR/program.bolt"
  "$OBJDUMP" --section-headers "$OUTPUT_DIR/program.bolt"
  "$OBJDUMP" --reloc "$OUTPUT_DIR/program.bolt"
  "$OBJDUMP" --disassemble --no-show-raw-insn --symbolize-operands "$OUTPUT_DIR/program.bolt" | sed -E 's/^[[:space:]]*[0-9a-fA-F]+:/ADDR:/'
} > "$OUTPUT_DIR/rewritten-summary.txt"
status=passed
reason=""
write_report
printf '[passed] BOLT experiment artifacts: %s\n' "$OUTPUT_DIR"
