#!/usr/bin/env bash
# Add apt.llvm.org's signed repository for one LLVM major, and harden apt.
#
# Usage: tools/ci/add_llvm_apt_repo.sh <llvm-major>
#
# Two CI jobs need this -- the MLIR rail matrix and the LLVM 23 training corpus --
# and they used to carry a copy each. One predicate, because the copies had already
# started to matter: the defect below was present in both and surfaced in one
# (`docs/security/laws.md` L14).
#
# **The key fetch must fail loudly.** Both copies were
#
#     wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
#       | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null
#
# and a pipeline's exit status is its *last* command's. `tee` succeeds on an empty
# stream, so a failed fetch wrote an empty keyring, `wget`'s status was discarded,
# and the step passed. Twenty lines later `apt-get update` said
#
#     NO_PUBKEY 15CF4D18AF4F7421
#     E: The repository '...llvm-toolchain-noble-22 InRelease' is not signed.
#     ##[error]Process completed with exit code 100
#
# which reads like an upstream signing problem and is in fact this script's own
# swallowed error: the same step had passed on the previous commit fifteen minutes
# earlier, with the same key and the same workflow. That is L1 on a shell rail --
# every exit is a verdict, and a pipe is a place a verdict goes missing.
#
# So the key is fetched to a temporary file, retried, and *inspected* before it is
# installed: a keyring is only a keyring if it holds a PGP public key block.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <llvm-major>" >&2
    exit 2
fi
readonly LLVM_MAJOR="$1"
readonly KEY_URL="https://apt.llvm.org/llvm-snapshot.gpg.key"
readonly KEYRING="/etc/apt/trusted.gpg.d/apt.llvm.org.asc"

# shellcheck disable=SC1091  # provided by the runner image
. /etc/os-release  # VERSION_CODENAME (e.g. noble)

staged="$(mktemp)"
trap 'rm -f "$staged"' EXIT

fetched=""
for attempt in 1 2 3; do
    if wget --quiet --timeout=20 --tries=1 -O "$staged" "$KEY_URL"; then
        # An HTTP error page, a truncated transfer and an empty body all look like a
        # successful write to `tee`. A key says what it is on its first line.
        if grep -q "BEGIN PGP PUBLIC KEY BLOCK" "$staged"; then
            fetched="yes"
            break
        fi
        echo "attempt ${attempt}: ${KEY_URL} returned $(wc -c <"$staged") byte(s)" \
             "that are not a PGP public key block" >&2
    else
        echo "attempt ${attempt}: could not fetch ${KEY_URL}" >&2
    fi
    sleep "$((attempt * 5))"
done

if [ -z "$fetched" ]; then
    echo "error: no usable signing key from ${KEY_URL} after 3 attempts;" \
         "apt would report this as an unsigned repository, which it is not" >&2
    exit 1
fi

sudo install -m 0644 "$staged" "$KEYRING"
echo "deb http://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME}-${LLVM_MAJOR} main" \
    | sudo tee /etc/apt/sources.list.d/llvm.list >/dev/null

# Drop the slow ESM/Pro sources and the pre-installed Microsoft/azure runner repos,
# which began returning 403 "no longer signed" and hard-fail `apt-get update` with
# exit 100. BCIR needs neither.
sudo rm -f /etc/apt/sources.list.d/*ubuntu-pro* /etc/apt/sources.list.d/*esm* \
    /etc/apt/sources.list.d/*microsoft* /etc/apt/sources.list.d/*azure* 2>/dev/null || true

# Retries and a per-source timeout against mirror hangs; no translation indexes.
printf 'Acquire::Retries "3";\nAcquire::http::Timeout "20";\nAcquire::https::Timeout "20";\nAcquire::Languages "none";\n' \
    | sudo tee /etc/apt/apt.conf.d/99ci-fast >/dev/null

# The install action resolves package names against the CURRENT lists, before any
# update of its own, so this one is needed here.
sudo apt-get update
