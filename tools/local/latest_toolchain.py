#!/usr/bin/env python3
"""The newest LLVM/Clang/MLIR 23 and Node 24 on this host: what is installed, what upstream ships,
and how to get the newest -- the first leg of tools/local/check_latest.sh
(.claude/skills/bcir-latest-toolchain/SKILL.md).

Every BCIR change is judged on the CI-default toolchain (Ubuntu's Clang 18, GCC 13) AND on the
newest release of the majors the law rail and the WASM tests track: LLVM 23 and Node 24. A pass on
Clang 18 alone says nothing about 23 -- BUILD-2c's generated Q8 header built clean on Clang 18 and
GCC 13 and failed on Clang 23.1.2, which offers C23 `#embed` to C11 as an extension.

    latest_toolchain.py status          installed vs newest: exit 0 when everything is current,
                                        1 when something is behind, missing or incoherent, 2 when
                                        the newest release could not be determined
    latest_toolchain.py env             shell exports putting the LLVM 23 bin and Node 24 first
    latest_toolchain.py install-node    the newest Node 24 into the BCIR cache, its tarball checked
                                        against a signature-verified checksum list before extraction

"Newest" is read from the sources themselves: the highest `llvmorg-23.<minor>.<patch>` tag of
llvm/llvm-project (release candidates excluded) and the highest v24.x of nodejs.org's release
index. A source that cannot be reached makes the verdict UNKNOWN, never "current"
(docs/security/laws.md L1, L2). LLVM 23 comes from `BCIR_LOCAL_FULL=1 bash tools/local/setup_mlir.sh`
(conda-forge, which tracks the upstream point releases) or from apt.llvm.org; this tool finds either.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

LLVM_MAJOR = 23
NODE_MAJOR = 24
LLVM_REPO = "https://github.com/llvm/llvm-project.git"
NODE_DIST = "https://nodejs.org/dist"
RELEASE_KEYS_REPO = "https://github.com/nodejs/release-keys.git"
TIMEOUT = 120
# The Node tarball is about 30 MB and its index under 1 MB; a larger body is refused where it is
# read, not after (laws.md L3).
TARBALL_LIMIT = 256 * 1024 * 1024
TEXT_LIMIT = 16 * 1024 * 1024
PLATFORMS = {
    "x86_64": "linux-x64",
    "amd64": "linux-x64",
    "aarch64": "linux-arm64",
    "arm64": "linux-arm64",
}

TAG = re.compile(rf"^refs/tags/llvmorg-{LLVM_MAJOR}\.(\d+)\.(\d+)$")
CLANG_VERSION = re.compile(r"\bclang version (\d+)\.(\d+)\.(\d+)")
LLVM_VERSION = re.compile(r"\bLLVM version (\d+)\.(\d+)\.(\d+)")
NODE_VERSION = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
SHASUM_LINE = re.compile(r"^([0-9a-f]{64})  (\S+)$")

Version = tuple[int, int, int]


class Refused(Exception):
    """An install that must not proceed (exit 1): nothing is extracted."""


def fmt(version: Version | None) -> str:
    return ".".join(str(part) for part in version) if version else "-"


def cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "bcir"


def _last_line(data: bytes) -> str:
    lines = data.decode("utf-8", "replace").strip().splitlines()
    return lines[-1].strip() if lines else "no output"


# --- the newest release, from the sources ---------------------------------------------------------


def newest_llvm_from_tags(listing: str) -> Version | None:
    """The highest `llvmorg-23.<minor>.<patch>` of a `git ls-remote --tags` listing; release
    candidates, `-init` and peeled `^{}` lines do not match."""
    found: list[Version] = []
    for line in listing.splitlines():
        fields = line.split()
        match = TAG.match(fields[-1]) if fields else None
        if match:
            found.append((LLVM_MAJOR, int(match[1]), int(match[2])))
    return max(found) if found else None


def newest_node_from_index(entries: object) -> Version | None:
    """The highest v24.x of nodejs.org's index.json (a list of {version, ...} objects)."""
    found: list[Version] = []
    for entry in entries if isinstance(entries, list) else []:
        match = (
            NODE_VERSION.match(str(entry.get("version", ""))) if isinstance(entry, dict) else None
        )
        if match and int(match[1]) == NODE_MAJOR:
            found.append((int(match[1]), int(match[2]), int(match[3])))
    return max(found) if found else None


def fetch(url: str, limit: int) -> bytes:
    """GET `url` (HTTPS_PROXY and the CA bundle come from the environment), refusing a body past
    `limit` bytes as it is read."""
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{url}: the body exceeds {limit} bytes")
    return data


def query_llvm() -> tuple[Version | None, str]:
    try:
        listing = subprocess.run(
            [
                "git",
                "ls-remote",
                "--tags",
                "--refs",
                LLVM_REPO,
                f"refs/tags/llvmorg-{LLVM_MAJOR}.*",
            ],
            capture_output=True,
            timeout=TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"git ls-remote could not run: {exc}"
    if listing.returncode != 0:
        return None, f"git ls-remote {LLVM_REPO} failed: {_last_line(listing.stderr)}"
    newest = newest_llvm_from_tags(listing.stdout.decode("utf-8", "replace"))
    if newest is None:
        return None, f"{LLVM_REPO} lists no llvmorg-{LLVM_MAJOR}.x.y release tag"
    return newest, f"llvmorg-{fmt(newest)}"


def query_node() -> tuple[Version | None, str]:
    url = f"{NODE_DIST}/index.json"
    try:
        entries = json.loads(fetch(url, TEXT_LIMIT))
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return None, f"{url}: {exc}"
    newest = newest_node_from_index(entries)
    if newest is None:
        return None, f"{url} lists no v{NODE_MAJOR}.x release"
    return newest, f"v{fmt(newest)} in {url}"


# --- what is installed ------------------------------------------------------------------------------


def tool_version(tool: Path, pattern: re.Pattern[str]) -> Version | None:
    try:
        run = subprocess.run([str(tool), "--version"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = pattern.search(run.stdout.decode("utf-8", "replace"))
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def llvm_candidates() -> list[Path]:
    """Where an LLVM 23 bin may live: LLVM_BIN, the setup_mlir.sh env, apt.llvm.org's prefix."""
    dirs = [Path(os.environ["LLVM_BIN"])] if os.environ.get("LLVM_BIN") else []
    prefix = Path(os.environ.get("MAMBA_ROOT_PREFIX") or (cache_root() / "mamba"))
    dirs.append(prefix / "envs" / f"m{LLVM_MAJOR}" / "bin")
    dirs.append(Path(f"/usr/lib/llvm-{LLVM_MAJOR}/bin"))
    return dirs


def installed_llvm() -> dict:
    """The first candidate whose clang is major 23, with its llvm-config and mlir-opt versions."""
    for directory in llvm_candidates():
        clang = directory / "clang"
        version = tool_version(clang, CLANG_VERSION) if clang.is_file() else None
        if version and version[0] == LLVM_MAJOR:
            return {
                "bin": directory,
                "clang": version,
                "llvm": tool_version(directory / "llvm-config", re.compile(r"(\d+)\.(\d+)\.(\d+)")),
                "mlir": tool_version(directory / "mlir-opt", LLVM_VERSION),
            }
    return {"bin": None, "clang": None, "llvm": None, "mlir": None}


def node_candidates() -> list[Path]:
    found = [Path(os.environ["BCIR_NODE_BIN"]) / "node"] if os.environ.get("BCIR_NODE_BIN") else []
    on_path = shutil.which("node")
    if on_path:
        found.append(Path(on_path))
    for base, pattern in (
        (cache_root() / "node", f"node-v{NODE_MAJOR}.*/bin/node"),
        (Path("/opt/nvm/versions/node"), f"v{NODE_MAJOR}.*/bin/node"),
        (Path.home() / ".nvm" / "versions" / "node", f"v{NODE_MAJOR}.*/bin/node"),
    ):
        if base.is_dir():
            found.extend(sorted(base.glob(pattern)))
    return found


def installed_node() -> dict:
    """The highest Node 24 among the candidates."""
    best: dict = {"bin": None, "node": None}
    for node in node_candidates():
        version = tool_version(node, re.compile(r"^v(\d+)\.(\d+)\.(\d+)", re.M))
        if (
            version
            and version[0] == NODE_MAJOR
            and (best["node"] is None or version > best["node"])
        ):
            best = {"bin": node.parent, "node": version}
    return best


# --- verdicts -------------------------------------------------------------------------------------


def verdict(installed: Version | None, newest: Version | None) -> str:
    """CURRENT, OUTDATED or MISSING when the newest release is known; UNKNOWN when it is not --
    never CURRENT on an unread source."""
    if installed is None:
        return "MISSING"
    if newest is None:
        return "UNKNOWN"
    return "CURRENT" if installed >= newest else "OUTDATED"


def exit_code(verdicts: list[str]) -> int:
    if any(v in ("MISSING", "OUTDATED", "INCOHERENT") for v in verdicts):
        return 1
    if any(v == "UNKNOWN" for v in verdicts):
        return 2
    return 0 if verdicts else 2


def status(offline: bool = False) -> int:
    llvm = installed_llvm()
    node = installed_node()
    llvm_newest, llvm_source = (None, "not queried (--offline)") if offline else query_llvm()
    node_newest, node_source = (None, "not queried (--offline)") if offline else query_node()
    rows = []
    for name, installed in (
        ("clang", llvm["clang"]),
        ("llvm", llvm["llvm"]),
        ("mlir", llvm["mlir"]),
    ):
        state = verdict(installed, llvm_newest)
        if state != "MISSING" and llvm["clang"] and installed != llvm["clang"]:
            state = "INCOHERENT"  # one major, one release: every tool of the bin is the same build
        rows.append(
            (f"{name} {LLVM_MAJOR}", installed, llvm["bin"], llvm_newest, llvm_source, state)
        )
    node_state = verdict(node["node"], node_newest)
    rows.append(
        (f"node {NODE_MAJOR}", node["node"], node["bin"], node_newest, node_source, node_state)
    )
    for label, installed, where, newest, source, state in rows:
        print(
            f"latest-toolchain: {label:8} installed {fmt(installed):9} ({where or 'not found'})  "
            f"newest {fmt(newest):9} ({source})  {state}"
        )
    code = exit_code([row[-1] for row in rows])
    overall = {0: "CURRENT", 1: "BEHIND", 2: "UNKNOWN"}[code]
    print(f"latest-toolchain: {overall}")
    if llvm["bin"] is None or (llvm_newest and llvm["clang"] and llvm["clang"] < llvm_newest):
        print(
            "  LLVM 23: BCIR_LOCAL_FULL=1 bash tools/local/setup_mlir.sh  (or apt.llvm.org's llvm-toolchain-*-23)"
        )
    if node["bin"] is None or (node_newest and node["node"] and node["node"] < node_newest):
        print("  Node 24: python3 tools/local/latest_toolchain.py install-node")
    return code


def env() -> int:
    llvm = installed_llvm()
    node = installed_node()
    if llvm["bin"] is None or node["bin"] is None:
        print(
            f"latest-toolchain: env needs both (LLVM {LLVM_MAJOR}: {llvm['bin'] or 'missing'}, "
            f"Node {NODE_MAJOR}: {node['bin'] or 'missing'}); run `status` for the install commands",
            file=sys.stderr,
        )
        return 1
    print(f"export LLVM_BIN={shlex.quote(str(llvm['bin']))}")
    print(f"export BCIR_NODE_BIN={shlex.quote(str(node['bin']))}")
    print('export PATH="${LLVM_BIN}:${BCIR_NODE_BIN}:${PATH}"')
    return 0


# --- install-node ---------------------------------------------------------------------------------


def parse_shasums(text: str) -> dict[str, str]:
    sums: dict[str, str] = {}
    for line in text.splitlines():
        match = SHASUM_LINE.match(line.strip())
        if match:
            sums[match[2]] = match[1]
    return sums


def signed_shasums(asc: bytes, keys_dir: Path | None) -> str:
    """The checksum list out of the clearsigned SHASUMS256.txt.asc, only when gpg finds a good
    signature by a Node release key (the nodejs/release-keys set, or `keys_dir`)."""
    gpg = shutil.which("gpg")
    if gpg is None:
        raise Refused("gpg is not installed, so the checksum list's signature cannot be verified")
    with tempfile.TemporaryDirectory(prefix="bcir-node-keys-") as tmp:
        home = Path(tmp) / "gnupg"
        home.mkdir(mode=0o700)
        if keys_dir is None:
            keys_dir = Path(tmp) / "release-keys"
            try:
                cloned = subprocess.run(
                    ["git", "clone", "--depth", "1", "-q", RELEASE_KEYS_REPO, str(keys_dir)],
                    capture_output=True,
                    timeout=TIMEOUT,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise Refused(f"the Node release keys could not be fetched: {exc}") from exc
            if cloned.returncode != 0:
                raise Refused(
                    f"the Node release keys could not be fetched: {_last_line(cloned.stderr)}"
                )
            keys_dir = keys_dir / "keys"
        keys = sorted(keys_dir.glob("*.asc"))
        if not keys:
            raise Refused(f"no release keys (*.asc) in {keys_dir}")
        environ = {**os.environ, "GNUPGHOME": str(home)}
        subprocess.run(
            [gpg, "--batch", "--quiet", "--import", *map(str, keys)],
            capture_output=True,
            timeout=TIMEOUT,
            env=environ,
        )
        signed = Path(tmp) / "SHASUMS256.txt.asc"
        signed.write_bytes(asc)
        checked = subprocess.run(
            [gpg, "--batch", "--status-fd", "1", "--decrypt", str(signed)],
            capture_output=True,
            timeout=TIMEOUT,
            env=environ,
        )
        status_lines = checked.stdout.decode("utf-8", "replace")
        if checked.returncode != 0 or "[GNUPG:] GOODSIG" not in status_lines:
            raise Refused(f"no good signature on SHASUMS256.txt.asc: {_last_line(checked.stderr)}")
        # With --status-fd 1 the status lines share stdout with the plaintext; keep the plaintext.
        return "\n".join(
            line for line in status_lines.splitlines() if not line.startswith("[GNUPG:]")
        )


def install_node(
    version: str | None,
    dest: Path,
    keys_dir: Path | None,
    allow_unsigned: bool,
    machine: str | None = None,
    system: str | None = None,
) -> int:
    system = system or sys.platform
    machine = (machine or (os.uname().machine if hasattr(os, "uname") else "")).lower()
    platform = PLATFORMS.get(machine)
    if not system.startswith("linux") or platform is None:
        print(f"install-node: UNUSABLE: Linux x64/arm64 only (here: {system} {machine or '?'})")
        return 2
    if version is None:
        newest, source = query_node()
        if newest is None:
            print(
                f"install-node: UNKNOWN: the newest Node {NODE_MAJOR} could not be read ({source})"
            )
            return 2
        version = fmt(newest)
    version = version.lstrip("v")
    if not re.fullmatch(rf"{NODE_MAJOR}\.\d+\.\d+", version):
        print(f"install-node: UNUSABLE: {version!r} is not a Node {NODE_MAJOR}.x.y release")
        return 2
    name = f"node-v{version}-{platform}"
    target = dest / name
    node = target / "bin" / "node"
    if tool_version(node, re.compile(r"^v(\d+)\.(\d+)\.(\d+)", re.M)) == tuple(
        map(int, version.split("."))
    ):
        print(f"install-node: v{version} already installed at {target / 'bin'}")
        return 0
    base = f"{NODE_DIST}/v{version}"
    try:
        sums_text = fetch(f"{base}/SHASUMS256.txt", TEXT_LIMIT).decode("utf-8", "replace")
        if allow_unsigned:
            note = "checksum list NOT signature-verified (--allow-unsigned)"
        else:
            sums_text = signed_shasums(fetch(f"{base}/SHASUMS256.txt.asc", TEXT_LIMIT), keys_dir)
            note = "checksum list signature-verified against the Node release keys"
        expected = parse_shasums(sums_text).get(f"{name}.tar.xz")
        if expected is None:
            raise Refused(f"the checksum list has no entry for {name}.tar.xz")
        data = fetch(f"{base}/{name}.tar.xz", TARBALL_LIMIT)
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise Refused(f"{name}.tar.xz sha256 {actual} != the list's {expected}")
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".node-", dir=dest) as tmp:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as archive:
                try:  # the data filter: no absolute paths, no member or link escaping tmp
                    archive.extractall(tmp, filter="data")
                except tarfile.FilterError as exc:
                    raise Refused(f"the tarball writes outside its directory: {exc}") from exc
            unpacked = Path(tmp) / name
            if not (unpacked / "bin" / "node").is_file():
                raise Refused(f"the tarball does not unpack to {name}/bin/node")
            if target.exists():
                shutil.rmtree(target)
            unpacked.rename(target)
    except Refused as exc:
        print(f"install-node: REFUSED: {exc}")
        return 1
    except (OSError, ValueError, tarfile.TarError, http.client.HTTPException) as exc:
        print(f"install-node: UNAVAILABLE: {exc}")
        return 2
    print(
        f"install-node: v{version} installed at {target / 'bin'} ({note}; sha256 {actual[:16]}...)"
    )
    print(f"  export BCIR_NODE_BIN={shlex.quote(str(target / 'bin'))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    st = sub.add_parser("status", help="installed vs newest LLVM/Clang/MLIR 23 and Node 24")
    st.add_argument(
        "--offline", action="store_true", help="do not query upstream (verdict UNKNOWN)"
    )
    sub.add_parser("env", help="shell exports putting LLVM 23 and Node 24 first on PATH")
    inst = sub.add_parser("install-node", help="install the newest (or --version) Node 24")
    inst.add_argument("--version", help=f"a v{NODE_MAJOR}.x.y release (default: the newest)")
    inst.add_argument("--dest", type=Path, default=cache_root() / "node")
    inst.add_argument("--keys-dir", type=Path, help="Node release keys (*.asc); default: cloned")
    inst.add_argument(
        "--allow-unsigned",
        action="store_true",
        help="accept the checksum list without verifying its signature (recorded in the output)",
    )
    args = parser.parse_args(argv)
    if args.command == "env":
        return env()
    if args.command == "install-node":
        return install_node(args.version, args.dest, args.keys_dir, args.allow_unsigned)
    return status(offline=getattr(args, "offline", False))


if __name__ == "__main__":
    sys.exit(main())
