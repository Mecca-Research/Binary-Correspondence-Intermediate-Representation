# Autograder security model

## Trust boundary

Every attempt file, filename, manifest supplied outside the checkout, generator
response, and byte of generator stdout/stderr is **untrusted data**. Checked-in
exercise prompts, solutions, registries, and grader code are trusted only after
normal repository review. The baseline grader is a deterministic validation
harness, not a security sandbox.

The grader accepts answers only beneath the configured attempt root. It resolves
both the root and candidate (including symlinks) and rejects lexical or symlink
escapes. Answer files are size-limited. Generated shell scripts, build-system
files, pass plugins, native objects, libraries, and arbitrary executables are
rejected and are never executed by the baseline.

## External process policy

All repository grader process launches use argument arrays with `subprocess.run`;
submission text and filenames are never interpolated into a shell command. Each
launch has:

- a finite timeout;
- stdout/stderr redirected to files and read back with a fixed byte limit;
- a fresh temporary working directory and temporary `HOME`/`TMPDIR`;
- a small, controlled locale/timezone/path environment with proxy variables
  cleared; and
- no grader feature that implicitly downloads dependencies or contacts a
  network service.

These controls reduce accidental exposure but do **not** remove network access
from a process at the operating-system level. Production multi-tenant grading
must additionally use an OS/container sandbox with an unprivileged identity,
read-only inputs, resource and process limits, syscall restrictions, and a
network namespace with no egress.

## Allowed execution

Parsing and verification use fixed trusted LLVM/MLIR executables and fixed
built-in arguments. The grader does not honor plugin-loading options from an
answer or manifest. `lli` is more dangerous because submitted IR becomes code:
it is used only for exercises explicitly marked `safe_deterministic_lli` in the
trusted registry. New allowlist entries require review for determinism, bounded
behavior, and the absence of external calls or ambient-resource dependencies.

A request to **review a C++ pass skeleton** is textual/rubric review. Compiling,
linking, or loading an untrusted pass plugin is a different operation requiring
a stronger compiler/runtime sandbox; it remains disabled by default and is out
of scope for the baseline grader.

## Reporting and failures

Every score report records the exact resolved path and first line of version
output for each discovered tool; missing tools are recorded explicitly. Check
records distinguish:

- `grader_failure`: a trusted grader/configuration failure;
- `timeout`: an external tool exceeded its deadline;
- `missing_tool`: a required executable was not found;
- `invalid_answer`: a missing, empty, oversized, malformed, or rejected answer;
- `incorrect_answer`: a valid answer that did not satisfy a graded assertion;
- `pass`: a successful check.

Generator contract failures are infrastructure failures, while valid but wrong
or incomplete answers are quality failures.

## Maintainer rules

1. Never add `shell=True`, `os.system`, shell pipelines, or command-string
   interpolation to a submission path.
2. Never execute files selected by a submission, even if their extension or
   executable bit looks reasonable.
3. Keep plugin compilation/loading disabled in the baseline.
4. Add an explicit registry allowlist and tests before introducing any new form
   of submitted-code execution.
5. Treat expanded output limits, timeouts, environment inheritance, or network
   access as security-sensitive review items.
