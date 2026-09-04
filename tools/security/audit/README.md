# Dependency-audit tooling (report-only)

The gates are one directory up (`audit_dependencies.py` runs pip-audit, required in CI). The
scripts here reproduce the **dated audits** in `docs/security/DEPENDENCY_AUDIT_<date>.md`: they
report, they do not gate, and each exits non-zero only when it found something or could not
read a source it needed (a hole in a report must never read as clean).

| Script | Question it answers | Needs |
|---|---|---|
| `import_scan.py` | Which third-party modules does the tree actually import, and is each one declared in `pyproject.toml`? | nothing (stdlib) |
| `actions_currency.py` | Is every `uses:` SHA and every pre-commit `rev` a release commit; which release; how far behind upstream's latest and the latest in its major? | `git ls-remote` reaching github.com |
| `osv_pypi.py` | Offline second-source check of every `name==version` in pip-audit's JSON reports against the OSV PyPI export | `packaging` (pip-audit's environment), the OSV export |
| `osv_ubuntu.py` | Ubuntu advisories for the apt packages the workflows install, at the versions the archive resolves today, classified with `dpkg --compare-versions` | `apt-cache`/`dpkg` on a host of the runner's release, the OSV export |

Reproduce the 2026-09-04 audit from the repository root (a scratch venv with
`pip-audit==2.10.1` provides both the engine and `packaging`):

```bash
python tools/security/audit/import_scan.py --out work/import_scan.json
python tools/security/audit/actions_currency.py --out work/actions_currency.json
PIP_AUDIT=venv/bin/pip-audit python tools/security/audit_dependencies.py --require-advisory --json-out work/rail_report.json
venv/bin/pip-audit -r <declaration>  --format json --strict --desc off --progress-spinner off > work/resolved.json
venv/bin/pip-audit -r <floor pins>   --no-deps --disable-pip --format json --strict --desc off --progress-spinner off > work/floor.json
venv/bin/python tools/security/audit/osv_pypi.py --osv-dir work/osv --report work/resolved.json --report work/floor.json \
    --pin setuptools==84.0.0 --out work/osv_pypi.json
python tools/security/audit/osv_ubuntu.py --osv-dir work/osv --release 24.04 --binary clang --binary nodejs ... --out work/osv_ubuntu.json
```

The evidence each run produced is committed beside the report under
`docs/security/audit-<date>/`, so the next audit is a diff against it.
