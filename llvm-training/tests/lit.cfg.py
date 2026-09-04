# Minimal lit configuration for repository-local llvm-training checks.

import os
import shutil

import lit.formats

config.name = "LLVMTraining"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".test"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root

training_root = os.path.dirname(config.test_source_root)
repo_root = os.path.dirname(training_root)
config.substitutions.append(("%training_root", training_root))
config.substitutions.append(("%repo_root", repo_root))

config.environment["PATH"] = os.environ.get("PATH", "")


def find_versioned_tool(base):
    direct = shutil.which(base)
    if direct:
        return direct
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    candidates = []
    for directory in path_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name.startswith(base + "-") and name[len(base) + 1 :].isdigit():
                candidates.append((int(name[len(base) + 1 :]), os.path.join(directory, name)))
    return max(candidates, default=(None, None))[1]


for tool in ["bash", "diff", "llvm-as", "opt", "lli", "mlir-opt"]:
    resolved = find_versioned_tool(tool)
    if resolved:
        config.available_features.add(tool)
        config.environment[tool.upper().replace("-", "_") + "_BINARY"] = resolved
