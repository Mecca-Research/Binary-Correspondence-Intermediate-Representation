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

for tool in ["bash", "diff", "llvm-as", "opt"]:
    if shutil.which(tool):
        config.available_features.add(tool)
