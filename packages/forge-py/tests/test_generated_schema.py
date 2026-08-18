import os
import subprocess
import sys
import pytest


def test_generated_contracts_are_current():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    script_path = os.path.join(repo_root, "scripts", "generate_release1_contracts.py")
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "--check",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Schema drift detected:\n{result.stderr}\n{result.stdout}"
