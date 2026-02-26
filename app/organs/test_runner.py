from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class TestRunnerResult:
    ok: int
    applied: int
    apply_error: str
    results: List[Dict[str, Any]]
    sandbox: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": int(self.ok),
            "applied": int(self.applied),
            "apply_error": self.apply_error or "",
            "results": self.results,
            "sandbox": self.sandbox,
        }


def _safe_copytree(src: Path, dst: Path) -> None:
    # Copy repo into a sandbox, excluding obvious large/ephemeral folders.
    exclude = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        "dist", "build", ".idea", ".vscode",
    }
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        # prune dirs
        dirs[:] = [d for d in dirs if d not in exclude and not d.endswith(".egg-info")]
        if any(part in exclude for part in rel.parts):
            continue
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for fn in files:
            if fn in {"bunny.db", "main.exe"}:
                continue
            if fn.endswith((".pyc", ".pyo")):
                continue
            sp = Path(root) / fn
            dp = target_dir / fn
            try:
                shutil.copy2(sp, dp)
            except Exception:
                # best effort
                pass


def _apply_patch_git(repo_root: Path, patch_text: str) -> Tuple[int, str]:
    if not patch_text.strip():
        return 1, ""
    patch_file = repo_root / ".bunny_patch.diff"
    patch_file.write_text(patch_text, encoding="utf-8", errors="ignore")
    try:
        p = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_file)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
        if p.returncode == 0:
            return 1, ""
        return 0, (p.stdout or "")[-4000:]
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"
    finally:
        try:
            patch_file.unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass


def _run_commands(repo_root: Path, commands: List[List[str]], timeout_s: int = 240) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cmd in commands:
        try:
            p = subprocess.run(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                env={**os.environ},
            )
            out.append({"cmd": cmd, "rc": int(p.returncode), "out": (p.stdout or "")[-8000:]})
        except Exception as e:
            out.append({"cmd": cmd, "rc": 999, "out": f"{type(e).__name__}: {e}"})
    return out


def run_patch_tests(repo_root: str, patch_diff: str, commands: List[List[str]]) -> Dict[str, Any]:
    """Real test runner: applies the patch in an isolated sandbox and executes commands.

    - No network access assumed.
    - Uses git apply when available; if patch can't be applied, tests run on the unpatched sandbox and ok=0.
    """
    src = Path(repo_root).resolve()
    with tempfile.TemporaryDirectory(prefix="bunny_sandbox_") as td:
        sandbox = Path(td)
        _safe_copytree(src, sandbox)

        applied, apply_err = _apply_patch_git(sandbox, patch_diff)

        results = _run_commands(sandbox, commands or [["python", "-m", "compileall", "."]])
        ok = int(applied == 1 and all(r.get("rc") == 0 for r in results))
        tr = TestRunnerResult(ok=ok, applied=applied, apply_error=apply_err, results=results, sandbox=str(sandbox))
        # Return as plain dict (db storage)
        return tr.to_dict()
