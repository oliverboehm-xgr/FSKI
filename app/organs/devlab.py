from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.net import http_post_json


@dataclass
class DevLabConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.3"
    temperature: float = 0.2
    num_ctx: int = 6144
    stream: bool = False
    max_patch_chars: int = 18000


def _ollama_chat(cfg: DevLabConfig, system: str, user: str) -> str:
    url = cfg.host.rstrip("/") + "/api/chat"
    payload = {
        "model": cfg.model,
        "stream": cfg.stream,
        "options": {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    status, txt = http_post_json(url, payload, timeout=180)
    if status == 0:
        raise RuntimeError(txt)
    if status >= 400:
        raise RuntimeError(f"ollama /api/chat HTTP {status}: {txt[:200]}")
    data = json.loads(txt or "{}")
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        i = (text or "").find("{")
        if i < 0:
            return {}
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def run_tests(repo_root: str, commands: List[List[str]]) -> Dict[str, Any]:
    results = []
    for cmd in commands:
        try:
            p = subprocess.run(cmd, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
            results.append({"cmd": cmd, "rc": p.returncode, "out": (p.stdout or "")[-4000:]})
        except Exception as e:
            results.append({"cmd": cmd, "rc": 999, "out": f"{type(e).__name__}: {e}"})
    ok = int(all(r["rc"] == 0 for r in results)) if results else 0
    return {"ok": ok, "results": results}


def propose_patch_and_tests(
    cfg: DevLabConfig,
    axioms: Dict[str, str],
    cluster: Dict[str, Any],
    examples: List[Dict[str, Any]],
    skills: List[Dict[str, Any]],
    repo_hint: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Generate an engineering-grade proposal with optional patch diff + test plan.

    IMPORTANT: This organ does NOT auto-apply patches.
    """
    system = (
        "You are Bunny's DevLab organ (code-review + patch proposal + test plan). "
        "You read a failure cluster, examples, and existing skills. "
        "You must return STRICT JSON. No prose outside JSON. "
        "Output must be generic (no user-specific hacks). "
        "If you propose hardware/capability additions, include a minimal adapter/API plan and a mock-based test plan."
    )
    prompt = {
        "axioms": axioms,
        "cluster": cluster,
        "examples": examples[-12:],
        "skills": skills[-8:],
        "repo_hint": repo_hint or {},
        "output_schema": {
            "proposal_v2": {
                "title": "str",
                "linked_axioms": ["A1","A2","A3","A4"],
                "problem": "str",
                "evidence": [{"kind":"caught|selfeval|metric","ref":"id or note","detail":"str"}],
                "hypothesis": "str",
                "change": {
                    "type": "code_patch|new_capability|hardware|config",
                    "summary": "str",
                    "files": ["optional list of files/modules"],
                },
                "acceptance_tests": [{"name":"str","procedure":"str","pass":"str"}],
                "metrics_expected": {"helpfulness":"+0..1","coherence":"+0..1","pain_psych":"-0..1","err_rate":"-0..1","lat_p95":"-0..1"},
                "risks": [{"risk":"str","mitigation":"str"}],
                "rollback": "str",
            },
            "patch_unified_diff": "string (optional, unified diff, may be empty)",
            "test_commands": [["python","-m","compileall","."], ["pytest","-q"]],
        },
        "rules": [
            "Prefer smallest viable change.",
            "If patch_unified_diff is present, it must be plausible and reference existing paths.",
            "Never claim tests passed unless you actually ran them (DevLab will run them separately).",
        ],
    }
    txt = _ollama_chat(cfg, system, json.dumps(prompt, ensure_ascii=False))
    obj = _extract_json(txt)
    prop = obj.get("proposal_v2") if isinstance(obj.get("proposal_v2"), dict) else {}
    patch = str(obj.get("patch_unified_diff") or "")[:cfg.max_patch_chars]
    tcmds = obj.get("test_commands")
    if not isinstance(tcmds, list):
        tcmds = [["python","-m","compileall","."]]
    obj2 = {"proposal_v2": prop, "patch_unified_diff": patch, "test_commands": tcmds}
    return obj2
