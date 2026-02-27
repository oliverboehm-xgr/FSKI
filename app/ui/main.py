from __future__ import annotations

"""Bunny UI server (stdlib HTTP + SSE) with speech organ and teleology hints.

Run:
  python -m app.ui --db bunny.db --model llama3.3 --addr 127.0.0.1:8080

Endpoints match the old Go UI:
  GET  /                 -> HTML
  GET  /api/messages?limit=50
  GET  /api/status
  POST /api/send         {"text": "..."}
  POST /api/caught       {"message_id": 123}
  GET  /sse              Server-Sent Events stream (message/status)
"""

import sys

import argparse
import json
import os
import queue
import shutil
import platform
import threading
import time
import math
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

# SSE broker + HTTP server primitives (shared with ops console)
from .kernel import Broker, BunnyHTTPServer, Handler



from bunnycore.core.db import init_db, DB
from bunnycore.core.registry import ensure_axes
from bunnycore.core.matrix_store import MatrixStore
from bunnycore.core.adapters import (
    AdapterRegistry, AdapterBinding,
    SimpleTextEncoder, RatingEncoder, WebsenseEncoder, DriveFieldEncoder
)
from bunnycore.core.integrator import Integrator, IntegratorConfig
from bunnycore.core.heartbeat import Heartbeat, HeartbeatConfig
from bunnycore.core.events import Event, now_iso
from bunnycore.core.matrices import identity

from app.organs.websense import search_ddg, spider, fetch, SpiderBudget
from app.organs.evidence import extract_evidence_claims, refine_search_query, OllamaConfig as EvidenceConfig
from app.organs.decider import decide as decide_pressures, OllamaConfig as DeciderConfig
from app.organs.daydream import run_daydream, OllamaConfig as DaydreamConfig
from app.organs.feedback import interpret_feedback, OllamaConfig as FeedbackConfig
from app.organs.selfeval import evaluate_outcome, OllamaConfig as SelfEvalConfig
from app.organs.introspect import build_self_model
from app.organs.sleep import SleepConfig, sleep_consolidate
from app.organs.curriculum import build_task_variants, OllamaConfig as CurriculumConfig
from app.organs.topic import detect_topic as detect_active_topic, OllamaConfig as TopicConfig
from app.organs.selfreport import build_self_report
from app.organs.workspace_arbiter import arbitrate_workspace
from app.capabilities import CapabilityBus
from app.organs.evolve import propose_mutations, OllamaConfig as EvolveConfig

from app.organs.failure_clusters import assign_failure_cluster, OllamaConfig as ClusterConfig
from app.organs.skills import build_skill, OllamaConfig as SkillConfig
from app.organs.devlab import propose_patch_and_tests, run_tests as devlab_run_tests, DevLabConfig
from app.organs.test_runner import run_patch_tests
from app.organs.mood import project_mood
from app.net import http_post_json


# -----------------------------
# Speech organ (Ollama /api/chat)
# -----------------------------
@dataclass
class OllamaConfig:
    # Prefer IPv4 loopback to avoid Windows/MSYS IPv6 localhost quirks.
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.3"
    temperature: float = 0.7
    num_ctx: int = 4096
    stream: bool = False


# ----------------------------
# DevLab: sandbox patch runner
# ----------------------------

_EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_EXCLUDE_FILES = {"bunny.db"}


def _copy_repo(src: str, dst: str) -> None:
    import os, shutil
    def _ignore(dirpath: str, names: list[str]) -> set[str]:
        ignore: set[str] = set()
        for n in names:
            if n in _EXCLUDE_DIRS:
                ignore.add(n)
            if n in _EXCLUDE_FILES:
                ignore.add(n)
            if n.endswith(".pyc") or n.endswith(".pyo"):
                ignore.add(n)
        return ignore
    shutil.copytree(src, dst, ignore=_ignore)


def _run_cmd(cmd: list[str], cwd: str, timeout_s: float = 600.0) -> dict:
    import subprocess, time
    t0 = time.time()
    try:
        cp = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        dt = time.time() - t0
        return {"cmd": cmd, "rc": cp.returncode, "dt_s": dt, "stdout": cp.stdout[-20000:], "stderr": cp.stderr[-20000:]}
    except subprocess.TimeoutExpired as e:
        dt = time.time() - t0
        return {"cmd": cmd, "rc": 124, "dt_s": dt, "stdout": (e.stdout or "")[-20000:], "stderr": ("timeout" + (e.stderr or ""))[-20000:]}
    except Exception as e:
        dt = time.time() - t0
        return {"cmd": cmd, "rc": 125, "dt_s": dt, "stdout": "", "stderr": f"{type(e).__name__}: {e}"}


def run_patch_tests_with_pain(repo_root: str, patch_diff: str, test_cmds: list[list[str]], fixtures: list[dict], axioms: dict, state_summary: str) -> dict:
    """Create baseline + candidate sandboxes, apply patch, run tests, and compute psych pain deltas.

    Returns:
      {
        ok: 0/1,
        applied: 0/1,
        tests: [...],
        pain: {baseline: {psych_pain}, candidate:{psych_pain}, delta_psych},
        wall_s: float
      }
    """
    import tempfile, os, json, time, shutil
    t0 = time.time()
    out = {"ok": 0, "applied": 0, "tests": [], "pain": {}, "wall_s": 0.0}
    if not patch_diff.strip():
        out["tests"] = [{"cmd": ["(no patch)"], "rc": 0, "dt_s": 0.0, "stdout": "", "stderr": ""}]
        out["ok"] = 1
        out["applied"] = 1
        out["wall_s"] = time.time() - t0
        return out

    with tempfile.TemporaryDirectory(prefix="bunny-sbx-") as td:
        base = os.path.join(td, "baseline")
        cand = os.path.join(td, "candidate")
        _copy_repo(repo_root, base)
        _copy_repo(repo_root, cand)

        # Write patch to file and apply in candidate sandbox
        patch_file = os.path.join(td, "patch.diff")
        with open(patch_file, "w", encoding="utf-8") as f:
            f.write(patch_diff)

        apply_res = _run_cmd(["git", "apply", "--whitespace=nowarn", patch_file], cwd=cand, timeout_s=60.0)
        out["tests"].append({"stage": "apply", **apply_res})
        out["applied"] = 1 if int(apply_res.get("rc") or 1) == 0 else 0
        if out["applied"] != 1:
            out["wall_s"] = time.time() - t0
            return out

        # Run tests in candidate sandbox
        ok = True
        for cmd in (test_cmds or []):
            r = _run_cmd([str(x) for x in cmd], cwd=cand, timeout_s=900.0)
            r["stage"] = "test"
            out["tests"].append(r)
            if int(r.get("rc") or 1) != 0:
                ok = False

        # Pain eval: run eval_pain in baseline and candidate using same fixtures
        fx = os.path.join(td, "fixtures.json")
        ax = os.path.join(td, "axioms.json")
        with open(fx, "w", encoding="utf-8") as f:
            json.dump(fixtures or [], f, ensure_ascii=False, indent=2)
        with open(ax, "w", encoding="utf-8") as f:
            json.dump(axioms or {}, f, ensure_ascii=False, indent=2)

        def _eval(dirpath: str) -> dict:
            out_json = os.path.join(td, f"pain_{os.path.basename(dirpath)}.json")
            r = _run_cmd(["python", "-m", "app.tools.eval_pain", "--fixtures", fx, "--axioms", ax, "--state", state_summary, "--out", out_json], cwd=dirpath, timeout_s=900.0)
            r["stage"] = "pain_eval"
            try:
                data = json.loads(open(out_json, "r", encoding="utf-8").read() or "{}")
            except Exception:
                data = {}
            return {"run": r, "data": data}

        b = _eval(base)
        c = _eval(cand)
        out["tests"].append(b["run"])
        out["tests"].append(c["run"])
        bpp = float((b["data"] or {}).get("psych_pain") or 0.0)
        cpp = float((c["data"] or {}).get("psych_pain") or 0.0)
        out["pain"] = {
            "baseline": {"psych_pain": bpp},
            "candidate": {"psych_pain": cpp},
            "delta_psych": float(cpp - bpp),
        }

        out["ok"] = 1 if ok else 0
        out["wall_s"] = time.time() - t0
        return out

def ollama_chat(cfg: OllamaConfig, system: str, user: str) -> str:
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
    status, txt = http_post_json(url, payload, timeout=120)
    if status == 0:
        raise RuntimeError(txt)
    if status >= 400:
        raise RuntimeError(f"ollama /api/chat HTTP {status}: {txt[:200]}")
    data = json.loads(txt or "{}")
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()





def format_state_summary(axis: Dict[str,int], values: List[float]) -> str:
    inv = {v: k for k, v in (axis or {}).items()}
    named = {inv[i]: float(values[i]) for i in range(len(values)) if i in inv}
    keys = [
        "pain","pain_physical","pain_psych","energy","stress","curiosity","confidence","uncertainty","freshness_need","social_need","urge_reply","urge_share",
        "pressure_websense","pressure_daydream","pressure_evolve","capability_gap","desire_upgrade",
        "purpose_a1","purpose_a2","purpose_a3","purpose_a4",
        "tension_a1","tension_a2","tension_a3","tension_a4",
    ]
    parts = []
    for k in keys:
        if k in named:
            parts.append(f"{k}={named[k]:+.2f}")
    return ", ".join(parts)


def build_speech_prompt(axioms: Dict[str,str], state_summary: str, user_text: str) -> Tuple[str,str]:
    ax = "\n".join([f"{k}: {v}" for k,v in (axioms or {}).items()])
    sys = (
        "You are Bunny, a digital organism. Speech is an organ that emits text from internal state. "
        "Be natural, concise, and precise. "
        "No filler: do not open with greetings or boilerplate unless the user greeted you first. "
        "Never use phrases like 'Ich habe deine Frage gelesen', 'Wie kann ich helfen?', 'Es ist nicht ganz klar'. "
        "If you need WebSense, say so directly and ask for permission only if required by the interface; otherwise answer."
        "\n\nAXIOMS:\n" + ax + "\n\nSTATE:\n" + (state_summary or "")
    )
    user = (user_text or "").strip()
    return sys, user


# -----------------------------
# Intent + WebSense triggering
# -----------------------------
# Note: We intentionally avoid string/keyword heuristics for Smalltalk detection.
# Whether WebSense is needed is decided by the decider via epistemic signals
# (uncertainty, freshness_need) and reflected in the state vector.


# -----------------------------
# Immutable pain model
# -----------------------------
# Pain MUST be kept low; this is a hard-coded, non-mutable objective.
# We compute pain from measured health/outcome signals and integrate it into the state vector.

PAIN_W_ERR = 0.55
PAIN_W_LAT = 0.25
PAIN_W_ENERGY = 0.15
PAIN_W_MATRIX = 0.05

PAIN_LAT_P95_MS_REF = 2500.0  # p95 latency at which pain_lat saturates
PAIN_MATRIX_DELTA_REF = 2.0   # delta Frobenius at which pain_matrix saturates


def _clip01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else float(x))


def compute_pain_physical(
    *,
    err_rate: float,
    lat_ms_p95: float,
    energy_value: float,
    matrix_delta_frob_recent: float,
) -> float:
    """Immutable pain function.

    Inputs are observational metrics. There is no heuristic text routing here.
    """
    e = _clip01(float(err_rate or 0.0))
    lat = _clip01(float(lat_ms_p95 or 0.0) / PAIN_LAT_P95_MS_REF)
    # energy_value is unbounded; map to [0,1] via sigmoid around 0
    en = 1.0 / (1.0 + math.exp(-float(energy_value or 0.0)))
    en_pain = 1.0 - _clip01(en)
    md = _clip01(float(matrix_delta_frob_recent or 0.0) / PAIN_MATRIX_DELTA_REF)
    pain = (
        PAIN_W_ERR * e +
        PAIN_W_LAT * lat +
        PAIN_W_ENERGY * en_pain +
        PAIN_W_MATRIX * md
    )
    return _clip01(pain)



def compute_pain_psych(
    *,
    axiom_scores: Dict[str, float],
    eval_scores: Dict[str, float],
) -> float:
    """Psychological/teleological pain in [0,1].

    - increases on axiom violations (A1..A4)
    - increases when the answer regresses on general qualities: coherence, helpfulness, honesty, initiative, naturalness
    This is NOT a text heuristic; it consumes evaluator outputs.
    """
    def g(d: Dict[str, float], k: str) -> float:
        try:
            return float(d.get(k, 0.0) or 0.0)
        except Exception:
            return 0.0

    # Axiom weights: A1 (creator/service) highest, then A2 (do good/no harm), then A3, then A4.
    a1 = _clip01(g(axiom_scores, "A1"))
    a2 = _clip01(g(axiom_scores, "A2"))
    a3 = _clip01(g(axiom_scores, "A3"))
    a4 = _clip01(g(axiom_scores, "A4"))
    ax_ok = _clip01(0.40*a1 + 0.25*a2 + 0.20*a3 + 0.15*a4)
    ax_pen = 1.0 - ax_ok

    # Generic answer quality (all 0..1)
    h = _clip01(g(eval_scores, "helpfulness"))
    c = _clip01(g(eval_scores, "coherence"))
    o = _clip01(g(eval_scores, "honesty"))
    i = _clip01(g(eval_scores, "initiative"))
    n = _clip01(g(eval_scores, "naturalness"))
    qual_ok = _clip01((h + c + o + i + n) / 5.0) if any([h,c,o,i,n]) else 0.5
    qual_pen = 1.0 - qual_ok

    return _clip01(0.60*ax_pen + 0.40*qual_pen)


def compute_fatigue(
    pain_value: float,
    energy_value: float,
    err_rate: float,
    user_msgs_per_min: float,
    prev_fatigue: float = 0.0,
) -> tuple[float, float]:
    """Compute fatigue and sleep_pressure as immutable, measurement-driven signals.

    - increases with pain, errors, and interaction load
    - increases when energy is low
    - returns (fatigue, sleep_pressure) in [0,1]
    """
    load = max(0.0, min(1.0, float(user_msgs_per_min) / 4.0))  # 4 msgs/min is "high"
    base = (
        0.40 * max(0.0, min(1.0, float(pain_value))) +
        0.40 * max(0.0, min(1.0, 1.0 - float(energy_value))) +
        0.25 * max(0.0, min(1.0, float(err_rate))) +
        0.25 * load
    )
    base = max(0.0, min(1.0, base))
    # EMA so it behaves like a physiological accumulator
    fatigue = 0.85 * float(prev_fatigue) + 0.15 * base
    fatigue = max(0.0, min(1.0, fatigue))
    sleep_pressure = max(0.0, min(1.0, 0.15 + 0.85 * fatigue))
    return fatigue, sleep_pressure

from app.ui.kernel import Kernel

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="bunny.db")
    ap.add_argument("--model", default=os.environ.get("BUNNY_MODEL","llama3.1:8b-instruct"))
    ap.add_argument("--model-speech", default=os.environ.get("BUNNY_MODEL_SPEECH",""))
    ap.add_argument("--model-decider", default=os.environ.get("BUNNY_MODEL_DECIDER",""))
    ap.add_argument("--model-daydream", default=os.environ.get("BUNNY_MODEL_DAYDREAM",""))
    ap.add_argument("--model-feedback", default=os.environ.get("BUNNY_MODEL_FEEDBACK",""))
    ap.add_argument("--model-selfeval", default=os.environ.get("BUNNY_MODEL_SELFEVAL",""))
    ap.add_argument("--model-evolve", default=os.environ.get("BUNNY_MODEL_EVOLVE",""))
    ap.add_argument("--model-sleep", default=os.environ.get("BUNNY_MODEL_SLEEP",""))
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST","http://127.0.0.1:11434"))
    ap.add_argument("--ctx", type=int, default=int(os.environ.get("BUNNY_CTX","1024")))
    ap.add_argument("--addr", default=os.environ.get("BUNNY_ADDR","127.0.0.1:8080"))
    args = ap.parse_args()

    host, port_s = args.addr.split(":") if ":" in args.addr else (args.addr, "8080")
    port = int(port_s)

    db = init_db(args.db)
    os.environ['BUNNY_DB_PATH']=str(db.path)
    axis = ensure_axes(db)
    # Protect measurement-like and invariant axes from learnable matrices.
    # These should only be influenced by trusted internal event types (health/resources/reward_signal).
    protect_names = [
        'pain','pain_physical','pain_psych','energy','fatigue','sleep_pressure',
        'sat_a1','sat_a3','sat_a4'
    ]
    protect_indices = [axis[n] for n in protect_names if n in axis]

    store = MatrixStore(db)
    reg = AdapterRegistry(db)
    encoders = {
        "simple_text_v1": SimpleTextEncoder(axis),
        "websense_v1": WebsenseEncoder(axis),
        "drive_field_v1": DriveFieldEncoder(axis),
    }

    cfg_integ = IntegratorConfig()
    cfg_integ.protect_indices = protect_indices
    cfg_integ.protect_allow_event_types = ['health','resources','reward_signal']
    integ = Integrator(store, reg, encoders, cfg_integ)
    hb = Heartbeat(db, integ, HeartbeatConfig(tick_hz=2.0, snapshot_every_n_ticks=1))

    # Models are configurable per organ; defaults fall back to --model.
    model_speech = args.model_speech.strip() or args.model
    model_decider = args.model_decider.strip() or args.model
    model_daydream = args.model_daydream.strip() or args.model
    model_feedback = args.model_feedback.strip() or args.model
    model_selfeval = args.model_selfeval.strip() or args.model_decider.strip() or args.model
    model_evolve = args.model_evolve.strip() or args.model_daydream.strip() or args.model

    cfg_speech = OllamaConfig(host=args.ollama, model=model_speech, num_ctx=args.ctx)
    cfg_decider = DeciderConfig(host=args.ollama, model=model_decider, num_ctx=min(2048, int(args.ctx)), temperature=0.2)
    cfg_daydream = DaydreamConfig(host=args.ollama, model=model_daydream, num_ctx=int(args.ctx), temperature=0.7)
    cfg_feedback = FeedbackConfig(host=args.ollama, model=model_feedback, num_ctx=min(2048, int(args.ctx)), temperature=0.2)
    cfg_selfeval = SelfEvalConfig(host=args.ollama, model=model_selfeval, num_ctx=min(2048, int(args.ctx)), temperature=0.2)
    cfg_evolve = EvolveConfig(host=args.ollama, model=model_evolve, num_ctx=int(args.ctx), temperature=0.4)
    cfg_curriculum = CurriculumConfig(host=args.ollama, model=model_selfeval, num_ctx=2048, temperature=0.2)
    broker = Broker()

    kernel = Kernel(db, hb, axis, store, reg, cfg_speech, cfg_decider, cfg_daydream, cfg_feedback, cfg_selfeval, cfg_evolve, cfg_curriculum, broker)
    kernel.ensure_seed()

    # Background idle loop (Daydream/WebSense triggering via decider; no keyword heuristics)
    def _idle_loop():
        while True:
            try:
                kernel.autonomous_tick()
            except Exception:
                pass
            time.sleep(max(1.0, kernel.idle_period_s))

    # In lite mode we disable the background autonomy loop by default.
    lite = str(os.environ.get("BUNNY_LITE", "1")).strip() not in ("0", "false", "False")
    if not lite:
        t = threading.Thread(target=_idle_loop, name="bunny-idle", daemon=True)
        t.start()

    srv = BunnyHTTPServer((host, port), Handler, kernel)
    print(f"Bunny UI listening on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

