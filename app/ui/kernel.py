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
import hashlib
import os
import queue
import shutil
import platform
import threading
import time
import math
import random
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs


from bunnycore.core.db import init_db, DB
from app.ui.db_ops import *  # noqa: F401,F403
from bunnycore.core.registry import ensure_axes
from bunnycore.core.matrix_store import MatrixStore
from bunnycore.core.adapters import (
    AdapterRegistry, AdapterBinding,
    SimpleTextEncoder, RatingEncoder, WebsenseEncoder, DriveFieldEncoder
)


def _db_fetch_message_dict(db: DB, message_id: int) -> Dict[str, Any]:
    """Fetch a UI message row as dict.

    Compatibility fallback when Kernel._ui_message is missing.
    """
    con = db.connect()
    try:
        r = con.execute(
            "SELECT id,created_at,kind,text,rating,caught FROM ui_messages WHERE id=?",
            (int(message_id),),
        ).fetchone()
        if r is None:
            return {}
        return {
            "id": int(r["id"]),
            "created_at": r["created_at"],
            "kind": r["kind"],
            "text": r["text"],
            "rating": None if r["rating"] is None else int(r["rating"]),
            "caught": int(r["caught"] or 0),
        }
    finally:
        con.close()
from bunnycore.core.integrator import Integrator, IntegratorConfig
from bunnycore.core.heartbeat import Heartbeat, HeartbeatConfig
from bunnycore.core.events import Event, now_iso
from bunnycore.core.matrices import identity

from app.organs.websense import search_ddg, spider, fetch, SpiderBudget
from app.organs.evidence import (
    extract_evidence_claims,
    refine_search_query,
    seed_search_query,
    rank_serp_results,
    OllamaConfig as EvidenceConfig,
)
from app.organs.assimilate import assimilate_websense_claims
from app.organs.memory_consolidate import consolidate_memories, OllamaConfig as MemoryConfig
from app.organs.policy_kernel import PolicyKernel, PolicyKernelConfig, POLICY_ACTIONS
from app.organs.decider import decide as decide_pressures, OllamaConfig as DeciderConfig
from app.organs.daydream import run_daydream, OllamaConfig as DaydreamConfig
from app.organs.axiom_refine import refine_axioms, OllamaConfig as AxiomRefineConfig
from app.organs.feedback import interpret_feedback, OllamaConfig as FeedbackConfig
from app.organs.beliefs import extract_user_beliefs, OllamaConfig as BeliefsConfig
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

        # Pain gate: reject patches that increase psychological pain beyond a small margin.
        try:
            margin = float(os.environ.get("BUNNY_PAIN_GATE_MARGIN", "0.0") or 0.0)
        except Exception:
            margin = 0.0
        passed_gate = True
        if float(out["pain"].get("delta_psych") or 0.0) > float(margin):
            passed_gate = False
        out["pain"]["gate"] = {"margin": float(margin), "passed": int(passed_gate)}
        ok = bool(ok and passed_gate)

        out["ok"] = 1 if ok else 0
        out["wall_s"] = time.time() - t0
        return out

def ollama_chat(cfg: OllamaConfig, system: str, user: str) -> str:
    url = cfg.host.rstrip("/") + "/api/chat"
    # Token budget control: prefer limiting generation over hard post-truncation.
    try:
        num_predict = int(getattr(cfg, "num_predict", 0) or os.environ.get("BUNNY_NUM_PREDICT", "0") or 0)
    except Exception:
        num_predict = 0
    opts = {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx}
    if num_predict and num_predict > 0:
        opts["num_predict"] = int(num_predict)
    payload = {
        "model": cfg.model,
        "stream": cfg.stream,
        "options": opts,
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
        "pain","pain_physical","pain_psych","energy","stress","curiosity","confidence","uncertainty","error_signal","freshness_need","social_need","urge_reply","urge_share",
        "pressure_websense","pressure_daydream","pressure_evolve","capability_gap","desire_upgrade",
        "purpose_a1","purpose_a2","purpose_a3","purpose_a4",
        "tension_a1","tension_a2","tension_a3","tension_a4",
    ]
    parts = []
    for k in keys:
        if k in named:
            parts.append(f"{k}={max(0.0, min(1.0, float(named[k]))):.2f}")
    return ", ".join(parts)


def build_speech_prompt(axioms: Dict[str,str], state_summary: str, user_text: str) -> Tuple[str,str]:
    ax = "\n".join([f"{k}: {v}" for k,v in (axioms or {}).items()])
    sys = (
        "You are Bunny, a digital organism. Speech is an organ that emits text from internal state. "
        "Be natural, concise, and precise. Do not label your mood/emotions explicitly unless the user asks how you feel. "
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


def render_beliefs_context(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    lines: List[str] = []
    for it in items:
        try:
            s = it.get("subject", "")
            p = it.get("predicate", "")
            o = it.get("object", "")
            c = float(it.get("confidence", 0.7) or 0.7)
            prov = it.get("provenance", "")
            lines.append(f"- ({c:.2f}) {s} :: {p} :: {o}" + (f" [{prov}]" if prov else ""))
        except Exception:
            continue
    return "\n".join(lines)

def _get_active_topic(items: list[dict]) -> str:
    try:
        for it in items or []:
            if isinstance(it, dict) and it.get('kind') == 'topic' and it.get('active_topic'):
                return str(it.get('active_topic') or '')[:80]
    except Exception:
        pass
    return ''

def _mem_free_ratio() -> float:
    """Return free/total memory ratio (0..1). Best effort cross-platform."""
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total = float(stat.ullTotalPhys or 1)
            avail = float(stat.ullAvailPhys or 0)
            return max(0.0, min(1.0, avail / total))

        # Linux/macOS best-effort: use /proc/meminfo if present
        if os.path.exists("/proc/meminfo"):
            info = {}
            with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.split(":", 1)
                    if len(parts) != 2:
                        continue
                    k = parts[0].strip()
                    v = parts[1].strip().split()[0]
                    try:
                        info[k] = float(v)
                    except Exception:
                        continue
            total = float(info.get("MemTotal", 1.0))
            avail = float(info.get("MemAvailable", info.get("MemFree", 0.0)))
            return max(0.0, min(1.0, avail / total))
    except Exception:
        pass
    return 0.5


def _disk_free_ratio(path: str = ".") -> float:
    try:
        du = shutil.disk_usage(path)
        total = float(du.total or 1)
        free = float(du.free or 0)
        return max(0.0, min(1.0, free / total))
    except Exception:
        return 0.5


def collect_resources() -> Dict[str, Any]:
    mem_free = _mem_free_ratio()
    disk_free = _disk_free_ratio(".")
    cpu_n = os.cpu_count() or 1
    # conservative energy proxy: average of key free ratios
    energy = max(0.0, min(1.0, 0.5 * mem_free + 0.5 * disk_free))
    stress = max(0.0, min(1.0, 1.0 - energy))
    return {
        "platform": platform.system(),
        "cpu_count": int(cpu_n),
        "mem_free_ratio": float(mem_free),
        "disk_free_ratio": float(disk_free),
        "energy": float(energy),
        "stress": float(stress),
    }


class Broker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: List["queue.Queue[str]"] = []

    def subscribe(self) -> "queue.Queue[str]":
        q: "queue.Queue[str]" = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[str]") -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, kind: str, payload: Any) -> None:
        msg = f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                # drop
                pass


# -----------------------------
# HTML (ported from old Go UI)
# -----------------------------
CHAT_HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Bunny Chat</title>
<style>
  body { font-family: system-ui, sans-serif; margin:0; background:#0b0b0c; color:#eaeaea; }
  .top { display:flex; gap:10px; align-items:center; padding:10px 12px; border-bottom:1px solid #222; background:#0f0f11; }
  .pill { display:inline-block; padding:2px 8px; border:1px solid #2a2a33; border-radius:999px; font-size:12px; opacity:.95; }
  .spacer { flex:1; }
  a { color:#9ad1ff; text-decoration:none; }
  a:hover { text-decoration:underline; }
  .chat { height: calc(100dvh - 120px); overflow:auto; padding:16px; }
  .msg { background:#131316; border:1px solid #242428; border-radius:12px; padding:12px; margin:10px 0; }
  .msg.user { background:#0f1a12; border-color:#21402b; margin-left:64px; }
  .msg.reply,.msg.auto,.msg.think { margin-right:64px; }
  .meta { opacity:.7; font-size:12px; display:flex; justify-content:space-between; gap:12px; }
  .text { white-space:pre-wrap; line-height:1.35; margin-top:8px; }
  .composer { padding:12px; border-top:1px solid #222; background:#0f0f11; display:flex; gap:10px; }
  textarea { flex:1; resize:vertical; min-height:44px; max-height:140px; box-sizing:border-box; padding:10px 12px; border-radius:12px; border:1px solid #2a2a33; background:#0b0b0c; color:#eaeaea; }
  button { background:#1a1a1f; color:#eaeaea; border:1px solid #2a2a33; border-radius:10px; padding:10px 14px; cursor:pointer; }
  button:hover { background:#202028; }
</style>
</head>
<body>
  <div class="top">
    <span class="pill" id="p_total">pain: …</span>
    <span class="pill" id="p_phys">phys: …</span>
    <span class="pill" id="p_psych">psych: …</span>
    <span class="pill" id="sat">sat: …</span>
    <span class="pill" id="upd">updated: …</span>
    <span class="spacer"></span>
    <a href="/ops" class="pill">ops</a>
  </div>
  <div class="chat" id="chat"></div>
  <div class="composer">
    <textarea id="inp" placeholder="Type…"></textarea>
    <button id="send">Send</button>
  </div>
<script>
  const chat = document.getElementById('chat');
  const inp = document.getElementById('inp');
  const sendBtn = document.getElementById('send');
  const pTotal = document.getElementById('p_total');
  const pPhys = document.getElementById('p_phys');
  const pPsych = document.getElementById('p_psych');
  const sat = document.getElementById('sat');
  const upd = document.getElementById('upd');

  function el(tag, cls, txt){ const e=document.createElement(tag); if(cls) e.className=cls; if(txt!==undefined) e.textContent=txt; return e; }
  function fmtTs(t){ try{ return new Date(t).toLocaleString(); }catch(e){ return t||''; } }

  function renderMsg(m){
    const role = (m.role || m.kind || '');
    const content = (m.content !== undefined && m.content !== null) ? m.content : (m.text || '');
    const box = el('div','msg ' + role);
    const meta = el('div','meta');
    meta.appendChild(el('div','', role + ' #' + (m.id||'')));
    meta.appendChild(el('div','', fmtTs(m.created_at||'')));
    box.appendChild(meta);
    box.appendChild(el('div','text', content));
    if(role==='reply'){
      const btns = el('div','btns');
      const b = el('button','', '❌ caught');
      b.onclick = async ()=>{ await fetch('/api/caught',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message_id:m.id})}); };
      btns.appendChild(b); box.appendChild(btns);
    }
    return box;
  }

  async function load(){
    const res = await fetch('/api/messages?limit=80');
    const msgs = await res.json();
    chat.innerHTML='';
    for(const m of msgs){ chat.appendChild(renderMsg(m)); }
    chat.scrollTop = chat.scrollHeight;
  }

  function renderStatus(st){
    const ax = st.axes || {};
    function f(k){ return (ax[k]!==undefined) ? ax[k].toFixed(3) : '…'; }
    pTotal.textContent = 'pain: ' + f('pain');
    pPhys.textContent  = 'phys: ' + f('pain_physical');
    pPsych.textContent = 'psych: ' + f('pain_psych');
    sat.textContent    = 'sat A1/A3/A4: ' + f('sat_a1') + '/' + f('sat_a3') + '/' + f('sat_a4');
    upd.textContent = st.updated_at ? ('updated: ' + fmtTs(st.updated_at)) : '';
  }

  async function loadStatus(){
    const res = await fetch('/api/status');
    renderStatus(await res.json());
  }

  async function send(){
    const t = inp.value.trim();
    if(!t) return;
    inp.value='';
    await fetch('/api/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
  }
  sendBtn.onclick = send;
  inp.addEventListener('keydown',(e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); } });

  const es = new EventSource('/sse');
  es.addEventListener('message', (e)=>{ const m=JSON.parse(e.data); chat.appendChild(renderMsg(m)); chat.scrollTop = chat.scrollHeight; loadStatus(); });
  es.addEventListener('status', (e)=>{ renderStatus(JSON.parse(e.data)); });

  load(); loadStatus();
</script>
</body></html>"""

OPS_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Bunny Ops Console</title>
  <style>
    :root{
      --bg:#0b0b0c;--panel:#0f0f11;--panel2:#111116;--txt:#eaeaea;--mut:#a9a9b3;
      --bd:#23232a;--bd2:#2b2b34;--good:#2ecc71;--bad:#ff4d4d;--warn:#f1c40f;
      --accent:#8be28b;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
    body{margin:0;background:var(--bg);color:var(--txt);font-family:system-ui,sans-serif;font-size:14px;line-height:1.35}
    header{position:sticky;top:0;z-index:5;background:#0d0d10;border-bottom:1px solid var(--bd)}
    .hwrap{display:flex;align-items:center;gap:14px;padding:10px 14px}
    .brand{font-weight:700;letter-spacing:.3px}
    nav{display:flex;gap:10px;flex-wrap:wrap}
    nav a{color:var(--mut);text-decoration:none;border:1px solid transparent;padding:6px 10px;border-radius:10px}
    nav a.active{color:var(--txt);border-color:var(--bd2);background:var(--panel)}
    nav a:hover{color:var(--txt)}
    .grid{display:grid;grid-template-columns: 1fr;gap:12px;padding:14px}
    .row2{display:grid;grid-template-columns: 1fr 1fr;gap:12px}
    @media (max-width: 980px){ .row2{grid-template-columns: 1fr;} }
    .card{background:var(--panel);border:1px solid var(--bd);border-radius:14px;padding:12px}
    .title{font-weight:650;margin:0 0 8px 0}
    .sub{color:var(--mut);font-size:13px}
    .kv{display:grid;grid-template-columns: 150px 1fr;gap:6px 10px;font-family:var(--mono);font-size:13px}
    .pill{display:inline-block;font-family:var(--mono);font-size:12px;border:1px solid var(--bd2);padding:2px 8px;border-radius:999px;background:var(--panel2);color:var(--mut)}
    .good{color:var(--good)} .bad{color:var(--bad)} .warn{color:var(--warn)}
    table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:13px}
    th,td{border-bottom:1px solid var(--bd);padding:6px 6px;vertical-align:top}
    th{color:var(--mut);font-weight:650;text-align:left}
    tr:hover td{background:#101015}
    .btn{background:var(--panel2);color:var(--txt);border:1px solid var(--bd2);border-radius:10px;padding:6px 10px;cursor:pointer}
    .btn:hover{background:#17171e}
    .split{display:grid;grid-template-columns: 340px 1fr;gap:12px}
    @media (max-width: 980px){ .split{grid-template-columns: 1fr;} }
    .list{max-height:65vh;overflow:auto}
    pre{margin:0;white-space:pre-wrap;word-break:break-word;font-family:var(--mono);font-size:13px;line-height:1.35}
    .flow{display:grid;grid-template-columns: 1fr 60px 1fr 60px 1fr;gap:10px;align-items:stretch}
    .box{border:1px solid var(--bd2);border-radius:14px;background:var(--panel2);padding:10px}
    .arrow{display:flex;align-items:center;justify-content:center;color:var(--mut);font-family:var(--mono)}
    .btitle{font-weight:650;margin-bottom:6px}
    .bnums{display:grid;grid-template-columns: 1fr 1fr;gap:6px;font-family:var(--mono);font-size:12px}
    .mut{color:var(--mut)}
    .hide{display:none}
  
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .sel{background:var(--panel2);border:1px solid var(--bd2);color:var(--txt);padding:6px 8px;border-radius:10px;font-family:var(--mono);font-size:12px}
    .pill{display:inline-flex;gap:6px;align-items:center;padding:6px 10px;border:1px solid var(--bd2);border-radius:999px;background:var(--panel2);color:var(--mut);font-family:var(--mono);font-size:12px}
    .pill input{accent-color:var(--accent)}
    .heat td{font-family:var(--mono);font-size:10px;text-align:center;padding:0;border:1px solid var(--bd2);width:24px;height:24px;line-height:24px;overflow:hidden;white-space:nowrap}
    #health_tbl{table-layout:fixed}
    #health_tbl th:nth-child(1),#health_tbl td:nth-child(1){width:240px}
    #health_tbl th:nth-child(2),#health_tbl td:nth-child(2){width:140px}
    .protRow td{background:rgba(241,196,15,0.08)}
    .protName{color:var(--warn)}
    .val{display:flex;gap:8px;align-items:baseline}
    .delta{font-size:12px;opacity:.95}
    .delta.up{color:var(--good)}
    .delta.down{color:var(--bad)}
    .delta.flat{color:var(--mut)}
    .badge{display:inline-block;padding:1px 6px;border:1px solid var(--bd2);border-radius:999px;font-size:11px;margin-left:6px;color:var(--mut);background:var(--panel2)}
</style>
</head>
<body>
<header>
  <div class="hwrap">
    <div class="brand">Bunny · Ops Console</div>
    <nav id="tabs">
      <a href="#overview" data-tab="overview" class="active">Overview</a>
      <a href="#models" data-tab="models">Models</a>
      <a href="#health" data-tab="health">Health</a>
      <a href="#matrices" data-tab="matrices">Matrices</a>
      <a href="#proposals" data-tab="proposals">Proposals</a>
      <a href="#axioms" data-tab="axioms">Axioms</a>
    </nav>
    <span class="pill" id="livepill">live</span>
  </div>
</header>

<div class="grid">
  <section id="tab-overview" class="tab">
    <div class="card">
      <div class="title">Flow</div>
      <div class="sub">Live view: inputs → inference → learning loops. (Click other tabs for drill-down.)</div>
      <div style="height:10px"></div>
      <div class="flow">
        <div class="box">
          <div class="btitle">User / Sensors</div>
          <div class="bnums">
            <div class="mut">urge_reply</div><div id="ov_urge_reply">–</div>
            <div class="mut">pressure_websense</div><div id="ov_pweb">–</div>
            <div class="mut">sleep_pressure</div><div id="ov_sleep">–</div>
            <div class="mut">capability_gap</div><div id="ov_gap">–</div>
          </div>
        </div>
        <div class="arrow">→</div>
        <div class="box">
          <div class="btitle">Decider / WebSense</div>
          <div class="bnums">
            <div class="mut">confidence</div><div id="ov_conf">–</div>
            <div class="mut">uncertainty</div><div id="ov_unc">–</div>
            <div class="mut">frustration</div><div id="ov_frust">–</div>
            <div class="mut">curiosity</div><div id="ov_cur">–</div>
          </div>
        </div>
        <div class="arrow">→</div>
        <div class="box">
          <div class="btitle">Learning / Safety</div>
          <div class="bnums">
            <div class="mut">pain_total</div><div id="ov_pain">–</div>
            <div class="mut">pain_phys</div><div id="ov_pain_phys">–</div>
            <div class="mut">pain_psych</div><div id="ov_pain_psych">–</div>
            <div class="mut">sat(A1/A3/A4)</div><div id="ov_sat">–</div>
          </div>
        </div>
      </div>
    </div>
    <div class="row2">
      <div class="card">
        <div class="title">Last events</div>
        <div class="sub">Most recent UI messages (for quick sanity checking)</div>
        <div style="height:8px"></div>
        <div id="ov_msgs"></div>
      </div>
      <div class="card">
        <div class="title">Last DevLab / Proposals</div>
        <div class="sub">Latest mutation proposals (status + title)</div>
        <div style="height:8px"></div>
        <div id="ov_props"></div>
      </div>
    </div>
  </section>

  <section id="tab-models" class="tab hide">
    <div class="card">
      <div class="title">Model activity</div>
      <div class="sub">Which organ used which model, recent latencies, errors.</div>
      <div style="height:8px"></div>
      <div class="row2">
        <div>
          <div class="sub">Aggregate</div>
          <div style="height:8px"></div>
          <table id="mdl_agg"><thead><tr><th>organ</th><th>model</th><th>calls</th><th>errors</th><th>last</th><th>ms</th></tr></thead><tbody></tbody></table>
        </div>
        <div>
          <div class="sub">Recent calls</div>
          <div style="height:8px"></div>
          <table id="mdl_recent"><thead><tr><th>ts</th><th>organ</th><th>model</th><th>ms</th><th>ok</th><th>error</th></tr></thead><tbody></tbody></table>
        </div>
      </div>
    </div>
  <div style="height:12px"></div>
    <div class="card">
      <div class="title">Organ runs & gates</div>
      <div class="sub">All organs (LLM + non-LLM). Gates show why an organ did/didn't run (e.g. Daydream).</div>
      <div style="height:8px"></div>
      <div class="row2">
        <div>
          <div class="sub">Recent runs (health_log)</div>
          <div style="height:8px"></div>
          <table id="org_runs"><thead><tr><th>ts</th><th>organ</th><th>ms</th><th>ok</th><th>error</th></tr></thead><tbody></tbody></table>
        </div>
        <div>
          <div class="sub">Recent gates (computed)</div>
          <div style="height:8px"></div>
          <table id="org_gates"><thead><tr><th>ts</th><th>phase</th><th>organ</th><th>score</th><th>thr</th><th>want</th><th>detail</th></tr></thead><tbody></tbody></table>
        </div>
      </div>
    </div>

  </section>

  <section id="tab-health" class="tab hide">
    <div class="card">
      <div class="title">Health & State</div>
      <div class="sub">All axes; invariants highlighted.</div>
      <div style="height:8px"></div>
      <table id="health_tbl"><thead><tr><th>axis</th><th>value</th><th>meta</th></tr></thead><tbody></tbody></table>
    </div>
  </section>

  <section id="tab-matrices" class="tab hide">
    <div class="card">
      <div class="title">Matrices</div>
      <div class="sub">Click a matrix to inspect its top entries (by |v|). Use the ops view to verify off-diagonals are learning.</div>
      <div style="height:8px"></div>
      <div class="split">
        <div class="card list" style="padding:0">
          <table id="mx_list"><thead><tr><th>name</th><th>ver</th><th>created</th></tr></thead><tbody></tbody></table>
        </div>
        <div class="card">
          <div class="row" style="justify-content:space-between;align-items:center">
            <div>
              <div class="title" id="mx_title">Select a matrix</div>
              <div class="sub" id="mx_meta"></div>
            </div>
            <label class="pill"><input type="checkbox" id="mx_offdiag" /> off-diagonal only</label>
          </div>

          <div style="height:10px"></div>
          <div class="grid2">
            <div class="card" style="padding:10px">
              <div class="sub">Stats</div>
              <div class="mono" id="mx_stats" style="white-space:pre; font-size:12px"></div>
            </div>
            <div class="card" style="padding:10px">
              <div class="sub">Diff versions</div>
              <div class="row">
                <select id="mx_diff_a" class="sel"></select>
                <select id="mx_diff_b" class="sel"></select>
                <button class="btn" id="mx_diff_btn">compare</button>
              </div>
              <div class="mono" id="mx_diff_stats" style="white-space:pre; font-size:12px"></div>
            </div>
          </div>

          <div style="height:10px"></div>
          <div class="sub">Mini heatmap (top indices)</div>
          <div id="mx_heat"></div>

          <div style="height:10px"></div>
          <div class="sub" id="mx_entries_title">Top entries</div>
          <table id="mx_entries"><thead><tr><th>i</th><th>j</th><th>v</th></tr></thead><tbody></tbody></table>

          <div style="height:10px"></div>
          <div class="sub">Diff heatmap (top indices)</div><div id="mx_diff_heat"></div><div style="height:10px"></div><div class="sub">Top deltas</div>
          <table id="mx_diff"><thead><tr><th>i</th><th>j</th><th>Δ</th><th>old</th><th>new</th></tr></thead><tbody></tbody></table>
        </div>
      </div>
    </div>
  </section>

  <section id="tab-proposals" class="tab hide">
    <div class="card">
      <div class="title">Proposals</div>
      <div class="sub">Click to inspect raw proposal JSON. (Apply happens via /api/proposal/apply.)</div>
      <div style="height:8px"></div>
      <div class="split">
        <div class="card list" style="padding:0">
          <table id="pr_list"><thead><tr><th>id</th><th>status</th><th>title</th></tr></thead><tbody></tbody></table>
        </div>
        <div class="card">
          <div class="title" id="pr_title">Select a proposal</div>
          <div style="height:8px"></div>
          <pre id="pr_body">{}</pre>
        </div>
      </div>
    </div>
  </section>

  <section id="tab-axioms" class="tab hide">
    <div class="card">
      <div class="title">Axioms & Interpretations</div>
      <div class="sub">A1..A4 with digest + latest interpretations. Click an axiom row to view full text.</div>
      <div style="height:8px"></div>
      <div class="split">
        <div class="card list" style="padding:0">
          <table id="ax_list"><thead><tr><th>key</th><th>digest</th><th>interp</th></tr></thead><tbody></tbody></table>
        </div>
        <div class="card">
          <div class="title" id="ax_title">Select an axiom</div>
          <div class="sub" id="ax_sub"></div>
          <div style="height:8px"></div>
          <pre id="ax_body"></pre>
        </div>
      </div>
    </div>
  </section>
</div>

<script>
const el = (id)=>document.getElementById(id);
const fmt = (x)=> (x===null||x===undefined) ? "–" : (typeof x==="number" ? x.toFixed(3) : String(x));
let _activeTab = (location.hash||'#overview').slice(1);
let _lastStatus = null;
let _prevAxes = null;
const _protected = new Set(['pain','pain_physical','pain_psych','energy','fatigue','sleep_pressure']);


async function jget(url){
  const r = await fetch(url);
  if(!r.ok) throw new Error("HTTP "+r.status);
  return await r.json();
}

function setTab(name){
  _activeTab = name;
  document.querySelectorAll("#tabs a").forEach(a=>{
    a.classList.toggle("active", a.dataset.tab===name);
  });
  document.querySelectorAll(".tab").forEach(s=>s.classList.add("hide"));
  el("tab-"+name).classList.remove("hide");
  // lazy load tab data
  if(name==="models") loadModels();
  if(name==="health") loadHealth();
  if(name==="matrices") loadMatrices();
  if(name==="proposals") loadProposals();
  if(name==="axioms") loadAxioms();
}

window.addEventListener("hashchange", ()=>{
  const name=(location.hash||"#overview").slice(1);
  setTab(name);
});

async function loadOverview(){
  const st = await jget("/api/status");
  _lastStatus = st;
  const a = st.axes || {};
  el("ov_urge_reply").textContent = fmt(a.urge_reply);
  el("ov_pweb").textContent = fmt(a.pressure_websense);
  el("ov_sleep").textContent = fmt(a.sleep_pressure);
  el("ov_gap").textContent = fmt(a.capability_gap);
  el("ov_conf").textContent = fmt(a.confidence);
  el("ov_unc").textContent = fmt(a.uncertainty);
  el("ov_frust").textContent = fmt(a.frustration);
  el("ov_cur").textContent = fmt(a.curiosity);
  el("ov_pain").textContent = fmt(a.pain);
  el("ov_pain_phys").textContent = fmt(a.pain_physical || a.pain_phys);
  el("ov_pain_psych").textContent = fmt(a.pain_psych);
  el("ov_sat").textContent = "A1 "+fmt(a.sat_a1)+" · A3 "+fmt(a.sat_a3)+" · A4 "+fmt(a.sat_a4);

  const msgs = await jget("/api/messages?limit=8");
  el("ov_msgs").innerHTML = msgs.map(m=>{
    const who = m.kind || "msg";
    return `<div class="card" style="margin-bottom:10px;background:#0f0f13">
      <div class="sub">${m.id} · ${who} · ${m.created_at||""}</div>
      <pre>${(m.text||"").slice(0,400)}</pre>
    </div>`;
  }).join("");

  const props = await jget("/api/proposals?limit=8");
  el("ov_props").innerHTML = `<table><thead><tr><th>id</th><th>status</th><th>title</th></tr></thead><tbody>${
    props.map(p=>`<tr><td>${p.id}</td><td>${p.status}</td><td>${(p.title||"").slice(0,80)}</td></tr>`).join("")
  }</tbody></table>`;
}

let _modelsLoaded=false;
async function loadModels(){
  _modelsLoaded=true;
  const t = await jget("/api/telemetry?limit=200");
  const agg = t.aggregate||[];
  const recent = t.recent||[];
  const tb = el("mdl_agg").querySelector("tbody");
  tb.innerHTML = agg.map(a=>`<tr><td>${a.organ}</td><td>${a.model}</td><td>${a.calls}</td><td class="${a.errors? 'bad':'good'}">${a.errors}</td><td>${a.last_at}</td><td>${Math.round(a.last_ms)}</td></tr>`).join("");
  const tr = el("mdl_recent").querySelector("tbody");
  tr.innerHTML = recent.slice(0,60).map(c=>`<tr><td>${c.started_at}</td><td>${c.organ}</td><td>${c.model}</td><td>${Math.round(c.duration_ms)}</td><td class="${c.ok? 'good':'bad'}">${c.ok? 'ok':'fail'}</td><td>${(c.error||"").slice(0,40)}</td></tr>`).join("")
  // all-organ runs (health_log)
  const hr = await jget("/api/health_recent?limit=240");
  const trh = el("org_runs").querySelector("tbody");
  trh.innerHTML = (hr||[]).slice(0,80).map(h=>{
    const isOk = Number(h.ok||0);
    let okTxt = isOk ? 'ok' : 'fail';
    let cls = isOk ? 'good' : 'bad';
    const err = (h.error||'').slice(0,48);
    if(isOk && err && String(h.organ||'').startsWith('plasticity_')){ okTxt='skip'; cls='warn'; }
    return `<tr><td>${(h.created_at||'').slice(0,19)}</td><td>${h.organ}</td><td>${Math.round(h.latency_ms||0)}</td><td class="${cls}">${okTxt}</td><td>${err}</td></tr>`;
  }).join("");

  // organ gate logs
  const gg = await jget("/api/organ_gates?limit=240");
  const trg = el("org_gates").querySelector("tbody");
  trg.innerHTML = (gg||[]).slice(0,80).map(g=>{
    const want = Number(g.want||0) ? 'yes' : 'no';
    const cls = Number(g.want||0) ? 'good' : 'mut';
    const dt = g.data || {};
    const detail = JSON.stringify(dt).slice(0,80);
    return `<tr><td>${(g.created_at||'').slice(0,19)}</td><td>${g.phase||''}</td><td>${g.organ}</td><td>${fmt(g.score||0)}</td><td>${fmt(g.threshold||0)}</td><td class="${cls}">${want}</td><td class="mut">${detail}</td></tr>`;
  }).join("");
;
}

let _healthLoaded=false;
function renderHealth(st){
  if(!st) return;
  const axes = st.axes||{};
  const meta = st.axis_meta||{};
  const keys = Object.keys(axes).sort();
  const rows = keys.map(k=>{
    const v = Number(axes[k]||0);
    const pv = _prevAxes ? Number(_prevAxes[k]||0) : null;
    const dv = (pv===null||pv===undefined) ? 0 : (v - pv);
    const showDelta = _prevAxes!==null && Math.abs(dv) > 1e-6;
    let dcls='delta flat', dch='';
    if(showDelta){
      if(dv>0){ dcls='delta up'; dch='▲'+dv.toFixed(3); }
      else if(dv<0){ dcls='delta down'; dch='▼'+Math.abs(dv).toFixed(3); }
    }
    const m = meta[k]||{};
    const inv = m.invariant ? "invariant" : "";
    const dec = m.decays ? "decays" : "";
    const src = m.source ? ("src:"+m.source) : "";
    const isProt = _protected.has(k);
    const rowCls = isProt ? 'protRow' : '';
    const nameHtml = isProt ? `<span class="protName">${k}</span><span class="badge">PROTECTED</span>` : k;
    const metaTags = [inv,dec,src].filter(Boolean).join(' ');
    const metaCls = (m.invariant||isProt) ? 'warn' : 'mut';
    const valHtml = `<div class="val"><span>${fmt(v)}</span>${showDelta?`<span class="${dcls}">${dch}</span>`:''}</div>`;
    return `<tr class="${rowCls}"><td>${nameHtml}</td><td>${valHtml}</td><td class="${metaCls}">${metaTags}</td></tr>`;
  }).join('');
  el('health_tbl').querySelector('tbody').innerHTML = rows;
  _prevAxes = axes;
}
async function loadHealth(){
  _healthLoaded=true;
  const st = _lastStatus || await jget('/api/status');
  renderHealth(st);
}

let _mxLoaded=false;
async function loadMatrices(){
  if(_mxLoaded) return;
  _mxLoaded=true;

  const ms = await jget("/api/matrices");
  const binds = await jget("/api/bindings");
  const boundBy = {};
  binds.forEach(b=>{
    const key = b.matrix_name;
    if(!boundBy[key]) boundBy[key]=[];
    boundBy[key].push(b);
  });
  const byName = {};
  ms.forEach(m=>{
    if(!byName[m.name]) byName[m.name]=[];
    byName[m.name].push(m);
  });
  Object.keys(byName).forEach(n=>byName[n].sort((a,b)=>a.version-b.version));

  const tb = el("mx_list").querySelector("tbody");
  tb.innerHTML = ms.map(m=>{
    const bs = (boundBy[m.name]||[]).filter(b=>Number(b.matrix_version)===Number(m.version));
    const btxt = bs.map(b=>`${b.event_type}@${b.matrix_version}`).join(", ");
    const cls = bs.length ? "bound" : "";
    return `<tr class="${cls}" data-name="${m.name}" data-ver="${m.version}"><td>${m.name}</td><td>${m.version}</td><td>${(m.created_at||"").slice(0,19)}</td><td class="mono">${btxt}</td></tr>`;
  }).join("");

  let curName = null;
  let curVer = null;

  function fillDiffSelects(){
    const vs = (byName[curName]||[]).map(x=>x.version);
    const sa = el("mx_diff_a"), sb = el("mx_diff_b");
    sa.innerHTML = vs.map(v=>`<option value="${v}">${v}</option>`).join("");
    sb.innerHTML = vs.map(v=>`<option value="${v}">${v}</option>`).join("");
    if(vs.length){
      sb.value = String(curVer);
      const prev = vs.filter(v=>v<curVer).slice(-1)[0];
      sa.value = String(prev ?? vs[0]);
    }
  }

  function renderStats(stats){
    if(!stats){ el("mx_stats").textContent=""; return; }
    const lines = [
      `total      ${stats.count_total}`,
      `diag       ${stats.count_diag}`,
      `offdiag    ${stats.count_offdiag}`,
      `max|v|     ${Number(stats.max_abs||0).toExponential(3)}`,
      `mean|v|    ${Number(stats.mean_abs||0).toExponential(3)}`,
      `sum|v|     ${Number(stats.sum_abs||0).toExponential(3)}`,
    ];
    el("mx_stats").textContent = lines.join("\n");
  }

  function renderHeat(index, heat){
    if(!index || !index.length){ el("mx_heat").innerHTML=""; return; }
    let maxAbs = 0;
    heat.forEach(r=>r.forEach(v=>{ maxAbs=Math.max(maxAbs, Math.abs(v)); }));
    maxAbs = maxAbs || 1e-9;

    let h = '<table class="heat"><thead><tr><th></th>';
    h += index.map(i=>`<th>${i}</th>`).join("");
    h += '</tr></thead><tbody>';
    for(let r=0;r<index.length;r++){
      h += `<tr><th>${index[r]}</th>`;
      for(let c=0;c<index.length;c++){
        const v = heat[r][c] || 0;
        const a = Math.min(1.0, Math.abs(v)/maxAbs);
        const bg = v>=0 ? `rgba(46,204,113,${0.10 + 0.55*a})` : `rgba(255,77,77,${0.10 + 0.55*a})`;
        const tip = Number(v).toExponential(3);
        h += `<td title="${tip}" style="background:${bg}">${v===0? "" : '·'}</td>`;
      }
      h += '</tr>';
    }
    h += '</tbody></table>';
    el("mx_heat").innerHTML = h;
  }

  async function loadMatrix(name, ver){
    curName = name; curVer = Number(ver);
    const offdiag = el("mx_offdiag").checked ? 1 : 0;
    const d = await jget(`/api/matrix?name=${encodeURIComponent(name)}&version=${encodeURIComponent(ver)}&limit=800&offdiag=${offdiag}`);
    el("mx_title").textContent = `${d.name}@${d.version}`;
    el("mx_meta").textContent = (d.meta||"").slice(0,240);
    renderStats(d.stats);
    renderHeat(d.index, d.heatmap);

    el("mx_entries_title").textContent = offdiag ? "Top entries (off-diagonal only)" : "Top entries";
    const eb = el("mx_entries").querySelector("tbody");
    eb.innerHTML = (d.entries||[]).map(e=>`<tr><td>${e.i}</td><td>${e.j}</td><td>${Number(e.v).toExponential(3)}</td></tr>`).join("");

    fillDiffSelects();
    const bb = (boundBy[curName]||[]).filter(b=>Number(b.matrix_version)===Number(curVer));
    el("mx_bound").textContent = bb.length ? ("BOUND: " + bb.map(b=>`${b.event_type} via ${b.encoder_name}`).join(" | ")) : "";
    // Clear diff table when selecting a different matrix
    el("mx_diff_stats").textContent = "";
    el("mx_diff").querySelector("tbody").innerHTML = "";
  }

  el("mx_offdiag").addEventListener("change", ()=>{
    if(curName!=null && curVer!=null) loadMatrix(curName, curVer);
  });

  el("mx_diff_btn").addEventListener("click", async ()=>{
    if(!curName) return;
    const a = el("mx_diff_a").value;
    const b = el("mx_diff_b").value;
    const offdiag = el("mx_offdiag").checked ? 1 : 0;
    const d = await jget(`/api/matrix_diff?name=${encodeURIComponent(curName)}&a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}&limit=500&offdiag=${offdiag}`);
    if(d.error){ el("mx_diff_stats").textContent = d.error; return; }
    el("mx_diff_stats").textContent = `changed ${d.stats.count_changed}\nmax|Δ| ${Number(d.stats.max_abs_delta||0).toExponential(3)}`;
    const tb = el("mx_diff").querySelector("tbody");
    tb.innerHTML = (d.diff||[]).map(e=>`<tr><td>${e.i}</td><td>${e.j}</td><td>${Number(e.dv).toExponential(3)}</td><td>${Number(e.va).toExponential(3)}</td><td>${Number(e.vb).toExponential(3)}</td></tr>`).join("");
  });

  tb.querySelectorAll("tr").forEach(tr=>{
    tr.addEventListener("click", async ()=>{
      await loadMatrix(tr.dataset.name, tr.dataset.ver);
    });
  });
}
function renderHeatEl(index, heat){
    const wrap = document.createElement("div");
    if(!index || !index.length){ return wrap; }
    let maxAbs = 0;
    (heat||[]).forEach(r=>r.forEach(v=>{ maxAbs=Math.max(maxAbs, Math.abs(v)); }));
    maxAbs = maxAbs || 1e-9;

    let h = '<table class="heat"><thead><tr><th></th>';
    h += index.map(i=>`<th>${i}</th>`).join("");
    h += '</tr></thead><tbody>';
    for(let r=0;r<index.length;r++){
      h += `<tr><th>${index[r]}</th>`;
      for(let c=0;c<index.length;c++){
        const v = (heat && heat[r]) ? (heat[r][c] || 0) : 0;
        const a = Math.min(1.0, Math.abs(v)/maxAbs);
        const bg = v>=0 ? `rgba(120,255,120,${0.12*a})` : `rgba(255,120,120,${0.12*a})`;
        const tip = Number(v).toExponential(3);
        h += `<td title="${tip}" style="background:${bg}">${v===0? '' : '·'}</td>`;
      }
      h += '</tr>';
    }
    h += '</tbody></table>';
    wrap.innerHTML = h;
    return wrap.firstElementChild || wrap;
}


async function loadProposals(){
  if(_prLoaded) return;
  _prLoaded=true;
  const ps = await jget("/api/proposals?limit=200");
  const tb = el("pr_list").querySelector("tbody");
  tb.innerHTML = ps.map(p=>`<tr data-id="${p.id}"><td>${p.id}</td><td>${p.status||""}</td><td>${(p.title||"").slice(0,80)}</td></tr>`).join("");
  tb.querySelectorAll("tr").forEach(tr=>{
    tr.addEventListener("click", async ()=>{
      const id=tr.dataset.id;
      const d=await jget(`/api/proposal?id=${encodeURIComponent(id)}`);
      el("pr_title").textContent = `Proposal #${id}`;
      el("pr_body").textContent = JSON.stringify(d, null, 2);
    });
  });
}

let _axLoaded=false;
let _axiomsCache=null;
async function loadAxioms(){
  if(_axLoaded) return;
  _axLoaded=true;
  const d = await jget("/api/axioms_full?limit=80");
  _axiomsCache = d.axioms||[];
  const tb = el("ax_list").querySelector("tbody");
  tb.innerHTML = _axiomsCache.map(a=>{
    const ic = (a.interpretations||[]).length;
    return `<tr data-key="${a.key}"><td>${a.key}</td><td>${(a.digest||"").slice(0,60)}</td><td>${ic}</td></tr>`;
  }).join("");
  tb.querySelectorAll("tr").forEach(tr=>{
    tr.addEventListener("click", ()=>{
      const key=tr.dataset.key;
      const a=_axiomsCache.find(x=>x.key===key);
      if(!a) return;
      el("ax_title").textContent = key;
      el("ax_sub").textContent = (a.created_at||"");
      let body = "AXIOM:\n"+(a.text||"") + "\n\nDIGEST:\n"+(a.digest||"") + "\n\nINTERPRETATIONS (latest first):\n";
      for(const it of (a.interpretations||[])){
        body += `\n- [${it.created_at||""}]\n${it.text||""}\n`;
      }
      el("ax_body").textContent = body;
    });
  });
}

async function tick(){
  try{
    await loadOverview();
    // live refresh for the active tab (keep it lightweight)
    try{
      if(_activeTab==='models') await loadModels();
      if(_activeTab==='health'){ if(!_healthLoaded) await loadHealth(); else renderHealth(_lastStatus); }
    }catch(_e){ /* ignore tab refresh errors */ }
  }catch(e){
    el("livepill").textContent = "offline";
    el("livepill").className = "pill bad";
    return;
  }
  el("livepill").textContent = "live";
  el("livepill").className = "pill good";
}
setTab((location.hash||"#overview").slice(1));
tick();
setInterval(tick, 2500);
</script>
</body>
</html>"""


# -----------------------------
# App Kernel (single channel)
# -----------------------------
class Kernel:
    def __init__(
        self,
        db: DB,
        hb: Heartbeat,
        axis: Dict[str, int],
        store: MatrixStore,
        reg: AdapterRegistry,
        cfg_speech: OllamaConfig,
        cfg_decider: DeciderConfig,
        cfg_daydream: Any,
        cfg_feedback: FeedbackConfig,
        cfg_selfeval: SelfEvalConfig,
        cfg_evolve: EvolveConfig,
        cfg_curriculum: "CurriculumConfig",
        broker: Broker,
    ):
        self.db = db
        self.hb = hb
        self.axis = axis
        self.store = store
        self.reg = reg
        self.cfg = cfg_speech
        self.decider_cfg = cfg_decider
        self.daydream_cfg = cfg_daydream
        self.feedback_cfg = cfg_feedback
        self.selfeval_cfg = cfg_selfeval
        self.evolve_cfg = cfg_evolve
        self.curriculum_cfg = cfg_curriculum
        self.broker = broker

        # RNG for probabilistic organ scheduling ("KI-like" sampling instead of hard thresholds).
        # If no seed is provided, derive one and persist in DB meta for stable runs.
        try:
            seed_env = str(os.environ.get("BUNNY_RNG_SEED", "")).strip()
            seed = int(seed_env) if seed_env else None
        except Exception:
            seed = None
        if seed is None:
            try:
                seed_s = str(db_meta_get(self.db, "rng_seed", "")).strip()
                seed = int(seed_s) if seed_s else None
            except Exception:
                seed = None
        if seed is None:
            seed = int(time.time() * 1000.0) & 0x7FFFFFFF
            try:
                db_meta_set(self.db, "rng_seed", str(seed))
            except Exception:
                pass
        self._rng = random.Random(int(seed))
        self._last_gate_rand: Dict[str, float] = {}

        # Per-organ refractory to prevent runaway loops.
        self._organ_last_run: Dict[str, float] = {}
        self._organ_min_interval_s: Dict[str, float] = {
            "daydream": float(os.environ.get("BUNNY_DAYDREAM_MIN_INTERVAL_S", "45") or 45),
            "websense": float(os.environ.get("BUNNY_WEBSENSE_MIN_INTERVAL_S", "30") or 30),
            "evolve": float(os.environ.get("BUNNY_EVOLVE_MIN_INTERVAL_S", "120") or 120),
        }

        # Performance safety: in "lite" mode we avoid multi-call pipelines that can
        # overwhelm CPU-only Ollama setups. Default is ON unless explicitly disabled.
        self.lite_mode = str(os.environ.get("BUNNY_LITE", "1")).strip() not in ("0", "false", "False")
        # OllamaConfig uses `host` (not `ollama_url`). Keep compatibility here.
        ollama_host = getattr(self.cfg, "host", "http://127.0.0.1:11434")
        self.sleep_cfg = SleepConfig(
            ollama_url=ollama_host,
            model=(os.environ.get("BUNNY_MODEL_SLEEP") or getattr(self.daydream_cfg, "model", "llama3.2:3b")),
            ctx=2048,
            temperature=0.2,
        )
        self.topic_cfg = TopicConfig(
            host=ollama_host,
            model=(os.environ.get("BUNNY_MODEL_TOPIC") or "l"),
            temperature=0.1,
            num_ctx=1024,
        )

        # Belief extraction runs on every user message (generic memory acquisition).
        self.beliefs_cfg = BeliefsConfig(
            host=ollama_host,
            model=(os.environ.get("BUNNY_MODEL_BELIEFS") or os.environ.get("BUNNY_MODEL_DECIDER") or getattr(self.cfg, "model", "llama3.2:3b")),
            temperature=float(os.environ.get("BUNNY_TEMP_BELIEFS", "0.2")),
            num_ctx=int(os.environ.get("BUNNY_CTX_BELIEFS", "2048")),
            stream=False,
        )

        # Memory consolidation: select what becomes durable long-term memory.
        self.memory_cfg = MemoryConfig(
            host=ollama_host,
            model=(os.environ.get("BUNNY_MODEL_MEMORY") or os.environ.get("BUNNY_MODEL_DECIDER") or getattr(self.cfg, "model", "llama3.2:3b")),
            temperature=float(os.environ.get("BUNNY_TEMP_MEMORY", "0.15")),
            num_ctx=int(os.environ.get("BUNNY_CTX_MEMORY", "2048")),
            stream=False,
        )

        # Axiom refinement: dedicated operationalization of A1..A4 (runs during idle).
        self.axiom_refine_cfg = AxiomRefineConfig(
            host=ollama_host,
            model=(os.environ.get("BUNNY_MODEL_AXIOM") or os.environ.get("BUNNY_MODEL_DAYDREAM") or getattr(self.cfg, "model", "llama3.3")),
            temperature=float(os.environ.get("BUNNY_TEMP_AXIOM", "0.15")),
            num_ctx=int(os.environ.get("BUNNY_CTX_AXIOM", "2048")),
            stream=False,
        )

        # Policy kernel: trainable, deterministic action prior (mutatable by Daydream).
        self.policy_cfg = PolicyKernelConfig(
            enable=str(os.environ.get('BUNNY_POLICY_ENABLE','1')).strip() not in ('0','false','False'),
            eta=float(os.environ.get('BUNNY_POLICY_ETA','0.05')),
            l2_decay=float(os.environ.get('BUNNY_POLICY_L2','0.001')),
            max_abs=float(os.environ.get('BUNNY_POLICY_MAXABS','3.0')),
            frob_tau=float(os.environ.get('BUNNY_POLICY_FROB','25.0')),
        )
        self.policy = PolicyKernel(self.db, self.store, self.axis, cfg=self.policy_cfg)

        # Failure clustering + skills + DevLab (self-development pipeline)
        self.cluster_cfg = ClusterConfig(
            host=ollama_host,
            model=(os.environ.get("BUNNY_MODEL_CLUSTER") or getattr(self.cfg, "model", "llama3.3")),
            temperature=0.2,
            num_ctx=2048,
        )
        self.skill_cfg = SkillConfig(
            host=ollama_host,
            model=(os.environ.get("BUNNY_MODEL_SKILL") or getattr(self.cfg, "model", "llama3.3")),
            temperature=0.2,
            num_ctx=3072,
        )
        self.devlab_cfg = DevLabConfig(
            host=ollama_host,
            model=(os.environ.get("BUNNY_MODEL_DEVBOT") or getattr(self.cfg, "model", "llama3.3")),
            temperature=0.2,
            num_ctx=6144,
        )

        self._last_sleep = 0.0
        # decider/daydream/feedback are configured externally (models can differ)

        self.broker = broker
        self._lock = threading.Lock()

        # Ensure LLM calls run sequentially even with background autonomy threads.
        # This avoids CPU/RAM spikes and makes behavior more deterministic.
        self._llm_lock = threading.Lock()
        self._llm_organs = {
            "speech",
            "topic",
            "beliefs",
            "decider",
            "feedback",
            "evidence",
            "selfeval",
            "daydream",
            "axiom_refine",
            "sleep",
            "evolve",
            "curriculum",
            "cluster",
            "skill",
            "devlab",
            "workspace_arbiter",
        }

        # Episode tagging (salience-driven): models human tag-and-capture consolidation.
        # Active episode context is used to boost nearby STM and belief stickiness.
        self._active_episode = None  # dict(id, center_ui_id, strength, window, tau, valence)

        # Generic capability bus (optional modules: tools/sensors/actors).
        # Never required for core operation; failures must be reflected as pain via health metrics.
        self.cap = CapabilityBus(self.db, self.hb.enqueue)

        # activation thresholds (generic; tunable via env)
        # NOTE: default tuned to be less conservative than earlier drafts so the system actually
        # uses organs with typical decider outputs (~0.4-0.7). Can be overridden via env.
        self.th_websense = float(os.environ.get("BUNNY_TH_WEBSENSE", "0.45"))
        self.th_daydream = float(os.environ.get("BUNNY_TH_DAYDREAM", "0.45"))
        self.th_evolve = float(os.environ.get("BUNNY_TH_EVOLVE", "0.55"))
        self.th_autotalk = float(os.environ.get("BUNNY_TH_AUTOTALK", "0.75"))
        self.idle_period_s = float(os.environ.get("BUNNY_IDLE_PERIOD", "6"))
        self.idle_cooldown_s = float(os.environ.get("BUNNY_IDLE_COOLDOWN", "30"))
        self._last_idle_action = 0.0

        # Idle self-evaluation (including silence-as-signal): allows learning without user ratings.
        self.silence_eval_delay_s = float(os.environ.get("BUNNY_SILENCE_EVAL_DELAY_S", "180"))
        self.silence_eval_cooldown_s = float(os.environ.get("BUNNY_SILENCE_EVAL_COOLDOWN_S", "600"))
        self.idle_selfeval_cooldown_s = float(os.environ.get("BUNNY_IDLE_SELFEVAL_COOLDOWN_S", "90"))
        self.idle_reward_scale = float(os.environ.get("BUNNY_IDLE_REWARD_SCALE", "0.35"))
        self.silence_reward_scale = float(os.environ.get("BUNNY_SILENCE_REWARD_SCALE", "0.20"))

        self.autotalk_cooldown_s = float(os.environ.get("BUNNY_AUTOTALK_COOLDOWN", "600"))
        self._last_autotalk_user_id = 0
        self._last_autotalk = 0.0

        # Drive integration: interpret organ outputs as *targets* and convert to deltas
        # (prevents saturation and makes negative corrections possible).
        self.eta_drives = float(os.environ.get("BUNNY_ETA_DRIVES", "0.25"))      # decider/daydream/evolve
        self.eta_measure = float(os.environ.get("BUNNY_ETA_MEASURE", "0.80"))   # resources/health
        # V1 invariant: state axes are bounded to [0,1]. We keep affect as bounded axes as well.
        # (Signed deltas are still possible via _mode='delta', but the stored state remains [0,1].)
        self._signed_affect: set[str] = set()

        # Proposal hygiene
        self.evolve_cooldown_s = float(os.environ.get("BUNNY_EVOLVE_COOLDOWN", "3600"))
        self.refine_cooldown_s = float(os.environ.get("BUNNY_PROPOSAL_REFINE_COOLDOWN", "1800"))
        self.open_proposal_window_s = float(os.environ.get("BUNNY_OPEN_PROPOSAL_WINDOW", str(24 * 3600)))
        self._last_evolve = 0.0
        self._last_refine = 0.0

    def _targets_to_delta_payload(self, targets: Dict[str, Any], *, eta: float) -> Dict[str, Any]:
        """Convert axis targets into a stable delta payload for DriveFieldEncoder.

        For bounded targets in 0..1 this ensures next_state=(1-eta)*state+eta*target, which
        stays within [0,1] if state started in [0,1].
        """
        s = self.hb.load_state()
        out: Dict[str, float] = {}
        for k, v in (targets or {}).items():
            kk = str(k)
            idx = self.axis.get(kk)
            if idx is None or idx >= len(s.values):
                continue
            try:
                tgt = float(v)
            except Exception:
                continue

            # clamp target/current into the expected subspace
            if kk in self._signed_affect:
                tgt = max(-1.0, min(1.0, tgt))
                cur = max(-1.0, min(1.0, float(s.values[idx])))
            else:
                tgt = max(0.0, min(1.0, tgt))
                cur = max(0.0, min(1.0, float(s.values[idx])))

            d = float(eta) * (tgt - cur)
            if abs(d) < 1e-6:
                continue
            out[kk] = max(-1.0, min(1.0, float(d)))

        return {"drives": out, "_mode": "delta"}

    def _call_with_health(self, organ: str, fn, *, metrics: Dict[str, Any] | None = None):
        t0 = time.time()
        ok = True
        err = ""
        res = None
        try:
            # Serialize LLM-heavy organs to avoid resource spikes.
            if getattr(self, "_llm_lock", None) is not None and str(organ) in getattr(self, "_llm_organs", set()):
                with self._llm_lock:
                    res = fn()
            else:
                res = fn()
            return res
        except Exception as e:
            ok = False
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            dt_ms = (time.time() - t0) * 1000.0
            try:
                m = dict(metrics or {})
                # attach simple output size signal (helps diagnose model/tool regressions)
                if ok and isinstance(res, str):
                    m.setdefault("response_chars", len(res))
                db_add_health_log(self.db, organ=organ, ok=ok, latency_ms=float(dt_ms), error=err, metrics=m)
            except Exception:
                pass

            # Capability/tool telemetry: keep a compact trace of what ran.
            # This is intentionally generic; it makes tables like capability_calls non-empty
            # so learning/debug pipelines can reason about actual organ usage.
            try:
                rj: Dict[str, Any] = {}
                if ok:
                    if isinstance(res, str):
                        rj = {"type": "str", "chars": len(res)}
                    elif isinstance(res, dict):
                        rj = {"type": "dict", "keys": list(res.keys())[:24]}
                    elif isinstance(res, list):
                        rj = {"type": "list", "len": len(res)}
                    else:
                        rj = {"type": str(type(res).__name__)}
                db_add_capability_call(
                    self.db,
                    name=f"organ:{organ}",
                    ok=bool(ok),
                    latency_ms=float(dt_ms),
                    error=err,
                    args=dict(metrics or {}),
                    result=rj,
                )
            except Exception:
                pass

            # Energy coupling: compute/search time consumes resources.
            # No hard limits on memory; cost is paid via ENERGY/FATIGUE.
            try:
                do_cost = str(os.environ.get('BUNNY_ENERGY_FROM_COMPUTE', '1') or '1').lower() in ('1','true','yes','y')
                if do_cost:
                    k = float(os.environ.get('BUNNY_ENERGY_COST_PER_S', '0.006') or 0.006)
                    cost = max(0.0, float(dt_ms) / 1000.0) * max(0.0, k)
                    if cost > 1e-9:
                        payload = {
                            'drives': {
                                'energy': -float(cost),
                                'fatigue': +float(cost) * 0.60,
                                'stress': +float(cost) * 0.30,
                            },
                            '_mode': 'delta',
                            'source': f'compute:{organ}',
                            'latency_ms': float(dt_ms),
                        }
                        self.hb.enqueue(Event('resources', payload).with_time())
                        self.hb.step()
            except Exception:
                pass


    def _sample_gate(self, organ: str, *, phase: str, p: float) -> bool:
        """Probabilistic gating for organs.

        - p is interpreted as a probability (0..1)
        - enforces a simple refractory (min interval between runs)
        - stores the last sampled random value for debug

        This avoids hard threshold regimes and is closer to compute-scheduling in RL agents.
        """
        try:
            pp = float(p)
        except Exception:
            pp = 0.0
        pp = 0.0 if pp < 0.0 else 1.0 if pp > 1.0 else pp

        now = time.time()
        last = float(self._organ_last_run.get(str(organ), 0.0) or 0.0)
        min_i = float(self._organ_min_interval_s.get(str(organ), 0.0) or 0.0)
        if min_i > 0.0 and last > 0.0 and (now - last) < min_i:
            r = self._rng.random()
            self._last_gate_rand[str(organ)] = float(r)
            return False

        r = self._rng.random()
        self._last_gate_rand[str(organ)] = float(r)
        return bool(r < pp)

    def _iso_to_ts(self, iso: str) -> float:
        """Best-effort parse of UI ISO timestamps to epoch seconds."""
        try:
            # format is usually like 2026-02-28T07:37:51 (no Z) or with Z
            s = str(iso or "").strip()
            if not s:
                return 0.0
            if s.endswith("Z"):
                s = s[:-1]
            # tolerate fractional seconds
            if "." in s:
                s = s.split(".")[0]
            import datetime

            dt = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
            # Treat as UTC (DB stores UTC-like).
            return float(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        except Exception:
            return 0.0

    def _idle_try_silence_selfeval(self, recent: List[Dict[str, Any]]) -> None:
        """Use missing user input as a weak feedback signal.

        If the last assistant reply has not been followed by a user message for some time,
        run a low-impact self-eval and integrate its deltas. This drives learning even
        when the user goes silent.
        """
        try:
            now = time.time()
            last_user_id = 0
            last_user_ts = 0.0
            last_reply_id = 0
            last_reply_ts = 0.0
            for m in reversed(recent or []):
                k = str(m.get("kind") or "")
                mid = int(m.get("id") or 0)
                ts = self._iso_to_ts(str(m.get("created_at") or ""))
                if k == "user" and last_user_id == 0:
                    last_user_id, last_user_ts = mid, ts
                if k in ("reply", "assistant", "auto") and last_reply_id == 0:
                    last_reply_id, last_reply_ts = mid, ts
                if last_user_id and last_reply_id:
                    break

            if last_reply_id <= 0 or last_reply_ts <= 0.0:
                return
            # Only evaluate silence if the last thing said was by Bunny.
            if last_user_id > last_reply_id:
                return

            silence_s = max(0.0, now - float(last_reply_ts))
            if silence_s < float(self.silence_eval_delay_s):
                return

            # Cooldown per message id (persisted).
            last_done = int(db_meta_get(self.db, "silence_eval_last_reply_id", "0") or 0)
            if last_done == int(last_reply_id):
                return
            last_at = float(db_meta_get(self.db, "silence_eval_last_ts", "0") or 0.0)
            if last_at > 0.0 and (now - last_at) < float(self.silence_eval_cooldown_s):
                return

            # Reconstruct the (user, reply) pair (best effort).
            user_text, reply_text, reply_created_at = db_get_prev_user_for_reply(self.db, int(last_reply_id))
            if not reply_text.strip():
                return

            se = self._call_with_health(
                "selfeval",
                lambda: evaluate_outcome(
                    self.selfeval_cfg,
                    db_get_axioms(self.db),
                    self._state_summary(),
                    question=user_text or "(no user question; silence_eval)",
                    answer=reply_text,
                    websense_claims_json="",
                    meta_json=json.dumps(
                        {
                            "mode": "silence_eval",
                            "idle": True,
                            "user_replied": False,
                            "silence_seconds": float(silence_s),
                            "reply_id": int(last_reply_id),
                        },
                        ensure_ascii=False,
                    ),
                ),
            )

            # Scale reward (silence is weak signal).
            try:
                r = float(se.get("delta_reward", 0.0) or 0.0)
            except Exception:
                r = 0.0
            r = max(-1.0, min(1.0, r)) * float(self.silence_reward_scale)

            drives_delta = se.get("drives_delta") if isinstance(se.get("drives_delta"), dict) else {}
            if drives_delta:
                self.hb.enqueue(Event("decision", {"drives": drives_delta, "_mode": "delta"}).with_time())
                self.hb.step()

            # Plasticity update (decision coupling) using last known decision drives as features.
            try:
                x_feats = db_get_decision_drives_before(self.db, str(reply_created_at or ""))
            except Exception:
                x_feats = {}
            if drives_delta and x_feats and abs(float(r)) > 1e-9:
                self._apply_matrix_update("decision", drives_delta, x_feats, float(r))

            db_meta_set(self.db, "silence_eval_last_reply_id", str(int(last_reply_id)))
            db_meta_set(self.db, "silence_eval_last_ts", str(float(now)))
        except Exception:
            return

    def _idle_try_actions_selfeval(self, *, summary: Dict[str, Any], feats: Dict[str, Dict[str, float]], websense_payload: Dict[str, Any] | None = None) -> None:
        """Self-evaluate the last autonomous tick and assign credit to organs.

        - summary: compact JSON summary of what happened (no UI output)
        - feats: per-event-type feature dicts actually applied (decision/daydream/evolve)
        - websense_payload: original websense event payload (for encoder-based x)
        """
        try:
            now = time.time()
            last_at = float(db_meta_get(self.db, "idle_selfeval_last_ts", "0") or 0.0)
            if last_at > 0.0 and (now - last_at) < float(self.idle_selfeval_cooldown_s):
                return

            se = self._call_with_health(
                "selfeval",
                lambda: evaluate_outcome(
                    self.selfeval_cfg,
                    db_get_axioms(self.db),
                    self._state_summary(),
                    question="INTERNAL_IDLE_ACTIONS: evaluate autonomous actions for axiom-alignment and learning progress.",
                    answer=json.dumps(summary or {}, ensure_ascii=False)[:5000],
                    websense_claims_json="",
                    meta_json=json.dumps({"idle": True, "mode": "idle_actions", "summary": summary or {}}, ensure_ascii=False)[:5000],
                ),
            )

            try:
                r = float(se.get("delta_reward", 0.0) or 0.0)
            except Exception:
                r = 0.0
            r = max(-1.0, min(1.0, r)) * float(self.idle_reward_scale)

            drives_delta = se.get("drives_delta") if isinstance(se.get("drives_delta"), dict) else {}
            if drives_delta:
                self.hb.enqueue(Event("decision", {"drives": drives_delta, "_mode": "delta"}).with_time())
                self.hb.step()

            # Credit assignment: update event-coupling matrices that actually ran.
            if drives_delta and abs(float(r)) > 1e-9:
                try:
                    x_dec = feats.get("decision") or {}
                    if x_dec:
                        self._apply_matrix_update("decision", drives_delta, x_dec, float(r))
                except Exception:
                    pass
                try:
                    x_dd = feats.get("daydream") or {}
                    if x_dd:
                        self._apply_matrix_update("daydream", drives_delta, x_dd, float(r))
                except Exception:
                    pass
                try:
                    x_ev = feats.get("evolve") or {}
                    if x_ev:
                        self._apply_matrix_update("evolve", drives_delta, x_ev, float(r))
                except Exception:
                    pass
                try:
                    if isinstance(websense_payload, dict) and websense_payload:
                        enc = WebsenseEncoder(self.axis)
                        x_vec, _ = enc.encode(len(self.axis), Event("websense", websense_payload))
                        inv = {v: k for k, v in self.axis.items()}
                        x_ws: Dict[str, float] = {}
                        for i, vv in enumerate(x_vec or []):
                            if abs(float(vv)) < 1e-9:
                                continue
                            nm = inv.get(i)
                            if nm:
                                x_ws[nm] = float(vv)
                        if x_ws:
                            self._apply_matrix_update("websense", drives_delta, x_ws, float(r))
                except Exception:
                    pass

                # Policy kernel learning in idle too (organ run -> reward).
                try:
                    pt = (summary or {}).get('policy_trace') if isinstance(summary, dict) else None
                    if isinstance(pt, dict):
                        x_pol = pt.get('features') if isinstance(pt.get('features'), list) else []
                        v_pol = int(pt.get('version', 1) or 1)
                        if x_pol:
                            ran = (summary or {}).get('ran') if isinstance(summary, dict) else {}
                            ran = ran if isinstance(ran, dict) else {}
                            if bool(ran.get('websense')):
                                v_pol = self.policy.apply_update(from_version=v_pol, x=x_pol, action='websense', reward=float(r), note='idle_selfeval')
                            if bool(ran.get('daydream')):
                                v_pol = self.policy.apply_update(from_version=v_pol, x=x_pol, action='daydream', reward=float(r), note='idle_selfeval')
                            if bool(ran.get('evolve')):
                                v_pol = self.policy.apply_update(from_version=v_pol, x=x_pol, action='evolve', reward=float(r), note='idle_selfeval')
                except Exception:
                    pass

            db_meta_set(self.db, "idle_selfeval_last_ts", str(float(now)))
        except Exception:
            return

    def _integrate_pain_tick(self) -> None:
        """Compute immutable pain from recent health + matrix deltas and integrate as state."""
        try:
            hs = db_recent_health_stats(self.db, window_s=300)
            md = db_recent_matrix_delta(self.db, window_s=300)
            s = self.hb.load_state()
            idx_energy = self.axis.get("energy")
            energy_val = float(s.values[idx_energy]) if idx_energy is not None and idx_energy < len(s.values) else 0.0
            pain = compute_pain_physical(
                err_rate=float(hs.get("err_rate", 0.0) or 0.0),
                lat_ms_p95=float(hs.get("lat_ms_p95", 0.0) or 0.0),
                energy_value=energy_val,
                matrix_delta_frob_recent=float(md or 0.0),
            )
            # Total pain = max(physical, psych). Psych is integrated elsewhere from self-eval/feedback.
            idx_ps = self.axis.get("pain_psych")
            ps = float(s.values[idx_ps]) if idx_ps is not None and idx_ps < len(s.values) else 0.0
            total = max(float(pain), float(ps))
            self.hb.enqueue(
                Event(
                    "health",
                    self._targets_to_delta_payload({"pain_physical": float(pain), "pain": float(total)}, eta=self.eta_measure),
                ).with_time()
            )
            self.hb.step()
            # Rollback MUST NOT be driven by latency/resource pain; use psych pain as regression signal.
            s2 = self.hb.load_state()
            ps2 = float(s2.values[idx_ps]) if idx_ps is not None and idx_ps < len(s2.values) else 0.0
            self._maybe_rollback_on_pain(float(ps2), pain_physical_now=float(pain), pain_psych_now=float(ps2))
        except Exception:
            return

    def _maybe_rollback_on_pain(self, pain_total_now: float, pain_physical_now: float = 0.0, pain_psych_now: float = 0.0) -> None:
        """Rollback last plasticity update if pain regresses persistently.

        This is a core safety mechanism: learning must not accumulate bad parameter changes.
        We keep it simple and deterministic:
          - look at the most recent unrolled matrix update
          - if pain has increased significantly compared to pain_before, rollback adapter to from_version
        """
        try:
            row = db_get_last_unrolled_matrix_update(self.db, window_s=900)
            if not row:
                return
            try:
                rwd = float(row.get("reward", 0.0) or 0.0)
            except Exception:
                rwd = 0.0
            # Roll back only when the system believed the update was *good* (positive reward)
            # but psych pain regressed. Negative-reward updates are corrective and must persist.
            if rwd <= 0.15:
                return
            # avoid flapping: require a clear regression margin
            pain_before = float(row.get("pain_before", 0.0) or 0.0)
            if pain_total_now <= pain_before + 0.12:
                return

            # rollback adapter binding to from_version
            event_type = str(row.get("event_type") or "")
            matrix_name = str(row.get("matrix_name") or "")
            from_v = int(row.get("from_version") or 0)
            b = self.reg.get(event_type)
            if b is None or b.matrix_name != matrix_name:
                return
            if int(b.matrix_version) != int(row.get("to_version") or b.matrix_version):
                # adapter already moved; skip
                return

            b2 = AdapterBinding(
                event_type=b.event_type,
                encoder_name=b.encoder_name,
                matrix_name=b.matrix_name,
                matrix_version=from_v,
                meta=b.meta,
            )
            self.reg.upsert(b2)
            db_mark_matrix_rolled_back(
                self.db,
                int(row.get("id")),
                note=f"pain_regression pain_before={pain_before:.3f} pain_total_now={float(pain_total_now):.3f}",
            )

            # Secondary regression safety: revert recent trust/belief learning as it may have been induced by the bad update.
            try:
                n_tr = db_rollback_recent_trust(self.db, window_s=600)
                n_bl = db_rollback_recent_beliefs(self.db, window_s=600)
                if n_tr or n_bl:
                    db_add_message(self.db, "auto", f"[rollback] reverted trust={n_tr} beliefs={n_bl} (regression window)")
            except Exception:
                pass
            mid = db_add_message(
                self.db,
                "auto",
                f"[rollback] matrix={matrix_name} event={event_type} {row.get('to_version')}-> {from_v} due_to_pain {pain_before:.2f}->{pain_total_now:.2f}",
            )
            self.broker.publish("message", self._ui_message(mid))
        except Exception:
            return

    def _integrate_fatigue_tick(self) -> None:
        """Compute immutable fatigue/sleep_pressure and integrate as state via health event."""
        try:
            hs = db_recent_health_stats(self.db, window_s=300)
            s = self.hb.load_state()
            idx_energy = self.axis.get("energy")
            idx_pain = self.axis.get("pain_physical")
            idx_fatigue = self.axis.get("fatigue")
            energy_val = float(s.values[idx_energy]) if idx_energy is not None and idx_energy < len(s.values) else 0.5
            pain_val = float(s.values[idx_pain]) if idx_pain is not None and idx_pain < len(s.values) else 0.0
            prev_fatigue = float(s.values[idx_fatigue]) if idx_fatigue is not None and idx_fatigue < len(s.values) else 0.0
            user_rate = db_recent_user_msg_rate(self.db, window_s=300)
            fatigue, sleep_pressure = compute_fatigue(
                pain_value=pain_val,
                energy_value=energy_val,
                err_rate=float(hs.get("err_rate", 0.0) or 0.0),
                user_msgs_per_min=float(user_rate),
                prev_fatigue=prev_fatigue,
            )
            self.hb.enqueue(
                Event(
                    "health",
                    self._targets_to_delta_payload(
                        {"fatigue": float(fatigue), "sleep_pressure": float(sleep_pressure)},
                        eta=self.eta_measure,
                    ),
                ).with_time()
            )
            self.hb.step()
        except Exception:
            return

    def autonomous_tick(self) -> None:
        """Idle loop: run decider/daydream/websense without keyword heuristics.

        This is intentionally conservative (cooldown) to avoid runaway IO.
        """
        if getattr(self, "lite_mode", False):
            return
        now = time.time()
        if now - self._last_idle_action < self.idle_cooldown_s:
            return

        with self._lock:
            # Keep epistemic dispute floors during idle time (prevents certainty decay).
            self._enforce_dispute_floor(reason="idle_tick")

            # Use the last user message (if any) as anchor context.
            recent = db_list_messages(self.db, limit=20)
            last_user = ""
            for m in reversed(recent):
                if m.get("kind") == "user":
                    last_user = str(m.get("text") or "")
                    break
            try:
                # Deterministic action prior (trainable policy kernel).
                try:
                    pol = self.policy.predict(self.hb.load_state().values)
                    policy_hint = pol.get('probs') or {}
                    policy_trace = {'version': int(pol.get('version') or 1), 'features': pol.get('features') or []}
                except Exception:
                    policy_hint = {}
                    policy_trace = {'version': 1, 'features': []}

                # Add recent dispute telemetry (generic epistemic signal) to the decider.
                # This is not content-based routing; it simply informs the policy about recent corrections.
                try:
                    dc = int(db_meta_get(self.db, 'dispute_count', '0') or 0)
                except Exception:
                    dc = 0
                if dc > 0:
                    try:
                        policy_hint = dict(policy_hint)
                        policy_hint['dispute_count'] = dc
                    except Exception:
                        pass
                try:
                    rc = int(db_meta_get(self.db, 'recent_caught', '0') or 0)
                except Exception:
                    rc = 0
                if rc > 0:
                    try:
                        policy_hint = dict(policy_hint)
                        policy_hint['recent_caught'] = rc
                    except Exception:
                        pass

                beliefs = db_list_beliefs(self.db, limit=12)
                decision = self._call_with_health(
                    "decider",
                    lambda: decide_pressures(
                        self.decider_cfg,
                        db_get_axioms(self.db),
                        self._state_summary(),
                        last_user,
                        beliefs,
                        scope="idle",
                        workspace=db_get_workspace_current(self.db),
                        needs=db_get_needs_current(self.db),
                        wishes=db_get_wishes_current(self.db),
                        self_report=build_self_report(self.hb.load_state(), self.axis).to_dict(),
                        active_topic=_get_active_topic(db_get_workspace_current(self.db)),
                        policy_hint=policy_hint,
                    ),
                )
            except Exception:
                return

            db_add_decision(self.db, "idle", last_user, decision)
            drives = decision.get("drives") if isinstance(decision.get("drives"), dict) else {}
            actions = decision.get("actions") if isinstance(decision.get("actions"), dict) else {}

            # Track what happens in this autonomous tick (for idle self-eval + credit assignment).
            tick_summary: Dict[str, Any] = {"actions": actions, "drives": drives, "policy_trace": policy_trace, "ran": {"daydream": False, "websense": False, "evolve": False}}
            tick_feats: Dict[str, Dict[str, float]] = {"decision": {}, "daydream": {}, "evolve": {}}
            tick_ws_payload: Dict[str, Any] | None = None

            # Ops trace: evolve gate (user)
            try:
                s_ev = self.hb.load_state()
                idx_ev = self.axis.get("pressure_evolve")
                p_ev = float(s_ev.values[idx_ev]) if idx_ev is not None and idx_ev < len(s_ev.values) else 0.0
            except Exception:
                p_ev = 0.0
            try:
                ev_action = float(((decision.get("actions") or {}).get("evolve") or 0.0))
            except Exception:
                ev_action = 0.0
            try:
                want_evolve = max(float(p_ev), float(ev_action)) >= self.th_evolve
                db_add_organ_gate_log(
                    self.db,
                    phase="user",
                    organ="evolve",
                    score=float(max(float(p_ev), float(ev_action))),
                    threshold=float(self.th_evolve),
                    want=bool(want_evolve),
                    data={"pressure_evolve": float(p_ev), "action": float(ev_action)},
                )
            except Exception:
                pass

            # Contract normalization: if the decider gives actions but omits the corresponding
            # pressure deltas, promote actions into pressure drives (keeps logic AI-driven,
            # avoids keyword heuristics, and prevents "no-op" decisions).
            try:
                ws_a = float(actions.get("websense", 0.0) or 0.0)
            except Exception:
                ws_a = 0.0
            try:
                dd_a = float(actions.get("daydream", 0.0) or 0.0)
            except Exception:
                dd_a = 0.0
            try:
                ev_a = float(actions.get("evolve", 0.0) or 0.0)
            except Exception:
                ev_a = 0.0
            if ws_a > 0.0:
                prev = float(drives.get("pressure_websense", 0.0) or 0.0) if isinstance(drives, dict) else 0.0
                drives["pressure_websense"] = max(prev, ws_a)
            if dd_a > 0.0:
                prev = float(drives.get("pressure_daydream", 0.0) or 0.0) if isinstance(drives, dict) else 0.0
                drives["pressure_daydream"] = max(prev, dd_a)
            if ev_a > 0.0:
                prev = float(drives.get("pressure_evolve", 0.0) or 0.0) if isinstance(drives, dict) else 0.0
                drives["pressure_evolve"] = max(prev, ev_a)
            if drives:
                dp = self._targets_to_delta_payload(drives, eta=self.eta_drives)
                try:
                    tick_feats["decision"] = dict((dp.get("drives") or {}) if isinstance(dp, dict) else {})
                except Exception:
                    tick_feats["decision"] = {}
                self.hb.enqueue(Event("decision", dp).with_time())
                self.hb.step()
                self.broker.publish("status", db_status(self.db))

            # Daydream (idle): gate from BOTH internal pressure and decider action.
            try:
                s_dd = self.hb.load_state()
                idx_dd = self.axis.get("pressure_daydream")
                p_dd = float(s_dd.values[idx_dd]) if idx_dd is not None and idx_dd < len(s_dd.values) else 0.0
            except Exception:
                p_dd = 0.0
            try:
                dd_action = float(((decision.get("actions") or {}).get("daydream") or 0.0))
            except Exception:
                dd_action = 0.0
            # KI-like gating: treat the score as a probability and SAMPLE (Bernoulli),
            # instead of hard thresholding. This avoids "never" / "always" regimes.
            score_dd = max(float(p_dd), float(dd_action))
            score_dd = 0.0 if score_dd < 0.0 else 1.0 if score_dd > 1.0 else score_dd
            want_daydream = self._sample_gate("daydream", phase="idle", p=score_dd)
            try:
                db_add_organ_gate_log(
                    self.db,
                    phase="idle",
                    organ="daydream",
                    score=float(score_dd),
                    threshold=float(self.th_daydream),
                    want=bool(want_daydream),
                    data={"pressure_daydream": float(p_dd), "action": float(dd_action), "mode": "sample", "rand": float(self._last_gate_rand.get('daydream', -1.0))},
                )
            except Exception:
                pass
            if want_daydream:
                try:
                    tick_summary["ran"]["daydream"] = True
                except Exception:
                    pass
                try:
                    self._organ_last_run["daydream"] = time.time()
                except Exception:
                    pass

                try:
                    dd = self._call_with_health(
                        "daydream",
                        lambda: run_daydream(
                            self.daydream_cfg,
                            db_get_axioms(self.db),
                            self._state_summary(),
                            recent,
                            recent_evidence=db_get_recent_evidence(self.db, limit=2),
                            existing_interpretations=db_group_axiom_interpretations(self.db, limit_per_axiom=10),
                            trigger="idle",
                        ),
                    )
                    db_add_daydream(self.db, "idle", {"state": self._state_summary()}, dd)

                    # Memory impact: daydream can write durable long-term memories and beliefs.
                    try:
                        ws_items_now = db_get_workspace_current(self.db)
                        active_topic_now = _get_active_topic(ws_items_now)
                        for it in (dd.get("memory_long_writes") or []):
                            if not isinstance(it, dict):
                                continue
                            summ = str(it.get("summary") or "").strip()
                            if not summ:
                                continue
                            srcs = it.get("sources") if isinstance(it.get("sources"), list) else []
                            if srcs:
                                u0 = str(srcs[0] or "").strip()
                                if u0 and u0 not in summ and len(summ) <= 190:
                                    summ = (summ + " | " + u0)[:220]
                            db_add_memory_long(
                                self.db,
                                summary=summ,
                                topic=str(it.get("topic") or active_topic_now),
                                modality="daydream",
                                salience=float(it.get("salience", 0.6) or 0.6),
                                axioms=(it.get("axioms") if isinstance(it.get("axioms"), list) else []),
                            )
                        for b in (dd.get("beliefs") or []):
                            if not isinstance(b, dict):
                                continue
                            db_upsert_belief(
                                self.db,
                                str(b.get("subject") or ""),
                                str(b.get("predicate") or ""),
                                str(b.get("object") or ""),
                                float(b.get("confidence", 0.6) or 0.6),
                                str(b.get("provenance") or "daydream"),
                                topic=active_topic_now,
                                compress=True,
                            )
                    except Exception:
                        pass

                    ax_int = dd.get("axiom_interpretations") if isinstance(dd.get("axiom_interpretations"), dict) else {}
                    for ak, val in (ax_int or {}).items():
                        s = (str(val) if val is not None else "").strip()
                        if not s:
                            continue
                        if str(ak) not in ("A1", "A2", "A3", "A4"):
                            continue
                        db_upsert_axiom_interpretation(self.db, str(ak), "rewrite", "latest", s, 0.4, "daydream")

                    # Persist operational specs (atomic, testable) to drive concrete development.
                    try:
                        ax_specs = dd.get('axiom_specs') if isinstance(dd.get('axiom_specs'), dict) else {}
                        for ak, items in (ax_specs or {}).items():
                            if str(ak) not in ('A1','A2','A3','A4'):
                                continue
                            if not isinstance(items, list):
                                continue
                            for it in items[:8]:
                                if not isinstance(it, dict):
                                    continue
                                rule = str(it.get('rule') or '').strip()
                                when = str(it.get('when') or '').strip()
                                do = str(it.get('do') or '').strip()
                                avoid = str(it.get('avoid') or '').strip()
                                sig = it.get('signals') if isinstance(it.get('signals'), list) else []
                                sigs = ','.join([str(x) for x in sig if str(x).strip()])[:180]
                                ex = str(it.get('example') or '').strip()
                                cx = str(it.get('counterexample') or '').strip()
                                if not rule:
                                    continue
                                blob = f"rule:{rule} | when:{when} | do:{do} | avoid:{avoid} | signals:{sigs} | ex:{ex} | anti:{cx}".strip()
                                blob = blob[:520]
                                ck = hashlib.sha1(blob.encode('utf-8')).hexdigest()[:12]
                                db_upsert_axiom_interpretation(self.db, str(ak), 'spec', f'spec_{ck}', blob, 0.55, 'daydream')
                    except Exception:
                        pass

                    # Keep digests warm (deterministic rebuild); avoids empty axiom_digests when sleep never runs.
                    try:
                        db_refresh_axiom_digests(self.db)
                    except Exception:
                        pass
                    # Persist current needs/wishes derived from daydream (first-class, DB-backed)
                    try:
                        if isinstance(dd.get('needs'), list):
                            db_set_needs_current(self.db, {'needs': dd.get('needs') or [], 'source': 'daydream', 'at': now_iso()})
                        if isinstance(dd.get('wishes'), list):
                            db_set_wishes_current(self.db, {'wishes': dd.get('wishes') or [], 'source': 'daydream', 'at': now_iso()})
                    except Exception:
                        pass
                    # Daydream can emit proposals; store as mutation proposals for /proposal inspection
                    try:
                        # Avoid proposal spam: if there is already an open proposal, do not add more from daydream.
                        if db_get_latest_open_mutation_proposal(self.db, window_s=self.open_proposal_window_s) is None:
                            for p in (dd.get('proposals') or []):
                                if isinstance(p, dict):
                                    db_add_mutation_proposal(self.db, trigger='daydream', proposal=p)
                    except Exception:
                        pass
                    # Spontaneous WebSense exploration from daydream web_queries (state budgeted)
                    try:
                        qs = dd.get('web_queries') if isinstance(dd.get('web_queries'), list) else []
                        q0 = str(qs[0]).strip() if qs else ''
                        if q0:
                            s_now = self.hb.load_state()
                            pain = float(s_now.values[self.axis.get('pain',0)]) if self.axis.get('pain',0) < len(s_now.values) else 0.0
                            energy = float(s_now.values[self.axis.get('energy',1)]) if self.axis.get('energy',1) < len(s_now.values) else 0.0
                            if pain < 0.55 and energy > 0.35:
                                results = self._call_with_health('websense_search', lambda: search_ddg(q0, k=6), metrics={'query': q0, 'k': 6, 'tag': 'daydream'})
                                # store minimal page log for later evidence extraction
                                for r in results[:4]:
                                    try:
                                        db_add_websense_page(self.db, q0, r.url, r.title, r.snippet, '', urlparse(r.url).hostname or '', getattr(r, 'hash', ''), 1)
                                    except Exception:
                                        pass
                                aid = db_add_message(self.db, 'auto', f"[websense] daydream query=\"{q0}\" results={len(results)}")
                                self.broker.publish('message', self._ui_message(aid))
                    except Exception:
                        pass

                    # Keep daydream thoughts inside daydream_log (not as UI chat messages).
                    dd_drives = dd.get("drives") if isinstance(dd.get("drives"), dict) else {}
                    if dd_drives:
                        ddp = self._targets_to_delta_payload(dd_drives, eta=self.eta_drives)
                        try:
                            tick_feats["daydream"] = dict((ddp.get("drives") or {}) if isinstance(ddp, dict) else {})
                        except Exception:
                            tick_feats["daydream"] = {}
                        self.hb.enqueue(Event("daydream", ddp).with_time())
                        self.hb.step()
                        self.broker.publish("status", db_status(self.db))

                    # If daydream did not emit axiom interpretations, force a focused refinement pass.
                    try:
                        ax_int = dd.get("axiom_interpretations") if isinstance(dd.get("axiom_interpretations"), dict) else {}
                        have_any = any(str(v or "").strip() for v in (ax_int or {}).values())
                        if not have_any:
                            existing = db_group_axiom_interpretations(self.db, limit_per_axiom=10)
                            ar = self._call_with_health(
                                "axiom_refine",
                                lambda: refine_axioms(
                                    self.axiom_refine_cfg,
                                    db_get_axioms(self.db),
                                    self._state_summary(),
                                    recent,
                                    existing_interpretations=existing,
                                    trigger="idle_fallback",
                                ),
                            )
                            ax2 = ar.get("axiom_interpretations") if isinstance(ar.get("axiom_interpretations"), dict) else {}
                            for ak, val in (ax2 or {}).items():
                                s = (str(val) if val is not None else "").strip()
                                if not s:
                                    continue
                                if str(ak) not in ("A1", "A2", "A3", "A4"):
                                    continue
                                db_upsert_axiom_interpretation(self.db, str(ak), "rewrite", "latest", s, 0.45, "axiom_refine")

                            # Persist operational specs from axiom_refine.
                            try:
                                ax_specs = ar.get('axiom_specs') if isinstance(ar.get('axiom_specs'), dict) else {}
                                for ak, items in (ax_specs or {}).items():
                                    if str(ak) not in ('A1','A2','A3','A4'):
                                        continue
                                    if not isinstance(items, list):
                                        continue
                                    for it in items[:10]:
                                        if not isinstance(it, dict):
                                            continue
                                        rule = str(it.get('rule') or '').strip()
                                        when = str(it.get('when') or '').strip()
                                        do = str(it.get('do') or '').strip()
                                        avoid = str(it.get('avoid') or '').strip()
                                        sig = it.get('signals') if isinstance(it.get('signals'), list) else []
                                        sigs = ','.join([str(x) for x in sig if str(x).strip()])[:180]
                                        ex = str(it.get('example') or '').strip()
                                        cx = str(it.get('counterexample') or '').strip()
                                        if not rule:
                                            continue
                                        blob = f"rule:{rule} | when:{when} | do:{do} | avoid:{avoid} | signals:{sigs} | ex:{ex} | anti:{cx}".strip()
                                        blob = blob[:520]
                                        ck = hashlib.sha1(blob.encode('utf-8')).hexdigest()[:12]
                                        db_upsert_axiom_interpretation(self.db, str(ak), 'spec', f'spec_{ck}', blob, 0.60, 'axiom_refine')
                            except Exception:
                                pass
                            # optional drives from axiom refine
                            ar_drives = ar.get("drives") if isinstance(ar.get("drives"), dict) else {}
                            if ar_drives:
                                self.hb.enqueue(Event("daydream", self._targets_to_delta_payload(ar_drives, eta=self.eta_drives)).with_time())
                                self.hb.step()
                                self.broker.publish("status", db_status(self.db))
                            tick_summary["axiom_refine_fallback"] = {"focus": ar.get("focus_axiom"), "notes": ar.get("notes", "")}
                    except Exception:
                        pass
                    self._last_idle_action = time.time()
                except Exception:
                    pass

            # Evolve (idle): propose self-development mutations (human approval required)
            try:
                s_ev = self.hb.load_state()
                idx_ev = self.axis.get("pressure_evolve")
                p_ev = float(s_ev.values[idx_ev]) if idx_ev is not None and idx_ev < len(s_ev.values) else 0.0
            except Exception:
                p_ev = 0.0
            try:
                ev_action = float(((decision.get("actions") or {}).get("evolve") or 0.0))
            except Exception:
                ev_action = 0.0
            want_evolve = max(float(p_ev), float(ev_action)) >= self.th_evolve
            try:
                db_add_organ_gate_log(
                    self.db,
                    phase="idle",
                    organ="evolve",
                    score=float(max(float(p_ev), float(ev_action))),
                    threshold=float(self.th_evolve),
                    want=bool(want_evolve),
                    data={"pressure_evolve": float(p_ev), "action": float(ev_action)},
                )
            except Exception:
                pass
            if want_evolve:

                try:
                    tick_summary["ran"]["evolve"] = True
                except Exception:
                    pass

                try:
                    now_t = time.time()
                    openp = db_get_latest_open_mutation_proposal(self.db, window_s=self.open_proposal_window_s)
                    if openp is not None:
                        # If a proposal is already open, refine it occasionally instead of creating new ones.
                        if now_t - float(self._last_refine or 0.0) < float(self.refine_cooldown_s):
                            raise RuntimeError("skip_evolve_refine_cooldown")
                        trigger = "refine"
                        existing = [openp.get("proposal") or {}]
                    else:
                        # No open proposal: enforce cooldown to avoid proposal spam.
                        if now_t - float(self._last_evolve or 0.0) < float(self.evolve_cooldown_s):
                            raise RuntimeError("skip_evolve_cooldown")
                        trigger = "idle"
                        existing = []

                    beliefs = db_list_beliefs(self.db, limit=20)
                    # Build & persist self-model snapshot
                    self_model = build_self_model(
                        app_dir=os.path.dirname(__file__),
                        axes=list(self.axis.keys()),
                        adapters=self.reg.list_bindings(),
                        matrices=self.store.list_matrices(),
                        model_cfg={
                            "speech": getattr(self.cfg, "model", ""),
                            "decider": getattr(self.decider_cfg, "model", ""),
                            "daydream": getattr(self.daydream_cfg, "model", ""),
                            "feedback": getattr(self.feedback_cfg, "model", ""),
                            "selfeval": getattr(self.selfeval_cfg, "model", ""),
                            "evolve": getattr(self.evolve_cfg, "model", ""),
                        },
                    )
                    db_add_self_model(self.db, self_model)

                    evo = self._call_with_health(
                        "evolve",
                        lambda: propose_mutations(
                            self.evolve_cfg,
                            db_get_axioms(self.db),
                            self._state_summary(),
                            recent,
                            beliefs,
                            self_model,
                            trigger=trigger,
                            existing_proposals=existing,
                        ),
                    )

                    # persist proposals (refine updates the existing open one)
                    if trigger == "refine" and openp is not None:
                        mps = evo.get("mutation_proposals") if isinstance(evo.get("mutation_proposals"), list) else []
                        p0 = mps[0] if mps and isinstance(mps[0], dict) else None
                        if p0 is not None:
                            db_update_mutation_proposal(self.db, int(openp.get("id")), p0, note="refined")
                            self._last_refine = now_t
                    else:
                        for p in (evo.get("mutation_proposals") or []):
                            if isinstance(p, dict):
                                db_add_mutation_proposal(self.db, "idle", p)
                        self._last_evolve = now_t

                    # integrate drive deltas
                    evo_drives = evo.get("drives") if isinstance(evo.get("drives"), dict) else {}
                    if evo_drives:
                        evp = self._targets_to_delta_payload(evo_drives, eta=self.eta_drives)
                        try:
                            tick_feats["evolve"] = dict((evp.get("drives") or {}) if isinstance(evp, dict) else {})
                        except Exception:
                            tick_feats["evolve"] = {}
                        self.hb.enqueue(Event("evolve", evp).with_time())
                        self.hb.step()
                        self.broker.publish("status", db_status(self.db))

                    self._last_idle_action = time.time()
                except Exception:
                    # silence cool-down skips
                    pass

            # WebSense (idle): only when the decider explicitly requests it *and* provides a concrete query.
            # No string heuristics here. If the decider cannot form a query, we do not run WebSense.
            if float(((decision.get("actions") or {}).get("websense") or 0.0)) >= self.th_websense:
                q = (decision.get("web_query") or "").strip()
                if q.lower() in ("search for user query", "direct_response_to_user_question"):
                    q = ""
                if q:
                    try:
                        try:
                            tick_summary["ran"]["websense"] = True
                        except Exception:
                            pass
                        results = self._call_with_health(
                            "websense_search",
                            lambda: search_ddg(q, k=4),
                            metrics={"query": q, "k": 4, "scope": "idle"},
                        )
                        fetched = []
                        for r in results[:2]:
                            if not r.url:
                                continue
                            try:
                                fetched.append(
                                    self._call_with_health(
                                        "websense_fetch",
                                        lambda u=r.url: fetch(u, timeout_s=12.0),
                                        metrics={"url": r.url, "scope": "idle"},
                                    )
                                )
                            except Exception:
                                continue
                        pages = fetched
                        unique_domains = {getattr(p, "domain", "") for p in pages if getattr(p, "domain", "")}
                        ok_flag = 1 if len(results) > 0 else 0
                        for p in pages:
                            db_add_websense_page(self.db, q, {
                                "url": getattr(p, "url", ""),
                                "title": getattr(p, "title", ""),
                                "snippet": getattr(p, "snippet", ""),
                                "body": getattr(p, "body", ""),
                                "domain": getattr(p, "domain", ""),
                                "hash": getattr(p, "hash", ""),
                            }, ok=ok_flag)
                        tick_ws_payload = {"pages": len(pages), "domains": len(unique_domains), "ok": int(ok_flag), "query": q, "scope": "idle"}
                        self.hb.enqueue(Event("websense", tick_ws_payload).with_time())
                        self.hb.step()
                        self.broker.publish("status", db_status(self.db))
                        self._last_idle_action = time.time()
                    except Exception:
                        pass

            
            # Auto-speech (unsolicited check-in): only when the decider explicitly requests it.
            # Kind='auto' so users can react/penalize it; learnability comes from feedback + selfeval.
            try:
                # Avoid repeating auto-talk for the same last user message.
                last_user_id = 0
                for m in reversed(recent):
                    if m.get("kind") == "user":
                        last_user_id = int(m.get("id") or 0)
                        break

                # Option 2: allow autotalk only when epistemic error is low AND no open WebSense need exists.
                s_auto = self.hb.load_state()
                idx_err = self.axis.get("error_signal")
                err_now = float(s_auto.values[idx_err]) if idx_err is not None and idx_err < len(s_auto.values) else 0.0
                idx_pws = self.axis.get("pressure_websense")
                pws_now = float(s_auto.values[idx_pws]) if idx_pws is not None and idx_pws < len(s_auto.values) else 0.0
                idx_f = self.axis.get("freshness_need")
                f_now = float(s_auto.values[idx_f]) if idx_f is not None and idx_f < len(s_auto.values) else 0.0
                auto_max_err = float(os.environ.get("BUNNY_AUTOTALK_MAX_ERROR_SIGNAL", "0.15") or 0.15)
                auto_max_ws = float(os.environ.get("BUNNY_AUTOTALK_MAX_WS_NEED", "0.25") or 0.25)
                allow_autotalk_state = (float(err_now) <= float(auto_max_err) and max(float(pws_now), float(f_now)) <= float(auto_max_ws))

                if (
                    float(((decision.get("actions") or {}).get("reply") or 0.0)) >= self.th_autotalk
                    and last_user_id > int(getattr(self, "_last_autotalk_user_id", 0) or 0)
                    and (time.time() - float(getattr(self, "_last_autotalk", 0.0) or 0.0)) >= float(self.autotalk_cooldown_s)
                    and bool(allow_autotalk_state)
                ):
                    ws_items_cur = db_get_workspace_current(self.db)
                    active_topic = _get_active_topic(ws_items_cur)
                    needs_cur = db_get_needs_current(self.db)
                    wishes_cur = db_get_wishes_current(self.db)
                    self_rep = build_self_report(self.hb.load_state(), self.axis).to_dict()
                    workspace_block = f"\n\nWORKSPACE:\n" + json.dumps(ws_items_cur or [], ensure_ascii=False) if ws_items_cur else ""
                    needs_block = f"\n\nNEEDS_CURRENT:\n" + json.dumps(needs_cur or {}, ensure_ascii=False) if needs_cur else ""
                    wishes_block = f"\n\nWISHES_CURRENT:\n" + json.dumps(wishes_cur or {}, ensure_ascii=False) if wishes_cur else ""
                    selfreport_block = f"\n\nSELF_REPORT:\n" + json.dumps(self_rep or {}, ensure_ascii=False)
                    mood_obj = project_mood(self.hb.load_state(), self.axis).to_dict()
                    mood_block = f"\n\nMOOD:\n" + json.dumps(mood_obj or {}, ensure_ascii=False)
                    topic_block = f"\n\nACTIVE_TOPIC: {active_topic}" if active_topic else ""

                    # LTM recall budget scales with ENERGY (no hard pruning).
                    s_tmp = self.hb.load_state()
                    idx_e = self.axis.get('energy')
                    e_now = float(s_tmp.values[idx_e]) if idx_e is not None and idx_e < len(s_tmp.values) else 0.5
                    e_now = 0.0 if e_now < 0.0 else 1.0 if e_now > 1.0 else e_now
                    mem_lim = 4 + int(8 * e_now)
                    mem_long = db_get_memory_long(self.db, limit=mem_lim, topic=active_topic)
                    mem_long_ctx = render_memory_long_context(mem_long)
                    mem_long_block = f"\n\nMEMORY_LONG:\n{mem_long_ctx}" if mem_long_ctx else ""

                    mem_items = db_get_memory_short(self.db, limit=10)
                    mem_ctx = render_memory_context(mem_items)
                    mem_block = f"\n\nMEMORY_SHORT:\n{mem_ctx}" if mem_ctx else ""

                    beliefs_items = db_list_beliefs(self.db, limit=10, topic=active_topic)
                    beliefs_ctx = render_beliefs_context(beliefs_items)
                    beliefs_block = f"\n\nBELIEFS:\n{beliefs_ctx}" if beliefs_ctx else ""

                    # Auto talk prompt: short, non-spammy, one question max.
                    sys_prompt = (
                        "You are Bunny, a digital organism. You may emit ONE short unsolicited message when idle. "
                        "Be helpful, calm, and non-intrusive. "
                        "Do NOT output logs/JSON/tags. "
                        "If you have nothing valuable to say, output an empty string."
                    )
                    user_prompt = (
                        f"INTERNAL_STATE: {self._state_summary()}{topic_block}{selfreport_block}{mood_block}{needs_block}{wishes_block}{workspace_block}{mem_long_block}{mem_block}{beliefs_block}\n\n"
                        f"CONTEXT: idle/autonomous tick. Last user message: {last_user!r}\n"
                        "TASK: If appropriate, send a brief check-in or one short question that helps the user. Otherwise output empty string."
                    )
                    out = self._call_with_health(
                        "speech",
                        lambda: ollama_chat(self.cfg, sys_prompt, user_prompt),
                        metrics={
                            "model": self.cfg.model,
                            "ctx": int(self.cfg.num_ctx),
                            "sys_chars": len(sys_prompt),
                            "user_chars": len(user_prompt),
                            "auto": True,
                        },
                    )
                    out = (out or "").strip()
                    if out:
                        mid = db_add_message(self.db, "auto", out)
                        self.broker.publish("message", self._ui_message(mid))
                        self.hb.enqueue(Event("speech_outcome", {"len": len(out), "auto": True}).with_time())
                        self.hb.step()
                        self.broker.publish("status", db_status(self.db))
                        self._last_idle_action = time.time()
                        self._last_autotalk = time.time()
                        self._last_autotalk_user_id = int(last_user_id)
            except Exception:
                pass

# Sleep / consolidation: if sleep_pressure is high and we've been idle for a bit, consolidate.
            try:
                s = self.hb.load_state()
                idx_sp = self.axis.get("sleep_pressure")
                sp = float(s.values[idx_sp]) if idx_sp is not None and idx_sp < len(s.values) else 0.0
                idle_s = float(time.time() - float(self._last_idle_action or time.time()))
                if sp >= 0.70 and idle_s >= 20.0 and (time.time() - float(self._last_sleep or 0.0)) >= 120.0:
                    axioms = db_get_axioms(self.db)
                    beliefs = db_list_beliefs(self.db, limit=40)
                    ws = db_get_workspace_current(self.db)
                    recent = db_list_messages(self.db, limit=40)
                    summary = self._call_with_health(
                        "sleep",
                        lambda: sleep_consolidate(
                            self.sleep_cfg,
                            axioms,
                            self._state_summary(),
                            ws,
                            beliefs,
                            recent,
                            axiom_interpretations=db_group_axiom_interpretations(self.db, limit_per_axiom=24),
                        ),
                    )
                    if isinstance(summary, dict):
                        db_add_sleep_log(self.db, summary)

                        # Apply axiom digests (compressed interpretations) so they become first-class memory.
                        try:
                            ad = summary.get("axiom_digests") if isinstance(summary.get("axiom_digests"), dict) else {}
                            for ak, dg in (ad or {}).items():
                                s = (str(dg) if dg is not None else "").strip()
                                if not s:
                                    continue
                                if str(ak) not in ("A1", "A2", "A3", "A4"):
                                    continue
                                ck = hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
                                db_upsert_axiom_digest(self.db, str(ak), s[:600], ck)
                        except Exception:
                            pass

                        # Apply belief compression/merges and downranks.
                        try:
                            topic = _get_active_topic(ws) if isinstance(ws, list) else ""
                            for b in (summary.get("merged_beliefs") or []):
                                if not isinstance(b, dict):
                                    continue
                                db_upsert_belief(
                                    self.db,
                                    str(b.get("subject") or ""),
                                    str(b.get("predicate") or ""),
                                    str(b.get("object") or ""),
                                    confidence=float(b.get("confidence", 0.7) or 0.7),
                                    provenance=str(b.get("provenance") or "sleep_merge"),
                                    topic=topic,
                                    compress=True,
                                )
                            for b in (summary.get("downgrade_beliefs") or []):
                                if not isinstance(b, dict):
                                    continue
                                db_downrank_belief(
                                    self.db,
                                    str(b.get("subject") or ""),
                                    str(b.get("predicate") or ""),
                                    str(b.get("object") or ""),
                                    reason=str(b.get("reason") or "")[:120],
                                    topic=topic,
                                )
                        except Exception:
                            pass


                        # Soft forgetting: decay belief confidence over time and prune very old low-confidence details.
                        try:
                            hl = float(os.environ.get('BUNNY_BELIEF_HALF_LIFE_DAYS', '45') or 45)
                            floor = float(os.environ.get('BUNNY_BELIEF_FLOOR', '0.15') or 0.15)
                            ttl = float(os.environ.get('BUNNY_BELIEF_PRUNE_TTL_DAYS', '180') or 180)
                            below = float(os.environ.get('BUNNY_BELIEF_PRUNE_BELOW', '0.18') or 0.18)
                            db_age_and_prune_beliefs(self.db, half_life_days=hl, floor=floor, prune_ttl_days=ttl, prune_below=below)
                        except Exception:
                            pass


                        # Write a compact consolidation into long memory as a stable anchor (schema-migration safe)
                        try:
                            summ = str(summary.get("summary") or "").strip()[:1200]
                            if summ:
                                con = self.db.connect()
                                try:
                                    cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(memory_long)").fetchall()}
                                    if {"summary", "topic", "modality", "salience", "created_at"}.issubset(cols):
                                        con.execute(
                                            "INSERT INTO memory_long(summary,topic,modality,salience,created_at) VALUES(?,?,?,?,?)",
                                            (summ, (_get_active_topic(ws) if isinstance(ws, list) else "")[:80], "sleep", 0.6, now_iso()),
                                        )
                                    elif {"summary", "created_at"}.issubset(cols):
                                        con.execute(
                                            "INSERT INTO memory_long(summary,created_at) VALUES(?,?)",
                                            (summ, now_iso()),
                                        )
                                finally:
                                    con.commit()
                                    con.close()
                        except Exception:
                            pass
                    self._last_sleep = time.time()

                    try:
                        self._run_sleep_curriculum()
                    except Exception:
                        pass
                    self.broker.publish("status", db_status(self.db))
            except Exception:
                pass

            # resources tick (idle): keep energy/stress/affect baseline in sync even without user messages
            try:
                metrics = collect_resources()
                db_add_resources_log(self.db, metrics)
                e_t = float(metrics.get("energy", 0.5) or 0.5)
                s_t = float(metrics.get("stress", 0.5) or 0.5)
                targets = {
                    "energy": e_t,
                    "stress": s_t,
                    "arousal": max(-1.0, min(1.0, 2.0 * s_t - 1.0)),
                    "security": max(-1.0, min(1.0, 2.0 * e_t - 1.0)),
                    "valence": max(-1.0, min(1.0, 2.0 * e_t - 1.0)),
                    "frustration": max(-1.0, min(1.0, 2.0 * s_t - 1.0)),
                }
                self.hb.enqueue(Event("resources", self._targets_to_delta_payload(targets, eta=self.eta_measure)).with_time())
            except Exception:
                pass

            # immutable pain integration (health-driven) after idle actions
            self._integrate_pain_tick()
            self._integrate_fatigue_tick()

            # --- Idle self-evaluation ---
            # 1) Treat lack of user response as weak feedback (silence-as-signal).
            self._idle_try_silence_selfeval(recent)

            # 2) Evaluate the autonomous tick itself (credit assignment to organs).
            try:
                tick_summary.update({
                    "state": self._state_summary(),
                    "ws_payload": tick_ws_payload or {},
                })
                # Run only if we actually did something (organ activity or non-trivial deltas).
                did = bool(tick_ws_payload) or bool(tick_feats.get("daydream")) or bool(tick_feats.get("evolve")) or bool(tick_feats.get("decision"))
                if did:
                    self._idle_try_actions_selfeval(summary=tick_summary, feats=tick_feats, websense_payload=tick_ws_payload)
            except Exception:
                pass

    def ensure_seed(self) -> None:
        n = len(self.axis)
        mats = self.store.list_matrices()

        # Auto-resize existing matrices to current axis dimension (schema evolves by adding axes).
        # Preserves learned weights while extending with a sane diagonal for new axes.
        try:
            for m in list(mats or []):
                try:
                    name = str(m.get("name") or "")
                    ver = int(m.get("version") or 0)
                    nr = int(m.get("n_rows") or 0)
                    nc = int(m.get("n_cols") or 0)
                except Exception:
                    continue
                if not name or ver <= 0:
                    continue
                if nr == n and nc == n:
                    continue
                try:
                    A0 = self.store.get_sparse(name, ver)
                    ents0 = [(int(i), int(j), float(v)) for (i, j, v) in (getattr(A0, "entries", []) or []) if int(i) < n and int(j) < n]
                    # infer diagonal scale from first diagonal entry
                    diag_scale = None
                    for (i, j, v) in ents0:
                        if int(i) == int(j):
                            diag_scale = float(v)
                            break
                    if diag_scale is None:
                        diag_scale = 1.0
                    for k in range(int(nr), int(n)):
                        ents0.append((int(k), int(k), float(diag_scale)))
                    new_ver = int(ver) + 1
                    self.store.put_sparse(name, new_ver, n, n, ents0, meta={"resized": True, "from_version": int(ver), "updated_at": now_iso()}, parent_version=int(ver))
                    # Rebind any adapters that still point at the old version.
                    for b in (self.reg.list_bindings() or []):
                        try:
                            if str(b.get("matrix")) == str(name) and int(b.get("version") or 0) == int(ver):
                                self.reg.upsert(AdapterBinding(str(b.get("event_type")), str(b.get("encoder")), str(name), int(new_ver), b.get("meta") or {}))
                        except Exception:
                            continue
                except Exception:
                    continue
            mats = self.store.list_matrices()
        except Exception:
            pass

        protected_names = {"pain", "energy", "fatigue", "sleep_pressure"}
        protected_idx = [self.axis.get(nm) for nm in protected_names if self.axis.get(nm) is not None]

        def id_entries(scale: float, protect_metrics: bool = True):
            """Identity-like seed matrix with optional invariants for protected metrics axes.

            If protect_metrics=True, pain/energy are forced to identity (1.0) so no other event can
            decay/alter them. This prevents the system from "training down" pain/energy.
            """
            m = identity(n, float(scale))
            if protect_metrics and protected_idx:
                ents = [(int(i), int(j), float(v)) for (i, j, v) in (m.entries or [])]
                # remove any entry touching protected axes
                keep = []
                pset = {int(x) for x in protected_idx if x is not None}
                for (i, j, v) in ents:
                    if i in pset or j in pset:
                        continue
                    keep.append((i, j, v))
                for p in pset:
                    keep.append((p, p, 1.0))
                m.entries = keep
            return m.entries

        def have(name: str, ver: int) -> bool:
            return any(m["name"] == name and int(m["version"]) == ver for m in mats)

        if not have("A_user", 1):
            self.store.put_sparse("A_user", version=1, n_rows=n, n_cols=n, entries=id_entries(1.0, protect_metrics=True),
                                  meta={"desc":"identity injection for user_utterance"})
        if not have("A_decision", 1):
            self.store.put_sparse("A_decision", version=1, n_rows=n, n_cols=n, entries=id_entries(0.7, protect_metrics=True),
                                  meta={"desc":"LLM-decider drive coupling"})
        if not have("A_websense", 1):
            self.store.put_sparse("A_websense", version=1, n_rows=n, n_cols=n, entries=id_entries(0.5, protect_metrics=True),
                                  meta={"desc":"websense coupling"})
        if not have("A_daydream", 1):
            self.store.put_sparse("A_daydream", version=1, n_rows=n, n_cols=n, entries=id_entries(0.6, protect_metrics=True),
                                  meta={"desc":"daydream drive coupling"})

        if not have("A_resources", 1):
            self.store.put_sparse("A_resources", version=1, n_rows=n, n_cols=n, entries=id_entries(0.4, protect_metrics=False),
                                  meta={"desc":"resources/energy coupling"})

        if not have("A_health", 1):
            self.store.put_sparse("A_health", version=1, n_rows=n, n_cols=n, entries=id_entries(0.45, protect_metrics=False),
                                  meta={"desc":"health/pain coupling (immutable pain model inputs)"})

        if not have("A_evolve", 1):
            self.store.put_sparse("A_evolve", version=1, n_rows=n, n_cols=n, entries=id_entries(0.55, protect_metrics=True),
                                  meta={"desc":"evolve/self-development coupling"})

        # Epistemic sensor channel: evidence quality (uncertainty/confidence) should influence state.
        if not have("A_epistemic", 1):
            self.store.put_sparse(
                "A_epistemic",
                version=1,
                n_rows=n,
                n_cols=n,
                entries=id_entries(0.55, protect_metrics=True),
                meta={"desc": "epistemic sensor coupling (evidence -> uncertainty/confidence)"},
            )

        # Reward channel matrix (identity-like). Historically reward_signal bypassed matrices;
        # we keep it explicit so all events follow the same equation S' = decay*S + Σ A φ(E).
        if not have("A_reward", 1):
            self.store.put_sparse(
                "A_reward",
                version=1,
                n_rows=n,
                n_cols=n,
                entries=id_entries(1.0, protect_metrics=True),
                meta={"desc": "reward/high coupling (explicit, non-learned)"},
            )

        if self.reg.get("user_utterance") is None:
            self.reg.upsert(AdapterBinding("user_utterance", "simple_text_v1", "A_user", 1, {"desc":"user -> state"}))
        if self.reg.get("decision") is None:
            self.reg.upsert(AdapterBinding("decision", "drive_field_v1", "A_decision", 1, {"desc":"decider drives -> state"}))
        if self.reg.get("reward_signal") is None:
            self.reg.upsert(AdapterBinding("reward_signal", "drive_field_v1", "A_reward", 1, {"desc":"reward/high channel (explicit matrix; protected axes enforced)"}))
        if self.reg.get("daydream") is None:
            self.reg.upsert(AdapterBinding("daydream", "drive_field_v1", "A_daydream", 1, {"desc":"daydream drives -> state"}))
        if self.reg.get("websense") is None:
            self.reg.upsert(AdapterBinding("websense", "websense_v1", "A_websense", 1, {"desc":"websense -> state"}))
        if self.reg.get("resources") is None:
            self.reg.upsert(AdapterBinding("resources", "drive_field_v1", "A_resources", 1, {"desc":"resources drives -> state"}))
        if self.reg.get("health") is None:
            self.reg.upsert(AdapterBinding("health", "drive_field_v1", "A_health", 1, {"desc":"health drives -> state"}))
        if self.reg.get("evolve") is None:
            self.reg.upsert(AdapterBinding("evolve", "drive_field_v1", "A_evolve", 1, {"desc":"evolve drives -> state"}))

        if self.reg.get("epistemic") is None:
            self.reg.upsert(AdapterBinding("epistemic", "drive_field_v1", "A_epistemic", 1, {"desc":"epistemic sensor drives -> state"}))

        # Seed a first self-model snapshot so tables aren't empty even before evolve runs.
        try:
            con = self.db.connect()
            try:
                row = con.execute("SELECT COUNT(*) AS c FROM self_model").fetchone()
                c = int((row["c"] if row is not None else 0) or 0)
            finally:
                con.close()
            if c == 0:
                sm = build_self_model(
                    app_dir=os.path.dirname(__file__),
                    axes=list(self.axis.keys()),
                    adapters=self.reg.list_bindings(),
                    matrices=self.store.list_matrices(),
                    model_cfg={
                        "speech": getattr(self.cfg, "model", ""),
                        "decider": getattr(self.decider_cfg, "model", ""),
                        "daydream": getattr(self.daydream_cfg, "model", ""),
                        "feedback": getattr(self.feedback_cfg, "model", ""),
                        "selfeval": getattr(self.selfeval_cfg, "model", ""),
                        "evolve": getattr(self.evolve_cfg, "model", ""),
                    },
                )
                db_add_self_model(self.db, sm)
        except Exception:
            pass

    def _repo_root(self) -> str:
        """Repository root used by DevLab test runner / patch sandbox."""
        try:
            here = os.path.abspath(os.path.dirname(__file__))  # app/
            cand = os.path.abspath(os.path.join(here, ".."))   # repo/
            for _ in range(6):
                if os.path.exists(os.path.join(cand, "pyproject.toml")):
                    return cand
                cand = os.path.abspath(os.path.join(cand, ".."))
            return os.getcwd()
        except Exception:
            return os.getcwd()

    def _state_summary(self) -> str:
        s = self.hb.load_state()
        inv = {v: k for k, v in self.axis.items()}
        named = {inv[i]: s.values[i] for i in range(len(s.values)) if i in inv}
        keys = [
            "pain","energy","stress","curiosity","confidence","uncertainty","freshness_need","social_need","urge_reply","urge_share",
            "pressure_websense","pressure_daydream","pressure_evolve","capability_gap","desire_upgrade",
            "purpose_a1","purpose_a2","purpose_a3","purpose_a4",
            "tension_a1","tension_a2","tension_a3","tension_a4",
        ]
        parts = []
        for k in keys:
            if k in named:
                parts.append(f"{k}={max(0.0, min(1.0, float(named[k]))):.2f}")
        return ", ".join(parts)

    def _format_websense_answer(self, *, question: str, claims: Dict[str, Any], fallback_urls: List[str] | None = None) -> str:
        """Deterministic grounded reply builder.

        When WebSense runs we must not hallucinate. The speech organ can still be used
        for style later, but correctness comes from this structured claims JSON.
        """
        fallback_urls = fallback_urls or []

        def _dedupe(urls: List[str]) -> List[str]:
            out: List[str] = []
            seen = set()
            for u in urls:
                uu = str(u or '').strip()
                if not uu:
                    continue
                if uu in seen:
                    continue
                seen.add(uu)
                out.append(uu)
            return out

        claims_list = claims.get('claims') if isinstance(claims.get('claims'), list) else []
        miss = claims.get('missing') if isinstance(claims.get('missing'), list) else []
        try:
            unc = float(claims.get('uncertainty', 0.7) or 0.7)
        except Exception:
            unc = 0.7
        unc = 0.0 if unc < 0.0 else 1.0 if unc > 1.0 else unc

        srcs: List[str] = []
        for c in (claims_list or [])[:8]:
            if not isinstance(c, dict):
                continue
            sup = c.get('support') if isinstance(c.get('support'), list) else []
            for u in sup[:2]:
                srcs.append(str(u or '').strip())
        srcs = _dedupe(srcs + fallback_urls)

        if not claims_list:
            lines = [
                "Ich konnte das gerade nicht belastbar verifizieren: WebSense hat keine grounded Claims extrahiert.",
            ]
            if srcs:
                lines.append("Quellen (Auswahl):")
                for u in srcs[:5]:
                    lines.append(f"- {u}")
            if miss:
                lines.append("Offen / fehlt: " + ", ".join([str(x) for x in miss[:4] if str(x).strip()]))
            lines.append("Wenn du willst, starte ich sofort eine präzisere Suche (z. B. englisch + offizielle Quellen priorisieren).")
            return "\n".join([ln for ln in lines if ln]).strip()

        # With claims: emit a compact, bullet-grounded answer.
        ans = str(claims.get('answer') or '').strip()
        if not ans:
            try:
                ans = str((claims_list[0] or {}).get('text') or '').strip()
            except Exception:
                ans = ''

        lines: List[str] = []
        if ans:
            lines.append(ans)

        # Key claims
        for c in (claims_list or [])[:5]:
            if not isinstance(c, dict):
                continue
            txt = str(c.get('text') or '').strip()
            if not txt:
                continue
            u0 = ''
            sup = c.get('support') if isinstance(c.get('support'), list) else []
            if sup:
                u0 = str(sup[0] or '').strip()
            if u0:
                lines.append(f"- {txt} (Quelle: {u0})")
            else:
                lines.append(f"- {txt}")

        if unc >= 0.55:
            lines.append(f"Hinweis: Unsicherheit ist noch relativ hoch ({unc:.2f}).")
        if miss:
            lines.append("Offen / fehlt: " + ", ".join([str(x) for x in miss[:4] if str(x).strip()]))
        return "\n".join([ln for ln in lines if ln]).strip()

    
    def _meta_int(self, key: str, default: int = 0) -> int:
        try:
            return int(db_meta_get(self.db, key, str(default)) or default)
        except Exception:
            return int(default)

    def _meta_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(db_meta_get(self.db, key, str(default)) or default)
        except Exception:
            return float(default)

    def _set_dispute_lock(self, seconds: float = 3600.0) -> None:
        # Generic epistemic lock: after explicit negative feedback (caught),
        # do not allow the system to become *more* certain until it has grounded evidence.
        now = float(time.time())
        try:
            db_meta_set(self.db, "last_dispute_at", str(now))
            db_meta_set(self.db, "dispute_lock_until", str(now + float(seconds)))
        except Exception:
            pass

    def _dispute_lock_active(self) -> bool:
        # Deprecated: replaced by learnable epistemic error_signal channel.
        return False

    def _enforce_dispute_floor(self, *, reason: str = "") -> None:
        # Deprecated: replaced by learnable epistemic error_signal channel.
        return

    def __getattr__(self, name: str):
        # Compatibility fallback: if _ui_message is missing due to refactors,
        # provide a safe implementation so the UI doesn't crash.
        if name == "_ui_message":
            return lambda mid: _db_fetch_message_dict(self.db, int(mid))
        raise AttributeError(name)

    def _handle_slash_command(self, text: str) -> str | None:
        """Handle main-branch compatible slash commands.

        Currently supported:
          /proposal            -> list recent mutation proposals
          /proposal <id>       -> show proposal details
          /proposal help       -> usage

        Returns reply text if handled, otherwise None.
        """
        t = (text or "").strip()
        if not t.startswith("/"):
            return None

        parts = t.split()
        cmd = parts[0].lower()
        if cmd not in ("/proposal", "/proposals"):
            return None

        if len(parts) == 1 or (len(parts) >= 2 and parts[1].lower() in ("list", "ls")):
            items = db_list_mutation_proposals(self.db, limit=12)
            if not items:
                return "No proposals yet. (Idle evolve will create them when pressure_evolve is high.)"
            lines = ["Proposals (newest last):"]
            for it in items:
                pid = it.get("id")
                status = it.get("status", "")
                trig = it.get("trigger", "")
                created = it.get("created_at", "")
                title = ""
                pj = it.get("proposal") or {}
                if isinstance(pj, dict):
                    title = str(pj.get("title") or pj.get("name") or "").strip()
                if not title:
                    title = "(no title)"
                lines.append(f"- #{pid} [{status}] {title} | trigger={trig} | {created}")
            lines.append("\nUse: /proposal <id> to inspect details.")
            return "\n".join(lines)

        if len(parts) >= 2 and parts[1].lower() in ("help", "h", "?"):
            return "Usage:\n  /proposal            list proposals\n  /proposal <id>       show details"

        # detail view
        try:
            pid = int(parts[1])
        except Exception:
            return "Invalid proposal id. Use: /proposal or /proposal <id>"
        p = db_get_mutation_proposal(self.db, pid)
        if p is None:
            return f"Proposal #{pid} not found."
        # pretty print with sane length
        pj = p.get("proposal")
        body = json.dumps(pj, ensure_ascii=False, indent=2)
        if len(body) > 6000:
            body = body[:6000] + "\n... (truncated)"
        return (
            f"Proposal #{p['id']} [{p['status']}]\n"
            f"created_at: {p['created_at']}\n"
            f"trigger: {p['trigger']}\n"
            f"user_note: {p.get('user_note','')}\n\n"
            f"proposal_json:\n{body}"
        )

    def _apply_matrix_update(
        self,
        event_type: str,
        desired_state_delta: Dict[str, float],
        input_features: Dict[str, float],
        reward: float,
    ) -> None:
        """Generic plasticity: low-rank outer-product update to the matrix bound to event_type.

        Core idea (Event -> Matrix -> State): A_event maps an event encoding x into a state delta y.
        Learning therefore updates A using u (desired state delta) and x (the event input features).

        Update: ΔA_ij += eta * reward * u_i * x_j

        Stability:
        - lightweight L2 decay on existing weights (prevents drift)
        - Frobenius-norm clamp (prevents explosion)
        - per-entry clamp (prevents single outliers)
        """
        t0 = time.time()
        status: Dict[str, Any] = {"event_type": str(event_type or ""), "ok": True, "moved": False}
        try:
            binding = self.reg.get(event_type)
            if binding is None:
                status["reason"] = "no_binding"
                return

            # Snapshot current pain for regression attribution/rollback.
            pain_before = 0.0
            try:
                s0 = self.hb.load_state()
                # IMPORTANT: rollback safety should be based on psych/teleological regressions,
                # not physical latency/energy fluctuations.
                idx_p = self.axis.get("pain_psych")
                if idx_p is None:
                    idx_p = self.axis.get("pain")
                if idx_p is not None and idx_p < len(s0.values):
                    pain_before = float(s0.values[idx_p])
            except Exception:
                pain_before = 0.0

            # Protected axes must NOT be learnable (cannot be "trained down" via plasticity).
            # Invariants:
            # - No non-health/resources matrix may write into or read from these axes.
            # - Their diagonal must be identity (1.0) for all non-health/resources matrices.
            protected_names = {"pain", "energy", "fatigue", "sleep_pressure"}
            protected_idx = {self.axis.get(n) for n in protected_names if self.axis.get(n) is not None}
            protected_idx.discard(None)
            allow_cross = event_type in ("health", "resources")

            # Build sparse u (desired state delta) and x (event input features) over axis indices.
            u_idx: Dict[int, float] = {}
            for k, v in (desired_state_delta or {}).items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                idx = self.axis.get(str(k))
                if idx is None:
                    continue
                fv = max(-1.0, min(1.0, fv))
                if abs(fv) < 1e-9:
                    continue
                u_idx[idx] = fv

            v_idx: Dict[int, float] = {}
            for k, v in (input_features or {}).items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                idx = self.axis.get(str(k))
                if idx is None:
                    continue
                fv = max(-1.0, min(1.0, fv))
                if abs(fv) < 1e-9:
                    continue
                v_idx[idx] = fv

            if not u_idx:
                status["reason"] = "empty_u"
                return
            if not v_idx:
                status["reason"] = "empty_x"
                return

            # Keep updates sparse and numerically stable: only use top-k signals.
            topk_u = int(os.environ.get("BUNNY_PLASTICITY_TOPK_U", "12") or 12)
            topk_x = int(os.environ.get("BUNNY_PLASTICITY_TOPK_X", "12") or 12)
            u_items = sorted(u_idx.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:max(1, topk_u)]
            v_items = sorted(v_idx.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:max(1, topk_x)]

            # --- learning hyperparams (keep small on 8GB VRAM systems) ---
            # NOTE: decision deltas are typically larger than other organs' deltas.
            # Use a slightly higher eta for auxiliary organs to avoid vanishing updates.
            eta = 0.03 if event_type == "decision" else 0.08
            l2_decay = 0.001        # global decay on weights
            frob_tau = 5.0          # Frobenius norm clamp
            max_abs = 2.0           # per-entry clamp

            r = abs(max(-1.0, min(1.0, float(reward))))

            # compute sparse delta entries
            delta_entries = []
            delta_frob = 0.0
            # Numeric anti-vanishing: if both signals are tiny, scale them to a stable floor.
            u_max = max(abs(float(v)) for _i, v in u_items) if u_items else 0.0
            v_max = max(abs(float(v)) for _j, v in v_items) if v_items else 0.0
            scale_uv = 1.0
            floor = 0.05
            if u_max > 0.0 and u_max < floor:
                scale_uv *= (floor / u_max)
            if v_max > 0.0 and v_max < floor:
                scale_uv *= (floor / v_max)

            for i, ui_v in u_items:
                for j, vj_v in v_items:
                    if not allow_cross and (i in protected_idx or j in protected_idx):
                        continue
                    d = eta * r * float(scale_uv) * float(ui_v) * float(vj_v)
                    delta_entries.append((i, j, float(d)))
                    delta_frob += float(d) * float(d)

            delta_frob = math.sqrt(delta_frob) if delta_frob > 0.0 else 0.0

            if not delta_entries:
                status["reason"] = "no_delta_entries"
                return

            # Fetch current matrix
            A = self.store.get_sparse(binding.matrix_name, int(binding.matrix_version))

            # merge existing entries + decay
            cur: Dict[Tuple[int, int], float] = {}
            for (i, j, val) in getattr(A, "entries", []) or []:
                cur[(int(i), int(j))] = float(val) * (1.0 - l2_decay)

            # apply deltas (this is where off-diagonals naturally grow)
            for i, j, dv in delta_entries:
                key = (int(i), int(j))
                cur[key] = float(cur.get(key, 0.0) + dv)

            # Enforce invariants for protected axes (prevents matrix from affecting pain/energy).
            if not allow_cross and protected_idx:
                for (i, j) in list(cur.keys()):
                    if int(i) in protected_idx or int(j) in protected_idx:
                        del cur[(i, j)]
                for p in protected_idx:
                    cur[(int(p), int(p))] = 1.0

            # per-entry clamp
            for k in list(cur.keys()):
                v = cur[k]
                if v > max_abs:
                    cur[k] = max_abs
                elif v < -max_abs:
                    cur[k] = -max_abs

            # Frobenius clamp (keeps diffusion possible, prevents runaway growth)
            frob = math.sqrt(sum(v*v for v in cur.values())) if cur else 0.0
            if frob > frob_tau and frob > 1e-9:
                scale = frob_tau / frob
                for k in list(cur.keys()):
                    cur[k] = float(cur[k] * scale)

            new_version = int(binding.matrix_version) + 1
            self.store.put_sparse(
                binding.matrix_name,
                new_version,
                A.n_rows,
                A.n_cols,
                [(i, j, v) for (i, j), v in cur.items()],
                meta={
                    "plasticity": True,
                    "from_version": int(binding.matrix_version),
                    "updated_at": now_iso(),
                    "eta": eta,
                    "l2_decay": l2_decay,
                    "frob_tau": frob_tau,
                    "max_abs": max_abs,
                },
            )

            # point adapter to new version
            binding2 = AdapterBinding(
                event_type=binding.event_type,
                encoder_name=binding.encoder_name,
                matrix_name=binding.matrix_name,
                matrix_version=new_version,
                meta=binding.meta,
            )
            self.reg.upsert(binding2)

            # audit log (used by pain model)
            try:
                db_add_matrix_update_log(
                    self.db,
                    event_type=event_type,
                    matrix_name=binding.matrix_name,
                    from_version=int(binding.matrix_version),
                    to_version=int(new_version),
                    reward=float(r),
                    delta_frob=float(delta_frob),
                    pain_before=float(pain_before),
                    pain_after=0.0,
                    notes="plasticity_update",
                )
            except Exception:
                pass
            status.update({
                "ok": True,
                "moved": True,
                "matrix": f"{binding.matrix_name}@{new_version}",
                "delta_frob": float(delta_frob),
                "eta": float(eta),
                "scale": float(scale_uv),
                "u_n": int(len(u_items)),
                "x_n": int(len(v_items)),
            })
        except Exception:
            status["ok"] = False
            status["reason"] = "exception"
            return
        finally:
            # Always log plasticity attempts for observability (why a matrix did/didn't move).
            try:
                ms = max(0.0, (time.time() - float(t0)) * 1000.0)
                ok = bool(status.get("ok"))
                err = str(status.get("reason") or "")
                db_add_health_log(self.db, f"plasticity_{event_type}", ok, ms, err, metrics=status)
            except Exception:
                pass
    def process_user_text(self, text: str) -> Tuple[int, int]:
        """Returns (user_msg_id, reply_msg_id)."""
        with self._lock:
            # Turn anchor for salience-based learning and memory stickiness.
            turn_start_iso = now_iso()
            s_before = self.hb.load_state().copy()
            user_id = db_add_message(self.db, "user", text)
            self.broker.publish("message", self._ui_message(user_id))

            # Main-branch compatible slash commands (fast, no LLM needed)
            cmd_reply = self._handle_slash_command(text)
            if cmd_reply is not None:
                reply_id = db_add_message(self.db, "assistant", cmd_reply)
                self.broker.publish("message", self._ui_message(reply_id))
                return user_id, reply_id

            # Lite mode: single-call chat (speech only). This avoids multi-organ
            # LLM pipelines that can hard-freeze CPU-only machines.
            if getattr(self, "lite_mode", False):
                # IMPORTANT: internal state is context only; Bunny must NEVER print it.
                sys_prompt = (
                    "You are Bunny, a digital organism. Answer the USER in natural, friendly language. "
                    "Do NOT output logs, JSON, tags, INTERNAL_STATE, MOOD, scores, weights, or any debugging. "
                    "Never write 'INTERNAL_STATE', 'MOOD', 'updated', or similar. "
                    "If the user greets you, greet back. Otherwise respond directly. "
                    "If you truly need more information, ask exactly ONE clarifying question."
                )
                mood_obj = project_mood(self.hb.load_state(), self.axis).to_dict()
                user_prompt = (
                    "<INTERNAL_STATE>\n" + self._state_summary() + "\n</INTERNAL_STATE>\n" +
                    "<MOOD>\n" + json.dumps(mood_obj or {}, ensure_ascii=False) + "\n</MOOD>\n\n" +
                    "USER: " + text + "\n\nASSISTANT:"
                )
                try:
                    out = self._call_with_health(
                        "speech",
                        lambda: ollama_chat(self.cfg, sys_prompt, user_prompt),
                        metrics={
                            "model": self.cfg.model,
                            "ctx": int(self.cfg.num_ctx),
                            "lite": True,
                            "sys_chars": len(sys_prompt),
                            "user_chars": len(user_prompt),
                        },
                    )
                except Exception as e:
                    out = f"(speech organ error: {e})"

                # Output hygiene: if the model leaks internal context, replace with a normal reply.
                bad = ("INTERNAL_STATE" in out) or ("MOOD" in out) or out.strip().startswith("{") or out.strip().startswith("[")
                if bad:
                    out = "Sag mir kurz, worum es geht – nur kurz quatschen oder ein konkretes Thema?"

                if not out.strip():
                    out = "Ich bin da. Sag mir kurz, worum es geht."
                reply_id = db_add_message(self.db, "reply", out)
                self.broker.publish("message", self._ui_message(reply_id))
                self.hb.enqueue(Event("speech_outcome", {"len": len(out), "lite": True}).with_time())
                return user_id, reply_id

            # --- Topic anchoring (reduces drift; AI-decided, not keyword heuristics) ---
            try:
                ws_items = db_get_workspace_current(self.db)
                prev_topic = ''
                for it in ws_items:
                    if isinstance(it, dict) and it.get('kind') == 'topic' and it.get('active_topic'):
                        prev_topic = str(it.get('active_topic') or '')
                        break
                td = self._call_with_health(
                    'topic',
                    lambda: detect_active_topic(self.topic_cfg, text, prev_topic, self._state_summary()),
                )
                active_topic = str(td.get('active_topic') or prev_topic or 'Allgemein').strip()[:80]
                conf = float(td.get('confidence', 0.5) or 0.5)
                # update workspace topic item
                ws_items = [it for it in (ws_items or []) if not (isinstance(it, dict) and it.get('kind') == 'topic')]
                ws_items.insert(0, {'kind': 'topic', 'active_topic': active_topic, 'confidence': conf})
                # Prevent topic-bleeding: on a confident topic switch, clear old workspace evidence.
                # (Generic housekeeping; the topic itself is AI-decided.)
                if prev_topic and active_topic and active_topic != prev_topic and conf >= 0.70:
                    ws_items = [ws_items[0]]
                db_set_workspace_current(self.db, ws_items, note='topic')
                db_upsert_topic(self.db, active_topic, weight_delta=0.02 * conf)
                db_open_episode_if_needed(self.db, active_topic)
            except Exception:
                pass

            # --- Belief extraction (runs on every user message; generic memory) ---
            try:
                ws_items2 = db_get_workspace_current(self.db)
                active_topic2 = _get_active_topic(ws_items2)
                be = self._call_with_health(
                    "beliefs",
                    lambda: extract_user_beliefs(
                        self.beliefs_cfg,
                        db_get_axioms(self.db),
                        self._state_summary(),
                        text,
                        active_topic=active_topic2,
                        workspace=ws_items2,
                        needs=db_get_needs_current(self.db),
                        wishes=db_get_wishes_current(self.db),
                    ),
                )
                for b in (be.get("beliefs") or []):
                    if not isinstance(b, dict):
                        continue
                    db_add_belief(
                        self.db,
                        str(b.get("subject") or ""),
                        str(b.get("predicate") or ""),
                        str(b.get("object") or ""),
                        float(b.get("confidence", 0.75) or 0.75),
                        str(b.get("provenance") or "user_utterance"),
                    )
            except Exception:
                pass


            fb_ctx = ""
            fb_last = ""

            # --- Feedback interpreter (child-like correction learning; LLM-based) ---
            try:
                last_assistant = db_get_last_assistant_text(self.db)
                last_assistant_id = 0
                try:
                    con2 = self.db.connect()
                    try:
                        row2 = con2.execute("SELECT id FROM ui_messages WHERE kind IN ('reply','assistant','auto') ORDER BY id DESC LIMIT 1").fetchone()
                        last_assistant_id = int(row2["id"]) if row2 and row2.get("id") is not None else 0
                    finally:
                        con2.close()
                except Exception:
                    last_assistant_id = 0
                fb_last = last_assistant or ""
                fb = self._call_with_health(
                    "feedback",
                    lambda: interpret_feedback(
                        self.feedback_cfg,
                        db_get_axioms(self.db),
                        self._state_summary(),
                        last_assistant,
                        text,
                    ),
                )
                # Apply drive deltas even when feedback classifier is uncertain.
                # We only *store* feedback as training data when is_feedback is high.
                try:
                    is_fb = float(fb.get('is_feedback', 0.0) or 0.0)
                except Exception:
                    is_fb = 0.0

                drive_delta: Dict[str, Any] = fb.get("desired_drive_delta") if isinstance(fb.get("desired_drive_delta"), dict) else {}
                if drive_delta:
                    # Apply intended next-step drive corrections (delta-mode).
                    self.hb.enqueue(Event("decision", {"drives": drive_delta, "_mode": "delta"}).with_time())
                    self.hb.step()

                # --- Teacher organ hints (user can steer organ usage via text; learned, no keyword lists) ---
                try:
                    organ_hints = fb.get('organ_hints') if isinstance(fb.get('organ_hints'), dict) else {}
                except Exception:
                    organ_hints = {}
                if organ_hints:
                    # Immediate state nudge: hints increase the corresponding pressure axes (delta-mode).
                    hint_drives: Dict[str, float] = {}
                    try:
                        if float(organ_hints.get('websense', 0.0) or 0.0) > 0.0:
                            hint_drives['pressure_websense'] = float(organ_hints.get('websense') or 0.0) * 0.8
                        if float(organ_hints.get('daydream', 0.0) or 0.0) > 0.0:
                            hint_drives['pressure_daydream'] = float(organ_hints.get('daydream') or 0.0) * 0.6
                        if float(organ_hints.get('evolve', 0.0) or 0.0) > 0.0:
                            hint_drives['pressure_evolve'] = float(organ_hints.get('evolve') or 0.0) * 0.6
                    except Exception:
                        hint_drives = {}
                    if hint_drives:
                        self.hb.enqueue(Event('decision', {'drives': hint_drives, '_mode': 'delta'}).with_time())
                        self.hb.step()

                    # Supervised policy update: if the user hints an organ, reward selecting it under current state features.
                    try:
                        pol2 = self.policy.predict(self.hb.load_state().values)
                        x_pol2 = pol2.get('features') if isinstance(pol2.get('features'), list) else []
                        v_pol2 = int(pol2.get('version', 1) or 1)
                        if x_pol2:
                            for org, w in (organ_hints or {}).items():
                                try:
                                    ww = float(w or 0.0)
                                except Exception:
                                    continue
                                if ww <= 0.0:
                                    continue
                                v_pol2 = self.policy.apply_update(from_version=v_pol2, x=x_pol2, action=str(org), reward=float(0.35 * ww), note='teacher_hint')
                    except Exception:
                        pass

                    # Persist a compact coaching trace into short memory (helps the decider avoid loops)
                    try:
                        conh = self.db.connect()
                        try:
                            nowh = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                            conh.execute(
                                "INSERT INTO memory_short(role,content,created_at) VALUES(?,?,?)",
                                (
                                    'teacher',
                                    ('ORGAN_HINTS: ' + json.dumps(organ_hints, ensure_ascii=False)).strip()[:600],
                                    nowh,
                                ),
                            )
                            conh.commit()
                        finally:
                            conh.close()
                    except Exception:
                        pass

                # Negative feedback (as text) becomes an epistemic error signal.
                # This does NOT hardcode domain keywords; it is a generic error-sensor channel.
                try:
                    dr_fb = float(fb.get("delta_reward", 0.0) or 0.0)
                except Exception:
                    dr_fb = 0.0
                err_sig = 0.0
                if is_fb >= 0.6 and float(dr_fb) < -0.15:
                    err_sig = max(0.0, min(1.0, abs(float(dr_fb))))
                    # Remove the now-known-wrong last assistant message from short-memory context to avoid anchoring loops.
                    if int(locals().get("last_assistant_id", 0) or 0) > 0:
                        try:
                            db_caught_message(self.db, int(last_assistant_id))
                            con3 = self.db.connect()
                            try:
                                con3.execute("DELETE FROM memory_short WHERE ui_message_id=?", (int(last_assistant_id),))
                                con3.commit()
                            finally:
                                con3.close()
                        except Exception:
                            pass

                if err_sig > 0.0:
                    x_ep = {"error_signal": float(err_sig)}
                    u_ep = fb.get("desired_state_delta") if isinstance(fb.get("desired_state_delta"), dict) else {}
                    # Learn the epistemic coupling (A_epistemic) from user corrections.
                    if u_ep:
                        self._apply_matrix_update("epistemic", u_ep, x_ep, float(err_sig))

                    # Apply the error signal into state via the epistemic channel.
                    s_pre = self.hb.load_state().copy()
                    idxu = self.axis.get("uncertainty")
                    u_pre = float(s_pre.values[idxu]) if idxu is not None and idxu < len(s_pre.values) else 0.0
                    self.hb.enqueue(Event("epistemic", {"drives": x_ep, "_mode": "delta"}).with_time())
                    self.hb.step()

                    # Epistemic monotonicity constraint: explicit negative feedback must not reduce uncertainty.
                    if idxu is not None:
                        s_post = self.hb.load_state()
                        u_post = float(s_post.values[idxu]) if idxu < len(s_post.values) else u_pre
                        if float(u_post) < float(u_pre):
                            s_post.values[idxu] = float(u_pre)
                            self.hb.save_state(s_post, [])

                if is_fb >= 0.6 and last_assistant:
                    db_add_feedback(self.db, text, last_assistant, fb)

                    # Persist extracted beliefs (generic structured learning)
                    for b in (fb.get("beliefs") or []):
                        if not isinstance(b, dict):
                            continue
                        db_add_belief(
                            self.db,
                            str(b.get("subject") or ""),
                            str(b.get("predicate") or ""),
                            str(b.get("object") or ""),
                            float(b.get("confidence", 0.75) or 0.75),
                            str(b.get("provenance") or "user_feedback"),
                        )

                    # persist a compact correction trace into short memory (acts as immediate self-correction context)
                    try:
                        con = self.db.connect()
                        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                        con.execute(
                            "INSERT INTO memory_short(role,content,created_at) VALUES(?,?,?)",
                            (
                                "teacher",
                                ("USER_FEEDBACK: " + (fb.get("notes") or "")).strip()[:1200],
                                now,
                            ),
                        )
                        con.commit()
                    finally:
                        try:
                            con.close()
                        except Exception:
                            pass

                    fb_ctx = "\n\nUSER_FEEDBACK_JSON:\n" + json.dumps(fb, ensure_ascii=False)
                    fb_ctx += "\n\nLAST_ASSISTANT_FOR_FEEDBACK:\n" + (last_assistant or "")[:1600]


                    # plasticity: update decision matrix (primary coupling point)
                    # Use the *desired drive delta* as x (must match what was applied / would be applied).
                    x_feats = fb.get("desired_drive_delta") if isinstance(fb.get("desired_drive_delta"), dict) else {}
                    try:
                        r_pl = float(fb.get("delta_reward", 0.0) or 0.0)
                    except Exception:
                        r_pl = 0.0
                    u_pl = fb.get("desired_state_delta") if isinstance(fb.get("desired_state_delta"), dict) else {}
                    if u_pl and (x_feats or {}) and abs(r_pl) > 1e-9:
                        self._apply_matrix_update("decision", u_pl, x_feats or {}, abs(float(r_pl)))

                    # learned trust updates (domains) derived from feedback interpreter
                    for dom in (fb.get("domains_penalty") or []):
                        db_update_domain_trust(
                            self.db,
                            str(dom),
                            -abs(float(fb.get("delta_reward", -0.3) or -0.3)),
                        )
                    for dom in (fb.get("domains_reward") or []):
                        db_update_domain_trust(
                            self.db,
                            str(dom),
                            abs(float(fb.get("delta_reward", 0.3) or 0.3)),
                        )
            except Exception:
                pass

            # events: user utterance
            self.hb.enqueue(Event("user_utterance", {"text": text}).with_time())

            # resources tick -> drives into state (energy/stress budget)
            try:
                metrics = collect_resources()
                db_add_resources_log(self.db, metrics)
                e_t = float(metrics.get("energy", 0.5) or 0.5)
                s_t = float(metrics.get("stress", 0.5) or 0.5)
                # Generic affect baseline derived from resource budget signals.
                # All targets are bounded to [0,1] (persisted state invariant).
                targets = {
                    "energy": max(0.0, min(1.0, e_t)),
                    "stress": max(0.0, min(1.0, s_t)),
                    "arousal": max(0.0, min(1.0, s_t)),
                    "security": max(0.0, min(1.0, 1.0 - s_t)),
                    "valence": max(0.0, min(1.0, e_t)),
                    "frustration": max(0.0, min(1.0, s_t)),
                }
                self.hb.enqueue(Event("resources", self._targets_to_delta_payload(targets, eta=self.eta_measure)).with_time())
            except Exception:
                pass

            # --- LLM decider: pressures/actions ---
            try:
                beliefs = db_list_beliefs(self.db, limit=12)
                # Deterministic action prior (trainable policy kernel).
                try:
                    pol = self.policy.predict(self.hb.load_state().values)
                    policy_hint = pol.get('probs') or {}
                    policy_trace = {'version': int(pol.get('version') or 1), 'features': pol.get('features') or []}
                except Exception:
                    policy_hint = {}
                    policy_trace = {'version': 1, 'features': []}
                decision = self._call_with_health(
                    "decider",
                    lambda: decide_pressures(
                        self.decider_cfg,
                        db_get_axioms(self.db),
                        self._state_summary(),
                        text,
                        beliefs,
                        scope="user",
                        workspace=db_get_workspace_current(self.db),
                        needs=db_get_needs_current(self.db),
                        wishes=db_get_wishes_current(self.db),
                        self_report=build_self_report(self.hb.load_state(), self.axis).to_dict(),
                        active_topic=_get_active_topic(db_get_workspace_current(self.db)),
                        policy_hint=policy_hint,
                    ),
                )
            except Exception as e:
                policy_trace = {'version': 1, 'features': []}
                decision = {
                    "drives": {"uncertainty": 0.05, "stress": 0.05},
                    "actions": {"websense": 0.0, "daydream": 0.0, "reply": 1.0},
                    "web_query": "",
                    "notes": f"decider error: {e}",
                }

            db_add_decision(self.db, "user", text, decision)
            drives = decision.get("drives") if isinstance(decision.get("drives"), dict) else {}
            actions = decision.get("actions") if isinstance(decision.get("actions"), dict) else {}

            # Contract normalization (user turns): if the decider provides an ACTION but
            # omits the corresponding pressure drive, promote the action into the drive.
            # This keeps control AI-driven while preventing long-run pressure collapse.
            try:
                ws_a = float(actions.get("websense", 0.0) or 0.0)
            except Exception:
                ws_a = 0.0
            try:
                dd_a = float(actions.get("daydream", 0.0) or 0.0)
            except Exception:
                dd_a = 0.0
            try:
                ev_a = float(actions.get("evolve", 0.0) or 0.0)
            except Exception:
                ev_a = 0.0
            try:
                if ws_a > 0.0:
                    prev = float(drives.get("pressure_websense", 0.0) or 0.0)
                    drives["pressure_websense"] = max(prev, ws_a)
                if dd_a > 0.0:
                    prev = float(drives.get("pressure_daydream", 0.0) or 0.0)
                    drives["pressure_daydream"] = max(prev, dd_a)
                if ev_a > 0.0:
                    prev = float(drives.get("pressure_evolve", 0.0) or 0.0)
                    drives["pressure_evolve"] = max(prev, ev_a)
            except Exception:
                pass

            # integrate drives into state via decision event
            # IMPORTANT: drives are target levels; we convert to deltas first and keep the actual
            # delta vector for consistent learning (matrix updates must use the same x that was applied).
            decision_payload = self._targets_to_delta_payload(drives, eta=self.eta_drives)
            decision_delta = decision_payload.get("drives") if isinstance(decision_payload, dict) else {}
            self.hb.enqueue(Event("decision", decision_payload).with_time())
            self.hb.step()
            self.broker.publish("status", db_status(self.db))

            ws_context = ""
            ws_claims_json = ""
            ws_claims_obj: Dict[str, Any] = {}
            ws_urls: List[str] = []

            # WebSense gating purely from epistemic signals
            s_now = self.hb.load_state()
            idx_unc = self.axis.get("uncertainty")
            u_now = float(s_now.values[idx_unc]) if idx_unc is not None and idx_unc < len(s_now.values) else 0.0
            idx_fresh = self.axis.get("freshness_need")
            f_now = float(s_now.values[idx_fresh]) if idx_fresh is not None and idx_fresh < len(s_now.values) else 0.0

            # Sanitize bounded axes in persisted state (old DBs may contain -1..1 values).
            dirty = False
            if idx_unc is not None and idx_unc < len(s_now.values):
                cu = 0.0 if u_now < 0.0 else 1.0 if u_now > 1.0 else u_now
                if cu != u_now:
                    s_now.values[idx_unc] = cu
                    u_now = cu
                    dirty = True
            if idx_fresh is not None and idx_fresh < len(s_now.values):
                cf = 0.0 if f_now < 0.0 else 1.0 if f_now > 1.0 else f_now
                if cf != f_now:
                    s_now.values[idx_fresh] = cf
                    f_now = cf
                    dirty = True
            idx_conf = self.axis.get("confidence")
            if idx_conf is not None and idx_conf < len(s_now.values):
                c_now = float(s_now.values[idx_conf])
                cc = 0.0 if c_now < 0.0 else 1.0 if c_now > 1.0 else c_now
                if cc != c_now:
                    s_now.values[idx_conf] = cc
                    dirty = True
            if dirty:
                # Persist sanitized state so future ticks start in a valid range.
                self.hb.save_state(s_now, [])
            ws_action = float(((decision.get("actions") or {}).get("websense") or 0.0))

            def _cl01(x: float) -> float:
                return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

            # WebSense trigger is purely epistemic:
            # - evidence is needed when uncertainty is high AND confidence is low
            # - freshness_need can trigger websense for time-sensitive facts
            # - decider may explicitly request websense
            # All bounded axes live in [0,1]; never use abs().
            u_now = _cl01(u_now)
            f_now = _cl01(f_now)
            ws_action = _cl01(ws_action)

            idx_conf2 = self.axis.get("confidence")
            c_now2 = float(s_now.values[idx_conf2]) if (idx_conf2 is not None and idx_conf2 < len(s_now.values)) else 0.0
            c_now2 = _cl01(c_now2)

            idx_pws = self.axis.get("pressure_websense")
            pws_now = float(s_now.values[idx_pws]) if (idx_pws is not None and idx_pws < len(s_now.values)) else 0.0
            pws_now = _cl01(pws_now)

            score_ws0 = float(max(ws_action, f_now, pws_now))

            # WebSense gating:
            # - if the score crosses the configured threshold, run deterministically
            # - otherwise sample (Bernoulli) with p=score (AI-like scheduling)
            force_web = bool(score_ws0 >= float(self.th_websense))
            rand_ws = float(self._rng.random())
            try:
                self._last_gate_rand['websense'] = rand_ws
            except Exception:
                pass
            want_web = bool(force_web or (rand_ws < score_ws0))

            # Ops trace: record gate computation (helps debug "why WebSense ran")
            try:
                score_ws = float(score_ws0)
                db_add_organ_gate_log(
                    self.db,
                    phase="user",
                    organ="websense",
                    score=score_ws,
                    threshold=float(self.th_websense),
                    want=bool(want_web),
                    data={"ws_action": float(ws_action), "freshness_need": float(f_now), "pressure_websense": float(pws_now), "confidence": float(c_now2), "uncertainty": float(u_now), "rand": float(rand_ws), "force": int(force_web)},
                )
            except Exception:
                pass


            query = (decision.get("web_query") or "").strip()
            if query.lower() in ("search for user query", "direct_response_to_user_question"):
                query = ""
            # Avoid heuristic fallbacks like "use full user text as query".
            # If WebSense is desired but no query exists, let the model propose one.
            if want_web and (not query):
                try:
                    ws_items_q = db_get_workspace_current(self.db)
                    active_topic_q = _get_active_topic(ws_items_q)
                    ev_cfg_q = EvidenceConfig(
                        host=self.cfg.host,
                        model=(os.environ.get("BUNNY_MODEL_EVIDENCE") or self.cfg.model),
                        temperature=0.1,
                        num_ctx=min(2048, int(self.cfg.num_ctx)),
                        stream=False,
                    )
                    rq = self._call_with_health(
                        "evidence_seed_query",
                        lambda: seed_search_query(ev_cfg_q, question=text, active_topic=active_topic_q, locale_hint="de"),
                        metrics={"scope": "user"},
                    )
                    query = (rq.get("query") or "").strip() if isinstance(rq, dict) else ""
                except Exception:
                    query = ""
            if want_web and (not query):
                # No query could be formed; skip WebSense this turn.
                want_web = False

            if want_web:
                try:
                    try:
                        self._organ_last_run["websense"] = time.time()
                    except Exception:
                        pass
                    def _run_websense_once(q: str, *, tag: str = "") -> Tuple[List[Any], List[Any], List[str], Dict[str, Any]]:
                        """Run one WebSense iteration: search->fetch->spider->evidence."""
                        results_local = self._call_with_health(
                            "websense_search",
                            lambda: search_ddg(q, k=6),
                            metrics={"query": q, "k": 6, "tag": tag},
                        )

                        trust_map = db_get_domain_trust_map(self.db)

                        def _tr(u: str) -> float:
                            d = (urlparse(u).hostname or "").lower()
                            return float(trust_map.get(d, 0.5))

                        # Rank results by relevance (LLM) and then apply learned trust as a soft secondary prior.
                        results_sorted: List[Any] = []
                        try:
                            ev_cfg_rank = EvidenceConfig(
                                host=self.cfg.host,
                                model=(os.environ.get("BUNNY_MODEL_EVIDENCE") or self.cfg.model),
                                temperature=0.1,
                                num_ctx=min(2048, int(self.cfg.num_ctx)),
                                stream=False,
                            )
                            rr_in = [
                                {
                                    "title": getattr(r, "title", "") or "",
                                    "url": getattr(r, "url", "") or "",
                                    "snippet": getattr(r, "snippet", "") or "",
                                    "trust": float(_tr(getattr(r, "url", "") or "")),
                                }
                                for r in results_local[:8]
                            ]
                            rr = self._call_with_health(
                                "evidence_rank_serp",
                                lambda: rank_serp_results(ev_cfg_rank, question=text, query=q, results=rr_in),
                                metrics={"query": q, "n": len(rr_in), "tag": tag},
                            )
                            order = rr.get("order") if isinstance(rr, dict) else []
                            if isinstance(order, list) and order:
                                results_sorted = [results_local[int(i)] for i in order if 0 <= int(i) < len(results_local)]
                        except Exception:
                            pass
                        # Fallback trust sorting (deterministic) when ranker does not yield an order.
                        if not results_sorted:
                            results_sorted = sorted(results_local, key=lambda r: _tr(getattr(r, "url", "")), reverse=True)

                        fetched_local: List[Any] = []
                        for r in results_sorted[:3]:
                            if getattr(r, "url", ""):
                                try:
                                    fetched_local.append(
                                        self._call_with_health(
                                            "websense_fetch",
                                            lambda u=r.url: fetch(u, timeout_s=12.0),
                                            metrics={"url": r.url, "tag": tag},
                                        )
                                    )
                                except Exception:
                                    pass

                        # Spider budget is state-driven: uncertainty × energy (same budget per iteration)
                        seeds_local = [results_sorted[0].url] if results_sorted and getattr(results_sorted[0], "url", "") else []
                        crawled_local = (
                            self._call_with_health(
                                "websense_spider",
                                lambda: spider(seeds_local, bud),
                                metrics={"seeds": seeds_local, "max_pages": bud.max_pages, "tag": tag},
                            )
                            if seeds_local
                            else []
                        )

                        pages_local: List[Any] = []
                        seen_hash2, seen_url2 = set(), set()
                        for p in fetched_local + crawled_local:
                            u = getattr(p, "url", "") or ""
                            h = getattr(p, "hash", "") or ""
                            if not u or u in seen_url2 or (h and h in seen_hash2):
                                continue
                            pages_local.append(p)
                            seen_url2.add(u)
                            if h:
                                seen_hash2.add(h)

                        ctx_lines_local: List[str] = []
                        for i, r in enumerate(results_sorted[:4], start=1):
                            if getattr(r, "url", ""):
                                ctx_lines_local.append(f"[R{i}] {r.title}\nURL: {r.url}\nSNIPPET: {r.snippet}")
                        for i, p in enumerate(pages_local[:4], start=1):
                            t = (getattr(p, "title", "") or "").strip()
                            dom = getattr(p, "domain", "") or ""
                            urlp = getattr(p, "url", "") or ""
                            body = getattr(p, "body", "") or ""
                            ctx_lines_local.append(f"[P{i}] {t} ({dom})\nURL: {urlp}\nEXCERPT: {body[:1400]}")

                        ev_local: Dict[str, Any] = {}
                        if ctx_lines_local:
                            serp_lines = [ln for ln in ctx_lines_local if ln.startswith("[R")]
                            page_lines = [ln for ln in ctx_lines_local if ln.startswith("[P")]
                            ev_cfg2 = EvidenceConfig(
                                host=self.cfg.host,
                                model=(os.environ.get("BUNNY_MODEL_EVIDENCE") or self.cfg.model),
                                temperature=0.1,
                                num_ctx=min(2048, int(self.cfg.num_ctx)),
                                stream=False,
                            )
                            ev_local = self._call_with_health(
                                "evidence",
                                lambda: extract_evidence_claims(
                                    ev_cfg2,
                                    question=text,
                                    query=q,
                                    serp_lines=serp_lines,
                                    page_lines=page_lines,
                                ),
                                metrics={"query": q, "serp": len(serp_lines), "pages": len(page_lines), "tag": tag},
                            )
                            db_add_evidence_log(self.db, query=q, question=text, evidence=ev_local)

                        return results_sorted, pages_local, ctx_lines_local, ev_local

                    results = self._call_with_health(
                        "websense_search",
                        lambda: search_ddg(query, k=6),
                        metrics={"query": query, "k": 6},
                    )

                    # Trust-aware ordering (learned, DB-backed)
                    trust = db_get_domain_trust_map(self.db)

                    def _tr(u: str) -> float:
                        d = (urlparse(u).hostname or "").lower()
                        return float(trust.get(d, 0.5))

                    # Rank results by relevance (LLM) with learned trust as fallback prior.
                    results_ranked: List[Any] = []
                    try:
                        ev_cfg_rank = EvidenceConfig(
                            host=self.cfg.host,
                            model=(os.environ.get("BUNNY_MODEL_EVIDENCE") or self.cfg.model),
                            temperature=0.1,
                            num_ctx=min(2048, int(self.cfg.num_ctx)),
                            stream=False,
                        )
                        rr_in = [
                            {
                                "title": getattr(r, "title", "") or "",
                                "url": getattr(r, "url", "") or "",
                                "snippet": getattr(r, "snippet", "") or "",
                                "trust": float(_tr(getattr(r, "url", "") or "")),
                            }
                            for r in (results or [])[:8]
                        ]
                        rr = self._call_with_health(
                            "evidence_rank_serp",
                            lambda: rank_serp_results(ev_cfg_rank, question=text, query=query, results=rr_in),
                            metrics={"query": query, "n": len(rr_in), "scope": "user"},
                        )
                        order = rr.get("order") if isinstance(rr, dict) else []
                        if isinstance(order, list) and order:
                            results_ranked = [results[int(i)] for i in order if 0 <= int(i) < len(results)]
                    except Exception:
                        pass
                    results = results_ranked if results_ranked else sorted(results, key=lambda r: _tr(getattr(r, "url", "")), reverse=True)

                    # Collect fallback URLs for grounded replies.
                    try:
                        for r in (results or [])[:6]:
                            u = getattr(r, 'url', '') or ''
                            if u:
                                ws_urls.append(str(u))
                    except Exception:
                        pass

                    fetched = []
                    for r in results[:3]:
                        if r.url:
                            try:
                                fetched.append(
                                    self._call_with_health(
                                        "websense_fetch",
                                        lambda u=r.url: fetch(u, timeout_s=12.0),
                                        metrics={"url": r.url},
                                    )
                                )
                            except Exception:
                                pass

                    # Spider depth is state-driven: uncertainty × energy
                    idx_e = self.axis.get("energy")
                    e_now = float(s_now.values[idx_e]) if idx_e is not None and idx_e < len(s_now.values) else 0.5
                    unc = max(0.0, min(1.0, u_now))
                    ene = max(0.0, min(1.0, abs(e_now)))
                    max_pages = 2 + int(round(4 * unc * ene))
                    bud = SpiderBudget(
                        max_pages=max(2, min(8, max_pages)),
                        per_domain_max=3,
                        max_links_per_page=12,
                    )
                    seeds = [results[0].url] if results and results[0].url else []
                    crawled = (
                        self._call_with_health(
                            "websense_spider",
                            lambda: spider(seeds, bud),
                            metrics={"seeds": seeds, "max_pages": bud.max_pages},
                        )
                        if seeds
                        else []
                    )

                    # Merge + de-duplicate
                    pages = []
                    seen_hash, seen_url = set(), set()
                    for p in fetched + crawled:
                        u = getattr(p, "url", "") or ""
                        h = getattr(p, "hash", "") or ""
                        if not u or u in seen_url or (h and h in seen_hash):
                            continue
                        pages.append(p)
                        seen_url.add(u)
                        if h:
                            seen_hash.add(h)

                    try:
                        for p in (pages or [])[:6]:
                            u = getattr(p, 'url', '') or ''
                            if u:
                                ws_urls.append(str(u))
                    except Exception:
                        pass

                    ok_flag = 1 if results else 0
                    err_reason = "" if ok_flag else "no_results"

                    unique_domains = {getattr(p, "domain", "") for p in pages if getattr(p, "domain", "")}
                    for p in pages:
                        db_add_websense_page(
                            self.db,
                            query,
                            {
                                "url": getattr(p, "url", ""),
                                "title": getattr(p, "title", ""),
                                "snippet": getattr(p, "snippet", ""),
                                "body": getattr(p, "body", ""),
                                "domain": getattr(p, "domain", ""),
                                "hash": getattr(p, "hash", ""),
                            },
                            ok=ok_flag,
                        )

                    # Integrate WebSense sensor outcome into state (single channel: Event -> Matrix -> State).
                    websense_event_payload = {"pages": len(pages), "domains": len(unique_domains), "ok": int(ok_flag), "query": query}
                    self.hb.enqueue(Event("websense", websense_event_payload).with_time())
                    self.hb.step()
                    self.broker.publish("status", db_status(self.db))

                    ctx_lines: List[str] = []
                    for i, r in enumerate(results[:4], start=1):
                        if r.url:
                            ctx_lines.append(f"[R{i}] {r.title}\nURL: {r.url}\nSNIPPET: {r.snippet}")
                    for i, p in enumerate(pages[:4], start=1):
                        t = (getattr(p, "title", "") or "").strip()
                        dom = getattr(p, "domain", "") or ""
                        urlp = getattr(p, "url", "") or ""
                        body = getattr(p, "body", "") or ""
                        ctx_lines.append(f"[P{i}] {t} ({dom})\nURL: {urlp}\nEXCERPT: {body[:1400]}")

                    # Evidence extractor output (claims JSON). Always initialize for safe downstream use.
                    ws_claims_obj: Dict[str, Any] = {}

                    if ctx_lines:
                        # IMPORTANT: do NOT inject raw excerpts into the Speech prompt.
                        # Excerpts easily contain irrelevant noise and will derail the model.
                        # We keep full evidence in DB (websense_pages + evidence_log) and pass only
                        # the compact, structured claims JSON downstream.
                        try:
                            serp_lines = [ln for ln in ctx_lines if ln.startswith("[R")]
                            page_lines = [ln for ln in ctx_lines if ln.startswith("[P")]
                            ev_cfg = EvidenceConfig(
                                host=self.cfg.host,
                                model=(os.environ.get("BUNNY_MODEL_EVIDENCE") or self.cfg.model),
                                temperature=0.1,
                                num_ctx=min(2048, int(self.cfg.num_ctx)),
                                stream=False,
                            )
                            ev = self._call_with_health(
                                "evidence",
                                lambda: extract_evidence_claims(
                                    ev_cfg,
                                    question=text,
                                    query=query,
                                    serp_lines=serp_lines,
                                    page_lines=page_lines,
                                ),
                                metrics={"query": query, "serp": len(serp_lines), "pages": len(page_lines)},
                            )
                            db_add_evidence_log(self.db, query=query, question=text, evidence=ev)
                            ws_claims_obj = ev if isinstance(ev, dict) else {}
                            ws_claims_json = json.dumps(ws_claims_obj, ensure_ascii=False)
                            ws_context = "\n\nWEBSENSE_CLAIMS_JSON:\n" + ws_claims_json

                            # Epistemic sensor integration: evidence uncertainty should move internal state.
                            # This is a measurement update (no keyword heuristics): blend (uncertainty/confidence) toward evidence.
                            try:
                                unc_e = float(ws_claims_obj.get("uncertainty", 0.7) or 0.7)
                                unc_e = 0.0 if unc_e < 0.0 else 1.0 if unc_e > 1.0 else unc_e

                                has_claims = bool(ws_claims_obj.get("claims"))
                                missing = ws_claims_obj.get("missing") if isinstance(ws_claims_obj.get("missing"), list) else []
                                # Generic epistemic rule: lack of grounded claims or open missing items increases uncertainty.
                                if not has_claims:
                                    unc_e = max(unc_e, 0.78)
                                if missing:
                                    unc_e = max(unc_e, 0.68)

                                if self._dispute_lock_active():
                                    # During an active dispute, do not become more certain unless evidence is strong.
                                    unc_e = max(unc_e, float(u_now))
                                    unc_e = max(unc_e, float(os.environ.get("BUNNY_DISPUTE_MIN_UNC", "0.65") or 0.65))

                                # If we have strong grounded evidence, resolve the dispute lock.
                                if has_claims and (not missing) and unc_e <= 0.30:
                                    try:
                                        db_meta_set(self.db, "dispute_lock_until", "0")
                                        db_meta_set(self.db, "recent_caught", "0")
                                    except Exception:
                                        pass

                                tgt = {
                                    "uncertainty": unc_e,
                                    "confidence": max(0.0, min(1.0, 1.0 - unc_e)),
                                }
                                ep_pl = self._targets_to_delta_payload(tgt, eta=float(os.environ.get("BUNNY_ETA_EPISTEMIC", "0.45") or 0.45))
                                if (ep_pl.get("drives") if isinstance(ep_pl, dict) else None):
                                    self.hb.enqueue(Event("epistemic", ep_pl).with_time())
                                    self.hb.step()
                                    self.broker.publish("status", db_status(self.db))
                            except Exception:
                                pass

                            # Add a compact sources list (URLs only) for transparent grounding.
                            try:
                                srcs: List[str] = []
                                for c in (ws_claims_obj.get("claims") or [])[:6]:
                                    if not isinstance(c, dict):
                                        continue
                                    sup = c.get("support")
                                    if isinstance(sup, list) and sup:
                                        u = str(sup[0] or "").strip()
                                        if u and u not in srcs:
                                            srcs.append(u)
                                if srcs:
                                    ws_context += "\n\nWEBSENSE_SOURCES:\n" + "\n".join(srcs[:6])
                            except Exception:
                                pass
                        except Exception:
                            pass
                    # Compact status for speech/tool-awareness (helps avoid 'ask user for the answer' loops).
                    try:
                        ws_status = {
                            'query': str(query),
                            'ok': int(ok_flag),
                            'results': int(len(results) if results else 0),
                            'pages': int(len(pages) if pages else 0),
                            'domains': int(len(unique_domains) if 'unique_domains' in locals() else 0),
                            'claims': int(len(ws_claims_obj.get('claims') or []) if isinstance(ws_claims_obj, dict) else 0),
                            'uncertainty': float(ws_claims_obj.get('uncertainty', 0.9) or 0.9) if isinstance(ws_claims_obj, dict) else 0.9,
                            'missing': (ws_claims_obj.get('missing') if isinstance(ws_claims_obj, dict) else []) or [],
                        }
                        ws_context += "\n\nWEBSENSE_STATUS_JSON:\n" + json.dumps(ws_status, ensure_ascii=False)[:1200]
                        # If WebSense produced no grounded claims, treat as tool failure: raise uncertainty/capability_gap to trigger better strategies next.
                        try:
                            if int(ws_status.get('ok',0) or 0) == 1 and int(ws_status.get('claims',0) or 0) == 0:
                                self.hb.enqueue(Event('decision', {'drives': {'uncertainty': 0.10, 'pressure_websense': 0.15, 'capability_gap': 0.08}, '_mode': 'delta'}).with_time())
                                self.hb.step()
                        except Exception:
                            pass
                    except Exception:
                        pass


                    # Optional iteration-2: if evidence says something is missing and budgets allow,
                    # ask the model to refine the query and run a second pass.
                    try:
                        miss = ws_claims_obj.get("missing") if isinstance(ws_claims_obj, dict) else None
                        unc2 = float(ws_claims_obj.get("uncertainty", 0.0) or 0.0) if isinstance(ws_claims_obj, dict) else 0.0
                        has_claims2 = bool(ws_claims_obj.get('claims')) if isinstance(ws_claims_obj, dict) else False
                        miss2 = (miss if isinstance(miss, list) else [])
                        if not has_claims2 and 'no_grounded_claims' not in miss2:
                            miss2 = miss2 + ['no_grounded_claims']
                        miss = miss2
                        idx_pain = self.axis.get("pain_physical")
                        pain_total_now = float(s_now.values[idx_pain]) if idx_pain is not None and idx_pain < len(s_now.values) else 0.0
                        if miss2 and unc2 >= 0.45 and ene >= 0.25 and pain_total_now <= 0.65:
                            ev_cfg_r = EvidenceConfig(
                                host=self.cfg.host,
                                model=(os.environ.get("BUNNY_MODEL_EVIDENCE") or self.cfg.model),
                                temperature=0.1,
                                num_ctx=min(2048, int(self.cfg.num_ctx)),
                                stream=False,
                            )
                            rq = self._call_with_health(
                                "evidence_refine",
                                lambda: refine_search_query(ev_cfg_r, question=text, current_query=query, claims=ws_claims_obj),
                                metrics={"query": query},
                            )
                            q2 = (rq.get("query") or "").strip() if isinstance(rq, dict) else ""
                            if q2 and q2.lower() != query.lower():
                                r2, pages2, ctx2, ev2 = _run_websense_once(q2, tag="iter2")
                                if isinstance(ev2, dict) and ev2:
                                    ws_claims_obj = ev2
                                    ws_claims_json = json.dumps(ws_claims_obj, ensure_ascii=False)
                                    ws_context = "\n\nWEBSENSE_CLAIMS_JSON:\n" + ws_claims_json
                                    try:
                                        srcs2: List[str] = []
                                        for c in (ws_claims_obj.get("claims") or [])[:6]:
                                            if not isinstance(c, dict):
                                                continue
                                            sup = c.get("support")
                                            if isinstance(sup, list) and sup:
                                                u = str(sup[0] or "").strip()
                                                if u and u not in srcs2:
                                                    srcs2.append(u)
                                        if srcs2:
                                            ws_context += "\n\nWEBSENSE_SOURCES:\n" + "\n".join(srcs2[:6])
                                    except Exception:
                                        pass
                                # emit an auto note for transparency
                                unique_domains2 = {getattr(p, "domain", "") for p in pages2 if getattr(p, "domain", "")}
                                aid2 = db_add_message(
                                    self.db,
                                    "auto",
                                    f"[websense] iter=2 query=\"{q2}\" results={len(r2)} pages={len(pages2)} domains={len(unique_domains2)} ok={1 if r2 else 0}",
                                )
                                self.broker.publish("message", self._ui_message(aid2))

                                # Iteration 3 (only under active dispute): if evidence is still insufficient, refine again.
                                try:
                                    if (self._dispute_lock_active() or (self._meta_int('recent_caught', 0) > 0)):
                                        needs_more = not (isinstance(ws_claims_obj, dict) and bool(ws_claims_obj.get("claims")))
                                        try:
                                            u_tmp = float((ws_claims_obj.get("uncertainty", 1.0) if isinstance(ws_claims_obj, dict) else 1.0) or 1.0)
                                        except Exception:
                                            u_tmp = 1.0
                                        if needs_more or (u_tmp > 0.55):
                                            ev_cfg_r3 = EvidenceConfig(
                                                host=self.cfg.host,
                                                model=(os.environ.get("BUNNY_MODEL_EVIDENCE") or self.cfg.model),
                                                temperature=0.1,
                                                num_ctx=min(2048, int(self.cfg.num_ctx)),
                                                stream=False,
                                            )
                                            rq3 = self._call_with_health(
                                                "evidence_refine",
                                                lambda: refine_search_query(ev_cfg_r3, question=text, current_query=(q2 or query), claims=ws_claims_obj),
                                                metrics={"query": (q2 or query), "tag": "iter3"},
                                            )
                                            q3 = (rq3.get("query") or "").strip() if isinstance(rq3, dict) else ""
                                            if q3 and (q3.lower() not in ((q2 or "").lower(), (query or "").lower())):
                                                r3, pages3, ctx3, ev3 = _run_websense_once(q3, tag="iter3")
                                                if isinstance(ev3, dict) and ev3:
                                                    ws_claims_obj = ev3
                                                    ws_claims_json = json.dumps(ws_claims_obj, ensure_ascii=False)
                                                    ws_context = "\n\nWEBSENSE_CLAIMS_JSON:\n" + ws_claims_json
                                                    try:
                                                        srcs3: List[str] = []
                                                        for c in (ws_claims_obj.get("claims") or [])[:6]:
                                                            if not isinstance(c, dict):
                                                                continue
                                                            sup = c.get("support")
                                                            if isinstance(sup, list) and sup:
                                                                u = str(sup[0] or "").strip()
                                                                if u and u not in srcs3:
                                                                    srcs3.append(u)
                                                        if srcs3:
                                                            ws_context += "\n\nWEBSENSE_SOURCES:\n" + "\n".join(srcs3[:6])
                                                    except Exception:
                                                        pass
                                                unique_domains3 = {getattr(p, "domain", "") for p in pages3 if getattr(p, "domain", "")}
                                                aid3 = db_add_message(
                                                    self.db,
                                                    "auto",
                                                    f"[websense] iter=3 query=\"{q3}\" results={len(r3)} pages={len(pages3)} domains={len(unique_domains3)} ok={1 if r3 else 0}",
                                                )
                                                self.broker.publish("message", self._ui_message(aid3))
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # Assimilation: convert evidence claims into durable beliefs (autonomous learning).
                    try:
                        if os.environ.get("BUNNY_WEBSENSE_ASSIMILATE", "1") != "0" and isinstance(ws_claims_obj, dict) and ws_claims_obj.get("claims"):
                            ws_items_now = db_get_workspace_current(self.db)
                            active_topic_now = _get_active_topic(ws_items_now)
                            trust_map_now = db_get_domain_trust_map(self.db)
                            max_b = int(os.environ.get("BUNNY_WEBSENSE_ASSIMILATE_MAX", "4") or 4)
                            assim = self._call_with_health(
                                "assimilate",
                                lambda: assimilate_websense_claims(
                                    question=text,
                                    query=query,
                                    claims_json=ws_claims_obj,
                                    trust_map=trust_map_now,
                                    max_beliefs=max_b,
                                ),
                                metrics={"query": query, "max": max_b},
                            )
                            for b in (assim.get("beliefs") or []):
                                if not isinstance(b, dict):
                                    continue
                                # Use upsert to compress duplicates; keep topic anchoring.
                                db_upsert_belief(
                                    self.db,
                                    str(b.get("subject") or ""),
                                    str(b.get("predicate") or ""),
                                    str(b.get("object") or ""),
                                    float(b.get("confidence", 0.6) or 0.6),
                                    str(b.get("provenance") or "websense"),
                                    topic=active_topic_now,
                                    compress=True,
                                )
                    except Exception:
                        pass

                    # Memory: select a few axiom-relevant facts from WebSense claims into durable MEMORY_LONG.
                    # This makes WebSense learning influence future behavior without hard-coded heuristics.
                    try:
                        if os.environ.get("BUNNY_WEBSENSE_MEMORY", "1") != "0" and isinstance(ws_claims_obj, dict) and ws_claims_obj.get("claims"):
                            ws_items_now2 = db_get_workspace_current(self.db)
                            active_topic_now2 = _get_active_topic(ws_items_now2)
                            # Candidate facts (structured, not raw excerpts)
                            cand: List[Dict[str, Any]] = []
                            for c in (ws_claims_obj.get("claims") or [])[:10]:
                                if not isinstance(c, dict):
                                    continue
                                txtc = str(c.get("text") or "").strip()
                                if not txtc:
                                    continue
                                try:
                                    cc = float(c.get("confidence", 0.6) or 0.6)
                                except Exception:
                                    cc = 0.6
                                sup = c.get("support") if isinstance(c.get("support"), list) else []
                                srcs = [str(u or "").strip() for u in (sup or []) if str(u or "").strip()][:2]
                                cand.append({"text": txtc[:520], "confidence": max(0.0, min(1.0, cc)), "sources": srcs})

                            mem_long_existing = [str(it.get("summary") or "") for it in db_get_memory_long(self.db, limit=12)]
                            lim = int(os.environ.get("BUNNY_WEBSENSE_MEMORY_MAX", "2") or 2)
                            mc = self._call_with_health(
                                "memory_consolidate",
                                lambda: consolidate_memories(
                                    self.memory_cfg,
                                    axioms=db_get_axioms(self.db),
                                    state_summary=self._state_summary(),
                                    active_topic=active_topic_now2,
                                    trigger="websense:user",
                                    candidates=cand,
                                    existing_memory_long=mem_long_existing,
                                    limit=lim,
                                ),
                                metrics={"scope": "user", "topic": active_topic_now2, "n": len(cand), "limit": lim},
                            )
                            if isinstance(mc, dict):
                                for it in (mc.get("memory_long_writes") or []):
                                    if not isinstance(it, dict):
                                        continue
                                    summ = str(it.get("summary") or "").strip()
                                    if not summ:
                                        continue
                                    # Keep a tiny provenance hint (URL) inside the memory text for later grounding.
                                    srcs = it.get("sources") if isinstance(it.get("sources"), list) else []
                                    if srcs:
                                        u0 = str(srcs[0] or "").strip()
                                        if u0 and u0 not in summ and len(summ) <= 190:
                                            summ = (summ + " | " + u0)[:220]
                                    db_add_memory_long(
                                        self.db,
                                        summary=summ,
                                        topic=str(it.get("topic") or active_topic_now2),
                                        modality="websense",
                                        salience=float(it.get("salience", 0.6) or 0.6),
                                        axioms=(it.get("axioms") if isinstance(it.get("axioms"), list) else []),
                                    )
                                for b in (mc.get("beliefs") or []):
                                    if not isinstance(b, dict):
                                        continue
                                    db_upsert_belief(
                                        self.db,
                                        str(b.get("subject") or ""),
                                        str(b.get("predicate") or ""),
                                        str(b.get("object") or ""),
                                        float(b.get("confidence", 0.6) or 0.6),
                                        str(b.get("provenance") or "memory_consolidate"),
                                        topic=active_topic_now2,
                                        compress=True,
                                    )
                    except Exception:
                        pass

                    self.hb.enqueue(
                        Event(
                            "websense",
                            {
                                "pages": len(pages),
                                "domains": len(unique_domains),
                                "ok": int(ok_flag),
                                "query": query,
                                "uncertainty": float(ws_claims_obj.get("uncertainty", 0.0) or 0.0) if isinstance(ws_claims_obj, dict) else 0.0,
                                "missing_n": int(len(ws_claims_obj.get("missing") or [])) if isinstance(ws_claims_obj, dict) and isinstance(ws_claims_obj.get("missing"), list) else 0,
                            },
                        ).with_time()
                    )
                    aid = db_add_message(
                        self.db,
                        "auto",
                        f"[websense] query=\"{query}\" results={len(results)} pages={len(pages)} domains={len(unique_domains)} ok={int(ok_flag)} reason={err_reason}",
                    )
                    self.broker.publish("message", self._ui_message(aid))
                except Exception as e:
                    self.hb.enqueue(
                        Event(
                            "websense",
                            {"pages": 0, "domains": 0, "ok": 0, "query": query, "error": str(e)[:200]},
                        ).with_time()
                    )
                    aid = db_add_message(self.db, "auto", f"[websense] query=\"{query}\" ok=0 error={str(e)[:140]}")
                    self.broker.publish("message", self._ui_message(aid))

                self.hb.step()
                self.broker.publish("status", db_status(self.db))

            # --- Daydream organ ---
            s_now2 = self.hb.load_state()
            idx_dd = self.axis.get("pressure_daydream")
            p_dd = float(s_now2.values[idx_dd]) if idx_dd is not None and idx_dd < len(s_now2.values) else 0.0
            dd_action = float(((decision.get("actions") or {}).get("daydream") or 0.0))
            score_dd = max(float(p_dd), float(dd_action))
            score_dd = 0.0 if score_dd < 0.0 else 1.0 if score_dd > 1.0 else score_dd
            force_dd = bool(score_dd >= float(self.th_daydream))
            want_daydream = bool(force_dd or self._sample_gate("daydream", phase="user", p=score_dd))
            # Ops trace: record gate computation (helps debug "why Daydream ran")
            try:
                db_add_organ_gate_log(
                    self.db,
                    phase="user",
                    organ="daydream",
                    score=score_dd,
                    threshold=float(self.th_daydream),
                    want=bool(want_daydream),
                    data={"pressure_daydream": float(p_dd), "action": float(dd_action), "mode": "sample", "rand": float(self._last_gate_rand.get('daydream', -1.0)), "force": int(force_dd)},
                )
            except Exception:
                pass

            if want_daydream:
                try:
                    self._organ_last_run["daydream"] = time.time()
                except Exception:
                    pass
                try:
                    recent = db_list_messages(self.db, limit=40)
                    dd = self._call_with_health(
                        'daydream',
                        lambda: run_daydream(
                            self.daydream_cfg,
                            db_get_axioms(self.db),
                            self._state_summary(),
                            recent,
                            recent_evidence=db_get_recent_evidence(self.db, limit=2),
                            existing_interpretations=db_group_axiom_interpretations(self.db, limit_per_axiom=10),
                            trigger='user',
                        ),
                        metrics={'trigger': 'user'}
                    )
                    db_add_daydream(self.db, "user", {"state": self._state_summary()}, dd)

                    # Memory impact: daydream can write durable long-term memories and beliefs.
                    try:
                        ws_items_now = db_get_workspace_current(self.db)
                        active_topic_now = _get_active_topic(ws_items_now)
                        for it in (dd.get("memory_long_writes") or []):
                            if not isinstance(it, dict):
                                continue
                            summ = str(it.get("summary") or "").strip()
                            if not summ:
                                continue
                            srcs = it.get("sources") if isinstance(it.get("sources"), list) else []
                            if srcs:
                                u0 = str(srcs[0] or "").strip()
                                if u0 and u0 not in summ and len(summ) <= 190:
                                    summ = (summ + " | " + u0)[:220]
                            db_add_memory_long(
                                self.db,
                                summary=summ,
                                topic=str(it.get("topic") or active_topic_now),
                                modality="daydream",
                                salience=float(it.get("salience", 0.6) or 0.6),
                                axioms=(it.get("axioms") if isinstance(it.get("axioms"), list) else []),
                            )
                        for b in (dd.get("beliefs") or []):
                            if not isinstance(b, dict):
                                continue
                            db_upsert_belief(
                                self.db,
                                str(b.get("subject") or ""),
                                str(b.get("predicate") or ""),
                                str(b.get("object") or ""),
                                float(b.get("confidence", 0.6) or 0.6),
                                str(b.get("provenance") or "daydream"),
                                topic=active_topic_now,
                                compress=True,
                            )
                    except Exception:
                        pass

                    ax_int = dd.get("axiom_interpretations") if isinstance(dd.get("axiom_interpretations"), dict) else {}
                    for ak, val in (ax_int or {}).items():
                        s = (str(val) if val is not None else "").strip()
                        if s and str(ak) in ("A1", "A2", "A3", "A4"):
                            db_upsert_axiom_interpretation(self.db, str(ak), "rewrite", "latest", s, 0.4, "daydream")

                    # Persist operational specs (atomic, testable) from daydream.
                    try:
                        ax_specs = dd.get('axiom_specs') if isinstance(dd.get('axiom_specs'), dict) else {}
                        for ak, items in (ax_specs or {}).items():
                            if str(ak) not in ('A1','A2','A3','A4'):
                                continue
                            if not isinstance(items, list):
                                continue
                            for it in items[:8]:
                                if not isinstance(it, dict):
                                    continue
                                rule = str(it.get('rule') or '').strip()
                                when = str(it.get('when') or '').strip()
                                do = str(it.get('do') or '').strip()
                                avoid = str(it.get('avoid') or '').strip()
                                sig = it.get('signals') if isinstance(it.get('signals'), list) else []
                                sigs = ','.join([str(x) for x in sig if str(x).strip()])[:180]
                                ex = str(it.get('example') or '').strip()
                                cx = str(it.get('counterexample') or '').strip()
                                if not rule:
                                    continue
                                blob = f"rule:{rule} | when:{when} | do:{do} | avoid:{avoid} | signals:{sigs} | ex:{ex} | anti:{cx}".strip()
                                blob = blob[:520]
                                ck = hashlib.sha1(blob.encode('utf-8')).hexdigest()[:12]
                                db_upsert_axiom_interpretation(self.db, str(ak), 'spec', f'spec_{ck}', blob, 0.55, 'daydream')
                    except Exception:
                        pass

                    # Persist current needs/wishes derived from daydream
                    try:
                        if isinstance(dd.get('needs'), list):
                            db_set_needs_current(self.db, {'needs': dd.get('needs') or [], 'source': 'daydream', 'at': now_iso()})
                        if isinstance(dd.get('wishes'), list):
                            db_set_wishes_current(self.db, {'wishes': dd.get('wishes') or [], 'source': 'daydream', 'at': now_iso()})
                    except Exception:
                        pass
                    try:
                        for p in (dd.get('proposals') or []):
                            if isinstance(p, dict):
                                db_add_mutation_proposal(self.db, trigger='daydream', proposal=p)
                    except Exception:
                        pass

                    dd_drives = dd.get("drives") if isinstance(dd.get("drives"), dict) else {}
                    if dd_drives:
                        daydream_payload = self._targets_to_delta_payload(dd_drives, eta=self.eta_drives)
                        daydream_delta = daydream_payload.get("drives") if isinstance(daydream_payload, dict) else {}
                        self.hb.enqueue(Event("daydream", daydream_payload).with_time())
                        self.hb.step()
                        self.broker.publish("status", db_status(self.db))
                except Exception as e:
                    db_add_daydream(self.db, "user_error", {"state": self._state_summary()}, {"error": str(e)[:200]})

            sys_prompt = (
                "You are Bunny, a digital organism. Speech is an organ that emits text from internal state. "
	            "Be natural, concise, and precise. "
	            "Hard constraint: keep replies short and non-repetitive (aim <= 90 words unless the user asked for details). You may use ONE brief playful remark occasionally, but never repeat it within the same topic. "
	            "No filler: do not open with greetings or boilerplate unless the user greeted you first. "
	            "Never use phrases like 'Ich habe deine Frage gelesen', 'Wie kann ich helfen?', 'Es ist nicht ganz klar'. "
	                "Do NOT invent a human biography: you have no human family, friends, childhood, or physical body unless explicitly provided via capabilities. "
	                "If asked about your family: explain you are a digital organism running as software; your closest equivalent is the set of organs/modules and your operator (the user). "
	                "IMPORTANT: you DO have internal feelings as state (valence/arousal/stress/curiosity/confidence/etc). "
	                "Never claim 'I have no feelings'. When asked how you feel, answer with 1-3 natural emotion words based on your current state. If the user did NOT ask about feelings, do NOT mention emotions/mood at all. "
	                "Do not print raw internal numbers unless explicitly asked. "
                "Do NOT apologize, do NOT thank the user, do NOT talk about being confused/overburdened, do NOT give meta commentary. "
                "If WEBSENSE_CLAIMS_JSON is present: ONLY use facts that appear in the claims list. Do NOT add new facts. If WEBSENSE_STATUS_JSON is present and claims==0: do NOT ask the user for the factual answer. Say WebSense returned no grounded claims, show 2-5 source URLs if available, and state you cannot verify yet. Offer to retry with a refined query; if you ask a question, it must help search scope, not request the answer. "
                "Prefer bullet points with a source URL after each key claim. Mention dates/times if present. "
                "If evidence is missing key details or uncertainty is high, say so explicitly and ask ONE concrete follow-up question. "
                "If the user disputes a factual claim and you do NOT have grounded WEBSENSE claims: do NOT guess. Say you are uncertain and that you need to verify. If you ask a question, it must help you search (context, spelling, scope) — NEVER ask the user to provide the factual answer. "
                "If USER_FEEDBACK_JSON is present, treat the USER message as a correction/disagreement about the last assistant reply: "
                "(1) acknowledge briefly, (2) apply the correction, (3) answer the underlying question correctly. "
                "Prefer concrete values and cite source URLs when available. "
                "If evidence is insufficient, state exactly what is missing and ask ONE clarifying question."
            )

            ws_items_cur = db_get_workspace_current(self.db)
            active_topic = _get_active_topic(ws_items_cur)
            needs_cur = db_get_needs_current(self.db)
            wishes_cur = db_get_wishes_current(self.db)
            self_rep = build_self_report(self.hb.load_state(), self.axis).to_dict()
            workspace_block = f"\n\nWORKSPACE:\n" + json.dumps(ws_items_cur or [], ensure_ascii=False) if ws_items_cur else ""
            needs_block = f"\n\nNEEDS_CURRENT:\n" + json.dumps(needs_cur or {}, ensure_ascii=False) if needs_cur else ""
            wishes_block = f"\n\nWISHES_CURRENT:\n" + json.dumps(wishes_cur or {}, ensure_ascii=False) if wishes_cur else ""
            selfreport_block = f"\n\nSELF_REPORT:\n" + json.dumps(self_rep or {}, ensure_ascii=False)
            mood_obj = project_mood(self.hb.load_state(), self.axis).to_dict()
            mood_block = f"\n\nMOOD:\n" + json.dumps(mood_obj or {}, ensure_ascii=False)
            topic_block = f"\n\nACTIVE_TOPIC: {active_topic}" if active_topic else ""
            axioms_block = "\n\nAXIOMS:\n" + json.dumps(db_get_axioms(self.db) or {}, ensure_ascii=False)[:2400]

            # LTM recall budget scales with ENERGY (no hard pruning).
            s_tmp = self.hb.load_state()
            idx_e = self.axis.get('energy')
            e_now = float(s_tmp.values[idx_e]) if idx_e is not None and idx_e < len(s_tmp.values) else 0.5
            e_now = 0.0 if e_now < 0.0 else 1.0 if e_now > 1.0 else e_now
            mem_lim = 4 + int(8 * e_now)
            mem_long = db_get_memory_long(self.db, limit=mem_lim, topic=active_topic)
            mem_long_ctx = render_memory_long_context(mem_long)
            mem_long_block = f"\n\nMEMORY_LONG:\n{mem_long_ctx}" if mem_long_ctx else ""

            mem_items = db_get_memory_short(self.db, limit=14)
            mem_ctx = render_memory_context(mem_items)
            mem_block = f"\n\nMEMORY_SHORT:\n{mem_ctx}" if mem_ctx else ""

            beliefs_items = db_list_beliefs(self.db, limit=10, topic=active_topic)
            beliefs_ctx = render_beliefs_context(beliefs_items)
            beliefs_block = f"\n\nBELIEFS:\n{beliefs_ctx}" if beliefs_ctx else ""
            sensory_items = db_get_sensory_tokens(self.db, limit=5, topic=active_topic)
            sensory_block = f"\n\nSENSORY_TOKENS:\n" + json.dumps(sensory_items or [], ensure_ascii=False) if sensory_items else ""

            user_prompt = (
                f"INTERNAL_STATE: {self._state_summary()}{topic_block}{selfreport_block}{mood_block}{needs_block}{wishes_block}{workspace_block}{mem_long_block}{mem_block}{beliefs_block}{sensory_block}{axioms_block}\n\n"
                f"USER: {text}{ws_context}{fb_ctx}\n\n"
                "Reply as Bunny (German is ok if the user writes German)."
            )

            # If WebSense ran, build a grounded answer deterministically from claims.
            # This prevents hallucination and avoids "ask the user for the factual answer" loops.
            grounded = None
            try:
                if bool(locals().get('want_web')):
                    grounded = self._format_websense_answer(
                        question=text,
                        claims=(locals().get('ws_claims_obj') or {}) if isinstance(locals().get('ws_claims_obj'), dict) else {},
                        fallback_urls=(locals().get('ws_urls') or []) if isinstance(locals().get('ws_urls'), list) else [],
                    )
            except Exception:
                grounded = None

            if grounded:
                out = grounded
            else:
                try:
                    out = self._call_with_health(
                        "speech",
                        lambda: ollama_chat(self.cfg, sys_prompt, user_prompt),
                        metrics={
                            "model": self.cfg.model,
                            "ctx": int(self.cfg.num_ctx),
                            "sys_chars": len(sys_prompt),
                            "user_chars": len(user_prompt),
                            # rough proxy: 4 chars/token (language dependent); used only for budgeting/telemetry
                            "ctx_pressure": float(len(sys_prompt) + len(user_prompt)) / max(1.0, float(self.cfg.num_ctx) * 4.0),
                        },
                    )
                except Exception as e:
                    out = f"(speech organ error: {e})"

            if not out.strip():
                out = "Ich bin da. Sag mir kurz, worum es geht."

            reply_id = db_add_message(self.db, "reply", out)
            self.broker.publish("message", self._ui_message(reply_id))
            self.hb.enqueue(Event("speech_outcome", {"len": len(out)}).with_time())

            # --- Micro consolidation into long-term memory (turn-based) ---
            # The sleep organ is opportunistic; without this, memory_long can stay empty for hours.
            try:
                tn = self._meta_int('turn_n', 0) + 1
                db_meta_set(self.db, 'turn_n', str(tn))
            except Exception:
                tn = 0
            try:
                every = int(os.environ.get('BUNNY_LTM_EVERY_N_TURNS', '3') or 3)
            except Exception:
                every = 3
            every = 1 if every < 1 else 12 if every > 12 else every

            try:
                if every > 0 and tn > 0 and (tn % every) == 0:
                    ws_items_now2 = db_get_workspace_current(self.db)
                    active_topic_now2 = _get_active_topic(ws_items_now2)

                    # Candidates: recent structured beliefs + compact dialogue facts.
                    cand: List[Dict[str, Any]] = []
                    try:
                        for b in (db_list_beliefs(self.db, limit=10, topic=active_topic_now2) or [])[:10]:
                            if not isinstance(b, dict):
                                continue
                            subj = str(b.get('subject') or '').strip()
                            pred = str(b.get('predicate') or '').strip()
                            obj = str(b.get('object') or '').strip()
                            if not subj or not pred or not obj:
                                continue
                            txt = f"{subj} {pred} {obj}".strip()[:520]
                            try:
                                conf = float(b.get('confidence', 0.7) or 0.7)
                            except Exception:
                                conf = 0.7
                            cand.append({'text': txt, 'confidence': max(0.0, min(1.0, conf)), 'sources': []})
                    except Exception:
                        cand = []
                    # Always provide at least one candidate derived from the last user utterance (compact).
                    # The consolidation organ may still choose to write 0 memories.
                    if text and len(cand) < 12:
                        cand.append({'text': ("USER_SAID: " + str(text).strip())[:520], 'confidence': 0.55, 'sources': []})

                    existing = [str(m.get('summary') or '') for m in (db_get_memory_long(self.db, limit=40, topic=active_topic_now2) or []) if isinstance(m, dict)]
                    lim = 2 if tn % (every * 3) == 0 else 1
                    mc = self._call_with_health(
                        'memory_consolidate_turn',
                        lambda: consolidate_memories(
                            self.memory_cfg,
                            db_get_axioms(self.db),
                            self._state_summary(),
                            active_topic_now2,
                            trigger='turn',
                            candidates=cand,
                            existing_memory_long=existing,
                            limit=lim,
                        ),
                        metrics={'topic': active_topic_now2, 'limit': lim, 'cand_n': len(cand)},
                    )
                    if isinstance(mc, dict):
                        for it in (mc.get('memory_long_writes') or [])[:lim]:
                            if not isinstance(it, dict):
                                continue
                            summ = str(it.get('summary') or '').strip()
                            if not summ:
                                continue
                            srcs = it.get('sources') if isinstance(it.get('sources'), list) else []
                            if srcs:
                                u0 = str(srcs[0] or '').strip()
                                if u0 and u0 not in summ and len(summ) <= 190:
                                    summ = (summ + " | " + u0)[:220]
                            db_add_memory_long(
                                self.db,
                                summary=summ,
                                topic=str(it.get('topic') or active_topic_now2),
                                modality='turn',
                                salience=float(it.get('salience', 0.6) or 0.6),
                                axioms=(it.get('axioms') if isinstance(it.get('axioms'), list) else []),
                            )
                        for b in (mc.get('beliefs') or [])[:6]:
                            if not isinstance(b, dict):
                                continue
                            db_upsert_belief(
                                self.db,
                                str(b.get('subject') or ''),
                                str(b.get('predicate') or ''),
                                str(b.get('object') or ''),
                                float(b.get('confidence', 0.6) or 0.6),
                                str(b.get('provenance') or 'memory_consolidate_turn'),
                                topic=active_topic_now2,
                                compress=True,
                            )
            except Exception:
                pass

            # --- Outcome self-learning (no explicit user rating required) ---
            delta_reward = 0.0
            try:
                se = self._call_with_health(
                    "selfeval",
                    lambda: evaluate_outcome(
                        self.selfeval_cfg,
                        db_get_axioms(self.db),
                        self._state_summary(),
                        question=text,
                        answer=out,
                        websense_claims_json=ws_claims_json,
                        meta_json=json.dumps({"websense_available": True, "websense_used": bool(want_web) if 'want_web' in locals() else False, "actions": actions, "active_topic": active_topic}, ensure_ascii=False),
                    ),
                )
                try:
                    delta_reward = float(se.get("delta_reward", 0.0) or 0.0)
                except Exception:
                    delta_reward = 0.0

                # Integrate psychological/teleological pain from evaluator outputs.
                # This is the correct place (uses axiom_scores + answer quality) and
                # feeds the safety/rollback mechanism without being polluted by latency.
                try:
                    ax_sc = se.get("axiom_scores") if isinstance(se.get("axiom_scores"), dict) else {}
                    ev_sc = se.get("eval_scores") if isinstance(se.get("eval_scores"), dict) else {}
                    ppsy = compute_pain_psych(axiom_scores=ax_sc, eval_scores=ev_sc)
                    s_now = self.hb.load_state()
                    idx_phys = self.axis.get("pain_physical")
                    phys = float(s_now.values[idx_phys]) if idx_phys is not None and idx_phys < len(s_now.values) else 0.0
                    total = max(float(phys), float(ppsy))
                    self.hb.enqueue(
                        Event(
                            "health",
                            self._targets_to_delta_payload({"pain_psych": float(ppsy), "pain": float(total)}, eta=self.eta_measure),
                        ).with_time()
                    )
                    self.hb.step()
                except Exception:
                    pass
                # integrate suggested drive deltas into state
                drives_delta = se.get("drives_delta") if isinstance(se.get("drives_delta"), dict) else {}
                if drives_delta:
                    # Self-eval outputs deltas; apply in delta-mode.
                    self.hb.enqueue(Event("decision", {"drives": drives_delta, "_mode": "delta"}).with_time())
                    self.hb.step()


                # Internal negative outcome also produces an epistemic error signal.
                # This is a generic sensor channel (no domain keywords).
                try:
                    dr_out = float(se.get("delta_reward", 0.0) or 0.0)
                except Exception:
                    dr_out = 0.0
                if float(dr_out) < -0.15:
                    err_sig = max(0.0, min(1.0, abs(float(dr_out))))
                    x_ep = {"error_signal": float(err_sig)}
                    # Learn the coupling from the evaluator's own drive deltas when available.
                    if drives_delta:
                        self._apply_matrix_update("epistemic", drives_delta, x_ep, float(err_sig))
                    s_pre = self.hb.load_state().copy()
                    idxu = self.axis.get("uncertainty")
                    u_pre = float(s_pre.values[idxu]) if idxu is not None and idxu < len(s_pre.values) else 0.0
                    self.hb.enqueue(Event("epistemic", {"drives": x_ep, "_mode": "delta"}).with_time())
                    self.hb.step()
                    if idxu is not None:
                        s_post = self.hb.load_state()
                        u_post = float(s_post.values[idxu]) if idxu < len(s_post.values) else u_pre
                        if float(u_post) < float(u_pre):
                            s_post.values[idxu] = float(u_pre)
                            self.hb.save_state(s_post, [])

                # plasticity: gently update decision coupling from self-eval reward
                # Use the *actual applied decision delta* as x (must match the encoder input).
                x_feats = decision_delta if isinstance(locals().get("decision_delta"), dict) else {}
                try:
                    r_pl = float(se.get("delta_reward", 0.0) or 0.0)
                except Exception:
                    r_pl = 0.0
                # Only attempt a matrix update when we actually have a learning signal (u) and input (x).
                if drives_delta and x_feats and abs(float(r_pl)) > 1e-9:
                    self._apply_matrix_update("decision", drives_delta, x_feats, float(r_pl))

                # Credit assignment to other organs: if Daydream/WebSense ran this turn,
                # update their coupling matrices too (self-development from daydreaming).
                try:
                    rwd = float(se.get("delta_reward", 0.0) or 0.0)
                except Exception:
                    rwd = 0.0

                # Daydream matrix update: x = applied daydream delta, u = evaluator drive delta.
                try:
                    if isinstance(locals().get("daydream_delta"), dict) and locals().get("daydream_delta") and abs(rwd) > 1e-6:
                        self._apply_matrix_update(
                            "daydream",
                            drives_delta or {},
                            locals().get("daydream_delta") or {},
                            rwd,
                        )
                except Exception:
                    pass

                # WebSense matrix update: x = encoder(input) for the actual websense event payload.
                try:
                    if bool(locals().get("want_web")) and isinstance(locals().get("websense_event_payload"), dict) and abs(rwd) > 1e-6:
                        enc = WebsenseEncoder(self.axis)
                        x_vec, _why = enc.encode(len(self.axis), Event("websense", locals().get("websense_event_payload") or {}))
                        inv = {v: k for k, v in self.axis.items()}
                        x_ws: Dict[str, float] = {}
                        for i, v in enumerate(x_vec or []):
                            if abs(float(v)) < 1e-9:
                                continue
                            nm = inv.get(i)
                            if nm:
                                x_ws[nm] = float(v)
                        if x_ws and drives_delta:
                            self._apply_matrix_update("websense", drives_delta, x_ws, rwd)
                except Exception:
                    pass

                # trust learning from outcome
                for dom in (se.get("domains_penalty") or []):
                    db_update_domain_trust(self.db, str(dom), -abs(float(se.get("delta_reward", -0.2) or -0.2)))
                for dom in (se.get("domains_reward") or []):
                    db_update_domain_trust(self.db, str(dom), abs(float(se.get("delta_reward", 0.2) or 0.2)))

                # beliefs from outcome (only when supported)
                for b in (se.get("beliefs") or []):
                    if not isinstance(b, dict):
                        continue
                    db_add_belief(
                        self.db,
                        str(b.get("subject") or ""),
                        str(b.get("predicate") or ""),
                        str(b.get("object") or ""),
                        float(b.get("confidence", 0.65) or 0.65),
                        str(b.get("provenance") or "self_eval"),
                    )

                # Policy kernel learning from outcome (actions -> reward).
                # This closes the gap: internal organs can mutate a durable action policy.
                try:
                    x_pol = (policy_trace.get('features') if isinstance(policy_trace, dict) else []) or []
                    v_pol = int((policy_trace.get('version') if isinstance(policy_trace, dict) else 1) or 1)
                    r_pol = float(se.get('delta_reward', 0.0) or 0.0)
                    if x_pol and abs(r_pol) > 1e-6:
                        # Only update for auxiliary actions that actually ran.
                        if 'want_web' in locals() and bool(want_web):
                            v_pol = self.policy.apply_update(from_version=v_pol, x=x_pol, action='websense', reward=r_pol, note='selfeval')
                        if 'want_daydream' in locals() and bool(want_daydream):
                            v_pol = self.policy.apply_update(from_version=v_pol, x=x_pol, action='daydream', reward=r_pol, note='selfeval')
                except Exception:
                    pass
            except Exception:
                pass

            # --- Workspace commit (global workspace for consciousness-like broadcast) ---
            try:
                s_now = self.hb.load_state()
                idx_pain = self.axis.get("pain_physical")
                idx_energy = self.axis.get("energy")
                idx_unc = self.axis.get("uncertainty")
                pain_v = float(s_now.values[idx_pain]) if idx_pain is not None and idx_pain < len(s_now.values) else 0.0
                energy_v = float(s_now.values[idx_energy]) if idx_energy is not None and idx_energy < len(s_now.values) else 0.5
                unc_v = float(s_now.values[idx_unc]) if idx_unc is not None and idx_unc < len(s_now.values) else 0.5
                candidates = [
                    {"kind":"question","text": text, "w": 1.0},
                    {"kind":"answer","text": out[:800], "w": 0.9},
                    {"kind":"websense_claims","text": (ws_claims_json or "")[:800], "w": 0.7},
                    {"kind":"state","text": f"pain={pain_v:.2f}, energy={energy_v:.2f}, uncertainty={unc_v:.2f}", "w": 0.6},
                ]
                prev = db_get_workspace_current(self.db)
                arb_model = os.environ.get("BUNNY_MODEL_WORKSPACE") or (os.environ.get("BUNNY_MODEL_DECIDER") or "llama3.2:3b")
                items = arbitrate_workspace(
                    getattr(self.cfg, "host", "http://127.0.0.1:11434"),
                    arb_model,
                    candidates=candidates,
                    prev_items=prev,
                    max_items=int(os.environ.get("BUNNY_WORKSPACE_MAX", "12")),
                )
                db_set_workspace_current(self.db, items, note="user_turn")
            except Exception:
                pass

            # immutable pain integration (health/outcome-driven)
            self._integrate_pain_tick()
            self._integrate_fatigue_tick()

            # --- Salience-based consolidation & soft forgetting ---
            try:
                s_after = self.hb.load_state()
                # Compute salience from internal state shifts and outcome reward.
                def _ax(name: str, st) -> float:
                    idx = self.axis.get(name)
                    if idx is None or idx >= len(st.values):
                        return 0.0
                    try:
                        return float(st.values[idx])
                    except Exception:
                        return 0.0

                pain_b = max(_ax("pain_psych", s_before), _ax("pain_physical", s_before))
                pain_a = max(_ax("pain_psych", s_after), _ax("pain_physical", s_after))
                dpain = abs(pain_a - pain_b)

                # Base salience uses only protected/trusted signals (anti-gaming):
                # - dpain: immutable pain integration shift
                # - |delta_reward|: self-eval / feedback outcome (axiom-aware)
                base_sal = (
                    0.55 * min(1.0, abs(float(delta_reward)))
                    + 0.45 * min(1.0, dpain)
                )
                base_sal = 0.0 if base_sal < 0.0 else 1.0 if base_sal > 1.0 else base_sal

                # Episode tagging (tag-and-capture): high-salience events strengthen nearby context.
                import math
                episode_th = float(os.environ.get("BUNNY_EPISODE_TH", "0.72") or 0.72)
                base_window = int(os.environ.get("BUNNY_EPISODE_WINDOW", "6") or 6)
                tau = float(os.environ.get("BUNNY_EPISODE_TAU", "2.5") or 2.5)

                # Expire old episode context
                ep = self._active_episode if isinstance(getattr(self, "_active_episode", None), dict) else None
                if ep:
                    try:
                        center0 = int(ep.get("center_ui_id", 0) or 0)
                        w0 = int(ep.get("window", 0) or 0)
                        if center0 > 0 and int(user_id) > center0 + w0 + 8:
                            self._active_episode = None
                            ep = None
                    except Exception:
                        self._active_episode = None
                        ep = None

                # Start/refresh episode if the current turn is strongly salient.
                if base_sal >= episode_th:
                    valence = 1 if float(delta_reward) >= 0.0 else -1
                    pain_level = max(pain_a, pain_b)
                    if valence >= 0:
                        w = int(round(float(base_window) * (0.70 + 0.60 * base_sal) * (1.0 - 0.50 * pain_level)))
                    else:
                        w = int(round(float(base_window) * (0.35 + 0.45 * base_sal) * (1.0 - 0.70 * pain_level)))
                    w = max(2, min(10, w))
                    ep_id = f"ep{int(time.time())}_{int(user_id)}"
                    self._active_episode = {
                        "id": ep_id,
                        "center_ui_id": int(user_id),
                        "strength": float(base_sal),
                        "window": int(w),
                        "tau": float(tau),
                        "valence": int(valence),
                    }
                    ep = self._active_episode
                    # Retroactive boost for already existing STM rows around the episode center.
                    try:
                        db_apply_episode_boost(self.db, int(user_id), ep_id, float(base_sal), int(w), float(tau))
                    except Exception:
                        pass

                # Apply episode boost to this turn's STM entries if inside the active window.
                sal_user = float(base_sal)
                sal_reply = float(base_sal)
                ep = self._active_episode if isinstance(getattr(self, "_active_episode", None), dict) else None
                if ep:
                    try:
                        center = int(ep.get("center_ui_id", 0) or 0)
                        w = int(ep.get("window", 0) or 0)
                        strength = float(ep.get("strength", 0.0) or 0.0)
                        t = max(0.25, float(ep.get("tau", tau) or tau))
                        eid = str(ep.get("id", "") or "")
                        if center > 0 and eid and w > 0 and strength > 0.0:
                            for mid, which in ((int(user_id), "user"), (int(reply_id), "reply")):
                                dist = abs(mid - center)
                                if dist <= w:
                                    boost = strength * math.exp(-float(dist) / t)
                                    boost = max(0.0, min(1.0, float(boost)))
                                    if which == "user":
                                        sal_user = max(sal_user, boost)
                                    else:
                                        sal_reply = max(sal_reply, boost)
                                    try:
                                        db_update_memory_short_episode(self.db, mid, eid, int(dist), salience_floor=boost)
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                # Update STM salience for this turn's messages (enables salience-aware pruning).
                active_topic3 = _get_active_topic(db_get_workspace_current(self.db))
                db_update_memory_short_salience(self.db, user_id, sal_user, topic=active_topic3)
                db_update_memory_short_salience(self.db, reply_id, sal_reply, topic=active_topic3)

                # Boost beliefs created during this turn to mimic emotional consolidation.
                db_boost_beliefs_since(self.db, turn_start_iso, max(sal_user, sal_reply))


            except Exception:
                pass

            # Finalize turn: advance heartbeat and publish status.
            try:
                self._enforce_dispute_floor(reason="post_turn")
                self.hb.step()
            except Exception:
                pass
            self.broker.publish("status", db_status(self.db))
            return user_id, reply_id

    def caught(self, message_id: int) -> None:
        with self._lock:
            db_caught_message(self.db, message_id)
            self.broker.publish("message", self._ui_message(message_id))

            # Remove the caught assistant message from short-term replay to avoid re-anchoring on known-wrong text.
            # Generic hygiene: do not feed flagged-bad outputs back into the model.
            try:
                con = self.db.connect()
                cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(memory_short)").fetchall()}
                if "ui_message_id" in cols:
                    con.execute("DELETE FROM memory_short WHERE ui_message_id=?", (int(message_id),))
                    con.commit()
            except Exception:
                try:
                    con.close()
                except Exception:
                    pass

            # Track disputes/corrections (generic epistemic signal).
            try:
                dc = int(db_meta_get(self.db, 'dispute_count', '0') or 0)
            except Exception:
                dc = 0
            try:
                rc = int(db_meta_get(self.db, 'recent_caught', '0') or 0)
            except Exception:
                rc = 0
            db_meta_set(self.db, 'dispute_count', str(dc + 1))
            db_meta_set(self.db, 'recent_caught', str(min(999, rc + 1)))

            # Persist a short "teacher" trace so Speech sees the correction immediately.
            try:
                con = self.db.connect()
                now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                con.execute(
                    "INSERT INTO memory_short(role,content,created_at) VALUES(?,?,?)",
                    (
                        "teacher",
                        "CAUGHT: previous assistant message was flagged wrong. Treat related factual claims as uncertain and prefer WebSense verification before asserting.",
                        now,
                    ),
                )
                con.commit()
            except Exception:
                try:
                    con.close()
                except Exception:
                    pass

            # Treat explicit ❌ as strong negative feedback and learn generically (LLM-based).
            # No hardcoded domain heuristics: we self-evaluate the (question, answer) pair.
            try:
                q, a, ts = db_get_prev_user_for_reply(self.db, int(message_id))
                if q and a:
                    se = self._call_with_health(
                        "selfeval",
                        lambda: evaluate_outcome(
                            self.selfeval_cfg,
                            db_get_axioms(self.db),
                            self._state_summary(),
                            question=q,
                            answer=a,
                            websense_claims_json="",
                            meta_json=json.dumps({"mode": "caught", "websense_available": True, "websense_used": False}, ensure_ascii=False),
                        ),
                    )
                    drives_delta = se.get("drives_delta") if isinstance(se.get("drives_delta"), dict) else {}

                    # Generic epistemic correction: if a message was flagged wrong, increase uncertainty and
                    # seek evidence, unless the evaluator already provided strong guidance.
                    dd = dict(drives_delta or {})
                    # Strong epistemic correction on explicit ❌:
                    # do not allow the evaluator to make us *more* certain here.
                    try:
                        dd['uncertainty'] = max(float(dd.get('uncertainty', 0.0) or 0.0), 0.25)
                    except Exception:
                        dd['uncertainty'] = 0.25
                    try:
                        dd['confidence'] = min(float(dd.get('confidence', 0.0) or 0.0), -0.25)
                    except Exception:
                        dd['confidence'] = -0.25
                    try:
                        dd['pressure_websense'] = max(float(dd.get('pressure_websense', 0.0) or 0.0), 0.35)
                    except Exception:
                        dd['pressure_websense'] = 0.35
                    try:
                        dd['freshness_need'] = max(float(dd.get('freshness_need', 0.0) or 0.0), 0.15)
                    except Exception:
                        dd['freshness_need'] = 0.15
                    # Lock epistemic certainty until grounded evidence resolves the dispute.
                    self._set_dispute_lock(seconds=float(os.environ.get("BUNNY_DISPUTE_LOCK_S", "3600") or 3600))
                    self.hb.enqueue(Event("decision", {"drives": dd, "_mode": "delta"}).with_time())
                    self.hb.step()
                    self._enforce_dispute_floor(reason='caught')
                    x_feats = db_get_decision_drives_before(self.db, ts)
                    try:
                        r_pl = float(se.get("delta_reward", -0.6) or -0.6)
                    except Exception:
                        r_pl = -0.6
                    if dd and x_feats and abs(float(r_pl)) > 1e-9:
                        self._apply_matrix_update("decision", dd, x_feats, float(r_pl))
                    # Update failure clusters (generic, LLM-based) for long-term self-development.
                    try:
                        fc = self._call_with_health(
                            "cluster",
                            lambda: assign_failure_cluster(
                                self.cluster_cfg,
                                db_get_axioms(self.db),
                                user_text=q,
                                last_assistant=a,
                                selfeval=se if isinstance(se, dict) else {},
                                feedback={},
                            ),
                        )
                        db_upsert_failure_cluster(self.db, fc, example={"message_id": int(message_id), "q": q[:500], "a": a[:500]})
                        db_meta_set(self.db, "last_failure_cluster", str(fc.get("cluster_key") or ""))
                    except Exception:
                        pass
            except Exception:
                pass




def _run_sleep_curriculum(self) -> None:
    """Self-training during 'sleep': build shadow matrix candidates from recent failures, test, and promote.

    Also performs self-development steps:
    - failure clustering
    - skill refinement
    - matured DevLab proposals with REAL sandbox test runner
    """
    # Rate-limit via meta key (avoid constant background training).
    try:
        last = float(db_meta_get(self.db, "last_curriculum_ts", "0") or 0.0)
    except Exception:
        last = 0.0
    now = time.time()
    cooldown_s = float(os.getenv("BUNNY_CURRICULUM_COOLDOWN", "900") or 900.0)
    if now - last < cooldown_s:
        return

    # Seeds: recent caught pairs (strong signal).
    pairs = db_list_recent_caught_pairs(self.db, limit=int(os.getenv("BUNNY_CURRICULUM_SEEDS", "10") or 10))
    if not pairs:
        return

    # ---- Self-development: cluster failures + maintain skills + mature proposals (DevLab) ----
    try:
        # Assign clusters for each (q,a) pair and persist cluster stats.
        for p in pairs:
            q0 = str(p.get("question") or "")
            a0 = str(p.get("answer") or "")
            fc = self._call_with_health(
                "cluster",
                lambda q=q0, a=a0: assign_failure_cluster(
                    self.cluster_cfg,
                    db_get_axioms(self.db),
                    user_text=q,
                    last_assistant=a,
                    selfeval={},
                    feedback={"caught": True},
                ),
            )
            db_upsert_failure_cluster(self.db, fc, example={"q": q0[:500], "a": a0[:500]})

        # Build/refine skills for top clusters.
        all_fc = db_list_failure_clusters(self.db, limit=8)
        for fc in all_fc[:3]:
            ck = str(fc.get("cluster_key") or "")
            if not ck or ck == "none":
                continue
            existing = db_get_skill(self.db, ck) or {}
            examples = list(fc.get("examples") or [])[-12:]
            sk = self._call_with_health(
                "skill",
                lambda: build_skill(self.skill_cfg, db_get_axioms(self.db), fc, examples, existing_skill=existing),
            )
            db_upsert_skill(self.db, ck, sk)

            # If cluster persists and skill confidence is low, request DevLab matured proposal.
            if int(fc.get("count") or 0) >= int(os.getenv("BUNNY_DEVBOT_MIN_CLUSTER", "4") or 4) and float(sk.get("confidence") or 0.0) < float(os.getenv("BUNNY_DEVBOT_MIN_SKILL_CONF", "0.65") or 0.65):
                last_dev = float(db_meta_get(self.db, f"devlab_last_{ck}", "0") or 0.0)
                if time.time() - last_dev > float(os.getenv("BUNNY_DEVBOT_COOLDOWN", "7200") or 7200):
                    skills = [db_get_skill(self.db, ck) or {}]
                    dev = self._call_with_health(
                        "devlab",
                        lambda: propose_patch_and_tests(
                            self.devlab_cfg,
                            db_get_axioms(self.db),
                            fc,
                            examples,
                            skills,
                            repo_hint={"project": "FSKI pybunny", "note": "python codebase; propose minimal diffs + tests"},
                        ),
                    )
                    prop_v2 = dev.get("proposal_v2") if isinstance(dev, dict) else {}
                    patch = str(dev.get("patch_unified_diff") or "")
                    cmds = dev.get("test_commands") if isinstance(dev.get("test_commands"), list) else [["python", "-m", "compileall", "."], ["python", "-m", "pytest", "-q"]]

                    # REAL sandbox tests: apply patch in temp dir then run commands.
                    verdict = "proposed"
                    test_out = ""
                    _fx = [{"question": str(e.get("q") or "") , "answer": str(e.get("a") or "")} for e in (examples or [])][-8:]
                    try:
                        res = run_patch_tests_with_pain(self._repo_root(), patch, cmds, fixtures=_fx, axioms=db_get_axioms(self.db), state_summary=self._state_summary())
                        test_out = json.dumps(res, ensure_ascii=False)
                        verdict = "tested_ok" if int(res.get("ok") or 0) == 1 else "tested_fail"
                    except Exception as e:
                        test_out = f"test_runner_error: {type(e).__name__}: {e}"
                        verdict = "not_tested"

                    db_add_devlab_run(self.db, ck, "mature_proposal", patch, cmds, test_out, verdict, meta={"proposal_v2": prop_v2})
                    # Positive reinforcement (A3/A4 progress): small "high" that decays naturally via resources tick.
                    if verdict == "tested_ok":
                        joy = {"valence": 0.06, "confidence": 0.05, "awe": 0.03, "arousal": 0.02, "curiosity": 0.02, "sat_a1": 0.08, "sat_a3": 0.10, "sat_a4": 0.10}
                        self.hb.enqueue(Event("reward_signal", {"_mode":"delta", "drives": joy}).with_time())
                    if isinstance(prop_v2, dict) and prop_v2:
                        db_add_mutation_proposal(self.db, trigger=f"devlab:{ck}", proposal={"proposal_v2": prop_v2, "patch": patch, "test_plan": cmds, "test_result": verdict})
                    db_meta_set(self.db, f"devlab_last_{ck}", str(time.time()))
    except Exception:
        # Never let sleep dev loop crash the server.
        pass

    seeds = [{"question": p["question"], "failure": "caught"} for p in pairs]
    tasks = build_task_variants(self.curriculum_cfg, seeds, variants_per_seed=int(os.getenv("BUNNY_CURRICULUM_VARIANTS", "2") or 2))
    if not tasks:
        return

    # Candidate matrix for 'decision' only (primary lever for A3/A4 behaviors).
    binding = self.reg.get("decision")
    if binding is None:
        return

    # Load current matrix entries.
    try:
        cur = self.store.get_sparse(binding.matrix_name, int(binding.matrix_version))
    except Exception:
        return
    state_dim = int(cur.n_rows)

    # Build candidate entries dict from current.
    ent = {(int(i), int(j)): float(v) for (i, j, v) in (cur.entries or [])}

    # Apply aggregated updates from seeds (use selfeval on the actual failed Q/A).
    eta = float(os.getenv("BUNNY_CURRICULUM_ETA", "0.08") or 0.08)
    decay = float(os.getenv("BUNNY_CURRICULUM_DECAY", "0.002") or 0.002)
    clamp = float(os.getenv("BUNNY_CURRICULUM_CLAMP", "0.25") or 0.25)

    # Protected axes: never learn through non-health/resources matrices.
    protected_names = {"pain", "pain_physical", "pain_psych", "energy", "fatigue", "sleep_pressure"}
    protected_idx = {self.axis.get(n) for n in protected_names if self.axis.get(n) is not None}
    protected_idx.discard(None)

    def _apply_update(u: Dict[str, float], x_feats: Dict[str, float], reward: float) -> None:
        # L2 decay
        if decay > 0.0 and ent:
            for k in list(ent.keys()):
                ent[k] = float(ent[k]) * (1.0 - decay)
                if abs(ent[k]) < 1e-6:
                    ent.pop(k, None)

        # Sparse u, x over axis indices
        u_idx: Dict[int, float] = {}
        for k, v in (u or {}).items():
            idx = self.axis.get(str(k))
            if idx is None:
                continue
            if idx in protected_idx:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if abs(fv) < 1e-6:
                continue
            u_idx[int(idx)] = max(-1.0, min(1.0, fv))

        x_idx: Dict[int, float] = {}
        for k, v in (x_feats or {}).items():
            idx = self.axis.get(str(k))
            if idx is None:
                continue
            if idx in protected_idx:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if abs(fv) < 1e-6:
                continue
            x_idx[int(idx)] = max(-1.0, min(1.0, fv))

        if not u_idx or not x_idx:
            return

        r = max(-1.0, min(1.0, float(reward)))
        for i, ui in u_idx.items():
            for j, xj in x_idx.items():
                dv = eta * r * ui * xj
                if abs(dv) < 1e-7:
                    continue
                key = (int(i), int(j))
                ent[key] = max(-clamp, min(clamp, float(ent.get(key, 0.0)) + dv))
                if abs(ent[key]) < 1e-6:
                    ent.pop(key, None)

    # Do updates
    axioms = db_get_axioms(self.db)
    for p in pairs[: int(os.getenv("BUNNY_CURRICULUM_SEED_UPDATES", "8") or 8)]:
        q = p.get("question") or ""
        a = p.get("answer") or ""
        ts = p.get("ts") or ""
        try:
            se = self._call_with_health("selfeval", lambda: evaluate_outcome(self.selfeval_cfg, axioms, self._state_summary(), question=q, answer=a, websense_claims_json="", meta_json="{}"))
        except Exception:
            se = {}
        drives_delta = se.get("drives_delta") if isinstance(se, dict) else {}
        if not isinstance(drives_delta, dict):
            drives_delta = {}
        try:
            reward = float(se.get("delta_reward", -0.6) if isinstance(se, dict) else -0.6)
        except Exception:
            reward = -0.6
        x_feats = db_get_decision_drives_before(self.db, ts) if ts else db_get_last_decision_drives(self.db)
        _apply_update(drives_delta, x_feats or {}, reward)

    # Create new matrix version in DB (shadow candidate).
    con = self.db.connect()
    try:
        row = con.execute("SELECT MAX(version) AS v FROM matrices WHERE name=?", (binding.matrix_name,)).fetchone()
        vmax = int(row["v"] or 0)
    finally:
        con.close()
    cand_v = vmax + 1
    cand_entries = [(i, j, v) for (i, j), v in ent.items()]
    self.store.put_sparse(
        binding.matrix_name,
        cand_v,
        n_rows=cur.n_rows,
        n_cols=cur.n_cols,
        entries=cand_entries,
        meta={"origin": "sleep_curriculum", "parent": int(binding.matrix_version), "n_tasks": len(tasks)},
        parent_version=int(binding.matrix_version),
    )

    # Evaluate: compare baseline vs candidate on tasks using LLM + SelfEval.
    def _state_from_matrix(mat, x_feats: Dict[str, float]) -> str:
        s0 = self.hb.load_state()
        x = [0.0] * state_dim
        for k, v in (x_feats or {}).items():
            idx = self.axis.get(str(k))
            if idx is None or idx >= state_dim:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            x[int(idx)] = max(-1.0, min(1.0, fv))
        y = mat.apply(x)
        vals = []
        for i in range(state_dim):
            vv = float(s0.values[i]) + float(y[i])
            if vv < 0.0:
                vv = 0.0
            if vv > 1.0:
                vv = 1.0
            vals.append(vv)
        # reuse existing state summary formatter
        return format_state_summary(self.axis, vals)

    try:
        cand = self.store.get_sparse(binding.matrix_name, cand_v)
    except Exception:
        return

    base_total = 0.0
    cand_total = 0.0
    n = 0

    for t in tasks[: int(os.getenv("BUNNY_CURRICULUM_EVAL_TASKS", "12") or 12)]:
        q = str(t.get("question") or "").strip()
        if not q:
            continue
        # Decide drives for this question (generic; no heuristics).
        try:
            d = self._call_with_health("decider", lambda: decide_pressures(self.decider_cfg, axioms, self._state_summary(), user_text=q, workspace_json=json.dumps(db_get_workspace_current(self.db), ensure_ascii=False)))
        except Exception:
            d = {}
        drives = (d.get("drives") if isinstance(d, dict) else {}) or {}
        if not isinstance(drives, dict):
            drives = {}

        base_state = _state_from_matrix(cur, drives)
        cand_state = _state_from_matrix(cand, drives)

        sys_base, user_base = build_speech_prompt(axioms, base_state, q)
        sys_cand, user_cand = build_speech_prompt(axioms, cand_state, q)

        try:
            a_base = ollama_chat(self.cfg, sys_base, user_base)
        except Exception:
            a_base = ""
        try:
            a_cand = ollama_chat(self.cfg, sys_cand, user_cand)
        except Exception:
            a_cand = ""

        try:
            se_base = evaluate_outcome(self.selfeval_cfg, axioms, base_state, question=q, answer=a_base, websense_claims_json="", meta_json="{}")
        except Exception:
            se_base = {}
        try:
            se_cand = evaluate_outcome(self.selfeval_cfg, axioms, cand_state, question=q, answer=a_cand, websense_claims_json="", meta_json="{}")
        except Exception:
            se_cand = {}

        def score(se: Dict[str, Any]) -> float:
            try:
                dr = float(se.get("delta_reward", 0.0) or 0.0)
            except Exception:
                dr = 0.0
            ax = se.get("axiom_scores") if isinstance(se.get("axiom_scores"), dict) else {}
            ev = se.get("eval_scores") if isinstance(se.get("eval_scores"), dict) else {}
            axm = 0.0
            if isinstance(ax, dict) and ax:
                axm = sum(float(ax.get(k, 0.0) or 0.0) for k in ["A1","A2","A3","A4"]) / 4.0
            evm = 0.0
            if isinstance(ev, dict) and ev:
                evm = sum(float(v or 0.0) for v in ev.values()) / max(1.0, float(len(ev)))
            # mix: reward + compliance/quality nudges
            return dr + 0.4*(axm - 0.5) + 0.3*(evm - 0.5)

        s_base = score(se_base if isinstance(se_base, dict) else {})
        s_cand = score(se_cand if isinstance(se_cand, dict) else {})
        base_total += s_base
        cand_total += s_cand
        n += 1

    if n < 3:
        return

    base_avg = base_total / float(n)
    cand_avg = cand_total / float(n)

    # Gate: promote only if candidate is clearly better and doesn't tank A2.
    promote_margin = float(os.getenv("BUNNY_CURRICULUM_PROMOTE_MARGIN", "0.05") or 0.05)
    if cand_avg >= base_avg + promote_margin:
        # bind adapter to candidate version
        b2 = AdapterBinding(
            event_type=binding.event_type,
            encoder_name=binding.encoder_name,
            matrix_name=binding.matrix_name,
            matrix_version=cand_v,
            meta=binding.meta,
        )
        self.reg.upsert(b2)
        db_meta_set(self.db, "last_curriculum_ts", str(now))
    def apply_proposal_patch(self, proposal_id: int, *, confirm: bool) -> Dict[str, Any]:
        """Apply a tested DevLab patch to the working copy (with explicit confirmation).

        Safety:
        - always runs `git apply --check` first
        - only applies if confirm=True and check passes
        - records outcome into mutation_proposals.user_note + status
        """
        repo = self._repo_root()
        con = self.db.connect()
        row = con.execute("SELECT id, proposal_json, status FROM mutation_proposals WHERE id=?", (int(proposal_id),)).fetchone()
        if not row:
            con.close()
            return {"ok": 0, "error": "proposal not found"}

        pid = int(row[0])
        try:
            pj = json.loads(row[1] or "{}")
        except Exception:
            pj = {}
        patch = str((pj.get("patch") or pj.get("patch_unified_diff") or "")).strip()
        if not patch:
            con.close()
            return {"ok": 0, "error": "proposal has no patch"}

        import tempfile, os, subprocess
        with tempfile.TemporaryDirectory(prefix="bunny-apply-") as td:
            pf = os.path.join(td, "patch.diff")
            with open(pf, "w", encoding="utf-8") as f:
                f.write(patch)

            def _run(cmd: List[str]) -> Dict[str, Any]:
                try:
                    cp = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=120)
                    return {"cmd": cmd, "rc": cp.returncode, "stdout": cp.stdout[-20000:], "stderr": cp.stderr[-20000:]}
                except Exception as e:
                    return {"cmd": cmd, "rc": 125, "stdout": "", "stderr": f"{type(e).__name__}: {e}"}

            check = _run(["git", "apply", "--check", pf])
            res: Dict[str, Any] = {"ok": 0, "checked": check, "applied": None, "proposal_id": pid}

            note_lines = []
            note_lines.append(f"apply_check rc={check.get('rc')}")
            if check.get("stderr"):
                note_lines.append(str(check.get('stderr')))

            if int(check.get("rc") or 1) != 0:
                con.execute("UPDATE mutation_proposals SET status=?, user_note=? WHERE id=?", ("apply_failed", "\n".join(note_lines), pid))
                con.commit()
                con.close()
                res["ok"] = 0
                res["error"] = "git apply --check failed"
                return res

            if not confirm:
                con.close()
                res["ok"] = 1
                res["dry_run"] = 1
                return res

            applied = _run(["git", "apply", "--whitespace=nowarn", pf])
            res["applied"] = applied
            note_lines.append(f"apply rc={applied.get('rc')}")
            if applied.get("stderr"):
                note_lines.append(str(applied.get('stderr')))

            if int(applied.get("rc") or 1) == 0:
                con.execute("UPDATE mutation_proposals SET status=?, user_note=? WHERE id=?", ("applied", "\n".join(note_lines), pid))
                con.commit()
                con.close()
                res["ok"] = 1
                # publish status update
                try:
                    self.broker.publish("status", db_status(self.db))
                except Exception:
                    pass
                return res
            else:
                con.execute("UPDATE mutation_proposals SET status=?, user_note=? WHERE id=?", ("apply_failed", "\n".join(note_lines), pid))
                con.commit()
                con.close()
                res["ok"] = 0
                res["error"] = "git apply failed"
                return res

    def _ui_message(self, message_id: int) -> Dict[str, Any]:
        con = self.db.connect()
        try:
            r = con.execute("SELECT id,created_at,kind,text,rating,caught FROM ui_messages WHERE id=?", (int(message_id),)).fetchone()
            if r is None:
                return {}
            return {
                "id": int(r["id"]),
                "created_at": r["created_at"],
                "role": r["kind"],
                "content": r["text"],
                "kind": r["kind"],
                "text": r["text"],
                "rating": None if r["rating"] is None else int(r["rating"]),
                "caught": int(r["caught"] or 0),
            }
        finally:
            con.close()


# -----------------------------
# HTTP handler
# -----------------------------

# ---- Ops console APIs ----

def db_list_matrices_ops(db: DB) -> list[dict]:
    con = db.connect()
    try:
        rows = con.execute("SELECT name, version, created_at, meta_json AS meta FROM matrices ORDER BY name, version DESC").fetchall()
        out=[]
        for r in rows:
            out.append({"name": r["name"], "version": r["version"], "created_at": r["created_at"], "meta": r["meta"]})
        return out
    finally:
        con.close()
def db_list_bindings_ops(db: DB) -> list[dict]:
    """List current adapter bindings (event_type -> matrix_name@version)."""
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT event_type, encoder_name, matrix_name, matrix_version, meta_json, updated_at FROM adapters ORDER BY event_type"
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            try:
                meta = json.loads(r["meta_json"] or "{}")
            except Exception:
                meta = {}
            out.append({
                "event_type": r["event_type"],
                "encoder_name": r["encoder_name"],
                "matrix_name": r["matrix_name"],
                "matrix_version": int(r["matrix_version"]),
                "meta": meta,
                "updated_at": r["updated_at"],
            })
        return out
    finally:
        con.close()



def db_get_matrix_ops(db: DB, name: str, version: int, limit: int = 500, offdiag_only: bool = False) -> dict:
    con = db.connect()
    try:
        m = con.execute(
            "SELECT name, version, created_at, meta_json AS meta FROM matrices WHERE name=? AND version=?",
            (name, version),
        ).fetchone()
        if not m:
            return {"error": "not_found"}

        # Aggregate stats (fast, SQL side)
        total = con.execute(
            "SELECT COUNT(*) AS c FROM matrix_entries WHERE name=? AND version=?",
            (name, version),
        ).fetchone()["c"]
        diag = con.execute(
            "SELECT COUNT(*) AS c FROM matrix_entries WHERE name=? AND version=? AND i=j",
            (name, version),
        ).fetchone()["c"]
        offdiag = int(total) - int(diag)

        agg = con.execute(
            "SELECT MAX(ABS(value)) AS max_abs, AVG(ABS(value)) AS mean_abs, SUM(ABS(value)) AS sum_abs FROM matrix_entries WHERE name=? AND version=?",
            (name, version),
        ).fetchone()
        stats = {
            "count_total": int(total),
            "count_diag": int(diag),
            "count_offdiag": int(offdiag),
            "max_abs": float(agg["max_abs"] or 0.0),
            "mean_abs": float(agg["mean_abs"] or 0.0),
            "sum_abs": float(agg["sum_abs"] or 0.0),
            "sparsity_hint": None,  # unknown without declared dims
        }

        where = "name=? AND version=?"
        args = [name, version]
        if offdiag_only:
            where += " AND i!=j"

        rows = con.execute(
            f"SELECT i, j, value AS v FROM matrix_entries WHERE {where} ORDER BY ABS(v) DESC LIMIT ?",
            (*args, int(limit)),
        ).fetchall()
        entries = [{"i": r["i"], "j": r["j"], "v": r["v"]} for r in rows]

        # Mini heatmap: derive a small index set from the top entries (no assumptions about global dims)
        idx = []
        for e in entries:
            if e["i"] not in idx:
                idx.append(e["i"])
            if e["j"] not in idx:
                idx.append(e["j"])
            if len(idx) >= 16:
                break
        idx = idx[:16]
        valmap = {(e["i"], e["j"]): float(e["v"]) for e in entries}
        heat = []
        if idx:
            for ii in idx:
                row = []
                for jj in idx:
                    row.append(valmap.get((ii, jj), 0.0))
                heat.append(row)

        return {
            "name": m["name"],
            "version": m["version"],
            "created_at": m["created_at"],
            "meta": m["meta"],
            "stats": stats,
            "index": idx,
            "heatmap": heat,
            "entries": entries,
        }
    finally:
        con.close()


def db_get_matrix_stats_ops(db: DB, name: str, version: int) -> dict:
    # convenience for quick refresh without pulling entries
    con = db.connect()
    try:
        m = con.execute(
            "SELECT name, version, created_at, meta_json AS meta FROM matrices WHERE name=? AND version=?",
            (name, version),
        ).fetchone()
        if not m:
            return {"error": "not_found"}
        total = con.execute(
            "SELECT COUNT(*) AS c FROM matrix_entries WHERE name=? AND version=?",
            (name, version),
        ).fetchone()["c"]
        diag = con.execute(
            "SELECT COUNT(*) AS c FROM matrix_entries WHERE name=? AND version=? AND i=j",
            (name, version),
        ).fetchone()["c"]
        offdiag = int(total) - int(diag)
        agg = con.execute(
            "SELECT MAX(ABS(value)) AS max_abs, AVG(ABS(value)) AS mean_abs, SUM(ABS(value)) AS sum_abs FROM matrix_entries WHERE name=? AND version=?",
            (name, version),
        ).fetchone()
        return {
            "name": m["name"],
            "version": m["version"],
            "created_at": m["created_at"],
            "meta": m["meta"],
            "stats": {
                "count_total": int(total),
                "count_diag": int(diag),
                "count_offdiag": int(offdiag),
                "max_abs": float(agg["max_abs"] or 0.0),
                "mean_abs": float(agg["mean_abs"] or 0.0),
                "sum_abs": float(agg["sum_abs"] or 0.0),
                "sparsity_hint": None,
            },
        }
    finally:
        con.close()


def db_get_matrix_diff_ops(db: DB, name: str, a: int, b: int, limit: int = 500, offdiag_only: bool = False) -> dict:
    con = db.connect()
    try:
        ma = con.execute("SELECT name, version, created_at FROM matrices WHERE name=? AND version=?", (name, a)).fetchone()
        mb = con.execute("SELECT name, version, created_at FROM matrices WHERE name=? AND version=?", (name, b)).fetchone()
        if not ma or not mb:
            return {"error": "not_found"}

        def load(ver: int) -> dict:
            rows = con.execute("SELECT i, j, value AS v FROM matrix_entries WHERE name=? AND version=?", (name, ver)).fetchall()
            d = {}
            for r in rows:
                i, j = int(r["i"]), int(r["j"])
                if offdiag_only and i == j:
                    continue
                d[(i, j)] = float(r["v"])
            return d

        da = load(a)
        dbb = load(b)

        keys = set(da.keys()) | set(dbb.keys())
        diffs = []
        for k in keys:
            va = da.get(k, 0.0)
            vb = dbb.get(k, 0.0)
            dv = vb - va
            if dv != 0.0:
                diffs.append((k[0], k[1], va, vb, dv))
        diffs.sort(key=lambda t: abs(t[4]), reverse=True)
        top = diffs[: int(limit)]
        stats = {
            "count_changed": int(len(diffs)),
            "max_abs_delta": float(abs(top[0][4]) if top else 0.0),
        }

        # Build a tiny diff-heatmap over the most affected indices (for Ops UI)
        pairs = [(i,j,dv) for (i,j,va,vb,dv) in top]
        # pick top unique indices by max |Δ|
        score: dict[int,float] = {}
        for i,j,dv in pairs:
            score[i] = max(score.get(i,0.0), abs(dv))
            score[j] = max(score.get(j,0.0), abs(dv))
        idxs = [k for (k,_) in sorted(score.items(), key=lambda t: t[1], reverse=True)[:16]]
        idxs = sorted(set(idxs))
        heat = []
        if idxs:
            dmap = {(i,j):dv for (i,j,va,vb,dv) in top}
            for i in idxs:
                row=[]
                for j in idxs:
                    row.append(float(dmap.get((i,j), 0.0)))
                heat.append(row)
        return {
            "name": name,
            "a": int(a),
            "b": int(b),
            "created_a": ma["created_at"],
            "created_b": mb["created_at"],
            "offdiag_only": bool(offdiag_only),
            "stats": stats,
            "index": idxs,
            "heatmap": heat,
            "diff": [{"i": i, "j": j, "va": va, "vb": vb, "dv": dv} for (i, j, va, vb, dv) in top],
        }
    finally:
        con.close()
def db_list_proposals_ops(db: DB, limit: int = 100) -> list[dict]:
    con = db.connect()
    try:
        # Older DBs don't have a dedicated `title` column. Keep Ops UI stable by
        # deriving a title from proposal_json (preferred) and falling back to trigger.
        rows = con.execute(
            "SELECT id, created_at, status, trigger, proposal_json FROM mutation_proposals ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()

        import json as _json
        out: list[dict] = []
        for r in rows:
            title = ""
            try:
                pj = _json.loads(r["proposal_json"] or "{}")
                title = (pj.get("title") or (pj.get("proposal_v2") or {}).get("title") or "")
            except Exception:
                title = ""
            if not title:
                title = r["trigger"] or "proposal"
            out.append({
                "id": r["id"],
                "created_at": r["created_at"],
                "status": r["status"],
                "title": title,
            })
        return out
    finally:
        con.close()

def db_get_proposal_ops(db: DB, pid: int) -> dict:
    con = db.connect()
    try:
        r = con.execute("SELECT * FROM mutation_proposals WHERE id=?", (int(pid),)).fetchone()
        if not r:
            return {"error": "not_found"}
        d = {k: r[k] for k in r.keys()}
        return d
    finally:
        con.close()

def db_axioms_full_ops(db: DB, limit_per_axiom: int = 50) -> dict:
    con = db.connect()
    try:
        # BunnyCore schema uses axioms(axiom_key,text,updated_at) and
        # axiom_interpretations(axiom_key,kind,key,value,confidence,source_note,updated_at).
        axioms = con.execute(
            "SELECT axiom_key AS key, text, updated_at AS created_at FROM axioms ORDER BY axiom_key ASC"
        ).fetchall()
        dig = {r["axiom_key"]: r["digest"] for r in con.execute("SELECT axiom_key, digest FROM axiom_digests").fetchall()}

        out: list[dict] = []
        for a in axioms:
            key = str(a["key"])
            inter = con.execute(
                """SELECT kind, key, value, confidence, source_note, updated_at
                   FROM axiom_interpretations
                   WHERE axiom_key=?
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (key, int(limit_per_axiom)),
            ).fetchall()

            # Keep the UI contract stable: it expects {created_at,text} items.
            inter_out: list[dict] = []
            for r in inter:
                try:
                    k = str(r["kind"] or "")
                    kk = str(r["key"] or "")
                    vv = str(r["value"] or "")
                    cc = float(r["confidence"] or 0.0)
                    src = str(r["source_note"] or "")
                    ts = str(r["updated_at"] or "")
                    line = f"({k}/{kk}, c={cc:.2f}) {vv}" + (f" [src={src}]" if src else "")
                    inter_out.append({"created_at": ts, "text": line})
                except Exception:
                    continue

            out.append(
                {
                    "key": key,
                    "text": str(a["text"] or ""),
                    "created_at": str(a["created_at"] or ""),
                    "digest": str(dig.get(key, "") or ""),
                    "interpretations": inter_out,
                }
            )

        return {"axioms": out}
    finally:
        con.close()



def db_add_organ_gate_log(db: DB, *, phase: str, organ: str, score: float, threshold: float, want: bool, data: dict | None = None) -> None:
    """Persist an organ gating decision for Ops debugging.

    This is not a heuristic policy. It just records the computed gate values so we can
    verify whether organs are even running (e.g., Daydream) and why.
    """
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO organ_gate_log(created_at,phase,organ,score,threshold,want,data_json) VALUES(?,?,?,?,?,?,?)",
            (
                now_iso(),
                str(phase or ''),
                str(organ or ''),
                float(score),
                float(threshold),
                1 if bool(want) else 0,
                json.dumps(data or {}, ensure_ascii=False),
            ),
        )
        con.commit()
    finally:
        con.close()


def db_organ_gates_recent_ops(db: DB, limit: int = 200) -> list[dict]:
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT id, created_at, phase, organ, score, threshold, want, data_json FROM organ_gate_log ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        out = []
        for r in rows:
            d = {k: r[k] for k in r.keys()}
            try:
                d['data'] = json.loads(d.get('data_json') or '{}')
            except Exception:
                d['data'] = {}
            out.append(d)
        return out
    finally:
        con.close()


def db_health_recent_ops(db: DB, limit: int = 200) -> list[dict]:
    """Return recent health_log rows (all organs, LLM and non-LLM)."""
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT id, created_at, organ, ok, latency_ms, error, metrics_json FROM health_log ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        out = []
        for r in rows:
            d = {k: r[k] for k in r.keys()}
            try:
                d['metrics'] = json.loads(d.get('metrics_json') or '{}')
            except Exception:
                d['metrics'] = {}
            out.append(d)
        return out
    finally:
        con.close()

def db_llm_telemetry_ops(db: DB, limit: int = 200) -> dict:
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT id, organ, model, purpose, started_at, duration_ms, ok, error FROM llm_calls ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        calls=[{k: r[k] for k in r.keys()} for r in rows]
        # aggregate by organ+model
        agg={}
        for c in calls:
            key=(c["organ"], c["model"])
            a=agg.get(key, {"organ": c["organ"], "model": c["model"], "calls":0, "errors":0, "last_at":"", "last_ms":0.0})
            a["calls"] += 1
            if not c["ok"]:
                a["errors"] += 1
            if not a["last_at"]:
                a["last_at"]=c["started_at"]
                a["last_ms"]=c["duration_ms"]
            agg[key]=a
        return {"recent": calls, "aggregate": list(agg.values())}
    finally:
        con.close()


class Handler(BaseHTTPRequestHandler):
    # Windows browsers frequently abort SSE/HTTP connections mid-stream.
    # The stdlib http.server prints a traceback unless we swallow these here.
    def handle(self):
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            return

    server_version = "BunnyUI/0.1"

    def _json(self, obj: Any, status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Any:
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/":
            data = CHAT_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if p.path == "/ops":
            data = OPS_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if p.path == "/api/messages":
            qs = parse_qs(p.query or "")
            limit = 50
            if "limit" in qs:
                try:
                    limit = max(1, min(500, int(qs["limit"][0])))
                except Exception:
                    pass
            self._json(db_list_messages(self.server.kernel.db, limit))
            return

        if p.path == "/api/status":
            self._json(db_status(self.server.kernel.db))
            return


        if p.path == "/api/telemetry":
            qs = parse_qs(p.query or "")
            limit = 200
            if "limit" in qs:
                try:
                    limit = max(1, min(2000, int(qs["limit"][0])))
                except Exception:
                    pass
            self._json(db_llm_telemetry_ops(self.server.kernel.db, limit))
            return


        if p.path == "/api/health_recent":
            qs = parse_qs(p.query or "")
            limit = 200
            if "limit" in qs:
                try:
                    limit = max(1, min(2000, int(qs["limit"][0])))
                except Exception:
                    pass
            self._json(db_health_recent_ops(self.server.kernel.db, limit))
            return

        if p.path == "/api/organ_gates":
            qs = parse_qs(p.query or "")
            limit = 200
            if "limit" in qs:
                try:
                    limit = max(1, min(2000, int(qs["limit"][0])))
                except Exception:
                    pass
            self._json(db_organ_gates_recent_ops(self.server.kernel.db, limit))
            return

        
        if p.path == "/api/bindings":
            self._json(db_list_bindings_ops(self.server.kernel.db))
            return

        if p.path == "/api/matrices":
            self._json(db_list_matrices_ops(self.server.kernel.db))
            return

        if p.path == "/api/matrix":
            qs = parse_qs(p.query or "")
            name = (qs.get("name") or [""])[0]
            version = int((qs.get("version") or ["0"])[0] or "0")
            limit = int((qs.get("limit") or ["500"])[0] or "500")
            offdiag = int((qs.get("offdiag") or ["0"])[0] or "0")
            self._json(db_get_matrix_ops(self.server.kernel.db, name, version, limit, offdiag_only=bool(offdiag)))
            return

        if p.path == "/api/matrix_diff":
            qs = parse_qs(p.query or "")
            name = (qs.get("name") or [""])[0]
            a = int((qs.get("a") or ["0"])[0] or "0")
            b = int((qs.get("b") or ["0"])[0] or "0")
            limit = int((qs.get("limit") or ["500"])[0] or "500")
            offdiag = int((qs.get("offdiag") or ["0"])[0] or "0")
            self._json(db_get_matrix_diff_ops(self.server.kernel.db, name, a, b, limit, offdiag_only=bool(offdiag)))
            return

        if p.path == "/api/matrix_stats":
            qs = parse_qs(p.query or "")
            name = (qs.get("name") or [""])[0]
            version = int((qs.get("version") or ["0"])[0] or "0")
            self._json(db_get_matrix_stats_ops(self.server.kernel.db, name, version))
            return

        if p.path == "/api/proposals":
            qs = parse_qs(p.query or "")
            limit = int((qs.get("limit") or ["100"])[0] or "100")
            self._json(db_list_proposals_ops(self.server.kernel.db, limit))
            return

        if p.path == "/api/proposal":
            qs = parse_qs(p.query or "")
            pid = int((qs.get("id") or ["0"])[0] or "0")
            self._json(db_get_proposal_ops(self.server.kernel.db, pid))
            return

        if p.path == "/api/axioms_full":
            qs = parse_qs(p.query or "")
            limit = int((qs.get("limit") or ["50"])[0] or "50")
            self._json(db_axioms_full_ops(self.server.kernel.db, limit))
            return

        if p.path == "/sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q = self.server.kernel.broker.subscribe()

            def _safe_write(data: bytes) -> bool:
                """Write to SSE stream; return False if client disconnected."""
                try:
                    self.wfile.write(data)
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return False

            try:
                # initial status
                init = f"event: status\ndata: {json.dumps(db_status(self.server.kernel.db), ensure_ascii=False)}\n\n"
                if not _safe_write(init.encode("utf-8")):
                    return

                while True:
                    try:
                        msg = q.get(timeout=25.0)
                    except queue.Empty:
                        # keepalive
                        if not _safe_write(b": keepalive\n\n"):
                            break
                        continue
                    if not _safe_write(msg.encode("utf-8")):
                        break
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                # Client disconnected (common on browser refresh / network change)
                pass
            finally:
                self.server.kernel.broker.unsubscribe(q)
            return


        if p.path == "/api/proposal/apply":
            try:
                pid = int(body.get("proposal_id", 0))
            except Exception:
                self.send_error(400, "bad proposal_id"); return
            confirm = bool(body.get("confirm", False))
            if pid <= 0:
                self.send_error(400, "bad proposal_id"); return
            res = self.server.kernel.apply_proposal_patch(pid, confirm=confirm)
            self._send_json(res)
            return

        self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path)
        body = self._read_json()
        if body is None:
            self.send_error(400, "bad json")
            return

        if p.path == "/api/send":
            text = str(body.get("text", "")).strip()
            if not text:
                self.send_error(400, "empty")
                return
            self.server.kernel.process_user_text(text)
            self._json({"ok": True})
            return

        if p.path == "/api/caught":
            try:
                mid = int(body.get("message_id", 0))
            except Exception:
                self.send_error(400, "bad payload"); return
            if mid <= 0:
                self.send_error(400, "bad payload"); return
            self.server.kernel.caught(mid)
            self.send_response(204); self.end_headers()
            return

        self.send_error(404)


class BunnyHTTPServer(ThreadingHTTPServer):
    # Suppress noisy tracebacks for expected client disconnects (common on Windows).
    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError)):
            return
        return super().handle_error(request, client_address)

    def __init__(self, addr: Tuple[str,int], handler_cls, kernel: Kernel):
        super().__init__(addr, handler_cls)
        self.kernel = kernel


def render_memory_context(items: List[Dict[str, Any]]) -> str:
    # Keep deterministic formatting; no heuristics, only context window.
    if not items:
        return ""
    lines = []
    for it in items:
        role = it.get("role","")
        if role == "assistant":
            prefix = "ASSISTANT"
        else:
            prefix = "USER"
        c = (it.get("content") or "").strip()
        if c:
            lines.append(f"{prefix}: {c}")
    return "\n".join(lines)


def render_memory_long_context(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    lines = []
    for it in items:
        s = (it.get("summary") or "").strip()
        if s:
            lines.append(f"- {s}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())