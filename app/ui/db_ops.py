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

def db_get_axiom_digests(db: DB) -> Dict[str, str]:
    con = db.connect()
    try:
        rows = con.execute("SELECT axiom_key,digest FROM axiom_digests ORDER BY axiom_key").fetchall()
        return {str(r["axiom_key"]): str(r["digest"]) for r in rows}
    except Exception:
        return {}
    finally:
        con.close()

def db_upsert_axiom_digest(db: DB, axiom_key: str, digest: str, checksum: str = "") -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT OR REPLACE INTO axiom_digests(axiom_key,digest,checksum,updated_at) VALUES(?,?,?,?)",
            (str(axiom_key), str(digest), str(checksum or ""), time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())),
        )
        con.commit()
    finally:
        con.close()

def db_group_axiom_interpretations(db: DB, limit_per_axiom: int = 24) -> Dict[str, List[Dict[str, Any]]]:
    con = db.connect()
    try:
        rows = con.execute(
            """SELECT axiom_key, kind, key, value, confidence, source_note, updated_at
               FROM axiom_interpretations
               ORDER BY updated_at DESC, id DESC"""
        ).fetchall()
        out: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            ak = str(r["axiom_key"])
            out.setdefault(ak, [])
            if len(out[ak]) >= int(limit_per_axiom):
                continue
            out[ak].append({
                "kind": str(r["kind"]),
                "key": str(r["key"]),
                "value": str(r["value"]),
                "confidence": float(r["confidence"] or 0.0),
                "source_note": str(r["source_note"] or ""),
                "updated_at": str(r["updated_at"] or ""),
            })
        return out
    except Exception:
        return {}
    finally:
        con.close()

def db_get_axioms(db: DB) -> Dict[str, str]:
    """Load axioms from DB (single source of truth).

    We keep axioms in the DB to prevent UI/code drift and to allow later mutation.
    """
    con = db.connect()
    try:
        rows = con.execute("SELECT axiom_key,text FROM axioms ORDER BY axiom_key").fetchall()
        out: Dict[str, str] = {}
        for r in rows:
            out[str(r["axiom_key"])] = str(r["text"])
        # Prefer compact digests (sleep-generated) to prevent context bloat.
        dig = db_get_axiom_digests(db)
        if dig:
            for ak, d in dig.items():
                if ak in out and d:
                    out[ak] = out[ak].rstrip() + "\n\nInterpretations-Digest:\n" + str(d).strip()
        else:
            # Fallback: attach latest operationalizations/interpretations (bounded).
            try:
                ir = con.execute(
                    """SELECT axiom_key, kind, key, value, confidence, source_note
                       FROM axiom_interpretations
                       ORDER BY updated_at DESC, id DESC"""
                ).fetchall()
                by_ax: Dict[str, List[str]] = {}
                for row in ir:
                    ak = str(row["axiom_key"])
                    line = f"- ({row['kind']}/{row['key']}, c={float(row['confidence'] or 0.0):.2f}) {str(row['value'])}"
                    by_ax.setdefault(ak, []).append(line)
                for ak, lines_ in by_ax.items():
                    if ak in out and lines_:
                        out[ak] = out[ak].rstrip() + "\\n\\nInterpretationen (neueste zuerst):\\n" + "\\n".join(lines_[:6])
            except Exception:
                pass
        return out
    finally:
        con.close()

def db_upsert_axiom_interpretation(
    db: DB,
    axiom_key: str,
    kind: str,
    key: str,
    value: str,
    confidence: float = 0.4,
    source_note: str = "",
) -> None:
    con = db.connect()
    try:
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        con.execute(
            """INSERT OR REPLACE INTO axiom_interpretations(axiom_key,kind,key,value,confidence,source_note,updated_at)
               VALUES(?,?,?,?,?,?,?)""",
            (
                str(axiom_key),
                str(kind or "rewrite"),
                str(key or "latest"),
                str(value or "").strip(),
                float(confidence or 0.0),
                str(source_note or ""),
                now,
            ),
        )
        con.commit()
    finally:
        con.close()

def db_add_decision(db: DB, scope: str, input_text: str, decision: Dict[str, Any]) -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO decision_log(created_at,scope,input_text,decision_json) VALUES(?,?,?,?)",
            (
                time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                scope or "",
                input_text or "",
                json.dumps(decision, ensure_ascii=False),
            ),
        )
        con.commit()
    finally:
        con.close()

def db_get_last_decision_drives(db: DB) -> Dict[str, float]:
    """Return the most recent decision.drives dict (best effort)."""
    con = db.connect()
    try:
        row = con.execute(
            "SELECT decision_json FROM decision_log ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if not row or not row.get("decision_json"):
            return {}
        try:
            obj = json.loads(row["decision_json"] or "{}")
        except Exception:
            return {}
        drives = obj.get("drives") if isinstance(obj, dict) else {}
        if not isinstance(drives, dict):
            return {}
        out: Dict[str, float] = {}
        for k, v in drives.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    finally:
        con.close()

def db_get_last_assistant_text(db: DB) -> str:
    """Return last assistant-like utterance (reply/assistant/auto), used for feedback pairing."""
    con = db.connect()
    try:
        row = con.execute(
            "SELECT text FROM ui_messages WHERE kind IN ('reply','assistant','auto') ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return str(row["text"]) if row and row.get("text") is not None else ""
    finally:
        con.close()

def db_get_prev_user_for_reply(db: DB, reply_id: int) -> tuple[str, str, str]:
    """Return (user_text, reply_text, reply_created_at) for a given reply message id (best effort)."""
    con = db.connect()
    try:
        r = con.execute(
            "SELECT id,created_at,text FROM ui_messages WHERE id=? AND kind='reply'",
            (int(reply_id),),
        ).fetchone()
        if not r:
            return "", "", ""
        reply_text = str(r["text"] or "")
        created_at = str(r["created_at"] or "")
        u = con.execute(
            "SELECT text FROM ui_messages WHERE kind='user' AND id < ? ORDER BY id DESC LIMIT 1",
            (int(reply_id),),
        ).fetchone()
        user_text = str(u["text"] or "") if u else ""
        return user_text, reply_text, created_at
    finally:
        con.close()

def db_get_decision_drives_before(db: DB, created_at_iso: str) -> Dict[str, float]:
    """Return the most recent decision.drives with created_at <= created_at_iso (best effort)."""
    if not created_at_iso:
        return db_get_last_decision_drives(db)
    con = db.connect()
    try:
        row = con.execute(
            "SELECT decision_json FROM decision_log WHERE created_at <= ? ORDER BY rowid DESC LIMIT 1",
            (created_at_iso,),
        ).fetchone()
        if not row or not row.get("decision_json"):
            return {}
        try:
            obj = json.loads(row["decision_json"] or "{}")
        except Exception:
            return {}
        drives = obj.get("drives") if isinstance(obj, dict) else {}
        if not isinstance(drives, dict):
            return {}
        out: Dict[str, float] = {}
        for k, v in drives.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    finally:
        con.close()

def db_add_feedback(db: DB, user_text: str, last_assistant: str, parsed: Dict[str, Any]) -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO feedback_log(user_text,last_assistant,parsed_json,created_at) VALUES(?,?,?,?)",
            (user_text, last_assistant, json.dumps(parsed, ensure_ascii=False), now_iso()),
        )
        con.commit()
    finally:
        con.close()

def db_add_belief(
    db: DB,
    subject: str,
    predicate: str,
    obj: str,
    confidence: float = 0.7,
    provenance: str = "",
    *,
    salience: float = 0.0,
) -> None:
    subject = (subject or "").strip()
    predicate = (predicate or "").strip()
    obj = (obj or "").strip()
    if not (subject and predicate and obj):
        return
    con = db.connect()
    try:
        # best-effort topic anchoring from current workspace
        topic = ""
        try:
            row = con.execute("SELECT items_json FROM workspace_current WHERE id=1").fetchone()
            if row and row["items_json"]:
                items = json.loads(row["items_json"] or "[]") or []
                for it in items:
                    if isinstance(it, dict) and it.get("kind") == "topic" and it.get("active_topic"):
                        topic = str(it.get("active_topic") or "")[:80]
                        break
        except Exception:
            topic = ""

        s = max(0.0, min(1.0, float(salience or 0.0)))
        base_hl = float(os.environ.get('BUNNY_BELIEF_HALF_LIFE_DAYS', '45') or 45)
        hl = max(7.0, min(365.0, base_hl * (1.0 + float(os.environ.get('BUNNY_SALIENCE_HALF_LIFE_GAIN', '1.2') or 1.2) * s)))
        upd = now_iso()

        # migration-safe columns
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(beliefs)").fetchall()}
        except Exception:
            cols = set()

        if {"topic", "salience", "half_life_days", "updated_at"}.issubset(cols):
            con.execute(
                "INSERT INTO beliefs(created_at,subject,predicate,object,confidence,provenance,topic,salience,half_life_days,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    upd,
                    subject[:200],
                    predicate[:60],
                    obj[:600],
                    float(confidence or 0.7),
                    (provenance or "")[:120],
                    topic,
                    float(s),
                    float(hl),
                    upd,
                ),
            )
        else:
            # legacy fallback
            con.execute(
                "INSERT INTO beliefs(created_at,subject,predicate,object,confidence,provenance,topic) VALUES(?,?,?,?,?,?,?)",
                (
                    upd,
                    subject[:200],
                    predicate[:60],
                    obj[:600],
                    float(confidence or 0.7),
                    (provenance or "")[:120],
                    topic,
                ),
            )
        try:
            bid = int(con.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])
            con.execute(
                "INSERT INTO beliefs_history(created_at,belief_id,op,row_json) VALUES(?,?,?,?)",
                (
                    now_iso(),
                    bid,
                    "insert",
                    json.dumps(
                        {
                            "subject": subject[:200],
                            "predicate": predicate[:60],
                            "object": obj[:600],
                            "confidence": float(confidence or 0.7),
                            "provenance": (provenance or "")[:120],
                            "topic": topic,
                            "salience": float(s),
                            "half_life_days": float(hl),
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        except Exception:
            pass
        con.commit()
    finally:
        con.close()


def db_boost_beliefs_since(db: DB, since_iso: str, salience: float, *, conf_gain: float = 0.18) -> int:
    """Boost belief stickiness for beliefs created since `since_iso`.

    This is the generic mechanism for "high emotional" / "high consequence" learning:
    - salience increases belief.salience
    - salience increases half_life_days (slower decay)
    - optionally increases confidence slightly

    Returns number of rows updated.
    """
    s = max(0.0, min(1.0, float(salience or 0.0)))
    if s <= 0.0 or not since_iso:
        return 0
    con = db.connect()
    try:
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(beliefs)").fetchall()}
        except Exception:
            cols = set()
        if not {"salience", "half_life_days", "updated_at"}.issubset(cols):
            return 0
        base_hl = float(os.environ.get('BUNNY_BELIEF_HALF_LIFE_DAYS', '45') or 45)
        hl_gain = float(os.environ.get('BUNNY_SALIENCE_HALF_LIFE_GAIN', '1.2') or 1.2)
        upd = now_iso()

        rows = con.execute(
            "SELECT id,confidence,salience,half_life_days FROM beliefs WHERE created_at >= ?",
            (since_iso,),
        ).fetchall()
        n = 0
        for r in rows:
            bid = int(r["id"])
            c0 = float(r["confidence"] or 0.0)
            s0 = float(r["salience"] or 0.0)
            hl0 = float(r["half_life_days"] or base_hl)
            s1 = max(s0, s)
            hl1 = max(7.0, min(365.0, max(hl0, base_hl * (1.0 + hl_gain * s1))))
            c1 = max(0.0, min(1.0, c0 * (1.0 + float(conf_gain) * s)))
            if abs(c1 - c0) < 1e-9 and abs(s1 - s0) < 1e-9 and abs(hl1 - hl0) < 1e-9:
                continue
            con.execute(
                "UPDATE beliefs SET confidence=?, salience=?, half_life_days=?, updated_at=? WHERE id=?",
                (float(c1), float(s1), float(hl1), upd, bid),
            )
            try:
                con.execute(
                    "INSERT INTO beliefs_history(created_at,belief_id,op,row_json) VALUES(?,?,?,?)",
                    (
                        upd,
                        bid,
                        "boost",
                        json.dumps(
                            {
                                "confidence_before": c0,
                                "confidence_after": c1,
                                "salience_before": s0,
                                "salience_after": s1,
                                "half_life_before": hl0,
                                "half_life_after": hl1,
                                "since_iso": since_iso,
                                "boost_salience": s,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
            except Exception:
                pass
            n += 1
        con.commit()
        return n
    finally:
        con.close()


def db_upsert_belief(
    db: DB,
    subject: str,
    predicate: str,
    obj: str,
    confidence: float = 0.7,
    provenance: str = "",
    *,
    topic: str = "",
    compress: bool = True,
    salience: float = 0.0,
) -> None:
    """Insert or replace a belief by (subject,predicate,object).

    This is used by Sleep/Consolidation to *compress* redundant long-term memory.
    If compress=True, existing matching rows are removed and written to beliefs_history (op=delete).
    """
    subject = (subject or "").strip()
    predicate = (predicate or "").strip()
    obj = (obj or "").strip()
    if not (subject and predicate and obj):
        return
    con = db.connect()
    try:
        # Ensure new columns exist (migration-safe)
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(beliefs)").fetchall()}
            if "topic" not in cols:
                con.execute("ALTER TABLE beliefs ADD COLUMN topic TEXT NOT NULL DEFAULT ''")
                cols.add("topic")
            if "salience" not in cols:
                con.execute("ALTER TABLE beliefs ADD COLUMN salience REAL NOT NULL DEFAULT 0.0")
                cols.add("salience")
            if "half_life_days" not in cols:
                con.execute("ALTER TABLE beliefs ADD COLUMN half_life_days REAL NOT NULL DEFAULT 45.0")
                cols.add("half_life_days")
            if "updated_at" not in cols:
                con.execute("ALTER TABLE beliefs ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")
                cols.add("updated_at")
        except Exception:
            cols = set()

        topic = (topic or "").strip()[:80]

        s = max(0.0, min(1.0, float(salience or 0.0)))
        base_hl = float(os.environ.get('BUNNY_BELIEF_HALF_LIFE_DAYS', '45') or 45)
        hl = max(7.0, min(365.0, base_hl * (1.0 + float(os.environ.get('BUNNY_SALIENCE_HALF_LIFE_GAIN', '1.2') or 1.2) * s)))
        upd = now_iso()

        max_s = s
        max_hl = hl
        if compress:
            rows = con.execute(
                "SELECT id,subject,predicate,object,confidence,provenance,created_at,topic, salience, half_life_days FROM beliefs WHERE subject=? AND predicate=? AND object=? ORDER BY id DESC",
                (subject[:200], predicate[:60], obj[:600]),
            ).fetchall()
            for r in rows:
                bid = int(r["id"])
                try:
                    max_s = max(float(max_s), float(r["salience"] or 0.0))
                except Exception:
                    pass
                try:
                    max_hl = max(float(max_hl), float(r["half_life_days"] or 0.0))
                except Exception:
                    pass
                try:
                    con.execute(
                        "INSERT INTO beliefs_history(created_at,belief_id,op,row_json) VALUES(?,?,?,?)",
                        (
                            now_iso(),
                            bid,
                            "delete",
                            json.dumps(
                                {
                                    "subject": str(r["subject"]),
                                    "predicate": str(r["predicate"]),
                                    "object": str(r["object"]),
                                    "confidence": float(r["confidence"] or 0.0),
                                    "provenance": str(r["provenance"] or ""),
                                    "topic": str(r["topic"] or ""),
                                    "created_at": str(r["created_at"] or ""),
                                    "salience": float(r["salience"] or 0.0) if ("salience" in r.keys()) else 0.0,
                                    "half_life_days": float(r["half_life_days"] or 0.0) if ("half_life_days" in r.keys()) else 0.0,
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    )
                except Exception:
                    pass
                try:
                    con.execute("DELETE FROM beliefs WHERE id=?", (bid,))
                except Exception:
                    pass

        if {"topic", "salience", "half_life_days", "updated_at"}.issubset(cols):
            con.execute(
                "INSERT INTO beliefs(created_at,subject,predicate,object,confidence,provenance,topic,salience,half_life_days,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    upd,
                    subject[:200],
                    predicate[:60],
                    obj[:600],
                    float(confidence or 0.7),
                    (provenance or "")[:120],
                    topic,
                    float(max_s),
                    float(max_hl),
                    upd,
                ),
            )
        else:
            con.execute(
                "INSERT INTO beliefs(created_at,subject,predicate,object,confidence,provenance,topic) VALUES(?,?,?,?,?,?,?)",
                (
                    upd,
                    subject[:200],
                    predicate[:60],
                    obj[:600],
                    float(confidence or 0.7),
                    (provenance or "")[:120],
                    topic,
                ),
            )
        try:
            bid = int(con.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])
            con.execute(
                "INSERT INTO beliefs_history(created_at,belief_id,op,row_json) VALUES(?,?,?,?)",
                (
                    now_iso(),
                    bid,
                    "insert",
                    json.dumps(
                        {
                            "subject": subject[:200],
                            "predicate": predicate[:60],
                            "object": obj[:600],
                            "confidence": float(confidence or 0.7),
                            "provenance": (provenance or "")[:120],
                            "topic": topic,
                            "salience": float(max_s),
                            "half_life_days": float(max_hl),
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        except Exception:
            pass
        con.commit()
    finally:
        con.close()


def db_downrank_belief(
    db: DB,
    subject: str,
    predicate: str,
    obj: str,
    *,
    reason: str = "",
    floor: float = 0.15,
    target: float = 0.25,
    topic: str = "",
) -> None:
    """Lower confidence of a belief by replacing it with a low-confidence version."""
    prov = ("sleep_downgrade" + (":" + reason.strip()[:60] if reason else "")).strip()
    conf = max(float(floor), min(float(target), 1.0))
    db_upsert_belief(db, subject, predicate, obj, confidence=conf, provenance=prov, topic=topic, compress=True)


def db_list_beliefs(db: DB, limit: int = 12, topic: str = "") -> List[Dict[str, Any]]:
    con = db.connect()
    try:
        topic = (topic or "").strip()
        if topic:
            rows = con.execute(
                "SELECT subject,predicate,object,confidence,provenance,created_at,topic FROM beliefs WHERE topic=? ORDER BY id DESC LIMIT ?",
                (topic, int(limit or 12)),
            ).fetchall()
            # If we don't have enough, backfill with recent global beliefs.
            if len(rows) < int(limit or 12):
                more = con.execute(
                    "SELECT subject,predicate,object,confidence,provenance,created_at,topic FROM beliefs ORDER BY id DESC LIMIT ?",
                    (int(limit or 12),),
                ).fetchall()
                # append unique rows by created_at+subject+predicate+object
                seen = {(r["created_at"], r["subject"], r["predicate"], r["object"]) for r in rows}
                for r in more:
                    k = (r["created_at"], r["subject"], r["predicate"], r["object"])
                    if k in seen:
                        continue
                    rows.append(r)
                    seen.add(k)
                    if len(rows) >= int(limit or 12):
                        break
        else:
            rows = con.execute(
                "SELECT subject,predicate,object,confidence,provenance,created_at,topic FROM beliefs ORDER BY id DESC LIMIT ?",
                (int(limit or 12),),
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "subject": str(r["subject"]),
                    "predicate": str(r["predicate"]),
                    "object": str(r["object"]),
                    "confidence": float(r["confidence"]),
                    "provenance": str(r["provenance"]),
                    "topic": str(r["topic"] or ""),
                    "created_at": str(r["created_at"]),
                }
            )
        return out
    finally:
        con.close()


def db_age_and_prune_beliefs(
    db: DB,
    *,
    half_life_days: float = 45.0,
    floor: float = 0.15,
    prune_ttl_days: float = 180.0,
    prune_below: float = 0.18,
) -> None:
    """Soft forgetting for long-term beliefs.

    - Exponential confidence decay over time with half-life `half_life_days`.
    - Prunes very old, low-confidence beliefs to keep the store compact.

    Uses meta.key='belief_decay_last' to avoid over-decay on frequent ticks.
    """
    try:
        half_life_days = float(half_life_days or 0.0)
    except Exception:
        half_life_days = 45.0
    if half_life_days <= 0:
        return
    try:
        floor = float(floor or 0.0)
    except Exception:
        floor = 0.15
    floor = max(0.0, min(1.0, floor))

    try:
        prune_ttl_days = float(prune_ttl_days or 0.0)
    except Exception:
        prune_ttl_days = 180.0
    try:
        prune_below = float(prune_below or 0.0)
    except Exception:
        prune_below = 0.18
    prune_below = max(0.0, min(1.0, prune_below))

    con = db.connect()
    try:
        now = time.time()
        now_iso_str = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now))

        # read last decay timestamp
        last_iso = None
        try:
            r = con.execute("SELECT value FROM meta WHERE key='belief_decay_last'").fetchone()
            if r:
                last_iso = str(r["value"] or "").strip()
        except Exception:
            last_iso = None

        def _parse_iso(s: str) -> float:
            try:
                # format like 2026-02-27T12:34:56Z
                return time.mktime(time.strptime(s, '%Y-%m-%dT%H:%M:%SZ'))
            except Exception:
                return now

        last_t = _parse_iso(last_iso) if last_iso else (now - 86400.0)  # default: 1 day
        dt_days = max(0.0, (now - float(last_t)) / 86400.0)
        if dt_days <= 0.05:  # don't churn for tiny deltas
            return

        # Determine whether per-belief half-life exists.
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(beliefs)").fetchall()}
        except Exception:
            cols = set()
        has_hl = "half_life_days" in cols

        if has_hl:
            rows = con.execute(
                "SELECT id,subject,predicate,object,confidence,provenance,created_at,topic,half_life_days FROM beliefs WHERE confidence > ?",
                (float(floor),),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT id,subject,predicate,object,confidence,provenance,created_at,topic FROM beliefs WHERE confidence > ?",
                (float(floor),),
            ).fetchall()

        for r in rows:
            bid = int(r["id"])
            conf0 = float(r["confidence"] or 0.0)
            hl_row = float(r["half_life_days"] or half_life_days) if has_hl else float(half_life_days)
            hl_row = max(1.0, float(hl_row))
            decay_factor = 0.5 ** (dt_days / hl_row)
            conf1 = max(float(floor), min(1.0, conf0 * float(decay_factor)))
            if abs(conf1 - conf0) < 1e-6:
                continue
            # log history
            try:
                con.execute(
                    "INSERT INTO beliefs_history(created_at,belief_id,op,row_json) VALUES(?,?,?,?)",
                    (
                        now_iso(),
                        bid,
                        "decay",
                        json.dumps(
                            {
                                "subject": str(r["subject"]),
                                "predicate": str(r["predicate"]),
                                "object": str(r["object"]),
                                "confidence_before": conf0,
                                "confidence_after": conf1,
                                "provenance": str(r["provenance"] or ""),
                                "topic": str(r["topic"] or ""),
                                "created_at": str(r["created_at"] or ""),
                                "half_life_days": hl_row,
                                "dt_days": dt_days,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
            except Exception:
                pass
            con.execute("UPDATE beliefs SET confidence=? WHERE id=?", (float(conf1), bid))

        # prune very old low-confidence beliefs
        if prune_ttl_days > 0:
            cutoff = now - float(prune_ttl_days) * 86400.0
            cutoff_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(cutoff))
            old = con.execute(
                "SELECT id,subject,predicate,object,confidence,provenance,created_at,topic FROM beliefs WHERE created_at < ? AND confidence <= ?",
                (cutoff_iso, float(prune_below)),
            ).fetchall()
            for r in old:
                bid = int(r["id"])
                try:
                    con.execute(
                        "INSERT INTO beliefs_history(created_at,belief_id,op,row_json) VALUES(?,?,?,?)",
                        (
                            now_iso(),
                            bid,
                            "prune",
                            json.dumps(
                                {
                                    "subject": str(r["subject"]),
                                    "predicate": str(r["predicate"]),
                                    "object": str(r["object"]),
                                    "confidence": float(r["confidence"] or 0.0),
                                    "provenance": str(r["provenance"] or ""),
                                    "topic": str(r["topic"] or ""),
                                    "created_at": str(r["created_at"] or ""),
                                    "cutoff_iso": cutoff_iso,
                                    "prune_below": prune_below,
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    )
                except Exception:
                    pass
                con.execute("DELETE FROM beliefs WHERE id=?", (bid,))

        # update last decay timestamp
        try:
            con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('belief_decay_last',?)", (now_iso_str,))
        except Exception:
            pass

        con.commit()
    finally:
        con.close()

def db_update_domain_trust(db: DB, domain: str, delta: float, source: str = "") -> None:
    """Update learned trust for a domain.

    No hardcoded heuristics. Delta is typically derived from user feedback (reward/error).
    We keep trust in [0,1] and track n_obs.
    """
    domain = (domain or "").strip().lower()
    if not domain:
        return
    d = float(delta or 0.0)
    # map delta [-1,1] to a small step
    step = max(-0.2, min(0.2, d * 0.1))
    con = db.connect()
    try:
        row = con.execute("SELECT trust,n_obs FROM trust_domains WHERE domain=?", (domain,)).fetchone()
        if row:
            cur = float(row["trust"])
            n = int(row["n_obs"])
        else:
            cur = 0.5
            n = 0
        nxt = max(0.0, min(1.0, cur + step))
        try:
            con.execute(
                "INSERT INTO trust_history(created_at,domain,prev_trust,new_trust,delta,source) VALUES(?,?,?,?,?,?)",
                (now_iso(), domain, float(cur), float(nxt), float(step), (source or "")[:80]),
            )
        except Exception:
            pass
        con.execute(
            "INSERT OR REPLACE INTO trust_domains(domain,trust,n_obs,updated_at) VALUES(?,?,?,?)",
            (domain, nxt, n + 1, now_iso()),
        )
        con.commit()
    finally:
        con.close()

def db_get_domain_trust_map(db: DB, limit: int = 500) -> Dict[str, float]:
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT domain, trust FROM trust_domains ORDER BY trust DESC, n_obs DESC LIMIT ?",
            (int(limit or 500),),
        ).fetchall()
        return {str(r["domain"]): float(r["trust"]) for r in rows}
    finally:
        con.close()

def db_rollback_recent_trust(db: DB, window_s: int = 900) -> int:
    """Rollback recent trust updates by replaying trust_history backwards within a window.

    This is a regression safety net. Returns number of reverted rows.
    """
    cutoff = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - int(window_s)))
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT id,domain,prev_trust FROM trust_history WHERE created_at >= ? ORDER BY id DESC",
            (cutoff,),
        ).fetchall()
        n = 0
        for r in rows:
            dom = str(r["domain"])
            prev = float(r["prev_trust"])
            con.execute(
                "UPDATE trust_domains SET trust=?, updated_at=? WHERE domain=?",
                (prev, now_iso(), dom),
            )
            n += 1
        con.commit()
        return n
    finally:
        con.close()

def db_rollback_recent_beliefs(db: DB, window_s: int = 900) -> int:
    """Rollback recent belief inserts within a time window (soft delete).

    We keep the history table as audit.
    """
    cutoff = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - int(window_s)))
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT id, belief_id FROM beliefs_history WHERE op='insert' AND created_at >= ? ORDER BY id DESC",
            (cutoff,),
        ).fetchall()
        n = 0
        for r in rows:
            bid = int(r["belief_id"])
            # soft delete by copying to history and removing row
            row = con.execute(
                "SELECT subject,predicate,object,confidence,provenance,topic,created_at FROM beliefs WHERE id=?",
                (bid,),
            ).fetchone()
            if row:
                con.execute(
                    "INSERT INTO beliefs_history(created_at,belief_id,op,row_json) VALUES(?,?,?,?)",
                    (
                        now_iso(),
                        bid,
                        "delete",
                        json.dumps(dict(row), ensure_ascii=False),
                    ),
                )
                con.execute("DELETE FROM beliefs WHERE id=?", (bid,))
                n += 1
        con.commit()
        return n
    finally:
        con.close()

def db_add_resources_log(db: DB, metrics: Dict[str, Any]) -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO resources_log(created_at,metrics_json) VALUES(?,?)",
            (now_iso(), json.dumps(metrics or {}, ensure_ascii=False)),
        )
        con.commit()
    finally:
        con.close()

def db_add_health_log(db: DB, organ: str, ok: bool, latency_ms: float, error: str = "", metrics: Dict[str, Any] | None = None) -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO health_log(created_at,organ,ok,latency_ms,error,metrics_json) VALUES(?,?,?,?,?,?)",
            (
                now_iso(),
                str(organ or ""),
                1 if ok else 0,
                float(latency_ms or 0.0),
                (str(error or "")[:400] if error else ""),
                json.dumps(metrics or {}, ensure_ascii=False),
            ),
        )
        con.commit()
    finally:
        con.close()

def db_meta_get(db: DB, key: str, default: str = "") -> str:
    con = db.connect()
    try:
        row = con.execute("SELECT value FROM meta WHERE key=?", (str(key),)).fetchone()
        return str(row["value"]) if row and row.get("value") is not None else str(default)
    finally:
        con.close()

def db_meta_set(db: DB, key: str, value: str) -> None:
    con = db.connect()
    try:
        con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (str(key), str(value)))
        con.commit()
    finally:
        con.close()

def db_list_recent_caught_pairs(db: DB, limit: int = 12) -> List[Dict[str, str]]:
    """Return recent (question, answer) pairs where the reply was marked caught."""
    con = db.connect()
    out: List[Dict[str, str]] = []
    try:
        rows = con.execute(
            "SELECT id,created_at,text FROM ui_messages WHERE kind='reply' AND caught=1 ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        for r in rows:
            rid = int(r["id"])
            try:
                q, a, ts = db_get_prev_user_for_reply(db, rid)
            except Exception:
                q, a, ts = "", "", ""
            if q and a:
                out.append({"question": q, "answer": a, "ts": ts})
        return out
    finally:
        con.close()

def db_get_workspace_current(db: DB) -> list[dict]:
    con = db.connect()
    try:
        row = con.execute("SELECT items_json FROM workspace_current WHERE id=1").fetchone()
        if not row:
            return []
        try:
            return json.loads(row["items_json"] or "[]") or []
        except Exception:
            return []
    finally:
        con.close()

def db_set_workspace_current(db: DB, items: list[dict], note: str = "") -> None:
    payload = json.dumps(items or [], ensure_ascii=False)
    con = db.connect()
    try:
        con.execute(
            "INSERT OR REPLACE INTO workspace_current(id,items_json,updated_at) VALUES(1,?,?)",
            (payload, now_iso()),
        )
        con.execute(
            "INSERT INTO workspace_log(created_at,items_json,note) VALUES(?,?,?)",
            (now_iso(), payload, str(note or "")[:200]),
        )
        con.commit()
    finally:
        con.close()

def db_get_needs_current(db: DB) -> dict:
    con = db.connect()
    try:
        row = con.execute('SELECT json FROM needs_current WHERE id=1').fetchone()
        if not row:
            return {}
        try:
            return json.loads(row['json'] or '{}') or {}
        except Exception:
            return {}
    finally:
        con.close()

def db_set_needs_current(db: DB, payload: dict) -> None:
    con = db.connect()
    try:
        con.execute(
            'INSERT OR REPLACE INTO needs_current(id,json,updated_at) VALUES(1,?,?)',
            (json.dumps(payload or {}, ensure_ascii=False), now_iso()),
        )
        con.commit()
    finally:
        con.close()

def db_get_wishes_current(db: DB) -> dict:
    con = db.connect()
    try:
        row = con.execute('SELECT json FROM wishes_current WHERE id=1').fetchone()
        if not row:
            return {}
        try:
            return json.loads(row['json'] or '{}') or {}
        except Exception:
            return {}
    finally:
        con.close()

def db_set_wishes_current(db: DB, payload: dict) -> None:
    con = db.connect()
    try:
        con.execute(
            'INSERT OR REPLACE INTO wishes_current(id,json,updated_at) VALUES(1,?,?)',
            (json.dumps(payload or {}, ensure_ascii=False), now_iso()),
        )
        con.commit()
    finally:
        con.close()

def db_upsert_topic(db: DB, topic: str, weight_delta: float = 0.0) -> None:
    topic = (topic or '').strip()[:80]
    if not topic:
        return
    con = db.connect()
    try:
        row = con.execute('SELECT weight FROM topics WHERE topic=?', (topic,)).fetchone()
        if not row:
            con.execute('INSERT INTO topics(topic,weight,created_at,updated_at) VALUES(?,?,?,?)', (topic, 0.5, now_iso(), now_iso()))
        else:
            w = float(row['weight'] or 0.5) + float(weight_delta or 0.0)
            w = max(0.0, min(1.0, w))
            con.execute('UPDATE topics SET weight=?, updated_at=? WHERE topic=?', (w, now_iso(), topic))
        con.commit()
    finally:
        con.close()

def db_open_episode_if_needed(db: DB, active_topic: str) -> None:
    active_topic = (active_topic or '').strip()[:80]
    if not active_topic:
        return
    con = db.connect()
    try:
        # find open episode
        row = con.execute("SELECT id, topic, ended_at FROM episodes WHERE ended_at='' ORDER BY id DESC LIMIT 1").fetchone()
        if row and str(row['topic'] or '') == active_topic:
            return
        # close previous if open and different
        if row:
            con.execute("UPDATE episodes SET ended_at=? WHERE id=?", (now_iso(), int(row["id"])))
        # open new
        con.execute("INSERT INTO episodes(topic,started_at,ended_at,summary,created_at) VALUES(?,?,?,?,?)", (active_topic, now_iso(), '', '', now_iso()))
        con.commit()
    except Exception:
        try:
            con.rollback()
        except Exception:
            pass
    finally:
        con.close()

def db_add_sleep_log(db: DB, summary: dict) -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO sleep_log(created_at,summary_json) VALUES(?,?)",
            (now_iso(), json.dumps(summary or {}, ensure_ascii=False)),
        )
        con.commit()
    finally:
        con.close()

def db_add_matrix_update_log(
    db: DB,
    event_type: str,
    matrix_name: str,
    from_version: int,
    to_version: int,
    reward: float,
    delta_frob: float,
    pain_before: float = 0.0,
    pain_after: float = 0.0,
    rolled_back: int = 0,
    rollback_at: str = "",
    rollback_notes: str = "",
    notes: str = "",
) -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO matrix_update_log(created_at,event_type,matrix_name,from_version,to_version,reward,delta_frob,"
            "pain_before,pain_after,rolled_back,rollback_at,rollback_notes,notes) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                now_iso(),
                str(event_type or ""),
                str(matrix_name or ""),
                int(from_version),
                int(to_version),
                float(reward or 0.0),
                float(delta_frob or 0.0),
                float(pain_before or 0.0),
                float(pain_after or 0.0),
                int(rolled_back or 0),
                str(rollback_at or ""),
                (str(rollback_notes or "")[:200]),
                (str(notes or "")[:200]),
            ),
        )
        con.commit()
    finally:
        con.close()

def db_recent_health_stats(db: DB, window_s: int = 300) -> Dict[str, Any]:
    """Return aggregate health stats over a recent time window (immutable pain model inputs)."""
    con = db.connect()
    try:
        # SQLite time comparisons: store ISO Z; compare lexicographically by created_at.
        # We compute cutoff in UTC.
        cutoff = time.time() - float(window_s or 300)
        cutoff_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(cutoff))
        rows = con.execute(
            "SELECT organ, ok, latency_ms FROM health_log WHERE created_at>=?",
            (cutoff_iso,),
        ).fetchall()
        if not rows:
            return {"n": 0, "err_rate": 0.0, "lat_ms_p95": 0.0, "by_organ": {}}
        oks = [int(r["ok"]) for r in rows]
        lats = [float(r["latency_ms"]) for r in rows]
        err_rate = 1.0 - (sum(oks) / max(1, len(oks)))
        lats_sorted = sorted(lats)
        p95 = lats_sorted[int(0.95 * (len(lats_sorted) - 1))] if lats_sorted else 0.0
        by: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            o = str(r["organ"])
            by.setdefault(o, {"n": 0, "ok": 0, "lat": []})
            by[o]["n"] += 1
            by[o]["ok"] += int(r["ok"])
            by[o]["lat"].append(float(r["latency_ms"]))
        by_organ = {}
        for o, d in by.items():
            lat_list = sorted(d["lat"]) if d.get("lat") else []
            op95 = lat_list[int(0.95 * (len(lat_list) - 1))] if lat_list else 0.0
            by_organ[o] = {
                "n": int(d["n"]),
                "err_rate": float(1.0 - (d["ok"] / max(1, d["n"]))),
                "lat_ms_p95": float(op95),
            }
        return {"n": len(rows), "err_rate": float(err_rate), "lat_ms_p95": float(p95), "by_organ": by_organ}
    finally:
        con.close()

def db_recent_user_msg_rate(db: DB, window_s: int = 300) -> float:
    """Approx. user messages per minute over a window."""
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT COUNT(*) AS c FROM ui_messages WHERE kind='user' AND created_at >= datetime('now', ?)",
            (f"-{int(window_s)} seconds",),
        ).fetchone()
        c = int(rows["c"] or 0)
        minutes = max(1e-6, float(window_s) / 60.0)
        return float(c) / minutes
    finally:
        con.close()

def db_recent_matrix_delta(db: DB, window_s: int = 300) -> float:
    con = db.connect()
    try:
        cutoff = time.time() - float(window_s or 300)
        cutoff_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(cutoff))
        row = con.execute(
            "SELECT COALESCE(SUM(ABS(delta_frob)),0.0) AS s FROM matrix_update_log WHERE created_at>=?",
            (cutoff_iso,),
        ).fetchone()
        return float(row["s"] or 0.0) if row else 0.0
    finally:
        con.close()

def db_get_last_unrolled_matrix_update(db: DB, window_s: int = 900) -> Optional[Dict[str, Any]]:
    con = db.connect()
    try:
        cutoff = time.time() - float(window_s or 900)
        cutoff_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(cutoff))
        row = con.execute(
            """SELECT id, created_at, event_type, matrix_name, from_version, to_version, reward, delta_frob, pain_before
               FROM matrix_update_log
               WHERE created_at>=? AND rolled_back=0
               ORDER BY id DESC LIMIT 1""",
            (cutoff_iso,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        con.close()

def db_mark_matrix_rolled_back(db: DB, row_id: int, note: str) -> None:
    con = db.connect()
    try:
        con.execute(
            "UPDATE matrix_update_log SET rolled_back=1, rollback_at=?, rollback_notes=? WHERE id=?",
            (now_iso(), (str(note or "")[:200]), int(row_id)),
        )
        con.commit()
    finally:
        con.close()

def db_add_evidence_log(db: DB, query: str, question: str, evidence: Dict[str, Any]) -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO evidence_log(created_at,query,question,evidence_json) VALUES(?,?,?,?)",
            (now_iso(), query or "", question or "", json.dumps(evidence or {}, ensure_ascii=False)),
        )
        con.commit()
    finally:
        con.close()

def db_add_self_model(db: DB, model_json: Dict[str, Any]) -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO self_model(created_at,model_json) VALUES(?,?)",
            (now_iso(), json.dumps(model_json or {}, ensure_ascii=False)),
        )
        con.commit()
    finally:
        con.close()

def db_get_last_self_model(db: DB) -> Dict[str, Any]:
    con = db.connect()
    try:
        row = con.execute(
            "SELECT model_json FROM self_model ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["model_json"] or "{}")
        except Exception:
            return {}
    finally:
        con.close()

def db_add_mutation_proposal(db: DB, trigger: str, proposal: Dict[str, Any]) -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO mutation_proposals(created_at,trigger,proposal_json,status,user_note) VALUES(?,?,?,?,?)",
            (now_iso(), trigger or "", json.dumps(proposal or {}, ensure_ascii=False), "proposed", ""),
        )
        con.commit()
    finally:
        con.close()

def db_get_latest_open_mutation_proposal(db: DB, window_s: float = 24 * 3600) -> Dict[str, Any] | None:
    """Return newest proposal with status='proposed' within window_s."""
    con = db.connect()
    try:
        # SQLite stores created_at as ISO UTC; order by id is fine.
        row = con.execute(
            "SELECT id,created_at,trigger,proposal_json,status,user_note FROM mutation_proposals WHERE status='proposed' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        # time window check (best-effort)
        try:
            import calendar
            ts = time.strptime(str(row["created_at"]), '%Y-%m-%dT%H:%M:%SZ')
            age_s = time.time() - calendar.timegm(ts)
            if float(age_s) > float(window_s or 0.0):
                return None
        except Exception:
            pass
        try:
            pj = json.loads(row["proposal_json"] or "{}")
        except Exception:
            pj = {"_raw": (row["proposal_json"] or "")[:4000]}
        return {
            "id": int(row["id"]),
            "created_at": str(row["created_at"]),
            "trigger": str(row["trigger"]),
            "status": str(row["status"]),
            "user_note": str(row["user_note"] or ""),
            "proposal": pj,
        }
    finally:
        con.close()

def db_update_mutation_proposal(db: DB, proposal_id: int, proposal: Dict[str, Any], *, note: str = "") -> None:
    con = db.connect()
    try:
        con.execute(
            "UPDATE mutation_proposals SET proposal_json=?, user_note=? WHERE id=?",
            (
                json.dumps(proposal or {}, ensure_ascii=False),
                (str(note or "")[:200]),
                int(proposal_id),
            ),
        )
        con.commit()
    finally:
        con.close()

def db_list_mutation_proposals(db: DB, limit: int = 6) -> List[Dict[str, Any]]:
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT id,created_at,trigger,proposal_json,status,user_note FROM mutation_proposals ORDER BY id DESC LIMIT ?",
            (int(limit or 6),),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                pj = json.loads(r["proposal_json"] or "{}")
            except Exception:
                pj = {"_raw": (r["proposal_json"] or "")[:500]}
            out.append(
                {
                    "id": int(r["id"]),
                    "created_at": str(r["created_at"]),
                    "trigger": str(r["trigger"]),
                    "status": str(r["status"]),
                    "proposal": pj,
                }
            )
        out.reverse()
        return out
    finally:
        con.close()

def db_upsert_failure_cluster(db: DB, cluster: Dict[str, Any], example: Dict[str, Any] | None = None) -> None:
    ck = str(cluster.get("cluster_key") or "none")
    if ck == "none":
        return
    con = db.connect()
    try:
        row = con.execute("SELECT cluster_key,label,count,examples_json,stats_json FROM failure_clusters WHERE cluster_key=?", (ck,)).fetchone()
        now = now_iso()
        label = str(cluster.get("label") or ck)[:120]
        sev = float(cluster.get("severity") or 0.0)
        if row:
            cnt = int(row["count"] or 0) + 1
            try:
                examples = json.loads(row["examples_json"] or "[]")
            except Exception:
                examples = []
            if example:
                examples.append(example)
                examples = examples[-20:]
            try:
                stats = json.loads(row["stats_json"] or "{}")
            except Exception:
                stats = {}
            stats["severity_ema"] = float(stats.get("severity_ema") or sev) * 0.8 + sev * 0.2
            con.execute(
                "UPDATE failure_clusters SET label=?,count=?,last_seen=?,examples_json=?,stats_json=? WHERE cluster_key=?",
                (label, cnt, now, json.dumps(examples, ensure_ascii=False), json.dumps(stats, ensure_ascii=False), ck),
            )
        else:
            examples = [example] if example else []
            stats = {"severity_ema": sev}
            con.execute(
                "INSERT INTO failure_clusters(cluster_key,label,embedding_json,count,last_seen,examples_json,stats_json) VALUES(?,?,?,?,?,?,?)",
                (ck, label, "[]", 1, now, json.dumps(examples, ensure_ascii=False), json.dumps(stats, ensure_ascii=False)),
            )
        con.commit()
    finally:
        con.close()

def db_list_failure_clusters(db: DB, limit: int = 20) -> List[Dict[str, Any]]:
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT cluster_key,label,count,last_seen,examples_json,stats_json FROM failure_clusters ORDER BY count DESC, last_seen DESC LIMIT ?",
            (int(limit or 20),),
        ).fetchall()
        out = []
        for r in rows:
            try:
                ex = json.loads(r["examples_json"] or "[]")
            except Exception:
                ex = []
            try:
                st = json.loads(r["stats_json"] or "{}")
            except Exception:
                st = {}
            out.append({
                "cluster_key": str(r["cluster_key"]),
                "label": str(r["label"]),
                "count": int(r["count"] or 0),
                "last_seen": str(r["last_seen"]),
                "examples": ex,
                "stats": st,
            })
        return out
    finally:
        con.close()

def db_get_skill(db: DB, cluster_key: str) -> Dict[str, Any] | None:
    con = db.connect()
    try:
        r = con.execute("SELECT skill_key,cluster_key,strategy_digest,tests_json,confidence,updated_at FROM skills WHERE cluster_key=?", (cluster_key,)).fetchone()
        if not r:
            return None
        try:
            tests = json.loads(r["tests_json"] or "[]")
        except Exception:
            tests = []
        return {
            "skill_key": str(r["skill_key"]),
            "cluster_key": str(r["cluster_key"]),
            "strategy_digest": str(r["strategy_digest"] or ""),
            "tests": tests,
            "confidence": float(r["confidence"] or 0.4),
            "updated_at": str(r["updated_at"]),
        }
    finally:
        con.close()

def db_upsert_skill(db: DB, cluster_key: str, skill_obj: Dict[str, Any]) -> None:
    if not cluster_key or cluster_key == "none":
        return
    con = db.connect()
    try:
        now = now_iso()
        skey = f"skill_{cluster_key}"
        con.execute(
            "INSERT OR REPLACE INTO skills(skill_key,cluster_key,strategy_digest,tests_json,confidence,updated_at) VALUES(?,?,?,?,?,?)",
            (
                skey,
                cluster_key,
                str(skill_obj.get('strategy_digest') or ''),
                json.dumps(skill_obj.get('tests') or [], ensure_ascii=False),
                float(skill_obj.get('confidence') or 0.4),
                now,
            ),
        )
        con.commit()
    finally:
        con.close()

def db_add_devlab_run(db: DB, cluster_key: str, intent: str, patch_diff: str, test_plan: List[Any], test_output: str, verdict: str, meta: Dict[str, Any] | None = None) -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO devlab_runs(created_at,cluster_key,intent,patch_diff,test_plan_json,test_output,verdict,meta_json) VALUES(?,?,?,?,?,?,?,?)",
            (
                now_iso(),
                cluster_key,
                intent[:120],
                (patch_diff or '')[:20000],
                json.dumps(test_plan or [], ensure_ascii=False),
                (test_output or '')[-8000:],
                verdict[:120],
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        con.commit()
    finally:
        con.close()

def db_get_mutation_proposal(db: DB, proposal_id: int) -> Dict[str, Any] | None:
    con = db.connect()
    try:
        row = con.execute(
            "SELECT id,created_at,trigger,proposal_json,status,user_note FROM mutation_proposals WHERE id=?",
            (int(proposal_id),),
        ).fetchone()
        if row is None:
            return None
        try:
            pj = json.loads(row["proposal_json"] or "{}")
        except Exception:
            pj = {"_raw": (row["proposal_json"] or "")[:4000]}
        return {
            "id": int(row["id"]),
            "created_at": str(row["created_at"]),
            "trigger": str(row["trigger"]),
            "status": str(row["status"]),
            "user_note": str(row["user_note"] or ""),
            "proposal": pj,
        }
    finally:
        con.close()


# -----------------------------
# Resource / energy metrics (stdlib only)
# -----------------------------

def db_add_daydream(db: DB, trigger: str, state_json: Dict[str, Any], output_json: Dict[str, Any]) -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO daydream_log(created_at,trigger,state_json,output_json) VALUES(?,?,?,?)",
            (
                time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                trigger or "",
                json.dumps(state_json or {}, ensure_ascii=False),
                json.dumps(output_json or {}, ensure_ascii=False),
            ),
        )
        con.commit()
    finally:
        con.close()


# -----------------------------
# UI DB helpers
# -----------------------------

def _prune_memory_short(con, *, max_rows: int = 800, ttl_days: float = 30.0) -> None:
    """Soft forgetting for short-term memory.

    Keeps at most `max_rows` newest rows and drops very old rows beyond `ttl_days`.
    This prevents unbounded growth while still allowing recency-based context.
    """
    try:
        max_rows = int(max(0, int(max_rows or 0)))
    except Exception:
        max_rows = 800
    try:
        ttl_days = float(ttl_days or 0.0)
    except Exception:
        ttl_days = 30.0

    # TTL prune
    if ttl_days > 0:
        cutoff = time.time() - float(ttl_days) * 86400.0
        cutoff_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(cutoff))
        con.execute("DELETE FROM memory_short WHERE created_at < ?", (cutoff_iso,))

    # Count prune (keep last max_rows). Prefer retaining high-salience items.
    if max_rows > 0:
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(memory_short)").fetchall()}
        except Exception:
            cols = set()

        if "salience" in cols:
            rows = con.execute(
                "SELECT id, created_at, salience FROM memory_short ORDER BY id DESC"
            ).fetchall()
            if rows and len(rows) > max_rows:
                # Retention score combines salience (dominant) + recency.
                n = float(max(1, len(rows) - 1))
                scored = []
                for rank, r in enumerate(rows):
                    rid = int(r["id"])
                    sal = float(r["salience"] or 0.0)
                    rec = 1.0 - (float(rank) / n)  # newest ~1
                    score = 0.75 * max(0.0, min(1.0, sal)) + 0.25 * max(0.0, min(1.0, rec))
                    scored.append((score, rid))
                scored.sort(reverse=True)
                keep = {rid for _, rid in scored[: int(max_rows)]}
                drop = [rid for _, rid in scored[int(max_rows) :]]
                if drop:
                    # delete in chunks
                    for i in range(0, len(drop), 250):
                        chunk = drop[i : i + 250]
                        q = ",".join(["?"] * len(chunk))
                        con.execute(f"DELETE FROM memory_short WHERE id IN ({q})", tuple(chunk))
        else:
            # Fallback for older DBs: keep last max_rows by id threshold.
            row = con.execute(
                "SELECT id FROM memory_short ORDER BY id DESC LIMIT 1 OFFSET ?",
                (int(max_rows),),
            ).fetchone()
            if row and row["id"] is not None:
                thr = int(row["id"])
                con.execute("DELETE FROM memory_short WHERE id <= ?", (thr,))

def db_add_message(db: DB, kind: str, text: str) -> int:
    con = db.connect()
    try:
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        cur = con.execute(
            "INSERT INTO ui_messages(created_at,kind,text) VALUES(?,?,?)",
            (now, kind, text),
        )
        ui_mid = int(cur.lastrowid)
        # Short-term memory stream (like main: keep raw dialogue; summarization can be a separate organ).
        if kind in ("user","reply"):
            role = "user" if kind == "user" else "assistant"
            # best-effort topic anchoring
            topic = ""
            try:
                row = con.execute("SELECT items_json FROM workspace_current WHERE id=1").fetchone()
                if row and row["items_json"]:
                    items = json.loads(row["items_json"] or "[]") or []
                    for it in items:
                        if isinstance(it, dict) and it.get("kind") == "topic" and it.get("active_topic"):
                            topic = str(it.get("active_topic") or "")[:80]
                            break
            except Exception:
                topic = ""

            # Migration-safe insert (older DBs may not have the new columns)
            try:
                cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(memory_short)").fetchall()}
            except Exception:
                cols = set()
            if {"ui_message_id", "topic", "salience"}.issubset(cols):
                # Support optional episode tagging columns.
                if {"episode_id", "episode_dist"}.issubset(cols):
                    con.execute(
                        "INSERT INTO memory_short(role,content,created_at,ui_message_id,topic,salience,episode_id,episode_dist) VALUES(?,?,?,?,?,?,?,?)",
                        (role, text, now, ui_mid, topic, 0.0, "", 0),
                    )
                else:
                    con.execute(
                        "INSERT INTO memory_short(role,content,created_at,ui_message_id,topic,salience) VALUES(?,?,?,?,?,?)",
                        (role, text, now, ui_mid, topic, 0.0),
                    )
            else:
                con.execute(
                    "INSERT INTO memory_short(role,content,created_at) VALUES(?,?,?)",
                    (role, text, now),
                )
            # Soft forgetting: prune short memory (count + TTL)
            try:
                max_rows = int(os.environ.get('BUNNY_MEMORY_SHORT_MAX', '800') or 800)
                ttl_days = float(os.environ.get('BUNNY_MEMORY_SHORT_TTL_DAYS', '30') or 30)
                _prune_memory_short(con, max_rows=max_rows, ttl_days=ttl_days)
            except Exception:
                pass
        con.commit()
        return ui_mid
    finally:
        con.close()

def db_list_messages(db: DB, limit: int) -> List[Dict[str, Any]]:
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT id,created_at,kind,text,rating,caught FROM ui_messages ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        out = []
        for r in rows[::-1]:
            # Backward/forward compatible payload:
            # - UI HTML uses role/content (old Go UI convention)
            # - internal code stores kind/text in ui_messages
            out.append({
                "id": int(r["id"]),
                "created_at": r["created_at"],
                "role": r["kind"],
                "content": r["text"],
                "kind": r["kind"],
                "text": r["text"],
                "rating": None if r["rating"] is None else int(r["rating"]),
                "caught": int(r["caught"] or 0),
            })
        return out
    finally:
        con.close()

def db_get_memory_short(db: DB, limit: int = 12) -> List[Dict[str, Any]]:
    """Return last N short-memory items (chronological order)."""
    con = db.connect()
    try:
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(memory_short)").fetchall()}
        except Exception:
            cols = set()

        if {"salience", "topic", "ui_message_id"}.issubset(cols):
            rows = con.execute(
                "SELECT role,content,created_at,ui_message_id,topic,salience FROM memory_short ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            out = [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "created_at": r["created_at"],
                    "ui_message_id": int(r["ui_message_id"] or 0),
                    "topic": str(r["topic"] or ""),
                    "salience": float(r["salience"] or 0.0),
                }
                for r in rows
            ]
        else:
            rows = con.execute(
                "SELECT role,content,created_at FROM memory_short ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            out = [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows]
        out.reverse()
        return out
    finally:
        con.close()

def db_get_memory_long(db: DB, limit: int = 4) -> List[Dict[str, Any]]:
    """Return last N long-memory summaries (chronological order)."""
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT summary,created_at FROM memory_long ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        out = [{"summary": r["summary"], "created_at": r["created_at"]} for r in rows]
        out.reverse()
        return out
    finally:
        con.close()


def db_update_memory_short_salience(db: DB, ui_message_id: int, salience: float, *, topic: str = "") -> None:
    """Update salience/topic for a short-memory row linked to a UI message.

    Migration-safe: does nothing if the linkage columns don't exist.
    """
    try:
        ui_message_id = int(ui_message_id or 0)
    except Exception:
        ui_message_id = 0
    if ui_message_id <= 0:
        return
    s = float(salience or 0.0)
    s = max(0.0, min(1.0, s))
    topic = (topic or "").strip()[:80]

    con = db.connect()
    try:
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(memory_short)").fetchall()}
        except Exception:
            cols = set()
        if not {"ui_message_id", "salience"}.issubset(cols):
            return
        # Update salience (monotonic max) and topic if provided.
        if "topic" in cols and topic:
            con.execute(
                "UPDATE memory_short SET salience=MAX(COALESCE(salience,0.0),?), topic=CASE WHEN topic='' THEN ? ELSE topic END WHERE ui_message_id=?",
                (s, topic, ui_message_id),
            )
        else:
            con.execute(
                "UPDATE memory_short SET salience=MAX(COALESCE(salience,0.0),?) WHERE ui_message_id=?",
                (s, ui_message_id),
            )
        con.commit()
    finally:
        con.close()




def db_update_memory_short_episode(db: DB, ui_message_id: int, episode_id: str, episode_dist: int, *, salience_floor: float = 0.0) -> None:
    """Attach episode metadata to STM and optionally raise salience.

    Migration-safe: no-op if episode columns do not exist.
    """
    try:
        ui_message_id = int(ui_message_id or 0)
    except Exception:
        ui_message_id = 0
    if ui_message_id <= 0:
        return
    eid = (episode_id or "").strip()[:64]
    try:
        dist = int(episode_dist or 0)
    except Exception:
        dist = 0
    sf = float(salience_floor or 0.0)
    sf = max(0.0, min(1.0, sf))

    con = db.connect()
    try:
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(memory_short)").fetchall()}
        except Exception:
            cols = set()
        if not {"ui_message_id", "salience"}.issubset(cols):
            return
        if not {"episode_id", "episode_dist"}.issubset(cols):
            # If DB is old, just raise salience if possible.
            con.execute(
                "UPDATE memory_short SET salience=MAX(COALESCE(salience,0.0),?) WHERE ui_message_id=?",
                (sf, ui_message_id),
            )
            con.commit()
            return
        con.execute(
            "UPDATE memory_short SET episode_id=?, episode_dist=?, salience=MAX(COALESCE(salience,0.0),?) WHERE ui_message_id=?",
            (eid, dist, sf, ui_message_id),
        )
        con.commit()
    finally:
        con.close()



def db_apply_episode_boost(db: DB, center_ui_message_id: int, episode_id: str, strength: float, window: int, tau: float) -> int:
    """Boost salience for STM rows around a high-salience episode.

    Models human 'tag-and-capture': salient events strengthen nearby context.

    Only updates existing rows; future messages are boosted by Kernel when they arrive.
    Returns number of rows updated.
    """
    try:
        center = int(center_ui_message_id or 0)
    except Exception:
        center = 0
    if center <= 0:
        return 0
    eid = (episode_id or "").strip()[:64]
    try:
        w = max(0, int(window or 0))
    except Exception:
        w = 0
    try:
        t = max(0.25, float(tau or 2.5))
    except Exception:
        t = 2.5
    s0 = max(0.0, min(1.0, float(strength or 0.0)))
    if w <= 0 or s0 <= 0.0:
        return 0

    lo = max(0, center - w)
    hi = center + w

    import math
    con = db.connect()
    try:
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(memory_short)").fetchall()}
        except Exception:
            cols = set()
        if not {"ui_message_id", "salience"}.issubset(cols):
            return 0
        rows = con.execute(
            "SELECT id, ui_message_id FROM memory_short WHERE ui_message_id BETWEEN ? AND ?",
            (lo, hi),
        ).fetchall()
        n = 0
        for r in rows:
            rid = int(r["id"])
            mid = int(r["ui_message_id"] or 0)
            dist = abs(mid - center)
            boost = s0 * math.exp(-float(dist) / t)
            boost = max(0.0, min(1.0, boost))
            if {"episode_id", "episode_dist"}.issubset(cols):
                con.execute(
                    "UPDATE memory_short SET salience=MAX(COALESCE(salience,0.0),?), episode_id=?, episode_dist=? WHERE id=?",
                    (boost, eid, int(dist), rid),
                )
            else:
                con.execute(
                    "UPDATE memory_short SET salience=MAX(COALESCE(salience,0.0),?) WHERE id=?",
                    (boost, rid),
                )
            n += 1
        con.commit()
        return n
    finally:
        con.close()
def db_add_sensory_token(db: DB, modality: str, summary: str, tokens: Dict[str, Any], salience: float, topic: str = "") -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO sensory_tokens(modality,summary,tokens_json,salience,topic,created_at) VALUES(?,?,?,?,?,?)",
            (modality, summary[:400], json.dumps(tokens or {}, ensure_ascii=False), float(salience or 0.0), topic or "", now_iso()),
        )
        con.commit()
    finally:
        con.close()

def db_get_sensory_tokens(db: DB, limit: int = 6, topic: str = "") -> List[Dict[str, Any]]:
    con = db.connect()
    try:
        if topic:
            rows = con.execute(
                "SELECT modality,summary,tokens_json,salience,topic,created_at FROM sensory_tokens WHERE topic=? ORDER BY id DESC LIMIT ?",
                (topic, int(limit)),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT modality,summary,tokens_json,salience,topic,created_at FROM sensory_tokens ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "modality": str(r["modality"]),
                "summary": str(r["summary"]),
                "tokens": json.loads(r["tokens_json"] or "{}"),
                "salience": float(r["salience"] or 0.0),
                "topic": str(r["topic"] or ""),
                "created_at": str(r["created_at"]),
            })
        return out
    finally:
        con.close()

def db_caught_message(db: DB, message_id: int) -> None:
    con = db.connect()
    try:
        con.execute("UPDATE ui_messages SET caught=1 WHERE id=?", (int(message_id),))
        con.commit()
    finally:
        con.close()

def db_status(db: DB) -> Dict[str, Any]:
    con = db.connect()
    try:
        ax = con.execute("SELECT axis_index,axis_name FROM state_axes ORDER BY axis_index").fetchall()
        row = con.execute("SELECT vec_json,updated_at FROM state_current WHERE id=1").fetchone()
        vec = json.loads(row["vec_json"]) if row else []
        named = {a["axis_name"]: float(vec[int(a["axis_index"])]) for a in ax if int(a["axis_index"]) < len(vec)}
        # axis metadata
        try:
            meta_rows = con.execute("SELECT axis_name,invariant,decays,source FROM state_axes_meta").fetchall()
            axis_meta = {r['axis_name']: {'invariant': int(r['invariant']), 'decays': int(r['decays']), 'source': r['source']} for r in meta_rows}
        except Exception:
            axis_meta = {}
        # Debugging/observability: helps confirm persistence across restarts.
        try:
            ai_cnt = int(con.execute("SELECT COUNT(*) AS c FROM axiom_interpretations").fetchone()["c"])
        except Exception:
            ai_cnt = 0
        try:
            mut_cnt = int(con.execute("SELECT COUNT(*) AS c FROM mutation_proposals").fetchone()["c"])
        except Exception:
            mut_cnt = 0
        return {
            "updated_at": (row["updated_at"] if row else None),
            "axes": named,
            "axioms": db_get_axioms(db),
            "db_path": str(db.path),
            "axiom_interpretations_count": ai_cnt,
            "mutation_proposals_count": mut_cnt,
            "mutation_proposals": (db_list_mutation_proposals(db, limit=4) if mut_cnt > 0 else []),
        }
    finally:
        con.close()

def db_add_websense_page(db: DB, query: str, fr: Dict[str, Any], ok: int = 1) -> None:
    con = db.connect()
    try:
        con.execute(
            """INSERT INTO websense_pages(created_at,query,url,title,snippet,body,domain,hash,ok)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                query or "",
                str(fr.get("url") or ""),
                str(fr.get("title") or ""),
                str(fr.get("snippet") or ""),
                str(fr.get("body") or ""),
                str(fr.get("domain") or ""),
                str(fr.get("hash") or ""),
                int(ok),
            ),
        )
        con.commit()
    finally:
        con.close()


# -----------------------------
# SSE broker
# -----------------------------