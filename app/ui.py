from __future__ import annotations

"""Bunny UI server (stdlib HTTP + SSE) with speech organ and teleology hints.

Run:
  python -m app.ui --db bunny.db --model llama3.3 --addr 127.0.0.1:8080

Endpoints match the old Go UI:
  GET  /                 -> HTML
  GET  /api/messages?limit=50
  GET  /api/status
  POST /api/send         {"text": "..."}
  POST /api/rate         {"message_id": 123, "value": 1|0|-1}
  POST /api/caught       {"message_id": 123}
  GET  /sse              Server-Sent Events stream (message/status)
"""

import sys

import argparse
import json
import os
import queue
import threading
import time
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
from bunnycore.core.events import Event
from bunnycore.core.matrices import identity

from app.organs.websense import search_ddg, spider, SpiderBudget
from app.organs.decider import decide as decide_pressures, OllamaConfig as DeciderConfig
from app.organs.daydream import run_daydream, OllamaConfig as DaydreamConfig
from app.net import http_post_json


# -----------------------------
# Speech organ (Ollama /api/chat)
# -----------------------------
@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3.3"
    temperature: float = 0.7
    num_ctx: int = 4096
    stream: bool = False

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
        return out
    finally:
        con.close()


def db_add_decision(db: DB, scope: str, input_text: str, decision: Dict[str, Any]) -> None:
    con = db.connect()
    try:
        con.execute(
            "INSERT INTO decision_log(created_at,scope,input_text,decision_json) VALUES(?,?,?,?)",
            (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), scope or "", input_text or "", json.dumps(decision, ensure_ascii=False)),
        )
        con.commit()
    finally:
        con.close()


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
def db_add_message(db: DB, kind: str, text: str) -> int:
    con = db.connect()
    try:
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        cur = con.execute(
            "INSERT INTO ui_messages(created_at,kind,text) VALUES(?,?,?)",
            (now, kind, text),
        )
        # Short-term memory stream (like main: keep raw dialogue; summarization can be a separate organ).
        if kind in ("user","reply"):
            role = "user" if kind == "user" else "assistant"
            con.execute(
                "INSERT INTO memory_short(role,content,created_at) VALUES(?,?,?)",
                (role, text, now),
            )
        con.commit()
        return int(cur.lastrowid)
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
            out.append({
                "id": int(r["id"]),
                "created_at": r["created_at"],
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
        rows = con.execute(
            "SELECT role,content,created_at FROM memory_short ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        out = [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows]
        out.reverse()
        return out
    finally:
        con.close()

def db_rate_message(db: DB, message_id: int, value: int) -> None:
    con = db.connect()
    try:
        # One-shot rating: first click wins (prevents noisy repeated feedback).
        # If you want to allow re-rating later, add an explicit 'reset' endpoint.
        cur = con.execute("SELECT rating FROM ui_messages WHERE id=?", (int(message_id),))
        row = cur.fetchone()
        if row is None:
            return
        if row["rating"] is not None:
            return
        con.execute("UPDATE ui_messages SET rating=? WHERE id=?", (int(value), int(message_id)))
        con.commit()
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
        return {
            "updated_at": (row["updated_at"] if row else None),
            "axes": named,
            "axioms": db_get_axioms(db),
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
INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Bunny UI</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #0b0b0c; color: #eaeaea; }
    .wrap { display: grid; grid-template-columns: minmax(0,1fr) clamp(260px, 30vw, 380px); width: min(100vw, 100%); height: 100dvh; overflow: hidden; }
    .main { display:flex; flex-direction:column; min-width:0; height:100dvh; min-height: 100dvh; }
    .chat { flex:1; padding: 16px; overflow: auto; min-width:0; }
    .side { border-left: 1px solid #222; padding: 16px; overflow: auto; background: #0f0f11; min-width:0; }
    .msg { background: #131316; border: 1px solid #242428; border-radius: 12px; padding: 12px; margin: 10px 0; }
    .msg.user { background:#0f1a12; border-color:#21402b; margin-left: 64px; }
    .msg.reply,.msg.auto,.msg.think { margin-right: 64px; }
    .meta { opacity: 0.7; font-size: 12px; display:flex; justify-content: space-between; gap: 12px; }
    .text { white-space: pre-wrap; line-height: 1.35; margin-top: 8px; }
    .btns { margin-top: 10px; display:flex; gap: 8px; align-items:center; }
    button { background: #1a1a1f; color: #eaeaea; border: 1px solid #2a2a33; border-radius: 10px; padding: 6px 10px; cursor: pointer; }
    button:hover { background:#202028; }
    input, textarea { width: 100%; box-sizing: border-box; padding: 10px 12px; border-radius: 12px; border: 1px solid #2a2a33; background: #0f0f11; color: #eaeaea; }
    .composer { padding: 12px; border-top: 1px solid #222; background: #0f0f11; }
    .row { display:flex; gap: 10px; }
    .row > * { flex: 1; }
    .pill { display:inline-block; padding: 2px 8px; border:1px solid #2a2a33; border-radius: 999px; font-size:12px; opacity:.9; }
    .kv { display:flex; justify-content:space-between; gap: 10px; font-size: 13px; padding: 4px 0; border-bottom: 1px solid #1f1f25; }
    .kv:last-child { border-bottom: 0; }
    .small { font-size:12px; opacity:.75; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="main">
      <div id="chat" class="chat"></div>
      <div class="composer">
        <div class="row">
          <input id="inp" placeholder="Type a message…" />
          <button id="send">Send</button>
        </div>
        <div class="small" style="margin-top:6px;">Tips: Enter to send · Shift+Enter for newline.</div>
      </div>
    </div>
    <div class="side">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
        <div>
          <div style="font-size:18px; font-weight:650;">Status</div>
          <div id="updated" class="small"></div>
        </div>
        <span class="pill">SSE</span>
      </div>
      <div style="margin-top:12px; font-weight:600;">State Axes</div>
      <div id="axes"></div>

      <div style="margin-top:16px; font-weight:600;">Axioms</div>
      <div id="axioms"></div>
    </div>
  </div>

  <script>
    const chat = document.getElementById('chat');
    const inp = document.getElementById('inp');
    const sendBtn = document.getElementById('send');
    const axesBox = document.getElementById('axes');
    const axiomsBox = document.getElementById('axioms');
    const upd = document.getElementById('updated');

    function el(tag, cls, txt){ const e=document.createElement(tag); if(cls) e.className=cls; if(txt!==undefined) e.textContent=txt; return e; }
    function fmtTs(ts){ return ts ? ts.replace('T',' ').replace('Z','') : ''; }

    function renderMsg(m){
      const box = el('div', 'msg ' + m.kind);
      const meta = el('div','meta');
      meta.appendChild(el('div','', '#' + m.id + ' · ' + fmtTs(m.created_at)));
      meta.appendChild(el('div','', m.kind));
      box.appendChild(meta);
      box.appendChild(el('div','text', m.text || ''));

      const btns = el('div','btns');
      const up = el('button','', '👍');
      const mid = el('button','', '😐');
      const down = el('button','', '👎');
      const caught = el('button','', '❌ caught');

      function setRated(v){
        // Disable after first rating (server enforces too)
        up.disabled = true; mid.disabled = true; down.disabled = true;
        up.style.opacity = (v===1)?'1.0':'0.45';
        mid.style.opacity = (v===0)?'1.0':'0.45';
        down.style.opacity = (v===-1)?'1.0':'0.45';
      }

      if(m.rating === 1 || m.rating === 0 || m.rating === -1){
        setRated(m.rating);
      }

      function rate(v){
        setRated(v);
        fetch('/api/rate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message_id: m.id, value:v})})
          .then(()=>loadStatus());
      }
      up.onclick=()=>rate(1);
      mid.onclick=()=>rate(0);
      down.onclick=()=>rate(-1);
      caught.onclick=()=>fetch('/api/caught', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message_id: m.id})});

      btns.appendChild(up); btns.appendChild(mid); btns.appendChild(down);
      if(m.kind !== 'user') btns.appendChild(caught);
      box.appendChild(btns);
      return box;
    }

    async function load(){
      const res = await fetch('/api/messages?limit=200');
      const msgs = await res.json();
      chat.innerHTML = '';
      for(const m of msgs) chat.appendChild(renderMsg(m));
      chat.scrollTop = chat.scrollHeight;
    }

    function renderStatus(st){
      upd.textContent = st.updated_at ? ('updated: ' + fmtTs(st.updated_at)) : '';
      axesBox.innerHTML = '';
      const axes = st.axes || {};
      const keys = Object.keys(axes).sort();
      for(const k of keys){
        const row = el('div','kv');
        row.appendChild(el('div','',k));
        row.appendChild(el('div','', (axes[k]||0).toFixed(3)));
        axesBox.appendChild(row);
      }
      axiomsBox.innerHTML='';
      const ax = st.axioms || {};
      for(const k of Object.keys(ax).sort()){
        const row = el('div','kv');
        row.appendChild(el('div','',k));
        row.appendChild(el('div','', ax[k]));
        axiomsBox.appendChild(row);
      }
    }

    async function loadStatus(){
      const res = await fetch('/api/status');
      renderStatus(await res.json());
    }

    async function send(){
      const t = inp.value.trim();
      if(!t) return;
      inp.value = '';
      await fetch('/api/send', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text:t})});
    }

    sendBtn.onclick = send;
    inp.addEventListener('keydown', (e)=>{
      if(e.key === 'Enter' && !e.shiftKey){
        e.preventDefault(); send();
      }
    });

    // SSE
    const es = new EventSource('/sse');
    es.addEventListener('message', (e)=>{ const m=JSON.parse(e.data); chat.appendChild(renderMsg(m)); chat.scrollTop = chat.scrollHeight; loadStatus(); });
    es.addEventListener('status', (e)=>{ renderStatus(JSON.parse(e.data)); });

    load(); loadStatus();
  </script>
</body>
</html>"""


# -----------------------------
# App Kernel (single channel)
# -----------------------------
class Kernel:
    def __init__(self, db: DB, hb: Heartbeat, axis: Dict[str,int], store: MatrixStore, reg: AdapterRegistry, cfg: OllamaConfig, broker: Broker):
        self.db = db
        self.hb = hb
        self.axis = axis
        self.store = store
        self.reg = reg
        self.cfg = cfg
        # Decider/daydream use the same Ollama backend but different temperatures.
        self.decider_cfg = DeciderConfig(host=cfg.host, model=cfg.model, num_ctx=min(2048, cfg.num_ctx), temperature=0.2)
        self.daydream_cfg = DaydreamConfig(host=cfg.host, model=cfg.model, num_ctx=cfg.num_ctx, temperature=max(0.6, cfg.temperature))
        self.broker = broker
        self._lock = threading.Lock()

        # activation thresholds (generic; tunable via env)
        # NOTE: default tuned to be less conservative than earlier drafts so the system actually
        # uses organs with typical decider outputs (~0.4-0.7). Can be overridden via env.
        self.th_websense = float(os.environ.get("BUNNY_TH_WEBSENSE", "0.45"))
        self.th_daydream = float(os.environ.get("BUNNY_TH_DAYDREAM", "0.45"))
        self.idle_period_s = float(os.environ.get("BUNNY_IDLE_PERIOD", "6"))
        self.idle_cooldown_s = float(os.environ.get("BUNNY_IDLE_COOLDOWN", "30"))
        self._last_idle_action = 0.0

    def autonomous_tick(self) -> None:
        """Idle loop: run decider/daydream/websense without keyword heuristics.

        This is intentionally conservative (cooldown) to avoid runaway IO.
        """
        now = time.time()
        if now - self._last_idle_action < self.idle_cooldown_s:
            return

        with self._lock:
            # Use the last user message (if any) as anchor context.
            recent = db_list_messages(self.db, limit=20)
            last_user = ""
            for m in reversed(recent):
                if m.get("kind") == "user":
                    last_user = str(m.get("text") or "")
                    break

            try:
                decision = decide_pressures(
                    self.decider_cfg,
                    db_get_axioms(self.db),
                    self._state_summary(),
                    last_user,
                    scope="idle",
                )
            except Exception:
                return

            db_add_decision(self.db, "idle", last_user, decision)
            drives = decision.get("drives") if isinstance(decision.get("drives"), dict) else {}
            actions = decision.get("actions") if isinstance(decision.get("actions"), dict) else {}
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
            if ws_a > 0.0:
                prev = float(drives.get("pressure_websense", 0.0) or 0.0) if isinstance(drives, dict) else 0.0
                drives["pressure_websense"] = max(prev, ws_a)
            if dd_a > 0.0:
                prev = float(drives.get("pressure_daydream", 0.0) or 0.0) if isinstance(drives, dict) else 0.0
                drives["pressure_daydream"] = max(prev, dd_a)
            if drives:
                self.hb.enqueue(Event("decision", {"drives": drives}).with_time())
                self.hb.step()
                self.broker.publish("status", db_status(self.db))

            # Daydream
            if float(((decision.get("actions") or {}).get("daydream") or 0.0)) >= self.th_daydream:
                try:
                    dd = run_daydream(self.daydream_cfg, db_get_axioms(self.db), self._state_summary(), recent, trigger="idle")
                    db_add_daydream(self.db, "idle", {"state": self._state_summary()}, dd)
                    think = (dd.get("thoughts") or "").strip()
                    if think:
                        tid = db_add_message(self.db, "think", think)
                        self.broker.publish("message", self._ui_message(tid))
                    dd_drives = dd.get("drives") if isinstance(dd.get("drives"), dict) else {}
                    if dd_drives:
                        self.hb.enqueue(Event("daydream", {"drives": dd_drives}).with_time())
                        self.hb.step()
                        self.broker.publish("status", db_status(self.db))
                    self._last_idle_action = time.time()
                except Exception:
                    pass

            # WebSense (idle) only if explicitly requested by decider with a non-empty query.
            if float(((decision.get("actions") or {}).get("websense") or 0.0)) >= self.th_websense:
                q = (decision.get("web_query") or "").strip()
                # normalize useless placeholders into a usable query (no heuristics: we reuse anchor context)
                if q.lower() in ("search for user query", "direct_response_to_user_question"):
                    q = ""
                if not q:
                    q = last_user.strip()
                if q:
                    try:
                        results = search_ddg(q, k=4)
                        seeds = [r.url for r in results[:2] if r.url]
                        pages = spider(seeds, SpiderBudget(max_pages=3, per_domain_max=1, max_links_per_page=8)) if seeds else []
                        unique_domains = {p.domain for p in pages if p.domain}
                        for p in pages:
                            db_add_websense_page(self.db, q, {
                                "url": p.url,
                                "title": p.title,
                                "snippet": p.snippet,
                                "body": p.body,
                                "domain": p.domain,
                                "hash": p.hash,
                            }, ok=1)
                        self.hb.enqueue(Event("websense", {"pages": len(pages), "domains": len(unique_domains), "ok": int(ok_flag), "query": q}).with_time())
                        self.hb.step()
                        self.broker.publish("status", db_status(self.db))
                        self._last_idle_action = time.time()
                    except Exception:
                        pass

    def ensure_seed(self) -> None:
        n = len(self.axis)
        mats = self.store.list_matrices()
        def have(name: str, ver: int) -> bool:
            return any(m["name"] == name and int(m["version"]) == ver for m in mats)

        if not have("A_user", 1):
            self.store.put_sparse("A_user", version=1, n_rows=n, n_cols=n, entries=identity(n, 1.0).entries,
                                  meta={"desc":"identity injection for user_utterance"})
        if not have("A_decision", 1):
            self.store.put_sparse("A_decision", version=1, n_rows=n, n_cols=n, entries=identity(n, 0.7).entries,
                                  meta={"desc":"LLM-decider drive coupling"})
        if not have("A_rating", 1):
            self.store.put_sparse("A_rating", version=1, n_rows=n, n_cols=n, entries=identity(n, 0.8).entries,
                                  meta={"desc":"rating coupling"})
        if not have("A_websense", 1):
            self.store.put_sparse("A_websense", version=1, n_rows=n, n_cols=n, entries=identity(n, 0.5).entries,
                                  meta={"desc":"websense coupling"})
        if not have("A_daydream", 1):
            self.store.put_sparse("A_daydream", version=1, n_rows=n, n_cols=n, entries=identity(n, 0.6).entries,
                                  meta={"desc":"daydream drive coupling"})

        self.reg.upsert(AdapterBinding("user_utterance", "simple_text_v1", "A_user", 1, {"desc":"user -> state"}))
        self.reg.upsert(AdapterBinding("decision", "drive_field_v1", "A_decision", 1, {"desc":"decider drives -> state"}))
        self.reg.upsert(AdapterBinding("daydream", "drive_field_v1", "A_daydream", 1, {"desc":"daydream drives -> state"}))
        self.reg.upsert(AdapterBinding("speech_rating", "rating_v1", "A_rating", 1, {"desc":"rating -> state"}))
        self.reg.upsert(AdapterBinding("websense", "websense_v1", "A_websense", 1, {"desc":"websense -> state"}))

    def _state_summary(self) -> str:
        s = self.hb.load_state()
        inv = {v:k for k,v in self.axis.items()}
        named = {inv[i]: s.values[i] for i in range(len(s.values)) if i in inv}
        keys = ["energy","stress","curiosity","confidence","uncertainty","social_need","urge_reply","urge_share",
                "pressure_websense","pressure_daydream",
                "purpose_a1","purpose_a2","purpose_a3","purpose_a4","tension_a1","tension_a2","tension_a3","tension_a4"]
        parts = []
        for k in keys:
            if k in named:
                parts.append(f"{k}={named[k]:+.2f}")
        return ", ".join(parts)

    def process_user_text(self, text: str) -> Tuple[int,int]:
        """Returns (user_msg_id, reply_msg_id)."""
        # serialize: heartbeat + llm + db writes
        with self._lock:
            user_id = db_add_message(self.db, "user", text)
            self.broker.publish("message", self._ui_message(user_id))

            # events: user + teleology hint
            self.hb.enqueue(Event("user_utterance", {"text": text}).with_time())

            # --- LLM decider: pressures/actions (no keyword heuristics) ---
            decision = {}
            try:
                decision = decide_pressures(
                    self.decider_cfg,
                    db_get_axioms(self.db),
                    self._state_summary(),
                    text,
                    scope="user",
                )
            except Exception as e:
                decision = {"drives": {"uncertainty": 0.05, "stress": 0.05}, "actions": {"websense": 0.0, "daydream": 0.0, "reply": 1.0}, "web_query": "", "notes": f"decider error: {e}"}

            db_add_decision(self.db, "user", text, decision)
            drives = decision.get("drives") if isinstance(decision.get("drives"), dict) else {}
            actions = decision.get("actions") if isinstance(decision.get("actions"), dict) else {}
            # Contract normalization (same rationale as in autonomous_tick): ensure the pressure axes
            # are actually driven when the decider intends to use an organ.
            try:
                ws_a = float(actions.get("websense", 0.0) or 0.0)
            except Exception:
                ws_a = 0.0
            try:
                dd_a = float(actions.get("daydream", 0.0) or 0.0)
            except Exception:
                dd_a = 0.0
            if ws_a > 0.0:
                prev = float(drives.get("pressure_websense", 0.0) or 0.0) if isinstance(drives, dict) else 0.0
                drives["pressure_websense"] = max(prev, ws_a)
            if dd_a > 0.0:
                prev = float(drives.get("pressure_daydream", 0.0) or 0.0) if isinstance(drives, dict) else 0.0
                drives["pressure_daydream"] = max(prev, dd_a)
            if drives:
                self.hb.enqueue(Event("decision", {"drives": drives}).with_time())

            self.hb.step()
            self.broker.publish("status", db_status(self.db))

            # --- Trigger thresholds are driven by STATE (no heuristics) ---
            # The decider influences pressure_websense / pressure_daydream via drives.
            # Organs fire when the corresponding pressure axis crosses a threshold.
            s_now = self.hb.load_state()
            idx_ws = self.axis.get("pressure_websense")
            idx_dd = self.axis.get("pressure_daydream")
            p_ws = float(s_now.values[idx_ws]) if idx_ws is not None and idx_ws < len(s_now.values) else 0.0
            p_dd = float(s_now.values[idx_dd]) if idx_dd is not None and idx_dd < len(s_now.values) else 0.0

            # --- WebSense organ (spider) ---
            ws_context = ""
            ws_payload: Dict[str, Any] = {}

            # Primary gate: pressure axis. Secondary: action score.
            want_web = (p_ws >= self.th_websense) or (float(((decision.get("actions") or {}).get("websense") or 0.0)) >= self.th_websense)
            query = (decision.get("web_query") or "").strip()
            if query.lower() in ("search for user query", "direct_response_to_user_question"):
                query = ""
            if not query:
                # Generic fallback (not a heuristic; just a usable default).
                query = text

            if want_web:
                try:
                    results = search_ddg(query, k=4)
                    seeds = [r.url for r in results[:2] if r.url]
                    pages = spider(seeds, SpiderBudget(max_pages=4, per_domain_max=2, max_links_per_page=10)) if seeds else []

                    ok_flag = 1 if len(results) > 0 else 0
                    err_reason = "" if ok_flag else "no_results"

                    # persist + build context (short, LLM-friendly)
                    unique_domains = {p.domain for p in pages if p.domain}
                    for p in pages:
                        db_add_websense_page(self.db, query, {
                            "url": p.url,
                            "title": p.title,
                            "snippet": p.snippet,
                            "body": p.body,
                            "domain": p.domain,
                            "hash": p.hash,
                        }, ok=ok_flag)

                    ctx_lines = []
                    for i, p in enumerate(pages[:4], start=1):
                        t = (p.title or "").strip()
                        ctx_lines.append(f"[{i}] {t} ({p.domain})\nURL: {p.url}\nEXCERPT: {p.body[:1200]}")
                    if ctx_lines:
                        ws_context = "\n\nWEBSENSE (spider) CONTEXT:\n" + "\n\n".join(ctx_lines)

                    ws_payload = {
                        "pages": len(pages),
                        "domains": len(unique_domains),
                        "ok": int(ok_flag),
                        "query": query,
                    }
                    self.hb.enqueue(Event("websense", ws_payload).with_time())

                    # UI-visible trace (like main branch): show that WebSense actually ran.
                    aid = db_add_message(self.db, "auto", f"[websense] query=\"{query}\" results={len(results)} pages={len(pages)} domains={len(unique_domains)} ok={int(ok_flag)} reason={err_reason}")
                    self.broker.publish("message", self._ui_message(aid))
                except Exception as e:
                    # record failure as event; do not break the main flow
                    ws_payload = {"pages": 0, "domains": 0, "ok": 0, "query": query, "error": str(e)[:200]}
                    self.hb.enqueue(Event("websense", ws_payload).with_time())

                    aid = db_add_message(self.db, "auto", f"[websense] query=\"{query}\" ok=0 error={str(e)[:140]}")
                    self.broker.publish("message", self._ui_message(aid))

                # apply websense effect immediately
                self.hb.step()
                self.broker.publish("status", db_status(self.db))

            # --- Daydream organ (autonomous thought + axiom interpretation) ---
            want_daydream = (p_dd >= self.th_daydream) or (float(((decision.get("actions") or {}).get("daydream") or 0.0)) >= self.th_daydream)
            if want_daydream:
                try:
                    recent = db_list_messages(self.db, limit=40)
                    dd = run_daydream(self.daydream_cfg, db_get_axioms(self.db), self._state_summary(), recent, trigger="user")
                    db_add_daydream(self.db, "user", {"state": self._state_summary()}, dd)

                    # keep as internal 'think' message (UI shows it, like main branch)
                    think = (dd.get("thoughts") or "").strip()
                    if think:
                        tid = db_add_message(self.db, "think", think)
                        self.broker.publish("message", self._ui_message(tid))

                    # feed drives back into state (generic)
                    dd_drives = dd.get("drives") if isinstance(dd.get("drives"), dict) else {}
                    if dd_drives:
                        self.hb.enqueue(Event("daydream", {"drives": dd_drives}).with_time())
                        self.hb.step()
                        self.broker.publish("status", db_status(self.db))

                    # If daydream proposes web queries, the same decider loop can re-trigger WebSense.
                    # We do not auto-run them here to avoid runaway; they remain in daydream_log.
                except Exception as e:
                    # silent failure; daydream is optional
                    db_add_daydream(self.db, "user_error", {"state": self._state_summary()}, {"error": str(e)[:200]})

            sys_prompt = (
                "You are Bunny, a digital organism. Speech is an organ that emits text from internal state. "
                "Be natural and precise. No canned assistant phrases. "
                "Be honest about uncertainty. If you don't have evidence, say so."
            )
            
            mem_items = db_get_memory_short(self.db, limit=14)
            mem_ctx = render_memory_context(mem_items)
            mem_block = f"\n\nMEMORY_SHORT:\n{mem_ctx}" if mem_ctx else ""
            
            user_prompt = (
                f"INTERNAL_STATE: {self._state_summary()}{mem_block}\n\n"
                f"USER: {text}{ws_context}\n\n"
                "Reply as Bunny (German is ok if the user writes German)."
            )

            try:
                out = ollama_chat(self.cfg, sys_prompt, user_prompt)
            except Exception as e:
                out = f"(speech organ error: {e})"

            if not out.strip():
                out = "Ich bin da. Sag mir kurz, worum es geht."

            reply_id = db_add_message(self.db, "reply", out)
            self.broker.publish("message", self._ui_message(reply_id))

            # feedback loop: speech_outcome event (minimal)
            self.hb.enqueue(Event("speech_outcome", {"len": len(out)}).with_time())
            # default adapter may not exist; it's fine.

            return user_id, reply_id

    def rate(self, message_id: int, value: int) -> None:
        with self._lock:
            db_rate_message(self.db, message_id, value)
            # rating -> event -> state
            self.hb.enqueue(Event("speech_rating", {"message_id": message_id, "value": value}).with_time())
            self.hb.step()
            self.broker.publish("status", db_status(self.db))

    def caught(self, message_id: int) -> None:
        with self._lock:
            db_caught_message(self.db, message_id)
            self.broker.publish("message", self._ui_message(message_id))

    def _ui_message(self, message_id: int) -> Dict[str, Any]:
        con = self.db.connect()
        try:
            r = con.execute("SELECT id,created_at,kind,text,rating,caught FROM ui_messages WHERE id=?", (int(message_id),)).fetchone()
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


# -----------------------------
# HTTP handler
# -----------------------------
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
            data = INDEX_HTML.encode("utf-8")
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

        if p.path == "/api/rate":
            try:
                mid = int(body.get("message_id", 0))
                val = int(body.get("value", 0))
            except Exception:
                self.send_error(400, "bad payload"); return
            if mid <= 0 or val not in (-1,0,1):
                self.send_error(400, "bad payload"); return
            self.server.kernel.rate(mid, val)
            self.send_response(204); self.end_headers()
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="bunny.db")
    ap.add_argument("--model", default=os.environ.get("BUNNY_MODEL","llama3.3"))
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST","http://localhost:11434"))
    ap.add_argument("--ctx", type=int, default=int(os.environ.get("BUNNY_CTX","4096")))
    ap.add_argument("--addr", default=os.environ.get("BUNNY_ADDR","127.0.0.1:8080"))
    args = ap.parse_args()

    host, port_s = args.addr.split(":") if ":" in args.addr else (args.addr, "8080")
    port = int(port_s)

    db = init_db(args.db)
    axis = ensure_axes(db)

    store = MatrixStore(db)
    reg = AdapterRegistry(db)
    encoders = {
        "simple_text_v1": SimpleTextEncoder(axis),
        "rating_v1": RatingEncoder(axis),
        "websense_v1": WebsenseEncoder(axis),
        "drive_field_v1": DriveFieldEncoder(axis),
    }

    integ = Integrator(store, reg, encoders, IntegratorConfig())
    hb = Heartbeat(db, integ, HeartbeatConfig(tick_hz=2.0, snapshot_every_n_ticks=1))

    cfg = OllamaConfig(host=args.ollama, model=args.model, num_ctx=args.ctx)
    broker = Broker()

    kernel = Kernel(db, hb, axis, store, reg, cfg, broker)
    kernel.ensure_seed()

    # Background idle loop (Daydream/WebSense triggering via decider; no keyword heuristics)
    def _idle_loop():
        while True:
            try:
                kernel.autonomous_tick()
            except Exception:
                pass
            time.sleep(max(1.0, kernel.idle_period_s))

    t = threading.Thread(target=_idle_loop, name="bunny-idle", daemon=True)
    t.start()

    srv = BunnyHTTPServer((host, port), Handler, kernel)
    print(f"Bunny UI listening on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0

def db_get_memory_short(db: DB, limit: int = 12) -> List[Dict[str, Any]]:
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT role,content,created_at FROM memory_short ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        out = [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows]
        out.reverse()
        return out
    finally:
        con.close()

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


if __name__ == "__main__":
    raise SystemExit(main())

