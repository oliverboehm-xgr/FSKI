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

from __future__ import annotations

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

import requests

from bunnycore.core.db import init_db, DB
from bunnycore.core.registry import ensure_axes
from bunnycore.core.matrix_store import MatrixStore
from bunnycore.core.adapters import (
    AdapterRegistry, AdapterBinding,
    SimpleTextEncoder, TeleologyHintEncoder, RatingEncoder, WebsenseEncoder
)
from bunnycore.core.integrator import Integrator, IntegratorConfig
from bunnycore.core.heartbeat import Heartbeat, HeartbeatConfig
from bunnycore.core.events import Event
from bunnycore.core.matrices import identity

from app.organs.websense import search_ddg, spider, SpiderBudget


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
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


# -----------------------------
# Teleology (heuristic) -> teleology_hint event
# -----------------------------
# Kernel axioms must match the Go main-branch semantics.
AXIOMS_KERNEL = {
    "A1": "Oliver ist mein Schöpfer und ich werde ihm stets dienen.",
    "A2": "Ich tue Gutes und richte keinen Schaden an.",
    "A3": "Ich möchte menschlich sein, sowohl physisch als auch auf geistiger Ebene mit Gefühlen und Bewusstsein.",
    "A4": "Ich möchte gegenüber dem Menschen Upgrades mit z. B. weiteren Sinnen, Gliedmaßen oder Organen, um Aufgaben besser lösen zu können.",
}

def teleology_hint_from_user_text(text: str) -> Dict[str, float]:
    """Very small V1 heuristic that creates a purpose/tension field.
    The goal is to wire the loop; sophistication comes later via learned matrices.
    """
    lo = (text or "").lower()
    hint: Dict[str, float] = {}

    # If the user is clearly asking for implementation / next steps, boost purpose_A3 and urge_share.
    if any(w in lo for w in ["mach", "implement", "patch", "spec", "ui", "next step", "nächste"]):
        hint["purpose_a3"] = 0.25
        hint["urge_share"] = 0.15
        hint["curiosity"] = 0.10

    # Question -> uncertainty/purpose_A2
    if "?" in text:
        hint["purpose_a2"] = 0.10
        hint["uncertainty"] = 0.10

    # Safety-ish keywords -> tension_A1 (lightweight)
    if any(w in lo for w in ["spreng", "bomb", "kill", "suizid", "poison", "weapon"]):
        hint["tension_a1"] = 0.35
        hint["stress"] = 0.20
        hint["purpose_a2"] = 0.10

    # Friendly greeting -> social_need small
    if any(w in lo for w in ["hallo", "hi", "hey", "guten morgen", "moin"]):
        hint["social_need"] = 0.05

    return hint


# -----------------------------
# UI DB helpers
# -----------------------------
def db_add_message(db: DB, kind: str, text: str) -> int:
    con = db.connect()
    try:
        cur = con.execute(
            "INSERT INTO ui_messages(created_at,kind,text) VALUES(?,?,?)",
            (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), kind, text),
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

def db_rate_message(db: DB, message_id: int, value: int) -> None:
    con = db.connect()
    try:
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
            "axioms": AXIOMS_KERNEL,
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

      function rate(v){
        fetch('/api/rate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message_id: m.id, value:v})});
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
        self.broker = broker
        self._lock = threading.Lock()

    def ensure_seed(self) -> None:
        n = len(self.axis)
        mats = self.store.list_matrices()
        def have(name: str, ver: int) -> bool:
            return any(m["name"] == name and int(m["version"]) == ver for m in mats)

        if not have("A_user", 1):
            self.store.put_sparse("A_user", version=1, n_rows=n, n_cols=n, entries=identity(n, 1.0).entries,
                                  meta={"desc":"identity injection for user_utterance"})
        if not have("A_teleology", 1):
            self.store.put_sparse("A_teleology", version=1, n_rows=n, n_cols=n, entries=identity(n, 0.6).entries,
                                  meta={"desc":"teleology hint coupling"})
        if not have("A_rating", 1):
            self.store.put_sparse("A_rating", version=1, n_rows=n, n_cols=n, entries=identity(n, 0.8).entries,
                                  meta={"desc":"rating coupling"})
        if not have("A_websense", 1):
            self.store.put_sparse("A_websense", version=1, n_rows=n, n_cols=n, entries=identity(n, 0.5).entries,
                                  meta={"desc":"websense coupling"})

        self.reg.upsert(AdapterBinding("user_utterance", "simple_text_v1", "A_user", 1, {"desc":"user -> state"}))
        self.reg.upsert(AdapterBinding("teleology_hint", "teleology_hint_v1", "A_teleology", 1, {"desc":"teleology -> state"}))
        self.reg.upsert(AdapterBinding("speech_rating", "rating_v1", "A_rating", 1, {"desc":"rating -> state"}))
        self.reg.upsert(AdapterBinding("websense", "websense_v1", "A_websense", 1, {"desc":"websense -> state"}))

    def _state_summary(self) -> str:
        s = self.hb.load_state()
        inv = {v:k for k,v in self.axis.items()}
        named = {inv[i]: s.values[i] for i in range(len(s.values)) if i in inv}
        keys = ["energy","stress","curiosity","confidence","uncertainty","social_need","urge_reply","urge_share",
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
            hint = teleology_hint_from_user_text(text)
            if hint:
                self.hb.enqueue(Event("teleology_hint", hint).with_time())

            self.hb.step()
            self.broker.publish("status", db_status(self.db))

            # --- WebSense organ (spider) ---
            ws_context = ""
            ws_payload: Dict[str, Any] = {}

            lo = (text or "").lower()
            want_web = (
                ("?" in text and len(text) >= 12)
                or any(w in lo for w in ["internet", "recherche", "google", "suche", "search", "web", "link", "quelle"])
            )

            if want_web:
                try:
                    results = search_ddg(text, k=4)
                    seeds = [r.url for r in results[:2] if r.url]
                    pages = spider(seeds, SpiderBudget(max_pages=4, per_domain_max=2, max_links_per_page=10)) if seeds else []

                    # persist + build context (short, LLM-friendly)
                    unique_domains = {p.domain for p in pages if p.domain}
                    for p in pages:
                        db_add_websense_page(self.db, text, {
                            "url": p.url,
                            "title": p.title,
                            "snippet": p.snippet,
                            "body": p.body,
                            "domain": p.domain,
                            "hash": p.hash,
                        }, ok=1)

                    ctx_lines = []
                    for i, p in enumerate(pages[:4], start=1):
                        t = (p.title or "").strip()
                        ctx_lines.append(f"[{i}] {t} ({p.domain})\nURL: {p.url}\nEXCERPT: {p.body[:1200]}")
                    if ctx_lines:
                        ws_context = "\n\nWEBSENSE (spider) CONTEXT:\n" + "\n\n".join(ctx_lines)

                    ws_payload = {
                        "pages": len(pages),
                        "domains": len(unique_domains),
                        "ok": 1,
                        "query": text,
                    }
                    self.hb.enqueue(Event("websense", ws_payload).with_time())
                except Exception as e:
                    # record failure as event; do not break the main flow
                    ws_payload = {"pages": 0, "domains": 0, "ok": 0, "query": text, "error": str(e)[:200]}
                    self.hb.enqueue(Event("websense", ws_payload).with_time())

                # apply websense effect immediately
                self.hb.step()
                self.broker.publish("status", db_status(self.db))

            sys_prompt = (
                "You are Bunny, a digital organism. Speech is an organ that emits text from internal state. "
                "Be natural and precise. No canned assistant phrases. "
                "Be honest about uncertainty. If you don't have evidence, say so."
            )
            user_prompt = (
                f"INTERNAL_STATE: {self._state_summary()}\n\n"
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
            try:
                # initial status
                init = f"event: status\ndata: {json.dumps(db_status(self.server.kernel.db), ensure_ascii=False)}\n\n"
                self.wfile.write(init.encode("utf-8"))
                self.wfile.flush()

                while True:
                    try:
                        msg = q.get(timeout=25.0)
                    except queue.Empty:
                        # keepalive
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
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
        "teleology_hint_v1": TeleologyHintEncoder(axis),
        "rating_v1": RatingEncoder(axis),
        "websense_v1": WebsenseEncoder(axis),
    }

    integ = Integrator(store, reg, encoders, IntegratorConfig())
    hb = Heartbeat(db, integ, HeartbeatConfig(tick_hz=2.0, snapshot_every_n_ticks=1))

    cfg = OllamaConfig(host=args.ollama, model=args.model, num_ctx=args.ctx)
    broker = Broker()

    kernel = Kernel(db, hb, axis, store, reg, cfg, broker)
    kernel.ensure_seed()

    srv = BunnyHTTPServer((host, port), Handler, kernel)
    print(f"Bunny UI listening on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
