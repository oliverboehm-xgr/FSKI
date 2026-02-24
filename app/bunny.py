"""Bunny (Python) - interactive UI with state-core + first Speech organ.

Run:
  python -m app.bunny --db bunny.db --model llama3.3

Notes:
- This is V1. It is intentionally minimal but matches the "single channel" design:
  user text -> event -> heartbeat updates state -> speech organ reads state -> reply.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import requests

from bunnycore.core.db import init_db
from bunnycore.core.registry import ensure_axes
from bunnycore.core.matrix_store import MatrixStore
from bunnycore.core.adapters import AdapterRegistry, AdapterBinding, SimpleTextEncoder
from bunnycore.core.integrator import Integrator, IntegratorConfig
from bunnycore.core.heartbeat import Heartbeat, HeartbeatConfig
from bunnycore.core.events import Event
from bunnycore.core.matrices import identity


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
    # Ollama returns {'message': {'role':'assistant','content':...}, ...}
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()

def state_summary(named: dict[str, float]) -> str:
    # Keep it short and humanoid; the model will render final phrasing.
    keys = ["energy","stress","curiosity","confidence","uncertainty","social_need","urge_reply","urge_share"]
    parts = []
    for k in keys:
        if k in named:
            parts.append(f"{k}={named[k]:+.2f}")
    return ", ".join(parts)

def render_named_state(hb: Heartbeat, axis: dict[str,int]) -> dict[str,float]:
    s = hb.load_state()
    inv = {v:k for k,v in axis.items()}
    return {inv[i]: s.values[i] for i in range(len(s.values)) if i in inv}

def ensure_min_seed(db, axis, store: MatrixStore, reg: AdapterRegistry) -> None:
    # identity injection for user_utterance
    n = len(axis)
    mats = store.list_matrices()
    have = any(m["name"] == "A_user" and int(m["version"]) == 1 for m in mats)
    if not have:
        A = identity(n, scale=1.0)
        store.put_sparse("A_user", version=1, n_rows=n, n_cols=n, entries=A.entries,
                        meta={"desc":"identity injection for user_utterance"})
    reg.upsert(AdapterBinding(
        event_type="user_utterance",
        encoder_name="simple_text_v1",
        matrix_name="A_user",
        matrix_version=1,
        meta={"desc":"user text -> x -> add into state"},
    ))

def help_text() -> str:
    return """Commands:
  /help                 show this help
  /state                show current state (named axes)
  /matrices             list matrices in DB
  /seed                 seed default A_user + adapter binding
  /quit                 exit
Anything else is treated as user text input.
"""

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="bunny.db")
    ap.add_argument("--model", default=os.environ.get("BUNNY_MODEL","llama3.3"))
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST","http://localhost:11434"))
    ap.add_argument("--ctx", type=int, default=int(os.environ.get("BUNNY_CTX","4096")))
    args = ap.parse_args()

    db = init_db(args.db)
    axis = ensure_axes(db)

    store = MatrixStore(db)
    reg = AdapterRegistry(db)
    encoders = {"simple_text_v1": SimpleTextEncoder(axis)}

    integ = Integrator(store, reg, encoders, IntegratorConfig())
    hb = Heartbeat(db, integ, HeartbeatConfig(tick_hz=2.0, snapshot_every_n_ticks=1))

    ensure_min_seed(db, axis, store, reg)

    cfg = OllamaConfig(host=args.ollama, model=args.model, num_ctx=args.ctx)

    print("Bunny (py) online. Type /help.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            cmd = line.lower().split()
            if cmd[0] in ("/quit","/exit"):
                break
            if cmd[0] == "/help":
                print(help_text()); continue
            if cmd[0] == "/state":
                print(json.dumps(render_named_state(hb, axis), indent=2, ensure_ascii=False)); continue
            if cmd[0] == "/matrices":
                mats = store.list_matrices()
                for m in mats:
                    print(f"{m['name']}@{m['version']} {m['op_type']} {m['n_rows']}x{m['n_cols']}")
                continue
            if cmd[0] == "/seed":
                ensure_min_seed(db, axis, store, reg)
                print("OK seeded")
                continue
            print("Unknown command. /help")
            continue

        # --- single channel: user input -> event -> heartbeat -> speech organ ---
        hb.enqueue(Event("user_utterance", {"text": line}).with_time())
        hb.step()

        named = render_named_state(hb, axis)
        sys_prompt = (
            "You are Bunny, a digital organism. You must respond naturally, not like a robot. "
            "You have an internal state vector. Use it to describe your current mood and constraints honestly, "
            "but keep it human and conversational. Avoid canned phrases. "
            "If you mention an article, only do so if the user asked or if you have evidence; otherwise be transparent."
        )
        user_prompt = (
            f"INTERNAL_STATE: {state_summary(named)}\n\n"
            f"USER: {line}\n\n"
            "Reply as Bunny."
        )

        try:
            out = ollama_chat(cfg, sys_prompt, user_prompt)
        except Exception as e:
            out = f"(speech organ error: {e})"

        if out.strip() == "":
            out = "Ich bin da. Sag mir kurz, worum es geht."

        print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
