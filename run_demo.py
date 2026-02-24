"""BunnyCore demo.

Usage:
  python run_demo.py init demo.db
  python run_demo.py seed demo.db
  python run_demo.py step demo.db "Hello, how are you?"
  python run_demo.py show demo.db
"""
from __future__ import annotations
import sys, json
from bunnycore.core.db import init_db
from bunnycore.core.registry import ensure_axes
from bunnycore.core.matrix_store import MatrixStore
from bunnycore.core.adapters import AdapterRegistry, AdapterBinding, SimpleTextEncoder
from bunnycore.core.integrator import Integrator, IntegratorConfig
from bunnycore.core.heartbeat import Heartbeat, HeartbeatConfig
from bunnycore.core.events import Event
from bunnycore.core.matrices import identity

def usage():
    print(__doc__.strip())

def main():
    if len(sys.argv) < 3:
        usage(); return 2
    cmd = sys.argv[1].lower()
    db_path = sys.argv[2]
    db = init_db(db_path)
    axis = ensure_axes(db)
    store = MatrixStore(db)
    reg = AdapterRegistry(db)
    encoders = {"simple_text_v1": SimpleTextEncoder(axis)}
    integ = Integrator(store, reg, encoders, IntegratorConfig())
    hb = Heartbeat(db, integ, HeartbeatConfig(tick_hz=2.0, snapshot_every_n_ticks=1))

    if cmd == "init":
        print("OK init", db_path); return 0

    if cmd == "seed":
        n = len(axis)
        A = identity(n, scale=1.0)
        store.put_sparse("A_user", version=1, n_rows=n, n_cols=n, entries=A.entries, meta={"desc":"identity injection for user_utterance"})
        reg.upsert(AdapterBinding(event_type="user_utterance", encoder_name="simple_text_v1", matrix_name="A_user", matrix_version=1, meta={}))
        print("OK seeded A_user identity + adapter binding"); return 0

    if cmd == "step":
        if len(sys.argv) < 4:
            print("Missing text"); return 2
        text = sys.argv[3]
        hb.enqueue(Event("user_utterance", {"text": text}).with_time())
        hb.step()
        print("OK step"); return 0

    if cmd == "show":
        s = hb.load_state()
        inv = {v:k for k,v in axis.items()}
        named = {inv[i]: s.values[i] for i in range(len(s.values)) if i in inv}
        print(json.dumps(named, indent=2, ensure_ascii=False))
        return 0

    usage()
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
