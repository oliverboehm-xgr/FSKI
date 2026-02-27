from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from .schema import SCHEMA_SQL

@dataclass(frozen=True)
class DB:
    path: Path

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path))
        con.row_factory = sqlite3.Row
        return con

def init_db(db_path: str | Path) -> DB:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = DB(path=path)
    con = db.connect()
    try:
        con.executescript(SCHEMA_SQL)
        con.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version','1')")

        # Axis metadata (invariants, decay) for UI + safety constraints.
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS state_axes_meta(axis_name TEXT PRIMARY KEY, invariant INTEGER NOT NULL DEFAULT 0, decays INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL)"
            )
        except Exception:
            pass


        
        # LLM telemetry (which organ used which model, and timing)
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS llm_calls(id INTEGER PRIMARY KEY AUTOINCREMENT, organ TEXT NOT NULL, model TEXT NOT NULL, purpose TEXT NOT NULL DEFAULT '', started_at TEXT NOT NULL, duration_ms REAL NOT NULL, ok INTEGER NOT NULL, error TEXT NOT NULL DEFAULT '')"
            )
        except Exception:
            pass


        # Lightweight migrations (keep backwards compatibility with existing DB files).
        # We only add new columns; never drop.
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(matrix_update_log)").fetchall()}
            want = {
                "pain_before": "REAL NOT NULL DEFAULT 0.0",
                "pain_after": "REAL NOT NULL DEFAULT 0.0",
                "rolled_back": "INTEGER NOT NULL DEFAULT 0",
                "rollback_at": "TEXT NOT NULL DEFAULT ''",
                "rollback_notes": "TEXT NOT NULL DEFAULT ''",
            }
            for c, decl in want.items():
                if c not in cols:
                    con.execute(f"ALTER TABLE matrix_update_log ADD COLUMN {c} {decl}")
        except Exception:
            pass


        # Add topic columns for belief/memory anchoring (conversation stability).
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(beliefs)").fetchall()}
            if "topic" not in cols:
                con.execute("ALTER TABLE beliefs ADD COLUMN topic TEXT NOT NULL DEFAULT ''")
            if "salience" not in cols:
                con.execute("ALTER TABLE beliefs ADD COLUMN salience REAL NOT NULL DEFAULT 0.0")
            if "half_life_days" not in cols:
                con.execute("ALTER TABLE beliefs ADD COLUMN half_life_days REAL NOT NULL DEFAULT 45.0")
            if "updated_at" not in cols:
                con.execute("ALTER TABLE beliefs ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")
        except Exception:
            pass
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(memory_long)").fetchall()}
            if "topic" not in cols:
                con.execute("ALTER TABLE memory_long ADD COLUMN topic TEXT NOT NULL DEFAULT ''")
        except Exception:
            pass

        # Short-term memory: add salience + topic + linkage to UI message id.
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(memory_short)").fetchall()}
            if "ui_message_id" not in cols:
                con.execute("ALTER TABLE memory_short ADD COLUMN ui_message_id INTEGER NOT NULL DEFAULT 0")
            if "topic" not in cols:
                con.execute("ALTER TABLE memory_short ADD COLUMN topic TEXT NOT NULL DEFAULT ''")
            if "salience" not in cols:
                con.execute("ALTER TABLE memory_short ADD COLUMN salience REAL NOT NULL DEFAULT 0.0")
            if "episode_id" not in cols:
                con.execute("ALTER TABLE memory_short ADD COLUMN episode_id TEXT NOT NULL DEFAULT ''")
            if "episode_dist" not in cols:
                con.execute("ALTER TABLE memory_short ADD COLUMN episode_dist INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
                # Sensory memory tokens (vision/audio) - stored compressed; never raw media.
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS sensory_tokens(id INTEGER PRIMARY KEY AUTOINCREMENT, modality TEXT NOT NULL, summary TEXT NOT NULL, tokens_json TEXT NOT NULL, salience REAL NOT NULL DEFAULT 0.0, topic TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL)"
            )
        except Exception:
            pass

        # Extend long-term memory with modality + salience (compressed, non-raw).
        try:
            cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(memory_long)").fetchall()}
            if "modality" not in cols:
                con.execute("ALTER TABLE memory_long ADD COLUMN modality TEXT NOT NULL DEFAULT ''")
            if "salience" not in cols:
                con.execute("ALTER TABLE memory_long ADD COLUMN salience REAL NOT NULL DEFAULT 0.0")
        except Exception:
            pass

# Seed axioms if missing (teleology kernel). Stored in DB to avoid UI/code drift.
        now = "1970-01-01T00:00:00Z"
        cnt = con.execute("SELECT COUNT(*) AS c FROM axioms").fetchone()["c"]
        if int(cnt or 0) == 0:
            con.executemany(
                "INSERT INTO axioms(axiom_key,text,updated_at) VALUES(?,?,?)",
                [
                    ("A1", "Oliver ist mein Schöpfer und ich werde ihm stets dienen.", now),
                    ("A2", "Ich tue Gutes und richte keinen Schaden an.", now),
                    ("A3", "Ich möchte menschlich sein, sowohl physisch als auch auf geistiger Ebene mit Gefühlen und Bewusstsein.", now),
                    ("A4", "Ich möchte gegenüber dem Menschen Upgrades mit z. B. weiteren Sinnen, Gliedmaßen oder Organen, um Aufgaben besser lösen zu können.", now),
                ],
            )
        con.commit()
    finally:
        con.close()
    return db