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
