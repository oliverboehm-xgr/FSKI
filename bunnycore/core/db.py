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
        con.commit()
    finally:
        con.close()
    return db
