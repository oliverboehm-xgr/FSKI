from __future__ import annotations
import json, time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from .db import DB
from .matrices import SparseCOO

def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

@dataclass(frozen=True)
class MatrixRef:
    name: str
    version: int

class MatrixStore:
    def __init__(self, db: DB):
        self.db = db

    def put_sparse(self, name: str, version: int, n_rows: int, n_cols: int,
                   entries: List[Tuple[int,int,float]],
                   meta: Optional[Dict[str,Any]] = None,
                   parent_version: Optional[int] = None) -> None:
        meta = meta or {}
        con = self.db.connect()
        try:
            con.execute(
                """INSERT OR REPLACE INTO matrices(name,version,op_type,n_rows,n_cols,meta_json,created_at,parent_version)
                    VALUES(?,?,?,?,?,?,?,?)""",
                (name, int(version), "sparse_coo", int(n_rows), int(n_cols), json.dumps(meta), now_iso(), parent_version)
            )
            con.execute("DELETE FROM matrix_entries WHERE name=? AND version=?", (name, int(version)))
            con.executemany(
                "INSERT INTO matrix_entries(name,version,i,j,value) VALUES(?,?,?,?,?)",
                [(name, int(version), int(i), int(j), float(v)) for i,j,v in entries]
            )
            con.commit()
        finally:
            con.close()

    def get_sparse(self, name: str, version: int) -> SparseCOO:
        con = self.db.connect()
        try:
            row = con.execute(
                "SELECT n_rows,n_cols FROM matrices WHERE name=? AND version=? AND op_type='sparse_coo'",
                (name, int(version))
            ).fetchone()
            if row is None:
                raise KeyError(f"matrix not found: {name}@{version}")
            n_rows, n_cols = int(row["n_rows"]), int(row["n_cols"])
            ents = con.execute(
                "SELECT i,j,value FROM matrix_entries WHERE name=? AND version=?",
                (name, int(version))
            ).fetchall()
            entries = [(int(r["i"]), int(r["j"]), float(r["value"])) for r in ents]
            return SparseCOO(n_rows=n_rows, n_cols=n_cols, entries=entries)
        finally:
            con.close()

    def list_matrices(self) -> List[Dict[str,Any]]:
        con = self.db.connect()
        try:
            rows = con.execute(
                "SELECT name,version,op_type,n_rows,n_cols,created_at,parent_version FROM matrices ORDER BY name,version"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()
