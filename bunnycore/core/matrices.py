from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class SparseCOO:
    """Sparse operator in COO form. y = A x."""
    n_rows: int
    n_cols: int
    entries: List[Tuple[int,int,float]]

    def apply(self, x: List[float]) -> List[float]:
        y = [0.0] * self.n_rows
        n = len(x)
        for i, j, v in self.entries:
            if 0 <= j < n and 0 <= i < self.n_rows:
                y[i] += float(v) * float(x[j])
        return y

def identity(n: int, scale: float = 1.0) -> SparseCOO:
    n = int(n)
    return SparseCOO(n_rows=n, n_cols=n, entries=[(i,i,float(scale)) for i in range(n)])

def mask_entries(entries: List[Tuple[int,int,float]], allowed: Dict[Tuple[int,int], bool]) -> List[Tuple[int,int,float]]:
    out = []
    for i,j,v in entries:
        if allowed.get((i,j), True):
            out.append((i,j,v))
    return out
