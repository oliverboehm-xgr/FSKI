# BunnyCore (V1)

A minimal, generic Python core for a **state-is-everything** organism.

## Core equation
`S' = clip(decay*S + Σ A_k φ(E_k))`

- `S`: DB-backed state vector (axes are DB-defined and extendable)
- `E_k`: events (user, websense, memory, sensors, ...)
- `φ`: encoder per event_type
- `A_k`: versioned sparse matrix operator per event_type
- `decay`: scalar (V1), later replace with diagonal/sparse matrix

## Quickstart
```bash
python run_demo.py init demo.db
python run_demo.py seed demo.db
python run_demo.py step demo.db "Hey Bunny, how are you?"
python run_demo.py show demo.db
```

## Extensibility
- Add new axes: insert into `state_axes` (or use `ensure_axes`)
- Add new matrices: `matrices` + `matrix_entries` (sparse COO)
- Bind events to adapters: `adapters` table (event_type -> encoder + matrix)
- Replace encoders with embeddings, vision features, etc.


## UI
Run interactive Bunny:
```bash
python -m app.bunny --db bunny.db --model llama3.3
```


## Web UI (like old Go main)

Run:

```bash
python -m app.ui --db bunny.db --model llama3.3 --addr 127.0.0.1:8080
```

Then open the printed URL in your browser.
