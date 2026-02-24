from __future__ import annotations

"""Convenience entrypoint.

Historically the project exposed `python -m app.bunny`. The UI/server lives in
`app.ui`, so this module just forwards to it.

Run:
  python -m app.bunny --db bunny.db --model llama3.1:8b --addr 127.0.0.1:8080
"""

from app.ui import main


if __name__ == "__main__":
    raise SystemExit(main())
