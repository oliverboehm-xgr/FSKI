from __future__ import annotations

"""Thin wrapper to keep CLI stable while modularizing the server implementation.

The full implementation lives in app.ui_app.
"""

from app.ui_app import main

if __name__ == "__main__":
    raise SystemExit(main())
