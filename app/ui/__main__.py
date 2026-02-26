"""Module entrypoint.

Allows running the UI as:
  python -m app.ui --db bunny.db --model llama3.1:8b --addr 127.0.0.1:8080
"""

from .main import main


if __name__ == "__main__":
    raise SystemExit(main())
