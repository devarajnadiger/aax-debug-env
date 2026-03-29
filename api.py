"""Backward-compatible re-export. Canonical app lives in server/app.py."""
from server.app import app, main  # noqa: F401

if __name__ == "__main__":
    main()
