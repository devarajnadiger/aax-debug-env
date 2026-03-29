"""Backward-compatible re-export. The canonical app lives in server/app.py."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.app import app, main  # noqa: F401

if __name__ == "__main__":
    main()
