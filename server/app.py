"""OpenEnv server entry point — required by multi-mode deployment spec.

Uses openenv_core.create_fastapi_app to expose the AaxDebugEnv over HTTP
with auto-generated /reset, /step, /state, /schema, /health endpoints.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openenv_core import create_fastapi_app

from environment.env import AaxDebugEnv
from environment.models import AaxAction, AaxObservation

app = create_fastapi_app(
    AaxDebugEnv,
    AaxAction,
    AaxObservation,
)


def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
