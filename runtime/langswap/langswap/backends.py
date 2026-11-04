"""Single source of truth for the selectable pipeline backends.

The available ASR / translation / TTS engines and dubbing algorithms are declared
in ``backends.json`` (next to this file) — edit that to add or hide a backend, and
both the Gradio UI and any validation pick it up.  The actual client classes are
still wired in the per-service dispatch (``langswap/ml/*/__init__.py``); this module
only describes *what is offered*, not how it is constructed.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "backends.json"


@lru_cache(maxsize=1)
def load_backends() -> dict:
    """Load and cache the backend registry from backends.json."""
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def options(category: str) -> list[str]:
    """Selectable backend names for a category (asr / translation / tts / dubbing)."""
    return list(load_backends()[category]["options"].keys())


def default(category: str) -> str:
    """The default backend name for a category."""
    return load_backends()[category]["default"]
