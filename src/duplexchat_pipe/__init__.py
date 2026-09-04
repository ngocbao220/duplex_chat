from __future__ import annotations

import importlib
import sys


_MODULES = [
    "model_options",
    "logging_utils",
    "config",
    "audio",
    "db",
    "devices",
    "dialogue",
    "outputs",
    "rss",
    "sources",
    "tags",
    "wds",
    "youtube",
    "runtime_warnings",
    "diarize",
    "separate",
    "benchmark",
    "cholimex",
    "pipeline",
    "runner",
    "cli",
]


for _name in _MODULES:
    _module = importlib.import_module(_name)
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module


__all__ = list(_MODULES)
