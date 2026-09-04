import importlib
import importlib.util
import sys
import warnings
from pathlib import Path

from duplexchat_pipe.runtime_warnings import suppress_pyannote_tf32_warning


def _load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_suppresses_tf32_warning_message_without_pyannote(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyannote", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        suppress_pyannote_tf32_warning()
        warnings.warn("TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues.")


def test_entrypoints_call_warning_suppression(monkeypatch):
    calls = []
    end2end = _load_script("end2end_test_entrypoint", "end2end.py")
    cli = importlib.import_module("duplexchat_pipe.cli")

    monkeypatch.setattr(end2end, "suppress_pyannote_tf32_warning", lambda: calls.append("end2end"))
    monkeypatch.setattr(end2end, "load_config", lambda path: object())
    monkeypatch.setattr(end2end, "run_phase", lambda cfg, phase: None)
    monkeypatch.setattr(sys, "argv", ["end2end.py"])
    end2end.main()

    monkeypatch.setattr(cli, "suppress_pyannote_tf32_warning", lambda: calls.append("cli"))
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    monkeypatch.setattr(cli, "apply_overrides", lambda cfg, overrides: None)
    monkeypatch.setattr(cli, "_apply_run_args", lambda cfg, args: None)
    monkeypatch.setattr(cli, "run_phase", lambda cfg, phase: None)
    monkeypatch.setattr(sys, "argv", ["duplexchat-pipe", "run"])
    cli.main()

    assert calls == ["end2end", "cli"]
