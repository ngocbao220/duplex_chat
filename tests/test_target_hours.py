import importlib.util
from pathlib import Path

from duplexchat_pipe.config import Config
from duplexchat_pipe import pipeline


spec = importlib.util.spec_from_file_location("end2end", Path("end2end.py"))
end2end = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(end2end)


class FakePbar:
    def __init__(self):
        self.updates = 0
        self.postfix = ""

    def update(self, value):
        self.updates += value

    def set_postfix_str(self, value, refresh=True):
        self.postfix = value


class FakeDB:
    def __init__(self):
        self.marked = []

    def mark(self, *args):
        self.marked.append(args)


def test_end2end_does_not_override_config_target_hours(monkeypatch):
    cfg = Config(target_hours=1.0)
    seen = {}

    monkeypatch.setattr(end2end, "load_config", lambda path: cfg)
    monkeypatch.setattr(end2end, "run_phase", lambda cfg_arg, phase: seen.update(cfg=cfg_arg, phase=phase))
    monkeypatch.setattr("sys.argv", ["end2end.py"])

    end2end.main()

    assert seen["phase"] == "end2end"
    assert seen["cfg"].target_hours == 1.0


def test_end2end_overrides_target_hours_when_flag_is_passed(monkeypatch):
    cfg = Config(target_hours=1.0)
    seen = {}

    monkeypatch.setattr(end2end, "load_config", lambda path: cfg)
    monkeypatch.setattr(end2end, "run_phase", lambda cfg_arg, phase: seen.update(cfg=cfg_arg, phase=phase))
    monkeypatch.setattr("sys.argv", ["end2end.py", "--target_hours", "2"])

    end2end.main()

    assert seen["cfg"].target_hours == 2.0


def test_end2end_sets_youtube_only_when_flag_is_passed(monkeypatch):
    cfg = Config(youtube_only=False)
    seen = {}

    monkeypatch.setattr(end2end, "load_config", lambda path: cfg)
    monkeypatch.setattr(end2end, "run_phase", lambda cfg_arg, phase: seen.update({"cfg": cfg_arg, "phase": phase}))
    monkeypatch.setattr("sys.argv", ["end2end.py", "--youtube-only"])

    end2end.main()

    assert seen["cfg"].youtube_only is True
    assert seen["phase"] == "end2end"


def test_handle_processed_result_skips_write_after_target_reached(monkeypatch):
    writes = []
    cfg = Config(target_hours=1.0, cleanup_audio_cache=False)
    stats = pipeline.Stats(total_duration_sec=3600.0)
    result = pipeline.ProcessedResult(
        key="sample",
        status="ok",
        meta={"duration_sec": 10.0},
        audio_bytes=b"audio",
        diarization=None,
        diarization_error=None,
    )

    monkeypatch.setattr(pipeline.wds, "write_sample", lambda *args: writes.append(args))
    pipeline._handle_processed_result(result, cfg, object(), FakeDB(), FakePbar(), stats)

    assert writes == []
    assert stats.written == 0
    assert stats.skipped == 1
