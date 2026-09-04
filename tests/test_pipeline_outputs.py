from pathlib import Path
from types import SimpleNamespace

import torch

from duplexchat_pipe.config import Config
from duplexchat_pipe.pipeline import (
    AudioItem,
    SeparationTask,
    _build_dialogue_meta,
    _debug_episode_dir,
    _separate_dialogue,
    _write_debug_diarization_outputs,
    _write_debug_separation_outputs,
)


def test_debug_episode_dir_sanitizes_key(tmp_path: Path):
    cfg = Config(debug_outputs_dir=tmp_path)

    assert _debug_episode_dir(cfg, "abc/def ghi").name == "abc_def_ghi"


def test_write_debug_diarization_outputs(tmp_path: Path):
    cfg = Config(
        debug_outputs_enabled=True,
        debug_outputs_dir=tmp_path,
        diarization_backend="pyannote",
        diarization_model="model-id",
    )
    wav_path = tmp_path / "source.wav"
    wav_path.write_bytes(b"fake wav")
    dialogue = SimpleNamespace(
        start=0.0,
        end=2.0,
        duration=2.0,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=[
            {"speaker": "raw_a", "start": 0.0, "end": 1.0},
            {"speaker": "raw_b", "start": 1.0, "end": 2.0},
        ],
    )

    _write_debug_diarization_outputs(
        cfg,
        "episode-key",
        wav_path,
        dialogue.segments,
        [dialogue],
        duration_sec=2.0,
    )

    out_dir = tmp_path / "episode-key"
    assert (out_dir / "phase_01_preprocess" / "audio_16k_mono.wav").read_bytes() == b"fake wav"
    assert (out_dir / "phase_02_diarization" / "labels" / "SPEAKER_00.txt").read_text() == (
        "0.000\t1.000\tSPEAKER_00\n"
    )
    assert (out_dir / "phase_03_dialogues" / "dialogues.json").exists()


def test_write_debug_separation_outputs(tmp_path: Path, monkeypatch):
    cfg = Config(
        debug_outputs_enabled=True,
        debug_outputs_dir=tmp_path,
        separation_backend="sepformer",
        separation_model="speechbrain/sepformer-wsj02mix",
    )
    task = SeparationTask(
        dlg_key="episode_0000",
        episode_key="episode",
        dlg_wav=torch.zeros(1, 16),
        dlg_start=0.0,
        dlg_end=1.0,
        dialogue=SimpleNamespace(),
        item=AudioItem("audio", "rss", "vi", {}, {}),
        episode_duration=1.0,
        cfg=cfg,
        device="cpu",
    )

    def fake_save_wav(path, wav, sample_rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{tuple(wav.shape)} {sample_rate}", encoding="utf-8")
        return path

    monkeypatch.setattr("duplexchat_pipe.pipeline.outputs_mod.save_wav", fake_save_wav)

    _write_debug_separation_outputs(
        task,
        torch.zeros(1, 16),
        torch.ones(1, 16),
        sample_rate=16000,
        model_id="model-id",
    )

    phase_dir = tmp_path / "episode" / "phase_04_separation" / "episode_0000"
    assert (phase_dir / "speaker_A.wav").read_text() == "(1, 16) 16000"
    assert (phase_dir / "speaker_B.wav").read_text() == "(1, 16) 16000"
    assert "model-id" in (phase_dir / "separation.json").read_text()


def test_dialogue_meta_uses_configured_metadata_language():
    cfg = Config(metadata_language="vi")
    item = AudioItem("audio", "rss", "vi-vn", {}, {})
    dialogue = SimpleNamespace(
        start=0.0,
        end=12.0,
        duration=12.0,
        speakers=["SPEAKER_00", "SPEAKER_01"],
    )

    meta = _build_dialogue_meta(cfg, item, 60.0, 0, dialogue)

    assert meta["language"] == "vi"


def test_separate_dialogue_loads_model_for_task_device(monkeypatch):
    loaded_devices = []

    def fake_load(device, backend, model_id):
        loaded_devices.append(device)
        return {
            "backend": "dialoguesidon",
            "model_id": "fake-model",
            "sample_rate": 16000,
            "device": torch.device(device),
        }

    def fake_run(wav, sample_rate, num_steps, models, progress_callback=None):
        assert str(models["device"]) == "cuda:1"
        return wav, wav * 0, sample_rate

    monkeypatch.setattr("duplexchat_pipe.pipeline.separate_mod.load_separation_models", fake_load)
    monkeypatch.setattr("duplexchat_pipe.pipeline.separate_mod.run_separation", fake_run)
    monkeypatch.setattr("duplexchat_pipe.pipeline.audio.tensors_to_stereo_mp3_bytes", lambda *args: b"mp3")
    monkeypatch.setattr("duplexchat_pipe.pipeline._write_debug_separation_outputs", lambda *args: None)

    cfg = Config(enable_separation=True, separation_backend="dialoguesidon", separation_model="fake-model")
    task = SeparationTask(
        dlg_key="episode_0001",
        episode_key="episode",
        dlg_wav=torch.ones(1, 160),
        dlg_start=0.0,
        dlg_end=0.01,
        dialogue=SimpleNamespace(start=0.0, end=0.01, duration=0.01, segments=[], speakers=["A", "B"]),
        item=AudioItem(
            audio_url="https://example.com/audio.wav",
            rss_url="https://example.com/feed.xml",
            language="vi",
            feed_meta={},
            entry_meta={},
        ),
        episode_duration=1.0,
        cfg=cfg,
        device="cuda:1",
    )

    result = _separate_dialogue(task, None, None)

    assert result.status == "ok"
    assert loaded_devices == ["cuda:1"]
    assert result.meta is not None
    assert result.meta["device"] == "cuda:1"
