from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from duplexchat_pipe import cli
from duplexchat_pipe.cholimex.pipeline import _align_proposal_track, run_cholimex_file
from duplexchat_pipe.cholimex.models import ActivitySegment, Region
from duplexchat_pipe.cholimex.reconstruction import reconstruct_tracks
from duplexchat_pipe.cholimex.region_classifier import classify_regions
from duplexchat_pipe.cholimex.vad_masking import write_vad_artifacts
from duplexchat_pipe.config import Config, apply_overrides


class SignEmbedding:
    def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return torch.tensor([1.0, 0.0]) if wav.mean() >= 0 else torch.tensor([0.0, 1.0])


def test_cholimex_config_overrides():
    cfg = Config()

    apply_overrides(
        cfg,
        [
            "cholimex.enabled=true",
            "cholimex.overlap_padding=0.25",
            "cholimex.min_vad_duration=0.0",
            "cholimex.proposal_backend=dialoguesidon",
        ],
    )

    assert cfg.cholimex_enabled is True
    assert cfg.cholimex_overlap_padding == 0.25
    assert cfg.cholimex_min_vad_duration == 0.0
    assert cfg.cholimex_proposal_backend == "dialoguesidon"


def test_cholimex_cli_dispatches_single_audio_runner(monkeypatch, tmp_path: Path):
    calls = {}
    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_dir = tmp_path / "out"

    def fake_load_config(path: Path):
        calls["config_path"] = path
        return Config()

    def fake_apply_overrides(cfg: Config, overrides: list[str]):
        calls["overrides"] = overrides

    def fake_run(input_arg: Path, output_arg: Path, cfg: Config):
        calls["input"] = input_arg
        calls["output"] = output_arg
        calls["enabled"] = cfg.cholimex_enabled
        calls["device"] = cfg.runtime_device
        calls["steps"] = cfg.separation_num_steps
        return {}

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "apply_overrides", fake_apply_overrides)
    monkeypatch.setattr("duplexchat_pipe.cholimex.run_cholimex_file", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "duplexchat-pipe",
            "cholimex",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--runtime-device",
            "cpu",
            "--separation-num-steps",
            "2",
            "cholimex.overlap_padding=0.2",
        ],
    )

    cli.main()

    assert calls["config_path"] == Path("configs/config.json")
    assert calls["overrides"] == ["cholimex.overlap_padding=0.2"]
    assert calls["input"] == input_path
    assert calls["output"] == output_dir
    assert calls["enabled"] is True
    assert calls["device"] == "cpu"
    assert calls["steps"] == 2


def test_cholimex_cli_rejects_missing_input_cleanly(monkeypatch, tmp_path: Path, capsys):
    missing_input = tmp_path / "missing.wav"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "duplexchat-pipe",
            "cholimex",
            "--input",
            str(missing_input),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert f"Cholimex input audio file does not exist: {missing_input}" in capsys.readouterr().err


def test_cholimex_rejects_missing_input_before_transcode(monkeypatch, tmp_path: Path):
    missing_input = tmp_path / "missing.wav"
    output_dir = tmp_path / "out"

    def fail_transcode(*_):
        raise AssertionError("transcode should not run for a missing input")

    monkeypatch.setattr("duplexchat_pipe.cholimex.pipeline.audio.transcode_to_wav_16k_mono", fail_transcode)

    try:
        run_cholimex_file(missing_input, output_dir, Config())
    except FileNotFoundError as exc:
        assert str(missing_input) in str(exc)
    else:
        raise AssertionError("missing input should raise FileNotFoundError")


def test_cholimex_aligns_proposal_tracks_to_original_length():
    cropped = _align_proposal_track(
        torch.arange(7, dtype=torch.float32).reshape(1, -1),
        source_sample_rate=16000,
        target_sample_rate=16000,
        target_samples=5,
        label="sidon_track_0",
    )
    padded = _align_proposal_track(
        torch.ones(1, 3),
        source_sample_rate=16000,
        target_sample_rate=16000,
        target_samples=5,
        label="sidon_track_1",
    )

    assert torch.equal(cropped, torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.float32))
    assert torch.equal(padded, torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32))


def test_write_vad_artifacts_uses_only_active_one_labels(tmp_path: Path):
    segments = [
        ActivitySegment(0.0, 0.3, speaker=0),
        ActivitySegment(0.7, 1.0, speaker=0),
    ]

    write_vad_artifacts(tmp_path / "vad.json", tmp_path / "vad.txt", segments)

    assert (tmp_path / "vad.txt").read_text() == "0.000\t0.300\t1\n0.700\t1.000\t1\n"
    assert '"speaker": 0' in (tmp_path / "vad.json").read_text()


def test_region_classifier_preserves_short_backchannel():
    mask_0 = [ActivitySegment(0.0, 5.0, speaker=0)]
    mask_1 = [ActivitySegment(2.0, 2.4, speaker=1)]

    regions = classify_regions(mask_0, mask_1, duration_sec=5.0, backchannel_max_duration=1.0)

    assert [region.type for region in regions] == [
        "single_speaker",
        "overlap_backchannel",
        "single_speaker",
    ]
    assert regions[1].backchannel_speaker == 1
    assert regions[1].start == 2.0
    assert regions[1].end == 2.4


def test_single_speaker_reconstruction_preserves_original_and_skips_separator():
    original = torch.arange(10, dtype=torch.float32).reshape(1, -1)
    regions = [Region(0.0, 1.0, "single_speaker", speaker=0)]

    final_0, final_1, overlap_records = reconstruct_tracks(
        original,
        sample_rate=10,
        regions=regions,
        separator=lambda *_: (_ for _ in ()).throw(AssertionError("separator called")),
        references={},
        extractor=None,
        cosine_threshold=0.5,
        overlap_padding=0.0,
    )

    assert torch.equal(final_0, original)
    assert torch.equal(final_1, torch.zeros_like(original))
    assert overlap_records == []


def test_overlap_reconstruction_handles_separator_permutation_with_cosine():
    original = torch.zeros(1, 8)
    regions = [
        Region(0.0, 0.2, "overlap"),
        Region(0.2, 0.4, "overlap"),
    ]
    calls = {"count": 0}

    def separator(wav: torch.Tensor, sample_rate: int):
        calls["count"] += 1
        if calls["count"] == 1:
            return torch.ones_like(wav), -torch.ones_like(wav), sample_rate
        return -torch.ones_like(wav), torch.ones_like(wav), sample_rate

    final_0, final_1, overlap_records = reconstruct_tracks(
        original,
        sample_rate=10,
        regions=regions,
        separator=separator,
        references={0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])},
        extractor=SignEmbedding(),
        cosine_threshold=0.5,
        overlap_padding=0.0,
    )

    assert torch.equal(final_0[:, :4], torch.ones(1, 4))
    assert torch.equal(final_1[:, :4], -torch.ones(1, 4))
    assert overlap_records[0]["assignment"]["swapped"] is False
    assert overlap_records[1]["assignment"]["swapped"] is True


def test_overlap_padding_does_not_shift_reconstruction_boundaries():
    original = torch.zeros(1, 10)
    regions = [Region(0.2, 0.4, "overlap")]
    observed = {}

    def separator(wav: torch.Tensor, sample_rate: int):
        observed["samples"] = wav.shape[-1]
        return torch.ones_like(wav), -torch.ones_like(wav), sample_rate

    final_0, final_1, _ = reconstruct_tracks(
        original,
        sample_rate=10,
        regions=regions,
        separator=separator,
        references={},
        extractor=None,
        cosine_threshold=0.5,
        overlap_padding=0.1,
    )

    assert observed["samples"] == 4
    assert torch.equal(final_0, torch.tensor([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32))
    assert torch.equal(final_1, torch.tensor([[0, 0, -1, -1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32))
