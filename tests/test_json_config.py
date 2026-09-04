from pathlib import Path

import pytest

from duplexchat_pipe.config import Config, apply_config_data, apply_overrides, load_config


def test_load_config_maps_default_project_values(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
{
  "source": {
    "languages": ["vi", "vi-vn"],
    "feed_allowlist": "configs/allowlist.jsonl",
    "target_hours": 10
  },
  "metadata": {
    "language": "vi"
  },
  "runtime": {
    "device": "auto",
    "multi_gpu": {"enabled": true, "device_ids": [0, 1]},
    "debug_outputs": {"enabled": false, "dir": "outputs/debug"}
  },
  "diarization": {
    "enabled": true,
    "backend": "sortformer",
    "model": "nvidia/diar_sortformer_4spk-v1"
  },
  "separation": {
    "enabled": true,
    "backend": "mossformer2",
    "model": "alibabasglab/MossFormer2_SS_16K"
  },
  "benchmark": {"enabled": false}
}
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.languages == ["vi", "vi-vn"]
    assert cfg.metadata_language == "vi"
    assert cfg.feed_allowlist == Path("configs/allowlist.jsonl")
    assert cfg.target_hours == 10
    assert cfg.runtime_device == "auto"
    assert cfg.multi_gpu_enabled is True
    assert cfg.multi_gpu_device_ids == [0, 1]
    assert cfg.debug_outputs_enabled is False
    assert cfg.debug_outputs_dir == Path("outputs/debug")
    assert cfg.enable_diarization is True
    assert cfg.diarization_backend == "sortformer"
    assert cfg.diarization_model == "nvidia/diar_sortformer_4spk-v1"
    assert cfg.enable_separation is True
    assert cfg.separation_backend == "mossformer2"
    assert cfg.separation_model == "alibabasglab/MossFormer2_SS_16K"


def test_apply_overrides_parses_scalars_lists_and_nulls():
    cfg = Config()

    apply_overrides(
        cfg,
        [
            "source.target_hours=2.5",
            "runtime.multi_gpu.enabled=true",
            "runtime.multi_gpu.device_ids=[0,1]",
            "source.feed_allowlist=null",
            "audio.output_dir=data/wds_2h5",
        ],
    )

    assert cfg.target_hours == 2.5
    assert cfg.multi_gpu_enabled is True
    assert cfg.multi_gpu_device_ids == [0, 1]
    assert cfg.feed_allowlist is None
    assert cfg.output_dir == Path("data/wds_2h5")


def test_apply_overrides_supports_backend_profile_shortcuts():
    cfg = Config()

    apply_overrides(cfg, ["diarization=diarizen", "separation=sepformer"])

    assert cfg.enable_diarization is True
    assert cfg.diarization_backend == "diarizen"
    assert cfg.diarization_model == "BUT-FIT/diarizen-wavlm-large-s80-md"
    assert cfg.enable_separation is True
    assert cfg.separation_backend == "sepformer"
    assert cfg.separation_model == "speechbrain/sepformer-wsj02mix"
    assert cfg.separation_num_steps == 0


def test_apply_config_data_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown config key"):
        apply_config_data(Config(), {"source": {"missing": "value"}})
