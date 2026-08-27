from duplexchat_pipe.hydra_runner import config_from_hydra


def test_config_from_hydra_maps_vi_poc_values():
    cfg = config_from_hydra(
        {
            "source": {
                "languages": ["vi", "vi-vn"],
                "feed_allowlist": "sources/vi_allowlist.jsonl",
                "target_hours": 10,
            },
            "runtime": {
                "device": "auto",
                "multi_gpu": {"enabled": True, "device_ids": [0, 1]},
            },
            "diarization": {
                "enabled": True,
                "backend": "sortformer",
                "model": "nvidia/diar_sortformer_4spk-v1",
            },
            "separation": {
                "enabled": True,
                "backend": "mossformer2",
                "model": "alibabasglab/MossFormer2_SS_16K",
            },
        }
    )

    assert cfg.languages == ["vi", "vi-vn"]
    assert str(cfg.feed_allowlist) == "sources/vi_allowlist.jsonl"
    assert cfg.target_hours == 10
    assert cfg.runtime_device == "auto"
    assert cfg.multi_gpu_enabled is True
    assert cfg.multi_gpu_device_ids == [0, 1]
    assert cfg.enable_diarization is True
    assert cfg.diarization_backend == "sortformer"
    assert cfg.diarization_model == "nvidia/diar_sortformer_4spk-v1"
    assert cfg.separation_backend == "mossformer2"
