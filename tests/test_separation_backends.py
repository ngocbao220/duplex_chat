import torch

from duplexchat_pipe import separate


def test_separation_backend_routes_mossformer2(monkeypatch):
    called = {}

    def fake_load(device="auto", model_id=None):
        called["device"] = device
        called["model_id"] = model_id
        return {"backend": "mossformer2", "sample_rate": 16000}

    monkeypatch.setattr(separate, "_load_mossformer2_models", fake_load)

    models = separate.load_separation_models(
        "cpu", backend="mossformer2", model_id="alibabasglab/MossFormer2_SS_16K"
    )

    assert models["backend"] == "mossformer2"
    assert called == {"device": "cpu", "model_id": "alibabasglab/MossFormer2_SS_16K"}


def test_mossformer2_run_uses_adapter(monkeypatch):
    def fake_run(wav, sample_rate, models):
        return wav, wav * 0, sample_rate

    monkeypatch.setattr(separate, "_run_mossformer2_separation", fake_run)
    wav = torch.ones(1, 160)

    spk0, spk1, sr = separate.run_separation(
        wav, 16000, 30, {"backend": "mossformer2", "sample_rate": 16000}
    )

    assert sr == 16000
    assert torch.equal(spk0, wav)
    assert torch.equal(spk1, wav * 0)
