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


def test_separation_backend_routes_sepformer(monkeypatch):
    called = {}

    def fake_load(device="auto", model_id=None):
        called["device"] = device
        called["model_id"] = model_id
        return {"backend": "sepformer", "sample_rate": 16000}

    monkeypatch.setattr(separate, "_load_sepformer_models", fake_load)

    models = separate.load_separation_models(
        "cpu", backend="sepformer", model_id="sepformer"
    )

    assert models["backend"] == "sepformer"
    assert called == {"device": "cpu", "model_id": "speechbrain/sepformer-wsj02mix"}


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


def test_sepformer_run_uses_adapter():
    class FakeSepformer:
        def separate_batch(self, mixture):
            return torch.stack([mixture.squeeze(0), mixture.squeeze(0) * 0], dim=-1).unsqueeze(0)

    wav = torch.ones(1, 160)

    spk0, spk1, sr = separate.run_separation(
        wav,
        16000,
        0,
        {
            "backend": "sepformer",
            "sample_rate": 16000,
            "device": torch.device("cpu"),
            "model": FakeSepformer(),
        },
    )

    assert sr == 16000
    assert torch.equal(spk0, wav)
    assert torch.equal(spk1, wav * 0)


def test_dialoguesidon_separation_reports_chunk_progress(monkeypatch):
    events = []

    def fake_separate_chunk(wav, num_steps, models):
        return torch.stack([wav.reshape(-1), wav.reshape(-1) * 0], dim=0)

    monkeypatch.setattr(separate, "_separate_chunk", fake_separate_chunk)

    spk0, spk1, sr = separate.run_separation(
        torch.ones(1, 32000),
        16000,
        0,
        {
            "backend": "dialoguesidon",
            "sample_rate": 16000,
            "device": torch.device("cpu"),
        },
        chunk_seconds=1.0,
        overlap_seconds=0.5,
        progress_callback=lambda event, value: events.append((event, value)),
    )

    assert sr == 16000
    assert spk0.shape[0] == 1
    assert spk1.shape[0] == 1
    assert events[0] == ("start", 4)
    assert events.count(("advance", 1)) == 4
    assert events[-1] == ("close", 0)


def test_dialoguesidon_short_input_returns_original_timeline_length(monkeypatch):
    def fake_separate_chunk(wav, num_steps, models):
        return torch.stack([wav.reshape(-1), wav.reshape(-1) * 0], dim=0)

    monkeypatch.setattr(separate, "_separate_chunk", fake_separate_chunk)

    spk0, spk1, sr = separate.run_separation(
        torch.ones(1, 1600),
        16000,
        0,
        {
            "backend": "dialoguesidon",
            "sample_rate": 16000,
            "device": torch.device("cpu"),
        },
        chunk_seconds=30.0,
        overlap_seconds=5.0,
    )

    assert sr == 16000
    assert spk0.shape == (1, 1600)
    assert spk1.shape == (1, 1600)


def test_export_loader_maps_serialized_cuda_artifacts_to_cpu_and_restores(monkeypatch):
    import pytest
    from torch._export.serde import serialize
    original = serialize.deserialize_torch_artifact
    seen = []
    def fake_load(buffer, **kwargs):
        seen.append(kwargs)
        return {'tensor': 'loaded'}
    monkeypatch.setattr(torch, 'load', fake_load)
    def export_load(path):
        assert serialize.deserialize_torch_artifact(b'checkpoint') == {'tensor': 'loaded'}
        raise ValueError('test export failure')
    monkeypatch.setattr(torch.export, 'load', export_load)
    with pytest.raises(ValueError, match='test export failure'):
        separate._load_exported_module('model.pt2', torch.device('cpu'))
    assert serialize.deserialize_torch_artifact is original
    assert torch.load is fake_load
    assert seen == [{'weights_only': False, 'map_location': 'cpu'}]


def test_export_graph_device_constants_follow_requested_cpu():
    graph = torch.fx.Graph()
    value = graph.placeholder('value')
    graph.call_function(torch.ops.aten._assert_tensor_metadata.default,
                        (value,), {'device': torch.device('cuda:0'), 'dtype': torch.int64})
    graph.output(value)
    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    actual = torch.ones(2, dtype=torch.int64)
    import pytest
    with pytest.raises(RuntimeError, match='device mismatch'):
        module(actual)
    migrated = separate._retarget_exported_module(module, torch.device('cpu'))
    assert torch.equal(migrated(actual), actual)
