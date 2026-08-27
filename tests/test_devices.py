import pytest

from duplexchat_pipe.devices import resolve_device, validate_multi_gpu


def test_auto_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_auto_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_device_resolver_never_returns_mps(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_device("auto") != "mps"


def test_multi_gpu_validation_rejects_unavailable_ids(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError, match="unavailable CUDA device ids"):
        validate_multi_gpu(True, [0, 1])
