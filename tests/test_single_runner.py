import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("test_single", Path("test_single.py"))
test_single = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(test_single)


def test_release_diarization_gpu_memory_accepts_adapter_without_to(monkeypatch):
    monkeypatch.setattr(test_single.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(test_single.torch.cuda, "empty_cache", lambda: None)

    class AdapterWithoutTo:
        pass

    test_single.release_diarization_gpu_memory(AdapterWithoutTo())


def test_release_diarization_gpu_memory_moves_supported_pipeline(monkeypatch):
    monkeypatch.setattr(test_single.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(test_single.torch.cuda, "empty_cache", lambda: None)
    called = {}

    class PipelineWithTo:
        def to(self, device):
            called["device"] = str(device)

    test_single.release_diarization_gpu_memory(PipelineWithTo())

    assert called == {"device": "cpu"}


def test_resolve_output_dir_defaults_to_output_prefix_parent():
    assert test_single.resolve_output_dir("runs/model_a/speaker") == Path("runs/model_a")


def test_resolve_output_dir_uses_explicit_output_dir():
    assert test_single.resolve_output_dir("speaker", "outputs/item") == Path("outputs/item")
