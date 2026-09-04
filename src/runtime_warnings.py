from __future__ import annotations

import logging
import warnings


def suppress_pyannote_tf32_warning() -> None:
    """Hide noisy third-party runtime warnings without changing model behavior."""
    suppress_noisy_runtime_logs()
    try:
        from pyannote.audio.utils.reproducibility import ReproducibilityWarning
    except Exception:  # noqa: BLE001
        warnings.filterwarnings("ignore", message=r".*TensorFloat-32.*")
        warnings.filterwarnings("ignore", message=r".*TF32.*")
        return

    warnings.filterwarnings("ignore", category=ReproducibilityWarning)


def suppress_noisy_runtime_logs() -> None:
    warnings.filterwarnings("ignore", message=r".*legacy format.*torch\.export\.save.*")
    warnings.filterwarnings("ignore", message=r".*Please generate a new pt2 file.*")
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.load.*deprecated.*")
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "torch.export",
        "torch.export.pt2_archive",
        "torch.export.pt2_archive._package",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
