from __future__ import annotations

import warnings


def suppress_pyannote_tf32_warning() -> None:
    """Hide pyannote's TF32 reproducibility warning without changing TF32 settings."""
    try:
        from pyannote.audio.utils.reproducibility import ReproducibilityWarning
    except Exception:  # noqa: BLE001
        warnings.filterwarnings("ignore", message=r".*TensorFloat-32.*")
        warnings.filterwarnings("ignore", message=r".*TF32.*")
        return

    warnings.filterwarnings("ignore", category=ReproducibilityWarning)
