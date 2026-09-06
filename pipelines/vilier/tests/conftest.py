"""Generate the small audio fixture the upstream tests previously read from local data."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope='session', autouse=True)
def local_audio_fixture():
    path = Path('inputs/podcast_single_30s.wav')
    existed = path.exists()
    if not existed:
        path.parent.mkdir(parents=True, exist_ok=True)
        time = np.arange(30 * 16000) / 16000
        sf.write(path, .1 * np.sin(2 * np.pi * 220 * time), 16000)
    yield
    if not existed:
        path.unlink(missing_ok=True)
