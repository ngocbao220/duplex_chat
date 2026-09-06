from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
import uuid
from pathlib import Path


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n')
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(pipeline: str, source: Path, config: dict, code: str) -> str:
    payload = [pipeline, sha256(source), config, code]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def validate_tracks(source: Path, tracks: list[Path]) -> float:
    import numpy as np
    import soundfile as sf

    if len(tracks) != 2 or tracks[0].resolve() == tracks[1].resolve():
        raise ValueError('Expected exactly two distinct speaker tracks')
    reference = sf.info(source)
    duration = reference.frames / reference.samplerate
    if duration <= 0:
        raise ValueError('Empty input timeline')
    for path in tracks:
        info = sf.info(path)
        if info.channels != 1 or info.frames <= 0:
            raise ValueError(f'Expected nonempty mono track: {path}')
        if abs(info.frames / info.samplerate - duration) > max(1 / info.samplerate, 1 / reference.samplerate):
            raise ValueError(f'Track timeline differs from input: {path}')
        for block in sf.blocks(path, blocksize=65536):
            if not np.isfinite(block).all():
                raise ValueError(f'Nonfinite samples: {path}')
    return duration


def vilier_tracks(manifest: Path) -> list[Path]:
    speakers = json.loads(manifest.read_text())['speakers']
    if len(speakers) != 2:
        raise ValueError(f'Vilier must produce exactly two speakers; got {len(speakers)}')
    return [manifest.parent / speaker['track_wav'] for speaker in speakers]


def reusable(output: Path, identity: str, source: Path) -> dict | None:
    try:
        result = json.loads((output / 'run.json').read_text())
        if result['status'] != 'complete' or result['fingerprint'] != identity:
            return None
        tracks = [output / name for name in ('speaker_A.wav', 'speaker_B.wav')]
        validate_tracks(source, tracks)
        if result['track_sha256'] != [sha256(path) for path in tracks]:
            return None
        return result
    except (OSError, ValueError, KeyError, RuntimeError):
        return None


def run_sample(pipeline, sample, output, config, code, adapter, force=False) -> dict:
    source = Path(sample['mixture'])
    output = Path(output)
    started = time.perf_counter()
    result = {'key': sample['key'], 'pipeline': pipeline, 'config': config, 'code': code,
              'input': str(source), 'status': 'running', 'resumed': False}
    try:
        identity = fingerprint(pipeline, source, config, code)
        previous = reusable(output, identity, source) if not force else None
        if previous:
            return {**previous, 'resumed': True}
        if output.exists():
            archive = output.parent / '.history' / uuid.uuid4().hex / output.name
            archive.parent.mkdir(parents=True, exist_ok=True)
            output.rename(archive)
        output.mkdir(parents=True, exist_ok=True)
        result['fingerprint'] = identity
        write_json(output / 'run.json', result)
        tracks, metadata = adapter(source, output, config)
        duration = validate_tracks(source, tracks)
        canonical = [output / name for name in ('speaker_A.wav', 'speaker_B.wav')]
        for original, target in zip(tracks, canonical):
            if original.resolve() != target.resolve():
                shutil.copy2(original, target)
        elapsed = time.perf_counter() - started
        result.update(status='complete', duration_sec=duration, inference_seconds=elapsed,
                      rtf=elapsed / duration, metadata=metadata,
                      track_sha256=[sha256(path) for path in canonical])
    except Exception as exc:
        traceback.print_exc()
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      inference_seconds=time.perf_counter() - started)
    write_json(output / 'run.json', result)
    return result
