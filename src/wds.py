from __future__ import annotations

import json
from pathlib import Path

import webdataset as wds


def open_shard_writer(output_dir: Path, shard_size_gb: float) -> wds.ShardWriter:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "%06d.tar.gz")
    return wds.ShardWriter(pattern, maxsize=int(shard_size_gb * 1024 ** 3))


def write_sample(
    writer: wds.ShardWriter,
    key: str,
    audio_bytes: bytes | None,
    meta: dict,
    diarization: dict | None,
    diarization_error: dict | None,
) -> None:
    sample: dict = {
        "__key__": key,
        "meta.json": json.dumps(meta, ensure_ascii=False),
    }
    if audio_bytes:
        sample["audio.mp3"] = audio_bytes
    if diarization is not None:
        sample["diarization.json"] = json.dumps(diarization, ensure_ascii=False)
    if diarization_error is not None:
        sample["diarization_error.json"] = json.dumps(diarization_error, ensure_ascii=False)

    writer.write(sample)
