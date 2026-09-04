from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)


def ensure_ytdlp() -> None:
    if shutil.which("yt-dlp"):
        return
    try:
        subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("yt-dlp is required for youtube_allowlist sources") from exc


def _ytdlp_cmd() -> list[str]:
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def iter_entries(url: str, limit: int | None = None) -> list[dict]:
    ensure_ytdlp()
    cmd = [
        *_ytdlp_cmd(),
        "--dump-json",
        "--flat-playlist",
        "--ignore-errors",
        "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    entries: list[dict] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        webpage_url = raw.get("webpage_url") or raw.get("url")
        if webpage_url and not str(webpage_url).startswith(("http://", "https://")):
            webpage_url = f"https://www.youtube.com/watch?v={webpage_url}"
        if not webpage_url:
            continue
        entries.append(
            {
                "id": raw.get("id"),
                "title": raw.get("title"),
                "url": str(webpage_url),
                "duration": raw.get("duration"),
                "channel": raw.get("channel") or raw.get("uploader"),
            }
        )
        if limit is not None and len(entries) >= limit:
            break
    return entries


def download_audio(url: str, dest_dir: Path, key: str) -> Path:
    ensure_ytdlp()
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(dest_dir / f"{key}.%(ext)s")
    cmd = [
        *_ytdlp_cmd(),
        "--no-playlist",
        "--no-warnings",
        "-f",
        "bestaudio/best",
        "-o",
        output_template,
        url,
    ]
    LOGGER.info("Downloading YouTube audio: %s", url)
    subprocess.run(cmd, check=True)

    candidates = sorted(
        path for path in dest_dir.glob(f"{key}.*")
        if path.suffix != ".part" and path.stat().st_size > 0
    )
    if not candidates:
        raise RuntimeError(f"yt-dlp did not produce audio for {url}")
    return candidates[0]
