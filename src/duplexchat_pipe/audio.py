from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import requests
import torch


LOGGER = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be installed and on PATH")


def download_audio(url: str, dest: Path, timeout_seconds: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    temp_path = dest.with_suffix(dest.suffix + ".part")
    headers = DEFAULT_HEADERS
    LOGGER.info("Downloading audio: %s", url)
    with requests.get(url, stream=True, timeout=timeout_seconds, headers=headers) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temp_path.replace(dest)
    return dest


def probe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = result.stdout.strip()
    return float(output) if output else 0.0


def transcode_to_wav_16k_mono(in_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(in_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-f",
        "wav",
        str(out_path),
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return out_path


def extract_audio_segment(
    in_path: Path,
    out_path: Path,
    start: float,
    end: float,
    sample_rate: int,
    channels: int,
    bitrate_kbps: int,
) -> Path:
    """Extract [start, end] seconds from in_path and encode as CBR MP3."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bitrate = f"{bitrate_kbps}k"
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        str(in_path),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-minrate",
        bitrate,
        "-maxrate",
        bitrate,
        "-bufsize",
        bitrate,
        str(out_path),
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return out_path


def load_wav_tensor(wav_path: Path) -> tuple[torch.Tensor, int]:
    """Read a WAV file into a (1, T) float32 tensor using stdlib only."""
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        samples = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    elif sampwidth == 4:
        samples = torch.frombuffer(bytearray(raw), dtype=torch.int32).float() / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    # interleaved → (channels, time) → mix down to mono (1, time)
    waveform = samples.reshape(-1, n_channels).T.contiguous()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def tensors_to_stereo_mp3_bytes(
    spk0: torch.Tensor,
    spk1: torch.Tensor,
    sample_rate: int,
    bitrate_kbps: int,
) -> bytes:
    """Encode two mono (1, T) float32 tensors as a stereo CBR MP3 (spk0=L, spk1=R)."""
    left = np.clip(spk0.squeeze(0).cpu().float().numpy(), -1.0, 1.0)
    right = np.clip(spk1.squeeze(0).cpu().float().numpy(), -1.0, 1.0)
    # Interleave L R L R ... for s16le stereo
    length = min(len(left), len(right))
    interleaved = np.empty(length * 2, dtype=np.int16)
    interleaved[0::2] = (left[:length] * 32767).astype(np.int16)
    interleaved[1::2] = (right[:length] * 32767).astype(np.int16)
    pcm = interleaved.tobytes()
    bitrate = f"{bitrate_kbps}k"
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "2",
        "-i", "pipe:0",
        "-codec:a", "libmp3lame",
        "-b:a", bitrate, "-minrate", bitrate, "-maxrate", bitrate, "-bufsize", bitrate,
        "-f", "mp3", "pipe:1",
    ]
    result = subprocess.run(cmd, input=pcm, capture_output=True, check=True)
    return result.stdout


def transcode_to_mp3_cbr(
    in_path: Path,
    out_path: Path,
    sample_rate: int,
    channels: int,
    bitrate_kbps: int,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bitrate = f"{bitrate_kbps}k"
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(in_path),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-minrate",
        bitrate,
        "-maxrate",
        bitrate,
        "-bufsize",
        bitrate,
        str(out_path),
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return out_path
