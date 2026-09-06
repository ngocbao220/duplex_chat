from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torchaudio

from duplexchat_pipe.outputs import write_label_file
from duplexchat_pipe.runtime_warnings import suppress_pyannote_tf32_warning
from duplexchat_pipe.single_audio import run_single_audio


DEFAULT_DIARIZATION_BACKEND = "auto"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_SEPARATION_BACKEND = "dialoguesidon"
DEFAULT_SEPARATION_MODEL = None


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_text(temp_output_dir: Path, speaker_a: Path, speaker_b: Path) -> dict[str, str]:
    input_meta = _read_json(temp_output_dir / "phase_00_input" / "input.json")
    diarization_meta = _read_json(temp_output_dir / "phase_02_diarization" / "diarization.json")
    separation_meta = _read_json(temp_output_dir / "phase_04_separation" / "separation.json")
    segments = diarization_meta.get("segments") or []
    return {
        "input": f"Input loaded: {input_meta.get('audio_name') or 'unknown'}",
        "preprocess": "Converted input audio to 16 kHz mono WAV.",
        "diarization": f"Detected {len(segments)} diarization segments.",
        "separation": (
            "Separated audio into two speaker tracks "
            f"at {separation_meta.get('sample_rate') or 'unknown'} Hz."
        ),
        "output": f"Wrote {speaker_a.name} and {speaker_b.name}.",
    }


def _diarization_segments(temp_output_dir: Path) -> list[dict]:
    diarization_meta = _read_json(temp_output_dir / "phase_02_diarization" / "diarization.json")
    return list(diarization_meta.get("segments") or [])


def _fallback_duration_sec(segments: list[dict]) -> float:
    ends = [float(segment["end"]) for segment in segments if float(segment.get("end") or 0) > 0]
    return max(ends) if ends else 0.0


def _audio_duration_sec(path: Path, fallback: float = 0.0) -> float:
    if not path.exists():
        return fallback
    try:
        info = torchaudio.info(str(path))
    except Exception:  # noqa: BLE001
        return fallback
    if info.sample_rate <= 0:
        return fallback
    return float(info.num_frames) / float(info.sample_rate)


def _write_single_label_files(temp_output_dir: Path, output_dir: Path, speaker_a: Path, speaker_b: Path) -> dict[str, str]:
    labels_dir = temp_output_dir / "phase_02_diarization" / "labels"
    vad_path = output_dir / "vad.txt"
    diarization_path = output_dir / "diarization.txt"
    separation_path = output_dir / "separation.txt"

    if (labels_dir / "vad.txt").exists():
        vad_path.write_text((labels_dir / "vad.txt").read_text(encoding="utf-8"), encoding="utf-8")
    else:
        vad_path.write_text("", encoding="utf-8")

    if (labels_dir / "speakers.txt").exists():
        diarization_path.write_text((labels_dir / "speakers.txt").read_text(encoding="utf-8"), encoding="utf-8")
    else:
        diarization_path.write_text("", encoding="utf-8")

    segments = _diarization_segments(temp_output_dir)
    fallback_duration = _fallback_duration_sec(segments)
    duration = max(_audio_duration_sec(speaker_a, fallback_duration), _audio_duration_sec(speaker_b, fallback_duration))
    write_label_file(
        separation_path,
        [
            (0.0, duration, "speaker_A"),
            (0.0, duration, "speaker_B"),
        ],
    )
    return {
        "vad": str(vad_path),
        "diarization": str(diarization_path),
        "separation": str(separation_path),
    }


def main() -> None:
    suppress_pyannote_tf32_warning()
    parser = argparse.ArgumentParser(description="Debug DuplexChat on one local audio sample.")
    parser.add_argument("--input", required=True, help="Input audio path.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--diarize-chunk", type=float, default=60.0)
    parser.add_argument("--separate-chunk", type=float, default=30.0)
    parser.add_argument("--diarization-backend", default=DEFAULT_DIARIZATION_BACKEND)
    parser.add_argument("--diarization-model", default=DEFAULT_DIARIZATION_MODEL)
    parser.add_argument("--separation-backend", default=DEFAULT_SEPARATION_BACKEND)
    parser.add_argument("--separation-model", default=DEFAULT_SEPARATION_MODEL)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = args.output_dir or Path("outputs") / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / "speaker"

    with tempfile.TemporaryDirectory(prefix="duplexchat_single_") as temp_dir:
        temp_output_dir = Path(temp_dir)
        run_single_audio(
            args.input,
            diarize_chunk=args.diarize_chunk,
            separate_chunk=args.separate_chunk,
            diarization_backend=args.diarization_backend,
            diarization_model=args.diarization_model,
            separation_backend=args.separation_backend,
            separation_model=args.separation_model,
            output_prefix=str(output_prefix),
            output_dir=str(temp_output_dir),
        )

        speaker_a = output_dir / "speaker_A.wav"
        speaker_b = output_dir / "speaker_B.wav"
        label_files = _write_single_label_files(temp_output_dir, output_dir, speaker_a, speaker_b)
        run_manifest = {
            "input": args.input,
            "output_dir": str(output_dir),
            "speaker_A": str(speaker_a),
            "speaker_B": str(speaker_b),
            "labels": label_files,
            "models": {
                "diarization": {
                    "backend": args.diarization_backend,
                    "model": args.diarization_model,
                    "chunk_seconds": args.diarize_chunk,
                },
                "separation": {
                    "backend": args.separation_backend,
                    "model": args.separation_model or "default",
                    "chunk_seconds": args.separate_chunk,
                },
            },
            "phase_results": _phase_text(temp_output_dir, speaker_a, speaker_b),
        }
        (output_dir / "run.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
