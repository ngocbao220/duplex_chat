from __future__ import annotations

import argparse
import json
from pathlib import Path

from test_single import run_single_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug DuplexChat on one local audio sample.")
    parser.add_argument("--input", required=True, help="Input audio path.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--diarize-chunk", type=float, default=60.0)
    parser.add_argument("--separate-chunk", type=float, default=30.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = args.output_dir or Path("outputs") / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / "speaker"

    run_single_audio(
        args.input,
        diarize_chunk=args.diarize_chunk,
        separate_chunk=args.separate_chunk,
        output_prefix=str(output_prefix),
        output_dir=str(output_dir),
    )

    speaker_a = output_dir / "speaker_A.wav"
    speaker_b = output_dir / "speaker_B.wav"

    run_manifest = {
        "input": args.input,
        "output_dir": str(output_dir),
        "speaker_A": str(speaker_a),
        "speaker_B": str(speaker_b),
        "phases": {
            "input": str(output_dir / "phase_00_input" / "input.json"),
            "preprocess": str(output_dir / "phase_01_preprocess" / "audio_16k_mono.wav"),
            "diarization": str(output_dir / "phase_02_diarization" / "diarization.json"),
            "separation": str(output_dir / "phase_04_separation" / "separation.json"),
        },
    }
    (output_dir / "run.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
