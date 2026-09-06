import sys
import torch
import torchaudio
import subprocess
from pathlib import Path

from tqdm import tqdm

from duplexchat_pipe.audio import load_wav_tensor
from duplexchat_pipe.diarize import load_diarization_pipeline, run_diarization
from duplexchat_pipe.outputs import (
    copy_file,
    save_wav,
    write_diarization_phase,
    write_json,
)
from duplexchat_pipe.separate import load_separation_models, run_separation

import argparse


def release_diarization_gpu_memory(diarize_pipeline):
    if hasattr(diarize_pipeline, "to"):
        diarize_pipeline.to(torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_output_dir(output_prefix: str, output_dir: str | None = None) -> Path:
    if output_dir:
        return Path(output_dir)
    parent = Path(output_prefix).parent
    return parent if str(parent) != "." else Path("outputs") / "single_audio"


def make_chunk_progress(desc: str, unit: str):
    pbar = None

    def progress_callback(event: str, value: int) -> None:
        nonlocal pbar
        if event == "start":
            pbar = tqdm(total=value, desc=desc, unit=unit, leave=False)
        elif event == "advance" and pbar is not None:
            pbar.update(value)
        elif event == "close" and pbar is not None:
            pbar.close()
            pbar = None

    return progress_callback


def run_single_audio(
    audio_path_str,
    diarize_chunk=60.0,
    separate_chunk=30.0,
    diarization_backend="auto",
    diarization_model="pyannote/speaker-diarization-community-1",
    separation_backend="dialoguesidon",
    separation_model=None,
    output_prefix="output_speaker",
    output_dir=None,
    runtime_device="auto",
    num_steps=30,
):
    from duplexchat_pipe.devices import resolve_device
    device = resolve_device(runtime_device, allow_cpu_fallback=True)
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    print(f"========= Phase 1: Preparing input ({audio_path.name}) =========", flush=True)
    print(f"Config: Diarize Chunk={diarize_chunk}s, Separate Chunk={separate_chunk}s")
    print(f"Diarization: backend={diarization_backend}, model={diarization_model}")
    print(f"Separation: backend={separation_backend}, model={separation_model or 'default'}")

    phase_output_dir = resolve_output_dir(output_prefix, output_dir)
    phase_output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        phase_output_dir / "phase_00_input" / "input.json",
        {"audio_path": str(audio_path), "audio_name": audio_path.name},
    )

    temp_wav = phase_output_dir / "phase_01_preprocess" / "audio_16k_mono.wav"
    temp_wav.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=2, desc=f"{audio_path.stem} / preprocess", unit="step", leave=False) as pbar:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1", str(temp_wav)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pbar.update(1)
        # The standardized input is already persisted in the phase directory.
        pbar.update(1)

    print("========= Phase 2: Loading models =========", flush=True)
    with tqdm(total=2, desc=f"{audio_path.stem} / load models", unit="model", leave=False) as pbar:
        diarize_pipeline = load_diarization_pipeline(
            diarization_model,
            device=device,
            backend=diarization_backend,
        )
        pbar.update(1)
        sep_models = load_separation_models(
            device=device,
            backend=separation_backend,
            model_id=separation_model,
        )
        pbar.update(1)

    print("========= Phase 3: Running Pipeline: Diarization =========", flush=True)
    diarization_progress = make_chunk_progress(f"{audio_path.stem} / diarization", "chunk")
    try:
        segments = run_diarization(
            diarize_pipeline,
            temp_wav,
            max_chunk_dur=diarize_chunk,
            progress_callback=diarization_progress,
        )
    finally:
        diarization_progress("close", 0)
    with tqdm(total=1, desc="write diarization", unit="file", leave=False) as pbar:
        write_diarization_phase(
            phase_output_dir,
            segments,
            model=diarization_model,
            backend=diarization_backend,
        )
        pbar.update(1)

    print(f"Found {len(segments)} diarization segments.")
    for seg in segments[:5]:
        print(f"  {seg['speaker']}: {seg['start']:.2f}s - {seg['end']:.2f}s")
    if len(segments) > 5:
        print("  ...")

    if str(device).startswith("cuda"):
        release_diarization_gpu_memory(diarize_pipeline)

    wav, sr = load_wav_tensor(temp_wav)
    overlap = max(1.0, separate_chunk / 6.0)
    print("========= Phase 4: Running Pipeline: Separation =========", flush=True)
    separation_progress = make_chunk_progress(f"{audio_path.stem} / separation", "chunk")
    try:
        spk0, spk1, out_sr = run_separation(
            wav,
            sr,
            num_steps=num_steps,
            models=sep_models,
            chunk_seconds=separate_chunk,
            overlap_seconds=overlap,
            progress_callback=separation_progress,
        )
    finally:
        separation_progress("close", 0)

    print("========= Phase 5: Writing outputs =========", flush=True)
    with tqdm(total=5, desc=f"{audio_path.stem} / save outputs", unit="file", leave=False) as pbar:
        output_path = Path(output_prefix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_A = str(output_path.parent / f"{output_path.name}A.wav")
        out_B = str(output_path.parent / f"{output_path.name}B.wav")
        save_wav(Path(out_A), spk0, out_sr)
        pbar.update(1)
        save_wav(Path(out_B), spk1, out_sr)
        pbar.update(1)
        save_wav(phase_output_dir / "phase_04_separation" / "speaker_A.wav", spk0, out_sr)
        pbar.update(1)
        save_wav(phase_output_dir / "phase_04_separation" / "speaker_B.wav", spk1, out_sr)
        pbar.update(1)
        write_json(
            phase_output_dir / "phase_04_separation" / "separation.json",
            {
                "backend": separation_backend,
                "model": separation_model,
                "sample_rate": out_sr,
                "speaker_A": out_A,
                "speaker_B": out_B,
            },
        )
        pbar.update(1)
    
    print(f"Done! Saved to:")
    print(f" - {out_A} (Người A)")
    print(f" - {out_B} (Người B)")

def main():
    parser = argparse.ArgumentParser(description="Test DuplexChat on a single audio file.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file (mp3/wav)")
    parser.add_argument("--diarize-chunk", type=float, default=60.0, help="Max chunk duration (seconds) for Diarization")
    parser.add_argument("--separate-chunk", type=float, default=30.0, help="Chunk duration (seconds) for Separation")
    parser.add_argument("--diarization-backend", default="auto", help="Diarization backend: auto, pyannote, sortformer, diarizen")
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-community-1", help="Diarization model id or alias")
    parser.add_argument("--separation-backend", default="dialoguesidon", help="Separation backend: dialoguesidon, sepformer, mossformer2")
    parser.add_argument("--separation-model", default=None, help="Separation model id or alias")
    parser.add_argument("--output-prefix", default="output_speaker", help="Output WAV prefix, e.g. runs/sortformer__sepformer/output")
    parser.add_argument("--output-dir", default=None, help="Directory for phase outputs and labels")
    
    args = parser.parse_args()
    run_single_audio(
        args.audio_path,
        args.diarize_chunk,
        args.separate_chunk,
        args.diarization_backend,
        args.diarization_model,
        args.separation_backend,
        args.separation_model,
        args.output_prefix,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
