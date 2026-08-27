import sys
import torch
import torchaudio
import subprocess
from pathlib import Path

from duplexchat_pipe.audio import load_wav_tensor
from duplexchat_pipe.diarize import load_diarization_pipeline, run_diarization
from duplexchat_pipe.separate import load_separation_models, run_separation

# Tự động tìm thiết bị (dùng GPU nếu có, ngược lại dùng CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chỉ patch torch.load ép về CPU nếu máy thực sự không có GPU (như Mac)
if device == "cpu":
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

import argparse

def test_single_audio(audio_path_str, diarize_chunk=60.0, separate_chunk=30.0):
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        print(f"Error: File '{audio_path}' does not exist.")
        sys.exit(1)

    print(f"--- Processing {audio_path.name} ---")
    print(f"Config: Diarize Chunk={diarize_chunk}s, Separate Chunk={separate_chunk}s")
    
    temp_wav = Path("temp_test_audio.wav")
    print("[0/4] Converting audio to 16kHz mono WAV...")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ar", "16000", "-ac", "1", str(temp_wav)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[1/4] Loading Models...")
    diarize_pipeline = load_diarization_pipeline("pyannote/speaker-diarization-community-1", device=device)
    sep_models = load_separation_models(device=device, backend="dialoguesidon")
    
    print("[2/4] Running Diarization...")
    segments = run_diarization(diarize_pipeline, temp_wav, max_chunk_dur=diarize_chunk)
    print(f"Found {len(segments)} diarization segments.")
    for seg in segments[:5]:
        print(f"  {seg['speaker']}: {seg['start']:.2f}s - {seg['end']:.2f}s")
    if len(segments) > 5:
        print("  ...")
    
    print("[3/4] Running Separation...")
    if device == "cuda":
        diarize_pipeline.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        
    wav, sr = load_wav_tensor(temp_wav)
    # Tự động tính overlap_seconds bằng 1/6 của separate_chunk (vd 30s -> 5s)
    overlap = max(1.0, separate_chunk / 6.0)
    spk0, spk1, out_sr = run_separation(wav, sr, num_steps=30, models=sep_models, chunk_seconds=separate_chunk, overlap_seconds=overlap)
    
    print("[4/4] Saving output...")
    out_A = "output_speaker_A.wav"
    out_B = "output_speaker_B.wav"
    torchaudio.save(out_A, spk0, out_sr)
    torchaudio.save(out_B, spk1, out_sr)
    
    print(f"Done! Saved to:")
    print(f" - {out_A} (Người A)")
    print(f" - {out_B} (Người B)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DuplexChat on a single audio file.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file (mp3/wav)")
    parser.add_argument("--diarize-chunk", type=float, default=60.0, help="Max chunk duration (seconds) for Diarization")
    parser.add_argument("--separate-chunk", type=float, default=30.0, help="Chunk duration (seconds) for Separation")
    
    args = parser.parse_args()
    test_single_audio(args.audio_path, args.diarize_chunk, args.separate_chunk)
