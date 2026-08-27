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

def test_single_audio(audio_path_str):
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        print(f"Error: File '{audio_path}' does not exist.")
        sys.exit(1)

    print(f"--- Processing {audio_path.name} ---")
    
    temp_wav = Path("temp_test_audio.wav")
    print("[0/4] Converting audio to 16kHz mono WAV...")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ar", "16000", "-ac", "1", str(temp_wav)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[1/4] Loading Models...")
    # Khởi tạo pipeline nhận diện giọng nói (Diarization)
    diarize_pipeline = load_diarization_pipeline("pyannote/speaker-diarization-community-1", device=device)
    
    # Khởi tạo mô hình tách âm (Separation)
    sep_models = load_separation_models(device=device, backend="dialoguesidon")
    
    print("[2/4] Running Diarization...")
    segments = run_diarization(diarize_pipeline, temp_wav)
    print(f"Found {len(segments)} diarization segments.")
    for seg in segments[:5]:
        print(f"  {seg['speaker']}: {seg['start']:.2f}s - {seg['end']:.2f}s")
    if len(segments) > 5:
        print("  ...")
    
    print("[3/4] Running Separation...")
    wav, sr = load_wav_tensor(temp_wav)
    spk0, spk1, out_sr = run_separation(wav, sr, num_steps=30, models=sep_models)
    
    print("[4/4] Saving output...")
    # Gộp 2 kênh thành file stereo (Left: Người A, Right: Người B)
    stereo = torch.cat([spk0, spk1], dim=0)
    output_path = "output_test_separated.wav"
    torchaudio.save(output_path, stereo, out_sr)
    print(f"Done! Saved separated stereo file to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_single_audio(sys.argv[1])
    else:
        print("Usage: uv run python test_single.py <path_to_audio_file>")
