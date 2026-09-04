# Cholimex Pipeline

Cholimex là pipeline single-audio kết hợp cách tách hai track của DuplexChat với
cơ chế giữ audio gốc cho vùng non-overlap của Sommelier.

Mục tiêu chính:

```text
DialogueSidon recall cho speech ngắn/nhỏ
+
Sommelier fidelity cho vùng chỉ một speaker nói
```

Nguyên tắc quan trọng:

```text
Use separated audio only when separation is necessary.
Otherwise preserve the original waveform.
```

## Flow

```text
original audio
  -> 16 kHz mono original.wav
  -> DialogueSidon provisional separation
  -> sidon_track_0.wav + sidon_track_1.wav
  -> Silero VAD trên từng Sidon track
  -> vad_track_0 / vad_track_1 activity masks
  -> region classification
  -> overlap-only separation trên original audio
  -> cosine speaker assignment
  -> speaker_0.wav + speaker_1.wav
```

DialogueSidon output chỉ là proposal để tìm activity/backchannel. Final
single-speaker audio được lấy trực tiếp từ original waveform.

## Quick Start

Chạy trên một file audio:

```bash
UV_CACHE_DIR=.uv-cache uv run duplexchat-pipe cholimex \
  --input path/to/input.wav \
  --output-dir outputs/cholimex/input_name \
  cholimex.overlap_padding=0.1
```

Ví dụ với sample trong sibling Vilier checkout:

```bash
UV_CACHE_DIR=.uv-cache uv run duplexchat-pipe cholimex \
  --input ../vilier/inputs/samples/easy_1.wav \
  --output-dir outputs/cholimex/easy_1 \
  cholimex.overlap_padding=0.1
```

Ép CPU:

```bash
UV_CACHE_DIR=.uv-cache uv run duplexchat-pipe cholimex \
  --input path/to/input.wav \
  --output-dir outputs/cholimex/input_name \
  --runtime-device cpu
```

Trên Kaggle, dùng path tuyệt đối hoặc path đúng trong `/kaggle/working`:

```bash
UV_CACHE_DIR=.uv-cache uv run duplexchat-pipe cholimex \
  --input /kaggle/working/duplex_chat/easy_1.wav \
  --output-dir outputs/cholimex/easy_1 \
  cholimex.overlap_padding=0.1
```

## Outputs

Output chính:

```text
outputs/cholimex/<name>/
├── speaker_0.wav
├── speaker_1.wav
├── run.json
└── debug/
```

Debug artifacts:

```text
debug/
├── original.wav
├── sidon_track_0.wav
├── sidon_track_1.wav
├── vad_track_0.json
├── vad_track_1.json
├── vad_track_0.txt
├── vad_track_1.txt
├── regions.json
├── overlap_regions.json
├── speaker_reference_0.wav
├── speaker_reference_1.wav
├── final_track_0.wav
└── final_track_1.wav
```

Các file `.txt` dùng format Audacity:

```text
start<TAB>end<TAB>1
```

Cholimex chỉ ghi label active `1`, không ghi vùng silence `0`.

## Region Types

`regions.json` là artifact quan trọng nhất để inspect logic masking:

```json
[
  {
    "start": 1.2,
    "end": 4.35,
    "type": "single_speaker",
    "speaker": 0
  },
  {
    "start": 4.35,
    "end": 4.72,
    "type": "overlap_backchannel",
    "backchannel_speaker": 1
  }
]
```

Các loại region:

- `silence`: cả hai Sidon VAD masks đều inactive.
- `single_speaker`: chỉ một Sidon VAD mask active; final audio lấy từ original.
- `overlap`: cả hai masks active đủ dài; chạy overlap separator.
- `overlap_backchannel`: cả hai masks active nhưng một activity ngắn hơn
  `cholimex.backchannel_max_duration`; vẫn được giữ và chạy overlap separator.

## Config

Các threshold nằm trong `configs/config.json` dưới key `cholimex` và có thể
override bằng dotted `key=value`:

```json
{
  "cholimex": {
    "backchannel_max_duration": 1.0,
    "min_vad_duration": 0.0,
    "vad_onset": null,
    "vad_offset": null,
    "merge_gap": 0.0,
    "min_reference_duration": 2.0,
    "cosine_similarity_threshold": 0.5,
    "overlap_padding": 0.1,
    "proposal_backend": "dialoguesidon",
    "proposal_model": "sarulab-speech/DialogueSidon",
    "overlap_separator_backend": "dialoguesidon",
    "overlap_separator_model": "sarulab-speech/DialogueSidon",
    "speaker_embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
    "output_stereo": false
  }
}
```

Ví dụ override:

```bash
UV_CACHE_DIR=.uv-cache uv run duplexchat-pipe cholimex \
  --input input.wav \
  --output-dir outputs/cholimex/input \
  cholimex.backchannel_max_duration=0.8 \
  cholimex.overlap_padding=0.2 \
  cholimex.output_stereo=true
```

Nếu `cholimex.output_stereo=true`, pipeline ghi thêm:

```text
stereo.wav
```

với left = `speaker_0`, right = `speaker_1`.

## Architecture

Implementation nằm trong `src/cholimex/`:

- `pipeline.py`: orchestration single-audio end to end.
- `vad_masking.py`: Silero VAD trên provisional Sidon tracks và ghi Audacity labels.
- `region_classifier.py`: phân loại timeline từ hai masks.
- `reconstruction.py`: reconstruct final tracks, chỉ gọi separator cho overlap.
- `speaker_assignment.py`: SpeechBrain embeddings và cosine matching.
- `models.py`: dataclass cho activity segments và regions.

Reuse từ DuplexChat:

- `audio.transcode_to_wav_16k_mono`
- `audio.load_wav_tensor`
- `outputs.save_wav`
- `outputs.write_json`
- `separate.load_separation_models`
- `separate.run_separation`
- `Config` và dotted overrides

Reuse ý tưởng từ Sommelier:

- original audio cho vùng non-overlap;
- overlap-only separation;
- reference speaker embeddings từ vùng non-overlap đủ dài;
- cosine similarity để xử lý permutation của separator output.

## Failure Modes

- Input path sai: `ffmpeg` sẽ fail khi không mở được file. Kiểm tra bằng
  `ls -l path/to/input.wav`.
- Model/token chưa sẵn sàng: DialogueSidon hoặc SpeechBrain có thể cần tải model
  từ Hugging Face.
- Không có reference non-overlap đủ dài: speaker assignment fallback về thứ tự
  channel separator.
- Sidon output lệch duration quá lớn: pipeline sẽ fail nếu không thể align về
  timeline original.
- CPU run có thể rất chậm; nên dùng GPU trên Kaggle/Colab cho real smoke.

## Tests

Chạy focused tests:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra dev python -m pytest \
  tests/test_cholimex.py tests/test_separation_backends.py
```

Chạy full suite:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra dev python -m pytest
```

Các tests hiện tại dùng fake tensors/fake separator/fake embedder cho logic
Cholimex. Chúng không chứng minh real DialogueSidon/Silero/SpeechBrain inference
đã chạy thành công.
