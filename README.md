# DuplexChat

DuplexChat là pipeline tạo dữ liệu hội thoại full-duplex hai người nói từ audio
thực tế. Pipeline thu thập nguồn audio, chuẩn hóa âm thanh, diarization để biết
ai nói lúc nào, tách các đoạn hội thoại hai speaker, chạy speech separation, rồi
ghi dữ liệu đầu ra kèm artifact để kiểm tra.

Các cấu hình nằm trong `configs/`: source, language metadata, diarization model,
separation model, benchmark model và runtime. Output chính của pipeline nằm ở
`data/`, `outputs/`, `reports/`, và log theo từng lần chạy nằm trong `logs/`.
Khi chạy CLI, từng phase có progress bar để theo dõi tiến trình.

## Lệnh chạy

1. Chạy toàn bộ pipeline để lấy 10 giờ data:

```bash
uv run python end2end.py --target_hours 10
```

Chỉ test các video/playlist trong `configs/youtube_allowlist.json`:

```bash
uv run --with yt-dlp python end2end.py --youtube-only --target_hours 1
```

Với `end2end.py --data oto-speech`, mỗi sample lưu ba file để nghe đối chiếu:

```text
outputs/otospeech_mixtures/<sample key>/mixture.wav
outputs/otospeech_mixtures/<sample key>/gt_speaker_1.wav
outputs/otospeech_mixtures/<sample key>/gt_speaker_2.wav
```

Hai file `gt_speaker_*` là bản sao nguyên gốc từ dataset, giữ nguyên sample rate
và độ dài. `mixture.wav` dùng `--sample-rate` (mặc định 16000 Hz) và độ dài
chung của hai nguồn. `--output-root` hoặc `--mixture-root` thay đổi nơi lưu.
Chạy lại cũng bổ sung hai file gốc khi prediction đã có và được bỏ qua.

2. Chạy 1 sample để debug pipeline:

```bash
uv run python single.py --input inputs/sample.wav \
  --diarize-chunk 90 \
  --separate-chunk 90
```

Output mặc định:

```text
outputs/<tên input>/run.json
outputs/<tên input>/vad.txt
outputs/<tên input>/diarization.txt
outputs/<tên input>/separation.txt
outputs/<tên input>/speaker_A.wav
outputs/<tên input>/speaker_B.wav
```

`<tên input>` là tên file cuối cùng, bỏ phần mở rộng. Ví dụ
`kaggle/inputs/adasdasd/demo1.wav` sẽ ghi vào `outputs/demo1/`.

`run.json` ghi model sử dụng và kết quả từng phase dạng text. Các file TXT dùng
format Audacity `start<TAB>end<TAB>label`. Hai file WAV là audio đích theo từng
speaker.

Pipeline thử nghiệm Cholimex có README riêng tại `README_CHOLIMEX.md`. Flow này
chạy giống `single.py` ở bước đầu để lấy hai provisional DialogueSidon tracks,
sau đó dùng VAD masks và chỉ separation vùng overlap để giữ fidelity audio gốc.

3. Chạy benchmark sau khi đã có hai audio speaker:

```bash
uv run python src/benchmark.py --single \
  --speakerA outputs/<tên input>/speaker_A.wav \
  --speakerB outputs/<tên input>/speaker_B.wav \
  --output outputs/<tên input>/benchmark.json
```

Benchmark tạo:

```text
outputs/<tên input>/benchmark.json
outputs/<tên input>/benchmark.metrics.md
outputs/<tên input>/benchmark.turn_taking.md
```

DNSMOS đang tắt mặc định. Nếu muốn bật DNSMOS, truyền metric và model ONNX:

```bash
uv run python src/benchmark.py --single \
  --speakerA outputs/<tên input>/speaker_A.wav \
  --speakerB outputs/<tên input>/speaker_B.wav \
  --output outputs/<tên input>/benchmark.json \
  --metrics dnsmos sq_stoi sq_pesq sq_si_sdr itc itd \
  --dnsmos-model DNSMOS/sig_bak_ovr.onnx
```

4. Chạy test:

```bash
uv run python -m pytest
```
