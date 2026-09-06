# DuplexChat

DuplexChat là pipeline tạo dữ liệu hội thoại full-duplex hai người nói từ audio
thực tế. Pipeline thu thập nguồn audio, chuẩn hóa âm thanh, diarization để biết
ai nói lúc nào, tách các đoạn hội thoại hai speaker, chạy speech separation, rồi
ghi dữ liệu đầu ra kèm artifact để kiểm tra.

Các cấu hình nằm trong `configs/`: source, language metadata, diarization model,
separation model, benchmark model và runtime. Output chính của pipeline nằm ở
`data/`, `outputs/`, `reports/`, và log theo từng lần chạy nằm trong `logs/`.
Khi chạy CLI, từng phase có progress bar để theo dõi tiến trình.

## So sánh Cholimex, DuplexChat và Vilier

Repo chứa cả ba pipeline. Các branch `cholimex`, `duplexchat`, `vilier` cùng có
runner chung; `--pipeline` quyết định implementation được chạy, không phụ thuộc
branch đang checkout. Branch `master` cũ được giữ làm mốc lịch sử.

```bash
MPLBACKEND=Agg uv run python end2end.py --pipeline cholimex --data oto-speech --size_gb 1
MPLBACKEND=Agg uv run python end2end.py --pipeline duplexchat --data oto-speech --size_gb 1
MPLBACKEND=Agg uv run python end2end.py --pipeline vilier --data oto-speech --size_gb 1
MPLBACKEND=Agg uv run python end2end.py --pipeline all --data oto-speech --size_gb 1
```

Trong notebook, thêm `!` trước lệnh. `--pipeline` hiện chỉ áp dụng cho OtoSpeech;
không truyền option này khi dùng luồng crawl. OtoSpeech mặc định dùng Cholimex.
`--size_gb` giới hạn dung lượng subset tải từ dataset, không bao gồm model và
môi trường Python. Download và tạo mixture chỉ diễn ra một lần cho mỗi lượt;
`all` chạy tuần tự Cholimex → DuplexChat → Vilier trên cùng các mixture.

Chạy smoke nhỏ trước:

```bash
MPLBACKEND=Agg uv run python end2end.py --pipeline all --data oto-speech \
  --size_gb 1 --max-samples 1 --max-seconds 10
```

`--max-seconds` cắt mixture và giới hạn vùng chấm điểm; hai file GT vẫn giữ
nguyên bản. Dùng `--otospeech-root /path/to/snapshot` để dùng dataset có sẵn,
không tải lại. Pipeline cần FFmpeg và quyền truy cập các model đã cấu hình.

Mỗi pipeline có project, lockfile và `.venv` riêng trong `pipelines/`; runner
sử dụng `uv run --locked`, tự đồng bộ môi trường khi cần. Vilier còn có worker
ClearVoice riêng: ClearVoice yêu cầu NumPy 1.x, pyannote Community 4 yêu cầu
NumPy 2.x. Worker này giữ model trong bộ nhớ xuyên suốt các vùng overlap của
một sample và đóng khi sample hoàn tất. Không dùng chung dependency hai stage.
Cholimex/DuplexChat pin Torch và TorchAudio 2.8 vì checkpoint DialogueSidon
được export bằng Torch 2.8; TorchCodec 0.7 theo
[bảng tương thích chính thức](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions).
Cách uv đồng bộ project/lockfile được mô tả trong
[tài liệu uv](https://docs.astral.sh/uv/guides/projects/).

Config dùng `--config` cho Cholimex, `--duplexchat-config` (mặc định
`configs/duplexchat.json`) và `--vilier-config` (mặc định `configs/vilier.json`).
Vilier giữ model hiện có, chạy đến xuất speaker tracks; profile so sánh luôn
tắt ASR/Qwen và dry-run. Device `auto` chọn CUDA nếu có, sau đó CPU.
Đường dẫn checkpoint tùy chỉnh nên là đường dẫn tuyệt đối; không đưa trọng số
vào Git. Dependency profile bổ sung vẫn nằm trong `pipelines/vilier/requirements/`.
Nếu đổi backend, cập nhật project dependency tương ứng và chạy `uv lock --project`
cho project đó; lockfile mặc định phục vụ model mặc định.

```text
outputs/otospeech_mixtures/<key>/mixture.wav
outputs/otospeech_mixtures/<key>/gt_speaker_1.wav
outputs/otospeech_mixtures/<key>/gt_speaker_2.wav
outputs/otospeech/<pipeline>/<key>/speaker_A.wav
outputs/otospeech/<pipeline>/<key>/speaker_B.wav
outputs/otospeech/<pipeline>/<key>/run.json
outputs/runs/<run id>/samples.json
outputs/runs/<run id>/<pipeline>/worker.log
reports/<pipeline>/summary.json
reports/<pipeline>/summary.md
reports/<pipeline>/sample_metrics.jsonl
reports/<pipeline>/runtime.json
reports/comparison.json
reports/comparison.md
```

`--pred-root` thay đổi root phía trước `<pipeline>`; `--benchmark-output
/path/summary.json` tạo report từng pipeline tại `/path/<pipeline>/summary.json`
và bảng tổng tại `/path/comparison.*`. `--output-root` đổi root audio/log,
report vẫn theo config benchmark hoặc `--benchmark-output`.

Terminal in bảng sau từng pipeline và bảng tổng. Bảng tổng chấm trên giao các
sample hợp lệ ở mọi pipeline; `N/A (n=0)` nghĩa là không có giá trị metric hợp lệ.
Cột ESTOI dùng `pystoi` với `extended=True`, PESQ dùng thư viện `pesq`; khi chưa
cài dependency tùy chọn, metric đó hiện `N/A`. Runtime/RTF chỉ tính inference
thành công mới chạy trong lượt này, gồm model loading; không tính download,
uv sync, benchmark hoặc thời gian inference từ lượt được resume.

Resume yêu cầu manifest hoàn tất, khớp hash input/config/source và cả hai WAV
nguyên vẹn, đúng độ dài. `--force` chạy lại. Kết quả bị thay thế được chuyển vào
`.history/` cạnh thư mục sample. Output cũ không có manifest hợp lệ không được
mặc nhiên dùng để so sánh. Lỗi sample hoặc worker được lưu và các phần còn lại
vẫn chạy; cuối lượt có report và exit code khác 0. Vilier có số speaker khác hai
sẽ được báo lỗi, không chọn speaker dựa trên GT.

Kiểm tra các môi trường và test độc lập:

```bash
uv lock --project pipelines/cholimex --check
uv lock --project pipelines/duplexchat --check
uv lock --project pipelines/vilier --check
uv lock --project pipelines/vilier/clearvoice-runtime --check
uv run --extra dev python -m pytest
(cd pipelines/vilier && uv run python -m pytest tests)
```

Nguồn mã và thay đổi khi nhập được ghi trong [SOURCES.md](SOURCES.md).

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
