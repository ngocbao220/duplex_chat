# DuplexChat Pipeline Architecture

Tài liệu này mô tả chi tiết các giai đoạn (phases) trong luồng xử lý thu thập, phân tích và trích xuất tập dữ liệu hội thoại song công (Full-duplex spoken-dialogue) từ các podcast nguồn mở. Pipeline được chia thành 6 giai đoạn nối tiếp nhau (tương ứng với các kịch bản trong `jobs/vi/`).

---

## Phase 1: Collect Sources (`01_collect_sources.sh`)
- **Mô tả:** Tải và truy vấn cơ sở dữ liệu RSS của podcast (từ podcastindex.org) để tìm kiếm các chương trình podcast phù hợp với tiêu chí ngôn ngữ (ví dụ: tiếng Việt `vi`, `vi-vn`). Kết hợp đối chiếu với các allowlist/blacklist để lọc bớt nguồn rác.
- **Input:** 
  - Cơ sở dữ liệu Podcast Index (`podcastindex_feeds.db.tgz`).
  - Các cấu hình lọc nguồn (`feed_allowlist`, `youtube_allowlist`, v.v.).
- **Output:** 
  - Một danh sách (manifest/queue) chứa thông tin metadata và URL MP3 gốc của các tập podcast (episodes) hợp lệ sẵn sàng để tải.

## Phase 2: Download & Clean (`02_download_clean.sh`)
- **Mô tả:** Sử dụng hệ thống tải song song để kéo các file âm thanh gốc về máy, sau đó xử lý bằng `ffmpeg` để chuẩn hoá cấu hình file âm thanh (định dạng chung, sample rate, số kênh).
- **Input:** 
  - Các URL âm thanh thô (MP3, M4A, WAV...) thu thập được từ Phase 1.
- **Output:** 
  - Các tệp âm thanh nguyên bản (whole-episode) đã được chuẩn hoá xuống **16 kHz mono** (hoặc format phù hợp với model) cùng với thời lượng chuẩn xác của file thu được qua `ffprobe`.

## Phase 3: Diarize & Segment (`03_diarize_segment.sh`)
- **Mô tả:** Chạy mô hình phân đoạn người nói (Speaker Diarization - mặc định sử dụng `pyannote/speaker-diarization-community-1`) trên các episode dài. Thuật toán sẽ quét qua toàn bộ thời lượng, gán nhãn người nói và cắt tỉa để trích xuất ra **những phân đoạn hội thoại chỉ có đúng 2 người**. Các quy tắc cắt gọt (ví dụ: hội thoại phải dài từ 10s đến 10 phút, không có khoảng lặng quá 5s) được áp dụng tại bước này.
- **Input:** 
  - File âm thanh chuẩn hoá (16 kHz mono).
- **Output:** 
  - Metadata `diarization.json` (chứa timestamps nhãn giọng nói).
  - Metadata chứa các toạ độ thời gian (timings) cắt nhỏ các đoạn hội thoại (dialogues) hợp lệ từ tệp gốc.

## Phase 4: Separate Speakers (`04_separate.sh`)
- **Mô tả:** Đưa các đoạn hội thoại 2 người vừa trích xuất vào mô hình tách nguồn âm thanh (Source Separation - ví dụ: `DialogueSidon` hoặc `MossFormer2`). Mô hình sẽ bóc tách giọng nói bị chồng chéo (overlap) và tách riêng mỗi người ra một kênh âm thanh độc lập. Cuối cùng, hệ thống đóng gói các file này vào WebDataset (định dạng `.tar`).
- **Input:** 
  - Các toạ độ đoạn hội thoại cắt được từ Phase 3.
- **Output:** 
  - Tập dữ liệu thô **WebDataset** chứa hàng loạt các mẫu (samples). Mỗi mẫu (sample) bao gồm:
    - `audio.mp3`: Tệp âm thanh **Stereo** (Kênh Trái = Người A, Kênh Phải = Người B).
    - `meta.json`: Các thông tin gốc của đoạn hội thoại, chỉ mục tập podcast và toạ độ thời gian.

## Phase 5: Filter (`05_filter.sh`)
- **Mô tả:** Lọc hậu kỳ dữ liệu và loại bỏ trùng lặp. Những đoạn hội thoại có dấu hiệu chứa nhạc nền (music-genre feeds), hoặc thời lượng đoạn cắt chiếm tỷ lệ quá lớn so với tập gốc (thường là lỗi mô hình trên các nội dung độc thoại) sẽ bị loại bỏ.
- **Input:** 
  - Dữ liệu WebDataset thô từ Phase 4 (vd: `data/wds_vi_poc`).
- **Output:** 
  - Các tệp WebDataset đã được lọc sạch (vd: `data/wds_vi_poc_filtered`).
  - Báo cáo thống kê số lượng dữ liệu bị loại `stats.json`.

## Phase 6: Benchmark (`06_benchmark.sh`)
- **Mô tả:** Tổng hợp, đánh giá và lập thống kê chuyên sâu về khối lượng dữ liệu đã thu thập, hiệu suất chạy của pipeline (thời gian xử lý, tài nguyên sử dụng GPU/CPU). 
- **Input:** 
  - Toàn bộ log xử lý và cache SQLite từ các bước trước.
- **Output:** 
  - Báo cáo `stats_table.md`, log ghi nhận tốc độ xử lý trên mỗi giờ audio.
