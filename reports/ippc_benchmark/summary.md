# IPPC Reference Benchmark

| Condition | Mục tiêu | Metric chính | Ý nghĩa | Mean | Median | P95 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| all | Chất lượng audio | PIT-SI-SDR | Prediction giống ground truth đến mức nào |  |  |  |
| all | Artifact/rè/nhiễu | SAR | Mức méo và artifact do pipeline tạo ra |  |  |  |
| overlap | Separation trong overlap | SIR | Mức speaker còn lại bị lọt vào kênh |  |  |  |
| all | Độ rõ speech | ESTOI/STOI | Khả năng giữ lại nội dung lời nói |  |  |  |
| all | Chất lượng nghe | PESQ/POLQA | Mức tương đồng về perceptual quality |  |  |  |
| overlap | Crosstalk | Crosstalk rate | Tỷ lệ frame bị lẫn speaker thứ hai |  |  |  |
| all | Timing | VAD F1, onset/offset error | Prediction có giữ đúng thời điểm nói không |  |  |  |
| overlap | Overlap timing | Overlap F1/IoU | Prediction có phát hiện đúng vùng overlap không |  |  |  |
