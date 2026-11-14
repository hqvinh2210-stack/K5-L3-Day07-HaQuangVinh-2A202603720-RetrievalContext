# Submission checklist — Day 7

Checklist này đối chiếu trực tiếp `SUBMISSION.md`, `BENCHMARK.md`, `RUBRIC.md`
và `ADVANCED.md`. Giao diện demo không thuộc bài nộp.

## Phần cơ bản (85 điểm)

- [x] Agent thật, điều kiện dừng, citation/context validation:
  `starter/agentic.py`, `starter/model_controller.py`.
- [x] Ít nhất ba retrieval method, có dense pgvector Day 6:
  `starter/retrieval.py`, `starter/day6_connection.py`.
- [x] Benchmark 24 câu, tách DEV/TEST trước thử nghiệm:
  `data/day6_questions.json` (15+2 DEV, 5+2 TEST).
- [x] Dữ liệu Day 6 và hướng tái tạo:
  `data/day6_chunks.json`, `starter/README.md`, `.env.example`.
- [x] So sánh TF-IDF, dense, RRF và hybrid interleave trên DEV:
  `starter/results/retrieval-dev-b3000/`.
- [x] Retrieval TEST baseline và cấu hình khóa:
  `starter/results/retrieval-test-final/`.
- [x] Đọc thủ công context của 5 câu TEST cho cả hai method, thang 0/1/2:
  `starter/results/retrieval-test-final/context_reviews.json`.
- [x] Agent DEV và audit nhãn/citation:
  `starter/results/agent-dev/`.
- [x] Agent TEST cấu hình chọn và agent TEST baseline TF-IDF:
  `starter/results/agent-test-final/`,
  `starter/results/agent-test-baseline-final/`.
- [x] Trace tool call, context, answer, citation, lỗi, token, thời gian:
  các file `summary.json` và `traces.json` trong thư mục kết quả.
- [x] Báo cáo đúng tên MSSV/họ tên:
  `report/individual/2A202603720_HaQuangVinh.md`.
- [x] Test code: `23 passed` với `python -m pytest -q`.

## Thử thách nâng cao (15 điểm)

- [x] Nguồn thực tế trên 100 trang, có metadata và đường truy cập hợp pháp:
  O'Reilly, ISBN `9781449374273`; file coursework là bản trích 130 trang và
  được index đủ 130/130 trang. Xem `advanced/README.md`.
- [x] 311 chunks: 277 text, 13 table, 21 visual; page/bbox/label được lưu:
  `advanced/extract_and_index.py`, `advanced/config/visual_annotations.json`.
- [x] Benchmark riêng 18 DEV + 30 TEST, đủ nhóm bắt buộc và evidence từng vùng:
  `advanced/benchmark/questions.json`.
- [x] Pipeline khóa trước TEST:
  `advanced/config/locked_pipeline.json`.
- [x] TEST 30/30 (100%), 0 lỗi, có nhãn/lý do từng câu và metrics theo nhóm:
  `advanced/submission/test_metrics.json`,
  `advanced/submission/test_reviews.json`.
- [x] Trace tái kiểm tra không phát tán toàn văn sách:
  `advanced/submission/test_trace_sanitized.json`.
- [x] Token, thời gian và chi phí:
  `advanced/submission/run_costs.json` và báo cáo cá nhân.

## An toàn bài nộp

- [x] `.env`, API key, PDF bản quyền, vector và extracted full text bị ignore.
- [x] Không có API key/token trong file tracked.
- [x] Không nộp `demo-ui/`.
- [x] Repo cá nhân dùng `origin`; repo đề bài chỉ nằm ở `upstream`.

