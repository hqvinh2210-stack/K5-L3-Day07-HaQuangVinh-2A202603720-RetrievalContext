# Thử thách nâng cao — Data Science for Business

Pipeline này lập chỉ mục toàn bộ bản PDF 130 trang ở thư mục gốc, tạo benchmark
DEV/TEST riêng và đánh giá agent mà không đưa gold vào lúc chạy.

## Nguồn và quyền sử dụng

- Tài liệu: *Data Science for Business*, Foster Provost và Tom Fawcett,
  O'Reilly Media, 2013, ISBN `9781449374273`. Trang publisher chính thức:
  <https://www.oreilly.com/library/view/data-science-for/9781449374273/>.
- File đầu vào cục bộ: `Data Science for Business_1.pdf` (bản trích 130 trang
  vật lý). Pipeline index toàn bộ 130/130 trang của file được cung cấp; không
  tuyên bố đây là toàn bộ ấn bản 414 trang của nhà xuất bản.
- Đây là sách có bản quyền. PDF, text trích xuất đầy đủ, vector và kết quả chứa
  context dài đều bị `.gitignore` loại khỏi gói nộp. Người chấm cần tự đặt bản
  PDF trích 130 trang hợp pháp cùng tên vào thư mục gốc rồi chạy lại pipeline.
  Người chấm có thể đối chiếu metadata/ấn bản và truy cập hợp pháp qua trang
  O'Reilly ở trên; repo không tái phân phối PDF.
- Benchmark chỉ lưu các trích đoạn ngắn cần thiết để kiểm chứng câu hỏi.

## Pipeline

`extract_and_index.py` dùng pypdf cho text, pdfplumber cho layout/bảng/vùng ảnh,
và 27 chú thích vùng được kiểm tra thủ công (6 bảng, 21 hình/sơ đồ/biểu đồ).
Mỗi chunk giữ `source_id`, trang PDF vật lý, loại nội dung và `bbox` nếu là vùng
trực quan. Embedding dùng model Day 6
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều,
chuẩn hóa L2, cosine trong pgvector, strategy `ADVANCED_DSFB`.

Thống kê hiện tại: 311 chunk gồm 277 text, 13 table và 21 visual; 62 image
object được phát hiện trên 53 trang. Các trang 1, 2, 6, 16, 42 và 104 có dưới
50 ký tự text vì là bìa/trang phân cách; không làm mất evidence benchmark.

## Benchmark

`benchmark/questions.json` có 18 câu DEV (15 có đáp án + 3 không đủ evidence)
và 30 câu TEST (24 có đáp án + 6 không đủ evidence). TEST có đúng 6 câu bảng,
5 câu ảnh/sơ đồ, 5 câu biểu đồ, 4 câu multi-hop và 4 câu text. Câu multi-hop
chỉ đạt retrieval coverage khi có đủ mọi vùng gold.

Kiểm tra cấu trúc và bằng chứng:

```powershell
..\lab06\starter\.venv\Scripts\python.exe advanced\benchmark\validate_benchmark.py
```

## Cách tái tạo

```powershell
# PostgreSQL/pgvector Day 6 phải đang chạy; .env chỉ tới DAY6_ENV_FILE.
..\lab06\starter\.venv\Scripts\python.exe advanced\extract_and_index.py

# Chỉ được thử cấu hình trên DEV.
..\lab06\starter\.venv\Scripts\python.exe advanced\evaluate_retrieval.py `
  --split dev --top-k 10 --budget 16000 `
  --output advanced\results\retrieval_dev

# Sau khi chốt cấu hình, chạy agent DEV rồi mới khóa pipeline và mở TEST.
..\lab06\starter\.venv\Scripts\python.exe starter\run_agent_cli.py `
  --chunks advanced\data\chunks.json `
  --questions advanced\benchmark\questions.json `
  --split dev --method hybrid_lexical2 --dense-strategy ADVANCED_DSFB `
  --top-k 6 --budget 14000 --max-searches 4 --max-output-tokens 2048 `
  --provider anthropic --output advanced\results\agent_dev
```

Kết quả agent phải được đọc và gắn nhãn từng câu trong `reviews.json`. Script
`score_agent.py` chỉ tổng hợp nhãn pass/fail; nó không dùng LLM judge và không
tự suy diễn rằng citation ID đồng nghĩa với evidence đúng.

## Kết quả đã khóa

- DEV chính: 15/15 answered, 3/3 abstained, 0 lỗi.
- TEST chạy đúng một lần sau khi khóa: **30/30 (100%)**, gồm table 6/6,
  image 5/5, chart 5/5, multi-hop 4/4, text 4/4 và unanswerable 6/6.
- TEST dùng 70 model call, 40 search call, 165.739 input token và 21.445 output
  token; chi phí ước tính 0,0272964 USD theo giá Haiku 5.5 cho prompt dưới
  100k token tại ngày 2026-11-14.
- `submission/test_trace_sanitized.json` giữ trace tái kiểm tra nhưng không chứa
  toàn văn chunk/context. Nhãn thủ công, metrics, retrieval summary, cost và
  manifest công khai nằm cùng thư mục `submission/`.
