# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

## Kết nối dữ liệu Day 6 của bài này

Adapter [`starter/day6_connection.py`](starter/day6_connection.py) đã được nối
với PostgreSQL/pgvector của thư mục `../lab06`. Cấu hình mặc định dùng strategy
`B` (cấu hình được Day 6 chọn trên dev), model
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` và vector 384
chiều. Mật khẩu không được sao chép sang repo này; adapter đọc file
`../lab06/starter/.env` hoặc biến `DAY6_DATABASE_URL`.

```powershell
# Khởi động store Day 6 nếu chưa chạy
docker compose -p day6_student_starter -f ..\lab06\starter\docker-compose.yml up -d --wait

# Dùng môi trường Day 6 đã cài đủ dependency để kiểm tra và xuất JSON Day 7
..\lab06\starter\.venv\Scripts\python.exe starter\day6_connection.py --check --export data

# Kiểm tra lexical baseline trên cùng chunk Day 6
python starter\baseline.py --chunks data\day6_chunks.json --questions data\day6_questions.json --split dev --output starter\results\day6-dev-tfidf
```

Trong agent thật, khởi tạo `Day6DenseRetriever()` rồi truyền vào
`run_agent`. Hàm `search` trả đúng contract Day 7 gồm `id`, `text`,
`source_id`, `source`, `page`, `metadata` và `score`; query được embed bằng
đúng model đã dùng để index Day 6.

Benchmark đã có đủ 24 câu tối thiểu: 15 dev + 5 test có đáp án và 2 câu thiếu
bằng chứng cho mỗi split. Ba phương pháp retrieval nằm trong
[`starter/retrieval.py`](starter/retrieval.py). Chạy benchmark **dev** bằng:

```powershell
$env:HF_HUB_OFFLINE="1"
..\lab06\starter\.venv\Scripts\python.exe starter\benchmark_retrieval.py `
  --chunks data\day6_chunks.json `
  --questions data\day6_questions.json `
  --split dev `
  --output starter\results\retrieval-dev
```

Script so sánh TF-IDF, dense không filter, dense có filter và hybrid RRF. Không
chạy `--split test` cho đến khi đã phân tích dev và chốt cấu hình.

Sau phân tích dev, cấu hình được chọn là hybrid xen kẽ dense/TF-IDF có khử
trùng, `top_k=3`, dùng metadata filter và context budget 3.000 ký tự. Cấu hình
này đạt Hit@3 `15/15` và Context Evidence Rate `15/15` trên dev. Kết quả và
trace nằm trong `starter/results/retrieval-dev-b3000/`; test chỉ được chạy sau
khi cấu hình này đã khóa.

## Kết quả cuối

- Retrieval test: TF-IDF và hybrid đều đạt Hit@3 `5/5`, Context Evidence Rate `5/5`; hybrid Hit@1 `3/5`.
- Agent dev: 15 câu trả lời + 2 abstain, grounded mean `1,93/2`, citation `22/22`, không lỗi.
- Agent test: 4 câu trả lời, 2 abstain đúng, 1 provider error; grounded mean `1,75/2` trên bốn câu hoàn tất, citation `5/5`.
- Provider/model: Anthropic `claude-haiku-5-5`; test chỉ chạy sau khi khóa cấu hình trên dev và không được chạy lại để tinh chỉnh.

### Kết quả thử thách nâng cao

Tài liệu *Data Science for Business* 130 trang được index toàn bộ bằng strategy
`ADVANCED_DSFB`: 311 chunk gồm 277 text, 13 table và 21 visual có page/bbox.
Benchmark riêng có 18 câu DEV và 30 câu TEST đúng cơ cấu bắt buộc. Pipeline được
khóa sau DEV rồi chạy TEST đúng một lần bằng `hybrid_lexical2`, top-k 6, tối đa
4 lượt tìm và context 14.000 ký tự.

- TEST agent: **30/30 (100%)**, vượt ngưỡng 90%; 24 answer, 6 abstain, 0 lỗi.
- Theo nhóm: table 6/6, image 5/5, chart 5/5, multi-hop 4/4, text 4/4,
  unanswerable 6/6.
- 70 model call, 40 search call; 165.739 input và 21.445 output token; chi phí
  ước tính 0,0272964 USD theo giá Haiku 5.5 cho prompt dưới 100k token.
- PDF và extracted full text không nằm trong gói nộp vì bản quyền. Trace công
  khai ở `advanced/submission/` đã bỏ context dài nhưng giữ đủ ID, trang, vùng,
  rank, answer, citation, thời gian và nhãn audit.

Xem hướng dẫn tái tạo ở [`advanced/README.md`](advanced/README.md), cấu hình đã
khóa ở `advanced/config/locked_pipeline.json`, benchmark ở
`advanced/benchmark/questions.json` và kết quả nộp ở `advanced/submission/`.

Xem báo cáo tại `report/individual/2A202603720_HaQuangVinh.md` và artifacts
cuối trong các thư mục `starter/results/*-final`, `starter/results/agent-dev`,
`starter/results/retrieval-dev-b3000`.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
