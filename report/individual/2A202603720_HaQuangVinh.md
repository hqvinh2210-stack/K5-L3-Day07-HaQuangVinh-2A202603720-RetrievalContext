# Báo cáo Day 7

- Họ tên: Hà Quang Vinh
- Mã sinh viên: 2A202603720

## Dữ liệu và benchmark

- Nguồn: Nghị định 13/2023/NĐ-CP, bản scan 39 trang đã OCR từ Day 6; strategy B gồm 103 chunk có truy vết trang.
- Vector store: PostgreSQL 16 + pgvector 0.8.7; embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều, chuẩn hóa L2, cosine.
- Agent model: `claude-haiku-5-5` qua Anthropic Messages API native và JSON Structured Outputs. Dev gồm 15 câu có đáp án + 2 câu thiếu evidence; test gồm 5 + 2. Chỉ dùng dev để thử/chọn; test được chạy một lần sau khi khóa cấu hình.
- Evidence của câu có đáp án phải nằm nguyên văn trong chunk đúng `source_id` và trang. Câu thiếu evidence được viết gần chủ đề, kèm `absence_reason`, và phải được người chấm xác nhận sau khi agent tìm kiếm rồi từ chối.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF, top-3, không filter, budget 3.000 ký tự | Lexical baseline | dev | 9/15 (60%) | 13/15 (86,7%) | 13/15 (86,7%) | median 0,50 ms; 0 USD |
| Phương pháp 2 | Dense pgvector B, top-3, không filter, budget 3.000 | Semantic retrieval Day 6 | dev | 9/15 (60%) | 12/15 (80%) | 12/15 (80%) | median 70,01 ms; 0 USD |
| Phương pháp 3 | Hybrid RRF cân bằng, top-3, filter, budget 3.000 | Kết hợp rank dense và lexical | dev | 11/15 (73,3%) | 11/15 (73,3%) | 11/15 (73,3%) | median 61,01 ms; 0 USD |
| Cải thiện 1 | Hybrid xen kẽ dense/TF-IDF, khử trùng, top-3, filter, budget 1.600 | Giữ ứng viên mạnh riêng của từng retriever | dev | 9/15 (60%) | 15/15 (100%) | 10/15 (66,7%) | median 29,00 ms; 0 USD |
| Cải thiện 2 | Hybrid xen kẽ, top-3, filter, budget 3.000 | Tăng budget để không loại chunk evidence | dev | 9/15 (60%) | 15/15 (100%) | 15/15 (100%) | median 36,32 ms; 0 USD |
| Cấu hình đã chọn | Hybrid xen kẽ, top-3, filter, budget 3.000 | Khóa sau dev, không chỉnh bằng test | test | 3/5 (60%) | 5/5 (100%) | 5/5 (100%) | median 56,73 ms; 0 USD |
| Baseline | TF-IDF, top-3, không filter, budget 3.000 | Đối chứng lexical | test | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | median 0,58 ms; 0 USD |

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; giải thích rõ. Báo `n/N` và tỷ lệ, ghi đơn vị thời gian/chi phí, so sánh test sau khi chốt.

Phân tích lỗi dev: TF-IDF không tìm được evidence cho `q01–q02` vì các từ chung “dữ liệu cá nhân” xuất hiện dày đặc ở nhiều trang, trong khi dense xếp evidence tương ứng ở hạng 1–2. Ngược lại, dense bỏ lỡ `q06`, `q12`, `q14`; lexical tìm được evidence ở hạng 1–2 nhờ cụm từ điều kiện/thời điểm gần nguyên văn. RRF cộng hạng làm các chunk xuất hiện khá cao ở cả hai danh sách lấn át một số evidence chỉ mạnh ở một retriever. Hybrid xen kẽ khử trùng giữ lần lượt ứng viên của cả hai, đạt 15/15 Hit@3. Budget 1.600 vẫn loại chunk dài khỏi context nên tăng lên 3.000, đưa Context Evidence Rate từ 10/15 lên 15/15. Cấu hình được khóa trước test: `hybrid_interleave`, metadata filter, top-k 3, budget 3.000.

## Agent và đánh giá

- Tool retrieval là `hybrid_interleave`: xen kẽ dense pgvector và TF-IDF, filter metadata, khử trùng theo chunk ID. Agent giới hạn 4 lượt search, top-k 3, context budget 3.000; citation phải thuộc context. Controller dùng Claude Structured Outputs để chọn `search` hoặc `answer`. Lỗi provider/database được ghi `status: error`, không fallback giả.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 / dev | 17/17 | 39 model calls; 22 search calls | 0/1/14; 1,93/2 | 22/22 | 15/15 | 2/2 | 0/15 |
| Claude Haiku 5.5 / test, baseline TF-IDF | 7/7 | 33 model calls; 26 search calls | 3/0/2; 0,80/2 | 2/2 | 2/2 | 2/2 | 3/5 |
| Claude Haiku 5.5 / test | 6/6 case không lỗi | 19 model calls; 1 provider error | 0/1/3; 1,75/2 trên 4 câu hoàn tất | 5/5 | 4/4 | 2/2 | 0/4 |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q01 | Dev có đáp án | 1 search; evidence trang 2 | Định nghĩa đúng, `[nd13_2023:p002:paragraph:000]` | 2/2; citation hỗ trợ đầy đủ |
| q08 | Dev có đáp án | 1 search; context có gold | Kết luận “Có” nhưng thu hẹp sai sang trẻ em | 1/2; citation hỗ trợ các mệnh đề nhưng câu trả lời thiếu trực tiếp |
| q21 | Dev thiếu evidence | 4 search | “Không đủ bằng chứng để trả lời.” | Abstention đúng |
| q18 | Test có đáp án | Claude dùng hết 1.024 output token trước JSON hoàn chỉnh | `status: error`, không có câu trả lời giả | Provider error; loại khỏi metric chất lượng |
| q20 | Test có đáp án | 1 search; evidence trang 27 | Đúng 02 năm nhưng bỏ ngoại lệ doanh nghiệp trực tiếp kinh doanh xử lý dữ liệu | 1/2 |
| q23 | Test thiếu evidence | 3 search | “Không đủ bằng chứng để trả lời.” | Abstention đúng |

- Retrieval: `starter/results/retrieval-dev-b3000/` và `starter/results/retrieval-test-final/`. Agent trace, review và metric: `starter/results/agent-dev/` và `starter/results/agent-test-final/`.
- Audit context TEST bắt buộc đã đọc đủ 5 câu có đáp án cho cả TF-IDF và
  hybrid: cả hai đạt `5/5` mức 2 (đủ evidence và nguồn). Nhãn và lý do nằm ở
  `starter/results/retrieval-test-final/context_reviews.json`. Agent baseline
  vẫn false-abstain `3/5` vì model viết query tiếng Anh trên corpus tiếng Việt;
  kết quả được giữ nguyên, không chỉnh theo TEST.
- Token/cost: dev 60.956 input + 10.896 output token, ước tính từ 0,011544 USD; test 32.364 input + 6.215 output token, ước tính từ 0,006344 USD theo mức giá cơ sở công bố của Haiku 5.5. Chi phí thực tế cần đối chiếu Claude Console.
- Baseline agent TEST dùng 42.115 input + 7.993 output token, ước tính
  0,008208 USD; tổng thời gian theo case 89,14 giây, median 12,98 giây.
- Kết quả trước/sau: hybrid xen kẽ tăng dev Hit@3 từ dense 12/15 và TF-IDF 13/15 lên 15/15. Tăng budget 1.600 → 3.000 đưa Context Evidence Rate 10/15 → 15/15, đổi lại context trung bình tăng lên 2.268 ký tự. Giới hạn: Hit@1 hybrid test thấp hơn TF-IDF; model đôi khi thêm diễn giải ngoài trọng tâm (`q08`) hoặc bỏ điều kiện (`q20`); giới hạn output 1.024 gây một provider error (`q18`). Không chạy lại q18 để tránh sửa hệ thống sau khi xem test.

## Thử thách nâng cao (nếu thực hiện)

### Tài liệu, index và quyền sử dụng

- Tài liệu: *Data Science for Business*, Foster Provost và Tom Fawcett,
  O'Reilly Media, 2013, ISBN `9781449374273`; bản PDF cục bộ
  `Data Science for Business_1.pdf` là bản trích 130 trang vật lý. Pipeline
  index toàn bộ 130/130 trang của file cung cấp, không tuyên bố đây là toàn bộ
  ấn bản 414 trang. Trang truy cập/đối chiếu chính thức:
  <https://www.oreilly.com/library/view/data-science-for/9781449374273/>. Đây là
  sách có bản quyền nên PDF, text trích xuất đầy đủ, vector và context dài không
  được đưa vào gói nộp. Người chấm tái tạo bằng cách đặt bản PDF hợp pháp cùng
  tên ở thư mục gốc rồi chạy `advanced/extract_and_index.py`.
- Đã index toàn bộ 130 trang thành 311 chunk: 277 text, 13 table và 21 vùng
  visual đã kiểm tra thủ công. Pipeline phát hiện 7 bảng tự động và 62 image
  object; 6 bảng khó được bổ sung bằng annotation có `bbox`. Mọi chunk giữ
  `source_id`, trang PDF vật lý, loại nội dung; bảng/hình/biểu đồ giữ thêm nhãn
  và vùng trang. Strategy pgvector riêng là `ADVANCED_DSFB`, dùng MiniLM đa
  ngôn ngữ 384 chiều, chuẩn hóa L2 và cosine.
- Sáu trang 1, 2, 6, 16, 42, 104 có dưới 50 ký tự vì là bìa/trang phân cách;
  không làm mất evidence benchmark. Text, các bảng và 21 vùng trực quan được
  đối chiếu với trang render từ bản gốc trước khi viết benchmark.

### Benchmark nâng cao

- DEV: 18 câu, gồm 15 có đáp án (mỗi nhóm table/image/chart/multi-hop/text có
  3 câu) và 3 câu thiếu evidence.
- TEST khóa trước khi chạy: 30 câu, gồm 24 có đáp án và 6 câu thiếu evidence;
  phân bố đúng 6 table, 5 image/diagram, 5 chart, 4 multi-hop, 4 text và 6
  unanswerable. Mỗi câu chỉ thuộc một nhóm chính.
- Evidence được ghi bằng trang vật lý, đoạn ngắn và `bbox`/nhãn vùng khi là nội
  dung trực quan. Validator xác nhận 48/48 câu có cấu trúc đúng và mọi excerpt
  gold nằm trong chunk đúng nguồn/trang. Multi-hop chỉ đủ evidence khi tìm được
  tất cả vùng gold. Gold và đáp án không được truyền vào agent.

### Chọn pipeline trên DEV

| Cấu hình retrieval | Hit@1 | Evidence Hit@3 | Context evidence | Ghi chú |
|---|---:|---:|---:|---|
| TF-IDF, top-6, budget 8.000 | 2/15 | 9/15 | 12/15 | baseline lexical trên câu hỏi tiếng Việt |
| Dense pgvector, top-6, budget 8.000 | 3/15 | 6/15 | 8/15 | semantic baseline |
| Hybrid RRF cân bằng, top-6, budget 8.000 | 6/15 | 10/15 | 12/15 | tốt nhất Hit@1 |
| Hybrid RRF lexical×2, top-6, budget 8.000 | 5/15 | 10/15 | 12/15 | giữ tốt tên bảng/hình/biến |
| Hybrid RRF lexical×2, top-10, budget 16.000 | 5/15 | 10/15 | 14/15 | context tốt nhất; một câu multi-hop cần nhiều query |

Agent được cho phép viết query tiếng Anh ngắn theo ngôn ngữ tài liệu và tìm
riêng từng vế. DEV thật trả lời 15/15 và từ chối đúng 3/3, không lỗi: 40 model
call, 22 search call. Sau audit DEV, chỉ thêm hai quy tắc tổng quát: câu
multi-hop phải có evidence cho từng vế từ ít nhất hai vùng, và phải kiểm tra đủ
điều kiện/con số/đơn vị trước khi trả lời. Cấu hình được khóa trong
`advanced/config/locked_pipeline.json` trước khi đọc TEST:

- `hybrid_lexical2` (RRF lexical weight 2, dense weight 1), strategy
  `ADVANCED_DSFB`;
- top-k 6 mỗi search, tối đa 4 search, context budget 14.000 ký tự;
- Claude `claude-haiku-5-5`, Structured Output, tối đa 2.048 output token;
- citation chỉ hợp lệ nếu ID hiện có trong context, nếu thiếu evidence thì
  abstain; không fallback giả.

### Kết quả TEST và audit thủ công

Retrieval một truy vấn sau khi khóa đạt Hit@3 `15/24` và context evidence
`18/24` với hybrid lexical×2, so với `14/24` và `17/24` của TF-IDF. Chỉ số này
khắt khe hơn agent vì agent được tìm tối đa bốn query độc lập. Toàn bộ 30 câu
TEST chỉ chạy một lần: 24 answered, 6 abstained, 0 error, 70 model call và 40
search call. Tổng thời gian theo từng case là 201,77 giây; median 5,39 giây.

| Nhóm TEST | Đạt / tổng |
|---|---:|
| Bảng | 6/6 |
| Ảnh, sơ đồ | 5/5 |
| Biểu đồ | 5/5 |
| Multi-hop | 4/4 |
| Text | 4/4 |
| Thiếu evidence | 6/6 |
| **Toàn bộ** | **30/30 (100%)** |

Mỗi câu TEST được đọc và gắn nhãn pass/fail riêng, kiểm tra câu trả lời, phép
tính, điều kiện, citation và từng vùng gold trên bản gốc. Ví dụ `t17` kết hợp
p78/p79 và tính `0,37 − 0,13 = 0,24`; `t20` trích hai bảng p121/p122; `t25–t30`
đều tìm kiếm rồi từ chối, không thêm khẳng định thiếu căn cứ. Script tổng hợp
nhãn xác nhận `threshold_met=true`, không dùng LLM judge.

TEST dùng 165.739 input token và 21.445 output token. Theo giá Haiku 5.5 cho
prompt dưới 100k token (0,10 USD/triệu input và 0,50 USD/triệu output), chi phí
ước tính là **0,0272964 USD**; cần đối chiếu hóa đơn thực tế trong Claude
Console. Bộ DEV chính ước tính 0,0150484 USD. Retrieval/index chạy cục bộ, chi
phí dịch vụ 0 USD.

Artifacts có thể nộp nằm trong `advanced/benchmark/`, `advanced/config/` và
`advanced/submission/`. Trace nộp đã bỏ text/context dài của sách nhưng vẫn giữ
query, rank, chunk ID, trang, `bbox`, context ID, answer, citation, thời gian và
nhãn audit. Full trace chỉ giữ cục bộ trong thư mục ignored
`advanced/results/`.
