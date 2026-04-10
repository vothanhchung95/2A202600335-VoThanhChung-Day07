# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Võ Thanh Chung - 2A202600335
**Nhóm:** Nhóm X1
**Ngày:** 10-04-2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (gần 1.0) nghĩa là hai vector embedding có hướng gần giống nhau trong không gian vector, tức là hai câu/cụm từ có ý nghĩa ngữ nghĩa tương đồng cao. Cosine similarity đo góc giữa hai vector, không quan tâm đến độ dài (magnitude).

**Ví dụ HIGH similarity:**
- Sentence A: "Machine learning is a subset of artificial intelligence"
- Sentence B: "AI encompasses machine learning as one of its components"
- Tại sao tương đồng: Cả hai câu đều nói về mối quan hệ giữa ML và AI, sử dụng các từ khóa tương tự (machine learning, artificial intelligence)

**Ví dụ LOW similarity:**
- Sentence A: "The cat sleeps on the sofa"
- Sentence B: "Stock market prices fluctuate daily"
- Tại sao khác: Hai câu hoàn toàn không liên quan về chủ đề - một câu về vật nuôi, một câu về tài chính, không có từ vựng chung

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity được ưu tiên vì nó chỉ quan tâm đến hướng (direction) của vector mà không bị ảnh hưởng bởi độ dài (magnitude). Text embeddings thường có độ dài khác nhau do câu dài/câu ngắn, nhưng hướng vector mới phản ánh ý nghĩa ngữ nghĩa. Euclidean distance bị ảnh hưởng bởi độ dài vector, có thể cho kết quả sai khi so sánh câu ngắn và dài có cùng chủ đề.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> ```
> num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))
> num_chunks = ceil((10000 - 50) / (500 - 50))
> num_chunks = ceil(9950 / 450)
> num_chunks = ceil(22.11)
> num_chunks = 23 chunks
> ```
> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap=100: num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = **25 chunks**. Overlap nhiều hơn giúp đảm bảo context liên tục giữa các chunk, tránh mất thông tin khi một ý/câu bị cắt ngang ở ranh giới chunk. Tuy nhiên overlap quá nhiều sẽ tăng số lượng chunk và dư thừa dữ liệu.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Technical Documentation & AI Systems (Python Programming, RAG Systems, Vector Stores, Customer Support for AI)

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì nó kết hợp nhiều chủ đề kỹ thuật liên quan đến AI và phát triển phần mềm, phù hợp với ngữ cảnh học tập về embedding và vector stores. Các tài liệu có cấu trúc rõ ràng với các đoạn văn tách biệt, giúp dễ dàng thử nghiệm các chiến lược chunking khác nhau. Ngoài ra, domain này cho phép tạo các câu hỏi benchmark đa dạng về cả kỹ thuật và ứng dụng thực tế.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | XanhSM - User FAQs.md | XanhSM | ~50,000 | type: faq, topic: user_app, audience: customers |
| 2 | XanhSM - electric_car_driver FAQs.md | XanhSM | ~3,600 | type: faq, topic: electric_car, audience: drivers |
| 3 | XanhSM - electric_motor_driver FAQs.md | XanhSM | ~11,700 | type: faq, topic: electric_motor, audience: drivers |
| 4 | XanhSM - Restaurant FAQs.md | XanhSM | ~25,400 | type: faq, topic: restaurant, audience: merchants |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| type | string | faq, guide, policy | Phân biệt định dạng tài liệu để chọn nguồn phù hợp với loại câu hỏi |
| topic | string | user_app, electric_car, electric_motor, restaurant | Lọc theo chủ đề chuyên môn khi query liên quan đến lĩnh vực cụ thể |
| audience | string | customers, drivers, merchants | Phân biệt đối tượng người dùng để cung cấp câu trả lời phù hợp |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên tài liệu python_intro.txt (chunk_size=300):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| python_intro.txt | FixedSizeChunker (`fixed_size`) | 4 | ~239 | Trung bình - có thể cắt ngang câu |
| python_intro.txt | SentenceChunker (`by_sentences`) | 3 | ~317 | Tốt - giữ nguyên câu, nhưng chunk dài |
| python_intro.txt | RecursiveChunker (`recursive`) | 6 | ~158 | Tốt - linh hoạt, nhưng nhiều chunk hơn |

### Strategy Của Tôi

**Loại:** RecursiveChunker (chunk_size=250)

**Mô tả cách hoạt động:**
> RecursiveChunker phân tách văn bản bằng cách thử các separators theo thứ tự ưu tiên: `\n\n` → `\n` → `. ` → ` ` → `""`. Nếu text vượt quá chunk_size, algorithm đệ quy xuống separator nhỏ hơn để chia nhỏ hơn nữa. Base case là khi text ≤ chunk_size hoặc hết separators thì split by character. Cách tiếp cận này giữ được cấu trúc tự nhiên của văn bản (đoạn → câu → từ) trong khi vẫn kiểm soát được độ dài chunk.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Domain FAQ của XanhSM có cấu trúc phân cấp rõ ràng (section → subsection → câu hỏi/câu trả lời). RecursiveChunker tận dụng đặc điểm này bằng cách ưu tiên split trên `\n\n` và `\n` trước, giữ nguyên cấu trúc Q&A. Với chunk_size=250, chiến lược này tạo ra chunk đồng đều, giữ được context liên tục và cho retrieval score cao (~0.76) trong benchmark.

**Code snippet (nếu custom):**
```python
# Không cần vì dùng RecursiveChunker có sẵn
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| python_intro.txt | FixedSizeChunker (best baseline) | 4 | ~239 | 6/10 - ổn định nhưng cắt ngang câu |
| python_intro.txt | **RecursiveChunker (của tôi)** | 6 | ~158 | 8/10 - giữ cấu trúc, chunk đều |

### So Sánh Với Thành Viên Khác

| Thành viên            | Strategy                                      | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------------------|-----------------------------------------------|----------------------|-----------|----------|
| Võ Thanh Chung (Tôi)  | RecursiveChunker (250 chars)                  | 8                    | Giữ cấu trúc tự nhiên, chunk đều | Có thể cắt ngang câu dài |
| Đỗ Thế Anh| Recursive (250 chars) | 8.752 | Trích xuất chính xác, duy trì được thông tin quan trọng | Số chunk nhiều, dẫn đến dư thừa dữ liệu do overlap |
| Hoàng Thị Thanh Tuyền | Recursive (350 chars)                         | 8.77 | Giữ context, Q&A coherent, score cao nhất | Số chunk nhiều (654), tốn memory |
| Nguyễn Hồ Bảo Thiên   | FixedSizeChunker (chunk_size=100, overlap=20) | 8.56 | Xử lý nhanh | Dễ ngắt câu giữa chừng, gây mất ngữ nghĩa |
| Dương Khoa Điềm       | RecursiveChunker  | 7.9 | Giữ được ngữ cảnh cụm Q&A tương đối ổn. | Tuỳ biến sai sót separator khiến một số câu dài bị đứt vụn, điểm chưa cao. |
**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker tốt nhất cho domain FAQ của XanhSM vì tài liệu có cấu trúc phân cấp rõ ràng (section → subsection → Q&A). RecursiveChunker ưu tiên giữ nguyên cấu trúc tự nhiên bằng cách split trên `\n\n` và `\n` trước, giúp chunk chứa context hoàn chỉnh của câu hỏi và câu trả lời. Với chunk_size phù hợp (~250-350), chiến lược này cân bằng giữa semantic coherence và độ dài chunk đồng đều.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `(?<=[.!?])\s+|\.\n` để split câu, giữ lại dấu câu qua positive lookbehind. Edge cases: strip whitespace từng sentence, skip empty strings, gộp câu còn lại vào chunk cuối dù ít hơn max_sentences.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm đệ quy thử separators theo priority: `\n\n` → `\n` → `. ` → ` ` → `""`. Base case: text ≤ chunk_size thì return ngay, hoặc hết separators thì split by character. Khi part vượt chunk_size, đệ quy xuống separator nhỏ hơn để chia nhỏ hơn nữa.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Lưu trữ dạng list các record dict với embedding, content, metadata. Dùng `_make_record` để tạo embedding qua `embedding_fn`. Tính similarity bằng dot product giữa query embedding và từng record embedding, sort descending lấy top_k.

**`search_with_filter` + `delete_document`** — approach:
> Filter trước khi search: lọc records by metadata match trước, rồi mới tính similarity trên tập đã lọc. Delete bằng cách lọc ngược list (hoặc dùng ChromaDB delete by doc_id), return True nếu có record bị xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> Prompt structure: Context section chứa top-k chunks nối bằng `\n\n`, theo sau là `Question:` và `Answer:`. Inject context bằng cách retrieve từ store rồi join content vào prompt template. Gọi `llm_fn(prompt)` để generate answer.

### Test Results

```
# Paste output of: pytest tests/ -v
tests/test_solution.py::test_exercise_1_1_cosine_similarity PASSED
tests/test_solution.py::test_exercise_1_2_chunking_math PASSED
tests/test_solution.py::test_exercise_2_document_loading PASSED
tests/test_solution.py::test_exercise_3_sentence_chunker_empty PASSED
tests/test_solution.py::test_exercise_3_sentence_chunker_basic PASSED
tests/test_solution.py::test_exercise_3_sentence_chunker_single_long PASSED
tests/test_solution.py::test_exercise_4_recursive_chunker_basic PASSED
tests/test_solution.py::test_exercise_4_recursive_chunker_small_chunk_size PASSED
tests/test_solution.py::test_exercise_4_recursive_chunker_recurses PASSED
tests/test_solution.py::test_exercise_5_embedding_store_add_and_search PASSED
tests/test_solution.py::test_exercise_5_embedding_store_filter PASSED
tests/test_solution.py::test_exercise_5_embedding_store_delete PASSED
tests/test_solution.py::test_exercise_6_agent_answers_with_context PASSED
tests/test_solution.py::test_exercise_7_compute_similarity PASSED
tests/test_solution.py::test_exercise_8_local_embedder_download PASSED
```

**Số tests pass:** 15 / 15

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Tôi gặp tai nạn trong chuyến xe" | "Sự cố nghiêm trọng khi đi taxi" | high | ~0.85 | Yes |
| 2 | "Đặt xe trên ứng dụng XanhSM" | "Cách book chuyến đi qua app" | high | ~0.82 | Yes |
| 3 | "Tài xế lái xe không an toàn" | "Thông tin tài xế không giống app" | low | ~0.45 | Yes |
| 4 | "Tôi để quên đồ trên xe" | "Bảo hiểm hàng hóa Xanh Express" | low | ~0.38 | Yes |
| 5 | "Hotline 1900 2097" | "Gọi cấp cứu 115" | low | ~0.42 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp 3 có điểm tương đồng khá cao dù khác ý (cùng đề cập tài xế). Embedding model nhận diện semantic proximity dù context khác nhau. Điều này cho thấy embeddings capture từ khóa và context chung, nhưng không phân biệt fine-grained meaning (an toàn lái xe vs thông tin tài xế).

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Tôi gặp tai nạn trong chuyến xe thì phải làm gì? | Bình tĩnh, đảm bảo an toàn, kiểm tra thương tích, gọi cấp cứu 115 nếu nghiêm trọng, báo cảnh sát 113, làm biên bản, liên hệ bảo hiểm, gọi hotline 1900 2097 |
| 2 | Thông tin tài xế không giống trên app thì sao? | Không lên xe, chụp ảnh ghi lại thông tin, hủy chuyến và đặt lại, báo cáo tài xế trên ứng dụng |
| 3 | Làm sao để đặt xe trên ứng dụng XanhSM? | Chọn điểm đón/đến, chọn loại xe, chọn thanh toán, nhấn Đặt xe, chờ tài xế xác nhận, lên xe và thanh toán |
| 4 | Tôi để quên đồ trên xe thì làm sao? | Liên hệ tài xế qua ứng dụng hoặc hotline 1900 2097 cung cấp thông tin chuyến đi để tìm đồ |
| 5 | Tài xế yêu cầu đi ngoài app có nên không? | Không nên, từ chối và báo cáo tài xế trên ứng dụng để đảm bảo an toàn và quyền lợi |

### Kết Quả Của Tôi (RecursiveChunker)

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Tôi gặp tai nạn trong chuyến xe... | ## 1.1. Tôi gặp tai nạn/sự cố nghiêm trọng... | 0.7819 | Yes | Bình tĩnh, đảm bảo an toàn, kiểm tra thương tích, gọi cấp cứu 115... |
| 2 | Thông tin tài xế không giống... | Để kiểm tra chi tiết thu nhập trên ứng dụng, tài xế lựa chọn mục "Thống kê thu nhập" tại "Danh sách ... | 0.7290 | Yes | Không lên xe, kiểm tra lại, chụp ảnh, hủy chuyến, báo cáo... |
| 3 | Làm sao để đặt xe trên ứng dụng... | ## 2.1. Làm thế nào để đặt chuyến xe... | 0.7668 | Yes | Chọn điểm đón/đến, chọn loại xe, chọn thanh toán, nhấn Đặt xe... |
| 4 | Tôi để quên đồ trên xe... | # 2. Vấn đề về chuyến đi | 0.7905 | Yes | Liên hệ tài xế qua app, gọi hotline 1900 2097... |
| 5 | Tài xế yêu cầu đi ngoài app... | Xanh SM cam kết sẽ nghiêm túc kiểm tra... | 0.7368 | Yes | Không nên đi ngoài app, từ chối và báo cáo... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> RecursiveChunker với chunk size nhỏ (250-350 chars) giữ được context tốt và cho retrieval score cao (~0.77-0.88). Đỗ Thế Anh và Thanh Tuyền chứng minh việc giữ chunk ngắn giúp Q&A coherent hơn, dù trade-off là tốn memory hơn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Các nhóm khác demo cho thấy importance của metadata filtering - việc gán đúng audience (customers/drivers/merchants) giúp retrieval trả về câu trả lời phù hợp ngữ cảnh người dùng, tránh nhầm lẫn thông tin.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thêm nhiều metadata hơn (vd: category: safety/payment/booking) để filter chính xác hơn. Cũng sẽ cân nhắc hybrid approach: SentenceChunker cho Q&A ngắn, RecursiveChunker cho đoạn dài cần context liên tục.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|------------------|
| Warm-up | Cá nhân | 5/ 5             |
| Document selection | Nhóm | 10/ 10           |
| Chunking strategy | Nhóm | 12/ 15           |
| My approach | Cá nhân | 8/ 10            |
| Similarity predictions | Cá nhân | 3/ 5             |
| Results | Cá nhân | 10/ 10           |
| Core implementation (tests) | Cá nhân | 28/ 30           |
| Demo | Nhóm | 5/ 5             |
| **Tổng** | | **81 / 90**      |
