# Kế hoạch thực hiện project Smart RAG QA

Dataset chính: **Zalo AI 2021 — Legal Text Retrieval**
Nguồn: https://www.kaggle.com/datasets/hariwh0/zaloai2021-legal-text-retrieval

## 1. Tổng quan dataset

| File | Nội dung | Vai trò trong project |
|---|---|---|
| `legal_corpus.json` (~119 MB) | ~7.000 văn bản pháp luật, cấu trúc `law_id` → `articles[]`, mỗi article có `article_id`, `title`, `text` | Corpus — đơn vị chunking tự nhiên là **điều luật (article)** |
| `train_question_answer.json` | ~5.000 câu hỏi tiếng Việt + danh sách điều luật relevant (đã gán nhãn) | Relevance judgments cho đánh giá retrieval + dữ liệu fine-tune |
| `public_test_question.json` | Câu hỏi test công khai (không nhãn) | Có thể dùng làm nguồn câu hỏi demo; không dùng tính metric |
| `stopwords.txt` | Danh sách stop-word tiếng Việt | Dùng cho phòng tiền xử lý |

Điểm mạnh với đồ án:

- **Không cần gán nhãn relevance từ đầu** — `train_question_answer.json` cung cấp ground truth ở mức điều luật, đủ để tính Recall@K, MRR, MAP cho RQ1–RQ4.
- Corpus có cấu trúc rõ (law → article) nên chunking theo cấu trúc là khả thi, đáp ứng Chunking Lab.
- Đơn vị relevance chốt: **article** (`law_id` + `article_id`). Lưu ý: hệ thống truy hồi theo chunk; nếu 1 article = 1 chunk thì hai đơn vị trùng nhau.

Quy ước: mọi metric báo cáo theo **article-level relevance**; nếu một article bị chia thành nhiều chunk thì tính hit khi bất kỳ chunk nào của article đó nằm trong Top-K.

## 2. Phân chia dữ liệu (chốt trước khi làm bất kỳ thứ gì)

- Từ `train_question_answer.json`: tách theo câu hỏi, tỷ lệ **train/dev/test = 70/10/20** (khoảng 3.500/500/1.000 câu hỏi), `random_seed = 42`, stratify theo số điều relevant.
- `split` lưu vào từng bản ghi câu hỏi; split file commit vào `data/processed/question_split.json`.
- **RQ1, RQ2, RQ3** (đánh giá retriever): chỉ dùng **dev + test**. Train set không dùng gì ở giai đoạn này.
- **RQ4** (fine-tune): chỉ dùng **train** để fine-tune, **dev** để chọn checkpoint, **test** để báo cáo. Đây là biện pháp chống data leakage bắt buộc.

## 3. Kế hoạch theo tuần

### Tuần 1 — Khởi động + Giai đoạn 1 (bắt đầu MVP)

| Việc | Màn hình tương ứng |
|---|---|
| Tạo cấu trúc project, venv, `requirements.txt`, cấu hình base | — |
| Script tải dataset Kaggle → `data/raw/`; ghi `dataset_version` | — |
| Parser `legal_corpus.json` → `Document` + `Chunk` (article-level); kiểm tra rỗng/trùng/encoding | 4.1 Tổng quan, 4.2 Quản lý dữ liệu |
| Phòng tiền xử lý: normalization, tách câu, tách từ (none/underthesea/PyVi), so sánh trước/sau | 4.3 Phòng tiền xử lý |

Sản phẩm: app Streamlit chạy được, nạp và xem trước corpus.

### Tuần 2 — Giai đoạn 1 (tiếp)

| Việc | Màn hình |
|---|---|
| Chunking Lab: fixed-size + theo article; hiển thị ranh giới, overlap, thống kê | 4.4 Chunking Lab |
| TF-IDF + cosine baseline (scikit-learn); lưu model + config | 4.5 Lập chỉ mục |
| Sentence-BERT tiếng Việt (ví dụ `VoVanPhuc/simcse-VietNamese-phobert-base` hoặc `keepitreal/vietnamese-sbert`) + ChromaDB; batch + cache | 4.5 Lập chỉ mục |
| Retrieval Playground: Top-K, score, so sánh 2 retriever cạnh nhau | 6 Retrieval Playground |

Sản phẩm: MVP trình diễn được từ ingestion → retrieval.

### Tuần 3 — Giai đoạn 1 (hoàn tất)

| Việc | Màn hình |
|---|---|
| Tích hợp LLM mã nguồn mở (ví dụ Qwen2.5 / PhoGPT qua Ollama hoặc transformers) | 4.7 Hỏi–đáp RAG |
| Prompt grounded + trích dẫn nguồn + chế độ "không đủ thông tin" | 4.7 |
| Prompt/context viewer, preset tham số generation | 4.7 |

Sản phẩm: **MVP hoàn chỉnh — demo được**. Milestone: chốt Giai đoạn 1.

### Tuần 4 — Giai đoạn 2 (thực nghiệm)

| Việc | Màn hình |
|---|---|
| Module metric: Precision@K, Recall@K, Hit@K, MRR, MAP, NDCG; unit test cho từng metric | 4.8 Retrieval evaluation |
| **RQ1:** chạy none/underthesea/PyVi trên dev/test, giữ mọi biến khác cố định | 4.8 |
| **RQ2:** TF-IDF vs SBERT pretrained trên cùng test | 4.8 |
| Bảng + biểu đồ xuất ra `results/` | 4.8 |

### Tuần 5 — Giai đoạn 2 (tiếp)

| Việc | Màn hình |
|---|---|
| **RQ3:** sweep `chunk_size`/`overlap` (ví dụ 128/256/512, overlap 0/25%); đo retrieval trước, generation sau | 4.8 |
| Generation evaluation: faithfulness/answer relevance (rubric + automatic nếu có reference); do Zalo không có reference answer cho train, human rubric là chính | 4.8 Generation evaluation |
| Error analysis theo nhóm lỗi; lưu case + ghi chú | 4.9 Error Analysis |

Sản phẩm: bảng metric đầy đủ RQ1–RQ3 + error analysis.

### Tuần 6 — Giai đoạn 2.5 (fine-tune embedding)

| Việc | Màn hình |
|---|---|
| Sinh bộ ba (query, positive, hard negative) từ train split; hard negative = article sai trong Top-K của SBERT pretrained | 4.10 Training Lab |
| Fine-tune bi-encoder (sentence-transformers, MNRL); chọn checkpoint theo dev Recall@K/MRR | 4.10 |
| Embed lại corpus, collection ChromaDB mới, cache key thêm `checkpoint_version` | 4.5 |
| **RQ4:** so sánh 3 nhóm trên cùng test set | 4.8 |

Sản phẩm: checkpoint + kết quả RQ4 + so sánh với pretrained.

### Tuần 7 — Báo cáo + slides

- Viết báo cáo (Word) theo cấu trúc Chương 1–6; nhúng bảng/biểu đồ từ `results/`.
- Slides theo flow: Problem → Method → Architecture → RQ1–RQ4 → Demo → Conclusion.
- Kiểm tra checklist mục 10 trong `DINH_HUONG_CHUC_NANG.md`.

Giai đoạn 3 (query expansion, hybrid, LoRA LLM) chỉ làm nếu còn thời gian.

## 4. Thứ tự phụ thuộc quan trọng

```text
Split dữ liệu (tuần 1) ── phải hoàn thành TRƯỚC mọi thí nghiệm
        ↓
Baseline TF-IDF (tuần 2) ── phải có trước khi so sánh SBERT
        ↓
RQ1–RQ3 trên pretrained (tuần 4–5) ── phải xong trước fine-tune
        ↓
Fine-tune (tuần 6) ── dùng train split, đánh giá trên test
```

Không fine-tune trước khi có kết quả baseline — nếu không sẽ không chứng minh được lợi ích.

## 5. Rủi ro và phương án

| Rủi ro | Phương án |
|---|---|
| Corpus 119 MB, embedding chậm trên CPU | Batching + cache embedding; chỉ embed lại khi config đổi; nếu không có GPU, dùng Google Colab cho bước embed + fine-tune rồi tải vector về |
| Câu hỏi Zalo không có reference answer | Generation evaluation dùng rubric tự đánh giá (faithfulness/relevance) thay vì EM/ROUGE; ghi rõ hạn chế này trong báo cáo |
| RAM/VRAM không đủ fine-tune | Fine-tune với batch nhỏ + gradient accumulation; hoặc chỉ fine-tune các layer pooler; giới hạn số epochs |
| Thay đổi preprocessing làm lệch so sánh | Mọi thực nghiệm ghi config JSON vào `experiments/`; một thí nghiệm = một thay đổi biến |
| LLM local quá chậm khi demo | Cài Ollama với model nhỏ (3–4B); chuẩn bị sẵn câu hỏi demo + cache câu trả lời để fallback |

## 6. Tham số cố định mặc định (baseline config)

```text
random_seed = 42
chunk_size = 256 (token) —(article-level chunking là mặc định cấu trúc)
chunk_overlap = 0 (article không cần overlap)
top_k = 10 (báo cáo tại 1, 3, 5, 10)
embedding_model = <chốt cụ thể tuần 2, ghi version chính xác>
tfidf: word_ngram_range = (1,2), tách từ theo nhóm thí nghiệm
llm: temperature = 0.2, max_tokens = 512
```

Mọi thay đổi tham số phải tạo config mới, không ghi đè kết quả cũ.
