# Chốt kiến thức và công nghệ sử dụng trong project Smart RAG QA

Ngày chốt: 2026-08-24. Mọi thay đổi sau ngày này phải ghi rõ lý do và ngày thay đổi ở mục 6.

## 1. Tổng quan stack

| Thành phần | Công nghệ chốt | Phiên bản tối thiểu | Vai trò |
|---|---|---|---|
| Ngôn ngữ | Python | 3.10+ | Toàn bộ backend |
| Giao diện | Streamlit | 1.30+ | Web app-demo các màn hình 4.1–4.10 |
| Vector database | ChromaDB (persistent mode) | 0.4+ | Lưu embedding + tìm kiếm cosine |
| Baseline retrieval | scikit-learn (`TfidfVectorizer`) | 1.3+ | TF-IDF + cosine |
| Dense embedding | sentence-transformers | 2.2+ | Bi-encoder pretrained + fine-tune |
| LLM generation | Ollama (Qwen2.5 3B/7B) | — | Sinh câu trả lời local |
| Quản lý môi trường | venv + `requirements.txt` | — | Tái lập môi trường |

## 2. Kiến thức NLP vận dụng (ánh xạ chương môn học)

| Kiến thức | Áp dụng ở đâu | Chương |
|---|---|---|
| Text normalization, Unicode NFC | Phòng tiền xử lý (4.3) | Chương 1 |
| Sentence segmentation, tokenization | Phòng tiền xử lý (4.3) | Chương 1, mục 4.1–4.3 |
| Vietnamese word segmentation (underthesea/PyVi) | Phòng tiền xử lý + RQ1 | Chương 1 |
| Bag of Words, TF-IDF, Vector Space Model | Baseline retriever (4.5) | Chương 2 mục 5.1, Chương 3 mục 5 |
| BERT / Sentence-BERT encoder | Dense retriever (4.5) | Chương 2, mục 11 |
| Contrastive learning (bi-encoder fine-tune) | Training Lab (4.10), RQ4 | Chương 2, mục 11 (mở rộng) |
| Ranked retrieval, Top-K scoring | Retrieval Playground (4.6) | Chương 3, mục 4 |
| Cosine similarity | Cả hai retriever | Chương 3, mục 5 |
| Decoder-only Transformer + prompting | Hỏi–đáp RAG (4.7) | Chương 2, mục 11.3–11.4 |
| IR evaluation (P@K, R@K, MRR, MAP, NDCG) | Đánh giá (4.8) | Chương 3 |
| Clustering (mở rộng, không bắt buộc) | Phân tích corpus | Chương 5 |

## 3. Chi tiết lựa chọn

### 3.1. Tiền xử lý tiếng Việt

- **Word segmentation:** so sánh 3 nhóm trong RQ1: `none` / `underthesea` / `PyVi`.
- `underthesea`: chuẩn, tích hợp dễ; `PyVi`: nhẹ, nhanh. VnCoreNLP chỉ để mở rộng (chạy Java server, phức tạp hơn).
- **Stop-word:** dùng `stopwords.txt` đi kèm dataset Zalo; có bật/tắt trên giao diện kèm cảnh báo mất nghĩa pháp lý.
- **Stemming/lemmatization:** KHÔNG dùng mặc định — tiếng Việt không có stemmer chuẩn; nếu bật thì phải là thí nghiệm riêng.
- **Không ép lowercase** cho dense model (chưa kiểm chứng); TF-IDF có thể lowercase vì BOW không phân biệt hoa thường.

### 3.2. Embedding model (dense)

- **Ứng viên chính:** `keepitreal/vietnamese-sbert` (phổ biến cho tiếng Việt) hoặc `VoVanPhuc/simcse-VietNamese-phobert-base`.
- Quy ước: chọn 1 model tuần 2 sau khi thử nhanh trên ~50 câu hỏi dev; ghi chính xác tên + version vào config và báo cáo. Khóa tên model ở mục 6 ngay khi chốt.
- Max sequence length mặc định 256 token; article dài hơn sẽ được chia chunk con.

### 3.3. Vector database

- **ChromaDB persistent** (thư mục local) — đủ cho corpus ~7.000 văn bản / ~30.000 article trên một máy.
- Mỗi (model × preprocessing × chunking config) một collection riêng, không ghi đè.
- Milvus chỉ cân nhắc nếu mở rộng multi-machine (không dùng trong phạm vi học phần).

### 3.4. LLM sinh câu trả lời

- **Ollama + Qwen2.5 3B-Instruct** (tối thiểu 8GB RAM) hoặc **7B** nếu có GPU ≥ 8GB VRAM. Qwen hỗ trợ tiếng Việt tốt trong các model mở nhỏ.
- Phương án thay thế: PhoGPT 4B (thuần Việt nhưng theo sau về instruction-following).
- Prompt: grounded + trích dẫn `law_id/article_id` + chế độ "không đủ thông tin"; `temperature = 0.2`.
- Không fine-tune LLM trong phạm vi cốt lõi (chỉ là Giai đoạn 3 mở rộng).

### 3.5. Fine-tune (Giai đoạn 2.5)

- **Thư viện:** sentence-transformers, loss `MultipleNegativesRankingLoss`.
- **Dữ liệu:** bộ ba (query, positive article, hard negative) sinh từ train split của `train_question_answer.json`; hard negative mined từ Top-K sai của SBERT pretrained.
- **Chọn checkpoint:** theo Recall@10 / MRR trên dev, không theo training loss.
- Sau fine-tune: embed lại toàn corpus, collection mới, cache key thêm `checkpoint_version`.
- Không có GPU local → chạy Google Colab (T4 miễn phí), lưu checkpoint + vectors tải về.

### 3.6. Đánh giá

- **Retrieval:** tự triển khai metric trong `src/evaluation/` (P@K, R@K, Hit@K, MRR, MAP, NDCG) + unit test; không phụ thuộc thư viện ngoài để kiểm soát đúng định nghĩa.
- **Generation:** rubric tự đánh giá (faithfulness, answer relevance, context utilization) vì dataset Zalo không có reference answer; GPT-as-judge chỉ nếu được phép và phải ghi rõ trong báo cáo.

## 4. Dataset

| Hạng mục | Chốt |
|---|---|
| Dataset | Zalo AI 2021 Legal Text Retrieval (mirror Kaggle: `hariwh0/zaloai2021-legal-text-retrieval`) |
| Corpus | `legal_corpus.json` (~119 MB, ~7.000 văn bản, cấu trúc law → articles) |
| Relevance judgments | `train_question_answer.json` (~5.000 câu hỏi + điều relevant) |
| Đơn vị relevance | Article (`law_id` + `article_id`) |
| Split | 70/10/20 theo câu hỏi, seed 42, lưu `data/processed/question_split.json` |
| License | Kaggle ghi "Unknown" — cite cuộc thi Zalo AI Challenge 2021 trong báo cáo |

## 5. Cấu trúc thư viện Python chính (`requirements.txt` dự kiến)

```text
streamlit
chromadb
scikit-learn
sentence-transformers
torch
underthesea
pyvi
pandas
numpy
matplotlib
tqdm
```

Ollama cài riêng (system binary), không nằm trong pip.

## 6. Sổ theo dõi thay đổi quyết định

| Ngày | Quyết định | Lý do |
|---|---|---|
| 2026-08-24 | Chốt ban đầu toàn bộ stack theo các mục trên | Kế hoạch tuần 1 trong `KE_HOACH_THUC_HIEN.md` |
| | (các dòng tiếp theo ghi khi có thay đổi) | |
