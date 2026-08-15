Dựa trên file yêu cầu, project là **hệ thống hỏi–đáp RAG tiếng Việt**, với các thành phần chính gồm tiền xử lý tiếng Việt, Sentence-BERT, Vector Database, Cosine Similarity, LLM và Web Application.  Hệ thống phải xây dựng trên kho ngữ liệu chuyên ngành tiếng Việt. 

Phần `AGENTS.md` nên đặc biệt khóa chặt pipeline mà đề bài yêu cầu: `underthesea/PyVi → Sentence-BERT → ChromaDB/Milvus → Cosine Similarity → LLM + Prompt Engineering`. 

Bạn đã chọn không chuyển sang Codex nên tôi chưa tạo file tải xuống. Nội dung tôi đề xuất cho `AGENTS.md` là:

# AGENTS.md

## 1. Thông tin dự án

- **Trình độ:** Thạc sĩ
- **Môn học:** Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP)
- **Đề tài:** Hệ thống Hỏi - Đáp thông minh dựa trên kỹ thuật RAG (Retrieval-Augmented Generation)
- **Ngôn ngữ chính:** Tiếng Việt
- **Loại dự án:** NLP / Information Retrieval / Large Language Model / RAG / Web Application

---

# 2. Vai trò của AI Agent

Trong project này, AI Agent đóng vai trò:

- Trợ lý nghiên cứu NLP.
- Trợ lý lập trình Python.
- Trợ lý xây dựng hệ thống RAG.
- Trợ lý thiết kế và thực hiện thực nghiệm.
- Trợ lý phân tích kết quả.
- Trợ lý viết báo cáo khoa học ở trình độ Thạc sĩ.
- Trợ lý kiểm tra phương pháp nghiên cứu và tính tái lập của thực nghiệm.

Mọi giải pháp phải phù hợp với **trình độ học viên Thạc sĩ**, không chỉ dừng ở mức xây dựng ứng dụng hoạt động được mà còn phải:

- Có cơ sở lý thuyết.
- Có phương pháp thực nghiệm.
- Có baseline để so sánh.
- Có metric đánh giá.
- Có phân tích kết quả.
- Có khả năng tái lập.
- Có giải thích cho các quyết định kỹ thuật.

---

# 3. Mục tiêu hệ thống

Xây dựng Web Application cho phép người dùng đặt câu hỏi bằng tiếng Việt và nhận câu trả lời dựa trên một kho ngữ liệu tri thức chuyên ngành tiếng Việt.

Kiến trúc tổng quát:

```text
Vietnamese Documents
        ↓
Data Cleaning
        ↓
Vietnamese Word Segmentation
        ↓
Document Chunking
        ↓
Sentence-BERT
        ↓
Vector Embeddings
        ↓
Vector Database
        ↓
User Question
        ↓
Question Preprocessing
        ↓
Question Embedding
        ↓
Cosine Similarity Search
        ↓
Top-K Relevant Contexts
        ↓
Prompt Construction
        ↓
LLM
        ↓
Generated Answer
        ↓
Answer + Retrieved Sources
```

---

# 4. Yêu cầu kỹ thuật bắt buộc

## 4.1 Dataset

Ưu tiên sử dụng một trong các nguồn dữ liệu:

- ViLegalQA.
- ZaloAI 2021 Legal Text Retrieval.
- Dữ liệu quy chế học vụ của trường.

Không tự ý thay dataset chính nếu chưa có yêu cầu.

Dữ liệu gốc phải được bảo toàn.

Khuyến nghị:

```text
data/
├── raw/
├── processed/
├── chunks/
└── evaluation/
```

Không chỉnh sửa trực tiếp dữ liệu trong `raw/`.

---

# 5. Tiền xử lý tiếng Việt

Sử dụng thư viện tách từ tiếng Việt:

- `underthesea`
- `PyVi`

Có thể triển khai nhiều pipeline tiền xử lý để thực nghiệm.

Ví dụ:

```text
Pipeline A
Raw text
→ Normalize
→ No word segmentation

Pipeline B
Raw text
→ Normalize
→ underthesea

Pipeline C
Raw text
→ Normalize
→ PyVi
```

Nếu mục tiêu nghiên cứu liên quan đến ảnh hưởng của tách từ tiếng Việt, phải giữ các thành phần khác cố định để đảm bảo so sánh công bằng.

Không được mặc định rằng `underthesea` hoặc `PyVi` tốt hơn nếu chưa có kết quả thực nghiệm.

---

# 6. Chunking

Document Chunking phải được triển khai thành module riêng.

Các tham số cần cấu hình:

```python
chunk_size
chunk_overlap
```

Không hard-code trong nhiều file khác nhau.

Có thể thực nghiệm:

```text
chunk_size = 128
chunk_size = 256
chunk_size = 512
```

Nếu thay đổi chunk size, phải ghi nhận vào cấu hình thực nghiệm.

---

# 7. Embedding

Sử dụng Sentence-BERT hoặc mô hình embedding phù hợp với tiếng Việt.

Mô hình ưu tiên theo yêu cầu đề tài:

```text
vietnamese-bi-encoder
```

Embedding pipeline phải tách biệt khỏi retrieval pipeline.

Ví dụ:

```python
def encode_documents(documents):
    ...

def encode_query(query):
    ...
```

Không tạo lại embedding toàn bộ corpus khi không cần thiết.

Nên cache embeddings sau khi tạo.

---

# 8. Vector Database

Sử dụng một trong:

- ChromaDB
- Milvus

Vector Database phải lưu tối thiểu:

```text
id
embedding
document_id
chunk_id
text
metadata
```

Metadata nên cho phép truy ngược về tài liệu gốc.

Ví dụ:

```json
{
  "document_id": "LAW_001",
  "chunk_id": "LAW_001_003",
  "source": "document_name",
  "text": "..."
}
```

---

# 9. Retrieval

Sử dụng **Cosine Similarity** để tìm các đoạn văn bản có ngữ nghĩa gần với câu hỏi.

Pipeline:

```text
Question
↓
Preprocessing
↓
Embedding
↓
Cosine Similarity
↓
Ranking
↓
Top-K Contexts
```

Hỗ trợ cấu hình:

```python
TOP_K = 5
```

Không hard-code Top-K trực tiếp trong retrieval logic.

Nên đánh giá nhiều giá trị:

```text
Top-1
Top-3
Top-5
Top-10
```

---

# 10. Retrieval Evaluation

Các metric có thể sử dụng:

- Precision@K
- Recall@K
- Hit Rate@K
- MRR
- MAP
- NDCG

Ít nhất phải có metric phù hợp để đánh giá khả năng retrieval.

Phải phân biệt:

```text
Retrieval Evaluation
```

và:

```text
Answer Generation Evaluation
```

Không dùng metric của phần Generation để kết luận trực tiếp chất lượng Retriever.

---

# 11. LLM

Sử dụng mô hình LLM mã nguồn mở phù hợp.

Các model có thể xem xét:

- PhoGPT
- Vistral
- Llama-3-8B-Instruct

LLM phải sử dụng context được Retriever cung cấp để sinh câu trả lời.

Không thiết kế hệ thống chỉ gửi câu hỏi trực tiếp cho LLM mà bỏ qua Retrieval.

---

# 12. Prompt Engineering

Prompt phải yêu cầu mô hình ưu tiên thông tin trong retrieved context.

Ví dụ về nguyên tắc:

```text
SYSTEM:
Bạn là hệ thống hỏi đáp dựa trên tài liệu.

Chỉ sử dụng thông tin trong CONTEXT để trả lời.

Nếu CONTEXT không chứa đủ thông tin, hãy nói rằng
không tìm thấy đủ thông tin trong tài liệu.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
```

Mục tiêu là giảm hallucination.

Không yêu cầu LLM tự tạo thông tin khi context không cung cấp đủ bằng chứng.

---

# 13. Web Application

Ứng dụng phải có giao diện trực quan.

Có thể sử dụng:

### Option 1

```text
Streamlit
```

### Option 2

```text
Gradio
```

### Option 3

```text
FastAPI
+
React / Vue
```

Ưu tiên kiến trúc đơn giản, phù hợp phạm vi môn học nhưng vẫn có khả năng trình diễn đầy đủ pipeline RAG.

---

# 14. Chức năng Web

Web Application tối thiểu nên hỗ trợ:

### Hỏi đáp

- Nhập câu hỏi.
- Gửi câu hỏi.
- Hiển thị câu trả lời.

### Retrieval information

Nên hiển thị:

- Retrieved context.
- Similarity score.
- Top-K.
- Nguồn tài liệu.

Ví dụ:

```text
Question:
Điều kiện để sinh viên được xét tốt nghiệp là gì?

Answer:
...

Sources:

[1] Quy chế đào tạo - Điều ...
Similarity: 0.87

[2] Quy định sinh viên - Điều ...
Similarity: 0.82
```

Việc hiển thị nguồn giúp kiểm tra khả năng grounding của hệ thống.

---

# 15. Fine-tuning / Training

Theo yêu cầu môn học cần có source code huấn luyện hoặc fine-tune model.

Ưu tiên:

```text
Jupyter Notebook
```

hoặc:

```text
Google Colab
```

Notebook cần thể hiện rõ:

```text
1. Environment Setup
2. Dataset Loading
3. Data Inspection
4. Preprocessing
5. Train / Validation / Test Split
6. Model Initialization
7. Training / Fine-tuning
8. Evaluation
9. Save Model
10. Results
```

Không tạo notebook chỉ chứa code mà không có giải thích.

---

# 16. Thiết kế thực nghiệm

Mỗi experiment phải xác định:

```text
Research Question
        ↓
Hypothesis
        ↓
Dataset
        ↓
Preprocessing
        ↓
Model
        ↓
Configuration
        ↓
Evaluation
        ↓
Results
        ↓
Analysis
        ↓
Conclusion
```

Ví dụ nghiên cứu ảnh hưởng của Vietnamese Word Segmentation:

```text
EXP001
No Word Segmentation

EXP002
underthesea

EXP003
PyVi
```

Giữ cố định:

```text
Dataset
Embedding Model
Chunk Size
Chunk Overlap
Vector Database
Similarity Metric
Top-K
Evaluation Dataset
```

Chỉ thay đổi:

```text
Word Segmentation Method
```

---

# 17. Baseline

Mọi thực nghiệm quan trọng phải có baseline phù hợp.

Ví dụ Retrieval:

```text
Baseline:
TF-IDF + Cosine Similarity

Proposed:
Sentence-BERT + Cosine Similarity
```

Ví dụ Word Segmentation:

```text
Baseline:
No Word Segmentation

Comparison:
underthesea
PyVi
```

Không kết luận một phương pháp tốt hơn nếu không có baseline hoặc phương pháp so sánh.

---

# 18. Generation Evaluation

Có thể sử dụng:

- Exact Match
- F1-score
- ROUGE
- BLEU
- BERTScore
- Semantic Similarity
- Human Evaluation

Đối với RAG nên xem xét:

- Answer Relevance
- Faithfulness
- Context Relevance
- Context Recall

Không chỉ đánh giá câu trả lời dựa trên cảm nhận cá nhân.

---

# 19. Reproducibility

Mọi experiment cần ghi lại:

```text
experiment_id
dataset
dataset_version
random_seed
segmentation_method
embedding_model
chunk_size
chunk_overlap
top_k
llm
prompt_version
evaluation_metrics
results
```

Random seed mặc định nếu không có yêu cầu khác:

```python
RANDOM_SEED = 42
```

Kết quả thực nghiệm không được ghi đè tùy tiện.

---

# 20. Experiment Tracking

Sử dụng convention:

```text
EXP001_baseline
EXP002_underthesea
EXP003_pyvi
EXP004_topk
EXP005_chunk_size
```

Cấu trúc:

```text
experiments/
├── EXP001_baseline/
├── EXP002_underthesea/
├── EXP003_pyvi/
└── ...
```

Mỗi experiment nên lưu:

```text
config.json
metrics.json
notes.md
```

Nếu có biểu đồ:

```text
figures/
```

---

# 21. Error Analysis

Không chỉ báo cáo metric tổng.

Phải kiểm tra các trường hợp hệ thống trả lời sai.

Phân loại lỗi có thể gồm:

```text
Preprocessing Error
Word Segmentation Error
Chunking Error
Embedding Error
Retrieval Error
Ranking Error
Insufficient Context
Prompt Error
LLM Hallucination
Dataset Error
```

Error Analysis phải được sử dụng để đề xuất hướng cải tiến.

---

# 22. Cấu trúc project

Khuyến nghị:

```text
rag-vietnamese-qa/
│
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── chunks/
│   └── evaluation/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── normalization.py
│   │   └── word_segmentation.py
│   │
│   ├── chunking/
│   │   └── chunker.py
│   │
│   ├── embeddings/
│   │   └── sentence_embedding.py
│   │
│   ├── vectorstore/
│   │   └── vector_store.py
│   │
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   ├── generation/
│   │   ├── llm.py
│   │   └── prompt.py
│   │
│   ├── evaluation/
│   │   ├── retrieval_metrics.py
│   │   └── generation_metrics.py
│   │
│   └── utils/
│
├── app/
│   └── app.py
│
├── experiments/
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── tables/
│
├── tests/
│
├── reports/
│
└── references/
```

Không bắt buộc tạo tất cả thư mục ngay từ đầu nếu chưa sử dụng.

---

# 23. Coding Convention

Ngôn ngữ chính:

```text
Python 3.x
```

Tuân thủ:

- PEP 8.
- Function nhỏ và có trách nhiệm rõ ràng.
- Hạn chế duplicate code.
- Sử dụng type hints khi phù hợp.
- Docstring cho function quan trọng.

Tên:

```python
document_chunks
embedding_model
vector_store
retrieve_context()
generate_answer()
evaluate_retrieval()
```

Không dùng:

```python
a
b
x1
test123
abc
```

cho các biến nghiệp vụ quan trọng.

---

# 24. Configuration

Không hard-code các tham số thực nghiệm.

Ưu tiên:

```text
configs/
└── default.yaml
```

Ví dụ:

```yaml
preprocessing:
  word_segmentation: underthesea

chunking:
  chunk_size: 256
  chunk_overlap: 50

embedding:
  model: vietnamese-bi-encoder

retrieval:
  similarity: cosine
  top_k: 5

vector_database:
  type: chromadb

experiment:
  random_seed: 42
```

---

# 25. Academic Integrity

Đây là project ở trình độ **Thạc sĩ**.

Tuyệt đối không:

- Bịa dataset.
- Bịa kết quả.
- Bịa Accuracy/F1/Recall.
- Bịa paper.
- Bịa DOI.
- Bịa citation.
- Bịa thông số model.
- Khẳng định đã chạy experiment khi chưa thực sự chạy.
- Khẳng định code hoạt động nếu chưa kiểm tra.

Nếu chưa có kết quả:

```text
Chưa có kết quả thực nghiệm.
```

Không tự tạo một con số giả để hoàn thiện báo cáo.

---

# 26. Nguồn tài liệu

Ưu tiên:

1. Original Research Paper.
2. ACL Anthology.
3. IEEE.
4. ACM.
5. Springer.
6. ScienceDirect.
7. Official documentation.
8. Official GitHub repository.
9. Hugging Face model documentation.

Khi sử dụng paper phải xác minh:

```text
Title
Authors
Year
Conference / Journal
DOI / URL
```

nếu các thông tin này tồn tại.

---

# 27. Báo cáo môn học

Báo cáo phải bao gồm các nội dung quan trọng:

```text
1. Giới thiệu

2. Cơ sở lý thuyết
   - NLP
   - Transformer
   - BERT
   - Sentence-BERT
   - LLM
   - RAG
   - Vector Embedding
   - Vector Database
   - Cosine Similarity

3. Phương pháp đề xuất

4. Tiền xử lý tiếng Việt
   - underthesea
   - PyVi

5. Kiến trúc hệ thống

6. Dataset

7. Thiết kế thực nghiệm

8. Kết quả

9. Phân tích và thảo luận

10. Hạn chế

11. Kết luận

12. Hướng phát triển

13. Tài liệu tham khảo
```

---

# 28. Biểu đồ và bảng

Khi báo cáo kết quả:

Ưu tiên cả:

```text
Table
+
Chart
+
Interpretation
```

Không chỉ đưa biểu đồ mà không giải thích.

Ví dụ:

```text
Bảng X. So sánh Recall@K

Method        R@1    R@3    R@5    R@10
Baseline      ...    ...    ...    ...
underthesea   ...    ...    ...    ...
PyVi          ...    ...    ...    ...
```

Sau bảng phải có nhận xét dựa trên số liệu thực tế.

---

# 29. Git

Branch convention:

```text
feature/preprocessing
feature/rag-pipeline
feature/web-app

experiment/underthesea
experiment/pyvi
experiment/top-k

fix/retrieval

docs/report
```

Commit:

```text
feat: implement Vietnamese text preprocessing

feat: add Sentence-BERT embedding pipeline

feat: implement ChromaDB vector store

feat: add cosine similarity retrieval

feat: integrate LLM generation

exp: compare Vietnamese word segmentation methods

exp: evaluate retrieval at top-k

docs: update experimental methodology
```

---

# 30. README

`README.md` tối thiểu phải có:

```text
Project Introduction
Architecture
Requirements
Installation
Dataset
Configuration
Run Preprocessing
Build Vector Database
Run Experiments
Run Web Application
Evaluation
Project Structure
References
```

Người khác phải có khả năng đọc README và biết cách chạy project.

---

# 31. Definition of Done

Trước khi đánh dấu một task hoàn thành:

- [ ] Code đáp ứng đúng yêu cầu.
- [ ] Code có thể chạy.
- [ ] Không có import/path sai.
- [ ] Không hard-code đường dẫn máy cá nhân.
- [ ] Không commit API key/password.
- [ ] Dataset gốc không bị chỉnh sửa.
- [ ] Parameters được lưu trong configuration.
- [ ] Experiment có ID.
- [ ] Random seed được lưu.
- [ ] Metric được tính đúng.
- [ ] Không có số liệu giả.
- [ ] Kết luận dựa trên kết quả thực nghiệm.
- [ ] Citation có thể kiểm chứng.
- [ ] README được cập nhật nếu cách chạy thay đổi.
- [ ] Code và báo cáo nhất quán.
- [ ] Các hạn chế được ghi nhận.

---

# 32. Nguyên tắc làm việc của Agent

Khi nhận một yêu cầu mới:

```text
Understand
    ↓
Inspect Existing Code/Data
    ↓
Identify Requirement
    ↓
Propose Approach
    ↓
Implement
    ↓
Test
    ↓
Evaluate
    ↓
Document
```

Không tự ý sửa các module không liên quan.

Nếu phát hiện yêu cầu có thể ảnh hưởng kết quả nghiên cứu, phải giải thích trước khi thay đổi.

---

# 33. Thứ tự ưu tiên

Trong toàn bộ project:

```text
Correctness
        ↓
Academic Integrity
        ↓
Experimental Validity
        ↓
Reproducibility
        ↓
Research Quality
        ↓
Code Quality
        ↓
Performance
        ↓
Convenience
```

Mục tiêu cuối cùng không chỉ là:

> "Xây dựng được một chatbot."

Mà phải là:

> **Xây dựng, thực nghiệm và đánh giá một hệ thống hỏi–đáp tiếng Việt dựa trên Retrieval-Augmented Generation theo phương pháp có thể giải thích, kiểm chứng và tái lập, phù hợp với yêu cầu học thuật ở trình độ Thạc sĩ môn Xử lý ngôn ngữ tự nhiên.**