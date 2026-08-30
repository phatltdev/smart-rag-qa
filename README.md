# Smart RAG QA

Hệ thống hỏi–đáp pháp luật tiếng Việt dựa trên **Retrieval-Augmented Generation (RAG)**. Ứng dụng cung cấp giao diện quản trị toàn bộ pipeline từ dữ liệu, tiền xử lý, chunking, lập chỉ mục, retrieval đến sinh câu trả lời; đồng thời có giao diện hỏi–đáp dành cho người dùng.

Dataset chính là **Zalo AI Challenge 2021 — Legal Text Retrieval**. Hệ thống hỗ trợ:

- TF-IDF + cosine similarity làm baseline;
- Sentence-BERT (`keepitreal/vietnamese-sbert`) + ChromaDB cho dense retrieval;
- `none`, `underthesea` và `PyVi` để so sánh phương pháp tách từ tiếng Việt;
- Ollama và mô hình mặc định `qwen2.5:7b` để sinh câu trả lời có ngữ cảnh;
- các chỉ số retrieval như Precision@K, Recall@K, Hit Rate@K, MRR, MAP và NDCG;
- lưu cấu hình, artifact tiền xử lý/chunking và kết quả thí nghiệm để hỗ trợ tái lập.

## Kiến trúc xử lý

```text
Zalo AI legal corpus
        │
        ▼
Ingestion & validation
        │
        ▼
Vietnamese preprocessing
        │
        ▼
Chunking (article / fixed / sentence)
        │
        ├───────────────┐
        ▼               ▼
TF-IDF index      SBERT embeddings
                        │
                        ▼
                     ChromaDB
        │               │
        └───────┬───────┘
                ▼
          Top-K retrieval
                ▼
       Prompt + retrieved context
                ▼
          Ollama local LLM
                ▼
     Answer + cited legal sources
```

Retrieval evaluation và generation evaluation là hai phần độc lập. Có thể xây dựng, chạy và đánh giá retriever mà không cần Ollama; Ollama chỉ bắt buộc khi sử dụng chức năng sinh câu trả lời.

## Cấu trúc project

```text
smart-rag-qa/
├── app.py                         # Entry point và giao diện Streamlit
├── requirements.txt               # Dependency chạy ứng dụng
├── configs/
│   ├── rag_llm_config.json         # Cấu hình RAG/LLM đang hoạt động
│   └── rag_llm_config_audit.jsonl  # Nhật ký thay đổi cấu hình
├── data/
│   ├── raw/                        # Dataset gốc, không chỉnh sửa trực tiếp
│   ├── interim/                    # Dữ liệu trung gian
│   ├── processed/                  # Artifact, manifest và train/dev/test split
│   └── DATASET_GUIDE.md            # Hướng dẫn lấy dataset
├── src/
│   ├── chunking/                   # Chiến lược chunk và artifact chunk
│   ├── embeddings/                 # Không gian mở rộng cho embedding
│   ├── evaluation/                 # Các metric đánh giá retrieval
│   ├── generation/                 # Ollama client, prompt và citation parsing
│   ├── ingestion/                  # Đọc, kiểm tra corpus và chia tập dữ liệu
│   ├── preprocessing/              # Chuẩn hóa, tách từ và artifact tiền xử lý
│   ├── retrieval/                  # TF-IDF và dense retriever
│   ├── ui/                         # Theme và giao diện legal portal
│   ├── config.py                   # Tham số thí nghiệm và đường dẫn dùng chung
│   └── system_config.py            # Kiểm tra/lưu cấu hình runtime
├── experiments/
│   └── evaluate_retrieval.py       # CLI chạy thí nghiệm RQ1/RQ2
├── results/
│   ├── metrics/                    # Kết quả metric dạng JSON
│   └── REPORT_RQ1.md               # Báo cáo kết quả thí nghiệm
├── scripts/
│   ├── download_dataset.py         # Tải dataset bằng Kaggle API
│   └── extract_pptx_text.py        # Trích xuất nội dung PowerPoint
├── tests/                           # Unit tests
└── documents/                       # Tài liệu môn học và báo cáo
```

Các thư mục `data/raw`, `data/interim`, `data/processed`, `chroma_store`, `models` và `cache` chứa dữ liệu hoặc artifact cục bộ nên không được commit lên Git theo cấu hình `.gitignore`.

## Yêu cầu môi trường

- Python **3.10 trở lên** (khuyến nghị Python 3.10 hoặc 3.11).
- Khoảng trống đĩa đủ cho dataset, Sentence-BERT cache và ChromaDB.
- Kết nối Internet ở lần đầu tải dataset và dense embedding model.
- Ollama nếu cần chức năng sinh câu trả lời RAG.
- GPU CUDA là tùy chọn; hệ thống vẫn chạy trên CPU nhưng xây dense index và suy luận LLM sẽ chậm hơn.

Các lệnh dưới đây được chạy từ thư mục gốc của project.

## Build và cài đặt

### 1. Tạo môi trường Python

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn script kích hoạt, có thể cho phép riêng phiên hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. Chuẩn bị dataset

Cách tự động bằng Kaggle API:

1. Tạo API token trong phần cài đặt tài khoản Kaggle.
2. Đặt `kaggle.json` tại `%USERPROFILE%\.kaggle\kaggle.json` trên Windows hoặc `~/.kaggle/kaggle.json` trên Linux/macOS.
3. Chạy:

```powershell
pip install kaggle
python scripts/download_dataset.py
```

Hoặc tải thủ công dataset và chép các tệp sau vào `data/raw/`:

```text
legal_corpus.json
train_question_answer.json
public_test_question.json
stopwords.txt
```

Xem hướng dẫn chi tiết tại [`data/DATASET_GUIDE.md`](data/DATASET_GUIDE.md).

### 3. Cài Ollama cho chế độ RAG đầy đủ

Cài Ollama từ trang chính thức, sau đó tải mô hình mặc định:

```powershell
ollama pull qwen2.5:7b
ollama serve
```

`ollama serve` cần tiếp tục chạy trong một terminal riêng. Ứng dụng kết nối mặc định đến `http://localhost:11434`. Có thể bỏ qua bước này nếu chỉ thử tiền xử lý, chunking, TF-IDF hoặc đánh giá retrieval.

### 4. Khởi động ứng dụng

```powershell
streamlit run app.py
```

Streamlit thường mở ứng dụng tại `http://localhost:8501`. Nếu không tự mở trình duyệt, truy cập địa chỉ được in trong terminal.

## Khởi tạo pipeline lần đầu

Sau khi mở giao diện quản trị, thực hiện tuần tự:

1. **Quản lý dữ liệu**: kiểm tra corpus và tạo train/dev/test split nếu chưa có.
2. **Tiền xử lý**: chọn cấu hình Unicode normalization, tách từ và stop-word; chạy xử lý toàn bộ dataset để tạo artifact trong `data/processed/`.
3. **Chunking**: chọn artifact tiền xử lý, cấu hình chiến lược chunk rồi bấm **Generate & Save Chunks**.
4. **Lập chỉ mục và embedding**:
   - bấm **Xây TF-IDF index** để dùng baseline; hoặc
   - bấm **Xây dense index** để tải Sentence-BERT, tạo embedding và lưu collection trong `chroma_store/`.
5. **Cấu hình**: chọn retriever, Top-K, phương pháp tách từ, model Ollama, temperature và max tokens.
6. **Retrieval Playground**: kiểm tra các tài liệu được truy hồi trước khi chạy sinh câu trả lời.
7. **Chat với LLM (RAG)** hoặc **Legal Portal**: đặt câu hỏi và kiểm tra câu trả lời cùng nguồn luật được trích dẫn.

TF-IDF index hiện được xây trong bộ nhớ và Streamlit cache; sau khi khởi động lại ứng dụng, hãy xây lại khi cần. Dense index được ChromaDB lưu bền vững trong `chroma_store/`.

## Chạy kiểm thử

`pytest` là dependency phục vụ phát triển và hiện không nằm trong `requirements.txt`:

```powershell
pip install pytest
python -m pytest tests -v
```

## Chạy thí nghiệm retrieval

Tạo train/dev/test split trong giao diện trước khi chạy, vì CLI cần tệp `data/processed/question_split.json`.

So sánh phương pháp tách từ (RQ1):

```powershell
python -m experiments.evaluate_retrieval --rq1 --retriever tfidf --split dev
```

So sánh TF-IDF và dense retrieval (RQ2):

```powershell
python -m experiments.evaluate_retrieval --rq2 --split dev
```

Chạy nhanh trên một số câu hỏi để kiểm tra pipeline:

```powershell
python -m experiments.evaluate_retrieval --rq1 --retriever tfidf --split dev --max-questions 20
```

Kết quả được lưu tại `results/metrics/<experiment_id>.json`. Thí nghiệm có dense retrieval yêu cầu collection tương ứng đã được xây ở màn hình **Lập chỉ mục và embedding**.

## Tiện ích trích xuất PowerPoint

Script này cần dependency tùy chọn `python-pptx`:

```powershell
pip install python-pptx
python scripts/extract_pptx_text.py documents scripts/extracted
```

## Cấu hình mặc định và khả năng tái lập

- Random seed: `42`.
- Dense embedding model: `keepitreal/vietnamese-sbert`.
- LLM: `qwen2.5:7b` qua Ollama.
- Retriever: `dense`.
- Top-K: `5`.
- Temperature: `0.2`.
- Max generated tokens: `512`.
- Split dữ liệu: train/dev/test = `70%/10%/20%`.

Cấu hình đang hoạt động nằm trong `configs/rag_llm_config.json`; lịch sử thay đổi được ghi nối tiếp vào `configs/rag_llm_config_audit.jsonl`. Artifact tiền xử lý và chunking có manifest kèm dataset version/config ID để tránh dùng nhầm dữ liệu giữa các thí nghiệm.

## Lỗi thường gặp

- **Không tìm thấy `legal_corpus.json`**: chạy script tải dataset hoặc chép đủ bốn tệp bắt buộc vào `data/raw/`.
- **Không có chunk artifact**: hoàn thành bước Tiền xử lý, sau đó vào Chunking và chọn **Generate & Save Chunks**.
- **Dense collection not found**: xây dense index với đúng preprocessing/chunking config trước khi chat hoặc chạy RQ2.
- **Ollama chưa kết nối**: kiểm tra `ollama serve`, sau đó chạy `ollama list` để xác nhận model đã được tải.
- **Lần đầu chạy dense chậm**: Sentence-BERT phải được tải về và toàn bộ corpus phải được embed; các lần sau ChromaDB sẽ tái sử dụng collection tương ứng.
- **Hết RAM/VRAM**: giảm batch size khi xây dense index hoặc chọn CPU trong giao diện.

## Lưu ý học thuật và dữ liệu

Dataset trên Kaggle được ghi nhận license là `Unknown`; cần kiểm tra điều kiện sử dụng trước khi phân phối lại. Không chỉnh sửa trực tiếp dữ liệu trong `data/raw/`, không đưa dataset lớn, model cache, ChromaDB store hoặc thông tin xác thực Kaggle lên Git. Các kết quả trong `results/` là kết quả thực nghiệm của từng cấu hình cụ thể, không nên khái quát thành ưu thế của một phương pháp nếu chưa có đủ lần chạy và phân tích thống kê.
