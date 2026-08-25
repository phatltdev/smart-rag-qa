# 📜 DEMO — Hướng dẫn chạy demo hệ thống Smart RAG QA

> Cheat sheet đầy đủ các lệnh từ khởi tạo đến khi stop project.
> Dataset: Zalo AI 2021 — Legal Text Retrieval | LLM: Ollama + Qwen2.5:7b

---

## Giai đoạn 0 — Khởi tạo project (CHỈ làm 1 lần đầu — ✅ đã xong)

```powershell
# 0.1. Vào thư mục project
cd D:\htttk32\smart-rag-qa

# 0.2. Tạo môi trường ảo Python
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 0.3. Cài dependencies
pip install streamlit torch scikit-learn chromadb sentence-transformers underthesea pyvi matplotlib kaggle pandas numpy requests

# 0.4. Auth Kaggle (để tải dataset)
kaggle auth login

# 0.5. Tải dataset Zalo AI 2021 Legal
python scripts/download_dataset.py
```

## Giai đoạn 1 — Cài Ollama + model LLM (✅ đã xong)

```powershell
# 1.1. Cài Ollama (chạy installer từ https://ollama.com/download)

# 1.2. Tải model Qwen2.5 7B (~4.7 GB)
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen2.5:7b
```

## Giai đoạn 2 — Build index (✅ đã xong, tự động qua UI)

```powershell
# Không cần lệnh riêng — vào app:
#   Màn 4.5 → nhấn "Xây TF-IDF index"  (kết quả: models/tfidf_retriever.pkl, 341 MB)
#   (Tuỳ chọn) dense SBERT index cũng build qua màn 4.5
```

---

## Giai đoạn 3 — KHỞI ĐỘNG ĐỂ DEMO (làm mỗi lần chạy) ⭐

```powershell
# 3.1. Khởi động Ollama server
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList 'serve'

# 3.2. Kiểm tra Ollama + model sẵn sàng
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list     # phải thấy qwen2.5:7b

# 3.3. Khởi động app
cd D:\htttk32\smart-rag-qa
.\.venv\Scripts\Activate.ps1
streamlit run app.py                                       # mở http://localhost:8501
```

---

## Giai đoạn 4 — Kịch bản DEMO (7 màn trong browser)

| Thứ tự | Màn | Demo gì |
|---|---|---|
| 1 | **4.1 Tổng quan** | Số liệu: 3.271 laws / 60.152 articles / 3.196 câu hỏi |
| 2 | **4.2 Quản lý dữ liệu** | Dataset `zalo2021-fae7d632ad`, split 2.236/318/642 (seed 42) |
| 3 | **4.3 Phòng tiền xử lý** | So sánh none / underthesea / pyvi trên văn bản mẫu |
| 4 | **4.4 Chunking Lab** | Chiến lược article, chunk size/overlap |
| 5 | **4.5 Lập chỉ mục** | TF-IDF index đã build sẵn |
| 6 | **4.6 Retrieval Playground** | So TF-IDF vs dense, query: *"thời hiệu khiếu kiện hành chính"* |
| 7 | **4.7 Chat với LLM (RAG)** | Retriever=tfidf, top_k=5, model=qwen2.5:7b |

### Câu hỏi demo gợi ý cho màn 4.7

- ✅ **Trong phạm vi:** *"Thời hiệu khiếu kiện hành chính là bao lâu?"*
  → trả lời kèm trích dẫn `[Điều nguồn n]` đúng law_id/article_id
- ✅ **Ngoài phạm vi:** *"Cách nấu phở?"*
  → LLM phải chối "Các điều luật được cung cấp không đủ thông tin..." (demo chống hallucination)

---

## Giai đoạn 5 — STOP project

```powershell
# 5.1. Dừng Streamlit: nhấn Ctrl+C trong terminal đang chạy app

# 5.2. Dừng Ollama: right-click biểu tượng Ollama trong system tray → Quit
#      Hoặc bằng lệnh:
Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue

# 5.3. (Tuỳ chọn) Thoát môi trường ảo
deactivate

# 5.4. Đóng tab browser localhost:8501
```

---

## ⚠️ Troubleshooting nhanh

| Lỗi | Xử lý |
|---|---|
| `ollama: not recognized` | Terminal cũ mở trước khi cài → mở terminal **mới**, hoặc dùng đường dẫn đầy đủ `& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" ...` |
| Màn 4.7 báo "Không kết nối được Ollama" | Chưa làm bước 3.1 |
| Pull model lỗi `unexpected EOF` | Chạy lại lệnh pull (resume được) |
| LLM trả lời chậm | 7B trên CPU = 1–5 phút/câu, bình thường; giảm max_tokens nếu cần |
| `No module named streamlit` | Sai môi trường — chắc chắn đã `Activate.ps1` của `.venv` |
