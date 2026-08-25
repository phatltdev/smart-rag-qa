# Hướng dẫn tải dataset Zalo AI 2021 Legal Text Retrieval

Dataset: https://www.kaggle.com/datasets/hariwh0/zaloai2021-legal-text-retrieval

## Cách 1 — Tự động bằng Kaggle CLI (khuyến nghị)

1. Tạo tài khoản Kaggle → Account Settings → Create New API Token → tải `kaggle.json`.
2. Đặt `kaggle.json` vào thư mục (Windows): `%USERPROFILE%\.kaggle\`
3. Chạy:

```powershell
pip install kaggle
python scripts/download_dataset.py
```

Script sẽ tải và giải nén vào `data/raw/`, chỉ giữ các file cần thiết.

## Cách 2 — Tải tay

1. Vào link Kaggle trên, nhấn Download (cần đăng nhập).
2. Giải nén và copy các file sau vào `data/raw/`:
   - `legal_corpus.json` (~119 MB)
   - `train_question_answer.json`
   - `public_test_question.json`
   - `stopwords.txt`

## File bắt buộc phải có sau khi tải

| File | Dùng để |
|---|---|
| `legal_corpus.json` | Corpus (Document/Chunk) |
| `train_question_answer.json` | Relevance judgments + dữ liệu fine-tune |
| `public_test_question.json` | Câu hỏi demo |
| `stopwords.txt` | Phòng tiền xử lý |

Ghi chú license: Kaggle ghi "Unknown". Trong báo cáo cite cuộc thi **Zalo AI Challenge 2021 — Legal Text Retrieval**.
