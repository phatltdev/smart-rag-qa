# Báo cáo thực nghiệm RQ1 — Ảnh hưởng của tách từ đến Retrieval (Tuần 4)

- **Ngày:** 2026-08-25
- **Dataset:** Zalo AI 2021 Legal Text Retrieval (`zalo2021-fae7d632ad`)
- **Split:** dev (100 câu hỏi đầu tiên — smoke test)
- **Retriever:** TF-IDF (ngram 1–2, sublinear_tf) + cosine similarity
- **Chunking:** strategy=article (1 điều luật / chunk, 60.152 chunks)
- **Preprocessing (cố định):** NFC, whitespace normalization, giữ hoa/thường, không bỏ stopwords
- **Biến duy nhất:** `word_segmentation` ∈ {none, underthesea, pyvi}
- **Random seed:** 42 · Đánh giá: Hit@K, Recall@K, Precision@K, MRR

## Kết quả

| Tách từ | MRR | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@5 | P@5 | Thời gian (s) |
|---|---|---|---|---|---|---|---|---|
| none | 0.4079 | 0.28 | 0.48 | **0.63** | **0.72** | 0.625 | 0.126 | 93 |
| underthesea | 0.4172 | 0.29 | 0.49 | 0.60 | 0.71 | 0.595 | 0.120 | 741 |
| pyvi | **0.4260** | **0.30** | **0.50** | 0.59 | 0.70 | 0.585 | 0.118 | 273 |

File chi tiết: `results/metrics/retrieval_tfidf_seg-{none,underthesea,pyvi}_dev.json`

## Phân tích

### Quan sát
1. **Chênh lệch rất nhỏ** giữa 3 chiến lược trên mọi metric (≤ 0.04 tuyệt đối).
2. Tách từ **không cải thiện Hit@5/Hit@10** — thậm chí slightly thấp hơn none
   (0.63 → 0.60/0.59). Ngược lại, **MRR và Hit@1 lại nhích lên nhẹ**
   (0.408 → 0.417/0.426).
3. underthesea chậm nhất (741s build index cho 60k chunks), pyvi trung bình, none nhanh nhất.

### Diễn giải
- TF-IDF với ngram_range=(1,2) vốn đã capture được cụm từ gần đúng ("thời hiệu",
  "khiếu kiện"...) nên tách từ không mang thông tin mới đáng kể — kết quả nhất
  quán với literature cho kho ngữ liệu luật (vốn nhiều thuật ngữ ghép).
- Việc MRR/Hit@1 nhích lên gợi ý tách từ giúp **ranking đúng hơn ở top đầu**
  nhưng có thể đẩy một số điều đúng ra khỏi top 5–10 (recall thấp hơn nhẹ).

### Hạn chế (bắt buộc ghi nhận)
- **n = 100 câu hỏi** (dev có 318) — smoke test, chưa đủ để kết luận;
  chênh lệch 0.02–0.03 nằm trong biên độ nhiễu. Cần: (a) chạy đủ 318 câu,
  (b) lặp lại trên test, và (c) kiểm định thống kê (paired bootstrap/test)
  trước khi phát biểu "có/không khác biệt".
- Chỉ mới đánh giá với TF-IDF; cần lặp lại với dense SBERT (RQ2) vì embedding
  model có tokenizer riêng — tác động của tách từ có thể khác.

## Kết luận sơ bộ
Trên smoke test 100 câu dev với TF-IDF: **tách từ không tạo khác biệt có ý nghĩa
thực hành**; lựa chọn `none` là hợp lý về chi phí (nhanh nhất, đơn giản nhất).
Chờ kết quả full dev + test + RQ2 để chốt kết luận cuối.

## Bước tiếp theo
1. Chạy RQ1 đầy đủ trên 318 câu dev + 642 câu test (đã có sẵn harness).
2. RQ2: xây dense index (màn 4.5) rồi chạy `--rq2`.
3. Thêm paired bootstrap test cho MRR/Hit@5.
