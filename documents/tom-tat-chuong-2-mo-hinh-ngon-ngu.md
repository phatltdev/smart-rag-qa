# Tóm tắt Chương 2: Mô hình Ngôn ngữ (Language Models)

> Nguồn: `Chapter2-LanguageModels.pptx` (84 slides) — Giảng viên: Trương Quốc Định, Khoa Hệ thống Thông tin, Trường CNTT & TT.

---

## 1. Động lực & Vai trò của mô hình ngôn ngữ

- Nhiều tác vụ NLP có đầu ra là ngôn ngữ tự nhiên: **dịch máy, nhận dạng giọng nói, sinh ngôn ngữ, kiểm tra chính tả**.
- **Language Model (LM)** định nghĩa **phân phối xác suất** trên các chuỗi/câu ngôn ngữ tự nhiên:
  - Nếu $P_{LM}(A) > P_{LM}(B)$ → chọn câu A thay vì B.
- Dạng $P(w_t \mid \text{context})$ cũng dùng để **sinh văn bản**: tại mỗi bước, lấy mẫu token từ phân phối xác suất của LM trên các token kế tiếp.

### LM đóng vai trò:
1. **Trọng tài ngữ pháp**: ưu tiên "The boy runs." hơn "The boy run."
2. **Trọng tài tính hợp lý ngữ nghĩa**: ưu tiên "The woman spoke." hơn "The sandwich spoke."
3. **Thực thi nhất quán phong cách**: hội thoại hỏi–đáp phải đồng nhất văn phong.
4. **Kho tri thức (có thể)**: LM "biết" các fact như "Barack Obama was the 44th President of the United States".

### Vì sao cần LM?
- **Dịch máy**: $p(\text{strong winds}) > p(\text{large winds})$
- **Sửa chính tả**: $p(\text{about fifteen minutes from}) > p(\text{about fifteen minuets from})$
- **Nhận dạng giọng nói**: $p(\text{I saw a van}) \gg p(\text{eyes awe of an})$
- Tóm tắt, hỏi đáp, nhận dạng chữ viết, OCR...

## 2. Cơ sở Lý thuyết Xác suất

- **Quy tắc chuỗi (chain rule)**: $P(X, Y) = P(X \mid Y) \cdot P(Y)$ — mở rộng cho chuỗi dài chính là cơ sở phân rã xác suất câu.
- **Độc lập**: X, Y độc lập nếu $P(X, Y) = P(X) \cdot P(Y)$; khi đó $P(X \mid Y) = P(X)$.
- Xây mô hình xác suất gồm 2 bước: (1) **định nghĩa mô hình**, (2) **ước lượng tham số** (training/learning).
- Mô hình (gần như) luôn **giả định độc lập** để giảm số tham số phải ước lượng (VD: từ $n^2$ xuống $2n$).

## 3. Mô hình ngôn ngữ N-gram

- LM trên từ vựng V gán xác suất cho chuỗi trong $V^*$.
- **Giả định n-gram**: mỗi từ chỉ phụ thuộc **n−1 từ trước đó** (đây chính là giả định độc lập):
  $$P(w_1, \ldots, w_N) = \prod_i P(w_i \mid w_{i-n+1}, \ldots, w_{i-1})$$

### Ước lượng (Estimating)
1. Bọc câu bằng ký hiệu bắt đầu/kết thúc: `<s> Alice was beginning to get very tired… </s>`
2. Đếm tần suất từng n-gram: $C(\text{Alice}) = 1$, $C(\text{Alice was}) = 1$, ...
3. Chuẩn hóa tần suất để được xác suất — **ước lượng tần suất tương đối**:
  $$P(w_i = \text{`the'} \mid w_{i-1} = \text{`on'}) = \frac{C(\text{`on the'})}{C(\text{`on'})}$$

### Sử dụng LM
- Dùng LM làm **bộ sinh câu ngẫu nhiên**.
- Với các hệ thống đa ứng viên (MT, speech, spell-check, generation), chọn câu có xác suất cao hơn:
  $$\underset{S_{Out}}{\arg\max}\ P(S_{Out} \mid Input) = \underset{S_{Out}}{\arg\max}\ P(Input \mid S_{Out}) \cdot P(S_{Out})$$

### Sinh ngôn ngữ bằng n-gram (sampling)
Chia khoảng $[0,1]$ thành N khoảng con theo xác suất từng kết quả → sinh số ngẫu nhiên $r \in [0,1]$ → trả về $x_i$ chứa $r$. (Slide minh họa sinh văn bản kiểu Wall Street Journal.)

## 4. Đánh giá Mô hình Ngôn ngữ

Quy trình: định nghĩa metric → train trên tập train (tune trên held-out) → test trên tập test **rời rạc** với train/held-out → so sánh điểm.

| Loại | Ý nghĩa | Metric tiêu biểu |
|---|---|---|
| **Intrinsic (nội tại)** | Đo mức độ mô hình nắm bắt điều nó cần nắm bắt (VD: xác suất) | **Perplexity** |
| **Extrinsic (bên ngoài/task-based)** | Đo tính hữu ích của mô hình trong một tác vụ cụ thể | **Word Error Rate (WER)** |

- **Perplexity (độ bối rối)** = nghịch đảo xác suất tập test, chuẩn hóa theo số token; LM1 tốt hơn LM2 nếu gán **perplexity thấp hơn** (= xác suất cao hơn) cho corpus test. Chỉ so sánh trực tiếp khi **cùng từ vựng**.
- **WER**: gốc từ nhận dạng giọng nói — đo khác biệt giữa chuỗi từ dự đoán và chuỗi từ đúng trong transcript; lưu ý dữ liệu test chưa từng thấy sẽ chứa từ chưa từng thấy.

## 5. Word Embedding (Nhúng từ)

- Kỹ thuật biểu diễn từ/tài liệu bằng **vector số thực** trong không gian chiều thấp, khiến từ cùng nghĩa có biểu diễn giống nhau, ghi nhận **ngữ nghĩa liên từ**.

### 5.1. TF-IDF
- **TF** đo tần suất từ trong một văn bản; **IDF** đo độ hiếm của từ trong toàn bộ kho.
- Ứng dụng: information retrieval, loại stopword, trích từ khóa, phân tích văn bản cơ bản.

### 5.2. Bag of Words (BOW)
- Mỗi giá trị trong vector là **số lần xuất hiện** của từ trong văn bản/câu.
- Các bước: tách câu → tách từ → bỏ stopword/dấu câu → thường hóa in thường → dựng bảng phân bố tần suất.

### 5.3. Mạng nơ-ron cơ bản & Softmax
- **Softmax** biến vector K giá trị thực thành vector K giá trị ∈ (0, 1) có tổng bằng 1 → có thể diễn giải là **xác suất**.

### 5.4. One-Hot Encoding
- Biểu diễn từ bằng vector 0/1 chỉ một vị trí bật — đơn giản nhưng **không ghi nhận ngữ nghĩa**.

## 6. word2vec

- **Ý tưởng then chốt**: dự đoán các từ quanh mỗi từ. Lợi ích: nhanh, dễ thêm văn bản/từ mới.
- Hai kiến trúc:
  - **CBOW (Continuous Bag of Words)**: dùng các từ ngữ cảnh trong cửa sổ để **dự đoán từ giữa**. VD: "The cat sat on floor" (window size 2).
  - **Skip-gram**: dùng **từ giữa** để dự đoán các từ xung quanh trong cửa sổ.
- **So sánh**: CBOW không tốt cho **từ hiếm**, cần ít dữ liệu train hơn; Skip-gram tốt cho **từ hiếm**, cần nhiều dữ liệu hơn.
- **Word Analogies (loại suy từ)**: phát biểu dạng "a is to b as x is to y" — khẳng định a, x biến đổi cùng cách để được b, y (kiểm chứng bằng hiệu vector: $\vec{b} - \vec{a} \approx \vec{y} - \vec{x}$).
- **doc2vec**: mở rộng word2vec ra mức **văn bản**.

## 7. GloVe (Global Vectors for Word Representation)

- Dựa trên **tỷ lệ xác suất đồng xuất hiện (ratio of co-occurrence probabilities)** $P_{ik}/P_{jk}$. Ví dụ thí nghiệm "ice"/"steam":
  - Từ liên quan "ice" nhưng không liên quan "steam" → tỷ lệ lớn (8.9).
  - Từ liên quan "steam" nhưng không liên quan "ice" → tỷ lệ nhỏ (0.085).
  - Từ liên quan cả hai → tỷ lệ gần 1 (1.36).
  - Từ không liên quan cả hai → tỷ lệ gần 1 (0.96).
- Vì không gian vector có tính tuyến tính, thông tin tỷ lệ được mã hóa bằng **hiệu vector**, sau đó lấy **tích vô hướng**: $F(w_i, w_k, \tilde{w}_j) = (w_i - w_j)^T \tilde{w}_k$.
- $\log(X_i)$ độc lập với k nên được hấp thụ vào bias $b_i$ cho $w_i$.
- Huấn luyện GloVe là tìm giá trị nhỏ nhất của hàm chi phí có **hàm trọng số f** thỏa:
  - $f(0) = 0$.
  - f không giảm để **đồng xuất hiện hiếm không bị quá trọng số**.
  - f tương đối nhỏ với x lớn để **đồng xuất hiện thường xuyên không bị quá trọng số**.

## 8. Vấn đề thực tiễn (Practical issues)

- Quá nhiều bản sao mô hình (mỗi tác vụ một mô hình).
- Cần dữ liệu gán nhãn quy mô lớn để fine-tune.
- Làm thế nào train mô hình thực hiện tác vụ NLP theo kiểu **zero-shot**?

→ Dẫn tới các kiến trúc mạng nơ-ron hiện đại và LLM.

## 9. RNN và các kiến trúc tuần tự

### 9.1. RNN (Recurrent Neural Network)
- Giới thiệu khái niệm **recurrence**: trạng thái ẩn $h_t$ mang thông tin, được truyền vào bước dự đoán kế tiếp cùng với input $x_t$.
- Các cấu hình: **one-to-many** (sinh chú thích ảnh), **many-to-one** (phân tích cảm xúc văn bản), **many-to-many** (dịch máy).

**Ưu điểm**: xử lý được input độ dài bất kỳ; bước t (lý thuyết) dùng được thông tin từ nhiều bước trước; kích thước mô hình không tăng khi ngữ cảnh dài; cùng trọng số mỗi bước → đối xứng trong xử lý.

**Nhược điểm**: tính toán tuần tự **chậm**; thực tế khó truy cập thông tin từ nhiều bước trước (vanishing gradient).

### 9.2. LSTM (Long Short Term Memory)
- Loại RNN thiết kế riêng cho **phụ thuộc dài hạn** trong dữ liệu chuỗi (time series, giọng nói, văn bản).
- Giới thiệu **memory cell** giữ thông tin lâu, được điều khiển bởi **3 cổng (gates)**:
  - **Forget gate**: loại thông tin không còn hữu ích khỏi cell state (sigmoid → 0 là quên, 1 là giữ).
  - **Input gate**: thêm thông tin hữu ích vào cell state (tanh sinh giá trị −1..+1).
  - **Output gate**: trích thông tin hữu ích từ cell state hiện tại làm đầu ra.

## 10. Sequence-to-Sequence (seq2seq)

- Ra đời đầu tiên cho **dịch máy** (Google). Dùng cho dịch, tóm tắt, image captioning.
- Gồm 2 thành phần: **Encoder** + **Decoder**, train bằng tập cặp input–output (mỗi bên là chuỗi token).
- **Encoder Stack**: chuyển từ input thành hidden vector (mỗi vector chứa từ hiện tại + ngữ cảnh); hidden state cuối được truyền làm **context vector** cho decoder.
- **Decoder Stack**: nhận hidden vector của encoder, hidden state của chính nó và từ hiện tại để sinh hidden vector tiếp theo và dự đoán từ kế tiếp: tại mỗi bước, dùng hidden state + context vector + token đầu ra trước để sinh **phân phối xác suất trên các token kế tiếp**, chọn token xác suất cao nhất.

## 11. Large Language Models (LLM)

- LM có **rất nhiều tham số (hơn 1 tỷ)** và thực hiện nhiều tác vụ thông qua **prompting**; VD: GPT, Llama2, Gemini, PaLM, Mistral, Mixtral...
- Kiến trúc **Transformer** cho phép tính toán song song nhanh trên nhiều GPU → train trên lượng dữ liệu lớn, xếp nhiều lớp (mô hình lớn), train trong thời gian dài.
- **Tiền huấn luyện trong NLP**: word embedding (word2vec, GloVe) pre-train từ thống kê đồng xuất hiện; **hạn chế**: embedding được áp dụng **không phụ thuộc ngữ cảnh (context-free)**.

### 11.1. Ba kiến trúc LLM chính

| Kiến trúc | Tiêu biểu | Pre-training | Đặc điểm |
|---|---|---|---|
| **Encoder-only** | BERT | Masked Language Modeling (MLM) | Tốt cho phân loại; khó sinh văn bản |
| **Decoder-only** | GPT | Auto-regressive LM | Train ổn định, hội tụ nhanh, tổng quát hóa tốt sau pre-train |
| **Encoder-decoder** | T0/T5 | Masked Span Prediction | Tốt cho dịch máy, tóm tắt |

### 11.2. Kiến trúc BERT
- **Input Embedding**: văn bản → vector nhúng bao hàm ngữ nghĩa (dùng được GloVe, FastText, Word2Vec).
- **Positional Encoding**: mã hóa vị trí token.
- **Self-Attention**: cho phép mô hình liên kết từ với nhau theo ngữ cảnh — VD câu "The animal didn't cross the street because it was too tired", self-attention giúp liên kết "it" với "animal".

**Masked LM (MLM)**:
- Che **k% = 15%** từ input rồi dự đoán từ bị che.
- Che quá ít: tốn kém khi train; che quá nhiều: thiếu ngữ cảnh.
- Vấn đề: token [MASK] không xuất hiện lúc fine-tuning → giải pháp **không thay [MASK] 100%**:
  - 80% thay bằng [MASK]: "went to the store" → "went to the [MASK]"
  - 10% thay bằng từ ngẫu nhiên: → "went to the running"
  - 10% giữ nguyên: → "went to the store"

**Next Sentence Prediction (NSP)**: học quan hệ giữa câu — dự đoán câu B là câu thực sự theo sau câu A hay là câu ngẫu nhiên.

### 11.3. Decoder-Only Transformer (GPT)
- **GPT** (Generative Pre-trained Transformer) là mô hình Transformer decoder-only đầu tiên.
- **Layer Normalization**: tính mean/std theo chiều đặc trưng cho từng mẫu độc lập → ổn định học, giảm số bước train.
- **Masked Self-Attention**: ngăn mô hình attention tới token phía sau token hiện tại khi train/sinh (VD trong dịch máy); cài đặt bằng cách đặt trọng số thành $-\infty$ trước Softmax.

### 11.4. Từ raw text đến trợ lý căn chỉnh & Prompting
- Quy trình phát triển: raw text → pre-train → align thành assistant.
- **Prompting như "lập trình lúc suy luận" (inference-time programming)**: điều khiển LLM thực hiện tác vụ thông qua prompt.

---

## Ghi chú liên hệ đề tài RAG

- **TF-IDF / BOW** (mục 5): baseline retrieval trước khi so sánh với embedding hiện đại.
- **word2vec / GloVe** (mục 6–7): embedding tĩnh context-free → hạn chế với từ đa nghĩa.
- **BERT (encoder-only)** (mục 11): nền tảng của các mô hình embedding (Sentence-BERT) cho retriever RAG.
- **GPT (decoder-only) & LLM** (mục 11): thành phần generation của RAG.
- **Perplexity / đánh giá intrinsic–extrinsic** (mục 4): tư duy đánh giá tách biệt retrieval vs. generation.
