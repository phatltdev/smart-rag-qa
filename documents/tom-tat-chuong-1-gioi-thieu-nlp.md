# Tóm tắt Chương 1: Giới thiệu về Xử lý Ngôn ngữ Tự nhiên (NLP)

> Nguồn: `Chapter1-Introduction_to_NLP.pptx` (63 slides) — Giảng viên: Trương Quốc Định, Khoa Hệ thống Thông tin, Trường CNTT & TT.

---

## 1. Động lực (Motivation)

- Sự gia tăng quan tâm đến giao tiếp người–máy cùng sự sẵn có của **big data (dữ liệu lớn)** cho phép máy tính đọc văn bản, nghe giọng nói, diễn giải, đo lường cảm xúc và xác định phần thông tin quan trọng.
- Lượng **dữ liệu phi cấu trúc (unstructured data)** khổng lồ được sinh ra mỗi ngày (từ hồ sơ y tế đến mạng xã hội) buộc máy phải phân tích nhiều dữ liệu ngôn ngữ hơn cả con người.
- Mục tiêu: **vượt qua mức khớp từ khóa (keyword matching)** để nhận diện cấu trúc và ý nghĩa của từ, câu, văn bản và hội thoại.

## 2. Định nghĩa

- **NLP** là lĩnh vực kết hợp **khoa học máy tính, trí tuệ nhân tạo (AI) và ngôn ngữ học tính toán**, quan tâm đến tương tác giữa máy tính và ngôn ngữ tự nhiên của con người.
- Nói ngắn gọn: các **phương pháp tính toán để hiểu hoặc sinh ra ngôn ngữ tự nhiên**.

## 3. Các cách tiếp cận NLP

| Cách tiếp cận | Ý tưởng | Đặc điểm |
|---|---|---|
| **Classical divide & conquer (chia để trị)** | Chia bài toán lớn thành các tác vụ nhỏ, giải từng tác vụ, rồi ghép thành một **pipeline** | Mỗi tác vụ có nhiều cách giải; một số/không tác vụ con dùng machine learning |
| **End-to-end (đầu cuối)** | Một mô hình (deep learning) duy nhất làm toàn bộ | Vẫn cần tokenization và các thao tác ML (data-loaders, batching...) |

## 4. Các tác vụ NLP cơ bản

### 4.1. Sentence splitting (tách câu)
- Tách văn bản tự do thành các câu — bước đầu tiên trong mọi ứng dụng NLP.
- Dấu chấm `.` có thể kết thúc câu, nhưng cũng có thể là viết tắt, số thập phân, dấu ellipsis, email... → **nhiều mơ hồ**. Dấu `?` và `!` cũng mơ hồ do emoticon, code, tiếng lóng.

### 4.2. Tokenization (tách từ/token)
- Câu hỏi cốt lõi: *Thế nào là một từ?* Ví dụ: "It's", "don't", "data-mining", "trường", "trường học".
- Chia văn bản thành đơn vị xử lý gọi là **Token**.
- Tách theo khoảng trắng **không hoạt động** với một số ngôn ngữ (Trung, Thổ Nhĩ Kỳ, tiếng Việt) → cần **sub-word units (đơn vị nhỏ hơn từ)**.
- Có thể có token nhiều từ: "New York Times", "kick the bucket", "Công nghệ thông tin".
- Kết quả: văn bản được biểu diễn thành **chuỗi token**.

### 4.3. Vietnamese word segmentation (tách từ tiếng Việt)
Bảng so sánh các công cụ (Precision / Recall / F1, %):

| Công cụ | Precision | Recall | F1 |
|---|---|---|---|
| vnTokenizer | 96.98 | 97.69 | 97.33 |
| JVnSegmenter-Maxent | 96.60 | 97.40 | 97.00 |
| JVnSegmenter-CRFs | 96.63 | 97.49 | 97.06 |
| DongDu | 96.35 | 97.46 | 96.90 |
| UETsegmenter | 97.51 | 98.23 | 97.87 |
| **VnCoreNLP (RDRSegmenter)** | 97.46 | **98.35** | **97.90** |

**VnCoreNLP (RDRSegmenter)** — quy trình:
1. Xuất phát từ câu đã tách từ thủ công (VD: "thuế_thu_nhập cá_nhân") và biểu diễn dạng **BI** ("thuế/B thu/I nhập/I cá/B nhân/I").
2. Trích âm tiết để dựng **raw corpus** ("thuế thu nhập cá nhân").
3. Áp dụng bộ tách khởi tạo theo chiến lược **longest matching** lên raw corpus.
4. So sánh corpus gold standard dạng BI với corpus khởi tạo để sinh **từ điển ngữ cảnh 5 âm tiết** D (khóa = cửa sổ 5 âm tiết, giá trị = nhãn gold).
5. Dựa trên D, **rule selector** chọn luật phù hợp xây cây **SCRDR** (Single Classification Ripple Down Rules).

### 4.4. Stemming & Lemmatization
- **Stemming (gốc từ hóa)**: sinh các biến thể hình thái của từ gốc/cơ sở — "retrieval", "retrieved", "retrieves" → "retrieve".
- **Lemmatization (chuẩn hóa từ điển)**: nhóm các dạng biến cách khác nhau của cùng một từ về một từ điển chung; giúp chatbot, truy vấn tìm kiếm chính xác hơn.

### 4.5. POS tagging (gán nhãn từ loại)
- Gán mỗi từ một **phần từ loại (part of speech)**: danh từ, động từ, trạng từ, tính từ, đại từ, liên từ và các tiểu loại.
- Ba hướng chính:
  - **Rule-based**: gán nhãn theo đặc điểm từ và ngữ cảnh (VD: đuôi "-tion"/"-ment" → danh từ; viết hoa toàn bộ → danh từ riêng).
  - **Stochastic (ngẫu nhiên/thống kê)**:
    - *Word Frequency Approach*: theo xác suất từ xuất hiện với nhãn nào đó (có thể sinh chuỗi nhãn không hợp lệ).
    - *Tag Sequence Probabilities (n-gram)*: nhãn tốt nhất determined theo xác suất xảy ra với n nhãn trước đó.
  - **Hidden Markov Model (HMM)**: tập trạng thái (mỗi trạng thái = một nhãn POS) + chuyển tiếp giữa các trạng thái. Học từ dữ liệu huấn luyện:
    - Xác suất chuyển nhãn: $P(t_i \mid t_{i-1})$
    - Xác suất quan sát từ: $P(w_i \mid t_i)$

### 4.6. Parsing (phân tích cú pháp)
- Còn gọi là **Syntactic analysis / syntax analysis**. Đối chiếu văn bản với các quy tắc **ngữ pháp hình thức** để kiểm tra tính có nghĩa.

### 4.7. Named Entity Recognition (NER — nhận dạng thực thể có tên)
- Xác định và phân loại **thực thể (entity)** trong văn bản: tên người, tổ chức, địa điểm, ngày tháng, giá trị số...
- Các bước:
  1. **Text Preprocessing**: tokenization, POS tagging.
  2. **Entity Identification**: quét tìm chuỗi từ tương ứng thực thể.
  3. **Entity Classification**: xếp vào loại định sẵn (Person, Organization, Location, Date...).
  4. **Contextual Analysis**: dùng ngữ cảnh đảm bảo phân loại chính xác.
- **Use cases**: tuyển dụng (quét CV), cập nhật tin tức (giám sát thị trường, công nghệ, tuân thủ), bồi thường bảo hiểm, phân tích khách hàng (review, ticket, khảo sát), y tế (hồ sơ bệnh nhân, kết quả xét nghiệm).

## 5. Các ứng dụng NLP

Hai hướng chính: **quản lý khối lượng lớn nguồn văn bản** (cho con người / thu thập tài nguyên ngôn ngữ tự động) và **tương tác người–máy**.

### 5.1. Machine Translation (dịch máy)
- Dịch văn bản từ ngôn ngữ nguồn sang đích, giữ nguyên tính chất — quan trọng nhất là **giữ nghĩa**.
- Mô hình cho mỗi từ nguồn: bản dịch, số từ cần ở đích, vị trí bản dịch trong câu, số từ phải sinh từ đầu.

### 5.2. Information Retrieval (IR — truy vấn thông tin)
- **Input**: tập văn bản (Web, kho tài liệu doanh nghiệp...) + nhu cầu thông tin của người dùng (query).
- **Output**: các văn bản thỏa mãn nhu cầu.

### 5.3. Question Answering (QA — hỏi đáp)
- Phần mở rộng tự nhiên của IR: trả về **câu trả lời trực tiếp** (thường là một fact) thay vì văn bản chứa câu trả lời.

### 5.4. Information Extraction (IE — trích xuất thông tin)
- Rút thông tin từ văn bản, CSDL, website... từ văn bản phi cấu trúc/bán cấu trúc → có cấu trúc.
- Quy trình 6 bước:
  1. **Initial processing**: tách văn bản thành vùng/cụm/đoạn/token; POS tagging; nhận diện cụm danh/tính từ.
  2. **Proper names identification**: nhận diện tên riêng (người, tổ chức, ngày, tiền, địa chỉ...) bằng **regular expressions**.
  3. **Parsing**: tìm cụm danh từ quanh thực thể và cụm động từ.
  4. **Extraction of events and relations**: dựng quan hệ giữa các ý bằng **luật trích xuất theo pattern**; khớp pattern thì gán nhãn và truy hồi sau.
  5. **Coreference/Anaphora resolution**: quyết định các cụm danh từ có chỉ cùng một thực thể hay không.
  6. **Output results generation**: chuyển cấu trúc thu được thành template theo định dạng người dùng.

### 5.5. Text Classification (phân loại văn bản)
- Kỹ thuật machine learning gán **tập hạng mục định trước** cho văn bản mở; dùng tổ chức, cấu trúc, phân loại mọi loại văn bản (tài liệu, nghiên cứu y tế, web...).

### 5.6. Text Clustering (phân cụm văn bản)
- Nhóm tập văn bản **không nhãn** sao cho văn bản cùng cụm tương tự nhau hơn với văn bản cụm khác; thuật toán xác định xem có **cụm tự nhiên** trong dữ liệu hay không.

### 5.7. Automatic Summarization (tóm tắt tự động)
- Tóm tắt là **phép biến đổi thu gọn** từ văn bản nguồn thành văn bản tóm tắt, theo hai hướng:
  - **Extractive** (trích nguyên câu).
  - **Abstractive** (sinh lại bằng ngôn ngữ mới).

### 5.8. Sentiment Analysis / Opinion Mining (phân tích cảm xúc)
- Nhận diện **sắc thái cảm xúc** đằng sau văn bản; tổ chức dùng để xác định và phân loại ý kiến về sản phẩm/dịch vụ/ý tưởng.
- Trích xuất: **polarity (cực tính)**, mức tích cực/tiêu cực, chủ thể và người nắm giữ ý kiến.
- Các bước tổng quát: thu thập dữ liệu → làm sạch → trích đặc trưng → chọn mô hình ML (rule-based / tự động / hybrid) → phân loại cảm xúc (positive/negative/neutral).
- **Aspect-based analysis**: xem xét cụ thể khía cạnh nào được khen/chê (VD: "pin quá ngắn").

### 5.9. Chatbot
- Mô phỏng hiểu ngôn ngữ người, xử lý và tương tác lại khi thực hiện tác vụ cụ thể; là **conversational agent**.
- Phân loại: **text-based** / **voice-based**.
- Hai hướng thiết kế: **rule-based** (theo luật, đơn giản đến phức tạp) và **self-learning** (machine learning, hiệu quả hơn).

## 6. Các thách thức của NLP

| Thách thức | Giải thích | Ví dụ |
|---|---|---|
| **Synonyms (đồng nghĩa)** | Từ/câu khác nhau nhưng cùng nghĩa | Fall = Autumn; "When will my book arrive?" = "When will I receive my book?" |
| **Polysemy (đa nghĩa)** | Một từ có nhiều nghĩa liên quan | "Fall": mùa thu / rơi xuống; "The door is open": phát ngôn sự thật / yêu cầu đóng cửa |
| **Homonymy (đồng âm/dồng viết)** | Từ phát âm/viết giống nhau nhưng nghĩa khác hẳn | "Con đường này thật rộng! Chúng ta nên pha thêm đường."; "Ông ấy cười khanh khách! Nhà ông ấy đang có khách." |
| **Attachment ambiguity (mơ hồ gắn kết)** | Nghi ngờ về cấu trúc cú pháp của câu | (ví dụ minh họa trên slide) |
| **Discourse analysis (phân tích diễn ngôn)** | Đại từ chỉ về ai? | "Alice understands that you like your mother, but she..." — *she* là Alice hay mẹ? |
| **Compositionality (tính tổ hợp)** | Ngôn ngữ tổ hợp từ đơn vị nhỏ thành nghĩa lớn | — |
| **Scale (quy mô)** | Từ vựng và kho dữ liệu khổng lồ | Oxford Dictionary ~273.000 headwords; Penn Treebank ~7 triệu từ POS-tagged; Wikipedia tiếng Anh ~2,9 tỷ từ; Web hàng tỷ từ |

---

## Ghi chú liên hệ đề tài RAG

Các kiến thức trực tiếp phục vụ hệ thống Hỏi–Đáp RAG:
- **Tokenization / Vietnamese word segmentation** (mục 4.2–4.3): bước tiền xử lý quan trọng trước khi embedding tiếng Việt.
- **Information Retrieval** (mục 5.2): thành phần retriever trong RAG.
- **Question Answering** (mục 5.3): đúng bài toán của hệ thống.
- **Chatbot** (mục 5.9): hình thức giao tiếp của hệ thống.
