# Tóm tắt Chương 4: Khai thác Ý kiến (Opinion Mining)

> Nguồn: `Chapter4-OpinionMining.pptx` (32 slides) — Giảng viên: Trương Quốc Định, Khoa Hệ thống Thông tin, Trường CNTT & TT.

---

## 1. Giới thiệu

- **Opinion mining** là nghiên cứu tính toán về **ý kiến, cảm xúc, đánh giá và cảm xúc (opinion, sentiment, evaluation, emotion)**.

### Vì sao quan trọng?
- Ý kiến là **yếu tố ảnh hưởng chính đến hành vi**: khi ra quyết định, chúng ta thường tìm ý kiến từ người khác; sự trỗi dậy của mạng xã hội → **opinion data** khổng lồ.
- Sự trỗi dậy của **AI và chatbot**: cảm xúc và sentiment là then chốt trong giao tiếp con người.

### Sentiment vs. Opinion
- **Sentiment**: thái độ, suy nghĩ hoặc phán đoán được khơi gợi bởi **cảm xúc**.
- **Opinion**: quan điểm, phán đoán hoặc đánh giá hình thành trong **tâm trí** về một vấn đề cụ thể.

### Vì sao là bài toán hấp dẫn?
- Thách thức về trí tuệ + ứng dụng lớn → chủ đề nghiên cứu phổ biến trong NLP và Web data mining những năm gần đây.
- Chạm đến mọi khía cạnh của NLP nhưng bị giới hạn: quá khứ ít nghiên cứu trong NLP/Ngôn ngữ học.
- Có tiềm năng trở thành công nghệ lớn từ NLP — nhưng "chưa" và không dễ! Việc lấy nguồn dữ liệu và tích hợp dữ liệu cũng khó.

## 2. Phân loại ý kiến

Hai loại chính:

| Loại | Định nghĩa | Ví dụ |
|---|---|---|
| **Regular opinions (ý kiến thông thường)** | Biểu hiện sentiment/opinion trên một số **thực thể đích** | Trực tiếp: "The touch screen is really cool"; Gián tiếp: "After taking the drug, my pain has gone" |
| **Comparative opinions (ý kiến so sánh)** | So sánh **nhiều hơn một** thực thể | "iPhone is better than Blackberry" |

- Tài liệu tập trung vào regular opinions trước (gọi tắt là opinions).

## 3. Mô hình Opinion Mining (Liu, 2012)

- Ban đầu: opinion là **tứ bộ (quadruple)** `(target, sentiment, holder, time)` — ngắn gọn nhưng **không dễ dùng** (VD: "The voice quality of iPhone is amazing." → target = voice quality? Chưa hẳn).
- Hoàn chỉnh: opinion là **ngũ bộ (quintuple)**:

$$(\text{entity}, \text{aspect}, \text{sentiment}, \text{holder}, \text{time})$$

| Thành phần | Ý nghĩa |
|---|---|
| **Entity** | Thực thể đích (đối tượng) |
| **Aspect** | Đặc tính/tính năng của thực thể |
| **Sentiment** | positive, negative, neutral, rating hoặc một cảm xúc |
| **Holder** | Người nắm giữ ý kiến |
| **Time** | Thời điểm ý kiến được phát biểu |

### Ví dụ trích ngũ bộ từ blog
Blog Id XYZ567 ngày 08/10/2023: *"I bought iPhone13 yesterday. It is such a nice phone. The touch screen is really cool. The voice quality is great too. However, my mother was mad with me as I did not tell her before I bought the phone. She also thought the phone was too expensive."*

Quan sát thấy: opinion targets (entities + features), sentiments (positive/negative), opinion holders, time. Trích xuất:

```
(iPhone13, GENERAL,       +, XYZ567,     08/10/2023)
(iPhone13, touch_screen,  +, XYZ567,     08/10/2023)
(iPhone13, prix,          -, his_mother, 08/10/2023)
...
```

## 4. Tóm tắt Ý kiến (Opinion Summary)

- Với khối lượng lớn ý kiến, cần **bản tóm tắt** — là tác vụ **multi-document summarization**.
- Khác văn bản sự thật (1 fact = bất kỳ số lượng fact giống nhau): **ý kiến có mặt định lượng và có đích** → **1 opinion ≠ một số lượng opinion**.
- → **Aspect-based summary** phù hợp hơn.

### Aspect-based summary
Ví dụ review iPhone:

```
Feature 1: Touch screen
  Positive: 212
    - The touch screen was really cool.
    - The touch screen was so easy to use and can do amazing things.
  Negative: 6
    - The screen is easily scratched.
    - I have a lot of difficulty in removing finger marks from the touch screen.

Feature 2: battery life
  ...
```

## 5. Lý do & Điều kiện của Ý kiến

- **Reason (lý do)**: sự biện minh/giải thích cho ý kiến. Hai trường hợp chính:
  - Tiêu cực về thực thể do một aspect tệ: "I hate this car as it eats too much gas."
  - Tiêu cực về một aspect vì một lý do: "This car is too small."
- **Qualifier (từ hạn định)**: giới hạn hoặc thay đổi nghĩa của ý kiến, cho biết ý kiến hữu ích cho trường hợp nào:
  - "This car is too small **for a tall person**."
  - "The picture quality of **night shots** is bad."
- **Lưu ý**: không phải opinion nào cũng kèm reason/qualifier tường minh: "This car is bad."

## 6. Sentiment Classification (Phân loại cảm xúc cấp văn bản)

- Phân loại **toàn bộ tài liệu ý kiến (review)** theo cảm xúc tổng thể của người nắm giữ.
- Các lớp: **Positive, Negative, Neutral**.
- Về bản chất là bài toán **text classification**.

### Giả định
- Tài liệu do **một người** viết và phát biểu ý kiến về **một thực thể duy nhất**.
- Review thường thỏa; forum posting và blog thì **không** (đề cập/so sánh nhiều thực thể; nhiều posting không có sentiment).

### Các hướng tiếp cận
1. **Supervised learning (học có giám sát)**: áp dụng trực tiếp kỹ thuật học có giám sát. Các đặc trưng tốt đã được nghiên cứu nhiều: **term weighting schemes, POS tags, opinion words, negation (phủ định), syntactic dependency**.
2. **Lexicon-based (Taboada et al., 2011)**: dùng **sentiment lexicon** — tập thuật ngữ cảm xúc:
   - Từ tích cực: great, beautiful, amazing...
   - Từ tiêu cực: bad, unreliable, terrible, awful...
   - Mỗi thuật ngữ sentiment được gán điểm trong **[−5, +5]**; quyết định sentiment của review bằng **cộng dồn điểm** từ mọi thuật ngữ.
3. **Review rating prediction (dự đoán điểm review)**: dự đoán điểm rating (VD 1–5 sao); train/test là review có sao; bài toán được phát biểu là **hồi quy (regression)** vì điểm rating là **thứ tự (ordinal)**.

## 7. Sentence-level Sentiment Analysis (Phân tích cảm xúc cấp câu)

- Thường gồm **hai bước**:
  1. **Subjective classification**: nhận diện câu chủ quan.
  2. **Sentiment classification** trên câu chủ quan.
- **Cần nhớ**:
  - Nhiều câu **khách quan vẫn ngầm chứa sentiment**.
  - Nhiều câu **chủ quan không** phát biểu ý kiến tích cực/tiêu cực: "I believe he went home yesterday."

### Giả định
- Mỗi câu do một người viết và phát biểu **một ý kiến duy nhất** — đúng với câu đơn giản ("I like this car.") nhưng **không đúng** với nhiều câu ghép/phức: "I like the picture quality **but** battery life sucks."

### Segmentation and classification (phân đoạn & phân loại)
- Một câu có thể chứa **nhiều ý kiến** và cả mệnh đề chủ quan lẫn sự thật → cần nghiên cứu **phân loại cảm xúc cấp mệnh đề (clause)** tự động.

## 8. Aspect Extraction (Trích xuất khía cạnh)

Cho trước corpus ý kiến, trích xuất **mọi aspect**. Bốn hướng chính:

### 8.1. Finding frequent nouns and noun phrases (danh từ/danh cụm tần suất cao)
- Danh từ (NN) được nhắc tới **thường xuyên** nhiều khả năng là aspect thật.
- Vì: phần lớn aspect là danh từ/danh cụm; khi bàn về tính năng sản phẩm, từ ngữ dùng thường **hội tụ**; những từ tần suất cao thường là các aspect chính mà mọi người quan tâm.

### 8.2. Exploiting opinion and target relations (khai thác quan hệ ý kiến–đích)
- **Ý tưởng then chốt**: ý kiến luôn có đích; opinion term được dùng để **bổ nghĩa** cho aspect và entity:
  - "The pictures are absolutely amazing."
  - "This is an amazing software."
- Quan hệ cú pháp được **xấp xỉ** bằng cụm danh từ **gần nhất** với opinion word.
- Phân biệt:
  - **Explicit aspects**: aspect được nêu tường minh là danh từ/danh cụm trong câu — "The picture quality of this phone is great."
  - **Implicit aspects**: aspect không được nêu tường minh nhưng được ngụ ý — "This car is so expensive." (giá), "This phone will not easily fit in a pocket." (kích thước), "Include 16GB is stingy." (dung lượng).

### 8.3. Supervised learning (học có giám sát)
- Dùng phương pháp **gán nhãn chuỗi (sequence labelling)** như **HMM, CRF**...
- Sau khi phát hiện aspect expression, **nhóm** chúng vào **aspect category** (VD: power usage, battery life) dựa trên phương pháp tương tự, cần **taxonomy of aspects**:
  - **Mapping**: hệ ánh xạ mỗi aspect phát hiện được vào một node aspect trong taxonomy.
  - **Similarity metrics**: độ tương tự chuỗi, đồng nghĩa và các đo khoảng cách khác dùng **WordNet**.

### 8.4. Unsupervised learning & Topic modelling
- **Nhóm đồng nghĩa aspect không giám sát**: Clustering, Constrained topic modelling... dùng nhiều loại thông tin/độ tương tự để gom aspect expression vào aspect category:
  - Tương tự từ vựng dựa trên **WordNet**; thông tin phân bố; ràng buộc cú pháp...
- **Topic modelling**:
  - Một **topic về cơ bản là một aspect**: tài liệu là phân phối trên topic; topic là phân phối trên từ (price, cost, cheap, expensive...).
  - Aspect extraction có hai tác vụ: **trích aspect expression** + **gom cụm** chúng (picture, photo, image là một).
  - Mô hình tiêu biểu: **pLSA** (Hofmann, 1999), **LDA** (Blei et al., 2003).
  - Dùng topic model để mô hình hóa aspect; mô hình hóa **đồng thời** aspect và sentiment.
  - **Knowledge-based modelling**: mô hình không giám sát thường chưa đủ (topic/aspect thiếu nhất quán) → đề xuất topic model **dẫn dắt bởi tri thức chuyên ngành do người dùng chỉ định trước**.

## 9. Sentiment (Opinion) Lexicon

- Danh sách từ/cụm được dùng để diễn tả **cảm giác chủ quan và sentiment/opinion** của con người — **công cụ then chốt** cho sentiment analysis.
- Có vẻ có vô số biến thể biểu thức chứa sentiment.
- Nhiều biểu thức **phụ thuộc ngữ cảnh**, không chỉ phụ thuộc miền ứng dụng.
- **Ba cách biên soạn** lexicon:
  1. **Manual approach** (thủ công).
  2. **Corpus-based approach** (dựa trên corpus).
  3. **Dictionary-based approach** (dựa trên từ điển).

---

## Ghi chú liên hệ đề tài RAG

Chương 4 liên hệ với đề tài ở các điểm:
- **Sentiment lexicon + quy trình 5 bước** (mục 1.8 chương 1 + mục 9): nếu corpus tài liệu có chứa đánh giá/quan điểm, aspect extraction giúp cấu trúc hóa nội dung trước khi đưa vào vector DB.
- **Text classification supervised** (mục 6): kỹ thuật phân loại câu hỏi/ngữ cảnh tài liệu có thể dùng để **routing/filtering** trong RAG.
- **Topic modelling (LDA/pLSA)** (mục 8.4): phương pháp phân tích/khám phá chủ đề trong kho tài liệu — hỗ trợ đánh giá chất lượng chunking.
