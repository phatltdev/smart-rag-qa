# Tóm tắt Chương 3: Truy vấn Thông tin (Information Retrieval)

> Nguồn: `Chapter3-InformationRetrieval.pptx` (49 slides) — Giảng viên: Trương Quốc Định, Khoa Hệ thống Thông tin, Trường CNTT & TT.

---

## 1. Định nghĩa

- **Information Retrieval (IR)** là tìm kiếm các tài liệu (thường là **phi cấu trúc**, chủ yếu là văn bản) thỏa mãn **nhu cầu thông tin** từ các kho lớn (thường lưu trên máy tính).

### Giả định cơ bản
- **Collection**: tập tài liệu cố định.
- **Goal**: truy hồi tài liệu có thông tin **liên quan (relevant)** đến nhu cầu, giúp người dùng hoàn thành tác vụ.

## 2. Inverted Index (Chỉ mục nghịch đảo)

- Với mỗi term $t$, lưu **danh sách mọi tài liệu chứa $t$**, mỗi tài liệu định danh bằng **docID** (số serial).
- Cần **postings list kích thước biến đổi**:
  - Trên đĩa: chuỗi postings liên tục là chuẩn và tốt nhất.
  - Trong bộ nhớ: linked list hoặc mảng dài biến đổi — đánh đổi kích thước/tính dễ chèn.

### Xử lý truy vấn AND
VD: `Brutus AND Caesar`:
1. Tìm `Brutus` trong Dictionary → lấy postings.
2. Tìm `Caesar` trong Dictionary → lấy postings.
3. **Merge** hai postings.

**Thuật toán merge**: đi đồng thời qua hai postings, thời gian **tuyến tính** theo tổng số phần tử; nếu độ dài hai danh sách là x, y thì merge tốn $O(x+y)$. **Điều kiện then chốt**: postings được **sắp xếp theo docID**.

### Boolean queries (truy vấn Boolean)
- **Boolean retrieval model**: truy vấn là biểu thức Boolean với **AND, OR, NOT**.
- Nhìn mỗi tài liệu là **tập hợp từ** — chính xác tuyệt đối: tài liệu khớp điều kiện hoặc không.
- Là mô hình đơn giản nhất để xây hệ IR.

### Phrase queries (truy vấn cụm từ)
- Muốn trả lời truy vấn dạng cụm: "stanford university" — câu "I went to university at Stanford" **không** khớp.
- Là một trong ít ý tưởng "tìm kiếm nâng cao" thực sự hiệu quả và dễ hiểu với người dùng.

### Positional indexes (chỉ mục vị trí)
- Trong postings, lưu với mỗi term các **vị trí** token xuất hiện:
  `<term, số doc chứa term; doc1: vị trí1, vị trí2…; doc2: vị trí1, …>`
- Xử lý phrase query: trích mục nghịch đảo từng term (to, be, or, not) → merge danh sách `doc:position` để liệt kê vị trí khớp "to be or not to be". Cùng phương pháp cho **tìm kiếm lân cận (proximity search)**.

## 3. Nén chỉ mục (Index Compression)

### Vì sao nén?
- **Dictionary**: nhỏ đủ để giữ trong bộ nhớ chính; nhỏ đến mức giữ được một phần postings trong bộ nhớ.
- **Postings**: giảm dung lượng đĩa, giảm thời gian đọc postings; máy tìm kiếm lớn giữ phần đáng kể postings trong bộ nhớ — nén giúp giữ được nhiều hơn.

### Nén Dictionary
- Mảng mục cố định: ~400.000 terms × 28 bytes/term = 11,2 MB — phần lớn byte bị lãng phí (phân 20 bytes cho term 1 ký tự), vẫn không chứa nổi từ siêu dài; tiếng Anh viết trung bình ~4,5 ký tự/từ, từ điển trung bình ~8 ký tự.
- **Dictionary-as-a-String**: lưu dictionary thành một chuỗi ký tự dài, con trỏ đến từ kế tiếp đánh dấu kết thúc từ hiện tại — tiết kiệm tới 60%.
- **Blocking**: lưu con trỏ cho mỗi term thứ k (VD k=4) + lưu độ dài term (1 byte thêm).
- **Front coding**: từ đã sắp xếp thường có tiền tố chung dài — chỉ lưu phần khác biệt (cho k−1 từ cuối mỗi block): `8automata8automate9automatic10automation`.

### Nén Postings
- Postings file lớn hơn dictionary **ít nhất 10 lần**; mục tiêu lưu mỗi posting (docID) cực gọn.
- Reuters 800.000 tài liệu: 32 bits/docID (int 4 byte) hoặc $\log_2 800.000 \approx 20$ bits/docID — mục tiêu dùng **ít hơn nhiều 20 bits**.
- Danh sách doc chứa term lưu **tăng dần theo docID** → chỉ cần lưu **khoảng cách (gaps)**: 33, 47, 154, 159, 202 → gaps 33, 14, 107, 5, 43... → hy vọng phần lớn gaps được mã hóa với ít hơn 20 bits.

## 4. Ranked Retrieval (Truy hồi có xếp hạng)

- Thay vì trả về tập tài liệu khớp biểu thức truy vấn, hệ trả về **thứ bậc (ordering)** các tài liệu (top) theo mức độ liên quan đến truy vấn.
- **Free text queries**: truy vấn chỉ là một/vài từ trong ngôn ngữ con người, không dùng ngôn ngữ toán tử/biểu thức.
- Trong thực tế, ranked retrieval thường đi cùng free text queries và ngược lại.

### Chấm điểm (Scoring) làm nền tảng
- Trả về theo thứ tự tài liệu **có khả năng hữu ích nhất**.
- Gán **điểm (score)** — VD trong [0, 1] — đo mức độ "khớp" giữa tài liệu và truy vấn.

## 5. Mô hình Không gian Vector (Vector Space Model)

### Trọng số tf-idf
- Mỗi tài liệu được biểu diễn bằng **vector thực các trọng số tf-idf** $\in \mathbb{R}^{|V|}$ (ma trận trọng số).

### Tài liệu là vector
- Không gian $|V|$ chiều, các **trục là term**, tài liệu là điểm/vector trong không gian này.
- Chiều rất cao: hàng chục triệu chiều với web search engine; vector rất **thưa (sparse)** — phần lớn phần tử bằng 0.

### Truy vấn là vector
- **Ý tưởng 1**: biểu diễn truy vấn cũng là vector trong không gian đó.
- **Ý tưởng 2**: xếp hạng tài liệu theo **độ gần** với truy vấn: proximity = độ tương tự của vector ≈ nghịch đảo khoảng cách.

### Vì sao khoảng cách là ý tưởng tồi?
- Lấy tài liệu $d$ và ghép nối với chính nó thành $d'$: về "ngữ nghĩa" $d$ và $d'$ có nội dung như nhau.
- **Khoảng cách Euclidean** giữa chúng có thể rất lớn, nhưng **góc** giữa hai document là 0 — tương ứng **độ tương tự cực đại**.
- → **Ý tưởng then chốt: xếp hạng tài liệu theo góc với truy vấn.**

### Từ góc đến cosine
- Hai khái niệm tương đương:
  - Xếp hạng tài liệu **giảm dần theo góc** giữa truy vấn và tài liệu.
  - Xếp hạng tài liệu **tăng dần theo cosine(query, document)**.
- Cosine là hàm **đơn điệu giảm** trên $[0°, 180°]$:

$$\text{cosine}(q, d) = \frac{q \cdot d}{|q| \cdot |d|} = \frac{\sum_i q_i d_i}{\sqrt{\sum_i q_i^2} \sqrt{\sum_i d_i^2}}$$

## 6. Spell Correction (Sửa lỗi chính tả)

### Hai hướng chính
| Loại | Cách làm | Hạn chế |
|---|---|---|
| **Isolated word (từ độc lập)** | Kiểm tra từng từ riêng lẻ | Không bắt được lỗi chính tả tạo ra từ đúng (VD: from ↔ form) |
| **Context-sensitive (nhạy ngữ cảnh)** | Nhìn các từ xung quanh (VD: "I flew form Heathrow to Narita") | Phức tạp hơn |

### Isolated word correction
- **Tiền đề**: có lexicon (từ điển) chứa các chính tả đúng. Hai lựa chọn lexicon:
  - Lexicon chuẩn (Webster's English Dictionary) hoặc lexicon chuyên ngành (bảo trì thủ công).
  - Lexicon của chính corpus được index (mọi từ trên web, tên riêng, từ viết tắt... kể cả từ sai chính tả).
- Cho trước lexicon và chuỗi ký tự Q, trả về từ trong lexicon **gần Q nhất**. Các thước đo "gần nhất":

**Edit distance (Levenshtein)**: số phép biến đổi tối thiểu để chuyển chuỗi này thành chuỗi khác, phép biến đổi cấp ký tự: **Insert, Delete, Replace, (Transposition)**.
- VD: dof → dog là 1; cat → act là 2 (chỉ 1 nếu có transpose); cat → dog là 3.
- Tìm bằng **quy hoạch động (dynamic programming)**.

**Weighted edit distance**: gán trọng số khác nhau cho các phép biến đổi.

**n-gram overlap**: liệt kê mọi n-gram của chuỗi truy vấn và của lexicon; dùng n-gram index truy hồi các term khớp bất kỳ n-gram nào của truy vấn; ngưỡng theo số n-gram khớp (biến thể: trọng số theo bố cục bàn phím...).
- VD trigram: "november" → nov, ove, vem, emb, mbe, ber; "december" → dec, ece, cem, emb, mbe, ber → **3 trigram trùng** (trên 6 mỗi từ).

**Jaccard coefficient**: thước đo trùng lắp phổ biến, với X, Y là hai tập hợp:

$$J(X, Y) = \frac{|X \cap Y|}{|X \cup Y|}$$

- Bằng 1 khi X, Y có cùng phần tử; bằng 0 khi rời rạc; không yêu cầu X, Y cùng kích thước; luôn ∈ [0, 1].
- Đặt ngưỡng để quyết định khớp: VD nếu $J > 0{,}8$ → coi là match.

### Context-sensitive spell correction
- Cần ngữ cảnh xung quanh để bắt lỗi. Cách làm:
  1. Truy hồi các term trong từ điển **gần** mỗi term truy vấn (theo weighted edit distance).
  2. Thử mọi cụm từ kết quả với mỗi lần "sửa" một từ: "flew from heathrow" / "fled form heathrow" / "flea form heathrow"...
  3. **Hit-based**: đề xuất phương án có **nhiều kết quả trả về (hits)** nhất — VD "flew form Heathrow" không tài liệu nào khớp → gợi ý "Did you mean 'flew from Heathrow'?"

## 7. Query Expansion (Mở rộng truy vấn) & Relevance Feedback

### Vì sao cần cải thiện kết quả?
- Cho **recall cao**: tìm "aircraft" không khớp "plane"; "thermodynamic" không khớp "heat".
- Hai nhóm phương pháp:
  - **Global methods** (độc lập truy vấn): query expansion bằng **thesauri** (từ điển đồng nghĩa) hoặc **tự sinh thesaurus tự động**.
  - **Local methods**: **relevance feedback**, **pseudo relevance feedback**.

### Relevance Feedback (phản hồi liên quan)
- Người dùng phản hồi về tính liên quan của tài liệu trong kết quả ban đầu:
  1. Người dùng đưa truy vấn (ngắn, đơn giản).
  2. Đánh dấu một số kết quả là **relevant / non-relevant**.
  3. Hệ tính **biểu diễn tốt hơn** cho nhu cầu thông tin dựa trên phản hồi.
  4. Có thể lặp một/vài vòng.
- Ý tưởng: khó đưa truy vấn tốt khi chưa biết rõ kho tài liệu → **lặp dần**.

### Khái niệm Centroid (tâm cụm)
- Centroid là **tâm khối (center of mass)** của một tập điểm, với C là tập tài liệu:

$$\vec{c} = \frac{1}{|C|} \sum_{d \in C} \vec{d}$$

### Thuật toán Rocchio 1971 (SMART)
- Dùng trong thực hành, với:
  - $D_r$: tập vector tài liệu **liên quan** đã biết; $D_{nr}$: tập tài liệu **không liên quan** đã biết.
  - $q_0$: vector truy vấn gốc; $q_m$: vector truy vấn **đã hiệu chỉnh**; $\alpha, \beta, \gamma$: trọng số (chọn tay hoặc đặt thực nghiệm):

$$q_m = \alpha q_0 + \beta \frac{1}{|D_r|} \sum_{d_i \in D_r} d_i - \gamma \frac{1}{|D_{nr}|} \sum_{d_i \in D_{nr}} d_i$$

- **Truy vấn mới dịch về phía tài liệu liên quan và lùi khỏi tài liệu không liên quan.**

### Query Expansion vs. Relevance Feedback
- Relevance feedback: người dùng cho input bổ sung (relevant/non-relevant) về **tài liệu** → dùng để tái trọng số các term trong tài liệu.
- Query expansion: người dùng cho input (từ khóa tốt/xấu) về **từ/cụm từ**.

### Thesaurus-based query expansion
- Với mỗi term $t$ trong truy vấn, mở rộng bằng **từ đồng nghĩa và từ liên quan** từ thesaurus: feline → "feline cat".
- Có thể trọng số thấp hơn cho term thêm so với term gốc.
- **Tăng recall** nói chung; dùng rộng trong nhiều lĩnh vực khoa học/kỹ thuật.
- **Có thể giảm precision đáng kể**, đặc biệt với term mơ hồ: "interest rate" → "interest rate fascinate evaluate".
- Chi phí thủ công cao khi xây dựng thesaurus và cập nhật theo thay đổi khoa học.

### Automatic Thesaurus Generation (tự sinh thesaurus)
- Phân tích kho tài liệu để tự sinh thesaurus. Khái niệm nền tảng: **độ tương tự giữa hai từ**:
  - **Định nghĩa 1**: hai từ tương tự nếu **đồng xuất hiện** với các từ tương tự nhau.
  - **Định nghĩa 2**: hai từ tương tự nếu xuất hiện trong cùng **quan hệ ngữ pháp** với cùng các từ.
- Đồng xuất hiện **mạnh mẽ (robust) hơn**; quan hệ ngữ pháp **chính xác hơn**.

### Co-occurrence Thesaurus
- Cách đơn giản nhất dựa trên độ tương tự term–term trong $C = AA^T$, với A là ma trận **term-document**, $w_{i,j}$ là trọng số (đã chuẩn hóa) cho $(t_i, d_j)$.
- Với mỗi $t_i$, chọn các term có giá trị cao trong C.

---

## Ghi chú liên hệ đề tài RAG

Chương 3 là **lý thuyết nền tảng của thành phần Retriever** trong RAG:
- **Inverted index + Boolean/Phrase queries** (mục 2): retrieval từ khóa truyền thống (BM25/keyword search dựa trên inverted index).
- **TF-IDF + Vector Space Model + Cosine similarity** (mục 4–5): **baseline kinh điển** cần dựng trước khi so sánh với dense embedding (theo yêu cầu baseline trong AGENTS.md).
- **Spell correction** (mục 6): tăng độ robust cho câu hỏi của người dùng (query chuẩn hóa trước khi truy hồi).
- **Query expansion / Relevance feedback** (mục 7): kỹ thuật cải thiện recall — ý tưởng mở rộng query bằng từ đồng nghĩa tiếng Việt trước khi truy hồi.
