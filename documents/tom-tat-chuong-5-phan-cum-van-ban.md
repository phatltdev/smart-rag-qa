# Tóm tắt Chương 5: Phân cụm Văn bản (Text Clustering)

> Nguồn: `Chapter5-Clustering.pptx` (40 slides) — Giảng viên: Trương Quốc Định, Khoa Hệ thống Thông tin, Trường CNTT & TT.

---

## 1. Giới thiệu

- **Cluster (cụm)**: tập đối tượng dữ liệu **tương tự (hoặc liên quan)** với nhau trong cùng nhóm và **khác biệt (không liên quan)** với đối tượng ở các nhóm khác.
- **Cluster analysis**: tìm sự tương tự giữa dữ liệu theo các đặc trưng có trong dữ liệu và nhóm các đối tượng tương tự thành cụm.
- **Unsupervised learning (học không giám sát)**: **không có class định trước**.

### Ứng dụng tiêu biểu
- **Công cụ độc lập để hiểu dữ liệu**:
  - Information retrieval: document clustering (phân cụm tài liệu).
  - Land use: nhận diện vùng sử dụng đất tương tự trong CSDL quan sát trái đất.
  - Marketing: khám phá nhóm khách hàng riêng biệt → chương trình marketing nhắm mục tiêu.
  - Economic Science: nghiên cứu thị trường.
- **Công cụ tiền xử lý cho thuật toán khác (Utility)**:
  - Summarization: tiền xử lý cho regression, association analysis...
  - Finding K-nearest Neighbours: giới hạn tìm kiếm vào một/vài cụm.
  - Outlier detection: outlier thường được xem là đối tượng "xa" mọi cụm.

## 2. Chất lượng Phân cụm

### Thế nào là phân cụm tốt?
- **Độ tương tự cao trong cụm (high intra-class similarity)**: gắn kết bên trong cụm.
- **Độ tương tự thấp giữa các cụm (low inter-class similarity)**: phân biệt giữa các cụm.

### Chất lượng phụ thuộc vào:
- **Đo tương tự/khác biệt** mà phương pháp dùng.
- Cách cài đặt của nó.
- Khả năng phát hiện một phần hoặc toàn bộ pattern ẩn.

### Đo lường
- Tương tự được biểu diễn qua **hàm khoảng cách** $d(i, j)$ (thường là metric):
  - Định nghĩa hàm khoảng cách khác nhau nhau cho biến **interval-scaled, boolean, categorical, ordinal, ratio, vector**.
  - Nên gán **trọng số** cho các biến khác nhau theo ứng dụng và ngữ nghĩa dữ liệu.
- Có hàm "chất lượng" riêng đo "độ tốt" của cụm — nhưng **khó định nghĩa** "tương tự đủ" hay "tốt đủ".

## 3. Các Cân nhắc khi Phân cụm (Considerations)

| Tiêu chí | Các lựa chọn |
|---|---|
| **Partitioning criteria** | Phân hoạch đơn cấp vs. **phân cấp (hierarchical)** — thường mong muốn phân cấp đa tầng |
| **Separation of clusters** | **Exclusive** (một khách hàng thuộc một vùng) vs. **non-exclusive** (một tài liệu có thể thuộc nhiều class) |
| **Similarity measure** | **Distance-based** (Euclidean, road network, vector) vs. **connectivity-based** (density, contiguity) |
| **Clustering space** | **Full space** (thường khi chiều thấp) vs. **subspaces** (thường trong phân cụm chiều cao) |

## 4. Yêu cầu & Thách thức

- **Scalability (khả năng mở rộng)**: phân cụm **toàn bộ dữ liệu** thay vì chỉ mẫu.
- **Xử lý nhiều loại thuộc tính**: numerical, binary, categorical, ordinal, linked và hỗn hợp.
- **Constraint-based clustering**: người dùng có thể đưa ràng buộc; dùng tri thức chuyên ngành xác định tham số đầu vào.
- **Interpretability and usability** (khả năng diễn giải & sử dụng).
- Khác:
  - Phát hiện cụm **hình dạng tùy ý**.
  - Xử lý **dữ liệu nhiễu (noise)**.
  - **Incremental clustering** và không nhạy cảm thứ tự đầu vào.
  - **Chiều cao (high dimensionality)**.

## 5. Các Hướng tiếp cận Chính (Major Clustering Approaches)

| Hướng | Ý tưởng | Phương pháp tiêu biểu |
|---|---|---|
| **Partitioning** | Dựng các phân hoạch rồi đánh giá bằng tiêu chí nào đó (VD: tối thiểu hóa tổng bình phương sai số) | k-means, k-medoids |
| **Hierarchical** | Tạo phân rã phân cấp của tập dữ liệu theo tiêu chí nào đó | Diana, Agnes |
| **Density-based** | Dựa trên kết nối và hàm mật độ | DBSCAN |
| **Model-based** | Giả thuyết mô hình cho từng cụm, tìm cách khớp mô hình tốt nhất | SOM, GHSOM |
| **Frequent pattern-based** | Dựa trên phân tích mẫu hình frequent | p-Cluster |
| **Link-based** | Đối tượng liên kết với nhau nhiều cách; dùng link khổng lồ để phân cụm | SimRank, LinkClus |

## 6. Thuật toán Phân hoạch (Partitioning Algorithms)

- Phân hoạch n đối tượng thành k cụm sao cho **tổng bình phương khoảng cách nhỏ nhất** (với $c_i$ là centroid/medoid của cụm $C_i$):

$$\min \sum_{i=1}^{k} \sum_{x \in C_i} d(x, c_i)^2$$

- Cho trước k, tìm phân hoạch k cụm tối ưu tiêu chí đã chọn:
  - **Tối ưu toàn cục**: liệt kê vét cạn mọi phân hoạch (không khả thi).
  - **Phương pháp heuristic**: k-means, k-medoids.

### 6.1. K-Means (MacQueen '67, Lloyd '57/'82)
Mỗi cụm đại diện bởi **tâm (center)** của cụm. Bốn bước:
1. Phân hoạch đối tượng thành k tập con **không rỗng**.
2. Tính seed point là **centroid** của các cụm trong phân hoạch hiện tại.
3. Gán mỗi đối tượng vào cụm có seed point **gần nhất**.
4. Quay lại bước 2; dừng khi phép gán **không đổi**.

**Ưu điểm**: hiệu quả — $O(tkn)$ với n là số đối tượng, k số cụm, t số vòng lặp; thường $k, t \ll n$.

**Nhược điểm**:
- Phải chỉ định **k trước** (có cách tự động xác định k tối ưu — xem Hastie et al., 2009).
- **Nhạy với nhiễu và outlier**.
- Không phù hợp phát hiện cụm **hình dạng không lồi**.

### 6.2. K-Medoids / PAM (Kaufman & Rousseeuw '87)
Mỗi cụm đại diện bởi **một đối tượng thực** trong cụm. Thuật toán:
1. Chọn ngẫu nhiên k điểm trong n điểm làm **medoid**.
2. Gán mỗi điểm dữ liệu vào medoid **gần nhất** (bằng metric khoảng cách thông thường).
3. Trong khi **chi phí giảm**:
   - Với mỗi medoid m, mỗi điểm o không phải medoid: **hoán đổi** m và o, gán lại điểm vào medoid gần nhất, tính lại chi phí.
   - Nếu tổng chi phí **lớn hơn** bước trước → **hoàn tác** hoán đổi.

## 7. Phân cụm Phân cấp (Hierarchical Clustering)

- Dùng **ma trận khoảng cách** làm tiêu chí phân cụm.
- **Không yêu cầu số cụm k** làm đầu vào.

### Hai hướng
- **Bottom-up (Agglomerative — nhập)**:
  1. Dựng mọi cụm đơn (mỗi cụm một đối tượng).
  2. Tham lam **gộp** hai cụm "tương tự nhất" (khoảng cách nhỏ nhất) thành cụm mới.
  3. Lặp đến khi mọi đối tượng nằm trong một cụm duy nhất.
- **Top-down (Divisive — phân)**:
  1. Xuất phát với một cụm chứa mọi đối tượng.
  2. Tham lam **tách** cụm thành hai, gán đối tượng sao cho **tối đa tương tự trong nhóm**.
  3. Tiếp tục tách các cụm **ít gắn kết nhất** cho đến khi chỉ còn cụm đơn hoặc đạt số cụm mong muốn.

### Khoảng cách giữa hai cụm
| Đo lường | Định nghĩa |
|---|---|
| **MIN (Single Link)** | Khoảng cách **ngắn nhất** giữa hai điểm x, y thuộc hai cụm A, B khác nhau |
| **MAX (Complete Link)** | Khoảng cách **xa nhất** giữa hai điểm x, y thuộc hai cụm A, B khác nhau |
| **Group Average** | Khoảng cách **trung bình** giữa mọi cặp điểm hai cụm (số điểm cụm j là $n_j$): $\frac{1}{n_i n_j}\sum_{x \in A}\sum_{y \in B} d(x,y)$ |

## 8. Phân cụm Dựa trên Mật độ (Density-Based Clustering)

- Phân cụm dựa trên **mật độ (local cluster criterion)**, như các điểm **density-connected**.

### Đặc điểm chính
- Phát hiện cụm **hình dạng tùy ý**.
- Xử lý được **noise**.
- Chỉ cần **quét một lần** (one scan).
- Cần tham số mật độ làm điều kiện dừng.

### Hai tham số
- **Eps**: bán kính tối đa của lân cận.
- **MinPts**: số điểm tối thiểu trong lân cận Eps của một điểm.
- $N_{Eps}(p) = \{q \in D \mid dist(p, q) \le Eps\}$

### Các khái niệm then chốt
- **Directly density-reachable**: p trực tiếp đạt được mật độ từ q (w.r.t. Eps, MinPts) nếu $p \in N_{Eps}(q)$ và $|N_{Eps}(q)| \ge MinPts$.
- **Density-reachable**: p đạt được mật độ từ q nếu tồn tại chuỗi điểm $p_1, \ldots, p_n$, $p_1 = q$, $p_n = p$ sao cho $p_{i+1}$ trực tiếp đạt được mật độ từ $p_i$.
- **Density-connected**: p kết nối mật độ với q nếu tồn tại điểm o sao cho cả p và q đều density-reachable từ o.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **Cụm** được định nghĩa là tập các điểm **density-connected** tối đại (maximal).
- Thuật toán:
  1. Chọn tùy ý một điểm p.
  2. Truy hồi mọi điểm density-reachable từ p (w.r.t. Eps, MinPts).
  3. Nếu p là **core point** → hình thành một cụm.
  4. Nếu p là **border point** → không có điểm nào density-reachable từ p, DBSCAN chuyển sang điểm kế tiếp.
  5. Lặp lại cho đến khi mọi điểm được xử lý.

## 9. SOM — Self Organizing Map (Bản đồ tự tổ chức)

### Mô tả
- SOM ánh xạ dữ liệu **chiều cao** onto **không gian 2 chiều**.
- Không quá nhạy với một số dữ liệu nhiễu, chất lượng phân cụm vẫn được bảo đảm.

### Kiến trúc
- Mỗi **nơ-ron (node)** tương ứng một tập mẫu từ dataset.
- Mỗi nơ-ron liên kết một **vector trọng số (codebook)**.

### Thuật toán
Input: dataset, kích thước và topology của map; Output: codebook cho mỗi node.
1. Khởi tạo ngẫu nhiên trọng số các node của map.
2. Chọn ngẫu nhiên một instance.
3. Tìm node gần nhất: **Best Matching Unit (BMU)**.
4. Cập nhật codebook của node này.
5. Codebook của các node **lân cận** cũng được cập nhật, nhưng **không cùng mức độ**.
6. **Giảm dần cường độ** cập nhật theo thời gian.
7. Lặp bước 2–6 trong $T_{max}$ vòng lặp.

### Ưu điểm
- Rất đơn giản.
- Phân loại dữ liệu tốt, dễ đánh giá chất lượng (tính được map tốt thế nào và độ tương tự giữa các đối tượng mạnh ra sao).

### Nhược điểm
- SOM tổ chức dữ liệu sao cho trong sản phẩm cuối, các mẫu thường được bao quanh bởi mẫu tương tự — nhưng **mẫu tương tự không phải lúc nào cũng nằm cạnh nhau**.

## 10. GHSOM — Growing Hierarchical SOM

- **Ý tưởng then chốt**: dùng **cấu trúc phân cấp nhiều tầng**, mỗi tầng gồm **một số SOM độc lập**.
- Khắc phục hạn chế của SOM phẳng: cụm lớn/khó tách được "trồng" thêm SOM con ở tầng dưới.

## 11. Cluster Labelling (Gán nhãn cụm)

### 11.1. Entropy hợp đồng (Joint entropy)
- Entropy hợp đồng của hai biến ngẫu nhiên rời rạc X, Y với phân phối hợp đồng $p(x, y)$:

$$H(X, Y) = -\sum_{x}\sum_{y} p(x, y) \log p(x, y)$$

- **Entropy điều kiện** $H(Y \mid X)$ lượng hóa **mức bất định còn lại của Y** khi biết kết quả của X.

### 11.2. Chain rule cho entropy
- Tương tự (và suy ra trực tiếp từ) chain rule của xác suất:

$$H(X, Y) = H(X) + H(Y \mid X)$$

### 11.3. Mutual Information (Thông tin tương hỗ, MI)
- MI mô tả **lượng thông tin về một biến** thu được qua biến kia, hoặc mức phân phối hợp đồng khác biệt so với độc lập thuần túy.
- $I(X; Y)$ là **Kullback–Leibler Divergence** giữa phân phối hợp đồng và phân phối tích $p(x)p(y)$.
- Định nghĩa viết lại:

$$I(X; Y) = \sum_{x}\sum_{y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)} = H(X) + H(Y) - H(X, Y)$$

### 11.4. Chi-square test (Kiểm định Chi bình phương)
- **Kiểm định thống kê hình thức** để xác định kết quả có **ý nghĩa thống kê (statistically significant)** hay không.
- Ví dụ: giả định dịch outbreak Salmonella trên du thuyền — phỏng vấn 300 người, 60 người có triệu chứng; khảo sát cho thấy nhiều bệnh nhân ăn cà chua từ salad bar.

**Điều kiện tiến hành chi-square**:
- Tổng ít nhất **30 quan sát** trong bảng.
- **Mỗi ô chứa ≥ 5** quan sát.

**Cách tiến hành**: so sánh dữ liệu **quan sát (observed)** với dữ liệu **kỳ vọng (expected)**:

$$\chi^2 = \sum_{\text{các ô}} \frac{(O - E)^2}{E}$$

**Diễn giải**: nhìn chung, **giá trị chi-square càng cao, khả năng có khác biệt ý nghĩa thống kê giữa hai nhóm so sánh càng lớn**.

---

## Ghi chú liên hệ đề tài RAG

Chương 5 liên hệ với đề tài ở các điểm:
- **Document clustering** (mục 1): phân cụm kho tài liệu/chunk để hiểu phân bố chủ đề trước khi embedding; hỗ trợ xác định chủ đề của corpus.
- **K-Means / Hierarchical / DBSCAN** (mục 6–8): có thể dùng đánh giá chất lượng chunking (cụm chunk nhỏ quá/lớn quá), hoặc gom chunk tương tự giảm trùng lặp.
- **Mutual Information & Chi-square** (mục 11): chọn đặc trưng/từ khóa đại diện cho từng cụm tài liệu; kiểm định ý nghĩa thống kê khi so sánh kết quả các thí nghiệm RAG (theo mục "Statistical Interpretation" trong AGENTS.md).
