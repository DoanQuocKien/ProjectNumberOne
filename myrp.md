# Báo cáo Phân tích Kinh doanh và Chất lượng Dữ liệu

## 1. Tổng quan bộ dữ liệu

Bộ dữ liệu là một **bảng sự kiện (event log)** ở mức dòng, ghi nhận hành vi phát sinh của khách hàng theo thời gian. 
Mỗi dòng tương ứng với một sự kiện gắn với một khách hàng, một mặt hàng, số lượng và thời điểm phát sinh.

### Quy mô dữ liệu

| Chỉ số | Giá trị |
|---|---:|
| Số dòng | 37,684,823 |
| Số dòng trùng | 23,815 |
| Tỷ lệ trùng ước tính | 0.0632% |

### Khoảng thời gian dữ liệu

| Mốc thời gian | Giá trị |
|---|---|
| Sớm nhất | 2025-01-01 06:51:05.690000 |
| Muộn nhất | 2025-12-01 22:32:29.003000 |

## 2. Kiểm tra chất lượng dữ liệu

### Đánh giá nhanh

- **Độ phủ thời gian:** Dữ liệu trải dài gần **11 tháng**, đủ tốt để phân tích xu hướng theo thời gian và xem tính mùa vụ.
- **`quantity` hợp lệ:** Không phát hiện dòng nào có số lượng nhỏ hơn hoặc bằng 0 ở mức kiểm tra cơ bản.
- **Dòng trùng:** Tỷ lệ trùng lặp ở mức **0.0632%**, khá thấp. Tuy nhiên, vẫn nên đối chiếu nghiệp vụ để xác định đây là sự kiện mua hàng hợp lệ hay lỗi ghi log.

## 3. Phân tích hoạt động kinh doanh

### 3.1. Xu hướng doanh thu theo thời gian

Dựa trên biểu đồ tổng hợp doanh thu theo tháng:

- Doanh thu duy trì ở mức tốt trong hầu hết các tháng.
- **Tháng 8, tháng 10 và đặc biệt tháng 11** ghi nhận mức tăng trưởng mạnh, đạt đỉnh cao nhất trong năm.
- **Tháng 2** có dấu hiệu chững lại, có thể liên quan đến kỳ nghỉ lễ hoặc tính mùa vụ.
- **Tháng 12 thấp** là do dữ liệu mới chỉ ghi nhận đến hết ngày 01/12/2025.

### 3.2. Top địa điểm kinh doanh

Hoạt động bán hàng tập trung mạnh vào một số chi nhánh trọng điểm tại các thành phố lớn.

| Xếp hạng | Địa điểm | Doanh thu ước tính | Khách hàng duy nhất | Hóa đơn |
|---|---|---:|---:|---:|
| 1 | HCM - 9 – 11 – 13 Nguyễn Trãi | ~49.48 tỷ VNĐ | 26,955 | 70,622 |
| 2 | DNA - 81 - 83 Nguyễn Văn Linh | ~36.48 tỷ VNĐ | - | - |
| 3 | HCM - 55B Phan Đăng Lưu | ~35.77 tỷ VNĐ | - | - |

**Nhận xét:**
- Các chi nhánh dẫn đầu đều có lượng khách hàng duy nhất rất lớn.
- Nhóm cửa hàng top doanh thu thường nằm ở khu vực có mật độ giao dịch cao và tệp khách hàng rộng.

### 3.3. Sản phẩm chủ lực (Top Items)

Một số mã sản phẩm đóng góp doanh thu vượt trội so với phần còn lại.

| Xếp hạng | Mã hàng | Doanh thu ước tính | Sản lượng | Khách hàng |
|---|---|---:|---:|---:|
| 1 | 3773000000004 | ~151.17 tỷ VNĐ | 196,977 | gần 40,000 |
| 2 | 3774000000003 | ~105.13 tỷ VNĐ | - | - |
| 3 | 7176000000002 | ~75.8 tỷ VNĐ | 1.68 triệu | - |

**Nhận xét:**
- Mã hàng **3773000000004** đang là sản phẩm chủ lực về doanh thu.
- Mã hàng **7176000000002** có giá trung bình thấp nhưng sản lượng cực lớn, cho thấy đây là nhóm hàng bán theo volume.

### 3.4. Hành vi khách hàng tiêu biểu (Top Customers)

Tệp khách hàng giá trị cao thể hiện tần suất mua và giá trị đơn hàng rất tốt.

| Xếp hạng | Khách hàng | Tổng chi tiêu | Số hóa đơn | AOV |
|---|---|---:|---:|---:|
| 1 | 8338390 | >966 triệu VNĐ | 378 | - |
| 2 | 4243161 | 811 triệu VNĐ | - | >5.7 triệu VNĐ |
| 3 | 5405875 | gần 247 triệu VNĐ | 47 | - |

**Nhận xét:**
- Các khách hàng top đầu đều là **khách lặp lại** (`repeat_customer_flag = 1`).
- Có cả nhóm khách chi tiêu cao theo tần suất lớn và nhóm mua số lượng lớn theo kiểu mua sỉ.

## 4. Kết luận và định hướng

### Kết luận

- **Dữ liệu:** Cấu trúc đã ổn định, thời gian được chuẩn hóa tốt và phù hợp cho các bài toán phân tích sâu như dự báo doanh số, phân khúc khách hàng hoặc phát hiện bất thường.
- **Kinh doanh:** Doanh nghiệp có một tệp khách hàng trung thành chi tiêu lớn, đồng thời sở hữu một số mã hàng đóng góp doanh thu rất mạnh.
- **Mùa vụ:** Các tháng cuối năm đang là giai đoạn cao điểm cần được ưu tiên theo dõi.

### Hành động tiếp theo

1. Tối ưu tồn kho cho các sản phẩm doanh thu cao như **3773000000004** và các sản phẩm có volume lớn như **7176000000002**. 
2. Nhân rộng mô hình vận hành tại các chi nhánh có hiệu suất cao như **Nguyễn Trãi** và **Phan Đăng Lưu**. 
3. Thiết kế các chương trình **Loyalty Program** và ưu đãi riêng cho nhóm khách hàng mua nhiều, khách hàng sỉ và khách có AOV cao.


# TÀI LIỆU THIẾT KẾ & TRIỂN KHAI KIẾN TRÚC RECOMMENDATION SYSTEM

**Quy mô bài toán:** Dữ liệu Implicit Feedback, 37.6 triệu sự kiện giao dịch, 11 tháng.
**Mục tiêu đầu ra:** API trả về `Dictionary[customer_id] -> List[Top_K_Items]` với K=10.
**Metrics cốt lõi:** MAP, Precision@10, 1/Rank (MRR), IoU, Total Hits (đánh giá trên cả Warm-start và Cold-start).

---

## PHẦN 1: PIPELINE XỬ LÝ DỮ LIỆU ĐẦU VÀO (DATA PREPARATION)

Trước khi đưa vào bất kỳ mô hình nào, tập dữ liệu gốc cần được chuyển đổi thành các cấu trúc ma trận hoặc chuỗi (sequence) chuẩn mực.

### 1.1. Xử lý Implicit Feedback Matrix (Cho MF/KNN)
Thay vì dùng số lượng `quantity`, trong bài toán Ranking, ta quan tâm đến "tần suất mua" (confidence) hoặc "có mua hay không" (binary).
* **Biến đổi:** Khởi tạo ma trận $R_{u,i}$ (User-Item Matrix). 
    * $R_{u,i} = 1$ nếu khách hàng $u$ từng mua mặt hàng $i$.
    * $R_{u,i} = 0$ cho các item chưa tương tác (Unobserved).
* **Công cụ:** Sử dụng `scipy.sparse.csr_matrix` để lưu trữ nhằm tiết kiệm RAM, vì ma trận này có độ thưa thớt (sparsity) cực kỳ cao.

### 1.2. Xử lý Sequential Data (Cho Deep Learning/SASRec)
Nhóm dữ liệu theo `customer_id` và sắp xếp tăng dần theo `timestamp`.
* **Cấu trúc:** `User_A: [item_x, item_y, item_z, ..., item_target]`
* **Kỹ thuật Padding/Truncating:** Đặt một ngưỡng chiều dài chuỗi tối đa (ví dụ $L = 20$). Nếu khách hàng mua nhiều hơn 20 món, cắt lấy 20 món mới nhất (Truncating). Nếu ít hơn, đệm thêm giá trị `0` (Padding).

---

## PHẦN 2: CHI TIẾT CÁC MÔ HÌNH TRIỂN KHAI (IMPLEMENTATION STRATEGIES)

Để tối ưu toàn bộ các metrics được yêu cầu, hệ thống cần được thiết kế theo dạng **Hybrid Pipeline** (Kết hợp nhiều mô hình tùy thuộc vào phân khúc khách hàng).

### Mô hình 1: Heuristic Fallback (Quy tắc nghiệp vụ)
**Mục đích:** Đảm bảo điểm số không bị trượt về 0 đối với tập khách hàng Cold-start (tạo tài khoản tháng 1/2026).
* **Kiến trúc:** Không dùng Machine Learning. Dựa hoàn toàn vào thống kê đếm (Frequency Counting).
* **Chiến lược 1 (Global Top-K):** Lấy 10 mặt hàng có doanh thu cao nhất (như mã `3773000000004`) và sản lượng cao nhất (`7176000000002`).
* **Chiến lược 2 (Location-based Top-K):** Nhóm dữ liệu theo `store_id` (VD: Nguyễn Trãi, Phan Đăng Lưu). Lấy Top 10 item phổ biến nhất của từng cửa hàng. Khi user mới tạo tài khoản thuộc khu vực nào, map với list của khu vực đó.
* **Cài đặt:** Dùng `Polars` cho tốc độ xử lý song song. Cấu trúc output chuẩn bị sẵn một Dictionary tĩnh, load thẳng vào RAM (Redis) để query với độ trễ < 5ms.

### Mô hình 2: BPR - Matrix Factorization (Tối ưu Ranking)
**Mục đích:** Xử lý nhóm khách hàng Warm-start, tập trung tối ưu hóa các Ranking metrics như MAP và 1/Rank.
* **Lý thuyết:** Thay vì cố gắng dự đoán giá trị tương tác, BPR (Bayesian Personalized Ranking) tối ưu hóa sự khác biệt giữa một mặt hàng đã mua (positive item $i$) và một mặt hàng chưa mua (negative item $j$).
* **Hàm Loss:** Hàm mục tiêu của BPR được định nghĩa bằng LaTeX như sau:
    $$\min_{\Theta} \sum_{(u,i,j) \in D} -\ln \sigma(\hat{x}_{ui} - \hat{x}_{uj}) + \lambda ||\Theta||^2$$
    *(Trong đó $\hat{x}_{ui}$ là điểm số dự đoán user $u$ thích item $i$, $\sigma$ là hàm Sigmoid).*
* **Triển khai:** Khuyến nghị sử dụng thư viện `implicit` trong Python (`implicit.bpr.BayesianPersonalizedRanking`). Cực kỳ tối ưu nhờ nhân C++ và hỗ trợ GPU/CUDA.

### Mô hình 3: SASRec (Self-Attentive Sequential Recommendation)
**Mục đích:** Tận dụng tối đa yếu tố thời gian và sự liên tục trong giỏ hàng (bắt trend cá nhân trong ngắn hạn).
* **Kiến trúc:** Là một phiên bản thu gọn của Transformer (chỉ dùng Decoder/Encoder blocks với Masked Self-Attention).
* **Cách hoạt động:** Mô hình nhìn vào chuỗi hành vi $[i_1, i_2, ..., i_{t-1}]$ để học các pattern tương quan phức tạp, từ đó sinh ra embedding dự đoán cho item thứ $i_t$.
* **Ưu điểm:** Khắc phục được nhược điểm của MF là "quên mất thứ tự mua hàng". Phù hợp với đặc thù khách lặp lại (repeat_customer) mua sỉ theo chu kỳ.
* **Triển khai:** Xây dựng bằng `PyTorch`. Do dữ liệu 37.6M dòng là khá lớn, cần cấu hình Dataloader dùng Negative Sampling (tỉ lệ 1 positive : 4 negatives) cho mỗi batch.

### Mô hình 4: Two-Tower (Kiến trúc chuẩn Công nghiệp)
**Mục đích:** Hệ thống End-to-End mạnh nhất, tích hợp được cả tính năng linh hoạt và tốc độ truy vấn tại Kiosk.
* **Kiến trúc (PyTorch/TensorFlow):**
    * **User Tower:** Nhận input là lịch sử các item đã mua (Sequence), `store_id`, `created_time`. Đẩy qua các lớp MLP (Multi-Layer Perceptron) hoặc GRU/Transformer để ra một vector đại diện (User Embedding) độ dài 64/128 chiều.
    * **Item Tower:** Nhận input là `item_id`, `category`, `price_level`, `volume_flag`. Ra một vector đại diện (Item Embedding).
* **Cách huấn luyện:** Dùng In-batch Negative Loss (InfoNCE Loss). Tối ưu hóa Dot-Product (Tích vô hướng) giữa User Vector và Item Vector.
* **Tích hợp Kiosk/Web (Inference):**
    * Toàn bộ hàng trăm ngàn Item Vectors được tính trước (pre-compute) và nạp vào **FAISS** (Facebook AI Similarity Search).
    * Khi có khách quẹt thẻ (Query), đẩy đặc trưng khách qua User Tower (mất ~10ms), lấy User Vector query vào FAISS để lấy Top-10 (mất ~1ms). Tổng thời gian hoàn hảo cho Kiosk thời gian thực.

---

## PHẦN 3: ĐO LƯỜNG VÀ ĐÁNH GIÁ (EVALUATION PROTOCOL)

Kịch bản chia Data: Train (Tháng 1 -> 11/2025) / Validation (Tuần cuối tháng 11) / Test (Tập giao dịch sinh ra trong tháng 1/2026).

Việc tính toán Matrix bằng code Python thuần túy:

**1. 1/Rank (Mean Reciprocal Rank - MRR):**
Tìm vị trí $pos$ của item đúng (ground-truth) xuất hiện sớm nhất trong list gợi ý (Top K).
$$MRR = \frac{1}{|U|} \sum_{u=1}^{|U|} \frac{1}{pos_u}$$
*(Nếu không có item nào trúng trong Top K, điểm MRR = 0).*

**2. Precision@10:**
Với list 10 món (K=10), tính phần trăm số món thực sự được khách mua.
$$Precision@K = \frac{| \text{Relevant Items} \cap \text{Top-K Items} |}{K}$$

**3. MAP (Mean Average Precision):**
Tính Average Precision (AP) cho từng user, ưu tiên cực mạnh việc item trúng nằm ở top đầu.
$$AP@K = \frac{1}{\min(m, K)} \sum_{k=1}^K P(k) \times rel(k)$$
*(Trong đó $m$ là tổng số món khách đã mua, $P(k)$ là precision tại vị trí $k$, $rel(k) \in \{0,1\}$ báo hiệu hit/miss tại vị trí $k$).* Tổng hợp toàn user ta có MAP.


# KHUNG PHÂN TÍCH CHUYÊN SÂU MÔ HÌNH RECOMMENDATION (MODEL DEEP-DIVE ANALYSIS)

**Mục tiêu:** Không chỉ dừng lại ở việc nhìn vào các chỉ số tổng (Global Metrics) như MAP hay Precision@10, báo cáo này đưa ra các hướng đi chi tiết để "mổ xẻ" hành vi của mô hình, phát hiện bias (thiên lệch) và đảm bảo list recommend đáp ứng đúng bài toán kinh doanh trên Web và Kiosk.

---

## 1. Phân tích Thiên lệch Độ phổ biến (Popularity Bias & Coverage Analysis)

Các mô hình Implicit Feedback rất dễ rơi vào trạng thái "lười biếng", tức là chỉ gợi ý những mặt hàng bán chạy nhất cho tất cả mọi người. Với dữ liệu ghi nhận mã hàng `7176000000002` có volume lên tới 1.68 triệu, rủi ro này là cực kỳ lớn.

**Các hướng phân tích cần thực hiện:**
* **Đo lường Item Coverage (Độ phủ danh mục):** Tính tỷ lệ phần trăm các SKU trong kho hàng từng xuất hiện trong danh sách Top-K recommend. Nếu Coverage < 10%, model của bạn đang bị overfit vào nhóm Best-seller.
* **Head vs. Torso vs. Tail Analysis:** * Chia toàn bộ danh mục sản phẩm thành 3 nhóm dựa trên tần suất bán lịch sử (Head: Top 20% volume, Tail: Bottom 50% volume).
    * Tính `Hit Rate` và `Precision@10` riêng cho từng nhóm. 
    * *Insight kỳ vọng:* Nếu model chỉ đoán trúng ở nhóm Head mà hoàn toàn miss ở nhóm Tail, model này không mang lại tính "khám phá" (Novelty) cho người dùng trên Web.
* **Gini Coefficient của tập dự đoán:** Kiểm tra mức độ bất bình đẳng trong phân phối các item được gợi ý. Nếu chỉ số Gini tiến sát 1, hệ thống của bạn thực chất chỉ là một "Global Popularity Model" trá hình.

## 2. Phân rã Hiệu năng theo Phân khúc Khách hàng (Cohort-based Breakdown)

Đánh giá trung bình trên toàn bộ user sẽ che lấp sự thất bại của model trên các nhóm khách hàng yếu thế (như Cold-start user trong tháng 1/2026).

**Các hướng phân tích cần thực hiện:**
* **Phân tích theo History Length (Độ dài lịch sử):**
    * Nhóm 1: `history_len == 0` (Cold-start, user mới tạo tháng 1/2026).
    * Nhóm 2: `1 <= history_len <= 5` (Casual user).
    * Nhóm 3: `history_len >= 50` (VIP, Wholesale user - ví dụ khách hàng `8338390`).
    * *Hành động:* Vẽ biểu đồ xu hướng của chỉ số `MRR (1/rank)` theo `history_len`. Trọng tâm là debug nhóm 1 (phải dùng Rule-based) và nhóm 3 (nơi model Collaborative Filtering phải tỏa sáng).
* **Phân tích theo Khu vực Địa lý (Location-based):**
    * Trích xuất `store_id` từ lịch sử. So sánh `MAP` của khách hàng mua tại HCM (Nguyễn Trãi) vs. khách hàng tại Đà Nẵng (Nguyễn Văn Linh).
    * *Insight kỳ vọng:* Phát hiện xem model có đang thiên vị (bias) thị hiếu của người dân miền Nam và ép kết quả đó lên Kiosk ở miền Trung hay không.

## 3. Phân tích Sự suy thoái theo Thời gian (Temporal Degradation & Leakage)

Dữ liệu kéo dài 11 tháng và có tính mùa vụ mạnh, đạt đỉnh cao nhất vào tháng 11. Việc dự đoán cho tháng 1/2026 có thể bị nhiễu nghiêm trọng.

**Các hướng phân tích cần thực hiện:**
* **Đo lường Time-to-Last-Purchase (Tác động của độ trễ):**
    * Chia khách hàng theo thời điểm mua hàng cuối cùng (Recency): < 7 ngày, 7-30 ngày, > 90 ngày (Dormant user).
    * Chỉ số `IoU` và `Hit Rate` chắc chắn sẽ giảm khi Recency tăng lên. Cần đo lường tốc độ "quên" (forgetting curve) của mô hình.
* **Kiểm tra Rò rỉ Mùa vụ (Seasonality Leakage Trap):**
    * Phân tích top các item được model gợi ý nhiều nhất cho tháng 1/2026. Nếu danh sách này chứa toàn các mã hàng đặc thù của tháng 11 hoặc đồ mùa đông/lễ hội, model đã bị overfit vào Time-series.
    * *Hành động:* Thực hiện Ablation Study bằng cách train thử một model loại bỏ hoàn toàn dữ liệu tháng 10 và tháng 11, sau đó so sánh `Precision@10` trên tập test tháng 12.

## 4. Phân tích Sâu Nhóm Xếp hạng (Rank-Aware Analysis)

Chỉ số `Precision@10` không cho biết item đúng nằm ở vị trí thứ 1 hay thứ 10. `1/rank (first hit)` và `MAP` giải quyết được việc này, nhưng cần phân tích trực quan hơn.

**Các hướng phân tích cần thực hiện:**
* **Phân phối Vị trí Trúng (Hit Position Distribution):**
    * Vẽ biểu đồ Histogram thể hiện tần suất `first hit item` rơi vào các rank từ 1 đến 10.
    * *Insight kỳ vọng:* Đường cong lý tưởng phải lệch trái cực mạnh (Right-skewed), tức là đa số Hits rơi vào Rank 1, 2, 3. Nếu phân phối đồng đều hoặc tập trung ở cuối, thuật toán Ranking (như BPR loss) đang cấu hình sai hyper-parameters.
* **Phân tích IoU theo Kích thước Giỏ hàng (Basket Size Analysis):**
    * Với những khách hàng mua sỉ (ví dụ mua 20 mã SKU cùng lúc), `IoU` của Top 10 recommend toán học bị giới hạn (tối đa chỉ đạt 0.5).
    * *Hành động:* Tách riêng metric `IoU` cho nhóm khách mua lẻ (AOV thấp, basket size < 3) và khách mua sỉ để có góc nhìn công bằng về độ bao phủ.

## 5. Đánh giá Mức độ Đóng góp Kinh doanh (Business Metric Alignment)

Model có Hit Rate cao chưa chắc đã tốt cho doanh nghiệp nếu nó chỉ gợi ý các mặt hàng rẻ tiền. 

**Các hướng phân tích cần thực hiện:**
* **Revenue Proxy vs. Hit Rate:**
    * Tính toán Giá trị Trung bình (AOV) của danh sách Top-K recommend. 
    * So sánh doanh thu mô phỏng (Simulated Revenue = ∑ (Hit Items × Unit Price)) của Model so với baseline (ví dụ baseline là gợi ý Random hoặc Top Trending).
* **Discount Cannibalization Check:**
    * Kiểm tra xem model có đang ưu tiên xếp hạng (rank) các mặt hàng đang giảm giá sâu (Promotions) lên trên hay không. Nếu có, model đang vô tình bào mòn lợi nhuận (Gross Margin) của công ty bằng cách biến khách mua nguyên giá thành khách săn sale.

