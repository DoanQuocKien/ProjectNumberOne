# **ĐỀ BÀI: HỆ THỐNG GỢI Ý SẢN PHẨM CÁ NHÂN HÓA (PERSONALIZED ITEM RECOMMENDATION \- PIR)**

## **1\. Tổng quan bài toán (Problem Statement)**

Mục tiêu của bài toán là xây dựng một mô hình Machine Learning/Deep Learning để dự đoán và gợi ý danh sách các sản phẩm mà một người dùng cụ thể có khả năng sẽ mua trong tháng tiếp theo. Bài toán đặc biệt chú trọng vào việc giải quyết bài toán gợi ý theo thời gian thực (time-aware) và xử lý nhóm người dùng mới (cold-start users).

## **2\. Dữ liệu cung cấp (Dataset)**

Hệ thống được cung cấp các luồng dữ liệu lịch sử trong năm **2025**, bao gồm:

* **Dữ liệu giao dịch (Transaction Data):** Dữ liệu mua hàng chính thức, được ghi nhận dưới nhãn sự kiện là purchased.  
* **Dữ liệu hành vi (Event Data):** Các tín hiệu ngầm (implicit feedback) thể hiện sự quan tâm của người dùng, bao gồm các sự kiện như xem sản phẩm (view\_item) và thêm vào giỏ hàng (add-to-cart / ATC).  
* *Lưu ý:* Cần kết hợp và trích xuất đặc trưng (Feature Extraction) từ cả hai nguồn dữ liệu này để tạo thành một tập dataset hoàn chỉnh cho việc huấn luyện.

## **3\. Giao thức Đánh giá & Phân chia Dữ liệu (Evaluation Protocol & Data Split)**

Để tránh hiện tượng rò rỉ dữ liệu (Data Leakage) — một lỗi rất nghiêm trọng trong các mô hình dự báo chuỗi thời gian, tập dữ liệu sẽ được chia cắt nghiêm ngặt theo trục thời gian (Out-of-Time Validation):

* **Train Set:** Dữ liệu từ đầu năm **2025** đến hết tháng **10/2025**.  
* **Validation Set (val):** Dữ liệu trong tháng **11/2025** (Dùng để tinh chỉnh tham số).  
* **Internal Test Set (test nội bộ):** Dữ liệu trong tháng **12/2025** (Dùng để so sánh các kiến trúc mô hình).  
* **Private Test Set (blind test):** Dữ liệu trong tháng **1/2026**. Tập dữ liệu này bị ẩn nhãn trong quá trình phát triển và **chỉ tính sự kiện purchased** làm Ground Truth để chấm điểm xếp hạng cuối cùng.

## **4\. Quy trình Huấn luyện và Suy diễn (The 5-Step Pipeline)**

Dựa trên sơ đồ bảng trắng, luồng công việc yêu cầu thực hiện chuẩn xác theo 5 bước sau:

* **Bước 1 \- Huấn luyện mô hình cơ sở:** Sử dụng tập $X\_{train}$ (dữ liệu $\\le$ 10/2025) kết hợp nhãn $Y\_{train}$ để huấn luyện các kiến trúc mô hình ứng viên khác nhau (ví dụ: mô hình $f\_A, f\_B, f\_C$).  
* **Bước 2 \- Suy diễn kiểm thử:** Dùng các mô hình $f\_A, f\_B, f\_C$ đã huấn luyện tiến hành dự đoán (inference) trên tập dữ liệu đánh giá, cho ra các kết quả dự đoán tương ứng $\\tilde{y}\_A, \\tilde{y}\_B, \\tilde{y}\_C$.  
* **Bước 3 \- Đánh giá và Chọn lọc (Model Selection):** Đối chiếu các dự đoán $\\tilde{y}\_A, \\tilde{y}\_B, \\tilde{y}\_C$ với nhãn thực tế $y\_{test}$ (của tập Internal Test 12/2025). Tính toán các độ đo để tìm ra mô hình có hiệu năng cao nhất. Giả sử mô hình $B$ cho kết quả tốt nhất ($B$ là best).  
* **Bước 4 \- Tái huấn luyện (Retrain):** Gộp toàn bộ dữ liệu lịch sử đã biết bao gồm Train, Val và Internal Test thành tập $(X\_{train+val+test}, y\_{train+val+test})$. Dùng dữ liệu này để huấn luyện lại kiến trúc mô hình $B$, ta thu được siêu mô hình cuối cùng là $f^\*\_B$.  
* **Bước 5 \- Dự đoán thực tế (Final Inference):** Dùng mô hình $f^\*\_B$ để trích xuất đặc trưng và dự đoán $\\tilde{y}\_{private}$ cho tập Private Test tháng **1/2026**.

## **5\. Yêu cầu Đầu ra (Output Format)**

Mô hình phải trả về một file kết quả (submission) dưới định dạng **Dictionary**, mapping giữa định danh khách hàng và danh sách các sản phẩm được gợi ý:

JSON

{  
  "customer\_id\_1": \["item\_A", "item\_B", "item\_C", ...\],  
  "customer\_id\_2": \["item\_X", "item\_Y", "item\_Z", ...\],  
  ...  
}

*Mục đích:* Đầu ra này sẽ được cắm trực tiếp vào hệ thống để hiển thị luồng gợi ý trên Nền tảng Web hoặc màn hình Kiosk.

## **6\. Tiêu chí Đánh giá (Metrics)**

* **Phạm vi đánh giá:** Hệ thống *chỉ* tính điểm trên những khách hàng thực sự có phát sinh giao dịch (mua hàng) trong tháng **1/2026**. Đặc biệt chú ý: Tập đánh giá **bao gồm cả những khách hàng mới** tạo tài khoản trong tháng 1/2026 (Bài toán Cold Start).  
* **Các độ đo (Metrics) sử dụng:**  
  1. **Total Correct Hits:** Đếm tổng số lượng sản phẩm gợi ý trúng (khớp với sản phẩm khách hàng thực tế mua).  
  2. **IoU (Intersection over Union):** Tỷ lệ phần giao giữa tập sản phẩm gợi ý và tập sản phẩm thực mua, chia cho phần hợp của hai tập này.  
  3. **MRR (Mean Reciprocal Rank):** Đánh giá dựa trên công thức $\\frac{1}{rank(\\text{first hit item})}$. Độ đo này trừng phạt rất mạnh nếu sản phẩm trúng đích đầu tiên bị xếp ở vị trí quá thấp.  
  4. **Precision@10:** Tỷ lệ sản phẩm gợi ý đúng trong Top 10 sản phẩm đầu tiên. (Rất quan trọng cho không gian hiển thị giới hạn như màn hình Kiosk).  
  5. **MAP (Mean Average Precision):** Độ đo trung bình tính toán chất lượng của toàn bộ thứ tự xếp hạng các sản phẩm được gợi ý.

