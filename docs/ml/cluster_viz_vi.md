# Advanced clustering · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm. Thuật toán EM và DBSCAN chạy ngay trong trình duyệt, đúng như thuật toán P.1 và P.2 ở bài [Phân cụm nâng cao](ml/cluster_vi).

## EM từng bước và DBSCAN

<small class="viz-ref">Mục P.2–P.5 · hình P.3, P.6, P.7</small>

**1. EM cho mô hình hỗn hợp Gauss.** Thử các việc sau:

- Bấm xen kẽ **Bước E** và **Bước M**. Sau bước E, màu mỗi điểm trộn theo xác suất thuộc từng cụm. Sau bước M, các elip co giãn và xoay theo dữ liệu. Log-likelihood ở biểu đồ bên phải không bao giờ giảm.
- Với **Bị kéo dẹt** và **Khác kích thước**, EM tìm đúng các cụm mà K-means làm sai ([Hình P.1](ml/cluster_vi)). Ở dữ liệu khác kích thước, trọng số π của cụm nhỏ chỉ khoảng 0.11.
- Bấm **Khởi tạo lại** nhiều lần với K = 3 hoặc 4: đôi khi EM kẹt ở một cực đại địa phương, với log-likelihood cuối thấp hơn.
- Với **Nửa trăng**, các elip không ôm được hình cong. GMM cũng thất bại như K-means.

**2. DBSCAN.** Thử các việc sau:

- Kéo **ε** từ nhỏ đến lớn, và để ý đường ngang trên đồ thị k-distance. Khi ε nằm dưới chỗ bẻ gập, các cụm vỡ vụn và rất nhiều điểm thành nhiễu. Khi ε ở đúng chỗ bẻ gập, ta được đúng các cụm. Khi ε nằm trên chỗ bẻ gập, mọi thứ dính thành một cụm.
- Rê chuột lên các điểm: vòng nét đứt là vùng ε, các đoạn thẳng nối tới mọi láng giềng trong vùng. Điểm lõi có ít nhất minPts láng giềng tính cả chính nó.
- Chọn **Khác mật độ**: không giá trị ε nào tách đúng cả ba cụm.

<iframe class="viz-frame" src="viz/ml-cluster.html?embed" title="Mô phỏng: Phân cụm nâng cao" loading="lazy"></iframe>
