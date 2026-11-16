# Chương 10 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## K-means từng bước

<small class="viz-ref">Mục 10.2 · Thuật toán 10.1 · hình 10.2, 10.11–10.12</small>

Chạy Thuật toán 10.1 từng bước: gán nhãn, rồi cập nhật tâm. Màu nền là vùng của mỗi tâm, tức tập các điểm gần tâm đó nhất. Biểu đồ bên dưới cho thấy mất mát không bao giờ tăng. Thử ba việc:

1. Khởi tạo ngẫu nhiên nhiều lần trên dữ liệu 5 cụm để gặp cực tiểu địa phương.
2. Tự bấm đặt các tâm ở chỗ tệ, ví dụ cả 5 tâm trong một cụm.
3. Chọn dữ liệu "Cụm lớn nhỏ" hoặc "Trăng khuyết" để thấy các trường hợp K-means thất bại ở mục 10.7.1.

<iframe class="viz-frame" src="viz/ml-kmeans.html?embed" title="Mô phỏng: Phân cụm K-means" loading="lazy"></iframe>
