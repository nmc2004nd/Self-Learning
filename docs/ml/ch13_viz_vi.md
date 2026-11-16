# Chương 13 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## PLA từng lần cập nhật

<small class="viz-ref">Mục 13.2–13.3 · Thuật toán 13.1 · hình 13.3</small>

Xem Thuật toán 13.1 chạy từng bước: chọn một điểm bị phân loại sai, cộng $y_i\bar{\mathbf{x}}_i$ vào $\mathbf{w}$, rồi lặp lại. Thử bốn bộ dữ liệu:

- **Tách biệt** và **Sát nhau:** PLA luôn dừng, nhưng hai lớp càng sát thì càng cần nhiều lần cập nhật.
- **Chồng lấn** và **XOR:** PLA không bao giờ dừng. Bật "Hiện pocket" để thấy thuật toán pocket giữ lại đường tốt nhất.

Bấm lên đồ thị để thêm điểm, ví dụ thêm một điểm lẫn sang lớp kia để biến dữ liệu tách biệt thành không tách biệt.

<iframe class="viz-frame" src="viz/ml-pla.html?embed" title="Mô phỏng: Thuật toán học perceptron" loading="lazy"></iframe>
