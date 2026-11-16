# Chương 16 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc. Để mạng học đủ nhanh trên trình duyệt, mô phỏng dùng bộ tối ưu Adam ([trang Dùng thư viện chương 12](ml/ch12_lib_vi)) thay vì gradient descent thường. Gradient vẫn được tính bằng lan truyền ngược đúng như mục 16.4.

## Sân chơi mạng neuron

<small class="viz-ref">Mục 16.1–16.6 · hình 16.9–16.11</small>

Gợi ý:

1. **XOR, 0 tầng ẩn:** hồi quy softmax thất bại. Thêm 1 tầng ẩn với chỉ 2–4 neuron là đủ (mục 16.1.2).
2. **Xoắn ốc:** giảm dần số neuron ẩn để thấy mạng cần bao nhiêu neuron. So sánh ranh giới gấp khúc của ReLU với ranh giới cong mượt của tanh.
3. **Suy giảm trọng số:** kéo $\lambda$ lên để thấy ranh giới mượt dần, giống hình 16.11, cho tới khi mạng không còn khớp được dữ liệu.

<iframe class="viz-frame" src="viz/ml-mlp.html?embed" title="Mô phỏng: Mạng neuron đa tầng" loading="lazy"></iframe>
