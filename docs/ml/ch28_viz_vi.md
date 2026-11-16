# Chương 28 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Hạt nhân và vùng quyết định

<small class="viz-ref">Mục 28.2, 28.3.2, 28.4 · công thức (28.5), (28.6), (28.7), (28.9)–(28.11) · Hình 28.2, 28.3</small>

Bài toán đối ngẫu (28.5) được giải ngay trong trình duyệt bằng SMO, cùng thuật toán với trang [Code từ đầu](ml/ch28_code_vi).

Gợi ý:

1. **XOR với hạt nhân tuyến tính rồi RBF:** đường thẳng không tách được bốn đám điểm; RBF cho bốn vùng đối xứng như Hình 28.2c.
2. **Kéo $\gamma$ của RBF lên 100:** ranh giới co lại quanh từng điểm, gần như mọi điểm là vector hỗ trợ, dù vẫn đúng 100% trên tập huấn luyện. Kéo xuống 0.01: mô hình gần như tuyến tính.
3. **"Hình tròn" với hạt nhân đa thức bậc 2:** ranh giới là một đường ellipse, đúng như Hình 28.1 (mặt phẳng cắt mặt parabol). Với bậc 1, nó trở về đường thẳng.
4. **Hạt nhân sigmoid:** dòng "Mercer" cho biết ma trận hạt nhân trên tập điểm hiện tại có trị riêng âm hay không. Thử đổi $\gamma$ và $r$: kết quả thường kém ổn định.

<iframe class="viz-frame" src="viz/ml-kernel.html?embed" title="Mô phỏng: SVM hạt nhân" loading="lazy"></iframe>
