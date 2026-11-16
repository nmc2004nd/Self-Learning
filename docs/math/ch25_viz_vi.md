# Chương 25 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Hàm đối ngẫu Lagrange

<small class="viz-ref">Mục 25.2, 25.3 · Ví dụ 1, công thức (25.8) · Hình 25.1</small>

Đây là Hình 25.1 của sách, có thể điều chỉnh. Gợi ý:

1. **Kéo $\lambda$ từ 0 lên:** đáy của hàm Lagrange (chấm xanh) bị kéo dần vào vùng khả thi. Với hàm của sách, nó chạm đúng $p^*$ tại $\lambda \approx 2.5$, rồi đi xuống nếu tiếp tục tăng $\lambda$. Đồ thị bên phải luôn là một hàm lõm.
2. **Hàm $(x + 1)^2$ (lồi):** đổi tâm $c$ của ràng buộc. Khi $-1$ nằm trong $[c - 2, c + 2]$, ràng buộc không chặt và $\lambda^* = 0$; khi không, $\lambda^* > 0$. Khoảng cách đối ngẫu luôn bằng 0.
3. **Hàm $0.5x^2 - 4\cos 2x$ (không lồi):** với $c = 3$, $d^* < p^*$: có khoảng cách đối ngẫu. Đối ngẫu mạnh không được đảm bảo khi bài toán không lồi.

<iframe class="viz-frame" src="viz/math-dual.html?embed" title="Mô phỏng: Hàm đối ngẫu Lagrange" loading="lazy"></iframe>
