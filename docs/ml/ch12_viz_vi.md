# Chương 12 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## GD, momentum và NAG trên bốn hàm

<small class="viz-ref">Mục 12.3–12.5 · công thức (12.7)–(12.10) · hình 12.8–12.10</small>

Ba thuật toán chạy song song từ cùng một điểm xuất phát. Gợi ý:

1. **Bát dẹt:** tăng dần $\eta$. GD bắt đầu dao động theo phương dốc ($y$) trong khi vẫn bò chậm theo phương thoải ($x$). Momentum đi nhanh hơn nhiều.
2. **Rosenbrock:** so số bước cần để tới gần $(1, 1)$.
3. **Hai hố:** đây là hàm của hình 12.9. Kéo $\gamma$ từ 0 lên 0.9 để xem khi nào momentum có đủ đà vượt qua đồi.

<iframe class="viz-frame" src="viz/ml-gd.html?embed" title="Mô phỏng: Gradient descent, momentum và NAG" loading="lazy"></iframe>
