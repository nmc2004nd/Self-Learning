# Chương 12 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc. Các thuật toán được viết lại bằng JavaScript đúng như các mục 12.4–12.10 (RMSProp dùng $\gamma = 0.9$, Adam dùng $\beta_1 = 0.9$, $\beta_2 = 0.999$). Mô phỏng hạ gradient, momentum và NAG trên một chiều có ở [chương 12 khoá Học máy](ml/ch12_viz_vi).

## Đua các bộ tối ưu

<small class="viz-ref">Mục 12.1, 12.3–12.10 · hàm $0.1x_1^2 + 2x_2^2$ của mục 12.6</small>

Gợi ý:

1. **Elip kém điều kiện:** với $\eta = 0.4$ như mục 12.6, kéo $\eta$ lên quá $0.5$ để thấy SGD phân kỳ theo hướng dốc, trong khi Adam vẫn hội tụ. Rồi giảm $\beta$ về 0 để thấy động lượng trở lại thành SGD.
2. **Rosenbrock:** không thuật toán nào đi nhanh dọc thung lũng. Tăng số bước lên 400 và so sánh xem ai đến gần $(1, 1)$ nhất.
3. **Yên ngựa:** điểm xuất phát gần điểm yên ngựa. Bấm thật sát trục ngang $x_2 = 0$: càng sát, các thuật toán càng mất nhiều bước mới thoát (nếu xuất phát đúng trên trục thì gradient theo $x_2$ luôn bằng 0 và không thuật toán nào thoát được). Bật nhiễu gradient để thấy nhiễu giúp thoát nhanh.
4. **Nhiều cực tiểu:** đổi điểm xuất phát vài lần để thấy mỗi thuật toán rơi vào những hố khác nhau.

<iframe class="viz-frame" src="viz/dl-optim.html?embed" title="Mô phỏng: Đua các bộ tối ưu" loading="lazy"></iframe>
