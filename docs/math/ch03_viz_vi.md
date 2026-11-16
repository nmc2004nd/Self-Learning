# Chương 3 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Phân phối chuẩn hai chiều

<small class="viz-ref">Mục 3.1.3–3.1.9, 3.2.4 · công thức (3.46) · Hình 3.1, 3.3</small>

Đây là phiên bản liên tục của Hình 3.1: xác suất đồng thời ở giữa, xác suất biên ở hai mép, và một lát cắt cho xác suất có điều kiện. Gợi ý:

1. **Kéo $\rho$ về 0:** phân phối có điều kiện $p(y \mid x_0)$ trùng với phân phối biên $p(y)$, tức $x$ và $y$ độc lập.
2. **Kéo $\rho$ gần $\pm 1$:** mật độ co thành một dải hẹp, và $p(y \mid x_0)$ rất hẹp: biết $x$ gần như biết luôn $y$.
3. **Bấm nhiều vị trí $x_0$:** kỳ vọng có điều kiện luôn nằm trên đường nét đứt (đường hồi quy), còn độ rộng của $p(y \mid x_0)$ thì không đổi.
4. **Bật "400 mẫu":** các mẫu được tạo bằng phân tích Cholesky của $\mathbf{\Sigma}$, như ở [trang Code](math/ch03_code_vi).

<iframe class="viz-frame" src="viz/math-gauss.html?embed" title="Mô phỏng: Phân phối chuẩn hai chiều" loading="lazy"></iframe>
