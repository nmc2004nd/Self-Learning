# Chương 8 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Bậc đa thức, suy giảm trọng số và lượng dữ liệu

<small class="viz-ref">Mục 8.1–8.3 · hình 8.1</small>

Hồi quy đa thức trên dữ liệu $y = \sin(2\pi x)$ cộng nhiễu, giống trang [Code từ đầu](ml/ch08_code_vi). Bên phải là sai số huấn luyện và sai số xác thực theo bậc. Thử ba việc:

1. Kéo bậc từ 0 lên 15 và xem sai số xác thực đi theo hình chữ U.
2. Giữ bậc cao, kéo $\lambda$ lên để thấy suy giảm trọng số "là phẳng" đường cong.
3. Giữ bậc cao, tăng số điểm huấn luyện để thấy quá khớp giảm dần.

<iframe class="viz-frame" src="viz/ml-overfit.html?embed" title="Mô phỏng: Quá khớp" loading="lazy"></iframe>
