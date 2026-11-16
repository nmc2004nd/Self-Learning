# Chương 14 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Bề mặt xác suất, suy giảm trọng số và ngưỡng

<small class="viz-ref">Mục 14.2–14.4 · công thức (14.13)–(14.14) · hình 14.4–14.5</small>

Huấn luyện hồi quy logistic bằng gradient descent và xem bề mặt xác suất thay đổi. Thử ba việc:

1. Với dữ liệu **tách biệt** và $\lambda = 0$, chạy lâu và xem $\|\mathbf{w}\|$ tăng không ngừng. Sau đó kéo $\lambda$ lên.
2. Kéo **ngưỡng** để thấy đường quyết định dịch song song, đánh đổi giữa hai loại lỗi.
3. Bấm thêm một điểm rất xa nhưng đúng phía. Đường quyết định gần như không đổi, khác với hồi quy tuyến tính (mục 3 của [trang Code](ml/ch14_code_vi)).

<iframe class="viz-frame" src="viz/ml-logistic.html?embed" title="Mô phỏng: Hồi quy logistic" loading="lazy"></iframe>
