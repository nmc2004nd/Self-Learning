# Chương 15 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Từ điểm số đến xác suất

<small class="viz-ref">Mục 15.2–15.5 · công thức (15.1), (15.13) · hình 15.6</small>

Huấn luyện hồi quy softmax bằng gradient descent trên dữ liệu 3 hoặc 4 lớp. Chọn "Xem xác suất" rồi bấm lên đồ thị để xem điểm số $z_k$ và xác suất $a_k$ của từng lớp tại vị trí đó. Kéo $\lambda$ lên để thấy suy giảm trọng số làm các xác suất "mềm" đi: vùng chắc chắn (màu đậm) thu hẹp lại.

<iframe class="viz-frame" src="viz/ml-softmax.html?embed" title="Mô phỏng: Hồi quy softmax" loading="lazy"></iframe>
