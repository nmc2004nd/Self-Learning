# Chương 9 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Vùng quyết định của KNN

<small class="viz-ref">Mục 9.1–9.3 · hình 9.1–9.2</small>

Mỗi vị trí trên mặt phẳng được tô màu theo lớp mà KNN dự đoán tại đó. Đặt điểm cần đoán (ngôi sao) để xem $K$ lân cận và bảng phiếu bầu. Kéo $K$ từ 1 lên khoảng 30 để thấy ranh giới đi từ lởm chởm (quá khớp) đến mượt (có thể chưa khớp). Chỉ số "bỏ-một-ra" bên dưới là cách chọn $K$ mà [chương 8](ml/ch08_vi) gọi là xác thực chéo.

<iframe class="viz-frame" src="viz/ml-knn.html?embed" title="Mô phỏng: K lân cận" loading="lazy"></iframe>
