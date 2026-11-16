# Chương 18 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Lọc cộng tác lân cận

<small class="viz-ref">Mục 18.2, 18.3 · Hình 18.2, 18.3</small>

Gợi ý:

1. **Ô $(i_1, u_1)$** (ô viền đậm ở Hình 18.2e): hai láng giềng là $u_0$ và $u_5$, dự đoán chuẩn hoá 0.48 như trong sách.
2. **Đổi sang item-item:** cùng ô đó được dự đoán từ các sản phẩm mà $u_1$ đã đánh giá; kết quả khác (Hình 18.3).
3. **Kéo $k$:** với $k = 1$, mỗi dự đoán chỉ dựa vào một láng giềng; với $k$ lớn, các láng giềng ít giống (độ tương tự âm) cũng được tính vào.
4. **Ô $(i_2, u_3)$:** dự đoán $-0.5$, ngoài thang điểm.

<iframe class="viz-frame" src="viz/ml-nbcf.html?embed" title="Mô phỏng: Lọc cộng tác lân cận" loading="lazy"></iframe>
