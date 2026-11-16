# Chương 22 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## LDA và PCA: chiếu để làm gì?

<small class="viz-ref">Mục 22.1, 22.2 · công thức (22.5), (22.13) · Hình 22.1</small>

Gợi ý:

1. **Bấm "Hướng PCA":** với hai lớp dẹt nằm song song, hướng có phương sai lớn nhất chạy dọc theo cả hai lớp. Chiếu lên đó, hai biểu đồ tần suất chồng khít lên nhau.
2. **Bấm "Hướng LDA":** hai biểu đồ tách hẳn nhau. Hướng này là đỉnh của đường $J(\mathbf{w})$ bên phải.
3. **Giảm "Độ dẹt" về 1 và tăng khoảng cách hai lớp:** khi các lớp tròn và xa nhau, PCA và LDA gần như chọn cùng một hướng.

<iframe class="viz-frame" src="viz/ml-lda.html?embed" title="Mô phỏng: LDA và PCA" loading="lazy"></iframe>
