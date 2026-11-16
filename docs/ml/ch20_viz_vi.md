# Chương 20 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## SVD: xoay, co giãn, xoay

<small class="viz-ref">Mục 20.2.1, 20.2.5, 20.2.6</small>

Gợi ý:

1. **Bấm "Chạy lại từ đầu":** đường tròn đơn vị lần lượt bị xoay ($\mathbf{V}^T$), kéo thành elip thẳng trục ($\mathbf{\Sigma}$), rồi xoay tới vị trí cuối ($\mathbf{U}$). Hai mũi tên luôn vuông góc.
2. **Bật "Chỉ giữ $\sigma_1$":** elip bị ép thành đoạn thẳng, đó là xấp xỉ hạng 1. Sai số bằng đúng $\sigma_2$.
3. **Ma trận $\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$ (phản xạ):** $\sigma_1 = \sigma_2 = 1$, đường tròn giữ nguyên, chỉ lá cờ bị lật.

<iframe class="viz-frame" src="viz/ml-svd.html?embed" title="Mô phỏng: SVD là xoay, co giãn, xoay" loading="lazy"></iframe>
