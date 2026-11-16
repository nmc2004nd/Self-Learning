# Chương 4 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Tung đồng xu: MLE và MAP

<small class="viz-ref">Mục 4.2.3, 4.3 · công thức (4.10), (4.40) · Hình 4.1</small>

Gợi ý:

1. **Tung từng lần một:** sau lần tung đầu, MLE bằng đúng 0 hoặc 1. MAP nằm giữa đỉnh của tiên nghiệm và MLE.
2. **Đặt $\alpha = \beta = 1$:** tiên nghiệm là phân phối đều, MAP trùng với MLE (đúng như nhận xét sau công thức (4.40)).
3. **Tiên nghiệm sai:** đặt $\lambda$ thật bằng 0.1 và $\alpha = \beta = 30$. Tiên nghiệm tin chắc rằng đồng xu cân đối, nên cần hàng trăm lần tung để MAP tiến được về 0.1.
4. **Tung 100 lần vài lượt:** hàm hợp lý và hậu nghiệm hẹp dần, gần như trùng nhau.

<iframe class="viz-frame" src="viz/math-mapcoin.html?embed" title="Mô phỏng: Tung đồng xu, MLE và MAP" loading="lazy"></iframe>
