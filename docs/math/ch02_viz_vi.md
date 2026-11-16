# Chương 2 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Gradient và đạo hàm số

<small class="viz-ref">Mục 2.1, 2.6 · công thức (2.21)–(2.24)</small>

Gợi ý:

1. **Bấm nhiều điểm trên bản đồ:** gradient luôn vuông góc với đường đồng mức (nét đứt trắng) và dài hơn ở nơi các đường đồng mức sít nhau.
2. **Kéo $\varepsilon$:** so cột sai số của xấp xỉ một phía và hai phía. Trên đồ thị bên dưới, sai số có hình chữ V. Phía phải, độ dốc của đường hai phía gấp đôi đường một phía ($\varepsilon^2$ so với $\varepsilon$).
3. **Hàm bậc hai:** xấp xỉ hai phía không có sai số cắt Taylor, vì đạo hàm bậc ba bằng 0.
4. **Log-sum-exp:** gradient của nó chính là hàm softmax ([chương 15](ml/ch15_vi)).

<iframe class="viz-frame" src="viz/math-grad.html?embed" title="Mô phỏng: Gradient và đạo hàm số" loading="lazy"></iframe>
