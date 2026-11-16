# Chương 29 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Điểm số, vùng an toàn và mất mát bản lề tổng quát

<small class="viz-ref">Mục 29.2.1, 29.2.2, 29.2.3 · công thức (29.2), (29.5) · Hình 29.3, 29.4</small>

Gợi ý:

1. **Bấm vào một điểm nằm gần ranh giới** và xem cột bên phải: các cột điểm số của lớp khác vượt qua đường $z_y - \Delta$ bị phạt đúng bằng phần vượt (khung nét đứt), như Hình 29.4. Điểm nằm sâu trong vùng của lớp mình có mất mát bằng 0.
2. **Đếm "Điểm có mất mát dương"**: chỉ vài điểm sát ranh giới, giống vector hỗ trợ của SVM hai lớp. Chuyển sang **entropy chéo (softmax)**: con số này bằng tổng số điểm.
3. **Kéo $\lambda$ lên 1:** $\mathbf{W}$ bị ép nhỏ, vùng nhạt (hai điểm số cao nhất chênh nhau dưới 1) phủ gần hết mặt phẳng.

<iframe class="viz-frame" src="viz/ml-multisvm.html?embed" title="Mô phỏng: SVM đa lớp" loading="lazy"></iframe>
