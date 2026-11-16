# Decision trees · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm. Cây được dựng ngay trong trình duyệt bằng thuật toán CART ở [mục C.5](ml/tree_vi).

## Tự dựng cây và xem vùng quyết định

<small class="viz-ref">Mục C.1–C.7 · hình C.1, C.5</small>

Bên trái là 200 điểm huấn luyện. Nền được tô theo tỉ lệ lớp hồng ở mỗi lá, còn các đường cam là nhát cắt của cây. Bên phải là chính cây đó. Thử các việc sau:

1. Kéo **độ sâu tối đa** từ 0 lên 1, 2, 3 để xem cây chọn nhát cắt nào trước. Nhát cắt đầu tiên luôn là nhát làm Gini giảm nhiều nhất.
2. Kéo độ sâu lên 12 và xem biểu đồ phía dưới: accuracy kiểm tra tăng rồi giảm, trong khi accuracy huấn luyện lên tới 1.
3. Giữ độ sâu 12, tăng **số mẫu tối thiểu ở lá** lên 5–10. Các ô nhỏ xíu biến mất và accuracy kiểm tra tăng lại.
4. Chọn dữ liệu **Đường chéo**. Ranh giới thật là một đường thẳng nghiêng, nhưng cây chỉ cắt được song song trục, nên phải dùng cả bậc thang.
5. Chọn **XOR** với độ sâu 1: không nhát cắt đơn lẻ nào tốt, cây gần như đoán mò. Độ sâu 2 thì tách được ngay.
6. Bấm vào hình để thêm một điểm nhiễu, rồi xem cây sâu thay đổi cả một vùng lớn chỉ vì một điểm.

<iframe class="viz-frame" src="viz/ml-tree.html?embed" title="Mô phỏng: Cây quyết định" loading="lazy"></iframe>
