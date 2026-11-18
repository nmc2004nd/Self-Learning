# Chương 2 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc. Mọi phép tính chạy ngay trong trình duyệt.

## Đồ thị tính toán và lan truyền ngược

<small class="viz-ref">Mục 2.5 · bài tập 5–7</small>

Gợi ý:

1. Bấm *Bước tiếp* bảy lần để đi hết lần tính xuôi: mỗi nút chỉ được tính khi mọi đầu vào của nó đã có giá trị.
2. Bấm tiếp để đi ngược. Theo dõi nút $x$: nó nhận gradient từ ba nhánh và cộng dồn lại. So kết quả cuối với đạo hàm tính tay ở dòng chữ bên dưới.
3. Kéo $x$ đến gần $\pi \approx 3.14$: $\sin x \approx 0$ nên nhánh qua $b$ gần như không đóng góp gradient. Với $x$ nhỏ, nhánh $e = 1/x$ áp đảo.

Bản code của chính phép tính này nằm ở mục 4 trang [Code từ đầu](dl/ch02_code_vi).

<iframe class="viz-frame" src="viz/dl-autograd.html?embed" title="Mô phỏng: Đồ thị tính toán và lan truyền ngược" loading="lazy"></iframe>
