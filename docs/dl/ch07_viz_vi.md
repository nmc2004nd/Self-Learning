# Chương 7 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc. Mọi phép tính chạy ngay trong trình duyệt.

## Cửa sổ trượt và vùng tiếp nhận

<small class="viz-ref">Mục 7.2–7.5 · hình 7.2.1, 7.3.2, 7.3.3, 7.5.1</small>

Gợi ý:

1. **Hình 7.2.1:** bấm *Bước tiếp* để đi qua bốn vị trí của cửa sổ và đối chiếu với bốn phép tính 19, 25, 37, 43 của [mục 7.2.1](dl/ch07_s2_vi).
2. **Biên:** chọn đầu vào *Biên* và kernel $[1, -1]$ để thấy lại kết quả phát hiện biên của mục 7.2.4. Đổi sang *Sobel dọc*, rồi *Làm mờ*.
3. **Đệm và sải bước:** với kernel $3 \times 3$, tăng đệm lên 1 để đầu ra giữ nguyên kích thước; tăng sải bước lên 2 để đầu ra nhỏ đi một nửa ([mục 7.3](dl/ch07_s3_vi)). Dòng chữ bên dưới tính lại kích thước đầu ra theo công thức.
4. **Gộp:** chuyển sang *Gộp cực đại* trên *Chữ số 7*: nét chữ vẫn còn nhận ra được sau khi kích thước giảm một nửa ([mục 7.5](dl/ch07_s5_vi)).
5. **Vùng tiếp nhận:** ở phần 2, tăng số tầng với $k = 3$, $s = 1$ và xem vùng tiếp nhận lớn thêm 2 mỗi tầng; rồi đổi sang $s = 2$.

<iframe class="viz-frame" src="viz/dl-conv.html?embed" title="Mô phỏng: Tích chập và gộp" loading="lazy"></iframe>
