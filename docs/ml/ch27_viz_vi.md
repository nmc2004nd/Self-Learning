# Chương 27 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Lề mềm, biến lỏng lẻo và mất mát bản lề

<small class="viz-ref">Mục 27.2, 27.3.4, 27.4.2, 27.5.4 · công thức (27.11), (27.25) · Hình 27.1, 27.2, 27.3, 27.5</small>

SVM lề mềm được giải ngay trong trình duyệt bằng SMO, giống [mô phỏng chương 26](ml/ch26_viz_vi). Với dữ liệu của sách và $C = 100$, mô phỏng cho cùng nghiệm như mục 27.5: 17 vector hỗ trợ, 14 điểm có $\lambda_n = C$.

Gợi ý:

1. **Kéo thanh $C$ từ 1000 xuống 0.01** và quan sát ba con số: lề rộng ra, số vector hỗ trợ tăng, $\sum_n \xi_n$ tăng. Đó là Hình 27.5 chạy liên tục.
2. **So sánh hai hình.** Điểm có dấu × ở hình trái ($\lambda_n = C$) luôn nằm trên phần dốc của mất mát bản lề ở hình phải; điểm đặc ($\lambda_n = 0$) nằm ở phần bằng 0; điểm rỗng nằm đúng ở góc $y_nz_n = 1$.
3. **Chọn "Một điểm nhiễu"** (Hình 27.1a): với $C = 1000$, cả lề bị một điểm vuông lạc kéo hẹp lại. Giảm $C$ xuống dưới 1 thì lề rộng ra, và điểm đó chỉ được đánh dấu bằng một đoạn cam dài.

<iframe class="viz-frame" src="viz/ml-softsvm.html?embed" title="Mô phỏng: SVM lề mềm" loading="lazy"></iframe>
