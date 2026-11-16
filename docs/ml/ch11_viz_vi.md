# Chương 11 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Gaussian naive Bayes

<small class="viz-ref">Mục 11.1, 11.2.1</small>

Gợi ý:

1. **$\rho = 0$:** hai đặc trưng độc lập trong mỗi lớp, giả thiết của naive Bayes đúng. Hai mô hình cho gần như cùng ranh giới.
2. **$\rho = 0.8$ hoặc $-0.8$:** dữ liệu mỗi lớp là một elip nghiêng. Naive Bayes chỉ vẽ được elip thẳng trục, nên ranh giới của nó lệch so với mô hình đầy đủ.
3. **Giảm số điểm xuống 5–10 và bấm "Dữ liệu mới" vài lần:** với rất ít dữ liệu, mô hình đơn giản hơn (ít tham số hơn) đôi khi lại thắng, đó là lý do naive Bayes vẫn được dùng dù giả thiết của nó thường sai.

<iframe class="viz-frame" src="viz/ml-naivebayes.html?embed" title="Mô phỏng: Gaussian naive Bayes" loading="lazy"></iframe>
