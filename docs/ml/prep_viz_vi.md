# Data preprocessing · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm. Các bộ co giãn, kNN và các cách điền giá trị thiếu chạy ngay trong trình duyệt, đúng như mô tả ở bài [Tiền xử lý dữ liệu](ml/prep_vi).

## Co giãn, ngoại lai và điền giá trị thiếu

<small class="viz-ref">Mục D.3, D.4, D.6 · hình D.1, D.4, D.5</small>

**1. Co giãn, ngoại lai và kNN.** Nhãn phụ thuộc cả thu nhập lẫn tuổi (một vùng tròn ở giữa). Thử các việc sau:

- Chọn **MinMax**, rồi kéo ngoại lai ◆ từ gần đám đông sang phải. Phần thân dữ liệu bị dồn thành một dải hẹp theo trục thu nhập, vùng quyết định của kNN bị dẹt lại, và accuracy giảm so với khi không có ngoại lai (con số so sánh nằm trong khung nhận xét).
- Làm lại với **Standard**: ảnh hưởng nhẹ hơn nhưng vẫn có. Với **Robust**, tâm và độ rộng gần như không đổi dù ngoại lai ở đâu.
- Chọn **Quantile**: ngoại lai bị kéo về sát mép đám đông, nhưng hình tròn bị méo.
- Bỏ chọn **Có ngoại lai**: mọi bộ co giãn cho kết quả gần như nhau.

**2. Điền giá trị thiếu.** $x$ luôn quan sát được, $y$ có ô trống. Thử các việc sau:

- Với **MCAR** và **Trung bình**, tăng tỉ lệ thiếu: độ lệch chuẩn và tương quan co lại dần, như Hình D.1b.
- Chuyển sang **MAR**: **Bỏ dòng** làm lệch trung bình của $y$ nhưng hệ số góc vẫn đúng. **KNN** gặp khó ở vùng $x$ lớn, nơi gần như mọi $y$ đều bị ẩn.
- So **Hồi quy** với **Hồi quy + nhiễu**: điền đúng trên đường hồi quy làm tương quan bị thổi phồng; cộng thêm nhiễu thì độ phân tán trở lại như thật.
- Chuyển sang **MNAR**: mọi cách đều lệch, vì các giá trị $y$ lớn đã mất hẳn.

<iframe class="viz-frame" src="viz/ml-prep.html?embed" title="Mô phỏng: Tiền xử lý dữ liệu" loading="lazy"></iframe>
