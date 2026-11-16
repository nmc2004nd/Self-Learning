# Ensemble learning · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm. Mọi cây được dựng ngay trong trình duyệt bằng thuật toán CART ở [mục C.5](ml/tree_vi).

## Gradient boosting từng bước, bagging và random forest

<small class="viz-ref">Mục T.3–T.6 · hình T.4, T.7, T.8</small>

Trang có hai phần.

**1. Gradient boosting.** Phía trên là mô hình $F_M$ (hồng) và dữ liệu. Phía dưới là phần dư trước bước $M$ (tím) và cây vừa học để khớp phần dư đó (cam). Thử:

- Bấm **+1 bước** vài lần từ $M = 0$: mỗi cây chỉ khớp phần dư, còn mô hình thì tiến dần tới hàm thật.
- Kéo $M$ lên 300 với $\nu = 1$: MSE kiểm tra đạt đáy chỉ sau vài bước rồi tăng lại. Đổi sang $\nu = 0.03$: đáy đến chậm hơn nhưng thấp hơn, và tăng lại chậm hơn nhiều.
- Tăng **độ sâu mỗi cây** lên 5: mỗi cây mạnh hơn, nên quá khớp đến sớm hơn.

**2. Bagging và random forest.** Mỗi cây học trên một mẫu bootstrap và mọc hết cỡ. Thử:

- Với **1 cây**, so ba cây nhỏ bên phải: mỗi mẫu bootstrap cho một vùng quyết định khác hẳn.
- Kéo **số cây** lên 100: các ô lởm chởm triệt tiêu nhau, ranh giới mượt dần và accuracy kiểm tra tăng rồi đi ngang. Thêm cây không gây quá khớp.
- Chuyển sang **Random forest**: từng cây nhỏ trông "kém" hơn vì mỗi lần chia chỉ được xét một đặc trưng ngẫu nhiên, nhưng tổ hợp vẫn tốt ngang hoặc hơn.

<iframe class="viz-frame" src="viz/ml-ensemble.html?embed" title="Mô phỏng: Học tổ hợp" loading="lazy"></iframe>
