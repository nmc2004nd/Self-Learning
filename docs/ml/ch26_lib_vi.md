# Chương 26 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang đọc kỹ những gì `sklearn.svm.SVC` trả về sau khi khớp ([mục 26.4.2](ml/ch26_vi)), đối chiếu với các đại lượng của bài toán đối ngẫu, và so với `LinearSVC`, một lớp khác của scikit-learn cũng có tên SVM tuyến tính nhưng giải một bài toán hơi khác.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `scikit-learn`.

| Lớp | Bài toán giải | Ghi chú |
| --- | --- | --- |
| `SVC(kernel="linear")` | đối ngẫu, bằng libsvm | có `support_`, `dual_coef_`; dùng được hạt nhân ([chương 28](ml/ch28_vi)) |
| `LinearSVC` | gốc, bằng liblinear | nhanh với dữ liệu lớn; mặc định dùng mất mát bản lề *bình phương* và phạt cả $b$ |
| `SGDClassifier(loss="hinge")` | gốc, bằng SGD | xem [trang Thư viện chương 27](ml/ch27_lib_vi) |

scikit-learn không có SVM lề cứng riêng. Như sách, ta dùng SVM lề mềm ([chương 27](ml/ch27_vi)) với $C$ rất lớn.

## 1. Thuộc tính của `SVC` sau khi khớp

<small class="viz-ref">Mục 26.3.5, 26.4.2 · công thức (26.18), (26.19)</small>

<!-- run -->
```python
import numpy as np
from sklearn.svm import SVC, LinearSVC

np.random.seed(22)
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -np.ones(N)))

model = SVC(kernel="linear", C=1e5).fit(X, y)
print("coef_ (w)        =", model.coef_)
print("intercept_ (b)   =", model.intercept_)
print("support_         =", model.support_, " n_support_ =", model.n_support_)
print("dual_coef_       =", model.dual_coef_)
print("classes_         =", model.classes_)
```

<!-- output -->
```output
coef_ (w)        = [[-2.00971102  0.64194082]]
intercept_ (b)   = [4.66595309]
support_         = [12  0  2]  n_support_ = [1 2]
dual_coef_       = [[-2.22524303  0.85022369  1.37501935]]
classes_         = [-1.  1.]
```
<!-- /output -->

`coef_` và `intercept_` khớp đúng với kết quả in trong sách. `SVC` chỉ lưu các vector hỗ trợ: `support_` là chỉ số của chúng (ba điểm 12, 0, 2, giống [trang Code](ml/ch26_code_vi)), `n_support_` là số vector hỗ trợ của mỗi lớp theo thứ tự `classes_`. `dual_coef_` không phải $\lambda_n$ mà là $\lambda_n y_n$, nên mang dấu của nhãn.

Nghiệm của libsvm lệch với nghiệm chính xác ở chữ số thứ tư ($b = 4.6660$ so với $4.6686$), vì thuật toán dừng khi sai số KKT dưới `tol=1e-3`. Giảm `tol` sẽ cho nghiệm gần hơn.

## 2. Dựng lại $\mathbf{w}$ và $b$ từ các thuộc tính

<small class="viz-ref">Công thức (26.18), (26.19)</small>

<!-- run -->
```python
SV = model.support_vectors_
coef = model.dual_coef_.ravel()
w_rebuilt = coef @ SV                                   # (26.18)
b_rebuilt = np.mean(y[model.support_] - SV @ w_rebuilt)  # (26.19)
print("w dựng lại =", w_rebuilt, " b dựng lại =", round(b_rebuilt, 6))
print("lambda_n = |dual_coef_| =", np.abs(coef))
print("sum lambda_n =", round(np.abs(coef).sum(), 4), " ||w||^2 =", round(model.coef_.ravel() @ model.coef_.ravel(), 4))
```

<!-- output -->
```output
w dựng lại = [-2.00971102  0.64194082]  b dựng lại = 4.665953
lambda_n = |dual_coef_| = [2.22524303 0.85022369 1.37501935]
sum lambda_n = 4.4505  ||w||^2 = 4.451
```
<!-- /output -->

Từ các vector hỗ trợ và `dual_coef_` dựng lại được $\mathbf{w}$ và $b$ đúng như (26.18), (26.19). Đẳng thức $\sum_n \lambda_n = \|\mathbf{w}\|_2^2$ (xem [trang Code](ml/ch26_code_vi), mục 2) cũng đúng với nghiệm của thư viện. Với hạt nhân phi tuyến ([chương 28](ml/ch28_vi)), `coef_` không còn tồn tại, nhưng `support_vectors_` và `dual_coef_` vẫn dùng được để dự đoán.

## 3. `decision_function` và khoảng cách tới mặt phân chia

<small class="viz-ref">Mục 26.1.1</small>

`decision_function(x)` trả về $\mathbf{w}^T\mathbf{x} + b$, chưa chia cho $\|\mathbf{w}\|_2$. Khoảng cách thật là giá trị này chia cho $\|\mathbf{w}\|_2$.

<!-- run -->
```python
f = model.decision_function(X)
dist = np.abs(f) / np.linalg.norm(model.coef_)
order = np.argsort(dist)
print("5 điểm gần mặt phân chia nhất:")
for i in order[:5]:
    print(f"  điểm {i:2d}: y = {y[i]:+.0f}, w^T x + b = {f[i]:+.4f}, khoảng cách = {dist[i]:.4f}")
print("lề (khoảng cách nhỏ nhất) =", round(dist.min(), 4), " = 1/||w|| =", round(1 / np.linalg.norm(model.coef_), 4))
```

<!-- output -->
```output
5 điểm gần mặt phân chia nhất:
  điểm  0: y = +1, w^T x + b = +0.9999, khoảng cách = 0.4739
  điểm 12: y = -1, w^T x + b = -1.0001, khoảng cách = 0.4741
  điểm  2: y = +1, w^T x + b = +1.0003, khoảng cách = 0.4741
  điểm 13: y = -1, w^T x + b = -1.5112, khoảng cách = 0.7163
  điểm 10: y = -1, w^T x + b = -1.7649, khoảng cách = 0.8365
lề (khoảng cách nhỏ nhất) = 0.4739  = 1/||w|| = 0.474
```
<!-- /output -->

Ba vector hỗ trợ có $\mathbf{w}^T\mathbf{x} + b = \pm 1$ và cùng cách mặt phân chia một khoảng $1/\|\mathbf{w}\|_2 = 0.474$: đó là lề của mỗi lớp trong Hình 26.2. Muốn biết một điểm mới cách mặt phân chia bao xa, cần chia `decision_function` cho $\|\mathbf{w}\|_2$; giá trị thô chỉ dùng được để so sánh các điểm với nhau trong cùng một mô hình.

## 4. `LinearSVC` không cho cùng nghiệm

<small class="viz-ref">Mục 26.2</small>

<!-- run -->
```python
for kw in [dict(), dict(loss="hinge"), dict(loss="hinge", intercept_scaling=100)]:
    lin = LinearSVC(C=1e5, max_iter=1_000_000, tol=1e-8, **kw).fit(X, y)
    m = y * lin.decision_function(X)
    print(f"LinearSVC{kw}:\n   w = {lin.coef_.ravel()}, b = {lin.intercept_[0]:.4f}, "
          f"min y(w^T x + b) = {m.min():.4f}")
print("SVC (đối ngẫu):        w =", model.coef_.ravel(), " b =", round(model.intercept_[0], 4))
```

<!-- output -->
```output
LinearSVC{}:
   w = [-1.88178534  1.85559801], b = 2.2765, min y(w^T x + b) = 1.0000
LinearSVC{'loss': 'hinge'}:
   w = [-1.8818272   1.85564493], b = 2.2765, min y(w^T x + b) = 1.0000
LinearSVC{'loss': 'hinge', 'intercept_scaling': 100}:
   w = [-2.00987065  0.6405853 ], b = 4.6688, min y(w^T x + b) = 1.0000
SVC (đối ngẫu):        w = [-2.00971102  0.64194082]  b = 4.666
```
<!-- /output -->

Với tham số mặc định, `LinearSVC` cho một đường thẳng khác hẳn, dù vẫn tách đúng mọi điểm với $y(\mathbf{w}^T\mathbf{x} + b) \geq 1$. Lý do là liblinear coi $b$ như một hệ số ứng với một đặc trưng hằng bằng `intercept_scaling` (mặc định 1), nên phạt cả $b$: nó tối thiểu $\|\mathbf{w}\|_2^2 + b^2$ thay vì $\|\mathbf{w}\|_2^2$. Ở bài toán này, $b$ của nghiệm SVM đúng là $4.67$, rất lớn, nên phạt $b$ làm đường thẳng xoay đi để có $b$ nhỏ hơn. Đổi mất mát sang `"hinge"` không giúp gì; tăng `intercept_scaling` lên 100 (phạt $b^2/100^2$, gần như không phạt) thì nghiệm trở về nghiệm SVM.

Bài học: `LinearSVC` và `SVC(kernel="linear")` giải hai bài toán khác nhau. Khi đặc trưng đã được chuẩn hoá về trung bình 0 thì $b$ thường nhỏ và sự khác biệt này ít quan trọng hơn.

## Tóm tắt

| Thuộc tính | Ý nghĩa trong chương 26 |
| --- | --- |
| `coef_`, `intercept_` | $\mathbf{w}$, $b$ (chỉ có với `kernel="linear"`) |
| `support_`, `support_vectors_` | chỉ số và toạ độ các vector hỗ trợ, tập $\mathcal{S}$ |
| `dual_coef_` | $\lambda_n y_n$ với $n \in \mathcal{S}$ |
| `decision_function(x)` | $\mathbf{w}^T\mathbf{x} + b$; chia cho $\|\mathbf{w}\|_2$ để được khoảng cách |
| `LinearSVC` | giải bài toán gốc nhưng phạt cả $b$; cần `intercept_scaling` lớn để khớp `SVC` |
