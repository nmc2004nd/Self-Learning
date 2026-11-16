# Chương 22 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang chạy lại đoạn code `LinearDiscriminantAnalysis` của [mục 22.4](ml/ch22_vi), rồi giới thiệu hai vai trò của lớp này trong scikit-learn: giảm chiều (như trong sách) và phân loại, cùng các tuỳ chọn bộ giải và co rút cho dữ liệu nhiều chiều.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `scikit-learn`.

| Trong sách | `LinearDiscriminantAnalysis` |
| --- | --- |
| Vector chiếu $\mathbf{w}$ (hai lớp) | `coef_` (cùng phương, khác độ dài) |
| Ma trận chiếu $\mathbf{W}$ (đa lớp) | `scalings_`, dùng qua `transform` |
| Số chiều mới, tối đa $C - 1$ | `n_components` |
| Phân loại | `predict`, `predict_proba` |

## 1. Đoạn code của sách

<small class="viz-ref">Mục 22.4</small>

<!-- run -->
```python
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

np.random.seed(22)
means = [[0, 5], [5, 0]]
cov0 = [[4, 3], [3, 4]]
cov1 = [[3, 1], [1, 1]]
N0, N1 = 50, 40
X0 = np.random.multivariate_normal(means[0], cov0, N0)
X1 = np.random.multivariate_normal(means[1], cov1, N1)

X = np.concatenate((X0, X1))
y = np.array([0]*N0 + [1]*N1)
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
print('w_sklearn = ', clf.coef_[0]/np.linalg.norm(clf.coef_)) # normalize
```

<!-- output -->
```output
w_sklearn =  [ 0.75091074 -0.66040371]
```
<!-- /output -->

Trùng với kết quả in trong sách.

## 2. LDA là một bộ phân loại

`LinearDiscriminantAnalysis` không chỉ giảm chiều. Nó còn là một bộ phân loại sinh: giả sử dữ liệu mỗi lớp là một phân phối chuẩn với **cùng** ma trận hiệp phương sai, rồi dùng quy tắc Bayes ([chương 3](math/ch03_vi)). Ranh giới giữa hai lớp khi đó là một siêu phẳng vuông góc với hướng chiếu $\mathbf{w}$ của sách. `QuadraticDiscriminantAnalysis` bỏ giả sử cùng hiệp phương sai, nên cho ranh giới bậc hai (giống mô hình "Gaussian đầy đủ" ở [mô phỏng chương 11](ml/ch11_viz_vi)).

<!-- run -->
```python
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Xw, yw = load_wine(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, m in [("LDA", LinearDiscriminantAnalysis()), ("QDA", QuadraticDiscriminantAnalysis()),
                ("Gaussian naive Bayes", GaussianNB()),
                ("hồi quy logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))]:
    print(f"{name:22s}: độ chính xác kiểm định chéo trên Wine {cross_val_score(m, Xw, yw, cv=cv).mean():.3f}")
```

<!-- output -->
```output
LDA                   : độ chính xác kiểm định chéo trên Wine 0.994
QDA                   : độ chính xác kiểm định chéo trên Wine 0.989
Gaussian naive Bayes  : độ chính xác kiểm định chéo trên Wine 0.972
hồi quy logistic      : độ chính xác kiểm định chéo trên Wine 0.983
```
<!-- /output -->

Trên Wine, LDA (99.4%) còn nhỉnh hơn hồi quy logistic. Giả sử "mỗi lớp là phân phối chuẩn với cùng hiệp phương sai" khá hợp với số đo hoá học, và mô hình ít tham số nên không cần nhiều dữ liệu.

## 3. Giảm chiều: tối đa $C - 1$

<small class="viz-ref">Mục 22.3</small>

<!-- run -->
```python
lda = LinearDiscriminantAnalysis(n_components=2).fit(Xw, yw)
print("13 chiều ->", lda.transform(Xw).shape[1], "chiều; tỉ lệ 'phương sai giữa các lớp' của từng chiều:", lda.explained_variance_ratio_.round(3))
try:
    LinearDiscriminantAnalysis(n_components=3).fit(Xw, yw)
except ValueError as e:
    print("n_components = 3:", str(e)[:90])
```

<!-- output -->
```output
13 chiều -> 2 chiều; tỉ lệ 'phương sai giữa các lớp' của từng chiều: [0.687 0.313]
n_components = 3: n_components cannot be larger than min(n_features, n_classes - 1).
```
<!-- /output -->

## 4. Khi số chiều lớn: co rút

Với dữ liệu nhiều chiều và ít điểm, $\mathbf{S}_W$ có thể suy biến (không khả nghịch) hoặc ước lượng rất kém. Bộ giải `lsqr` hoặc `eigen` cho phép *co rút* (shrinkage) ma trận hiệp phương sai về phía một ma trận đường chéo, giống ý tưởng suy giảm trọng số ([chương 8](ml/ch08_vi)). `shrinkage="auto"` chọn mức co rút theo công thức Ledoit–Wolf.

<!-- run -->
```python
rng = np.random.default_rng(0)
n, d = 40, 200                                          # 40 điểm, 200 chiều
mu = np.zeros(d); mu[:10] = 0.8                          # chỉ 10 chiều đầu khác nhau giữa hai lớp
Xtr = np.vstack([rng.standard_normal((n // 2, d)), rng.standard_normal((n // 2, d)) + mu]); ytr = np.r_[np.zeros(n // 2), np.ones(n // 2)]
Xte = np.vstack([rng.standard_normal((1000, d)), rng.standard_normal((1000, d)) + mu]); yte = np.r_[np.zeros(1000), np.ones(1000)]
for name, m in [("svd, không co rút", LinearDiscriminantAnalysis(solver="svd")),
                ("lsqr, shrinkage='auto'", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]:
    print(f"{name:24s}: độ chính xác kiểm tra {m.fit(Xtr, ytr).score(Xte, yte):.3f}")
```

<!-- output -->
```output
svd, không co rút       : độ chính xác kiểm tra 0.656
lsqr, shrinkage='auto'  : độ chính xác kiểm tra 0.768
```
<!-- /output -->

Không co rút, $\mathbf{S}_W$ ước lượng từ 40 điểm trong 200 chiều là suy biến, và mô hình gần như chỉ học nhiễu (65.6%). Co rút đưa độ chính xác lên 76.8%.

## Tóm tắt

- `coef_` chứa hướng của sách (sau khi chuẩn hoá độ dài); `transform` chiếu xuống tối đa $C - 1$ chiều.
- LDA cũng là một bộ phân loại tuyến tính, thường là mô hình nền tốt cho dữ liệu dạng bảng.
- Khi số chiều lớn so với số điểm, dùng `solver="lsqr", shrinkage="auto"`.
