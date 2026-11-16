# Chương 17 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang giới thiệu các công cụ để làm việc với ma trận tiện ích ([mục 17.2](ml/ch17_vi)): lưu trữ dạng thưa với SciPy và pandas, và chọn hệ số phạt của hồi quy ridge, thứ mà [trang Code](ml/ch17_code_vi) cho thấy là nguyên nhân chính làm mô hình của sách kém.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `pandas`, `scipy`, `scikit-learn`. Dữ liệu MovieLens 100k được tải ở lần chạy đầu.

## 1. Ma trận tiện ích dạng thưa

<small class="viz-ref">Mục 17.2</small>

<!-- run -->
```python
import numpy as np
import pandas as pd
import scipy.sparse as sp
import urllib.request, zipfile
from pathlib import Path

root = Path.home() / ".cache"
if not (root / "ml-100k" / "ua.base").exists():
    root.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-100k.zip", root / "ml-100k.zip")
    zipfile.ZipFile(root / "ml-100k.zip").extractall(root)
path = root / "ml-100k"
cols = ["user_id", "movie_id", "rating", "ts"]
train = pd.read_csv(path / "ua.base", sep="\t", names=cols)
test = pd.read_csv(path / "ua.test", sep="\t", names=cols)
n_users, n_items = 943, 1682

# Ma trận tiện ích Y: hàng = phim, cột = người dùng, như quy ước của sách
Y = sp.csr_matrix((train.rating, (train.movie_id - 1, train.user_id - 1)), shape=(n_items, n_users))
print(f"ma trận tiện ích {Y.shape}, {Y.nnz:,} ô đã biết, tỉ lệ đã biết {Y.nnz / np.prod(Y.shape):.2%}")
print(f"bộ nhớ dạng thưa: {(Y.data.nbytes + Y.indices.nbytes + Y.indptr.nbytes) / 1e6:.2f} MB; dạng đầy đủ: {np.prod(Y.shape) * 8 / 1e6:.1f} MB")

# pandas: pivot thành bảng (NaN ở ô chưa biết), tiện để xem nhưng tốn bộ nhớ
table = train.pivot_table(index="movie_id", columns="user_id", values="rating")
print("pivot_table:", table.shape, "; 5 phim x 6 người dùng đầu:\n", table.iloc[:5, :6])

per_user = np.diff(Y.tocsc().indptr)
per_item = np.diff(Y.indptr)
print(f"\nsố đánh giá mỗi người dùng: ít nhất {per_user.min()}, trung vị {int(np.median(per_user))}, nhiều nhất {per_user.max()}")
print(f"số đánh giá mỗi phim: ít nhất {per_item.min()}, trung vị {int(np.median(per_item))}, nhiều nhất {per_item.max()}; số phim có không quá 5 đánh giá: {np.sum(per_item <= 5)}")
```

<!-- output -->
```output
ma trận tiện ích (1682, 943), 90,570 ô đã biết, tỉ lệ đã biết 5.71%
bộ nhớ dạng thưa: 1.09 MB; dạng đầy đủ: 12.7 MB
pivot_table: (1680, 943) ; 5 phim x 6 người dùng đầu:
 user_id     1    2   3   4   5    6
movie_id                           
1         5.0  4.0 NaN NaN NaN  4.0
2         3.0  NaN NaN NaN NaN  NaN
3         4.0  NaN NaN NaN NaN  NaN
4         3.0  NaN NaN NaN NaN  NaN
5         3.0  NaN NaN NaN NaN  NaN

số đánh giá mỗi người dùng: ít nhất 10, trung vị 55, nhiều nhất 727
số đánh giá mỗi phim: ít nhất 0, trung vị 24, nhiều nhất 495; số phim có không quá 5 đánh giá: 404
```
<!-- /output -->

Ma trận chỉ có 5.7% ô đã biết. Số đánh giá phân bố rất lệch: vài phim được hàng trăm người đánh giá, nhiều phim chỉ vài người. Đây là hiện tượng *đuôi dài* ở mục 17.1.1.

## 2. Chọn hệ số phạt

<small class="viz-ref">Mục 17.3.2</small>

Sách dùng `Ridge(alpha=0.01)` cho mọi người dùng. Thử các giá trị khác:

<!-- run -->
```python
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge, RidgeCV

items = pd.read_csv(path / "u.item", sep="|", header=None, encoding="latin-1")
X = TfidfTransformer(smooth_idf=True, norm="l2").fit_transform(items.iloc[:, -19:].to_numpy()).toarray()
by_user_tr = {u: g for u, g in train.groupby("user_id")}
u_te, i_te, r_te = test.user_id.to_numpy() - 1, test.movie_id.to_numpy() - 1, test.rating.to_numpy()

def fit_all(make_model):
    W = np.zeros((X.shape[1], n_users)); b = np.zeros(n_users)
    for u in range(n_users):
        g = by_user_tr[u + 1]
        m = make_model().fit(X[g.movie_id - 1], g.rating)
        W[:, u], b[u] = m.coef_, m.intercept_
    return W, b

def rmse_test(W, b):
    pred = np.clip(np.einsum("ij,ji->i", X[i_te], W[:, u_te]) + b[u_te], 1, 5)
    return np.sqrt(np.mean((pred - r_te) ** 2))

for a in [0.01, 0.1, 1, 3, 10, 30]:
    print(f"alpha = {a:5}: RMSE kiểm tra {rmse_test(*fit_all(lambda: Ridge(alpha=a))):.4f}")
W_cv, b_cv = fit_all(lambda: RidgeCV(alphas=np.logspace(-2, 2, 13)))
print(f"RidgeCV (mỗi người dùng tự chọn alpha bằng kiểm định chéo bỏ-một-ra): RMSE kiểm tra {rmse_test(W_cv, b_cv):.4f}")
```

<!-- output -->
```output
alpha =  0.01: RMSE kiểm tra 1.1899
alpha =   0.1: RMSE kiểm tra 1.1288
alpha =     1: RMSE kiểm tra 1.0498
alpha =     3: RMSE kiểm tra 1.0289
alpha =    10: RMSE kiểm tra 1.0250
alpha =    30: RMSE kiểm tra 1.0306
RidgeCV (mỗi người dùng tự chọn alpha bằng kiểm định chéo bỏ-một-ra): RMSE kiểm tra 1.0455
```
<!-- /output -->

Hệ số phạt lớn hơn nhiều so với 0.01 làm RMSE giảm từ 1.19 xuống 1.025 (với $\alpha = 10$). `RidgeCV` chọn $\alpha$ riêng cho từng người dùng bằng kiểm định chéo bỏ-một-ra, nhưng lại kém hơn một $\alpha$ chung: với 10–20 đánh giá, kiểm định chéo trên dữ liệu của riêng một người rất nhiễu. Chọn một $\alpha$ chung cho mọi người dùng (bằng một tập xác thực) ổn định hơn.

## 3. Vì sao phạt nặng lại tốt hơn

Với $\alpha$ rất lớn, $\mathbf{w} \to \mathbf{0}$ và dự đoán chỉ còn là hệ số tự do $b_n$, tức điểm trung bình của người dùng đó. Hệ số phạt cho phép mô hình "lùi về" mốc an toàn này khi thông tin về thể loại không đủ tin cậy.

<!-- run -->
```python
W_big, b_big = fit_all(lambda: Ridge(alpha=1e6))
user_mean = train.groupby("user_id").rating.mean().to_numpy()
print("alpha = 1e6: b_n trùng với trung bình của người dùng:", np.allclose(b_big, user_mean, atol=1e-3))
print(f"RMSE của 'trung bình người dùng' (cắt về [1, 5]): {rmse_test(np.zeros_like(W_big), user_mean):.4f}")
```

<!-- output -->
```output
alpha = 1e6: b_n trùng với trung bình của người dùng: True
RMSE của 'trung bình người dùng' (cắt về [1, 5]): 1.0431
```
<!-- /output -->

Mô hình tốt nhất ($\alpha = 10$, RMSE 1.025) chỉ tốt hơn mốc "trung bình người dùng" (1.043) một chút: thể loại phim mang rất ít thông tin về sở thích cá nhân. [Trang Ví dụ áp dụng](ml/ch17_app_vi) xem khi nào mô hình dựa trên nội dung thật sự có giá trị.

## Tóm tắt

- Lưu ma trận tiện ích bằng `scipy.sparse` (chỉ 5.7% ô có giá trị); `pivot_table` của pandas chỉ dùng để xem.
- Mỗi người dùng một mô hình ridge: đừng giữ $\alpha$ rất nhỏ; một $\alpha$ chung chọn bằng tập xác thực tốt hơn $\alpha$ riêng chọn trên dữ liệu ít ỏi của từng người.
