# Chương 17 · Code từ đầu

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang chạy lại ví dụ MovieLens 100k của [mục 17.4](ml/ch17_vi) (đã sửa để chạy được với pandas và Python 3 hiện nay), tự giải bài toán hồi quy ridge cho từng người dùng bằng công thức thay vì gọi thư viện, và chỉ ra một lỗi thụt lề trong đoạn code đánh giá in trong sách.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `pandas`, `scikit-learn`. Mục 1 tải bộ MovieLens 100k (khoảng 5 MB) từ GroupLens ở lần chạy đầu.

## 1. Dữ liệu

<small class="viz-ref">Mục 17.4.1</small>

Code đọc dữ liệu giữ như sách, với hai thay đổi bắt buộc: `as_matrix()` đã bị gỡ khỏi pandas (thay bằng `to_numpy()`), và tệp `u.item` dùng bảng mã Latin-1.

<!-- run -->
```python
import numpy as np
import pandas as pd
import urllib.request, zipfile
from pathlib import Path

root = Path.home() / ".cache"
if not (root / "ml-100k" / "ua.base").exists():
    root.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-100k.zip", root / "ml-100k.zip")
    zipfile.ZipFile(root / "ml-100k.zip").extractall(root)
path = root / "ml-100k"

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(path / 'u.user', sep='|', names=u_cols)
n_users = users.shape[0]
print('Number of users:', n_users)

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv(path / 'ua.base', sep='\t', names=r_cols)
ratings_test = pd.read_csv(path / 'ua.test', sep='\t', names=r_cols)
rate_train = ratings_base.to_numpy()          # sách: as_matrix()
rate_test = ratings_test.to_numpy()
print('Number of traing rates:', rate_train.shape[0])
print('Number of test rates:', rate_test.shape[0])

i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv(path / 'u.item', sep='|', names=i_cols, encoding='latin-1')
n_items = items.shape[0]
print('Number of items:', n_items)
print(rate_train[:4, :])
```

<!-- output -->
```output
Number of users: 943
Number of traing rates: 90570
Number of test rates: 9430
Number of items: 1682
[[        1         1         5 874965758]
 [        1         2         3 876893171]
 [        1         3         4 878542960]
 [        1         4         3 876893119]]
```
<!-- /output -->

Các con số trùng với sách: 943 người dùng, 90 570 đánh giá huấn luyện, 9430 đánh giá kiểm tra, 1682 bộ phim.

## 2. Vector đặc trưng của phim: TF-IDF trên thể loại

<small class="viz-ref">Mục 17.4.2</small>

Mỗi phim có 19 cột 0/1 cho thể loại. Sách biến chúng thành vector TF-IDF ([chương 6](ml/ch06_vi)). Ta tự tính, rồi so với `TfidfTransformer` của sách:

<!-- run -->
```python
X_train_counts = items.iloc[:, -19:].to_numpy().astype(float)
df = (X_train_counts > 0).sum(0)                                  # số phim thuộc mỗi thể loại
idf = np.log((1 + n_items) / (1 + df)) + 1                         # smooth_idf=True như sách
X = X_train_counts * idf
X = X / np.linalg.norm(X, axis=1, keepdims=True)                   # norm='l2'

from sklearn.feature_extraction.text import TfidfTransformer
X_book = TfidfTransformer(smooth_idf=True, norm='l2').fit_transform(X_train_counts.tolist()).toarray()
print("khớp TfidfTransformer:", np.allclose(X, X_book))
print("phim 'Toy Story':", items.iloc[0, 1], "->", {str(g): float(v) for g, v in zip(np.array(i_cols[-19:])[X[0] > 0], X[0][X[0] > 0].round(3))})
```

<!-- output -->
```output
khớp TfidfTransformer: True
phim 'Toy Story': Toy Story (1995) -> {'Animation': 0.741, "Children's": 0.574, 'Comedy': 0.349}
```
<!-- /output -->

Thể loại hiếm (Animation) có trọng số lớn hơn thể loại phổ biến (Comedy) trong vector của *Toy Story*.

## 3. Một mô hình hồi quy ridge cho mỗi người dùng

<small class="viz-ref">Mục 17.3.2, 17.4.3 · công thức (17.2)–(17.6)</small>

Với người dùng $n$ đã đánh giá các phim có vector đặc trưng $\hat{\mathbf{X}}_n$ và điểm $\hat{\mathbf{y}}_n$, ta tìm $(\mathbf{w}_n, b_n)$ cực tiểu $\|\hat{\mathbf{X}}_n\mathbf{w} + b\mathbf{1} - \hat{\mathbf{y}}_n\|^2 + \alpha\|\mathbf{w}\|^2$. Sách gọi `Ridge(alpha=0.01)` của scikit-learn. Ta tự giải: trừ trung bình để khử $b$, rồi giải hệ $(\hat{\mathbf{X}}_c^T\hat{\mathbf{X}}_c + \alpha\mathbf{I})\mathbf{w} = \hat{\mathbf{X}}_c^T\hat{\mathbf{y}}_c$ (dạng của [chương 7](ml/ch07_vi), [chương 8](ml/ch08_vi)).

<!-- run -->
```python
def get_items_rated_by_user(rate_matrix, user_id):
    y = rate_matrix[:, 0]
    ids = np.where(y == user_id + 1)[0]
    item_ids = rate_matrix[ids, 1] - 1
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)

def ridge_fit(Xn, yn, alpha):
    xm, ym = Xn.mean(0), yn.mean()                       # b không bị phạt: trừ trung bình rồi khôi phục
    Xc, yc = Xn - xm, yn - ym
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xn.shape[1]), Xc.T @ yc)
    return w, ym - xm @ w

d = X.shape[1]
W = np.zeros((d, n_users)); b = np.zeros(n_users)
for n in range(n_users):
    ids, scores = get_items_rated_by_user(rate_train, n)
    W[:, n], b[n] = ridge_fit(X[ids], scores, alpha=0.01)
Yhat = X.dot(W) + b

from sklearn.linear_model import Ridge
ids, scores = get_items_rated_by_user(rate_train, 10)
m = Ridge(alpha=0.01, fit_intercept=True).fit(X[ids], scores)
print("người dùng 10, khớp Ridge của scikit-learn:", np.allclose(m.coef_, W[:, 10]), np.isclose(m.intercept_, b[10]))

n = 10
np.set_printoptions(precision=2)
ids, scores = get_items_rated_by_user(rate_test, n)
print('Rated movies ids :', ids)
print('True ratings     :', scores)
print('Predicted ratings:', Yhat[ids, n])
```

<!-- output -->
```output
người dùng 10, khớp Ridge của scikit-learn: True True
Rated movies ids : [ 37 109 110 226 424 557 722 724 731 739]
True ratings     : [3 3 4 3 4 3 5 3 3 4]
Predicted ratings: [3.18 3.13 3.42 3.09 3.35 5.2  4.01 3.35 3.42 3.72]
```
<!-- /output -->

Ba dòng cuối trùng với sách.

## 4. Hàm đánh giá: một lỗi thụt lề

<small class="viz-ref">Mục 17.4.4</small>

Trong hàm `evaluate` in trong sách, lệnh `return` nằm **bên trong** vòng lặp `for` (thụt lề một mức quá sâu). Chạy đúng như vậy, hàm trả về ngay sau người dùng đầu tiên. Ta chạy cả bản in trong sách lẫn bản đã sửa:

<!-- run -->
```python
def evaluate_book(Yhat, rates, W, b):                     # nguyên văn sách, chỉ thay xrange bằng range
    se = cnt = 0
    for n in range(n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred
        se += (e*e).sum(axis = 0)
        cnt += e.size
        return np.sqrt(se/cnt)                            # <- nằm trong vòng lặp

def evaluate(Yhat, rates, W, b):                          # đã sửa: return sau vòng lặp
    se = cnt = 0
    for n in range(n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        e = scores_truth - Yhat[ids, n]
        se += (e * e).sum()
        cnt += e.size
    return np.sqrt(se / cnt)

print('code của sách: RMSE for training: %.2f' % evaluate_book(Yhat, rate_train, W, b))
print('code của sách: RMSE for test    : %.2f' % evaluate_book(Yhat, rate_test, W, b))
print('đã sửa:        RMSE for training: %.2f' % evaluate(Yhat, rate_train, W, b))
print('đã sửa:        RMSE for test    : %.2f' % evaluate(Yhat, rate_test, W, b))
```

<!-- output -->
```output
code của sách: RMSE for training: 1.08
code của sách: RMSE for test    : 0.81
đã sửa:        RMSE for training: 0.91
đã sửa:        RMSE for test    : 1.27
```
<!-- /output -->

Bản đã sửa cho đúng hai con số trong sách (0.91 và 1.27); bản in nguyên văn cho 1.08 và 0.81, tức chỉ là sai số của người dùng đầu tiên. Vậy kết quả của sách đúng: tác giả hẳn đã chạy code đúng, chỉ có đoạn code in ra bị lệch thụt lề.

RMSE trên tập kiểm tra (1.27) cao hơn hẳn trên tập huấn luyện (0.91). Để biết con số này tốt hay xấu, ta cần một mốc so sánh.

## 5. So với những cách dự đoán đơn giản

<!-- run -->
```python
mu = rate_train[:, 2].mean()
user_mean = np.array([rate_train[rate_train[:, 0] == n + 1, 2].mean() for n in range(n_users)])
item_sum = np.bincount(rate_train[:, 1] - 1, weights=rate_train[:, 2], minlength=n_items)
item_cnt = np.bincount(rate_train[:, 1] - 1, minlength=n_items)
item_mean = np.where(item_cnt > 0, item_sum / np.maximum(item_cnt, 1), mu)

rmse = lambda pred: np.sqrt(np.mean((rate_test[:, 2] - pred) ** 2))
u, i = rate_test[:, 0] - 1, rate_test[:, 1] - 1
print(f"trung bình toàn bộ:         RMSE {rmse(np.full(len(u), mu)):.4f}")
print(f"trung bình của người dùng:  RMSE {rmse(user_mean[u]):.4f}")
print(f"trung bình của phim:        RMSE {rmse(item_mean[i]):.4f}")
print(f"dựa trên nội dung (sách):   RMSE {rmse(Yhat[i, u]):.4f}")
print(f"dựa trên nội dung, cắt về [1, 5]: RMSE {rmse(np.clip(Yhat[i, u], 1, 5)):.4f}")
```

<!-- output -->
```output
trung bình toàn bộ:         RMSE 1.1220
trung bình của người dùng:  RMSE 1.0431
trung bình của phim:        RMSE 1.0418
dựa trên nội dung (sách):   RMSE 1.2703
dựa trên nội dung, cắt về [1, 5]: RMSE 1.1899
```
<!-- /output -->

Mô hình dựa trên nội dung với hệ số phạt rất nhỏ ($\alpha = 0.01$) còn kém cả việc đoán bằng điểm trung bình toàn bộ (1.27 so với 1.12). Lý do: mỗi mô hình có 20 tham số (19 hệ số và hệ số tự do), trong khi một phần tư số người dùng có không quá 23 đánh giá huấn luyện, ít nhất chỉ 10. Với hệ số phạt gần như bằng 0, mô hình quá khớp ([chương 8](ml/ch08_vi)). Cắt dự đoán về khoảng $[1, 5]$ giúp một chút, nhưng vẫn kém các mốc đơn giản.

## Tóm tắt

- Dữ liệu và kết quả dự đoán cho người dùng số 10 trùng với sách.
- Đoạn code `evaluate` in trong sách có lỗi thụt lề; con số trong sách (0.91, 1.27) là của bản đúng.
- Mô hình dựa trên nội dung, chỉ dùng thể loại phim, còn kém cả việc đoán bằng điểm trung bình của mỗi người dùng.
- Trang [Ví dụ áp dụng](ml/ch17_app_vi) tìm cách cải thiện mô hình dựa trên nội dung.
