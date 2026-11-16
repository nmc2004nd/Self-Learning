# Chương 18 · Code từ đầu

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang dựng lại từng bước của ví dụ trong Hình 18.2 ([chương 18](ml/ch18_vi)), chạy lại lớp `uuCF` của sách trên MovieLens 100k, và chỉ ra một lỗi kiểu dữ liệu làm kết quả của sách kém hơn thực tế.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `pandas`, `scipy`, `scikit-learn`. Mục 2 tải bộ MovieLens 100k ở lần chạy đầu.

## 1. Ví dụ trong Hình 18.2

<small class="viz-ref">Mục 18.2 · Hình 18.2</small>

Ma trận tiện ích có 5 sản phẩm (hàng) và 7 người dùng (cột); `nan` là ô chưa biết.

<!-- run -->
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True)
nan = np.nan
Y = np.array([[5, 5, 2, 0, 1, nan, nan],
              [4, nan, nan, 0, nan, 2, nan],
              [nan, 4, 1, nan, nan, 1, 1],
              [2, 2, 3, 4, 4, nan, 4],
              [2, 0, 4, nan, nan, nan, 5]])
known = ~np.isnan(Y)

# a) trung bình của mỗi người dùng; b) chuẩn hoá: trừ trung bình, ô chưa biết = 0
ubar = np.nanmean(Y, axis=0)
Ybar = np.where(known, Y - ubar, 0.0)
print("a) trung bình người dùng:", ubar)
print("b) ma trận chuẩn hoá:\n", Ybar)

# c) tương tự cos giữa các cột
norms = np.linalg.norm(Ybar, axis=0)
S = (Ybar.T @ Ybar) / np.outer(norms, norms)
print("c) ma trận tương tự người dùng:\n", S)
```

<!-- output -->
```output
a) trung bình người dùng: [3.25 2.75 2.5  1.33 2.5  1.5  3.33]
b) ma trận chuẩn hoá:
 [[ 1.75  2.25 -0.5  -1.33 -1.5   0.    0.  ]
 [ 0.75  0.    0.   -1.33  0.    0.5   0.  ]
 [ 0.    1.25 -1.5   0.    0.   -0.5  -2.33]
 [-1.25 -0.75  0.5   2.67  1.5   0.    0.67]
 [-1.25 -2.75  1.5   0.    0.    0.    1.67]]
c) ma trận tương tự người dùng:
 [[ 1.    0.83 -0.58 -0.79 -0.82  0.2  -0.38]
 [ 0.83  1.   -0.87 -0.4  -0.55 -0.23 -0.71]
 [-0.58 -0.87  1.    0.27  0.32  0.47  0.96]
 [-0.79 -0.4   0.27  1.    0.87 -0.29  0.18]
 [-0.82 -0.55  0.32  0.87  1.    0.    0.16]
 [ 0.2  -0.23  0.47 -0.29  0.    1.    0.56]
 [-0.38 -0.71  0.96  0.18  0.16  0.56  1.  ]]
```
<!-- /output -->

Ba ma trận trùng với Hình 18.2a, b, c (sách làm tròn tới 2 chữ số). Bước tiếp theo dự đoán các ô chưa biết. Với ô (sản phẩm $i$, người dùng $u$), lấy $k = 2$ người dùng giống $u$ nhất **trong số những người đã đánh giá $i$**, rồi lấy trung bình có trọng số các đánh giá chuẩn hoá của họ:

$$
\hat{y}_{i, u} = \frac{\sum_{u_j \in \mathcal{N}(u, i)} \bar{y}_{i, u_j}\,\text{sim}(u, u_j)}{\sum_{u_j \in \mathcal{N}(u, i)} |\text{sim}(u, u_j)|}.
$$

<!-- run -->
```python
def predict_norm(i, u, k=2):
    raters = np.flatnonzero(known[i])                    # những người đã đánh giá i
    nn = raters[np.argsort(S[u, raters])[-k:]]           # k người giống u nhất
    return Ybar[i, nn] @ S[u, nn] / np.abs(S[u, nn]).sum()

Yd = Ybar.copy()
for i, u in zip(*np.where(~known)):
    Yd[i, u] = predict_norm(i, u)
print("d) ma trận chuẩn hoá sau hoàn thiện:\n", Yd)
print("e) ô (i1, u1):", round(predict_norm(1, 1), 2), "(sách: 0.48)")
print("f) ma trận tiện ích sau hoàn thiện:\n", Yd + ubar)
```

<!-- output -->
```output
d) ma trận chuẩn hoá sau hoàn thiện:
 [[ 1.75  2.25 -0.5  -1.33 -1.5   0.18 -0.63]
 [ 0.75  0.48 -0.17 -1.33 -1.33  0.5   0.05]
 [ 0.91  1.25 -1.5  -1.84 -1.78 -0.5  -2.33]
 [-1.25 -0.75  0.5   2.67  1.5   0.59  0.67]
 [-1.25 -2.75  1.5   1.57  1.56  1.59  1.67]]
e) ô (i1, u1): 0.48 (sách: 0.48)
f) ma trận tiện ích sau hoàn thiện:
 [[ 5.    5.    2.    0.    1.    1.68  2.7 ]
 [ 4.    3.23  2.33  0.    1.17  2.    3.38]
 [ 4.16  4.    1.   -0.5   0.72  1.    1.  ]
 [ 2.    2.    3.    4.    4.    2.09  4.  ]
 [ 2.    0.    4.    2.9   4.06  3.09  5.  ]]
```
<!-- /output -->

Các giá trị khớp với Hình 18.2d–f (sai khác ±0.01 do sách làm tròn từng bước), trừ một ô: Hình 18.2f in ô $(i_1, u_4)$ bằng 1.67, nhưng theo chính Hình 18.2d giá trị chuẩn hoá là $-1.33$, cộng trung bình 2.5 của $u_4$ được 1.17. Đây là một lỗi in trong hình. Một ô đáng chú ý khác: $(i_2, u_3)$ được dự đoán $-0.5$, nhỏ hơn mức thấp nhất của thang điểm. Người dùng $u_3$ vốn chấm điểm thấp, và hai người giống họ nhất đều chấm $i_2$ thấp hơn trung bình của chính họ. Khi dùng thực tế, ta cắt dự đoán về khoảng $[0, 5]$.

## 2. Lớp `uuCF` của sách trên MovieLens 100k

<small class="viz-ref">Mục 18.4</small>

Code giữ như sách, chỉ thay `xrange` bằng `range` và `as_matrix()` bằng `to_numpy()`.

<!-- run -->
```python
import pandas as pd
import urllib.request, zipfile
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class uuCF(object):
    def __init__(self, Y_data, k, sim_func=cosine_similarity):
        self.Y_data = Y_data
        self.k = k
        self.sim_func = sim_func
        self.Ybar = None
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def fit(self):
        users = self.Y_data[:, 0]
        self.Ybar = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            item_ids = self.Y_data[ids, 1]
            ratings = self.Y_data[ids, 2]
            self.mu[n] = np.mean(ratings) if ids.size > 0 else 0
            self.Ybar[ids, 2] = ratings - self.mu[n]
        self.Ybar = sparse.coo_matrix((self.Ybar[:, 2],
                                      (self.Ybar[:, 1], self.Ybar[:, 0])),
                                      (self.n_items, self.n_users)).tocsr()
        self.S = self.sim_func(self.Ybar.T, self.Ybar.T)

    def pred(self, u, i):
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        sim = self.S[u, users_rated_i]
        nns = np.argsort(sim)[-self.k:]
        nearest_s = sim[nns]
        r = self.Ybar[i, users_rated_i[nns]]
        eps = 1e-8
        return (r*nearest_s).sum()/(np.abs(nearest_s).sum()+eps)+self.mu[u]

root = Path.home() / ".cache"
if not (root / "ml-100k" / "ua.base").exists():
    root.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-100k.zip", root / "ml-100k.zip")
    zipfile.ZipFile(root / "ml-100k.zip").extractall(root)
path = root / "ml-100k"
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
rate_train = pd.read_csv(path / 'ua.base', sep='\t', names=r_cols).to_numpy()
rate_test = pd.read_csv(path / 'ua.test', sep='\t', names=r_cols).to_numpy()
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

def test_rmse(rs, rate_test):
    SE = 0
    for n in range(rate_test.shape[0]):
        pred = rs.pred(rate_test[n, 0], rate_test[n, 1])
        SE += (pred - rate_test[n, 2])**2
    return np.sqrt(SE/rate_test.shape[0])

rs = uuCF(rate_train, k = 40); rs.fit()
print('User-user CF, RMSE =', test_rmse(rs, rate_test))
rs = uuCF(rate_train[:, [1, 0, 2]], k = 40); rs.fit()
print('Item-item CF, RMSE =', test_rmse(rs, rate_test[:, [1, 0, 2]]))
```

<!-- output -->
```output
User-user CF, RMSE = 0.9766261152473883
Item-item CF, RMSE = 0.968867817361657
```
<!-- /output -->

Hai con số trùng với sách: 0.976614... và 0.968846...

## 3. Lỗi kiểu dữ liệu

Trong `fit`, `self.Ybar = self.Y_data.copy()` có kiểu số nguyên (dữ liệu đọc vào là số nguyên). Gán `ratings - self.mu[n]` (số thực) vào mảng số nguyên làm **cắt bỏ phần thập phân**: một đánh giá cao hơn trung bình 0.8 điểm trở thành 0, như thể người dùng không có ý kiến gì. Kiểm tra với người dùng đầu tiên:

<!-- run -->
```python
rs = uuCF(rate_train, k = 40); rs.fit()
ids = np.where(rate_train[:, 0] == 0)[0][:6]
print("điểm gốc:            ", rate_train[ids, 2], "; trung bình:", round(rs.mu[0], 3))
print("điểm chuẩn hoá đúng:  ", (rate_train[ids, 2] - rs.mu[0]).round(3))
print("trong Ybar của sách:  ", rs.Ybar[rate_train[ids, 1], 0].toarray().ravel())
```

<!-- output -->
```output
điểm gốc:             [5 3 4 3 3 5] ; trung bình: 3.603
điểm chuẩn hoá đúng:   [ 1.4 -0.6  0.4 -0.6 -0.6  1.4]
trong Ybar của sách:   [1 0 0 0 0 1]
```
<!-- /output -->

Sửa bằng một dòng: sao chép dữ liệu dưới dạng số thực.

<!-- run -->
```python
class uuCF_fixed(uuCF):
    def __init__(self, Y_data, k, sim_func=cosine_similarity):
        super().__init__(Y_data.astype(float), k, sim_func)       # sửa: dùng số thực

rs = uuCF_fixed(rate_train, k = 40); rs.fit()
print('User-user CF (đã sửa), RMSE =', test_rmse(rs, rate_test))
rs = uuCF_fixed(rate_train[:, [1, 0, 2]], k = 40); rs.fit()
print('Item-item CF (đã sửa), RMSE =', test_rmse(rs, rate_test[:, [1, 0, 2]]))
```

<!-- output -->
```output
User-user CF (đã sửa), RMSE = 0.9580454278556474
Item-item CF (đã sửa), RMSE = 0.935935038919037
```
<!-- /output -->

Cả hai phương pháp đều tốt hơn sau khi sửa (0.977 → 0.958 và 0.969 → 0.936), và nhận xét của sách (lọc cộng tác sản phẩm tốt hơn lọc cộng tác người dùng) vẫn đúng.

## 4. Ảnh hưởng của số láng giềng $k$

<!-- run -->
```python
rs_u = uuCF_fixed(rate_train, k=40); rs_u.fit()
rs_i = uuCF_fixed(rate_train[:, [1, 0, 2]], k=40); rs_i.fit()
for k in [5, 10, 20, 40, 80, 160]:
    rs_u.k = rs_i.k = k
    print(f"k = {k:3d}: user-user {test_rmse(rs_u, rate_test):.4f}, item-item {test_rmse(rs_i, rate_test[:, [1, 0, 2]]):.4f}")
```

<!-- output -->
```output
k =   5: user-user 1.0001, item-item 0.9649
k =  10: user-user 0.9709, item-item 0.9396
k =  20: user-user 0.9611, item-item 0.9354
k =  40: user-user 0.9580, item-item 0.9359
k =  80: user-user 0.9600, item-item 0.9403
k = 160: user-user 0.9640, item-item 0.9430
```
<!-- /output -->

Với user-user, $k$ quá nhỏ (5) cho kết quả kém hẳn: trung bình của quá ít láng giềng rất nhiễu. Trong khoảng $k$ từ 20 tới 80 kết quả gần như không đổi, nên $k = 40$ của sách là lựa chọn hợp lý. Item-item tốt nhất quanh $k = 20$–$40$.

## Tóm tắt

| Trong sách | Trên trang này |
| --- | --- |
| Ví dụ Hình 18.2 | dựng lại các ma trận a–f |
| `uuCF`, RMSE 0.9766 (user-user), 0.9688 (item-item) | khớp sách khi giữ nguyên code |
| — | lỗi cắt phần thập phân; sửa xong RMSE giảm còn khoảng 0.958 và 0.936 |
