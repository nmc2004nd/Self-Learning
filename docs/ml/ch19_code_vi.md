# Chương 19 · Code từ đầu

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang chạy lại lớp `MF` của [mục 19.3](ml/ch19_vi) trên MovieLens 100k, chỉ ra rằng với tham số của sách các vector ẩn bị triệt tiêu hoàn toàn, rồi tự cài đặt phân tích ma trận bằng bình phương tối thiểu luân phiên (alternating least squares, ALS) với hệ số phạt đặt đúng thang.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `pandas`. Mục 1 tải bộ MovieLens 100k ở lần chạy đầu. Mục 1 chạy mất khoảng nửa phút.

## 1. Lớp `MF` của sách

<small class="viz-ref">Mục 19.2, 19.3</small>

Code giữ như sách, sửa hai lỗi làm nó không chạy được: thân vòng lặp trong `loss` thiếu thụt lề, và có dấu cách sau dấu `\` nối dòng. Sách không cố định hạt giống ngẫu nhiên; ở đây dùng `np.random.seed(0)`.

<!-- run -->
```python
import numpy as np
import pandas as pd
import urllib.request, zipfile
from pathlib import Path

class MF(object):
    def __init__(self, Y, K, lam = 0.1, Xinit = None, Winit = None,
        learning_rate = 0.5, max_iter = 1000, print_every = 100):
        self.Y = Y  # represents the utility matrix
        self.K = K
        self.lam = lam  # regularization parameter
        self.learning_rate = learning_rate # for gradient descent
        self.max_iter = max_iter # maximum number of iterations
        self.print_every = print_every # print loss after each a few iters
        self.n_users = int(np.max(Y[:, 0])) + 1
        self.n_items = int(np.max(Y[:, 1])) + 1
        self.n_ratings = Y.shape[0] # number of known ratings
        self.X = np.random.randn(self.n_items, K) if Xinit is None else Xinit
        self.W = np.random.randn(K, self.n_users) if Winit is None else Winit
        self.b = np.random.randn(self.n_items) # item biases
        self.d = np.random.randn(self.n_users) # user biases

    def loss(self):
        L = 0
        for i in range(self.n_ratings):
            # user_id, item_id, rating
            n, m, rating = int(self.Y[i,0]), int(self.Y[i,1]), self.Y[i,2]
            L += 0.5*(self.X[m].dot(self.W[:, n]) + self.b[m] + self.d[n] - rating)**2
        L /= self.n_ratings
        # regularization, don't ever forget this
        return L + 0.5*self.lam*(np.sum(self.X**2) + np.sum(self.W**2))

    def updateXb(self):
        for m in range(self.n_items):
            ids = np.where(self.Y[:, 1] == m)[0]
            user_ids, ratings=self.Y[ids, 0].astype(np.int32),self.Y[ids, 2]
            Wm, dm = self.W[:, user_ids], self.d[user_ids]
            for i in range(30):
                xm = self.X[m]
                error = xm.dot(Wm) + self.b[m] + dm - ratings
                grad_xm = error.dot(Wm.T)/self.n_ratings + self.lam*xm
                grad_bm = np.sum(error)/self.n_ratings
                self.X[m] -= self.learning_rate*grad_xm.reshape(-1)
                self.b[m] -= self.learning_rate*grad_bm

    def updateWd(self):
        for n in range(self.n_users):
            ids = np.where(self.Y[:,0] == n)[0]
            item_ids,ratings=self.Y[ids, 1].astype(np.int32), self.Y[ids, 2]
            Xn, bn = self.X[item_ids], self.b[item_ids]
            for i in range(30):
                wn = self.W[:, n]
                error = Xn.dot(wn) + bn + self.d[n] - ratings
                grad_wn = Xn.T.dot(error)/self.n_ratings + self.lam*wn
                grad_dn = np.sum(error)/self.n_ratings
                self.W[:, n] -= self.learning_rate*grad_wn.reshape(-1)
                self.d[n] -= self.learning_rate*grad_dn

    def fit(self):
        for it in range(self.max_iter):
            self.updateWd()
            self.updateXb()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y)
                print('iter = %d, loss = %.4f, RMSE train = %.4f'%(it + 1, self.loss(), rmse_train))

    def pred(self, u, i):
        u, i = int(u), int(i)
        pred = self.X[i, :].dot(self.W[:, u]) + self.b[i] + self.d[u]
        return max(0, min(5, pred))

    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2])**2
        return np.sqrt(SE/n_tests)

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

np.random.seed(0)
rs = MF(rate_train, K = 50, lam = .01, print_every = 5, learning_rate = 50, max_iter = 30)
rs.fit()
RMSE = rs.evaluate_RMSE(rate_test)
print('\nMatrix Factorization CF, RMSE = %.4f' %RMSE)
```

<!-- output -->
```output
iter = 5, loss = 0.4449, RMSE train = 0.9431
iter = 10, loss = 0.4214, RMSE train = 0.9178
iter = 15, loss = 0.4173, RMSE train = 0.9133
iter = 20, loss = 0.4160, RMSE train = 0.9120
iter = 25, loss = 0.4155, RMSE train = 0.9113
iter = 30, loss = 0.4151, RMSE train = 0.9110

Matrix Factorization CF, RMSE = 0.9623
```
<!-- /output -->

Các dòng in ra gần như trùng với sách (loss 0.4447 → 0.4152, RMSE kiểm tra 0.9621 trong sách; khác ở chữ số cuối do khởi tạo ngẫu nhiên).

## 2. Các vector ẩn đã biến mất

Xem độ lớn của các tham số sau khi huấn luyện:

<!-- run -->
```python
print(f"|X| lớn nhất = {np.abs(rs.X).max():.1e},  |W| lớn nhất = {np.abs(rs.W).max():.1e}")
print(f"|b| lớn nhất = {np.abs(rs.b).max():.2f},  |d| lớn nhất = {np.abs(rs.d).max():.2f}")
pred_bias = np.clip(rs.b[rate_test[:, 1]] + rs.d[rate_test[:, 0]], 0, 5)
print(f"RMSE nếu bỏ hẳn phần X W, chỉ giữ b + d: {np.sqrt(np.mean((pred_bias - rate_test[:, 2]) ** 2)):.4f}")
```

<!-- output -->
```output
|X| lớn nhất = 5.2e-80,  |W| lớn nhất = 2.3e-78
|b| lớn nhất = 2.09,  |d| lớn nhất = 4.66
RMSE nếu bỏ hẳn phần X W, chỉ giữ b + d: 0.9623
```
<!-- /output -->

Các phần tử của $\mathbf{X}$ và $\mathbf{W}$ chỉ còn cỡ $10^{-80}$: phần "phân tích ma trận" không đóng góp gì, và mô hình chỉ còn là $\hat{y}_{mn} = b_m + d_n$ (một độ lệch cho mỗi phim và một cho mỗi người dùng). RMSE 0.962 hoàn toàn đến từ hai độ lệch này.

Nguyên nhân nằm ở thang đo của hàm mất mát. Phần sai số được chia cho số đánh giá $s = 90\,570$, phần phạt thì không:

$$
\mathcal{L} = \frac{1}{2s}\sum_{(m, n)} (\mathbf{x}_m\mathbf{w}_n + b_m + d_n - y_{mn})^2 + \frac{\lambda}{2}\left(\|\mathbf{X}\|_F^2 + \|\mathbf{W}\|_F^2\right).
$$

Nhân cả hai vế với $s$: $\lambda = 0.01$ tương đương với hệ số phạt $0.01 \times 90\,570 \approx 900$ khi sai số được tính bằng tổng. Một phim có vài chục đánh giá không thể "chống lại" mức phạt này, nên mọi vector ẩn bị kéo về 0. (Các độ lệch $b$, $d$ không bị phạt nên vẫn học được.)

## 3. Bình phương tối thiểu luân phiên

<small class="viz-ref">Mục 19.2.3</small>

Sách tối ưu luân phiên: cố định $\mathbf{X}$ để tối ưu $\mathbf{W}$ và ngược lại, mỗi bài toán con giải bằng 30 bước gradient descent. Nhưng mỗi bài toán con là một bài toán hồi quy ridge, có nghiệm dạng đóng ([chương 7](ml/ch07_vi), [chương 8](ml/ch08_vi)). Giải thẳng bằng nghiệm dạng đóng (ALS) vừa nhanh hơn vừa không cần chọn tốc độ học. Ta viết hàm mất mát với sai số tính bằng tổng, và trừ trước trung bình toàn bộ $\mu$:

$$
\sum_{(m, n)} (\mu + b_m + d_n + \mathbf{x}_m^T\mathbf{w}_n - y_{mn})^2 + \lambda\left(\|\mathbf{X}\|_F^2 + \|\mathbf{W}\|_F^2\right).
$$

<!-- run -->
```python
n_users, n_items = 943, 1682
u_tr, i_tr, y_tr = rate_train[:, 0], rate_train[:, 1], rate_train[:, 2].astype(float)
by_user = [np.flatnonzero(u_tr == n) for n in range(n_users)]
by_item = [np.flatnonzero(i_tr == m) for m in range(n_items)]
mu = y_tr.mean()

def als(K, lam, iters=15, seed=0, lam_bias=1.0):
    rng = np.random.default_rng(seed)
    P = 0.1 * rng.standard_normal((n_users, K)); Q = 0.1 * rng.standard_normal((n_items, K))   # P: w_n, Q: x_m
    bu = np.zeros(n_users); bi = np.zeros(n_items)
    reg = np.diag(np.r_[np.full(K, lam), lam_bias])
    for it in range(iters):
        for n in range(n_users):                                  # cố định Q, bi: ridge cho (w_n, d_n)
            ids = by_user[n]; A = np.c_[Q[i_tr[ids]], np.ones(len(ids))]
            sol = np.linalg.solve(A.T @ A + reg, A.T @ (y_tr[ids] - mu - bi[i_tr[ids]]))
            P[n], bu[n] = sol[:K], sol[K]
        for m in range(n_items):                                  # cố định P, bu: ridge cho (x_m, b_m)
            ids = by_item[m]
            if len(ids) == 0: continue
            A = np.c_[P[u_tr[ids]], np.ones(len(ids))]
            sol = np.linalg.solve(A.T @ A + reg, A.T @ (y_tr[ids] - mu - bu[u_tr[ids]]))
            Q[m], bi[m] = sol[:K], sol[K]
    return P, Q, bu, bi

def rmse(model, R):
    P, Q, bu, bi = model
    pred = np.clip(mu + bu[R[:, 0]] + bi[R[:, 1]] + np.sum(P[R[:, 0]] * Q[R[:, 1]], 1), 1, 5)
    return np.sqrt(np.mean((pred - R[:, 2]) ** 2))

# Chọn K và lambda trên một tập xác thực tách từ tập huấn luyện, không nhìn tập kiểm tra
rng = np.random.default_rng(1)
val_mask = rng.random(len(rate_train)) < 0.1
full = (u_tr, i_tr, y_tr, by_user, by_item)
u_tr, i_tr, y_tr = u_tr[~val_mask], i_tr[~val_mask], y_tr[~val_mask]
by_user = [np.flatnonzero(u_tr == n) for n in range(n_users)]; by_item = [np.flatnonzero(i_tr == m) for m in range(n_items)]
val = rate_train[val_mask]
scores = {}
for K in [0, 2, 10, 50]:
    for lam in [5, 15, 30]:
        scores[(K, lam)] = rmse(als(K, lam), val)
        print(f"K = {K:2d}, lambda = {lam:2d}: RMSE xác thực {scores[(K, lam)]:.4f}")
K_best, lam_best = min(scores, key=scores.get)

u_tr, i_tr, y_tr, by_user, by_item = full                         # huấn luyện lại trên toàn bộ tập huấn luyện
model = als(K_best, lam_best)
print(f"\nchọn K = {K_best}, lambda = {lam_best}; RMSE kiểm tra {rmse(model, rate_test):.4f}  (sách: 0.9621)")
print(f"chỉ có độ lệch (K = 0), cùng lambda: RMSE kiểm tra {rmse(als(0, lam_best), rate_test):.4f}")
```

<!-- output -->
```output
K =  0, lambda =  5: RMSE xác thực 0.9328
K =  0, lambda = 15: RMSE xác thực 0.9328
K =  0, lambda = 30: RMSE xác thực 0.9328
K =  2, lambda =  5: RMSE xác thực 0.9164
K =  2, lambda = 15: RMSE xác thực 0.9137
K =  2, lambda = 30: RMSE xác thực 0.9257
K = 10, lambda =  5: RMSE xác thực 0.9375
K = 10, lambda = 15: RMSE xác thực 0.9042
K = 10, lambda = 30: RMSE xác thực 0.9251
K = 50, lambda =  5: RMSE xác thực 0.9767
K = 50, lambda = 15: RMSE xác thực 0.9021
K = 50, lambda = 30: RMSE xác thực 0.9251

chọn K = 50, lambda = 15; RMSE kiểm tra 0.9331  (sách: 0.9621)
chỉ có độ lệch (K = 0), cùng lambda: RMSE kiểm tra 0.9599
```
<!-- /output -->

Với $K = 0$, mô hình chỉ có các độ lệch và cho RMSE khoảng 0.96, đúng mức của sách. Khi các vector ẩn được phép tồn tại (hệ số phạt hợp lý), RMSE kiểm tra giảm xuống 0.933. Bảng xác thực cũng cho thấy $K$ và $\lambda$ phải đi cùng nhau: $K = 50$ với $\lambda = 5$ quá khớp (0.977), nhưng với $\lambda = 15$ lại tốt nhất. (RMSE trên tập xác thực thấp hơn trên tập kiểm tra vì tập kiểm tra `ua.test` gồm đúng 10 đánh giá của mỗi người dùng, kể cả những người ít hoạt động, nên khó hơn.)

## Tóm tắt

- Code của sách tái lập được RMSE khoảng 0.962, nhưng với $\lambda = 0.01$ và hàm mất mát chia cho số đánh giá, mọi vector ẩn bị triệt tiêu: kết quả thực chất là của mô hình chỉ có độ lệch.
- Hệ số phạt phải được hiểu cùng thang đo với phần sai số.
- ALS giải mỗi bài toán con bằng nghiệm dạng đóng của hồi quy ridge; chọn $K$ và $\lambda$ trên tập xác thực.
