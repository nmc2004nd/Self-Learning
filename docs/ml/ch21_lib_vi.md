# Chương 21 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang chạy lại đoạn code `sklearn.decomposition.PCA` của [mục 21.6.1](ml/ch21_vi) (trên bộ khuôn mặt Olivetti thay cho Yale), đối chiếu với bản tự viết ở [trang Code](ml/ch21_code_vi), rồi giới thiệu các tuỳ chọn quan trọng của `PCA`: chọn số chiều theo tỉ lệ phương sai, làm trắng dữ liệu, các bộ giải cho dữ liệu lớn.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `scikit-learn`.

Khác với sách, scikit-learn quy ước mỗi **hàng** là một điểm dữ liệu, nên `pca.fit(X.T)` trong code của sách nhận ma trận đã chuyển vị.

| Trong sách | `sklearn.decomposition.PCA` |
| --- | --- |
| $\bar{\mathbf{x}}$ | `mean_` |
| $\mathbf{U}_K$ (thành phần chính) | `components_` (mỗi **hàng** là một thành phần) |
| trị riêng của $\mathbf{S}$ | `explained_variance_` (chia cho $N - 1$, không phải $N$) |
| tỉ lệ phương sai (mục 21.4) | `explained_variance_ratio_` |
| $\mathbf{z} = \mathbf{U}_K^T(\mathbf{x} - \bar{\mathbf{x}})$ | `transform` |
| $\mathbf{x} \approx \mathbf{U}_K\mathbf{z} + \bar{\mathbf{x}}$ | `inverse_transform` |

## 1. Đoạn code của sách và đối chiếu

<small class="viz-ref">Mục 21.6.1</small>

<!-- run -->
```python
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()
X = faces.data.T                                   # D x N như sách: 4096 x 400
K = 100
pca = PCA(n_components=K)
pca.fit(X.T)
U = pca.components_.T                              # projection matrix, như sách
print("U:", U.shape, "; các cột trực chuẩn:", np.allclose(U.T @ U, np.eye(K), atol=1e-5))

# Đối chiếu với cách tự tính ở trang Code: trị riêng của Xhat Xhat^T / N
Xhat = X - X.mean(1, keepdims=True)
s = np.linalg.svd(Xhat, compute_uv=False)
print("explained_variance_[:3]:", pca.explained_variance_[:3].round(4))
print("sigma_i^2 / N          :", (s[:3] ** 2 / X.shape[1]).round(4))
print("sigma_i^2 / (N - 1)    :", (s[:3] ** 2 / (X.shape[1] - 1)).round(4))
print(f"{K} thành phần giữ {pca.explained_variance_ratio_.sum():.1%} phương sai")
```

<!-- output -->
```output
U: (4096, 100) ; các cột trực chuẩn: True
explained_variance_[:3]: [18.8402 11.0718  6.3046]
sigma_i^2 / N          : [18.7931 11.0441  6.2889]
sigma_i^2 / (N - 1)    : [18.8402 11.0718  6.3046]
100 thành phần giữ 93.5% phương sai
```
<!-- /output -->

`explained_variance_` chia cho $N - 1$ (phương sai mẫu không chệch, [chương 4](math/ch04_vi)), còn sách dùng $\frac{1}{N}$ ở (21.16). Các thành phần chính giống nhau; chỉ thang đo của trị riêng khác một hệ số $\frac{N}{N - 1}$.

## 2. Chọn số chiều theo tỉ lệ phương sai

<small class="viz-ref">Mục 21.4</small>

Truyền một số thực trong $(0, 1)$ cho `n_components`, `PCA` tự chọn $K$ nhỏ nhất giữ đủ tỉ lệ phương sai đó:

<!-- run -->
```python
for r in [0.8, 0.9, 0.95, 0.99]:
    p = PCA(n_components=r, svd_solver="full").fit(X.T)
    print(f"giữ {r:.0%} phương sai: K = {p.n_components_}")
```

<!-- output -->
```output
giữ 80% phương sai: K = 27
giữ 90% phương sai: K = 66
giữ 95% phương sai: K = 123
giữ 99% phương sai: K = 260
```
<!-- /output -->

## 3. Làm trắng dữ liệu

`whiten=True` chia mỗi thành phần cho độ lệch chuẩn của nó, để dữ liệu mới có ma trận hiệp phương sai bằng ma trận đơn vị. Có ích khi bước tiếp theo là một thuật toán dựa trên khoảng cách, để thành phần đầu tiên không chi phối khoảng cách (giống chuẩn hoá ở [chương 6](ml/ch06_vi)).

<!-- run -->
```python
Z = PCA(10).fit_transform(X.T)
Zw = PCA(10, whiten=True).fit_transform(X.T)
print("độ lệch chuẩn mỗi thành phần, không làm trắng:", Z.std(0).round(2))
print("độ lệch chuẩn mỗi thành phần, làm trắng:      ", Zw.std(0, ddof=1).round(2))
print("hiệp phương sai sau khi làm trắng ~ I:", np.allclose(np.cov(Zw.T), np.eye(10), atol=1e-6))
```

<!-- output -->
```output
độ lệch chuẩn mỗi thành phần, không làm trắng: [4.34 3.32 2.51 1.99 1.69 1.58 1.38 1.27 1.24 1.15]
độ lệch chuẩn mỗi thành phần, làm trắng:       [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
hiệp phương sai sau khi làm trắng ~ I: True
```
<!-- /output -->

## 4. Dữ liệu lớn: bộ giải ngẫu nhiên và PCA từng phần

<small class="viz-ref">Mục 21.5.2</small>

Mục 21.5.2 của sách nói về bài toán quy mô lớn, khi chỉ cần vài thành phần đầu của một ma trận rất lớn. scikit-learn có hai lựa chọn:

- `svd_solver="randomized"`: SVD ngẫu nhiên ([trang Thư viện chương 20](ml/ch20_lib_vi)).
- `IncrementalPCA`: xử lý dữ liệu theo từng lô, khi dữ liệu không vừa bộ nhớ.

<!-- run -->
```python
p_full = PCA(20, svd_solver="full").fit(X.T)
p_rand = PCA(20, svd_solver="randomized", random_state=0).fit(X.T)
p_inc = IncrementalPCA(20, batch_size=100).fit(X.T)              # 4 lô, mỗi lô 100 ảnh
for name, p in [("randomized", p_rand), ("IncrementalPCA", p_inc)]:
    # góc giữa hai không gian con (cos của các góc chính, bằng 1 nếu trùng nhau)
    cos = np.linalg.svd(p_full.components_ @ p.components_.T, compute_uv=False)
    print(f"{name:15s}: tỉ lệ phương sai {p.explained_variance_ratio_.sum():.4f} (đầy đủ: {p_full.explained_variance_ratio_.sum():.4f}); "
          f"cos góc chính nhỏ nhất với không gian con đầy đủ: {cos.min():.4f}")
```

<!-- output -->
```output
randomized     : tỉ lệ phương sai 0.7634 (đầy đủ: 0.7634); cos góc chính nhỏ nhất với không gian con đầy đủ: 1.0000
IncrementalPCA : tỉ lệ phương sai 0.7557 (đầy đủ: 0.7634); cos góc chính nhỏ nhất với không gian con đầy đủ: 0.7106
```
<!-- /output -->

SVD ngẫu nhiên cho đúng không gian con. `IncrementalPCA` chỉ là xấp xỉ: tỉ lệ phương sai gần bằng, nhưng vài thành phần cuối lệch đáng kể (cos góc chính nhỏ nhất 0.71), vì mỗi lô chỉ có 100 ảnh. Dùng lô lớn hơn sẽ chính xác hơn.

## 5. PCA trong một pipeline

PCA thường là bước tiền xử lý. Đặt nó trong `Pipeline` để nó chỉ học từ tập huấn luyện ([trang Thư viện chương 6](ml/ch06_lib_vi)):

<!-- run -->
```python
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(5, shuffle=True, random_state=0)
for k in [10, 30, 100]:
    pipe = make_pipeline(PCA(k, whiten=True, random_state=0), KNeighborsClassifier(1))
    print(f"PCA({k:3d}, whiten) + 1-NN: độ chính xác nhận dạng khuôn mặt {cross_val_score(pipe, faces.data, faces.target, cv=cv).mean():.3f}")
print(f"1-NN trên 4096 điểm ảnh gốc:     {cross_val_score(KNeighborsClassifier(1), faces.data, faces.target, cv=cv).mean():.3f}")
```

<!-- output -->
```output
PCA( 10, whiten) + 1-NN: độ chính xác nhận dạng khuôn mặt 0.910
PCA( 30, whiten) + 1-NN: độ chính xác nhận dạng khuôn mặt 0.932
PCA(100, whiten) + 1-NN: độ chính xác nhận dạng khuôn mặt 0.907
1-NN trên 4096 điểm ảnh gốc:     0.945
```
<!-- /output -->

Ở bộ dữ liệu này, PCA **không** làm 1-NN chính xác hơn: 1-NN trên ảnh gốc đạt 94.5%, tốt nhất với PCA là 93.2%. Lợi ích của PCA ở đây là tốc độ và bộ nhớ: mỗi ảnh còn 30 số thay vì 4096, nên tính khoảng cách nhanh hơn khoảng 100 lần với độ chính xác gần như không đổi. Giảm chiều không tự động làm mô hình tốt hơn; cần kiểm tra bằng kiểm định chéo như ở đây.

## Tóm tắt

- `components_` chứa các thành phần chính theo **hàng**; `explained_variance_` chia cho $N - 1$.
- `n_components` nhận một tỉ lệ (như 0.95) để chọn $K$ theo mục 21.4.
- Dữ liệu lớn: `svd_solver="randomized"`; không vừa bộ nhớ: `IncrementalPCA`.
