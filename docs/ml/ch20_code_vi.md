# Chương 20 · Code từ đầu

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang chạy lại đoạn code của [chương 20](ml/ch20_vi), rồi tự tính SVD từ trị riêng và vector riêng ([chương 1](math/ch01_vi)), kiểm tra các tính chất trong chương (giá trị suy biến của ma trận nửa xác định dương, SVD giản lược và cắt ngọn), và định lý xấp xỉ hạng $k$ tốt nhất.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy` và `matplotlib`.

## 1. Đoạn code của sách

<small class="viz-ref">Mục 20.2.1</small>

`np.linalg.svd` trả về $\mathbf{U}$, các giá trị suy biến (một vector, không phải ma trận $\mathbf{\Sigma}$), và $\mathbf{V}^T$ (đã chuyển vị, như ghi chú trong code của sách).

<!-- run -->
```python
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

np.random.seed(0)
m, n = 3, 4
A = np.random.rand(m, n)
U, S, V = LA.svd(A) # A = U*S*V (no V transpose here)

print('Frobenius norm of (UU^T - I) =', LA.norm(U.dot(U.T) - np.eye(m)))
print('S = ', S)
print('Frobenius norm of (VV^T - I) =', LA.norm(V.dot(V.T) - np.eye(n)))

Sigma = np.zeros((m, n)); Sigma[:m, :m] = np.diag(S)        # Sigma cỡ m x n như công thức (20.2)
print('A = U Sigma V^T ?', np.allclose(U @ Sigma @ V, A))
```

<!-- output -->
```output
Frobenius norm of (UU^T - I) = 3.534614454593711e-16
S =  [2.16445627 0.55120174 0.22328251]
Frobenius norm of (VV^T - I) = 8.447945478530022e-16
A = U Sigma V^T ? True
```
<!-- /output -->

## 2. SVD từ trị riêng

<small class="viz-ref">Mục 20.2.1, 20.2.2</small>

Từ $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$ suy ra $\mathbf{A}^T\mathbf{A} = \mathbf{V}\mathbf{\Sigma}^T\mathbf{\Sigma}\mathbf{V}^T$: các cột của $\mathbf{V}$ là vector riêng của $\mathbf{A}^T\mathbf{A}$, và bình phương các giá trị suy biến là trị riêng tương ứng. Sau đó $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i / \sigma_i$.

<!-- run -->
```python
def svd_from_eig(A, tol=1e-7):
    lam, V = np.linalg.eigh(A.T @ A)                       # trị riêng tăng dần
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    s = np.sqrt(np.clip(lam, 0, None))
    r = int(np.sum(s > tol * s[0]))                        # hạng của A
    U = A @ V[:, :r] / s[:r]
    return U, s[:r], V[:, :r]                              # SVD giản lược (mục 20.2.4)

A = np.random.default_rng(0).standard_normal((6, 4))
U, s, V = svd_from_eig(A)
print("giá trị suy biến, tự tính:", s.round(6))
print("np.linalg.svd:            ", np.linalg.svd(A, compute_uv=False).round(6))
print("U^T U = I:", np.allclose(U.T @ U, np.eye(4)), "; V^T V = I:", np.allclose(V.T @ V, np.eye(4)), "; A = U diag(s) V^T:", np.allclose(U @ np.diag(s) @ V.T, A))

# Ma trận hạng thấp: SVD giản lược chỉ giữ r thành phần
B = np.random.default_rng(1).standard_normal((6, 2)) @ np.random.default_rng(2).standard_normal((2, 5))
Ub, sb, Vb = svd_from_eig(B)
print("\nma trận 6x5 hạng 2: số giá trị suy biến khác 0 =", len(sb), "; dựng lại đúng:", np.allclose(Ub @ np.diag(sb) @ Vb.T, B))
```

<!-- output -->
```output
giá trị suy biến, tự tính: [3.036073 2.03354  1.856326 0.895371]
np.linalg.svd:             [3.036073 2.03354  1.856326 0.895371]
U^T U = I: True ; V^T V = I: True ; A = U diag(s) V^T: True

ma trận 6x5 hạng 2: số giá trị suy biến khác 0 = 2 ; dựng lại đúng: True
```
<!-- /output -->

Ngưỡng `tol` ở đây phải khá lớn ($10^{-7}$ so với giá trị lớn nhất): với ngưỡng $10^{-12}$, các giá trị suy biến "bằng 0" của ma trận hạng 2 bị tính ra cỡ $10^{-8}$ và bị nhầm là khác 0. Lý do nằm ở chỗ: cách này dễ hiểu nhưng kém chính xác về số học: tạo $\mathbf{A}^T\mathbf{A}$ làm bình phương số điều kiện ([trang Thư viện chương 1](math/ch01_lib_vi)), nên các giá trị suy biến nhỏ bị mất chính xác. Các thư viện dùng những thuật toán làm việc trực tiếp trên $\mathbf{A}$.

<!-- run -->
```python
H = np.array([[1. / (i + j + 1) for j in range(8)] for i in range(10)])    # ma trận kiểu Hilbert, rất xấu về số học
_, s_eig, _ = svd_from_eig(H, tol=0)
s_lib = np.linalg.svd(H, compute_uv=False)
for k in [0, 4, 6, 7]:
    print(f"sigma_{k + 1}: np.linalg.svd {s_lib[k]:.3e}, qua A^T A {s_eig[k]:.3e}")
```

<!-- output -->
```output
sigma_1: np.linalg.svd 1.723e+00, qua A^T A 1.723e+00
sigma_5: np.linalg.svd 8.309e-05, qua A^T A 8.309e-05
sigma_7: np.linalg.svd 4.656e-08, qua A^T A 4.709e-08
sigma_8: np.linalg.svd 4.831e-10, qua A^T A 2.892e-09
```
<!-- /output -->

## 3. Giá trị suy biến của ma trận nửa xác định dương

<small class="viz-ref">Mục 20.2.3</small>

Với ma trận đối xứng nửa xác định dương, SVD trùng với phân tích trị riêng: $\mathbf{U} = \mathbf{V}$ và giá trị suy biến bằng trị riêng. Với ma trận đối xứng có trị riêng âm thì không:

<!-- run -->
```python
X = np.random.default_rng(3).standard_normal((5, 3))
P = X @ X.T                                               # nửa xác định dương, hạng 3
print("trị riêng:      ", np.sort(np.linalg.eigvalsh(P))[::-1].round(5) + 0.0)
print("giá trị suy biến:", np.linalg.svd(P, compute_uv=False).round(5))
Ssym = np.array([[1., 2], [2, -3]])                       # đối xứng nhưng không xác định dương
print("\nma trận [[1, 2], [2, -3]]: trị riêng", np.linalg.eigvalsh(Ssym).round(4), "; giá trị suy biến", np.linalg.svd(Ssym, compute_uv=False).round(4))
```

<!-- output -->
```output
trị riêng:       [20.81366  6.3676   2.00319  0.       0.     ]
giá trị suy biến: [20.81366  6.3676   2.00319  0.       0.     ]

ma trận [[1, 2], [2, -3]]: trị riêng [-3.8284  1.8284] ; giá trị suy biến [3.8284 1.8284]
```
<!-- /output -->

Khi có trị riêng âm, giá trị suy biến là trị tuyệt đối của trị riêng.

## 4. Xấp xỉ hạng $k$ tốt nhất

<small class="viz-ref">Mục 20.2.5, 20.2.6</small>

SVD cắt ngọn $\mathbf{A}_k = \sum_{i=1}^k \sigma_i\mathbf{u}_i\mathbf{v}_i^T$ là ma trận hạng $k$ gần $\mathbf{A}$ nhất (định lý Eckart–Young), với sai số $\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \sum_{i > k}\sigma_i^2$. Kiểm tra: không ma trận hạng $k$ ngẫu nhiên nào, kể cả sau khi tối ưu cục bộ, tốt hơn $\mathbf{A}_k$.

<!-- run -->
```python
rng = np.random.default_rng(4)
A = rng.standard_normal((30, 20))
U, s, Vt = np.linalg.svd(A, full_matrices=False)
for k in [1, 5, 10]:
    Ak = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
    err = np.linalg.norm(A - Ak) ** 2
    # đối thủ: tối ưu B C (B: 30 x k, C: k x 20) bằng bình phương tối thiểu luân phiên từ điểm ngẫu nhiên
    best_other = np.inf
    for trial in range(5):
        B = rng.standard_normal((30, k))
        for _ in range(20):
            C = np.linalg.lstsq(B, A, rcond=None)[0]
            B = np.linalg.lstsq(C.T, A.T, rcond=None)[0].T
        best_other = min(best_other, np.linalg.norm(A - B @ C) ** 2)
    print(f"k = {k:2d}: ||A - A_k||^2 = {err:8.4f}, tổng sigma_i^2 bỏ đi = {np.sum(s[k:] ** 2):8.4f}, "
          f"tốt nhất của đối thủ = {best_other:8.4f}, ma trận hạng {k} ngẫu nhiên: {np.linalg.norm(A - rng.standard_normal((30, k)) @ rng.standard_normal((k, 20))) ** 2:8.1f}")
```

<!-- output -->
```output
k =  1: ||A - A_k||^2 = 516.9096, tổng sigma_i^2 bỏ đi = 516.9096, tốt nhất của đối thủ = 516.9304, ma trận hạng 1 ngẫu nhiên:   1507.9
k =  5: ||A - A_k||^2 = 255.2026, tổng sigma_i^2 bỏ đi = 255.2026, tốt nhất của đối thủ = 255.2030, ma trận hạng 5 ngẫu nhiên:   2110.4
k = 10: ||A - A_k||^2 =  90.8305, tổng sigma_i^2 bỏ đi =  90.8305, tốt nhất của đối thủ =  90.8307, ma trận hạng 10 ngẫu nhiên:   7033.7
```
<!-- /output -->

Phương pháp bình phương tối thiểu luân phiên (cũng là ý tưởng của [chương 19](ml/ch19_vi)) đạt đúng sai số của SVD cắt ngọn, không thấp hơn: SVD cho nghiệm tối ưu toàn cục của bài toán xấp xỉ hạng thấp, dù bài toán này không lồi.

## 5. Phương pháp lũy thừa cho các thành phần đầu tiên

Khi chỉ cần $k$ giá trị suy biến lớn nhất của một ma trận lớn, không cần tính toàn bộ SVD. Áp dụng phương pháp lũy thừa ([trang Code chương 1](math/ch01_code_vi)) cho $\mathbf{A}^T\mathbf{A}$, nhưng không bao giờ dựng ma trận đó ra: chỉ cần các phép nhân $\mathbf{A}\mathbf{v}$ và $\mathbf{A}^T\mathbf{u}$.

<!-- run -->
```python
def top_k_svd(A, k, iters=200, oversample=10, seed=0):
    # lặp trên không gian con lớn hơn k một chút: hội tụ nhanh hơn khi các giá trị suy biến sát nhau
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((A.shape[1], k + oversample)))[0]
    for _ in range(iters):
        Q = np.linalg.qr(A.T @ (A @ Q))[0]                 # lũy thừa trên không gian con k chiều
    B = A @ Q                                              # m x k
    Ub, sb, Wt = np.linalg.svd(B, full_matrices=False)
    return Ub[:, :k], sb[:k], (Q @ Wt.T)[:, :k]

A = rng.standard_normal((2000, 300)) @ np.diag(np.linspace(1, 0.01, 300)) @ np.linalg.qr(rng.standard_normal((300, 300)))[0]
Uk, sk, Vk = top_k_svd(A, 5)
print("5 giá trị suy biến lớn nhất, phương pháp lũy thừa:", sk.round(6))
print("np.linalg.svd:                                    ", np.linalg.svd(A, compute_uv=False)[:5].round(6))
```

<!-- output -->
```output
5 giá trị suy biến lớn nhất, phương pháp lũy thừa: [48.840266 48.709552 48.556491 48.160566 47.215598]
np.linalg.svd:                                     [48.840266 48.709552 48.556491 48.160566 47.215598]
```
<!-- /output -->

## Tóm tắt

| Trong sách | Trên trang này |
| --- | --- |
| SVD (20.2.1) | `np.linalg.svd`; tự tính từ trị riêng của $\mathbf{A}^T\mathbf{A}$ |
| Ma trận nửa xác định dương (20.2.3) | SVD trùng phân tích trị riêng |
| SVD giản lược, cắt ngọn (20.2.4–20.2.5) | giữ $r$ hoặc $k$ thành phần đầu |
| Xấp xỉ hạng $k$ tốt nhất (20.2.6) | sai số bằng $\sum_{i > k}\sigma_i^2$; không phương pháp nào tốt hơn |
