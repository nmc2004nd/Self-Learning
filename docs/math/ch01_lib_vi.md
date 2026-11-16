# Chương 1 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang giới thiệu các hàm đại số tuyến tính của NumPy và SciPy dùng cho những khái niệm ở [chương 1](math/ch01_vi), và những cái bẫy số học thường gặp khi dùng chúng.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy` và `scipy`.

| Trong sách | NumPy | SciPy |
| --- | --- | --- |
| Tích, chuyển vị | `A @ B`, `A.T`, `A.conj().T` | |
| Nghịch đảo, giải hệ | `np.linalg.inv`, `np.linalg.solve` | `scipy.linalg.solve(..., assume_a="pos")` |
| Định thức | `np.linalg.det`, `np.linalg.slogdet` | |
| Hạng, range, null space | `np.linalg.matrix_rank` | `scipy.linalg.orth`, `scipy.linalg.null_space` |
| Trị riêng | `np.linalg.eig`, `np.linalg.eigh` | `scipy.linalg.eigh(..., subset_by_index=...)` |
| Phân tích ma trận | `np.linalg.qr`, `np.linalg.cholesky` | `scipy.linalg.lu`, `scipy.linalg.cho_factor` |
| Chuẩn, vết | `np.linalg.norm`, `np.trace` | |

## 1. Giải hệ: `solve` thay vì `inv`

<small class="viz-ref">Mục 1.4.2</small>

Công thức $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$ rất gọn trên giấy. Trên máy tính, ta giải trực tiếp bằng `np.linalg.solve` (khử Gauss, khoảng $\tfrac{2}{3}n^3$ phép tính) thay vì tính $\mathbf{A}^{-1}$ (khoảng $2n^3$ phép tính) rồi nhân với $\mathbf{b}$. Ngoài chuyện tốn thêm phép tính, cách thứ hai còn kém chính xác hơn.

Ma trận Hilbert $h_{ij} = 1/(i + j - 1)$ là ví dụ kinh điển về ma trận *gần suy biến*. **Số điều kiện** (condition number) $\kappa(\mathbf{A}) = \|\mathbf{A}\|_2 \|\mathbf{A}^{-1}\|_2$ đo mức độ nhạy của nghiệm với sai số. Số này lớn cỡ $10^k$ thì ta mất khoảng $k$ chữ số chính xác.

<!-- run -->
```python
import numpy as np
import scipy.linalg as sla

print(" n   số điều kiện  sai số (solve)  sai số (inv @ b)")
for n in [4, 8, 12]:
    H = sla.hilbert(n)
    x_true = np.ones(n)
    b = H @ x_true
    e1 = np.linalg.norm(np.linalg.solve(H, b) - x_true)
    e2 = np.linalg.norm(np.linalg.inv(H) @ b - x_true)
    print(f"{n:2d}   {np.linalg.cond(H):12.2e}  {e1:14.2e}  {e2:16.2e}")
```

<!-- output -->
```output
 n   số điều kiện  sai số (solve)  sai số (inv @ b)
 4       1.55e+04        8.27e-14          1.29e-12
 8       1.53e+10        1.73e-07          2.64e-06
12       1.75e+16        1.13e+00          3.26e+01
```
<!-- /output -->

Cả hai cách đều mất dần độ chính xác khi số điều kiện tăng, nhưng `inv` luôn kém hơn khoảng 10–30 lần. Với $n = 12$, số điều kiện vượt $10^{16}$, tức vượt quá độ chính xác của số thực 64 bit (khoảng 16 chữ số): nghiệm tìm được không còn ý nghĩa, dù máy không báo lỗi gì.

## 2. Định thức: `det` tràn số, `slogdet` thì không

<small class="viz-ref">Mục 1.6</small>

Định thức là tích của $n$ số. Với $n$ lớn, tích này dễ vượt quá khoảng biểu diễn được của số thực dấu phẩy động (khoảng $10^{\pm 308}$). `slogdet` trả về dấu và logarit của trị tuyệt đối, nên không bị tràn. Đây là cách tính $\log \det \mathbf{\Sigma}$ trong mật độ phân phối chuẩn nhiều chiều ([chương 3](math/ch03_vi)).

<!-- run -->
```python
rng = np.random.default_rng(0)
for n, c in [(500, 0.1), (500, 10.0)]:
    A = c * np.eye(n) + 0.01 * rng.standard_normal((n, n))
    sign, logdet = np.linalg.slogdet(A)
    print(f"{n}x{n}, đường chéo ≈ {c}: det = {np.linalg.det(A)}, slogdet: dấu {sign:+.0f}, log|det| = {logdet:.2f}")
```

<!-- output -->
```output
500x500, đường chéo ≈ 0.1: det = 0.0, slogdet: dấu +1, log|det| = -949.44
500x500, đường chéo ≈ 10.0: det = inf, slogdet: dấu +1, log|det| = 1151.29
```
<!-- /output -->

## 3. Trị riêng: `eig` và `eigh`

<small class="viz-ref">Mục 1.11, 1.12</small>

- `np.linalg.eig` dùng cho mọi ma trận vuông. Trị riêng có thể là số phức, và không được sắp xếp.
- `np.linalg.eigh` chỉ dùng cho ma trận đối xứng (hoặc Hermitian). Nó trả trị riêng thực theo thứ tự tăng dần và vector riêng trực chuẩn.

<!-- run -->
```python
R = np.array([[0., -1], [1, 0]])                     # phép quay 90 độ: không vector nào giữ nguyên phương
print("trị riêng của phép quay 90°:", np.linalg.eig(R).eigenvalues)

# Ma trận đối xứng có trị riêng lặp (1, 1, 1, 3)
Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
S4 = Q @ np.diag([1., 1., 1., 3.]) @ Q.T; S4 = (S4 + S4.T) / 2
w1, V1 = np.linalg.eig(S4); w2, V2 = np.linalg.eigh(S4)
print("eig: ", w1.round(6), "; sai lệch lớn nhất của V^T V so với I:", f"{np.abs(V1.T @ V1 - np.eye(4)).max():.2g}")
print("eigh:", w2.round(6), "; sai lệch lớn nhất của V^T V so với I:", f"{np.abs(V2.T @ V2 - np.eye(4)).max():.2g}")

X = rng.standard_normal((300, 300)); S = X + X.T
w2 = np.linalg.eigvalsh(S)
# Chỉ cần vài trị riêng lớn nhất: scipy tính riêng phần đó
w_top = sla.eigh(S, eigvals_only=True, subset_by_index=[297, 299])
print("ba trị riêng lớn nhất:", w_top.round(4), "; khớp:", np.allclose(w_top, w2[-3:]))
```

<!-- output -->
```output
trị riêng của phép quay 90°: [0.+1.j 0.-1.j]
eig:  [3. 1. 1. 1.] ; sai lệch lớn nhất của V^T V so với I: 0.63
eigh: [1. 1. 1. 3.] ; sai lệch lớn nhất của V^T V so với I: 4.4e-16
ba trị riêng lớn nhất: [45.62   46.1899 48.4158] ; khớp: True
```
<!-- /output -->

Phép quay 90° không có trị riêng thực: không vector khác 0 nào giữ nguyên phương sau khi quay. Với ma trận đối xứng có trị riêng lặp, `eig` vẫn tìm đúng trị riêng, nhưng các vector riêng ứng với trị riêng lặp có thể không trực giao với nhau. `eigh` luôn trả một hệ trực chuẩn, đúng như tính chất ở mục 1.12.

## 4. Hạng, range và null space

<small class="viz-ref">Mục 1.7.4, 1.8</small>

<!-- run -->
```python
u, v = rng.standard_normal(5), rng.standard_normal(5)
M = np.column_stack([u, v, u + v, u - 2 * v])          # 5x4, hạng 2
print("hạng:", np.linalg.matrix_rank(M))
Rg, Nl = sla.orth(M), sla.null_space(M)
print("cơ sở trực chuẩn của range(M):", Rg.shape, "; của null(M):", Nl.shape)
print("M @ null_space ≈ 0:", np.allclose(M @ Nl, 0))

# Hạng phụ thuộc ngưỡng: nhiễu rất nhỏ làm ma trận "về lý thuyết" có hạng đầy đủ
Mn = M + 1e-9 * rng.standard_normal(M.shape)
print("thêm nhiễu 1e-9: hạng mặc định =", np.linalg.matrix_rank(Mn), "; với tol=1e-6:", np.linalg.matrix_rank(Mn, tol=1e-6))
print("giá trị suy biến:", np.linalg.svd(Mn, compute_uv=False).round(10))
```

<!-- output -->
```output
hạng: 2
cơ sở trực chuẩn của range(M): (5, 2) ; của null(M): (4, 2)
M @ null_space ≈ 0: True
thêm nhiễu 1e-9: hạng mặc định = 4 ; với tol=1e-6: 2
giá trị suy biến: [6.07797196e+00 2.12346118e+00 2.50000000e-09 1.30000000e-09]
```
<!-- /output -->

`matrix_rank` đếm số giá trị suy biến (chương 20) lớn hơn một ngưỡng. Dữ liệu thật luôn có nhiễu, nên "hạng" trong thực tế là một khái niệm gần đúng: ta nhìn xem các giá trị suy biến có giảm đột ngột ở đâu đó không.

## 5. Các phân tích ma trận

<small class="viz-ref">Mục 1.9, 1.13</small>

<!-- run -->
```python
A = rng.standard_normal((4, 4))

P, L, U = sla.lu(A)                                    # A = P L U (khử Gauss có chọn trụ)
print("LU:       A = P L U ?", np.allclose(P @ L @ U, A), "; L tam giác dưới, U tam giác trên")

Q, R = np.linalg.qr(A)                                 # A = Q R (Gram–Schmidt, cách làm ổn định hơn)
print("QR:       A = Q R ?  ", np.allclose(Q @ R, A), "; Q trực giao:", np.allclose(Q.T @ Q, np.eye(4)))

G = A.T @ A + 1e-3 * np.eye(4)
L = np.linalg.cholesky(G)
print("Cholesky: G = L L^T ?", np.allclose(L @ L.T, G))

# Phân tích một lần, giải nhiều lần với các vế phải khác nhau
c = sla.cho_factor(G)
B = rng.standard_normal((4, 3))
print("cho_solve cho 3 vế phải cùng lúc:", np.allclose(sla.cho_solve(c, B), np.linalg.solve(G, B)))
try:
    np.linalg.cholesky(np.array([[1., 2], [2, 1]]))
except np.linalg.LinAlgError as e:
    print("ma trận không xác định dương:", e)
```

<!-- output -->
```output
LU:       A = P L U ? True ; L tam giác dưới, U tam giác trên
QR:       A = Q R ?   True ; Q trực giao: True
Cholesky: G = L L^T ? True
cho_solve cho 3 vế phải cùng lúc: True
ma trận không xác định dương: Matrix is not positive definite
```
<!-- /output -->

## 6. Chuẩn, vết và `einsum`

<small class="viz-ref">Mục 1.14, 1.15</small>

<!-- run -->
```python
x = np.array([3., -4, 1])
print("chuẩn 1, 2, vô cùng:", [round(float(np.linalg.norm(x, p)), 6) for p in (1, 2, np.inf)])
A = rng.standard_normal((3, 4))
print("Frobenius:", round(np.linalg.norm(A), 6), "; chuẩn 2 của ma trận (giá trị suy biến lớn nhất):", round(np.linalg.norm(A, 2), 6))

# Khoảng cách Euclid giữa từng cặp trong hai tập điểm, không cần vòng lặp
X, Y = rng.standard_normal((1000, 3)), rng.standard_normal((500, 3))
D = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)       # (1000, 500)
D2 = np.sqrt(np.maximum((X ** 2).sum(1)[:, None] - 2 * X @ Y.T + (Y ** 2).sum(1)[None, :], 0))
print("ma trận khoảng cách", D.shape, "; hai cách khớp:", np.allclose(D, D2))

# einsum: viết công thức theo chỉ số, như trong sách
B = rng.standard_normal((4, 3))
print("trace(AB) bằng einsum:", np.isclose(np.einsum("ij,ji->", A, B), np.trace(A @ B)))
M3 = rng.standard_normal((10, 3, 3))
print("vết của 10 ma trận một lúc:", np.einsum("kii->k", M3).round(2))
```

<!-- output -->
```output
chuẩn 1, 2, vô cùng: [8.0, 5.09902, 4.0]
Frobenius: 3.43276 ; chuẩn 2 của ma trận (giá trị suy biến lớn nhất): 2.481509
ma trận khoảng cách (1000, 500) ; hai cách khớp: True
trace(AB) bằng einsum: True
vết của 10 ma trận một lúc: [ 1.1  -2.22  2.52 -3.65  0.75 -0.21  1.61 -0.92 -2.26  2.26]
```
<!-- /output -->

Cách thứ hai trong khối trên dùng khai triển $\|\mathbf{x} - \mathbf{y}\|_2^2 = \|\mathbf{x}\|_2^2 - 2\mathbf{x}^T\mathbf{y} + \|\mathbf{y}\|_2^2$. Phần chính là một phép nhân ma trận $\mathbf{X}\mathbf{Y}^T$. Đây là cách các thư viện tính khoảng cách cho $K$ lân cận ([chương 9](ml/ch09_vi)) và $K$-means ([chương 10](ml/ch10_vi)).

## Tóm tắt

- Giải hệ bằng `solve` (hoặc `cho_solve` nếu ma trận xác định dương), gần như không bao giờ cần `inv`.
- Định thức của ma trận lớn: dùng `slogdet`.
- Ma trận đối xứng: dùng `eigh` để có trị riêng thực, đã sắp xếp và vector riêng trực chuẩn.
- Số điều kiện và giá trị suy biến cho biết một bài toán có "khó" về mặt số học hay không.
