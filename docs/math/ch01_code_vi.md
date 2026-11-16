# Chương 1 · Code từ đầu

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang cài đặt lại bằng Python thuần và NumPy những phép toán chính của [chương 1](math/ch01_vi): nhân ma trận, khử Gauss (giải hệ, định thức, nghịch đảo), hạng, Gram–Schmidt, trị riêng, phân tích Cholesky và các chuẩn. Mỗi kết quả tự viết được so với hàm có sẵn của NumPy.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Chỉ cần `numpy`.

Mục đích không phải viết lại thư viện (bản của NumPy nhanh và ổn định hơn nhiều), mà để thấy mỗi khái niệm trong chương ứng với phép tính cụ thể nào.

## 1. Phép nhân hai ma trận

<small class="viz-ref">Mục 1.3</small>

Theo định nghĩa, $c_{ij} = \sum_{k} a_{ik} b_{kj}$: ba vòng lặp lồng nhau.

<!-- run -->
```python
import numpy as np
from time import perf_counter

def matmul(A, B):
    m, n = len(A), len(A[0])
    assert n == len(B), "số cột của A phải bằng số hàng của B"
    p = len(B[0])
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

rng = np.random.default_rng(0)
A, B = rng.standard_normal((3, 4)), rng.standard_normal((4, 2))
print("khớp với A @ B:", np.allclose(matmul(A.tolist(), B.tolist()), A @ B))

# Không giao hoán, nhưng kết hợp: (AB)C = A(BC), và (AB)^T = B^T A^T
P, Q, R = (rng.standard_normal((3, 3)) for _ in range(3))
print("AB == BA ?          ", np.allclose(P @ Q, Q @ P))
print("(AB)C == A(BC) ?    ", np.allclose((P @ Q) @ R, P @ (Q @ R)))
print("(AB)^T == B^T A^T ? ", np.allclose((P @ Q).T, Q.T @ P.T))

n = 150
X, Y = rng.standard_normal((n, n)), rng.standard_normal((n, n))
t = perf_counter(); matmul(X.tolist(), Y.tolist()); t_py = perf_counter() - t
t = perf_counter(); X @ Y; t_np = perf_counter() - t
print(f"ma trận {n}x{n}: ba vòng lặp {t_py:.2f} s, NumPy {1000 * t_np:.2f} ms")
```

<!-- output -->
```output
khớp với A @ B: True
AB == BA ?           False
(AB)C == A(BC) ?     True
(AB)^T == B^T A^T ?  True
ma trận 150x150: ba vòng lặp 0.20 s, NumPy 0.44 ms
```
<!-- /output -->

Kết quả giống nhau, nhưng NumPy nhanh hơn vài trăm lần. NumPy gọi thư viện BLAS viết bằng C/Fortran, tối ưu cho bộ nhớ đệm và chạy song song. Vì vậy, khi làm việc thật, luôn viết phép toán dưới dạng ma trận thay vì vòng lặp Python.

## 2. Khử Gauss: giải hệ, định thức, nghịch đảo

<small class="viz-ref">Mục 1.4, 1.6</small>

Khử Gauss đưa $\mathbf{A}$ về ma trận tam giác trên bằng các phép biến đổi hàng. Từ đó ta có cả ba thứ:

- **Nghiệm của $\mathbf{Ax} = \mathbf{b}$:** thế ngược từ hàng cuối lên.
- **Định thức:** định thức của ma trận tam giác bằng tích các phần tử trên đường chéo. Phép cộng một bội của hàng này vào hàng khác không đổi định thức, còn mỗi lần đổi chỗ hai hàng thì đổi dấu.
- **Nghịch đảo:** khử Gauss–Jordan trên ma trận ghép $[\mathbf{A} \mid \mathbf{I}]$ cho ra $[\mathbf{I} \mid \mathbf{A}^{-1}]$.

Ở mỗi bước, ta chọn hàng có phần tử lớn nhất (về trị tuyệt đối) làm hàng trụ. Cách này tránh chia cho số rất nhỏ.

<!-- run -->
```python
def gauss_eliminate(A, b=None):
    """Trả về (U, c, số lần đổi hàng). U tam giác trên; c là b sau cùng các phép biến đổi."""
    U = np.array(A, float)
    n = len(U)
    c = None if b is None else np.array(b, float)
    swaps = 0
    for i in range(n):
        p = i + np.argmax(np.abs(U[i:, i]))          # chọn trụ
        if np.isclose(U[p, i], 0):
            continue                                  # cột này đã bằng 0: ma trận suy biến
        if p != i:
            U[[i, p]] = U[[p, i]]; swaps += 1
            if c is not None: c[[i, p]] = c[[p, i]]
        for j in range(i + 1, n):
            f = U[j, i] / U[i, i]
            U[j, i:] -= f * U[i, i:]
            if c is not None: c[j] -= f * c[i]
    return U, c, swaps

def solve(A, b):
    U, c, _ = gauss_eliminate(A, b)
    n = len(U); x = np.zeros(n)
    for i in range(n - 1, -1, -1):                     # thế ngược
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x

def det(A):
    U, _, swaps = gauss_eliminate(A)
    return (-1) ** swaps * np.prod(np.diag(U))

def inverse(A):
    n = len(A)
    M = np.hstack([np.array(A, float), np.eye(n)])    # [A | I]
    for i in range(n):
        p = i + np.argmax(np.abs(M[i:, i]))
        if np.isclose(M[p, i], 0):
            raise ValueError("ma trận suy biến, không khả nghịch")
        M[[i, p]] = M[[p, i]]
        M[i] /= M[i, i]
        for j in range(n):
            if j != i:
                M[j] -= M[j, i] * M[i]
    return M[:, n:]

A = rng.standard_normal((5, 5)); b = rng.standard_normal(5)
print("nghiệm khớp np.linalg.solve:", np.allclose(solve(A, b), np.linalg.solve(A, b)))
print(f"det tự viết {det(A):.6f}, np.linalg.det {np.linalg.det(A):.6f}")
print("A A^{-1} = I ?", np.allclose(A @ inverse(A), np.eye(5)))

# Các tính chất ở mục 1.6.2
B = rng.standard_normal((5, 5))
print("det(AB) = det(A) det(B) ?", np.isclose(det(A @ B), det(A) * det(B)))
print("det(A^T) = det(A) ?      ", np.isclose(det(A.T), det(A)))
print("det(A^{-1}) = 1/det(A) ? ", np.isclose(det(inverse(A)), 1 / det(A)))

S = np.array([[1., 2, 3], [2, 4, 6], [1, 0, 1]])   # hàng 2 = 2 x hàng 1
print("ma trận có hai hàng tỉ lệ: det =", abs(det(S)))
try:
    inverse(S)
except ValueError as e:
    print("inverse(S):", e)
```

<!-- output -->
```output
nghiệm khớp np.linalg.solve: True
det tự viết 2.630696, np.linalg.det 2.630696
A A^{-1} = I ? True
det(AB) = det(A) det(B) ? True
det(A^T) = det(A) ?       True
det(A^{-1}) = 1/det(A) ?  True
ma trận có hai hàng tỉ lệ: det = 0.0
inverse(S): ma trận suy biến, không khả nghịch
```
<!-- /output -->

## 3. Hạng của ma trận

<small class="viz-ref">Mục 1.7, 1.8</small>

Sau khi khử Gauss (không cần ma trận vuông), hạng bằng số hàng khác không. Vì làm việc với số thực dấu phẩy động, "bằng không" phải hiểu là "nhỏ hơn một ngưỡng".

<!-- run -->
```python
def rank(A, tol=1e-10):
    U = np.array(A, float)
    m, n = U.shape
    r = 0                                       # số hàng trụ đã tìm được
    for col in range(n):
        if r == m:
            break
        p = r + np.argmax(np.abs(U[r:, col]))
        if abs(U[p, col]) < tol:
            continue                            # cột này không có trụ
        U[[r, p]] = U[[p, r]]
        U[r + 1:] -= np.outer(U[r + 1:, col] / U[r, col], U[r])
        r += 1
    return r

# Ma trận 6x4 có hạng 2: mọi cột là tổ hợp tuyến tính của hai vector u, v
u, v = rng.standard_normal(6), rng.standard_normal(6)
M = np.column_stack([u, v, 2 * u - v, u + 3 * v])
print("rank tự viết:", rank(M), " np.linalg.matrix_rank:", np.linalg.matrix_rank(M))
print("rank(M) = rank(M^T) ?", rank(M) == rank(M.T))

# Định lý rank-nullity: dim(range) + dim(null) = số cột.
# Một cơ sở của null space: lấy từ phân tích SVD (xem chương 20), các vector ứng với giá trị suy biến bằng 0
_, s, Vt = np.linalg.svd(M)
N = Vt[rank(M):].T
print("dim null(M) =", N.shape[1], "; M @ N ≈ 0 ?", np.allclose(M @ N, 0))
print("rank + nullity =", rank(M) + N.shape[1], "= số cột", M.shape[1])
```

<!-- output -->
```output
rank tự viết: 2  np.linalg.matrix_rank: 2
rank(M) = rank(M^T) ? True
dim null(M) = 2 ; M @ N ≈ 0 ? True
rank + nullity = 4 = số cột 4
```
<!-- /output -->

## 4. Hệ trực chuẩn: Gram–Schmidt

<small class="viz-ref">Mục 1.9, 1.10</small>

Gram–Schmidt biến một hệ độc lập tuyến tính thành một hệ trực chuẩn sinh ra cùng không gian: lần lượt trừ đi phần chiếu lên các vector đã có, rồi chuẩn hoá.

<!-- run -->
```python
def gram_schmidt(A):
    """Các cột của A độc lập tuyến tính -> các cột của Q trực chuẩn, cùng không gian sinh."""
    Q = []
    for a in A.T:
        u = a - sum((q @ a) * q for q in Q)     # bỏ phần nằm trong span của các q trước
        Q.append(u / np.linalg.norm(u))
    return np.column_stack(Q)

A = rng.standard_normal((4, 4))
U = gram_schmidt(A)
print("U^T U = I ?", np.allclose(U.T @ U, np.eye(4)))
print("U U^T = I ?", np.allclose(U @ U.T, np.eye(4)), "(ma trận trực giao vuông)")
print("det(U) =", round(det(U), 6), "(luôn bằng 1 hoặc -1)")

x, y = rng.standard_normal(4), rng.standard_normal(4)
print("giữ nguyên tích vô hướng:", np.isclose((U @ x) @ (U @ y), x @ y), "; giữ nguyên độ dài:", np.isclose(np.linalg.norm(U @ x), np.linalg.norm(x)))

# Mục 1.10: toạ độ của x trong hệ cơ sở trực chuẩn U là y = U^T x (không cần nghịch đảo)
coords = U.T @ x
print("toạ độ mới:", coords.round(4), "; dựng lại x:", np.allclose(U @ coords, x))
```

<!-- output -->
```output
U^T U = I ? True
U U^T = I ? True (ma trận trực giao vuông)
det(U) = -1.0 (luôn bằng 1 hoặc -1)
giữ nguyên tích vô hướng: True ; giữ nguyên độ dài: True
toạ độ mới: [-1.1574  0.8879  0.0998  0.1474] ; dựng lại x: True
```
<!-- /output -->

## 5. Trị riêng và vector riêng

<small class="viz-ref">Mục 1.11, 1.12</small>

**Phương pháp lũy thừa** (power iteration): nhân liên tiếp một vector với $\mathbf{A}$ rồi chuẩn hoá. Thành phần theo vector riêng ứng với trị riêng có trị tuyệt đối lớn nhất tăng nhanh nhất, nên vector hội tụ về vector riêng đó. Với ma trận đối xứng, sau khi tìm được $(\lambda_1, \mathbf{x}_1)$, ta "xoá" nó bằng $\mathbf{A} - \lambda_1 \mathbf{x}_1 \mathbf{x}_1^T$ (deflation) rồi tìm tiếp.

<!-- run -->
```python
def power_iteration(A, iters=1000, tol=1e-12, seed=0):
    x = np.random.default_rng(seed).standard_normal(len(A))
    x /= np.linalg.norm(x)
    for it in range(iters):
        y = A @ x
        y /= np.linalg.norm(y)
        if np.linalg.norm(y - np.sign(y @ x) * x) < tol:
            break
        x = y
    lam = x @ A @ x                             # thương Rayleigh
    return lam, x, it + 1

def eig_symmetric(A):
    A = np.array(A, float); vals, vecs = [], []
    for _ in range(len(A)):
        lam, x, _ = power_iteration(A)
        vals.append(lam); vecs.append(x)
        A = A - lam * np.outer(x, x)            # deflation
    return np.array(vals), np.column_stack(vecs)

X = rng.standard_normal((4, 4))
S = X + X.T                                     # ma trận đối xứng
lam, V = eig_symmetric(S)
order = np.argsort(lam)
print("tự viết:   ", lam[order].round(6))
print("np.linalg: ", np.linalg.eigvalsh(S).round(6))
print("S v = λ v cho mọi cặp ?", np.allclose(S @ V, V * lam))
print("vector riêng trực giao (V^T V = I) ?", np.allclose(V.T @ V, np.eye(4)))
print("S = V Λ V^T (chéo hoá) ?", np.allclose(V @ np.diag(lam) @ V.T, S))
print(f"tổng trị riêng {lam.sum():.6f} = trace {np.trace(S):.6f};  tích trị riêng {lam.prod():.6f} = det {det(S):.6f}")

lam1, x1, its = power_iteration(S)
print(f"phương pháp lũy thừa: {its} vòng; |λ| lớn nhất = {abs(lam1):.6f}")
```

<!-- output -->
```output
tự viết:    [-4.79617  -0.969514  0.142702  1.900156]
np.linalg:  [-4.79617  -0.969514  0.142702  1.900156]
S v = λ v cho mọi cặp ? True
vector riêng trực giao (V^T V = I) ? True
S = V Λ V^T (chéo hoá) ? True
tổng trị riêng -3.722826 = trace -3.722826;  tích trị riêng 1.260864 = det 1.260864
phương pháp lũy thừa: 33 vòng; |λ| lớn nhất = 4.796170
```
<!-- /output -->

Phương pháp lũy thừa hội tụ chậm khi hai trị riêng lớn nhất có trị tuyệt đối gần nhau: sai số giảm theo tỉ lệ $|\lambda_2 / \lambda_1|$ sau mỗi vòng.

<!-- run -->
```python
for l2 in [0.5, 0.9, 0.99]:
    D = np.diag([1.0, l2, 0.1])
    Qr, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    _, _, its = power_iteration(Qr @ D @ Qr.T, iters=100000, tol=1e-10)
    print(f"|λ2/λ1| = {l2:4}: {its:5d} vòng")
```

<!-- output -->
```output
|λ2/λ1| =  0.5:    33 vòng
|λ2/λ1| =  0.9:   195 vòng
|λ2/λ1| = 0.99:  1909 vòng
```
<!-- /output -->

## 6. Ma trận xác định dương và phân tích Cholesky

<small class="viz-ref">Mục 1.13</small>

Một ma trận đối xứng xác định dương có thể viết thành $\mathbf{A} = \mathbf{L}\mathbf{L}^T$ với $\mathbf{L}$ tam giác dưới (phân tích Cholesky). Thuật toán chỉ gặp căn bậc hai của số âm (hoặc bằng 0) khi $\mathbf{A}$ không xác định dương, nên đây cũng là một cách *kiểm tra* tính xác định dương.

<!-- run -->
```python
def cholesky(A):
    n = len(A); L = np.zeros((n, n))
    for j in range(n):
        d = A[j, j] - L[j, :j] @ L[j, :j]
        if d <= 0:
            raise ValueError(f"không xác định dương (gặp {d:.3g} ở bước {j})")
        L[j, j] = np.sqrt(d)
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
    return L

X = rng.standard_normal((6, 4))
G = X.T @ X                                   # ma trận Gram: luôn nửa xác định dương; xác định dương vì X có hạng 4
L = cholesky(G)
print("L L^T = G ?", np.allclose(L @ L.T, G), "; khớp np.linalg.cholesky:", np.allclose(L, np.linalg.cholesky(G)))
print("trị riêng của G:", np.linalg.eigvalsh(G).round(3), "(đều dương)")

z = rng.standard_normal(4)
print("z^T G z =", round(z @ G @ z, 4), "= ||X z||^2 =", round(np.linalg.norm(X @ z) ** 2, 4))

for name, M in [("[[1, 2], [2, 1]]", np.array([[1., 2], [2, 1]])),
                ("X^T X với X có hạng 2", (lambda Y: Y.T @ Y)(rng.standard_normal((2, 4))))]:
    try:
        cholesky(M)
    except ValueError as e:
        print(f"{name}: {e}; trị riêng nhỏ nhất {np.linalg.eigvalsh(M).min():.3g}")
```

<!-- output -->
```output
L L^T = G ? True ; khớp np.linalg.cholesky: True
trị riêng của G: [0.072 1.583 4.288 9.33 ] (đều dương)
z^T G z = 18.9663 = ||X z||^2 = 18.9663
[[1, 2], [2, 1]]: không xác định dương (gặp -3 ở bước 1); trị riêng nhỏ nhất -1
X^T X với X có hạng 2: không xác định dương (gặp -2.22e-16 ở bước 2); trị riêng nhỏ nhất -6.71e-16
```
<!-- /output -->

Ví dụ thứ hai cho thấy một chỗ tế nhị. $\mathbf{X}^T\mathbf{X}$ với $\mathbf{X}$ có hạng 2 là nửa xác định dương: hai trị riêng của nó *đúng ra* bằng 0. Do sai số làm tròn, máy tính thấy chúng là những số cỡ $\pm 10^{-16}$. Trong thực tế, người ta thường cộng thêm một lượng nhỏ $\epsilon \mathbf{I}$ vào đường chéo trước khi phân tích Cholesky.

## 7. Chuẩn và vết

<small class="viz-ref">Mục 1.14, 1.15</small>

<!-- run -->
```python
def norm_p(x, p):
    if p == np.inf:
        return np.max(np.abs(x))
    return np.sum(np.abs(x) ** p) ** (1 / p)

x = np.array([3., -4, 1])
for p in [1, 2, 3, np.inf]:
    print(f"||x||_{p}: tự viết {norm_p(x, p):.6f}, np.linalg.norm {np.linalg.norm(x, p):.6f}")

# Ba điều kiện của chuẩn (Định nghĩa 1.1), kiểm tra trên nhiều cặp ngẫu nhiên
ok = True
for _ in range(1000):
    a, b, al = rng.standard_normal(5), rng.standard_normal(5), rng.standard_normal()
    for p in [1, 2, 3, np.inf]:
        ok &= norm_p(a + b, p) <= norm_p(a, p) + norm_p(b, p) + 1e-12
        ok &= np.isclose(norm_p(al * a, p), abs(al) * norm_p(a, p))
print("bất đẳng thức tam giác và tính thuần nhất đúng trên 1000 cặp:", ok)

# Với p < 1, công thức trên không còn là chuẩn: bất đẳng thức tam giác bị vi phạm
e1, e2 = np.array([1., 0]), np.array([0., 1])
print(f"p = 0.5: ||e1 + e2|| = {norm_p(e1 + e2, 0.5):.1f} > ||e1|| + ||e2|| = {norm_p(e1, 0.5) + norm_p(e2, 0.5):.1f}")

A = rng.standard_normal((3, 5))
print("||A||_F^2 = trace(A^T A) ?", np.isclose(np.sum(A ** 2), np.trace(A.T @ A)))
B = rng.standard_normal((5, 3))
print("trace(AB) = trace(BA) ?   ", np.isclose(np.trace(A @ B), np.trace(B @ A)), f"(AB là {A.shape[0]}x{A.shape[0]}, BA là {B.shape[0]}x{B.shape[0]})")
```

<!-- output -->
```output
||x||_1: tự viết 8.000000, np.linalg.norm 8.000000
||x||_2: tự viết 5.099020, np.linalg.norm 5.099020
||x||_3: tự viết 4.514357, np.linalg.norm 4.514357
||x||_inf: tự viết 4.000000, np.linalg.norm 4.000000
bất đẳng thức tam giác và tính thuần nhất đúng trên 1000 cặp: True
p = 0.5: ||e1 + e2|| = 4.0 > ||e1|| + ||e2|| = 2.0
||A||_F^2 = trace(A^T A) ? True
trace(AB) = trace(BA) ?    True (AB là 3x3, BA là 5x5)
```
<!-- /output -->

## Tóm tắt

| Khái niệm trong chương | Cách tính trên trang này | Hàm NumPy tương ứng |
| --- | --- | --- |
| Tích ma trận | ba vòng lặp | `A @ B` |
| Nghiệm hệ, định thức, nghịch đảo | khử Gauss có chọn trụ | `np.linalg.solve`, `det`, `inv` |
| Hạng | đếm hàng trụ sau khi khử | `np.linalg.matrix_rank` |
| Hệ trực chuẩn | Gram–Schmidt | `np.linalg.qr` |
| Trị riêng (ma trận đối xứng) | lũy thừa + deflation | `np.linalg.eigh` |
| Xác định dương | Cholesky thành công | `np.linalg.cholesky` |
| Chuẩn, vết | công thức trực tiếp | `np.linalg.norm`, `np.trace` |
