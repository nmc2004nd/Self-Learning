# Chương 25 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Các bộ giải tối ưu thường trả về cả nghiệm của bài toán đối ngẫu ([chương 25](math/ch25_vi)), tức các nhân tử Lagrange, cùng với nghiệm của bài toán gốc. Trang cho thấy cách lấy chúng ra từ `scipy.optimize`, cách đọc quy ước dấu, và kiểm tra chúng bằng các điều kiện KKT.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy` và `scipy`.

| Bộ giải | Nhân tử Lagrange nằm ở đâu |
| --- | --- |
| `linprog(method="highs")` | `res.ineqlin.marginals` (ràng buộc $\le$), `res.eqlin.marginals` (ràng buộc $=$), `res.lower/upper.marginals` (cận của biến) |
| `minimize(method="trust-constr")` | `res.v`: mỗi phần tử ứng với một ràng buộc truyền vào |
| CVXOPT (sách) | `sol['z']` (bất đẳng thức), `sol['y']` (đẳng thức) |

## 1. Nhân tử Lagrange từ `linprog`

<small class="viz-ref">Mục 25.3, 25.4</small>

<!-- run -->
```python
import numpy as np
from scipy.optimize import linprog, minimize, LinearConstraint

c = np.array([-5., -3])                                    # bài toán canh tác, min -5x - 3y
G = np.array([[1., 1], [2, 1], [1, 4]]); h = np.array([10., 16, 32])
res = linprog(c, A_ub=G, b_ub=h, bounds=[(0, None), (0, None)], method="highs")
print("x* =", res.x, "; p* =", res.fun)
print("ineqlin.marginals:", res.ineqlin.marginals + 0.0)     # + 0.0 để bỏ dấu của -0.0
print("lower.marginals (x >= 0, y >= 0):", res.lower.marginals)
```

<!-- output -->
```output
x* = [6. 4.] ; p* = -42.0
ineqlin.marginals: [-1. -2.  0.]
lower.marginals (x >= 0, y >= 0): [0. 0.]
```
<!-- /output -->

Quy ước của SciPy: `marginals` là đạo hàm của giá trị tối ưu theo vế phải $h_i$, tức $\partial p^* / \partial h_i$. Với ràng buộc $\le$ trong bài toán cực tiểu, giá trị này $\le 0$, và nhân tử Lagrange theo quy ước của sách là $\lambda_i = -\partial p^*/\partial h_i \ge 0$. Ta được $\boldsymbol{\lambda} = (1, 2, 0)$, khớp với nghiệm đối ngẫu tự giải ở [trang Code](math/ch25_code_vi).

## 2. Kiểm tra: nhân tử là độ nhạy của giá trị tối ưu

Nới ràng buộc $i$ thêm một lượng nhỏ $\delta$ (tức $h_i \to h_i + \delta$) rồi giải lại. Giá trị tối ưu thay đổi xấp xỉ $-\lambda_i\delta$:

<!-- run -->
```python
lam = -res.ineqlin.marginals
delta = 0.1
for i, name in enumerate(["x + y <= 10", "2x + y <= 16", "x + 4y <= 32"]):
    h2 = h.copy(); h2[i] += delta
    r2 = linprog(c, A_ub=G, b_ub=h2, bounds=[(0, None), (0, None)], method="highs")
    print(f"{name:14s}: lambda = {lam[i]:.1f}; p* thay đổi {r2.fun - res.fun:+.4f}, dự đoán -lambda * delta = {-lam[i] * delta + 0.0:+.4f}")
```

<!-- output -->
```output
x + y <= 10   : lambda = 1.0; p* thay đổi -0.1000, dự đoán -lambda * delta = -0.1000
2x + y <= 16  : lambda = 2.0; p* thay đổi -0.2000, dự đoán -lambda * delta = -0.2000
x + 4y <= 32  : lambda = 0.0; p* thay đổi +0.0000, dự đoán -lambda * delta = +0.0000
```
<!-- /output -->

## 3. Bài toán gốc và đối ngẫu cùng được giải

Bộ giải HiGHS giải đồng thời bài toán gốc và đối ngẫu, và dừng khi khoảng cách đối ngẫu (hiệu giữa hai giá trị mục tiêu) đủ nhỏ. Ta tự dựng bài toán đối ngẫu của (24.2) và giải riêng để đối chiếu.

Với $\min \mathbf{c}^T\mathbf{x}$ thoả mãn $\mathbf{G}\mathbf{x} \preceq \mathbf{h}$, $\mathbf{x} \succeq \mathbf{0}$, bài toán đối ngẫu là $\max\ -\mathbf{h}^T\boldsymbol{\lambda}$ thoả mãn $\mathbf{G}^T\boldsymbol{\lambda} \succeq -\mathbf{c}$, $\boldsymbol{\lambda} \succeq \mathbf{0}$ (các nhân tử của $\mathbf{x} \succeq \mathbf{0}$ đã được khử).

<!-- run -->
```python
dual = linprog(h, A_ub=-G.T, b_ub=c, bounds=[(0, None)] * 3, method="highs")   # min h^T lam  s.t. -G^T lam <= c
print("nghiệm đối ngẫu lambda* =", dual.x, "; d* =", -dual.fun, "; p* =", res.fun)
print("nhân tử của bài toán đối ngẫu chính là nghiệm của bài toán gốc:", -dual.ineqlin.marginals)
```

<!-- output -->
```output
nghiệm đối ngẫu lambda* = [1. 2. 0.] ; d* = -42.0 ; p* = -42.0
nhân tử của bài toán đối ngẫu chính là nghiệm của bài toán gốc: [6. 4.]
```
<!-- /output -->

Đối ngẫu của đối ngẫu là bài toán gốc: nhân tử Lagrange của bài toán đối ngẫu cho lại $\mathbf{x}^* = (6, 4)$.

## 4. Nhân tử từ `minimize(method="trust-constr")`

<small class="viz-ref">Mục 24.5.2</small>

<!-- run -->
```python
u = np.array([10., 10])
G5 = np.array([[1., 1], [2, 1], [1, 4], [-1, 0], [0, -1]]); h5 = np.array([10., 16, 32, 0, 0])
cons = LinearConstraint(G5, -np.inf, h5)
r = minimize(lambda x: 0.5 * np.sum((x - u) ** 2), np.zeros(2), jac=lambda x: x - u, hess=lambda x: np.eye(2),
             constraints=[cons], method="trust-constr", options={"gtol": 1e-10, "xtol": 1e-12})
print("x* =", r.x.round(5))
print("res.v (nhân tử):", np.round(r.v[0], 4))
# KKT: gradient của L theo x = (x - u) + G^T lambda = 0
lam5 = np.maximum(r.v[0], 0)
print("x* - u + G^T lambda =", (r.x - u + G5.T @ lam5).round(4) + 0.0)
```

<!-- output -->
```output
x* = [5. 5.]
res.v (nhân tử): [5. 0. 0. 0. 0.]
x* - u + G^T lambda = [0. 0.]
```
<!-- /output -->

Với `trust-constr`, dấu của `res.v` ngược với quy ước của `linprog`: ở đây nhân tử của ràng buộc chặt dương, $\lambda_1 \approx 5$, đúng như bài toán đối ngẫu tự giải. Vì quy ước dấu khác nhau giữa các thư viện, cách an toàn nhất là kiểm tra lại bằng điều kiện $\nabla_{\mathbf{x}}\mathcal{L} = \mathbf{0}$.

## Tóm tắt

- Nhân tử Lagrange (nghiệm đối ngẫu) thường có sẵn trong kết quả của bộ giải; chúng cho biết ràng buộc nào chặt và giá trị tối ưu nhạy thế nào với mỗi ràng buộc.
- Quy ước dấu khác nhau giữa các thư viện. Luôn kiểm tra bằng điều kiện KKT hoặc bằng cách nới ràng buộc rồi giải lại.
