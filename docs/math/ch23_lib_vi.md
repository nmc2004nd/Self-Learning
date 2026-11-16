# Chương 23 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang giới thiệu các công cụ làm việc với tập lồi và hàm lồi của [chương 23](math/ch23_vi): `scipy.spatial.ConvexHull` cho bao lồi, `scipy.optimize.linprog` để kiểm tra một điểm có là tổ hợp lồi của các điểm khác không, và SymPy để kiểm tra tính lồi bằng ma trận Hessian dạng ký hiệu.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `scipy`, `sympy`, `matplotlib`.

## 1. `ConvexHull`: bao lồi trong không gian nhiều chiều

<small class="viz-ref">Mục 23.2.4 · Hình 23.5</small>

`scipy.spatial.ConvexHull` dùng thư viện Qhull, làm việc được trong không gian nhiều chiều. Ngoài các đỉnh, nó trả về `equations`: mỗi mặt của bao lồi là một siêu phẳng $\mathbf{a}^T\mathbf{x} + b = 0$, và bao lồi chính là giao của các nửa không gian $\mathbf{a}^T\mathbf{x} + b \le 0$ (Hình 23.4b).

<!-- run -->
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

rng = np.random.default_rng(0)
for d in [2, 3, 5]:
    P = rng.standard_normal((200, d))
    hull = ConvexHull(P)
    print(f"d = {d}: {len(hull.vertices):3d} đỉnh, {len(hull.equations):4d} mặt (siêu phẳng), thể tích {hull.volume:.3f}")

P = rng.standard_normal((40, 2))
hull = ConvexHull(P)
z_in, z_out = P.mean(0), np.array([4.0, 0.0])
inside = lambda z: np.all(hull.equations[:, :2] @ z + hull.equations[:, 2] <= 1e-12)
print("\ntrọng tâm nằm trong bao lồi:", inside(z_in), "; điểm (4, 0):", inside(z_out))
```

<!-- output -->
```output
d = 2:   8 đỉnh,    8 mặt (siêu phẳng), thể tích 23.658
d = 3:  27 đỉnh,   50 mặt (siêu phẳng), thể tích 61.458
d = 5:  88 đỉnh, 1264 mặt (siêu phẳng), thể tích 374.764

trọng tâm nằm trong bao lồi: True ; điểm (4, 0): False
```
<!-- /output -->

## 2. Một điểm có là tổ hợp lồi của các điểm cho trước không?

<small class="viz-ref">Định nghĩa 23.5</small>

$\mathbf{z}$ thuộc bao lồi của $\mathbf{x}_1, \dots, \mathbf{x}_N$ khi và chỉ khi tồn tại $\theta_i \ge 0$, $\sum_i \theta_i = 1$, $\sum_i \theta_i\mathbf{x}_i = \mathbf{z}$. Đây là một bài toán **quy hoạch tuyến tính** ([chương 24](math/ch24_vi)) chỉ cần tìm một điểm khả thi. Cách này dùng được cả khi số chiều lớn, khi việc dựng bao lồi là quá đắt.

<!-- run -->
```python
def convex_weights(X, z):
    N = len(X)
    A_eq = np.vstack([X.T, np.ones(N)])                  # sum theta_i x_i = z, sum theta_i = 1
    b_eq = np.append(z, 1.0)
    res = linprog(c=np.zeros(N), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * N)
    return res.x if res.status == 0 else None

for z in [z_in, z_out]:
    w = convex_weights(P, z)
    print(f"z = {z.round(3)}: " + ("không phải tổ hợp lồi" if w is None else
          f"là tổ hợp lồi của {np.sum(w > 1e-9)} điểm, dựng lại z: {(w @ P).round(6)}"))

X50 = rng.standard_normal((500, 50))                    # 500 điểm trong không gian 50 chiều
print("\n50 chiều: trung bình các điểm là tổ hợp lồi?", convex_weights(X50, X50.mean(0)) is not None,
      "; điểm 3 * x_1?", convex_weights(X50, 3 * X50[0]) is not None)
```

<!-- output -->
```output
z = [-0.033  0.   ]: là tổ hợp lồi của 3 điểm, dựng lại z: [-0.033241  0.000162]
z = [4. 0.]: không phải tổ hợp lồi

50 chiều: trung bình các điểm là tổ hợp lồi? True ; điểm 3 * x_1? False
```
<!-- /output -->

Tổ hợp lồi tìm được chỉ dùng vài điểm, dù có 40 điểm để chọn. Đây là định lý Carathéodory: trong không gian $d$ chiều, mọi điểm của bao lồi là tổ hợp lồi của nhiều nhất $d + 1$ điểm.

## 3. SymPy: kiểm tra tính lồi bằng Hessian

<small class="viz-ref">Mục 23.3.6 · Điều kiện bậc hai</small>

Với hàm có công thức, ta có thể tính Hessian dạng ký hiệu rồi kiểm tra nó nửa xác định dương. Ở trang [Code](math/ch23_code_vi), việc này được làm bằng số tại vài trăm điểm; ở đây kết luận đúng trên cả miền xác định.

<!-- run -->
```python
import sympy as sp

x, y = sp.symbols("x y", positive=True)
u, v = sp.symbols("u v", real=True)
cases = [
    ("x log x + y log y (x, y > 0)", x * sp.log(x) + y * sp.log(y), (x, y)),
    ("log(e^u + e^v)",               sp.log(sp.exp(u) + sp.exp(v)),  (u, v)),
    ("u^2 + u v + v^2",              u ** 2 + u * v + v ** 2,         (u, v)),
    ("u v",                          u * v,                           (u, v)),
    ("-exp(u + v)",                  -sp.exp(u + v),                  (u, v)),
]
for name, f, xs in cases:
    H = sp.simplify(sp.hessian(f, xs))
    tr, det = sp.simplify(H.trace()), sp.simplify(H.det())
    # Ma trận đối xứng 2x2 nửa xác định dương khi và chỉ khi vết >= 0 và định thức >= 0
    print(f"{name:30s}: Hessian = {H.tolist()}, vết = {tr}, định thức = {det}")
```

<!-- output -->
```output
x log x + y log y (x, y > 0)  : Hessian = [[1/x, 0], [0, 1/y]], vết = (x + y)/(x*y), định thức = 1/(x*y)
log(e^u + e^v)                : Hessian = [[exp(u + v)/(exp(2*u) + exp(2*v) + 2*exp(u + v)), -exp(u + v)/(exp(u) + exp(v))**2], [-exp(u + v)/(exp(u) + exp(v))**2, exp(u + v)/(exp(2*u) + exp(2*v) + 2*exp(u + v))]], vết = 2*exp(u + v)/(exp(2*u) + exp(2*v) + 2*exp(u + v)), định thức = 0
u^2 + u v + v^2               : Hessian = [[2, 1], [1, 2]], vết = 4, định thức = 3
u v                           : Hessian = [[0, 1], [1, 0]], vết = 0, định thức = -1
-exp(u + v)                   : Hessian = [[-exp(u + v), -exp(u + v)], [-exp(u + v), -exp(u + v)]], vết = -2*exp(u + v), định thức = 0
```
<!-- /output -->

Đọc kết quả:

- $x\log x + y\log y$: vết và định thức đều dương với $x, y > 0$, nên lồi chặt.
- log-sum-exp: định thức bằng 0, vết dương: nửa xác định dương, lồi nhưng không lồi chặt.
- $u^2 + uv + v^2$: định thức 3, vết 4: lồi chặt.
- $uv$: định thức $-1 < 0$: Hessian có một trị riêng âm, hàm không lồi (hình yên ngựa).
- $-e^{u + v}$: vết âm, nên không lồi (thực ra là lõm), dù tựa lồi.

## 4. Xây hàm lồi từ các khối lồi

Trong thực tế, người ta hiếm khi chứng minh tính lồi từ định nghĩa. Thay vào đó, ta ghép những hàm đã biết là lồi bằng các phép toán bảo toàn tính lồi (mục 23.3.2): cộng có trọng số không âm, lấy max, hợp với hàm affine. Đó là ý tưởng của thư viện CVXPY: nó tự kiểm tra một biểu thức có được ghép đúng quy tắc hay không (gọi là *disciplined convex programming*). Ví dụ (đoạn code này không được chạy trên trang vì môi trường build không cài CVXPY):

```python
import cvxpy as cp
w = cp.Variable(3)
print((cp.norm(w, 1) + cp.sum_squares(w)).is_convex())    # True: tổng hai hàm lồi
print(cp.log_sum_exp(w).is_convex())                       # True
print((-cp.sum_squares(w)).is_convex())                    # False: hàm lõm
print(cp.sqrt(cp.sum_squares(w)).is_dcp())                # False: sqrt (lõm, tăng) của hàm lồi không theo quy tắc,
                                                           # dù kết quả là ||w||_2, một hàm lồi; viết cp.norm(w, 2) thì đúng
```

Ví dụ cuối cho thấy giới hạn của cách này: quy tắc ghép chỉ là điều kiện đủ. Một hàm lồi viết theo cách "sai quy tắc" sẽ bị từ chối, và ta phải viết lại nó bằng các khối có sẵn.

## Tóm tắt

| Việc cần làm | Công cụ |
| --- | --- |
| Bao lồi, các mặt của nó, kiểm tra một điểm nằm trong | `scipy.spatial.ConvexHull` (`vertices`, `equations`) |
| Kiểm tra tổ hợp lồi trong không gian nhiều chiều | `scipy.optimize.linprog` (bài toán khả thi) |
| Chứng minh tính lồi của một công thức | Hessian dạng ký hiệu với SymPy |
| Mô hình hoá bài toán lồi theo quy tắc ghép | CVXPY |
