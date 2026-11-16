# Chương 24 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. [Chương 24](math/ch24_vi) dùng thư viện CVXOPT. Trang này giải lại cùng các bài toán bằng `scipy.optimize`, thư viện có sẵn cùng SciPy, và giới thiệu thêm quy hoạch nguyên (khi các biến phải là số nguyên).

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy` và `scipy`.

Môi trường build trang này không cài CVXOPT nên code CVXOPT của sách không được chạy lại ở đây. Kết quả của nó (in trong sách) được dùng để đối chiếu.

| Lớp bài toán | CVXOPT (sách) | SciPy |
| --- | --- | --- |
| Quy hoạch tuyến tính | `solvers.lp(c, G, h, A, b)` | `linprog(c, A_ub=G, b_ub=h, A_eq=A, b_eq=b, bounds=...)` |
| Quy hoạch toàn phương | `solvers.qp(P, q, G, h)` | `minimize(..., constraints=...)` |
| Quy hoạch hình học | `solvers.gp(K, F, g)` | `minimize` sau khi đổi biến $x = e^u$ |
| Quy hoạch nguyên | không có | `milp` |

## 1. Quy hoạch tuyến tính với `linprog`

<small class="viz-ref">Mục 24.4.4</small>

Hai khác biệt cần nhớ so với CVXOPT:

- `linprog` luôn **cực tiểu** hoá; muốn cực đại hoá thì đổi dấu $\mathbf{c}$, như sách đã làm.
- `linprog` mặc định coi mọi biến **không âm** (`bounds=(0, None)`). Muốn biến nhận cả giá trị âm phải ghi rõ `bounds=(None, None)`.

<!-- run -->
```python
import numpy as np
from scipy.optimize import linprog, minimize, milp, LinearConstraint, Bounds

# Bài toán canh tác (24.2), viết đúng như code CVXOPT của sách
c = np.array([-5., -3])
G = np.array([[1., 1], [2, 1], [1, 4], [-1, 0], [0, -1]])
h = np.array([10., 16, 32, 0, 0])
res = linprog(c, A_ub=G, b_ub=h, bounds=(None, None))
print("canh tác:", res.x, "; lợi nhuận", -res.fun, "; trạng thái:", res.message)
print("sách (CVXOPT): [6.00e+00, 4.00e+00]")

# Bài toán nhà xuất bản (24.1): ràng buộc đẳng thức và biến không âm (mặc định của linprog)
res = linprog([5, 10, 15, 4], A_ub=[[1, 1, 0, 0], [0, 0, 1, 1]], b_ub=[800, 700],
              A_eq=[[1, 0, 1, 0], [0, 1, 0, 1]], b_eq=[600, 400])
print("\nnhà xuất bản: (x, y, z, t) =", res.x, "; chi phí", res.fun, "x 10k")
```

<!-- output -->
```output
canh tác: [6. 4.] ; lợi nhuận 42.0 ; trạng thái: Optimization terminated successfully. (HiGHS Status 7: Optimal)
sách (CVXOPT): [6.00e+00, 4.00e+00]

nhà xuất bản: (x, y, z, t) = [600.   0.   0. 400.] ; chi phí 4600.0 x 10k
```
<!-- /output -->

## 2. Quy hoạch toàn phương

<small class="viz-ref">Mục 24.5.3</small>

Bài toán của sách: tìm điểm trong tập khả thi của bài toán canh tác gần $\mathbf{u} = (10, 10)$ nhất. SciPy không có hàm riêng cho QP; `minimize` với phương pháp `SLSQP` hoặc `trust-constr` nhận ràng buộc dạng `LinearConstraint`.

<!-- run -->
```python
u = np.array([10., 10])
obj = lambda x: 0.5 * np.sum((x - u) ** 2)
grad = lambda x: x - u
cons = LinearConstraint(G, -np.inf, h)                     # G x <= h
res = minimize(obj, x0=np.zeros(2), jac=grad, constraints=[cons], method="SLSQP")
print("SLSQP:        ", res.x.round(6), "; sách (CVXOPT): [5.00e+00, 5.00e+00]")
res2 = minimize(obj, x0=np.zeros(2), jac=grad, hess=lambda x: np.eye(2), constraints=[cons], method="trust-constr")
print("trust-constr: ", res2.x.round(6))
print("ràng buộc nào chặt:", np.isclose(G @ res.x, h, atol=1e-6))
```

<!-- output -->
```output
SLSQP:         [5. 5.] ; sách (CVXOPT): [5.00e+00, 5.00e+00]
trust-constr:  [4.999988 5.000006]
ràng buộc nào chặt: [ True False False False False]
```
<!-- /output -->

Chỉ ràng buộc đầu tiên ($x + y \le 10$) chặt: nghiệm là hình chiếu vuông góc của $\mathbf{u}$ lên đường thẳng $x + y = 10$, tức $(5, 5)$.

## 3. Quy hoạch hình học

<small class="viz-ref">Mục 24.6.3, 24.6.4</small>

SciPy không có bộ giải GP, nhưng ta có thể tự làm bước đổi biến của mục 24.6.3 rồi gọi `minimize`. Lấy log của hàm mục tiêu sau khi đổi biến cho ta dạng log-sum-exp, cũng là dạng CVXOPT dùng bên trong.

<!-- run -->
```python
from scipy.special import logsumexp

F = np.array([[-1., -1, -1], [1, 1, 0], [0, 1, 1], [1, 0, 1]])   # như ma trận F của sách (mỗi hàng là một đơn thức)
g = np.log([40., 2, 2, 2])
res = minimize(lambda u: logsumexp(F @ u + g), x0=np.zeros(3), method="BFGS", options={"gtol": 1e-10})
x = np.exp(res.x)
print("x, y, z =", x.round(8), "; sách (CVXOPT): 1.58489319")
print("x^5 =", (x ** 5).round(7), "; sách: 9.9999998")
```

<!-- output -->
```output
x, y, z = [1.58489319 1.58489319 1.58489319] ; sách (CVXOPT): 1.58489319
x^5 = [10. 10. 10.] ; sách: 9.9999998
```
<!-- /output -->

## 4. Khi các biến phải là số nguyên

<small class="viz-ref">Mục 24.1.1</small>

Sách nhận xét: ràng buộc "số tự nhiên" làm bài toán rất khó giải khi có nhiều biến. Quy hoạch tuyến tính nguyên (mixed-integer linear programming, MILP) không còn là bài toán lồi: tập khả thi chỉ gồm các điểm nguyên rời rạc. Các bộ giải như HiGHS (được `milp` của SciPy dùng) kết hợp quy hoạch tuyến tính với tìm kiếm nhánh cận.

Sửa bài toán canh tác một chút để thấy khác biệt: anh nông dân có 11 ha đất nhưng chỉ có 15.5 triệu, và chỉ được trồng theo từng ha nguyên.

<!-- run -->
```python
from itertools import product

h2 = np.array([11., 15.5, 32, 0, 0])
lp = linprog(c, A_ub=G, b_ub=h2, bounds=(None, None))
ip = milp(c, constraints=LinearConstraint(G, -np.inf, h2), integrality=np.ones(2), bounds=Bounds(0, np.inf))
print("LP (số thực):     x =", lp.x.round(3), "; lợi nhuận", round(-lp.fun, 3))
print("MILP (số nguyên): x =", ip.x, "; lợi nhuận", -ip.fun)
print("\nbốn cách làm tròn nghiệm LP:")
for r in product(*[(np.floor(v), np.ceil(v)) for v in lp.x]):
    r = np.array(r); ok = np.all(G @ r <= h2)
    print(f"  {r}: " + (f"khả thi, lợi nhuận {-c @ r}" if ok else "KHÔNG khả thi"))
```

<!-- output -->
```output
LP (số thực):     x = [4.5 6.5] ; lợi nhuận 42.0
MILP (số nguyên): x = [4. 7.] ; lợi nhuận 41.0

bốn cách làm tròn nghiệm LP:
  [4. 6.]: khả thi, lợi nhuận 38.0
  [4. 7.]: khả thi, lợi nhuận 41.0
  [5. 6.]: KHÔNG khả thi
  [5. 7.]: KHÔNG khả thi
```
<!-- /output -->

Nghiệm LP là $(4.5, 6.5)$ với lợi nhuận 42. Trong bốn cách làm tròn, hai cách cho điểm không khả thi, một cách khả thi nhưng kém (38), và chỉ một cách trùng với nghiệm nguyên tối ưu $(4, 7)$ (41). Với $n$ biến có $2^n$ cách làm tròn, và nói chung không có gì đảm bảo cách nào trong số đó tối ưu hay thậm chí khả thi. Bài toán nguyên cần được giải như một bài toán riêng.

## 5. Mã CVXOPT của sách

Để tiện đối chiếu, đây là cách gọi CVXOPT tương ứng với mục 1 và 2 (không được chạy trên trang này):

```python
from cvxopt import matrix, solvers
c = matrix([-5., -3.])
G = matrix([[1., 2., 1., -1., 0.], [1., 1., 4., 0., -1.]])   # CVXOPT nhận ma trận theo CỘT
h = matrix([10., 16., 32., 0., 0.])
solvers.options['show_progress'] = False
print(solvers.lp(c, G, h)['x'])                               # [6.00e+00, 4.00e+00]
P = matrix([[1., 0.], [0., 1.]]); q = matrix([-10., -10.])
print(solvers.qp(P, q, G, h)['x'])                            # [5.00e+00, 5.00e+00]
```

Lưu ý `matrix([[...], [...]])` của CVXOPT nhận danh sách các **cột**, ngược với NumPy. Đó là lý do ma trận $\mathbf{G}$ trong code của sách trông như bị chuyển vị.

## Tóm tắt

- `linprog`: cực tiểu hoá, biến mặc định không âm.
- QP và các bài toán lồi trơn có ràng buộc: `minimize` với `SLSQP` hoặc `trust-constr`.
- GP: đổi biến $x = e^u$ rồi tối ưu dạng log-sum-exp.
- Biến nguyên: `milp`. Bài toán không còn lồi; đừng làm tròn nghiệm của LP.
