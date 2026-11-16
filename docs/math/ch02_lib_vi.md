# Chương 2 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trong thực tế, gần như không ai tự tính gradient bằng tay cho các mô hình lớn. Trang giới thiệu ba công cụ: **đạo hàm tự động** của PyTorch, **đạo hàm ký hiệu** của SymPy, và các hàm kiểm tra gradient của SciPy. Mỗi công cụ được dùng để kiểm tra lại các công thức của [chương 2](math/ch02_vi).

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `scipy`, `sympy`, `torch`.

| Việc cần làm | Công cụ |
| --- | --- |
| Gradient của một hàm viết bằng code | `torch.func.grad`, hoặc `loss.backward()` |
| Ma trận Jacobian, Hessian | `torch.func.jacrev`, `torch.func.hessian` |
| Kiểm tra gradient tự viết | `torch.autograd.gradcheck`, `scipy.optimize.check_grad` |
| Công thức đạo hàm dạng ký hiệu | `sympy.diff`, `sympy.hessian` |

## 1. Đạo hàm tự động với PyTorch

<small class="viz-ref">Mục 2.4 · Bảng 2.1</small>

Đạo hàm tự động (automatic differentiation) không phải đạo hàm số (không có $\varepsilon$) và cũng không phải đạo hàm ký hiệu. Thư viện ghi lại chuỗi phép tính khi hàm chạy, rồi áp dụng quy tắc chuỗi cho từng phép tính cơ bản. Kết quả chính xác tới sai số làm tròn.

<!-- run -->
```python
import numpy as np
import torch
from torch.func import grad, jacrev, hessian

torch.manual_seed(0)
n = 4
A = torch.randn(n, n, dtype=torch.float64)
a, b = torch.randn(n, dtype=torch.float64), torch.randn(n, dtype=torch.float64)
x = torch.randn(n, dtype=torch.float64)

checks = [
    ("x^T A x  -> (A + A^T) x", lambda x: x @ A @ x,          lambda x: (A + A.T) @ x),
    ("a^T x x^T b -> (ab^T + ba^T) x", lambda x: (a @ x) * (x @ b), lambda x: (torch.outer(a, b) + torch.outer(b, a)) @ x),
    ("||x||_2  -> x / ||x||_2", lambda x: torch.linalg.norm(x), lambda x: x / torch.linalg.norm(x)),
]
for name, f, g in checks:
    print(f"{name:34s}: sai lệch lớn nhất {(grad(f)(x) - g(x)).abs().max().item():.1e}")

X = torch.randn(n, 3, dtype=torch.float64)
print(f"{'trace(X^T A X) -> (A + A^T) X':34s}: sai lệch lớn nhất {(grad(lambda X: torch.trace(X.T @ A @ X))(X) - (A + A.T) @ X).abs().max().item():.1e}")

# Hessian của x^T A x là A + A^T (ma trận hằng)
print("Hessian của x^T A x = A + A^T:", torch.allclose(hessian(lambda x: x @ A @ x)(x), A + A.T))
```

<!-- output -->
```output
x^T A x  -> (A + A^T) x           : sai lệch lớn nhất 2.2e-16
a^T x x^T b -> (ab^T + ba^T) x    : sai lệch lớn nhất 4.4e-16
||x||_2  -> x / ||x||_2           : sai lệch lớn nhất 0.0e+00
trace(X^T A X) -> (A + A^T) X     : sai lệch lớn nhất 4.4e-16
Hessian của x^T A x = A + A^T: True
```
<!-- /output -->

Hàm $\|\mathbf{x}\|_2$ (không bình phương) không có trong Bảng 2.1. Gradient của nó là $\mathbf{x} / \|\mathbf{x}\|_2$, suy ra từ quy tắc chuỗi với $\|\mathbf{x}\|_2 = \sqrt{\|\mathbf{x}\|_2^2}$.

## 2. Jacobian và quy ước kích thước

<small class="viz-ref">Mục 2.2</small>

Cần chú ý quy ước. Sách viết gradient của $\mathbf{h}: \mathbb{R}^k \to \mathbb{R}^n$ thành ma trận $k \times n$. Các thư viện trả về **ma trận Jacobian** cỡ $n \times k$ (hàng $j$ là gradient của $h_j$), tức chuyển vị của ma trận trong sách.

<!-- run -->
```python
W = torch.randn(3, 4, dtype=torch.float64)
h = lambda x: torch.tanh(W @ x)
J = jacrev(h)(x)
book = W.T * (1 - torch.tanh(W @ x) ** 2)          # ma trận k x n theo cách viết của sách
print("Jacobian của PyTorch:", tuple(J.shape), "; ma trận trong sách:", tuple(book.shape))
print("J = (ma trận trong sách)^T:", torch.allclose(J, book.T))
```

<!-- output -->
```output
Jacobian của PyTorch: (3, 4) ; ma trận trong sách: (4, 3)
J = (ma trận trong sách)^T: True
```
<!-- /output -->

## 3. `backward()`: cách dùng trong huấn luyện

Khi huấn luyện mô hình, ta thường gọi `loss.backward()`. PyTorch tính gradient của một số vô hướng theo *mọi* tham số có `requires_grad=True`, và cộng dồn vào thuộc tính `.grad`. Ví dụ với hàm mất mát của hồi quy tuyến tính, $\mathcal{L}(\mathbf{w}) = \frac{1}{2N}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$ ([chương 7](ml/ch07_vi)):

<!-- run -->
```python
N, d = 50, 3
Xd, y = torch.randn(N, d, dtype=torch.float64), torch.randn(N, dtype=torch.float64)
w = torch.randn(d, dtype=torch.float64, requires_grad=True)

loss = ((Xd @ w - y) ** 2).sum() / (2 * N)
loss.backward()
formula = Xd.T @ (Xd @ w.detach() - y) / N           # từ Bảng 2.1: gradient của ||Ax - b||^2 là 2A^T(Ax - b)
print("w.grad khớp công thức:", torch.allclose(w.grad, formula))

# .grad được CỘNG DỒN: gọi backward lần hai mà không xoá thì gradient gấp đôi
loss = ((Xd @ w - y) ** 2).sum() / (2 * N); loss.backward()
print("sau hai lần backward, w.grad = 2 x công thức:", torch.allclose(w.grad, 2 * formula))
```

<!-- output -->
```output
w.grad khớp công thức: True
sau hai lần backward, w.grad = 2 x công thức: True
```
<!-- /output -->

Đây là lý do mọi vòng huấn luyện PyTorch đều có `optimizer.zero_grad()` trước `loss.backward()`.

## 4. Kiểm tra gradient tự viết

<small class="viz-ref">Mục 2.6</small>

`torch.autograd.gradcheck` so gradient của một phép tính tự định nghĩa với đạo hàm số hai phía. `scipy.optimize.check_grad` làm việc tương tự cho hàm NumPy, nhưng dùng xấp xỉ **một phía** nên kém chính xác hơn (xem [trang Code](math/ch02_code_vi), mục 5).

<!-- run -->
```python
from scipy.optimize import check_grad, approx_fprime

class Softplus(torch.autograd.Function):
    """softplus(x) = log(1 + e^x), viết tay cả chiều xuôi lẫn chiều ngược."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.log1p(torch.exp(x))
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.sigmoid(x)                      # đạo hàm đúng

class SoftplusBug(Softplus):
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.exp(x)                          # đạo hàm sai

xt = torch.randn(5, dtype=torch.float64, requires_grad=True)
print("gradcheck, đạo hàm đúng:", torch.autograd.gradcheck(Softplus.apply, (xt,)))
try:
    torch.autograd.gradcheck(SoftplusBug.apply, (xt,))
except Exception as e:
    print("gradcheck, đạo hàm sai: ", str(e).splitlines()[0])

An = A.numpy()
f_np = lambda x: x @ An @ x
g_np = lambda x: (An + An.T) @ x
x0 = np.random.default_rng(0).standard_normal(n)
print(f"\nscipy check_grad (một phía): {check_grad(f_np, g_np, x0):.1e}")
print(f"xấp xỉ hai phía tự viết:      {np.linalg.norm([(f_np(x0 + 1e-6 * e) - f_np(x0 - 1e-6 * e)) / 2e-6 for e in np.eye(n)] - g_np(x0)):.1e}")
```

<!-- output -->
```output
gradcheck, đạo hàm đúng: True
gradcheck, đạo hàm sai:  Jacobian mismatch for output 0 with respect to input 0,

scipy check_grad (một phía): 4.1e-08
xấp xỉ hai phía tự viết:      5.3e-11
```
<!-- /output -->

## 5. Đạo hàm ký hiệu với SymPy

Khi cần một *công thức* (để viết vào báo cáo, hoặc để tự kiểm tra một bước biến đổi), SymPy cho đạo hàm dạng ký hiệu. Ví dụ hàm mất mát logistic của một điểm dữ liệu ([chương 14](ml/ch14_vi)):

<!-- run -->
```python
import sympy as sp

w1, w2, x1, x2, yy = sp.symbols("w1 w2 x1 x2 y", real=True)
z = w1 * x1 + w2 * x2
sigma = 1 / (1 + sp.exp(-z))
L = -(yy * sp.log(sigma) + (1 - yy) * sp.log(1 - sigma))

dL = sp.simplify(sp.diff(L, w1))
print("dL/dw1 =", dL)
print("bằng (sigma(z) - y) x1 ?", sp.simplify(dL - (sigma - yy) * x1) == 0)
H = sp.hessian(L, (w1, w2))
xx = sp.Matrix([x1, x2])
print("Hessian = sigma(z)(1 - sigma(z)) x x^T ?", sp.simplify(H - sigma * (1 - sigma) * xx * xx.T) == sp.zeros(2, 2))
print("Hessian không phụ thuộc y:", all(sp.simplify(sp.diff(h_, yy)) == 0 for h_ in H))
```

<!-- output -->
```output
dL/dw1 = x1*(-y*exp(w1*x1 + w2*x2) - y + exp(w1*x1 + w2*x2))/(exp(w1*x1 + w2*x2) + 1)
bằng (sigma(z) - y) x1 ? True
Hessian = sigma(z)(1 - sigma(z)) x x^T ? True
Hessian không phụ thuộc y: True
```
<!-- /output -->

SymPy in ra một biểu thức rối, nhưng khi so với dạng gọn $(\sigma(z) - y)\mathbf{x}$ (công thức (14.15) trong sách) thì hiệu bằng 0. Hessian là $\sigma(z)(1 - \sigma(z))\mathbf{x}\mathbf{x}^T$ và không phụ thuộc vào nhãn $y$. Hai kết quả này được dùng ở [trang Ví dụ áp dụng](math/ch02_app_vi) để huấn luyện hồi quy logistic bằng phương pháp Newton.

## Tóm tắt

- Đạo hàm tự động là công cụ chính của học sâu: chính xác, nhanh, và chỉ cần viết hàm mất mát.
- Nắm quy ước kích thước: sách viết gradient của hàm trả về vector là ma trận $k \times n$, còn thư viện trả Jacobian $n \times k$.
- Khi tự viết chiều ngược cho một phép tính mới, luôn kiểm tra bằng `gradcheck` hoặc đạo hàm số hai phía.
