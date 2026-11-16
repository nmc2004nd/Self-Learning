# Chương 2 Giải tích ma trận

> Ghi chú của người biên soạn: chương này lấy nguyên văn từ sách *Machine Learning cơ bản* của **Vũ Hữu Tiệp** ([mã nguồn LaTeX](https://github.com/tiepvupsu/ebookMLCB), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); trang gốc [machinelearningcoban.com](https://machinelearningcoban.com)). Người biên soạn chỉ đổi định dạng sang trang web: số chương, mục, công thức và hình giữ như bản PDF, hình vector được vẽ lại dưới dạng SVG.
>
> **Đính chính** *(người biên soạn)*:
>
> - Code ở mục 2.6.2: với NumPy mới (từ bản 2.x), `fn2` trả về mảng cỡ $1 \times 1$ nên dòng `grad_flat[i] = (fn(Xp) - fn(Xn))/(2*eps)` báo lỗi `ValueError: setting an array element with a sequence`. Sửa bằng cách cho `fn2` trả về một số: `return x.T.dot(A).dot(x).item()`. Bản chạy được nằm ở trang [Code từ đầu](math/ch02_code_vi).

Giả sử rằng các gradient tồn tại trong toàn bộ chương. Tài liệu tham khảo chính của chương là *Matrix calculus – Stanford* (<https://goo.gl/BjTPLr>).

## 2.1 Gradient của hàm trả về một số vô hướng

*Gradient bậc nhất* (first-order gradient) hay viết gọn là gradient của một hàm số $f(\mathbf{x}): \mathbb{R}^n \rightarrow \mathbb{R}$ theo $\mathbf{x}$, ký hiệu là $\nabla_{\mathbf{x}} f(\mathbf{x})$, được định nghĩa bởi

$$
\nabla_{\mathbf{x}} f(\mathbf{x}) \triangleq
\left[
\begin{matrix}
\dfrac{\partial f(\mathbf{x})}{\partial x_1} \\ 
\dfrac{\partial f(\mathbf{x})}{\partial x_2} \\ 
\vdots \\ 
\dfrac{\partial f(\mathbf{x})}{\partial x_n}
\end{matrix}
\right] \in \mathbb{R}^n, \tag{2.1}
$$

trong đó $\displaystyle\dfrac{\partial f(\mathbf{x})}{\partial x_i}$ là *đạo hàm riêng* của hàm số theo thành phần thứ $i$ của vector $\mathbf{x}$. Đạo hàm này được tính khi tất cả các biến, ngoài $x_i$, được giả sử là hằng số. Nếu không có thêm biến nào khác, $\nabla_{\mathbf{x}}f(\mathbf{x})$ thường được viết gọn là $\nabla f(\mathbf{x})$. Gradient của hàm số này là một vector có cùng chiều với vector đang được lấy gradient. Tức nếu vector được viết ở dạng cột thì gradient cũng phải được viết ở dạng cột.

*Gradient bậc hai* (second-order gradient) của hàm số trên còn được gọi là *Hesse* (Hessian) và được định nghĩa như sau:

$$
\nabla^2 f(\mathbf{x}) \triangleq
\left[\begin{matrix}
\dfrac{\partial^2 f(\mathbf{x})}{\partial x_1^2} & \dfrac{\partial^2 f(\mathbf{x})}{\partial x_1\partial x_2} & \dots & \dfrac{\partial^2 f(\mathbf{x})}{\partial x_1\partial x_n} \\ 
\dfrac{\partial^2 f(\mathbf{x})}{\partial x_2\partial x_1} & \dfrac{\partial^2 f(\mathbf{x})}{\partial x_2^2} & \dots & \dfrac{\partial^2 f(\mathbf{x})}{\partial x_2\partial x_n} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\dfrac{\partial^2 f(\mathbf{x})}{\partial x_n\partial x_1} & \dfrac{\partial^2 f(\mathbf{x})}{\partial x_n\partial x_2} & \dots & \dfrac{\partial^2 f(\mathbf{x})}{\partial x_n^2}
\end{matrix}\right] \in \mathbb{S}^{n}. \tag{2.2}
$$

Gradient của một hàm số $f(\mathbf{X}): \mathbb{R}^{n \times m} \rightarrow \mathbb{R}$ theo ma trận $\mathbf{X}$ được định nghĩa là

$$
\nabla f(\mathbf{X}) =
\left[\begin{matrix}
\dfrac{\partial f(\mathbf{X})}{\partial x_{11}} & \dfrac{\partial f(\mathbf{X})}{\partial x_{12}} & \dots & \dfrac{\partial f(\mathbf{X})}{\partial x_{1m}} \\ 
\dfrac{\partial f(\mathbf{X})}{\partial x_{21}} & \dfrac{\partial f(\mathbf{X})}{\partial x_{22}} & \dots & \dfrac{\partial f(\mathbf{X})}{\partial x_{2m}} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\dfrac{\partial f(\mathbf{X})}{\partial x_{n1}} & \dfrac{\partial f(\mathbf{X})}{\partial x_{n2}} & \dots & \dfrac{\partial f(\mathbf{X})}{\partial x_{nm}}
\end{matrix}\right] \in \mathbb{R}^{n \times m}. \tag{2.3}
$$

> Gradient của hàm số $f: \mathbb{R}^{m\times n} \rightarrow \mathbb{R}$ là một ma trận trong $\mathbb{R}^{m\times n}.$

Cụ thể, để tính gradient của một hàm $f: \mathbb{R}^{m\times n} \rightarrow \mathbb{R}$, ta tính đạo hàm riêng của hàm số đó theo từng thành phần của ma trận khi toàn bộ các thành phần khác được giả sử là hằng số. Tiếp theo, ta sắp xếp các đạo hàm riêng tính được theo đúng thứ tự trong ma trận.

*Ví dụ*: Xét hàm số $f: \mathbb{R}^2 \rightarrow \mathbb{R}$, $f(\mathbf{x}) = x_1 ^2 + 2x_1x_2 + \sin(x_1) + 2$. Gradient bậc nhất theo $\mathbf{x}$ của hàm số đó là

$$
\nabla f(\mathbf{x}) =
\left[\begin{matrix}
\dfrac{\partial f(\mathbf{x})}{\partial x_1} \\ 
\dfrac{\partial f(\mathbf{x})}{\partial x_2}
\end{matrix}\right] = \left[\begin{matrix}
2x_1 + 2x_2 + \cos(x_1) \\ 
2x_1
\end{matrix}\right].
$$

Gradient bậc hai theo $\mathbf{x}$, hay Hesse là
$$
\nabla^2 f(\mathbf{x}) =
\left[\begin{matrix}
\dfrac{\partial^2 f(\mathbf{x})}{\partial x_1^2} & \dfrac{\partial f^2(\mathbf{x})}{\partial x_1\partial x_2} \\
\dfrac{\partial^2 f(\mathbf{x})}{\partial x_2\partial x_1} & \dfrac{\partial f^2(\mathbf{x})}{\partial x_2^2}
\end{matrix}\right] =
\left[\begin{matrix}
2 - \sin(x_1) & 2 \\
2 & 0
\end{matrix}\right].
$$

Chú ý rằng Hesse luôn là một ma trận đối xứng.

## 2.2 Gradient của hàm trả về vector 

Những hàm số mà đầu ra là một vector được gọi là *hàm trả về vector* (vector-valued function).

Xét một hàm trả về vector với đầu vào là một số thực $v(x): \mathbb{R} \rightarrow \mathbb{R}^n$:

$$
v(x) =
\left[\begin{matrix}
v_1(x) \\ 
v_2(x) \\ 
\vdots \\ 
v_n(x)
\end{matrix}\right]. \tag{2.4}
$$

Gradient của hàm số này theo $x$ là một vector hàng như sau:

$$
\nabla v(x) \triangleq
\left[\begin{matrix}
\dfrac{\partial v_1(x)}{\partial x} & \dfrac{\partial v_2(x)}{\partial x} &
\dots & \dfrac{\partial v_n(x)}{\partial x}
\end{matrix}\right]. \tag{2.5}
$$

Gradient bậc hai của hàm số này có dạng:

$$
\nabla^2 v(x) \triangleq
\left[
\begin{matrix}
\dfrac{\partial^2 v_1(x)}{\partial x^2} & \dfrac{\partial^2 v_2(x)}{\partial x^2} & \dots & \dfrac{\partial^2 v_n(x)}{\partial x^2}
\end{matrix}
\right]. \tag{2.6}
$$

**Ví dụ**: Cho một vector $\mathbf{a} \in \mathbb{R}^n$ và một hàm số trả về vector $v(x) = x\mathbf{a}$, gradient và Hesse của nó lần lượt là

$$
\nabla v(x) = \mathbf{a}^T, ~~~ \nabla^2 v(x) = \mathbf{0} \in \mathbb{R}^{1\times n}. \tag{2.7}
$$

Xét một hàm trả về vector với đầu vào là một vector $h(\mathbf{x}):\mathbb{R}^k \rightarrow \mathbb{R}^n$, gradient của nó là

$$
\begin{align*}
\nabla h(\mathbf{x}) &\triangleq 
\left[\begin{matrix}
\dfrac{\partial h_1(\mathbf{x})}{\partial x_1} & \dfrac{\partial h_2(\mathbf{x})}{\partial x_1} & \dots & \dfrac{\partial h_n(\mathbf{x})}{\partial x_1} \\ 
\dfrac{\partial h_1(\mathbf{x})}{\partial x_2} & \dfrac{\partial h_2(\mathbf{x})}{\partial x_2} & \dots & \dfrac{\partial h_n(\mathbf{x})}{\partial x_2} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\dfrac{\partial h_1(\mathbf{x})}{\partial x_k} & \dfrac{\partial h_2(\mathbf{x})}{\partial x_k} & \dots & \dfrac{\partial h_n(\mathbf{x})}{\partial x_k}
\end{matrix}\right] =
\left[\begin{matrix}
\nabla h_1(\mathbf{x}) & \dots & \nabla h_n(\mathbf{x})
\end{matrix}\right] \in \mathbb{R}^{k\times n}. \tag{2.8}
\end{align*}
$$

> Gradient của hàm số $g: \mathbb{R}^m \rightarrow \mathbb{R}^n$ là một ma trận thuộc $\mathbb{R}^{m \times n}$.

Gradient bậc hai của hàm số trên là một mảng ba chiều. Trong phạm vi của cuốn sách, chúng ta sẽ không xét gradient bậc hai của các hàm số $g: \mathbb{R}^m \rightarrow \mathbb{R}^n$.

Trước khi đến phần tính gradient của các hàm số thường gặp, chúng ta cần biết hai tính chất quan trọng khá giống với gradient của hàm một biến.

## 2.3 Tính chất quan trọng của gradient

### 2.3.1 Quy tắc tích

Giả sử các biến đầu vào là một ma trận và các hàm số có chiều phù hợp để phép nhân ma trận thực hiện được. Ta có

$$
\nabla\left( f(\mathbf{X})^Tg(\mathbf{X}) \right) = \left(\nabla f(\mathbf{X})\right) g(\mathbf{X}) +
\left(\nabla g(\mathbf{X})\right) f(\mathbf{X}). \tag{2.9}
$$

Quy tắc này tương tự như quy tắc tính đạo hàm của tích các hàm $f, g: \mathbb{R} \rightarrow \mathbb{R}$:

$$
\left(f(x)g(x)\right)' = f'(x)g(x) + g'(x)f(x).
$$

Lưu ý rằng tính chất giao hoán không còn đúng với vector và ma trận, vì vậy nhìn chung

$$
\nabla\left( f(\mathbf{X})^Tg(\mathbf{X}) \right) \neq  g(\mathbf{X})\left(\nabla f(\mathbf{X})\right) +
f(\mathbf{X})\left(\nabla g(\mathbf{X})\right). \tag{2.10}
$$

Biểu thức bên phải có thể không xác định khi chiều của các ma trận lệch nhau.

### 2.3.2 Quy tắc chuỗi

Quy tắc chuỗi được áp dụng khi tính gradient của các hàm hợp:

$$
\nabla_{\mathbf{X}} g(f(\mathbf{X})) = (\nabla_{\mathbf{X}} f) (\nabla_{f}g). \tag{2.11}
$$

Quy tắc này cũng giống với quy tắc trong hàm một biến:

$$
(g(f(x)))' = f'(x)g'(f).
$$

Một lưu ý nhỏ nhưng quan trọng khi làm việc với tích các ma trận là sự phù hợp về kích thước của các ma trận trong tích.

## 2.4 Gradient của các hàm số thường gặp 

### 2.4.1 $f(\mathbf{x}) = \mathbf{a}^T\mathbf{x}$

Giả sử $\mathbf{a}, \mathbf{x} \in \mathbb{R}^n$, ta viết lại $f(\mathbf{x}) = \mathbf{a}^T\mathbf{x} = a_1 x_1 + a_2 x_2 + \dots + a_nx_n$.

Nhận thấy $\dfrac{\partial f(\mathbf{x})}{\partial x_i} = a_i, ~ \forall i = 1, 2\dots, n$.

Vậy, $\nabla_{\mathbf{x}} (\mathbf{a}^T\mathbf{x}) = \left[\begin{matrix} a_1 & a_2 & \dots & a_n \end{matrix}\right]^T = \mathbf{a}$.

Ngoài ra, vì $\mathbf{a}^T\mathbf{x} = \mathbf{x}^T\mathbf{a}$ nên $\nabla_{\mathbf{x}} (\mathbf{x}^T\mathbf{a}) = \mathbf{a}$.

### 2.4.2 $f(\mathbf{x}) = \mathbf{Ax}$

Đây là một hàm trả về vector $f: \mathbb{R}^n \rightarrow \mathbb{R}^{m}$ với $\mathbf{x} \in \mathbb{R}^n, \mathbf{A} \in \mathbb{R}^{m\times n}$. Giả sử $\mathbf{a}_i$ là hàng thứ $i$ của ma trận $\mathbf{A}$. Ta có

$$
\mathbf{Ax}  =
\left[\begin{matrix}
\mathbf{a}_1\mathbf{x} \\ 
\mathbf{a}_2\mathbf{x} \\ 
\vdots\\ 
\mathbf{a}_m\mathbf{x}
\end{matrix}\right].
$$

Từ định nghĩa (2.8) và công thức gradient của $\mathbf{a}_i\mathbf{x}$, có thể suy ra

$$
\nabla_{\mathbf{x}} (\mathbf{Ax}) =
\left[\begin{matrix}
\mathbf{a}_1^T & \mathbf{a}_2^T & \dots & \mathbf{a}_m^T
\end{matrix}\right] = \mathbf{A}^T \tag{2.12}
$$

Từ đây suy ra đạo hàm của hàm số $f(\mathbf{x}) = \mathbf{x} = \mathbf{Ix}$ là

$$
\nabla \mathbf{x} = \mathbf{I}
$$

với $\mathbf{I}$ là ma trận đơn vị.

### 2.4.3 $f(\mathbf{x}) = \mathbf{x}^T\mathbf{A} \mathbf{x}$

Với $\mathbf{x} \in \mathbb{R}^n, \mathbf{A} \in \mathbb{R}^{n\times n}$, áp dụng quy tắc tích (2.9) ta có

$$
\begin{align*}
\nabla f(\mathbf{x}) &= \nabla \left(\left(\mathbf{x}^T\right) \left(\mathbf{Ax}\right)\right) \\
&= \left(\nabla (\mathbf{x})\right) \mathbf{Ax} + \left(\nabla (\mathbf{Ax})\right)\mathbf{x} \\
& =  \mathbf{IAx} + \mathbf{A}^T\mathbf{x} \\
& =  (\mathbf{A} + \mathbf{A}^T)\mathbf{x}. \tag{2.13}
\end{align*}
$$

Từ (2.13) và (2.12), có thể suy ra $\nabla^2 \mathbf{x}^T\mathbf{Ax} = \mathbf{A}^T + \mathbf{A}.$ Nếu $\mathbf{A}$ là một ma trận đối xứng, ta có $\nabla \mathbf{x}^T\mathbf{A}\mathbf{x} = 2\mathbf{Ax}, \nabla^2 \mathbf{x}^T\mathbf{Ax} = 2\mathbf{A}.$

Nếu $\mathbf{A}$ là ma trận đơn vị, tức $f(\mathbf{x}) = \mathbf{x}^T\mathbf{Ix} = \mathbf{x}^T\mathbf{x} = \|\mathbf{x}\|_2^2$, ta có

$$
\nabla \|\mathbf{x}\|_2^2 = 2\mathbf{x},~~~
\nabla^2 \|\mathbf{x}\|_2^2 = 2\mathbf{I}. \tag{2.14}
$$

### 2.4.4 $f(\mathbf{x}) = \|\mathbf{Ax} - \mathbf{b}\|_2^2$

Có hai cách tính gradient của hàm số này:

- Cách 1: Trước hết, khai triển:

  $$
  \begin{align*}
  f(\mathbf{x}) &= \|\mathbf{Ax} - \mathbf{b}\|_2^2 = (\mathbf{Ax} - \mathbf{b})^T(\mathbf{Ax} - \mathbf{b}) = (\mathbf{x}^T\mathbf{A}^T - \mathbf{b}^T) (\mathbf{Ax} - \mathbf{b}) \\
  &= \mathbf{x}^T\mathbf{A}^T\mathbf{Ax} - 2 \mathbf{b}^T\mathbf{Ax} + \mathbf{b}^T\mathbf{b}.
  \end{align*}
  $$

  Lấy gradient cho từng số hạng rồi cộng lại ta có

  $$
  \nabla \|\mathbf{Ax} - \mathbf{b}\|_2^2 = 2\mathbf{A}^T\mathbf{A}\mathbf{x} - 2\mathbf{A}^T\mathbf{b} = 2\mathbf{A}^T(\mathbf{Ax} - \mathbf{b}).
  $$

- Cách 2: Sử dụng $\nabla (\mathbf{Ax} - \mathbf{b}) = \mathbf{A}^T$ và $\nabla \|\mathbf{x}\|_2^2 = 2\mathbf{x}$ và quy tắc chuỗi (2.11), ta cũng sẽ thu được kết quả tương tự.

### 2.4.5 $f(\mathbf{x}) = \mathbf{a}^T\mathbf{x}\mathbf{x}^T\mathbf{b}$

Viết lại $f(\mathbf{x}) = (\mathbf{a}^T\mathbf{x})(\mathbf{x}^T\mathbf{b})$ và dùng quy tắc tích (2.9), ta có

$$
\nabla (\mathbf{a}^T\mathbf{x}\mathbf{x}^T\mathbf{b}) = \mathbf{a} \mathbf{x}^T\mathbf{b} +
\mathbf{b}\mathbf{a}^T\mathbf{x} = \mathbf{ab}^T\mathbf{x} + \mathbf{b}\mathbf{a}^T\mathbf{x}=
(\mathbf{ab}^T + \mathbf{ba}^T)\mathbf{x},
$$

ở đây ta đã sử dụng tính chất $\mathbf{y}^T\mathbf{z} = \mathbf{z}^T\mathbf{y}$.

### 2.4.6 $f(\mathbf{X}) = \text{trace}(\mathbf{A}\mathbf{X})$

Giả sử $\mathbf{A} \in \mathbb{R}^{n\times m}, \mathbf{X} = \mathbb{R}^{m\times n}$, và $\mathbf{B} = \mathbf{A}\mathbf{X} \in \mathbb{R}^ {n \times n}$. Theo định nghĩa của trace:

$$
f(\mathbf{X}) = \text{trace}(\mathbf{A}\mathbf{X}) = \text{trace}(\mathbf{B}) = \sum_{j = 1}^n {b_{jj}} =
\sum_{j=1}^n\sum_{i=1}^n a_{ji}x_{ji}. \tag{2.15}
$$

Từ đó suy ra $\displaystyle\frac{\partial f(\mathbf{X})}{\partial x_{ij}} = a_{ji}$. Theo định nghĩa (2.3), ta có $\nabla_{\mathbf{X}}\text{trace}(\mathbf{A}\mathbf{X}) = \mathbf{A}^T$.

### 2.4.7 $f(\mathbf{X}) = \mathbf{a}^T\mathbf{X}\mathbf{b}$

Giả sử rằng $\mathbf{a} \in \mathbb{R}^{m}, \mathbf{X} \in R^{m\times n}, \mathbf{b} \in \mathbb{R}^{n}$. Ta có thể chứng minh được
$$
f(\mathbf{X}) = \sum_{i=1}^m \sum_{j=1}^n x_{ij}a_ib_j.
$$

Từ đó, sử dụng định nghĩa (2.3), ta đạt được

$$
\nabla_{\mathbf{X}}(\mathbf{a}^T\mathbf{X}\mathbf{b}^T) = \left[\begin{matrix}
a_1b_1 &  a_1b_2 & \dots &a_1b_n \\
a_2b_1 &  a_2b_2 & \dots &a_2b_n \\
\dots & \dots & \ddots & \dots \\
a_mb_1 &  a_mb_2 & \dots &a_mb_n \\
\end{matrix}\right]
= \mathbf{a}\mathbf{b}^T. \tag{2.16}
$$

### 2.4.8 $f(\mathbf{X}) = \|\mathbf{X}\|_F^2$

Giả sử $\mathbf{X} \in \mathbb{R}^{n\times n}$, ta có
$$
\|\mathbf{X}\|_F^2 =
\sum_{i=1}^n\sum_{j=1}^n x_{ij}^2 \Rightarrow \frac{\partial f}{\partial x_{ij}} = 2x_{ij} \Rightarrow \nabla \|\mathbf{X}\|_F^2 = 2\mathbf{X}.
$$

### 2.4.9 $f(\mathbf{X}) = \text{trace}(\mathbf{X}^T\mathbf{A}\mathbf{X})$

Giả sử rằng $\mathbf{X} = \left[\begin{matrix} \mathbf{x}_1 & \mathbf{x}_2 & \dots & \mathbf{x}_n \end{matrix}\right] \in \mathbb{R}^{m\times n}, \mathbf{A} \in \mathbb{R}^{m\times m}$. Bằng cách khai triển

$$
\mathbf{X}^T\mathbf{A}\mathbf{X} = \left[\begin{matrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_n^T \end{matrix}\right] \mathbf{A} \left[\begin{matrix}
\mathbf{x}_1 & \mathbf{x}_2 & \dots, \mathbf{x}_n \end{matrix}\right] = \left[\begin{matrix}
\mathbf{x}_1^T\mathbf{A}\mathbf{x}_1 & \mathbf{x}_1^T\mathbf{A}\mathbf{x}_2 & \dots & \mathbf{x}_1^T\mathbf{A}\mathbf{x}_n \\
\mathbf{x}_2^T\mathbf{A}\mathbf{x}_1 & \mathbf{x}_2^T\mathbf{A}\mathbf{x}_2 & \dots & \mathbf{x}_2^T\mathbf{A}\mathbf{x}_n \\
\dots & \dots & \ddots & \dots \\
\mathbf{x}_n^T\mathbf{A}\mathbf{x}_1 & \mathbf{x}_n^T\mathbf{A}\mathbf{x}_2 & \dots & \mathbf{x}_n^T\mathbf{A}\mathbf{x}_n
\end{matrix}\right],
$$

ta tính được $\text{trace}(\mathbf{X}^T\mathbf{A}\mathbf{X}) = \sum_{i=1}^n \mathbf{x}_i^T\mathbf{A}\mathbf{x}_i$.

Sử dụng công thức $\nabla_{\mathbf{x}_i}\mathbf{x}_i^T\mathbf{A}\mathbf{x}_i = (\mathbf{A} + \mathbf{A}^T)\mathbf{x}_i$, ta có

$$
\nabla_{\mathbf{X}} \text{trace}(\mathbf{X}^T\mathbf{A}\mathbf{X}) = (\mathbf{A} + \mathbf{A}^T) \left[\begin{matrix} \mathbf{x}_1 & \mathbf{x}_2 & \dots &
\mathbf{x}_n \end{matrix}\right] = (\mathbf{A} + \mathbf{A}^T)\mathbf{X}. \tag{2.17}
$$

Bằng cách thay $\mathbf{A} = \mathbf{I}$, ta cũng thu được $\nabla_{\mathbf{X}}\text{trace}(\mathbf{X}^T\mathbf{X}) = \nabla_{\mathbf{X}} \|\mathbf{X}\|_F^2 = 2\mathbf{X}$.

### 2.4.10 $f(\mathbf{X}) = \|\mathbf{A}\mathbf{X} - \mathbf{B}\|_F^2$

Bằng kỹ thuật hoàn toàn tương tự như đã làm trong Mục 2.4.4, ta thu được
$$
\nabla_{\mathbf{X}} \|\mathbf{A}\mathbf{X} - \mathbf{B}\|_F^2 = 2\mathbf{A}^T(\mathbf{A}\mathbf{X} - \mathbf{B}).
$$

## 2.5 Bảng các gradient thường gặp

Bảng 2.1 bao gồm gradient của các hàm số thường gặp với biến là vector hoặc ma trận.

<div align=center>Bảng 2.1: Bảng các gradient cơ bản.</div>

| $f(\mathbf{x})$                                  | $\nabla f(\mathbf{x})$                                         | $f(\mathbf{X})$                         | $\nabla_{\mathbf{X}} f(\mathbf{X})$  |
|:------------------------------------------|:--------------------------------------------------------|:---------------------------------|:-----------------------|
| $\mathbf{x}$                                     | $\mathbf{I}$                                                   | $\text{trace}(\mathbf{X})$                    | $\mathbf{I}$                  |
| $\mathbf{a}^T \mathbf{x}$                        | $\mathbf{a}$                                            | $\text{trace}(\mathbf{A}^T\mathbf{X})$               | $\mathbf{A}$                  |
| $\mathbf{x}^T\mathbf{Ax}$                        | $(\mathbf{A} + \mathbf{A}^T) \mathbf{x}$                       | $\text{trace}(\mathbf{X}^T\mathbf{A}\mathbf{X})$            | $(\mathbf{A} + \mathbf{A}^T) \mathbf{X}$    |
| $\mathbf{x}^T \mathbf{x} = \|\mathbf{x}\|_2^2$                 | $2\mathbf{x}$                                                  | $\text{trace}(\mathbf{X}^T\mathbf{X}) = \|\mathbf{X}\|_F^2$ | $2\mathbf{X}$                 |
| $\|\mathbf{Ax-b} \|_2^2$                  | $2\mathbf{A}^T (\mathbf{Ax - b})$                       | $\|\mathbf{A}\mathbf{X} - \mathbf{B} \|_F^2$          | $2\mathbf{A}^T(\mathbf{A}\mathbf{X} - \mathbf{B})$ |
| $\mathbf{a}^T(\mathbf{x}^T\mathbf{x})\mathbf{b}$ | $2\mathbf{a}^T\mathbf{bx}$                              | $\mathbf{a}^T\mathbf{X}\mathbf{b}$                    | $\mathbf{a}\mathbf{b}^T$             |
| $\mathbf{a}^T\mathbf{x}\mathbf{x}^T\mathbf{b}$          | $(\mathbf{a}\mathbf{b}^T + \mathbf{b}\mathbf{a}^T) \mathbf{x}$ | $\text{trace}(\mathbf{A}^T\mathbf{X}\mathbf{B})$            | $\mathbf{A}\mathbf{B}^T$             |

## 2.6 Kiểm tra gradient

Việc tính gradient của hàm nhiều biến thông thường khá phức tạp và rất dễ mắc lỗi. Trong thực nghiệm, có một cách để kiểm tra liệu gradient tính được có chính xác không. Cách này dựa trên định nghĩa của đạo hàm cho hàm một biến.

### 2.6.1 Xấp xỉ đạo hàm của hàm một biến

Xét cách tính đạo hàm của hàm một biến theo định nghĩa:

$$
f'(x) = \lim_{\varepsilon \rightarrow 0}\frac{f(x + \varepsilon) -
f(x)}{\varepsilon}. \tag{2.18}
$$

Trên máy tính, ta có thể chọn $\varepsilon$ rất nhỏ, ví dụ $10^{-6}$, rồi xấp xỉ đạo hàm này bởi

$$
f'(x) \approx \lim_{\varepsilon \rightarrow 0}\frac{f(x + \varepsilon) -
f(x)}{\varepsilon}. \tag{2.19}
$$

Trên thực tế, công thức xấp xỉ đạo hàm hai phía thường được sử dụng:

$$
f'(x) \approx \frac{f(x + \varepsilon) - f(x - \varepsilon)}{2\varepsilon}. \tag{2.20}
$$

Cách tính này được gọi là *numerical gradient*. Có hai cách giải thích việc tại sao cách tính như (2.20) được sử dụng rộng rãi hơn:

\* *Bằng giải tích*

Sử dụng khai triển Taylor với $\varepsilon$ rất nhỏ, ta có hai xấp xỉ sau:

$$
\begin{align*}
f(x + \varepsilon) &\approx f(x) + f'(x)\varepsilon + \frac{f"(x)}{2}
\varepsilon^2 + \frac{f^{(3)}}{6}\varepsilon^3 + \dots \tag{2.21} \\
f(x - \varepsilon) &\approx f(x) - f'(x)\varepsilon + \frac{f"(x)}{2}
\varepsilon^2 - \frac{f^{(3)}}{6}\varepsilon^3 + \dots \tag{2.22}
\end{align*}
$$

Từ đó ta có:

$$
\begin{align*}
\frac{f(x + \varepsilon) - f(x)}{\varepsilon} &\approx f'(x) +
\frac{f"(x)}{2}\varepsilon + \dots =  f'(x) + O(\varepsilon). \tag{2.23} \\
\frac{f(x + \varepsilon) - f(x - \varepsilon)}{2\varepsilon} &\approx f'(x) +
\frac{f^{(3)}(x)}{6}\varepsilon^2 + \dots =  f'(x) + O(\varepsilon^2). \tag{2.24}
\end{align*}
$$

trong đó $O()$ là Big O notation.

Từ đó, nếu xấp xỉ đạo hàm bằng công thức (2.23), sai số sẽ là $O(\varepsilon)$. Trong khi đó, nếu xấp xỉ đạo hàm bằng công thức (2.24), sai số sẽ là $O(\varepsilon^2)$. Khi $\varepsilon$ rất nhỏ, $O(\varepsilon^2) \ll O(\varepsilon)$, tức cách đánh giá sử dụng công thức (2.24) có sai số nhỏ hơn, và vì vậy nó được sử dụng phổ biến hơn.

\* *Bằng hình học*

<div align=center>
<img width="540" src="../img/math/ch02/check_grad.svg"/>
</div>
<div align=center>Hình 2.1: Giải thích cách xấp xỉ đạo hàm bằng hình học</div>

Quan sát Hình 2.1, vector nét liền là đạo hàm chính xác của hàm số tại điểm có hoành độ bằng $x_0$. Hai vector nét đứt thể hiện xấp xỉ đạo hàm phía phải và phía trái. Vector chấm gạch thể hiện xấp xỉ đạo hàm hai phía. Trong ba vector xấp xỉ đó, vector chấm gạch gần với vector nét liền nhất nếu xét theo hướng.

Sự khác biệt giữa các phương pháp xấp xỉ còn lớn hơn nữa nếu tại điểm $x$, hàm số bị bẻ cong mạnh hơn. Khi đó, xấp xỉ trái và phải sẽ khác nhau rất nhiều. Xấp xỉ hai phía sẽ cho kết quả ổn định hơn.

### 2.6.2 Xấp xỉ gradient của hàm nhiều biến

Với hàm nhiều biến, công thức (2.24) được áp dụng cho từng biến khi các biến khác cố định. Cụ thể, ta sử dụng định nghĩa gradient của hàm số nhận đầu vào là một ma trận như công thức (2.3). Mỗi thành phần của ma trận kết quả là đạo hàm riêng của hàm số tại thành phần đó khi ta coi các thành phần còn lại cố định. Chúng ta sẽ thấy rõ điều này hơn ở cách lập trình so sánh hai cách tính gradient ngay sau đây.

Cách tính gradient xấp xỉ hai phía thường cho giá trị khá chính xác. Tuy nhiên, cách này không được sử dụng để tính gradient vì độ phức tạp quá cao so với cách tính trực tiếp. Tại mỗi thành phần, ta cần tính giá trị của hàm số tại phía trái và phía phải. Việc làm này không khả thi với các ma trận lớn. Khi so sánh đạo hàm xấp xỉ với gradient tính theo công thức, người ta thường giảm số chiều dữ liệu và giảm số điểm dữ liệu để thuận tiện cho tính toán. Nếu gradient tính được là chính xác, nó sẽ rất gần với gradient xấp xỉ này.

Đoạn code dưới đây giúp kiểm tra gradient của một hàm số khả vi $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$, có kèm theo hai ví dụ. Để sử dụng hàm kiểm tra `check_grad` này, ta cần viết hai hàm. Hàm thứ nhất là hàm `fn(X)` tính giá trị của hàm số tại `X`. Hàm thứ hai là hàm `gr(X)` tính giá trị của gradient của `fn(X)`.

```python
from __future__ import print_function
import numpy as np

def check_grad(fn, gr, X):
    X_flat    = X.reshape(-1) # convert X to an 1d array, 1 for loop needed
    shape_X   = X.shape                # original shape of X
    num_grad  = np.zeros_like(X)       # numerical grad, shape = shape of X
    grad_flat = np.zeros_like(X_flat)  # 1d version of grad
    eps       = 1e-6      # a small number, 1e-10 -> 1e-6 is usually good
    numElems  = X_flat.shape[0] # number of elements in X
    # calculate numerical gradient
    for i in range(numElems):          # iterate over all elements of X
        Xp_flat      = X_flat.copy()
        Xn_flat      = X_flat.copy()
        Xp_flat[i]  += eps
        Xn_flat[i]  -= eps
        Xp           = Xp_flat.reshape(shape_X)
        Xn           = Xn_flat.reshape(shape_X)
        grad_flat[i] = (fn(Xp) - fn(Xn))/(2*eps)

    num_grad = grad_flat.reshape(shape_X)

    diff = np.linalg.norm(num_grad - gr(X))
    print('Difference between two methods should be small:', diff)

# ==== check if grad(trace(A*X)) == A^T ====
m, n = 10, 20
A = np.random.rand(m, n)
X = np.random.rand(n, m)

def fn1(X):
    return np.trace(A.dot(X))

def gr1(X):
    return A.T

check_grad(fn1, gr1, X)
# ==== check if grad(x^T*A*x) == (A + A^T)*x  ====
A = np.random.rand(m, m)
x = np.random.rand(m, 1)

def fn2(x):
    return x.T.dot(A).dot(x)

def gr2(x):
    return (A + A.T).dot(x)

check_grad(fn2, gr2, x)
```

*Kết quả:*

```text
Difference between two methods should be small: 2.02303323394e-08
Difference between two methods should be small: 2.10853872281e-09
```

Kết quả cho thấy sự khác nhau giữa Frobenious norm (norm mặc định trong `np.linalg.norm`) trong kết quả của hai cách tính là rất nhỏ. Sau khi chạy lại đoạn code với các giá trị `m, n` khác nhau và biến `X` khác nhau, nếu sự khác nhau vẫn là nhỏ, ta có thể kết luận rằng gradient mà ta tính được là chính xác.

Bạn đọc có thể kiểm tra lại các công thức trong Bảng 2.1 bằng phương pháp này.
