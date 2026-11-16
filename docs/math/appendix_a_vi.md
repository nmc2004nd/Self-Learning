# Phụ lục A Phương pháp nhân tử Lagrange

> Ghi chú của người biên soạn: chương này lấy nguyên văn từ sách *Machine Learning cơ bản* của **Vũ Hữu Tiệp** ([mã nguồn LaTeX](https://github.com/tiepvupsu/ebookMLCB), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); trang gốc [machinelearningcoban.com](https://machinelearningcoban.com)). Người biên soạn chỉ đổi định dạng sang trang web: số chương, mục, công thức và hình giữ như bản PDF.
>
> **Đính chính** *(người biên soạn)*:
>
> - Ví dụ 2: vì $\mathbf{x}^T\mathbf{A}^T\mathbf{A}\mathbf{x} = \|\mathbf{A}\mathbf{x}\|_2^2$ là bình phương của chuẩn, kết luận đúng là $\|\mathbf{A}\|_2 = \sqrt{\lambda_{\max}(\mathbf{A}^T\mathbf{A})}$ (sách viết thiếu dấu căn). Tương tự, giá trị suy biến của $\mathbf{A}$ là **căn bậc hai** của các trị riêng của $\mathbf{A}^T\mathbf{A}$. Câu kết "chuẩn $\ell_2$ của một ma trận là giá trị suy biến lớn nhất của ma trận đó" vẫn đúng. Kiểm chứng bằng số ở trang [Code](math/appendix_a_code_vi).

Việc tối ưu hàm số một biến liên tục và khả vi trên miền xác định là một tập mở<sup>1</sup> thường được thực hiện dựa trên việc giải phương trình đạo hàm bằng không. Gọi hàm mục tiêu là $f(x): \mathbb{R} \rightarrow \mathbb{R}$, cực trị toàn cục nếu có thường được tìm bằng cách giải phương trình $f'(x) = 0$. Chú ý rằng điều ngược lại không đúng, tức một điểm thoả mãn đạo hàm bằng không chưa chắc đã là cực trị của hàm số. Ví dụ hàm $f(x) = x^3$ có đạo hàm bằng không tại $x =0$ nhưng điểm này không là một điểm cực trị. Với hàm nhiều biến, ta cũng có thể áp dụng quan sát này: giải phương trình gradient bằng không.

Cách làm trên đây được áp dụng vào các bài toán tối ưu không ràng buộc. Các bài toán có ràng buộc là một phương trình:

$$
\begin{align*}
\mathbf{x}=& \arg\min_{\mathbf{x}} f_0(\mathbf{x}) \\
\text{thoả mãn:}~& f_1(\mathbf{x}) = 0, \tag{A.1}
\end{align*}
$$

cũng có thể được đưa về bài toán không ràng buộc bằng *phương pháp nhân tử Lagrange*.

Xét hàm số $\mathcal{L}(\mathbf{x}, \lambda) = f_0(\mathbf{x}) + \lambda f_1(\mathbf{x})$ với biến $\lambda$ được gọi là *nhân tử Lagrange* (Lagrange multiplier). Hàm số $\mathcal{L}(\mathbf{x}, \lambda)$ được gọi là *hàm Lagrange* của bài toán. Người ta đã chứng minh được rằng, điểm tối ưu của bài toán (A.1) thoả mãn điều kiện $\nabla_{\mathbf{x}, \lambda} \mathcal{L}(\mathbf{x}, \lambda) = \mathbf{0}$. Điều này tương đương với:

$$
\begin{align*}
\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \lambda) &=  \nabla_{\mathbf{x}}f_0(\mathbf{x}) + \lambda
\nabla_{\mathbf{x}} f_1(\mathbf{x}) = 0 \tag{A.2} \\
\nabla_{\lambda}\mathcal{L}(\mathbf{x}, \lambda) &= f_1(\mathbf{x}) = 0 \tag{A.3}
\end{align*}
$$

Để ý rằng điều kiện thứ hai chính là ràng buộc trong bài toán (A.1).

Trong nhiều trường hợp, việc giải hệ phương trình (A.2) - (A.3) đơn giản hơn việc trực tiếp đi tìm nghiệm của bài toán (A.1).

**Ví dụ 1**:

Tìm giá trị lớn nhất và nhỏ nhất của hàm số $f_0(x, y) = x + y$ với $x, y$ thoả mãn điều kiện $f_1(x, y) = x^2 + y^2 = 2$.

***Lời giải***:

Điều kiện ràng buộc có thể được viết lại dưới dạng $x^2 + y^2 - 2 = 0$. Hàm Lagrange của bài toán này là $\mathcal{L}(x, y, \lambda) = x + y + \lambda(x^2 + y^2 - 2)$. Các điểm cực trị của hàm số Lagrange phải thoả mãn hệ điều kiện:

$$
\nabla_{x, y, \lambda} \mathcal{L}(x, y, \lambda) = 0 \Leftrightarrow
\left\{
\begin{matrix}
1 + 2\lambda x &= 0 \\ 
1 + 2\lambda y &= 0 \\ 
x^2 + y^2 &=     2
\end{matrix}
\right. \tag{A.4}
$$

Từ hai phương trình đầu của (A.4) suy ra $x = y = \frac{-1}{2\lambda}$. Thay vào phương trình cuối ta sẽ có $\lambda^2 = \frac{1}{4} \Rightarrow \lambda = \pm \frac{1}{2}$. Vậy ta được 2 cặp nghiệm $(x, y) \in \{(1, 1), (-1, -1)\}$. Bằng cách thay các giá trị này vào hàm mục tiêu, ta tìm được giá trị nhỏ nhất và lớn nhất của bài toán.

**Ví dụ 2**: Chuẩn $\ell_2$ của ma trận.

Nhắc lại chuẩn $\ell_2$ của một vector $\mathbf{x}: \|\mathbf{x}\|_2 = \sqrt{\mathbf{x}^T\mathbf{x}}$. Dựa trên chuẩn $\ell_2$ của vector, chuẩn $\ell_2$ của một ma trận $\mathbf{A} \in \mathbb{R}^{m\times n}$, ký hiệu là $\|\mathbf{A}\|_2$, được định nghĩa như sau:

$$
\|\mathbf{A}\|_2 = \max \frac{\|\mathbf{A} \mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \max \sqrt{\frac{\mathbf{x}^T\mathbf{A}^T\mathbf{A}\mathbf{x}}{\mathbf{x}^T\mathbf{x}}}, \text{với}~ \mathbf{x} \in \mathbb{R}^{n} \tag{A.5}
$$

Bài toán tối ưu này tương đương với:

$$
\begin{align*}
\begin{aligned}
\max\left(\mathbf{x}^T\mathbf{A}^T\mathbf{A}\mathbf{x}\right) \\
\text{thoả mãn:}~ \mathbf{x}^T\mathbf{x} = 1
\end{aligned} \tag{A.6}
\end{align*}
$$

Hàm Lagrange của bài toán này là

$$
\mathcal{L}(\mathbf{x}, \lambda) = \mathbf{x}^T\mathbf{A}^T\mathbf{A}\mathbf{x} + \lambda ( 1 - \mathbf{x}^T\mathbf{x}) \tag{A.7}
$$

Các điểm cực trị của hàm số Lagrange phải thoả mãn:

$$
\begin{align*}
\nabla_{\mathbf{x}}\mathcal{L} &= 2\mathbf{A}^T\mathbf{A}\mathbf{x} - 2\lambda\mathbf{x} = \mathbf{0} \tag{A.8} \\
\nabla_{\lambda}\mathcal{L} &= 1 - \mathbf{x}^T\mathbf{x} = 0 \tag{A.9}
\end{align*}
$$

Từ (A.8) ta có $\mathbf{A}^T\mathbf{A}\mathbf{x} = \lambda\mathbf{x}$. Vậy $\mathbf{x}$ phải là một vector riêng của $\mathbf{A}^T\mathbf{A}$ ứng với trị riêng $\lambda$. Nhân cả hai vế của biểu thức này với $\mathbf{x}^T$ vào bên trái và sử dụng (A.9), ta thu được:

$$
\mathbf{x}^T\mathbf{A}^T\mathbf{A}\mathbf{x} = \lambda \mathbf{x}^T\mathbf{x} = \lambda \tag{A.10}
$$

Từ đó suy ra $\|\mathbf{A}\mathbf{x}\|_2$ đạt giá trị lớn nhất khi $\lambda$ đạt giá trị lớn nhất. Nói cách khác, $\lambda$ phải là trị riêng lớn nhất của $\mathbf{A}^T\mathbf{A}$. Vậy, $\|\mathbf{A}\|_2 = \lambda_{\max}(\mathbf{A}^T\mathbf{A}).$

Các trị riêng của $\mathbf{A}^T\mathbf{A}$ còn được gọi là *giá trị suy biến* (singular value) của $\mathbf{A}$. Tóm lại, chuẩn $\ell_2$ của một ma trận là giá trị suy biến lớn nhất của ma trận đó.

Hoàn toàn tương tự, nghiệm của bài toán

$$
\min_{\|\mathbf{x}\| = 1} \|\mathbf{A}\mathbf{x}\|_2 \tag{A.11}
$$

chính là một vector riêng ứng với giá trị suy biến nhỏ nhất của $\mathbf{A}$.

---

**Chú thích**

1. Xem thêm: *Open sets, closed sets and sequences of real numbers* (<https://goo.gl/AgKhCn>).
