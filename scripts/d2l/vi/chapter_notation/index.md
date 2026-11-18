# Ký hiệu
:label:`chap_notation`

Trong suốt cuốn sách này, chúng tôi tuân theo các quy ước ký hiệu dưới đây. Lưu ý rằng một số ký hiệu chỉ là ký hiệu đại diện, trong khi những ký hiệu khác chỉ các đối tượng cụ thể. Theo nguyên tắc chung, mạo từ bất định "một" thường cho biết ký hiệu đó là đại diện và các ký hiệu có định dạng tương tự có thể biểu thị các đối tượng khác cùng loại. Chẳng hạn, "$x$: một số vô hướng" nghĩa là chữ thường nói chung biểu diễn các giá trị vô hướng, nhưng "$\mathbb{Z}$: tập các số nguyên" chỉ riêng ký hiệu $\mathbb{Z}$.

## Các Đối tượng Số

* $x$: một số vô hướng
* $\mathbf{x}$: một vector
* $\mathbf{X}$: một ma trận
* $\mathsf{X}$: một tensor tổng quát
* $\mathbf{I}$: ma trận đơn vị (với số chiều cho trước), tức là ma trận vuông có mọi phần tử trên đường chéo bằng $1$ và mọi phần tử ngoài đường chéo bằng $0$
* $x_i$, $[\mathbf{x}]_i$: phần tử thứ $i$ của vector $\mathbf{x}$
* $x_{ij}$, $x_{i,j}$,$[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: phần tử của ma trận $\mathbf{X}$ ở hàng $i$ và cột $j$.

## Lý thuyết Tập hợp

* $\mathcal{X}$: một tập hợp
* $\mathbb{Z}$: tập các số nguyên
* $\mathbb{Z}^+$: tập các số nguyên dương
* $\mathbb{R}$: tập các số thực
* $\mathbb{R}^n$: tập các vector số thực $n$ chiều
* $\mathbb{R}^{a\times b}$: tập các ma trận số thực có $a$ hàng và $b$ cột
* $|\mathcal{X}|$: lực lượng (số phần tử) của tập $\mathcal{X}$
* $\mathcal{A}\cup\mathcal{B}$: hợp của hai tập $\mathcal{A}$ và $\mathcal{B}$
* $\mathcal{A}\cap\mathcal{B}$: giao của hai tập $\mathcal{A}$ và $\mathcal{B}$
* $\mathcal{A}\setminus\mathcal{B}$: hiệu của tập $\mathcal{A}$ trừ tập $\mathcal{B}$ (chỉ gồm các phần tử của $\mathcal{A}$ không thuộc $\mathcal{B}$)

## Hàm số và Toán tử

* $f(\cdot)$: một hàm số
* $\log(\cdot)$: logarit tự nhiên (cơ số $e$)
* $\log_2(\cdot)$: logarit cơ số $2$
* $\exp(\cdot)$: hàm mũ
* $\mathbf{1}(\cdot)$: hàm chỉ thị; bằng $1$ nếu đối số logic là đúng, và bằng $0$ nếu ngược lại
* $\mathbf{1}_{\mathcal{X}}(z)$: hàm chỉ thị thuộc tập; bằng $1$ nếu phần tử $z$ thuộc tập $\mathcal{X}$ và bằng $0$ nếu ngược lại
* $\mathbf{(\cdot)}^\top$: chuyển vị của một vector hoặc ma trận
* $\mathbf{X}^{-1}$: nghịch đảo của ma trận $\mathbf{X}$
* $\odot$: tích Hadamard (theo từng phần tử)
* $[\cdot, \cdot]$: phép ghép nối
* $\|\cdot\|_p$: chuẩn $\ell_p$
* $\|\cdot\|$: chuẩn $\ell_2$
* $\langle \mathbf{x}, \mathbf{y} \rangle$: tích vô hướng (tích chấm) của hai vector $\mathbf{x}$ và $\mathbf{y}$
* $\sum$: phép lấy tổng trên một tập các phần tử
* $\prod$: phép lấy tích trên một tập các phần tử
* $\stackrel{\textrm{def}}{=}$: một đẳng thức được khẳng định như định nghĩa của ký hiệu ở vế trái

## Giải tích

* $\frac{dy}{dx}$: đạo hàm của $y$ theo $x$
* $\frac{\partial y}{\partial x}$: đạo hàm riêng của $y$ theo $x$
* $\nabla_{\mathbf{x}} y$: gradient của $y$ theo $\mathbf{x}$
* $\int_a^b f(x) \;dx$: tích phân xác định của $f$ từ $a$ đến $b$ theo $x$
* $\int f(x) \;dx$: tích phân bất định của $f$ theo $x$

## Xác suất và Lý thuyết Thông tin

* $X$: một biến ngẫu nhiên
* $P$: một phân phối xác suất
* $X \sim P$: biến ngẫu nhiên $X$ tuân theo phân phối $P$
* $P(X=x)$: xác suất gán cho sự kiện biến ngẫu nhiên $X$ nhận giá trị $x$
* $P(X \mid Y)$: phân phối xác suất có điều kiện của $X$ khi biết $Y$
* $p(\cdot)$: một hàm mật độ xác suất (PDF) gắn với phân phối $P$
* ${E}[X]$: kỳ vọng của biến ngẫu nhiên $X$
* $X \perp Y$: hai biến ngẫu nhiên $X$ và $Y$ độc lập
* $X \perp Y \mid Z$: hai biến ngẫu nhiên $X$ và $Y$ độc lập có điều kiện khi biết $Z$
* $\sigma_X$: độ lệch chuẩn của biến ngẫu nhiên $X$
* $\textrm{Var}(X)$: phương sai của biến ngẫu nhiên $X$, bằng $\sigma^2_X$
* $\textrm{Cov}(X, Y)$: hiệp phương sai của hai biến ngẫu nhiên $X$ và $Y$
* $\rho(X, Y)$: hệ số tương quan Pearson giữa $X$ và $Y$, bằng $\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
* $H(X)$: entropy của biến ngẫu nhiên $X$
* $D_{\textrm{KL}}(P\|Q)$: phân kỳ KL (hay entropy tương đối) từ phân phối $Q$ đến phân phối $P$

[Discussions](https://discuss.d2l.ai/t/25)
