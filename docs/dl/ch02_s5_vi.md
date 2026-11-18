# 2.5 Tính Vi phân Tự động

> Ghi chú của người biên soạn: trang này dịch nguyên văn từ sách *Dive into Deep Learning* của **Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola** ([bản gốc](https://d2l.ai/chapter_preliminaries/autograd.html), [mã nguồn](https://github.com/d2l-ai/d2l-en), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); code theo giấy phép MIT-0). Chỉ giữ code PyTorch. Số mục, hình và công thức giữ như bản HTML d2l.ai; thuật ngữ đối chiếu với bản dịch [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn).

Hãy nhớ lại từ [Mục 2.4](https://d2l.ai/chapter_preliminaries/calculus.html) rằng tính đạo hàm là bước then chốt trong mọi thuật toán tối ưu mà ta sẽ dùng để huấn luyện mạng sâu. Dù các phép tính này đơn giản, việc tự tính bằng tay có thể tẻ nhạt và dễ sai sót, và những vấn đề này chỉ tăng lên khi mô hình của ta trở nên phức tạp hơn.

May mắn thay, mọi framework học sâu hiện đại đều gánh hộ ta công việc này bằng cách cung cấp tính năng *tính vi phân tự động* (automatic differentiation, thường gọi tắt là *autograd*). Khi ta đưa dữ liệu qua từng hàm nối tiếp nhau, framework xây dựng một *đồ thị tính toán* (computational graph) theo dõi cách mỗi giá trị phụ thuộc vào các giá trị khác. Để tính đạo hàm, cơ chế tính vi phân tự động đi ngược qua đồ thị này và áp dụng quy tắc dây chuyền. Thuật toán tính toán áp dụng quy tắc dây chuyền theo cách này được gọi là *lan truyền ngược* (backpropagation).

Dù các thư viện autograd đã trở thành mối quan tâm nóng hổi trong thập kỷ qua, chúng có một lịch sử lâu dài. Thực ra, những tài liệu sớm nhất về autograd đã có từ hơn nửa thế kỷ trước ([Wengert, 1964](dl/ch02_s5_vi?id=ref-wengert-1964)). Những ý tưởng cốt lõi đằng sau lan truyền ngược hiện đại bắt nguồn từ một luận án tiến sĩ năm 1980 ([Speelpenning, 1980](dl/ch02_s5_vi?id=ref-speelpenning-1980)) và được phát triển thêm vào cuối những năm 1980 ([Griewank, 1989](dl/ch02_s5_vi?id=ref-griewank-1989)). Dù lan truyền ngược đã trở thành phương pháp mặc định để tính gradient, nó không phải là lựa chọn duy nhất. Chẳng hạn, ngôn ngữ lập trình Julia dùng lan truyền xuôi ([Revels *et al.*, 2016](dl/ch02_s5_vi?id=ref-revels-lubin-papamarkou-2016)). Trước khi tìm hiểu các phương pháp, hãy làm chủ gói autograd trước đã.

<!-- run -->
```python
import torch
```

## 2.5.1 Một Hàm Đơn giản

Giả sử ta muốn tính đạo hàm của hàm $y = 2\mathbf{x}^{\top}\mathbf{x}$ theo vector cột $\mathbf{x}$. Để bắt đầu, ta gán cho `x` một giá trị ban đầu.

<!-- run -->
```python
x = torch.arange(4.0)
x
```

<!-- output -->
```output
tensor([0., 1., 2., 3.])
```
<!-- /output -->

Trước khi tính gradient của $y$ theo $\mathbf{x}$, ta cần một nơi để lưu nó. Nói chung, ta tránh cấp phát bộ nhớ mới mỗi lần lấy đạo hàm vì học sâu đòi hỏi tính đạo hàm theo cùng các tham số rất nhiều lần liên tiếp, và ta có thể có nguy cơ hết bộ nhớ. Lưu ý rằng gradient của một hàm giá trị vô hướng theo một vector $\mathbf{x}$ là một vector có cùng kích thước với $\mathbf{x}$.

<!-- run -->
```python
# Cũng có thể tạo x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
x.grad  # Mặc định gradient là None
```

Giờ ta tính hàm của `x` và gán kết quả cho `y`.

<!-- run -->
```python
y = 2 * torch.dot(x, x)
y
```

<!-- output -->
```output
tensor(28., grad_fn=<MulBackward0>)
```
<!-- /output -->

Giờ ta có thể lấy gradient của `y` theo `x` bằng cách gọi phương thức `backward` của nó. Tiếp theo, ta có thể truy cập gradient qua thuộc tính `grad` của `x`.

<!-- run -->
```python
y.backward()
x.grad
```

<!-- output -->
```output
tensor([ 0.,  4.,  8., 12.])
```
<!-- /output -->

Ta đã biết rằng gradient của hàm $y = 2\mathbf{x}^{\top}\mathbf{x}$ theo $\mathbf{x}$ phải là $4\mathbf{x}$. Giờ ta có thể kiểm chứng rằng kết quả tính gradient tự động và kết quả mong đợi là giống hệt nhau.

<!-- run -->
```python
x.grad == 4 * x
```

<!-- output -->
```output
tensor([True, True, True, True])
```
<!-- /output -->

Giờ hãy tính một hàm khác của `x` và lấy gradient của nó. Lưu ý rằng PyTorch không tự động đặt lại bộ đệm gradient khi ta ghi nhận một gradient mới. Thay vào đó, gradient mới được cộng vào gradient đã lưu. Hành vi này tiện lợi khi ta muốn tối ưu tổng của nhiều hàm mục tiêu. Để đặt lại bộ đệm gradient, ta có thể gọi `x.grad.zero_()` như sau:

<!-- run -->
```python
x.grad.zero_()  # Đặt lại gradient
y = x.sum()
y.backward()
x.grad
```

<!-- output -->
```output
tensor([1., 1., 1., 1.])
```
<!-- /output -->

## 2.5.2 Lan truyền Ngược cho Biến Không phải Số vô hướng

Khi `y` là một vector, cách biểu diễn tự nhiên nhất cho đạo hàm của `y` theo một vector `x` là một ma trận gọi là *ma trận Jacobi* (Jacobian), chứa đạo hàm riêng của mỗi thành phần của `y` theo mỗi thành phần của `x`. Tương tự, với `y` và `x` bậc cao hơn, kết quả của phép lấy đạo hàm có thể là một tensor bậc còn cao hơn nữa.

Dù ma trận Jacobi có xuất hiện trong một số kỹ thuật học máy nâng cao, thông thường hơn ta muốn cộng gộp gradient của mỗi thành phần của `y` theo toàn bộ vector `x`, cho ra một vector có cùng kích thước với `x`. Chẳng hạn, ta thường có một vector biểu diễn giá trị hàm mất mát được tính riêng cho mỗi mẫu trong một *batch* các mẫu huấn luyện. Ở đây, ta chỉ muốn cộng gộp các gradient được tính riêng cho từng mẫu.

Vì các framework học sâu diễn giải gradient của các tensor không phải số vô hướng theo những cách khác nhau, PyTorch có một số bước để tránh nhầm lẫn. Gọi `backward` trên một đối tượng không phải số vô hướng sẽ gây lỗi trừ khi ta cho PyTorch biết cách quy đối tượng đó về một số vô hướng. Chính thức hơn, ta cần cung cấp một vector $\mathbf{v}$ nào đó để `backward` tính $\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$ thay vì $\partial_{\mathbf{x}} \mathbf{y}$. Phần tiếp theo có thể gây bối rối, nhưng vì những lý do sẽ trở nên rõ ràng sau này, đối số này (biểu diễn $\mathbf{v}$) được đặt tên là `gradient`. Để có mô tả chi tiết hơn, xem [bài viết trên Medium](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29) của Yang Zhang.

<!-- run -->
```python
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Nhanh hơn: y.sum().backward()
x.grad
```

<!-- output -->
```output
tensor([0., 2., 4., 6.])
```
<!-- /output -->

## 2.5.3 Tách Phép tính

Đôi khi, ta muốn đưa một số phép tính ra ngoài đồ thị tính toán được ghi nhận. Chẳng hạn, giả sử ta dùng đầu vào để tạo một số đại lượng trung gian phụ mà ta không muốn tính gradient. Trong trường hợp này, ta cần *tách* (detach) đồ thị tính toán tương ứng khỏi kết quả cuối cùng. Ví dụ đơn giản sau làm rõ điều này: giả sử ta có `z = x * y` và `y = x * x` nhưng ta muốn tập trung vào ảnh hưởng *trực tiếp* của `x` lên `z` thay vì ảnh hưởng được truyền qua `y`. Trong trường hợp này, ta có thể tạo một biến mới `u` nhận cùng giá trị với `y` nhưng *nguồn gốc* (cách nó được tạo ra) đã bị xóa. Do đó `u` không có tổ tiên nào trong đồ thị và gradient không chảy qua `u` về `x`. Chẳng hạn, lấy gradient của `z = x * u` sẽ cho kết quả `u` (chứ không phải `3 * x * x` như bạn có thể đã nghĩ vì `z = x * x * x`).

<!-- run -->
```python
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

<!-- output -->
```output
tensor([True, True, True, True])
```
<!-- /output -->

Lưu ý rằng dù thủ tục này tách các tổ tiên của `y` khỏi đồ thị dẫn tới `z`, đồ thị tính toán dẫn tới `y` vẫn còn và do đó ta có thể tính gradient của `y` theo `x`.

<!-- run -->
```python
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

<!-- output -->
```output
tensor([True, True, True, True])
```
<!-- /output -->

## 2.5.4 Gradient và Luồng Điều khiển của Python

Cho đến nay ta đã xem xét các trường hợp mà đường đi từ đầu vào đến đầu ra được xác định rõ qua một hàm như `z = x * x * x`. Lập trình cho ta nhiều tự do hơn hẳn trong cách tính kết quả. Chẳng hạn, ta có thể làm cho chúng phụ thuộc vào các biến phụ hoặc đặt điều kiện lựa chọn dựa trên các kết quả trung gian. Một lợi ích của việc dùng tính vi phân tự động là ngay cả khi việc xây dựng đồ thị tính toán của một hàm đòi hỏi đi qua một mê cung luồng điều khiển của Python (chẳng hạn các câu lệnh điều kiện, vòng lặp và các lời gọi hàm tùy ý), ta vẫn có thể tính gradient của biến kết quả. Để minh họa điều này, hãy xét đoạn code sau, trong đó số lần lặp của vòng `while` và kết quả của câu lệnh `if` đều phụ thuộc vào giá trị của đầu vào `a`.

<!-- run -->
```python
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Dưới đây, ta gọi hàm này với đầu vào là một giá trị ngẫu nhiên. Vì đầu vào là một biến ngẫu nhiên, ta không biết đồ thị tính toán sẽ có dạng nào. Tuy nhiên, mỗi khi thực thi `f(a)` trên một đầu vào cụ thể, ta hiện thực hóa một đồ thị tính toán cụ thể và sau đó có thể chạy `backward`.

<!-- run -->
```python
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

Dù hàm `f` của ta, vì mục đích minh họa, hơi gượng ép, sự phụ thuộc của nó vào đầu vào khá đơn giản: nó là một hàm *tuyến tính* của `a` với hệ số tỉ lệ được định nghĩa từng khúc. Do đó, `f(a) / a` là một vector gồm các phần tử hằng số và, hơn nữa, `f(a) / a` phải khớp với gradient của `f(a)` theo `a`.

<!-- run -->
```python
a.grad == d / a
```

<!-- output -->
```output
tensor(True)
```
<!-- /output -->

Luồng điều khiển động rất phổ biến trong học sâu. Chẳng hạn, khi xử lý văn bản, đồ thị tính toán phụ thuộc vào độ dài của đầu vào. Trong những trường hợp này, tính vi phân tự động trở nên thiết yếu cho mô hình hóa thống kê vì không thể tính gradient *trước* (a priori).

## 2.5.5 Thảo luận

Giờ bạn đã nếm thử sức mạnh của tính vi phân tự động. Sự phát triển của các thư viện tính đạo hàm vừa tự động vừa hiệu quả đã giúp tăng năng suất khổng lồ cho những người làm học sâu, giải phóng họ để có thể tập trung vào những việc ít nhàm chán hơn. Hơn nữa, autograd cho phép ta thiết kế những mô hình khổng lồ mà việc tính gradient bằng giấy bút sẽ tốn thời gian không thể chấp nhận được. Thú vị là, trong khi ta dùng autograd để *tối ưu* mô hình (theo nghĩa thống kê), việc *tối ưu* chính các thư viện autograd (theo nghĩa tính toán) là một chủ đề phong phú và có tầm quan trọng sống còn với những người thiết kế framework. Ở đây, các công cụ từ trình biên dịch và thao tác đồ thị được tận dụng để tính kết quả theo cách nhanh nhất và tiết kiệm bộ nhớ nhất.

Tạm thời, hãy cố nhớ những điều cơ bản sau: (i) gắn gradient cho những biến mà ta muốn lấy đạo hàm theo chúng; (ii) ghi nhận quá trình tính giá trị đích; (iii) thực thi hàm lan truyền ngược; và (iv) truy cập gradient thu được.

## 2.5.6 Bài tập

1. Vì sao đạo hàm bậc hai tốn kém hơn nhiều so với đạo hàm bậc nhất?
1. Sau khi chạy hàm lan truyền ngược, hãy chạy lại ngay lập tức và xem điều gì xảy ra. Hãy tìm hiểu nguyên nhân.
1. Trong ví dụ luồng điều khiển, nơi ta tính đạo hàm của `d` theo `a`, điều gì sẽ xảy ra nếu ta đổi biến `a` thành một vector hoặc ma trận ngẫu nhiên? Lúc này, kết quả của phép tính `f(a)` không còn là một số vô hướng. Điều gì xảy ra với kết quả? Ta phân tích điều này như thế nào?
1. Cho $f(x) = \sin(x)$. Vẽ đồ thị của $f$ và của đạo hàm $f'$ của nó. Đừng tận dụng việc $f'(x) = \cos(x)$ mà hãy dùng tính vi phân tự động để có kết quả.
1. Cho $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$. Hãy vẽ một đồ thị phụ thuộc lần theo các kết quả từ $x$ đến $f(x)$.
1. Dùng quy tắc dây chuyền để tính đạo hàm $\frac{df}{dx}$ của hàm số nêu trên, đặt mỗi số hạng lên đồ thị phụ thuộc mà bạn đã dựng trước đó.
1. Với đồ thị và các kết quả đạo hàm trung gian, bạn có một số lựa chọn khi tính gradient. Hãy tính kết quả một lần bắt đầu từ $x$ đến $f$ và một lần từ $f$ lần ngược về $x$. Đường đi từ $x$ đến $f$ thường được gọi là *vi phân xuôi* (forward differentiation), còn đường đi từ $f$ về $x$ được gọi là vi phân ngược (backward differentiation).
1. Khi nào bạn có thể muốn dùng vi phân xuôi, và khi nào dùng vi phân ngược? Gợi ý: xét lượng dữ liệu trung gian cần thiết, khả năng song song hóa các bước, và kích thước của các ma trận và vector liên quan.

[Thảo luận (tiếng Anh)](https://discuss.d2l.ai/t/35)

## Tài liệu tham khảo

- <a id="ref-wengert-1964"></a>Wengert, Robert Edwin (1964). *A simple automatic derivative evaluation program*. Communications of the ACM.
- <a id="ref-speelpenning-1980"></a>Speelpenning, Bert (1980). *Compiling fast partial derivatives of functions given by algorithms*.
- <a id="ref-griewank-1989"></a>Griewank, Andreas (1989). *On automatic differentiation*. Mathematical Programming: Recent Developments and Applications.
- <a id="ref-revels-lubin-papamarkou-2016"></a>Revels, Jarrett, Lubin, Miles, Papamarkou, Theodore (2016). *Forward-mode automatic differentiation in Julia*. ArXiv:1607.07892.
