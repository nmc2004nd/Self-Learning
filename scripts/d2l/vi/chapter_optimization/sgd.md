# Hạ Gradient Ngẫu nhiên
:label:`sec_sgd`

Trong các chương trước, ta đã liên tục dùng hạ gradient ngẫu nhiên trong quy trình huấn luyện mà không giải thích vì sao nó hoạt động. Để làm sáng tỏ điều này, ta vừa mô tả các nguyên lý cơ bản của hạ gradient trong :numref:`sec_gd`. Trong mục này, ta tiếp tục thảo luận chi tiết hơn về *hạ gradient ngẫu nhiên* (stochastic gradient descent).

```{.python .input}
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

## Cập nhật Gradient Ngẫu nhiên

Trong học sâu, hàm mục tiêu thường là trung bình của các hàm mất mát cho mỗi mẫu trong tập dữ liệu huấn luyện. Cho một tập dữ liệu huấn luyện gồm $n$ mẫu, ta giả sử $f_i(\mathbf{x})$ là hàm mất mát ứng với mẫu huấn luyện có chỉ số $i$, trong đó $\mathbf{x}$ là vector tham số. Khi đó ta có hàm mục tiêu

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

Gradient của hàm mục tiêu tại $\mathbf{x}$ được tính bằng

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

Nếu dùng hạ gradient, chi phí tính toán cho mỗi vòng lặp cập nhật biến độc lập là $\mathcal{O}(n)$, tăng tuyến tính theo $n$. Do đó, khi tập dữ liệu huấn luyện lớn hơn, chi phí của hạ gradient cho mỗi vòng lặp sẽ cao hơn.

Hạ gradient ngẫu nhiên (SGD) giảm chi phí tính toán ở mỗi vòng lặp. Ở mỗi vòng lặp của hạ gradient ngẫu nhiên, ta lấy mẫu đều ngẫu nhiên một chỉ số $i\in\{1,\ldots, n\}$ của các mẫu dữ liệu, và tính gradient $\nabla f_i(\mathbf{x})$ để cập nhật $\mathbf{x}$:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

trong đó $\eta$ là tốc độ học. Ta có thể thấy chi phí tính toán cho mỗi vòng lặp giảm từ $\mathcal{O}(n)$ của hạ gradient xuống hằng số $\mathcal{O}(1)$. Hơn nữa, chúng tôi muốn nhấn mạnh rằng gradient ngẫu nhiên $\nabla f_i(\mathbf{x})$ là một ước lượng không chệch của gradient đầy đủ $\nabla f(\mathbf{x})$ vì

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

Điều này nghĩa là, tính trung bình, gradient ngẫu nhiên là một ước lượng tốt của gradient.

Giờ ta sẽ so sánh nó với hạ gradient bằng cách thêm nhiễu ngẫu nhiên có trung bình 0 và phương sai 1 vào gradient để mô phỏng hạ gradient ngẫu nhiên.

```{.python .input}
def f(x1, x2):  # Hàm mục tiêu
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient của hàm mục tiêu
    return 2 * x1, 4 * x2
```

```{.python .input}
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Mô phỏng gradient có nhiễu
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Tốc độ học không đổi
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Như ta có thể thấy, quỹ đạo của các biến trong hạ gradient ngẫu nhiên nhiễu hơn nhiều so với quỹ đạo ta đã quan sát trong hạ gradient ở :numref:`sec_gd`. Điều này là do bản chất ngẫu nhiên của gradient. Tức là, ngay cả khi đến gần cực tiểu, ta vẫn chịu sự bất định do gradient tức thời đưa vào qua $\eta \nabla f_i(\mathbf{x})$. Ngay cả sau 50 bước, chất lượng vẫn chưa tốt lắm. Tệ hơn nữa, nó sẽ không cải thiện sau thêm nhiều bước (chúng tôi khuyến khích bạn thử nghiệm với số bước lớn hơn để xác nhận điều này). Điều này để lại cho ta lựa chọn duy nhất: thay đổi tốc độ học $\eta$. Tuy nhiên, nếu chọn nó quá nhỏ, ban đầu ta sẽ không có tiến triển đáng kể nào. Mặt khác, nếu chọn nó quá lớn, ta sẽ không có được nghiệm tốt, như đã thấy ở trên. Cách duy nhất để giải quyết những mục tiêu mâu thuẫn này là giảm tốc độ học một cách *động* khi quá trình tối ưu tiến triển.

Đây cũng là lý do thêm một hàm tốc độ học `lr` vào hàm bước `sgd`. Trong ví dụ trên, mọi chức năng lập lịch tốc độ học đều nằm im vì ta đặt hàm `lr` tương ứng là hằng số.

## Tốc độ học Động

Thay $\eta$ bằng một tốc độ học phụ thuộc thời gian $\eta(t)$ làm tăng độ phức tạp của việc kiểm soát sự hội tụ của một thuật toán tối ưu. Cụ thể, ta cần tìm ra $\eta$ nên giảm nhanh đến đâu. Nếu quá nhanh, ta sẽ dừng tối ưu quá sớm. Nếu giảm quá chậm, ta lãng phí quá nhiều thời gian cho việc tối ưu. Sau đây là một vài chiến lược cơ bản được dùng để điều chỉnh $\eta$ theo thời gian (ta sẽ thảo luận các chiến lược nâng cao hơn sau):

$$
\begin{aligned}
    \eta(t) & = \eta_i \textrm{ nếu } t_i \leq t \leq t_{i+1}  && \textrm{hằng số từng khúc} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \textrm{suy giảm theo hàm mũ} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \textrm{suy giảm theo đa thức}
\end{aligned}
$$

Trong kịch bản *hằng số từng khúc* (piecewise constant) đầu tiên, ta giảm tốc độ học, chẳng hạn bất cứ khi nào tiến trình tối ưu bị đình trệ. Đây là một chiến lược phổ biến để huấn luyện mạng sâu. Hoặc, ta có thể giảm nó mạnh hơn nhiều bằng *suy giảm theo hàm mũ* (exponential decay). Đáng tiếc là điều này thường dẫn đến việc dừng quá sớm trước khi thuật toán hội tụ. Một lựa chọn phổ biến là *suy giảm theo đa thức* (polynomial decay) với $\alpha = 0.5$. Trong trường hợp tối ưu lồi, có một số chứng minh cho thấy tốc độ này hoạt động tốt.

Hãy xem suy giảm theo hàm mũ trông như thế nào trong thực tế.

```{.python .input}
def exponential_lr():
    # Biến toàn cục được định nghĩa bên ngoài hàm này và được cập nhật bên trong
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

Đúng như dự đoán, phương sai của các tham số giảm đáng kể. Tuy nhiên, cái giá phải trả là không hội tụ được về nghiệm tối ưu $\mathbf{x} = (0, 0)$. Ngay cả sau 1000 bước lặp, ta vẫn còn rất xa nghiệm tối ưu. Thật vậy, thuật toán hoàn toàn không hội tụ được. Mặt khác, nếu dùng suy giảm theo đa thức, trong đó tốc độ học giảm theo nghịch đảo căn bậc hai của số bước, sự hội tụ tốt hơn chỉ sau 50 bước.

```{.python .input}
def polynomial_lr():
    # Biến toàn cục được định nghĩa bên ngoài hàm này và được cập nhật bên trong
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Có rất nhiều lựa chọn khác về cách đặt tốc độ học. Chẳng hạn, ta có thể bắt đầu với một tốc độ nhỏ, rồi tăng nhanh và sau đó giảm lại, dù chậm hơn. Ta thậm chí có thể xen kẽ giữa các tốc độ học nhỏ hơn và lớn hơn. Có rất nhiều lịch trình như vậy. Tạm thời, hãy tập trung vào các lịch tốc độ học mà có thể phân tích lý thuyết toàn diện, tức là tốc độ học trong bối cảnh lồi. Với các bài toán không lồi tổng quát, rất khó có được những đảm bảo hội tụ có ý nghĩa, vì nói chung cực tiểu hóa các bài toán phi tuyến không lồi là NP-khó. Để có một bài tổng quan, xem chẳng hạn [bài giảng](https://www.stat.cmu.edu/%7Eryantibs/convexopt-F15/lectures/26-nonconvex.pdf) xuất sắc của Tibshirani năm 2015.

## Phân tích Hội tụ cho Hàm mục tiêu Lồi

Phân tích hội tụ dưới đây của hạ gradient ngẫu nhiên cho các hàm mục tiêu lồi là tùy chọn và chủ yếu nhằm truyền đạt thêm trực giác về bài toán. Ta giới hạn ở một trong những chứng minh đơn giản nhất :cite:`Nesterov.Vial.2000`. Có những kỹ thuật chứng minh nâng cao hơn đáng kể, chẳng hạn bất cứ khi nào hàm mục tiêu đặc biệt dễ xử lý.

Giả sử hàm mục tiêu $f(\boldsymbol{\xi}, \mathbf{x})$ lồi theo $\mathbf{x}$ với mọi $\boldsymbol{\xi}$. Cụ thể hơn, ta xét phép cập nhật hạ gradient ngẫu nhiên:

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

trong đó $f(\boldsymbol{\xi}_t, \mathbf{x})$ là hàm mục tiêu ứng với mẫu huấn luyện $\boldsymbol{\xi}_t$ được lấy từ một phân phối nào đó ở bước $t$ và $\mathbf{x}$ là tham số mô hình. Ký hiệu

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

là rủi ro kỳ vọng và $R^*$ là cực tiểu của nó theo $\mathbf{x}$. Cuối cùng, gọi $\mathbf{x}^*$ là điểm cực tiểu (ta giả sử nó tồn tại trong miền xác định của $\mathbf{x}$). Trong trường hợp này, ta có thể theo dõi khoảng cách giữa tham số hiện tại $\mathbf{x}_t$ tại thời điểm $t$ và điểm cực tiểu rủi ro $\mathbf{x}^*$ và xem nó có cải thiện theo thời gian không:

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

Ta giả sử chuẩn $\ell_2$ của gradient ngẫu nhiên $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$ bị chặn bởi một hằng số $L$ nào đó, do đó ta có

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`

Ta chủ yếu quan tâm đến việc khoảng cách giữa $\mathbf{x}_t$ và $\mathbf{x}^*$ thay đổi thế nào *về kỳ vọng*. Thực ra, với bất kỳ chuỗi bước cụ thể nào, khoảng cách hoàn toàn có thể tăng, tùy thuộc vào $\boldsymbol{\xi}_t$ mà ta gặp. Do đó ta cần chặn tích vô hướng. Vì với bất kỳ hàm lồi $f$ nào, ta có $f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$ với mọi $\mathbf{x}$ và $\mathbf{y}$, theo tính lồi ta có

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

Thay cả hai bất đẳng thức :eqref:`eq_sgd-L` và :eqref:`eq_sgd-f-xi-xstar` vào :eqref:`eq_sgd-xt+1-xstar`, ta có được một cận cho khoảng cách giữa các tham số tại thời điểm $t+1$ như sau:

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

Điều này nghĩa là ta có tiến triển miễn là hiệu giữa mất mát hiện tại và mất mát tối ưu lớn hơn $\eta_t L^2/2$. Vì hiệu này chắc chắn hội tụ về không, suy ra tốc độ học $\eta_t$ cũng cần phải *tiến về không*.

Tiếp theo ta lấy kỳ vọng hai vế của :eqref:`eqref_sgd-xt-diff`. Điều này cho ta

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

Bước cuối cùng là lấy tổng các bất đẳng thức với $t \in \{1, \ldots, T\}$. Vì tổng có dạng triệt tiêu dần (telescoping) và bỏ đi số hạng dưới, ta được

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

Lưu ý rằng ta đã tận dụng việc $\mathbf{x}_1$ được cho trước và do đó có thể bỏ kỳ vọng. Cuối cùng, định nghĩa

$$\bar{\mathbf{x}} \stackrel{\textrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

Vì

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

theo bất đẳng thức Jensen (đặt $i=t$, $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$ trong :eqref:`eq_jensens-inequality`) và tính lồi của $R$, suy ra $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$, do đó

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

Thay điều này vào bất đẳng thức :eqref:`eq_sgd-x1-xstar` cho ta cận

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

trong đó $r^2 \stackrel{\textrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$ là một cận cho khoảng cách giữa lựa chọn tham số ban đầu và kết quả cuối cùng. Tóm lại, tốc độ hội tụ phụ thuộc vào việc chuẩn của gradient ngẫu nhiên bị chặn đến đâu ($L$) và giá trị tham số ban đầu cách điểm tối ưu bao xa ($r$). Lưu ý rằng cận được phát biểu theo $\bar{\mathbf{x}}$ thay vì $\mathbf{x}_T$. Đó là vì $\bar{\mathbf{x}}$ là một phiên bản làm trơn của đường đi tối ưu. Bất cứ khi nào biết $r, L$ và $T$, ta có thể chọn tốc độ học $\eta = r/(L \sqrt{T})$. Điều này cho cận trên $rL/\sqrt{T}$. Tức là, ta hội tụ với tốc độ $\mathcal{O}(1/\sqrt{T})$ về nghiệm tối ưu.

## Gradient Ngẫu nhiên và Mẫu Hữu hạn

Cho đến nay ta đã hơi tùy tiện khi nói về hạ gradient ngẫu nhiên. Ta đã giả định rằng ta lấy các mẫu $x_i$, thường kèm nhãn $y_i$, từ một phân phối $p(x, y)$ nào đó và dùng chúng để cập nhật tham số mô hình theo một cách nào đó. Cụ thể, với kích thước mẫu hữu hạn, ta chỉ đơn giản lập luận rằng phân phối rời rạc $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$ với các hàm $\delta_{x_i}$ và $\delta_{y_i}$ nào đó cho phép ta thực hiện hạ gradient ngẫu nhiên trên nó.

Tuy nhiên, đây không thực sự là những gì ta đã làm. Trong các ví dụ đơn giản ở mục này, ta chỉ đơn giản thêm nhiễu vào một gradient vốn không ngẫu nhiên, tức là ta giả vờ có các cặp $(x_i, y_i)$. Hóa ra điều này là hợp lý ở đây (xem phần bài tập để có thảo luận chi tiết). Đáng lo hơn là trong mọi thảo luận trước đây, rõ ràng ta không làm điều này. Thay vào đó, ta duyệt qua mọi mẫu *đúng một lần*. Để thấy vì sao cách này được ưa chuộng hơn, hãy xét trường hợp ngược lại, tức là ta lấy $n$ quan sát từ phân phối rời rạc *có hoàn lại*. Xác suất chọn ngẫu nhiên một phần tử $i$ là $1/n$. Do đó xác suất chọn nó *ít nhất* một lần là

$$P(\textrm{chọn~} i) = 1 - P(\textrm{bỏ~qua~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

Một lập luận tương tự cho thấy xác suất chọn một mẫu nào đó (tức một mẫu huấn luyện) *đúng một lần* được cho bởi

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

Lấy mẫu có hoàn lại dẫn đến phương sai tăng và hiệu quả dữ liệu giảm so với lấy mẫu *không hoàn lại*. Do đó, trong thực tế ta làm theo cách sau (và đây là lựa chọn mặc định xuyên suốt cuốn sách này). Lưu ý cuối cùng là các lượt duyệt lặp lại qua tập dữ liệu huấn luyện duyệt nó theo một thứ tự ngẫu nhiên *khác nhau*.

## Tóm tắt

* Với các bài toán lồi, ta có thể chứng minh rằng với nhiều lựa chọn tốc độ học, hạ gradient ngẫu nhiên sẽ hội tụ về nghiệm tối ưu.
* Với học sâu, nói chung điều này không đúng. Tuy nhiên, phân tích các bài toán lồi cho ta cái nhìn hữu ích về cách tiếp cận tối ưu, cụ thể là giảm tốc độ học dần dần, dù không quá nhanh.
* Vấn đề xảy ra khi tốc độ học quá nhỏ hoặc quá lớn. Trong thực tế, một tốc độ học phù hợp thường chỉ được tìm thấy sau nhiều thí nghiệm.
* Khi tập dữ liệu huấn luyện có nhiều mẫu hơn, mỗi vòng lặp của hạ gradient tốn nhiều chi phí tính toán hơn, nên hạ gradient ngẫu nhiên được ưa chuộng hơn trong những trường hợp này.
* Các đảm bảo về tính tối ưu cho hạ gradient ngẫu nhiên nói chung không có trong các trường hợp không lồi, vì số cực tiểu địa phương cần kiểm tra hoàn toàn có thể tăng theo hàm mũ.

## Bài tập

1. Thử nghiệm với các lịch tốc độ học khác nhau cho hạ gradient ngẫu nhiên và với số vòng lặp khác nhau. Cụ thể, vẽ khoảng cách đến nghiệm tối ưu $(0, 0)$ theo số vòng lặp.
1. Chứng minh rằng với hàm $f(x_1, x_2) = x_1^2 + 2 x_2^2$, thêm nhiễu chuẩn vào gradient tương đương với việc cực tiểu hóa một hàm mất mát $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ trong đó $\mathbf{x}$ được lấy từ một phân phối chuẩn.
1. So sánh sự hội tụ của hạ gradient ngẫu nhiên khi bạn lấy mẫu từ $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ có hoàn lại và khi lấy mẫu không hoàn lại.
1. Bạn sẽ thay đổi bộ giải hạ gradient ngẫu nhiên như thế nào nếu một gradient nào đó (hay đúng hơn là một tọa độ nào đó của nó) luôn lớn hơn mọi gradient khác?
1. Giả sử $f(x) = x^2 (1 + \sin x)$. $f$ có bao nhiêu cực tiểu địa phương? Bạn có thể thay đổi $f$ sao cho để cực tiểu hóa nó, người ta cần đánh giá mọi cực tiểu địa phương không?

[Discussions](https://discuss.d2l.ai/t/497)
