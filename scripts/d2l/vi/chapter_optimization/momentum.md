# Động lượng
:label:`sec_momentum`

Trong :numref:`sec_sgd`, ta đã xem xét điều gì xảy ra khi thực hiện hạ gradient ngẫu nhiên, tức là khi tối ưu mà chỉ có một phiên bản bị nhiễu của gradient. Cụ thể, ta nhận thấy rằng với gradient bị nhiễu, ta cần đặc biệt thận trọng khi chọn tốc độ học trước nhiễu. Nếu giảm nó quá nhanh, sự hội tụ bị đình trệ. Nếu quá dễ dãi, ta không hội tụ được về một nghiệm đủ tốt vì nhiễu cứ đẩy ta ra xa điểm tối ưu.

## Kiến thức Cơ bản

Trong mục này, ta sẽ tìm hiểu các thuật toán tối ưu hiệu quả hơn, đặc biệt cho một số loại bài toán tối ưu thường gặp trong thực tế.

### Trung bình Rò rỉ

Mục trước đã thảo luận về SGD theo minibatch như một phương tiện để tăng tốc tính toán. Nó cũng có tác dụng phụ tốt là việc lấy trung bình gradient làm giảm phương sai. Hạ gradient ngẫu nhiên theo minibatch có thể được tính bằng:

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

Để giữ ký hiệu đơn giản, ở đây ta dùng $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$ là gradient ngẫu nhiên cho mẫu $i$ dùng trọng số được cập nhật tại thời điểm $t-1$. Sẽ thật tốt nếu ta có thể hưởng lợi từ hiệu ứng giảm phương sai ngay cả vượt ra ngoài việc lấy trung bình gradient trên một minibatch. Một lựa chọn để làm được điều này là thay phép tính gradient bằng một "trung bình rò rỉ" (leaky average):

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

với một $\beta \in (0, 1)$ nào đó. Về thực chất, điều này thay gradient tức thời bằng một gradient đã được lấy trung bình trên nhiều gradient *quá khứ*. $\mathbf{v}$ được gọi là *vận tốc* (velocity). Nó tích lũy các gradient quá khứ, tương tự như cách một quả bóng nặng lăn xuống địa hình của hàm mục tiêu tích hợp các lực trong quá khứ. Để xem chi tiết hơn điều gì đang diễn ra, hãy khai triển đệ quy $\mathbf{v}_t$ thành

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

$\beta$ lớn tương đương với một trung bình dài hạn, trong khi $\beta$ nhỏ chỉ tương đương với một hiệu chỉnh nhẹ so với phương pháp gradient. Gradient thay thế mới không còn chỉ theo hướng giảm dốc nhất trên một mẫu cụ thể nữa mà theo hướng của một trung bình có trọng số của các gradient quá khứ. Điều này cho phép ta có được phần lớn lợi ích của việc lấy trung bình trên một lô mà không phải trả chi phí thực sự tính gradient trên lô đó. Ta sẽ quay lại quy trình lấy trung bình này chi tiết hơn sau.

Lập luận trên đã tạo nền tảng cho những gì ngày nay được gọi là các phương pháp gradient *tăng tốc* (accelerated), như gradient với động lượng (momentum). Chúng có thêm lợi ích là hiệu quả hơn nhiều trong những trường hợp bài toán tối ưu có điều kiện kém (ill-conditioned, tức là có một số hướng mà tiến triển chậm hơn nhiều so với các hướng khác, giống một hẻm núi hẹp). Hơn nữa, chúng cho phép ta lấy trung bình trên các gradient liên tiếp để có được hướng giảm ổn định hơn. Thật vậy, khía cạnh tăng tốc ngay cả với các bài toán lồi không nhiễu là một trong những lý do then chốt vì sao động lượng hoạt động và vì sao nó hoạt động tốt đến vậy.

Đúng như dự đoán, nhờ hiệu quả của nó, động lượng là một chủ đề được nghiên cứu kỹ trong tối ưu cho học sâu và rộng hơn. Xem chẳng hạn [bài viết giải thích](https://distill.pub/2017/momentum/) tuyệt đẹp của :citet:`Goh.2017` để có phân tích sâu và hoạt hình tương tác. Nó được đề xuất bởi :citet:`Polyak.1964`. :citet:`Nesterov.2018` có một thảo luận lý thuyết chi tiết trong bối cảnh tối ưu lồi. Động lượng trong học sâu từ lâu đã được biết là có lợi. Xem chẳng hạn thảo luận của :citet:`Sutskever.Martens.Dahl.ea.2013` để biết chi tiết.

### Một Bài toán có Điều kiện Kém

Để hiểu rõ hơn các tính chất hình học của phương pháp động lượng, ta xem lại hạ gradient, dù với một hàm mục tiêu kém dễ chịu hơn đáng kể. Hãy nhớ lại rằng trong :numref:`sec_gd`, ta đã dùng $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, tức là một hàm mục tiêu dạng elipxoit bị méo vừa phải. Ta làm méo hàm này thêm nữa bằng cách kéo giãn nó theo hướng $x_1$ thông qua

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Như trước, $f$ có cực tiểu tại $(0, 0)$. Hàm này *rất* phẳng theo hướng $x_1$. Hãy xem điều gì xảy ra khi ta thực hiện hạ gradient như trước trên hàm mới này. Ta chọn tốc độ học $0.4$.

```{.python .input}
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

Theo cách xây dựng, gradient theo hướng $x_2$ cao hơn *rất nhiều* và thay đổi nhanh hơn nhiều so với theo hướng ngang $x_1$. Do đó ta bị kẹt giữa hai lựa chọn không mong muốn: nếu chọn tốc độ học nhỏ, ta đảm bảo nghiệm không phân kỳ theo hướng $x_2$ nhưng phải chịu sự hội tụ chậm theo hướng $x_1$. Ngược lại, với tốc độ học lớn, ta tiến nhanh theo hướng $x_1$ nhưng phân kỳ theo $x_2$. Ví dụ dưới đây minh họa điều gì xảy ra ngay cả sau khi tăng nhẹ tốc độ học từ $0.4$ lên $0.6$. Sự hội tụ theo hướng $x_1$ được cải thiện nhưng chất lượng nghiệm tổng thể tệ hơn nhiều.

```{.python .input}
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### Phương pháp Động lượng

Phương pháp động lượng cho phép ta giải quyết vấn đề hạ gradient mô tả ở trên. Nhìn vào quỹ đạo tối ưu ở trên, ta có thể đoán rằng lấy trung bình các gradient trong quá khứ sẽ hoạt động tốt. Suy cho cùng, theo hướng $x_1$, điều này sẽ cộng gộp các gradient cùng hướng, qua đó tăng quãng đường ta đi được với mỗi bước. Ngược lại, theo hướng $x_2$ nơi gradient dao động, một gradient tổng hợp sẽ làm giảm kích thước bước do các dao động triệt tiêu lẫn nhau. Dùng $\mathbf{v}_t$ thay cho gradient $\mathbf{g}_t$ cho ta các phương trình cập nhật sau:

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

Lưu ý rằng với $\beta = 0$, ta thu lại hạ gradient thông thường. Trước khi đi sâu hơn vào các tính chất toán học, hãy xem nhanh thuật toán hoạt động thế nào trong thực tế.

```{.python .input}
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Như ta có thể thấy, ngay cả với cùng tốc độ học đã dùng trước đó, động lượng vẫn hội tụ tốt. Hãy xem điều gì xảy ra khi ta giảm tham số động lượng. Giảm một nửa xuống $\beta = 0.25$ dẫn đến một quỹ đạo gần như không hội tụ chút nào. Dù vậy, nó vẫn tốt hơn nhiều so với không dùng động lượng (khi đó nghiệm phân kỳ).

```{.python .input}
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Lưu ý rằng ta có thể kết hợp động lượng với hạ gradient ngẫu nhiên và đặc biệt là hạ gradient ngẫu nhiên theo minibatch. Thay đổi duy nhất là trong trường hợp đó ta thay các gradient $\mathbf{g}_{t, t-1}$ bằng $\mathbf{g}_t$. Cuối cùng, để tiện lợi, ta khởi tạo $\mathbf{v}_0 = 0$ tại thời điểm $t=0$. Hãy xem việc lấy trung bình rò rỉ thực sự tác động thế nào đến các bước cập nhật.

### Trọng số Mẫu Hiệu dụng

Hãy nhớ lại rằng $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$. Ở giới hạn, các số hạng cộng lại thành $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$. Nói cách khác, thay vì bước một bước có kích thước $\eta$ như trong hạ gradient hay hạ gradient ngẫu nhiên, ta bước một bước có kích thước $\frac{\eta}{1-\beta}$, đồng thời xử lý một hướng giảm có khả năng tốt hơn nhiều. Đây là hai lợi ích trong một. Để minh họa việc đánh trọng số diễn ra thế nào với các lựa chọn $\beta$ khác nhau, hãy xem biểu đồ dưới đây.

```{.python .input}
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## Thí nghiệm Thực tế

Hãy xem động lượng hoạt động thế nào trong thực tế, tức là khi được dùng trong bối cảnh của một bộ tối ưu thực thụ. Để làm điều này, ta cần một cách cài đặt có khả năng mở rộng hơn một chút.

### Cài đặt từ Đầu

So với hạ gradient ngẫu nhiên (theo minibatch), phương pháp động lượng cần duy trì một tập biến phụ, tức là vận tốc. Nó có cùng kích thước với gradient (và với các biến của bài toán tối ưu). Trong cách cài đặt dưới đây, ta gọi các biến này là `states`.

```{.python .input}
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

Hãy xem điều này hoạt động thế nào trong thực tế.

```{.python .input}
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

Khi tăng siêu tham số động lượng `momentum` lên 0.9, nó tương đương với một kích thước mẫu hiệu dụng lớn hơn đáng kể là $\frac{1}{1 - 0.9} = 10$. Ta giảm nhẹ tốc độ học xuống $0.01$ để giữ mọi thứ trong tầm kiểm soát.

```{.python .input}
train_momentum(0.01, 0.9)
```

Giảm tốc độ học thêm nữa giải quyết mọi vấn đề của các bài toán tối ưu không trơn. Đặt nó bằng $0.005$ cho các tính chất hội tụ tốt.

```{.python .input}
train_momentum(0.005, 0.9)
```

### Cài đặt Súc tích

Có rất ít việc phải làm trong Gluon vì bộ giải `sgd` tiêu chuẩn đã tích hợp sẵn động lượng. Đặt các tham số tương ứng cho ra một quỹ đạo rất giống.

```{.python .input}
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

## Phân tích Lý thuyết

Cho đến giờ, ví dụ hai chiều $f(x) = 0.1 x_1^2 + 2 x_2^2$ có vẻ khá gượng ép. Giờ ta sẽ thấy rằng nó thực ra khá đại diện cho các loại bài toán mà ta có thể gặp, ít nhất là trong trường hợp cực tiểu hóa các hàm mục tiêu bậc hai lồi.

### Hàm Bậc hai Lồi

Xét hàm số

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

Đây là một hàm bậc hai tổng quát. Với các ma trận xác định dương $\mathbf{Q} \succ 0$, tức là các ma trận có trị riêng dương, hàm này có điểm cực tiểu tại $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ với giá trị nhỏ nhất $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Do đó ta có thể viết lại $h$ là

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

Gradient được cho bởi $\partial_{\mathbf{x}} h(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$. Tức là, nó được cho bởi khoảng cách giữa $\mathbf{x}$ và điểm cực tiểu, nhân với $\mathbf{Q}$. Do đó vận tốc cũng là một tổ hợp tuyến tính của các số hạng $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$.

Vì $\mathbf{Q}$ xác định dương, nó có thể được phân tích thành hệ riêng của nó qua $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ với một ma trận trực giao (phép quay) $\mathbf{O}$ và một ma trận đường chéo $\boldsymbol{\Lambda}$ gồm các trị riêng dương. Điều này cho phép ta đổi biến từ $\mathbf{x}$ sang $\mathbf{z} \stackrel{\textrm{def}}{=} \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ để có được một biểu thức đơn giản hơn nhiều:

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

Ở đây $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Vì $\mathbf{O}$ chỉ là một ma trận trực giao nên điều này không làm nhiễu loạn gradient theo cách đáng kể nào. Biểu diễn theo $\mathbf{z}$, hạ gradient trở thành

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

Điều quan trọng trong biểu thức này là hạ gradient *không trộn lẫn* giữa các không gian riêng khác nhau. Tức là, khi được biểu diễn theo hệ riêng của $\mathbf{Q}$, bài toán tối ưu diễn ra theo từng tọa độ. Điều này cũng đúng với

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

Làm như vậy, ta vừa chứng minh định lý sau: hạ gradient có và không có động lượng cho một hàm bậc hai lồi phân tách thành việc tối ưu theo từng tọa độ theo hướng các vector riêng của ma trận bậc hai.

### Hàm Vô hướng

Với kết quả trên, hãy xem điều gì xảy ra khi ta cực tiểu hóa hàm $f(x) = \frac{\lambda}{2} x^2$. Với hạ gradient, ta có

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

Bất cứ khi nào $|1 - \eta \lambda| < 1$, quá trình tối ưu này hội tụ với tốc độ hàm mũ vì sau $t$ bước ta có $x_t = (1 - \eta \lambda)^t x_0$. Điều này cho thấy tốc độ hội tụ cải thiện ban đầu thế nào khi ta tăng tốc độ học $\eta$ cho đến khi $\eta \lambda = 1$. Vượt quá đó, mọi thứ bắt đầu dao động và với $\eta \lambda > 2$, bài toán tối ưu phân kỳ.

```{.python .input}
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

Để phân tích sự hội tụ trong trường hợp động lượng, ta bắt đầu bằng việc viết lại các phương trình cập nhật theo hai đại lượng vô hướng: một cho $x$ và một cho vận tốc $v$. Điều này cho ta:

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

Ta dùng $\mathbf{R}$ để ký hiệu ma trận $2 \times 2$ chi phối hành vi hội tụ. Sau $t$ bước, lựa chọn ban đầu $[v_0, x_0]$ trở thành $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$. Do đó, các trị riêng của $\mathbf{R}$ quyết định tốc độ hội tụ. Xem [bài viết trên Distill](https://distill.pub/2017/momentum/) của :citet:`Goh.2017` để có hoạt hình tuyệt vời và :citet:`Flammarion.Bach.2015` để có phân tích chi tiết. Có thể chứng minh rằng với $0 < \eta \lambda < 2 + 2 \beta$, vận tốc hội tụ. Đây là một khoảng tham số khả thi lớn hơn so với $0 < \eta \lambda < 2$ của hạ gradient. Nó cũng gợi ý rằng nói chung các giá trị $\beta$ lớn là đáng mong muốn. Các chi tiết sâu hơn đòi hỏi khá nhiều kỹ thuật và chúng tôi gợi ý độc giả quan tâm tham khảo các công bố gốc.

## Tóm tắt

* Động lượng thay gradient bằng một trung bình rò rỉ của các gradient quá khứ. Điều này tăng tốc sự hội tụ đáng kể.
* Nó đáng mong muốn cho cả hạ gradient không nhiễu lẫn hạ gradient ngẫu nhiên (có nhiễu).
* Động lượng ngăn quá trình tối ưu bị đình trệ, điều có khả năng xảy ra cao hơn nhiều với hạ gradient ngẫu nhiên.
* Số gradient hiệu dụng được cho bởi $\frac{1}{1-\beta}$ do việc giảm trọng số theo hàm mũ đối với dữ liệu quá khứ.
* Trong trường hợp các bài toán bậc hai lồi, điều này có thể được phân tích chi tiết một cách tường minh.
* Cài đặt khá đơn giản nhưng đòi hỏi ta lưu thêm một vector trạng thái (vận tốc $\mathbf{v}$).

## Bài tập

1. Dùng các tổ hợp khác của siêu tham số động lượng và tốc độ học, quan sát và phân tích các kết quả thí nghiệm khác nhau.
1. Thử hạ gradient và động lượng cho một bài toán bậc hai có nhiều trị riêng, tức là $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, chẳng hạn $\lambda_i = 2^{-i}$. Vẽ cách các giá trị của $x$ giảm với khởi tạo $x_i = 1$.
1. Suy ra giá trị nhỏ nhất và điểm cực tiểu của $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$.
1. Điều gì thay đổi khi ta thực hiện hạ gradient ngẫu nhiên với động lượng? Điều gì xảy ra khi ta dùng hạ gradient ngẫu nhiên theo minibatch với động lượng? Hãy thử nghiệm với các tham số.

[Discussions](https://discuss.d2l.ai/t/1070)
