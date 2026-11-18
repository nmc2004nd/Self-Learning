# Adadelta
:label:`sec_adadelta`

Adadelta là thêm một biến thể nữa của AdaGrad (:numref:`sec_adagrad`). Khác biệt chính nằm ở chỗ nó giảm mức độ thích ứng của tốc độ học theo các tọa độ. Hơn nữa, theo truyền thống, nó được coi là không có tốc độ học vì nó dùng chính lượng thay đổi để hiệu chỉnh cho lượng thay đổi trong tương lai. Thuật toán được đề xuất trong :citet:`Zeiler.2012`. Nó khá đơn giản, xét những gì đã thảo luận về các thuật toán trước cho đến nay.

## Thuật toán

Nói ngắn gọn, Adadelta dùng hai biến trạng thái, $\mathbf{s}_t$ để lưu trung bình rò rỉ của mô-men bậc hai của gradient và $\Delta\mathbf{x}_t$ để lưu trung bình rò rỉ của mô-men bậc hai của lượng thay đổi của chính các tham số trong mô hình. Lưu ý rằng ta dùng ký hiệu và cách đặt tên gốc của các tác giả để tương thích với các công bố và cài đặt khác (không có lý do thực sự nào khác khiến người ta nên dùng các chữ cái Hy Lạp khác nhau để chỉ một tham số có cùng mục đích trong động lượng, Adagrad, RMSProp và Adadelta).

Đây là các chi tiết kỹ thuật của Adadelta. Với tham số của thuật toán này là $\rho$, ta có các bước cập nhật rò rỉ sau, tương tự như :numref:`sec_rmsprop`:

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

Khác biệt so với :numref:`sec_rmsprop` là ta thực hiện cập nhật với gradient đã được co giãn lại $\mathbf{g}_t'$, tức là

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

Vậy gradient đã co giãn lại $\mathbf{g}_t'$ là gì? Ta có thể tính nó như sau:

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

trong đó $\Delta \mathbf{x}_{t-1}$ là trung bình rò rỉ của bình phương các gradient đã co giãn lại $\mathbf{g}_t'$. Ta khởi tạo $\Delta \mathbf{x}_{0}$ bằng $0$ và cập nhật nó ở mỗi bước với $\mathbf{g}_t'$, tức là

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

và $\epsilon$ (một giá trị nhỏ như $10^{-5}$) được thêm vào để duy trì ổn định số.

## Cài đặt

Adadelta cần duy trì hai biến trạng thái cho mỗi biến, $\mathbf{s}_t$ và $\Delta\mathbf{x}_t$. Điều này cho ta cách cài đặt sau.

```{.python .input}
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # Cập nhật tại chỗ qua [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

Chọn $\rho = 0.9$ tương đương với chu kỳ bán rã bằng 10 cho mỗi lần cập nhật tham số. Điều này thường hoạt động khá tốt. Ta có hành vi sau.

```{.python .input}
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

Để cài đặt súc tích, ta chỉ đơn giản dùng thuật toán Adadelta từ các API cấp cao. Điều này cho ta một dòng lệnh duy nhất gọn hơn nhiều.

```{.python .input}
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

## Tóm tắt

* Adadelta không có tham số tốc độ học. Thay vào đó, nó dùng chính tốc độ thay đổi của các tham số để điều chỉnh tốc độ học.
* Adadelta cần hai biến trạng thái để lưu mô-men bậc hai của gradient và của lượng thay đổi trong tham số.
* Adadelta dùng trung bình rò rỉ để duy trì ước lượng liên tục của các thống kê thích hợp.

## Bài tập

1. Điều chỉnh giá trị của $\rho$. Điều gì xảy ra?
1. Chỉ ra cách cài đặt thuật toán mà không dùng $\mathbf{g}_t'$. Vì sao đây có thể là một ý hay?
1. Adadelta có thực sự không cần tốc độ học không? Bạn có thể tìm những bài toán tối ưu làm Adadelta thất bại không?
1. So sánh Adadelta với Adagrad và RMSProp để thảo luận về hành vi hội tụ của chúng.

[Discussions](https://discuss.d2l.ai/t/1076)
