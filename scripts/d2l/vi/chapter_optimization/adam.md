# Adam
:label:`sec_adam`

Trong các thảo luận dẫn đến mục này, ta đã gặp một số kỹ thuật tối ưu hiệu quả. Hãy tóm tắt chi tiết chúng ở đây:

* Ta đã thấy :numref:`sec_sgd` hiệu quả hơn hạ gradient khi giải các bài toán tối ưu, chẳng hạn nhờ khả năng chống chịu vốn có với dữ liệu thừa.
* Ta đã thấy :numref:`sec_minibatch_sgd` mang lại hiệu quả bổ sung đáng kể nhờ vector hóa, dùng các tập quan sát lớn hơn trong một minibatch. Đây là chìa khóa cho xử lý song song hiệu quả trên nhiều máy, nhiều GPU và nói chung.
* :numref:`sec_momentum` thêm một cơ chế tổng hợp lịch sử các gradient quá khứ để tăng tốc hội tụ.
* :numref:`sec_adagrad` dùng phép co giãn theo từng tọa độ để có được một bộ tiền điều kiện hiệu quả về tính toán.
* :numref:`sec_rmsprop` tách rời phép co giãn theo từng tọa độ khỏi việc điều chỉnh tốc độ học.

Adam :cite:`Kingma.Ba.2014` kết hợp tất cả các kỹ thuật này vào một thuật toán học hiệu quả. Đúng như dự đoán, đây là một thuật toán đã trở nên khá phổ biến như một trong những thuật toán tối ưu bền vững và hiệu quả hơn để dùng trong học sâu. Tuy nhiên, nó không phải không có vấn đề. Cụ thể, :cite:`Reddi.Kale.Kumar.2019` chỉ ra rằng có những tình huống Adam có thể phân kỳ do kiểm soát phương sai kém. Trong một công trình tiếp theo, :citet:`Zaheer.Reddi.Sachan.ea.2018` đã đề xuất một bản sửa nhanh cho Adam, gọi là Yogi, giải quyết những vấn đề này. Sẽ nói thêm về điều này sau. Tạm thời hãy xem lại thuật toán Adam.

## Thuật toán

Một trong những thành phần then chốt của Adam là nó dùng trung bình động có trọng số hàm mũ (exponential weighted moving average, còn được gọi là trung bình rò rỉ) để có được ước lượng của cả động lượng lẫn mô-men bậc hai của gradient. Tức là, nó dùng các biến trạng thái

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

Ở đây $\beta_1$ và $\beta_2$ là các tham số trọng số không âm. Các lựa chọn thông dụng cho chúng là $\beta_1 = 0.9$ và $\beta_2 = 0.999$. Tức là, ước lượng phương sai di chuyển *chậm hơn rất nhiều* so với số hạng động lượng. Lưu ý rằng nếu khởi tạo $\mathbf{v}_0 = \mathbf{s}_0 = 0$, ban đầu ta có một độ chệch đáng kể về phía các giá trị nhỏ hơn. Điều này có thể được giải quyết bằng cách dùng thực tế $\sum_{i=0}^{t-1} \beta^i = \frac{1 - \beta^t}{1 - \beta}$ để chuẩn hóa lại các số hạng. Tương ứng, các biến trạng thái đã chuẩn hóa được cho bởi

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \textrm{ và } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

Với các ước lượng phù hợp, giờ ta có thể viết ra các phương trình cập nhật. Trước tiên, ta co giãn lại gradient theo cách rất giống RMSProp để có

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

Khác với RMSProp, bước cập nhật của ta dùng động lượng $\hat{\mathbf{v}}_t$ thay vì chính gradient. Hơn nữa, có một khác biệt nhỏ về hình thức vì phép co giãn dùng $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ thay vì $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$. Cách trước có thể nói là hoạt động tốt hơn một chút trong thực tế, do đó có sự khác biệt so với RMSProp. Thông thường ta chọn $\epsilon = 10^{-6}$ để cân bằng tốt giữa ổn định số và độ trung thực.

Giờ ta đã có mọi mảnh ghép để tính các bước cập nhật. Điều này hơi hụt hẫng một chút, ta có một bước cập nhật đơn giản dạng

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Xem lại thiết kế của Adam, nguồn cảm hứng của nó rất rõ ràng. Động lượng và thang đo hiện rõ trong các biến trạng thái. Định nghĩa khá đặc biệt của chúng buộc ta phải khử chệch các số hạng (điều này có thể được khắc phục bằng một cách khởi tạo và điều kiện cập nhật hơi khác). Thứ hai, việc kết hợp cả hai số hạng khá đơn giản, khi đã có RMSProp. Cuối cùng, tốc độ học tường minh $\eta$ cho phép ta kiểm soát độ dài bước để giải quyết các vấn đề hội tụ.

## Cài đặt

Cài đặt Adam từ đầu không đáng sợ lắm. Để tiện lợi, ta lưu bộ đếm bước thời gian $t$ trong từ điển `hyperparams`. Ngoài ra mọi thứ đều đơn giản.

```{.python .input}
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

Ta đã sẵn sàng dùng Adam để huấn luyện mô hình. Ta dùng tốc độ học $\eta = 0.01$.

```{.python .input}
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

Một cách cài đặt súc tích hơn rất đơn giản vì `adam` là một trong các thuật toán được cung cấp sẵn trong thư viện tối ưu `trainer` của Gluon. Do đó ta chỉ cần truyền các tham số cấu hình cho một cài đặt trong Gluon.

```{.python .input}
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

## Yogi

Một trong những vấn đề của Adam là nó có thể không hội tụ ngay cả trong bối cảnh lồi khi ước lượng mô-men bậc hai trong $\mathbf{s}_t$ bùng nổ. Để khắc phục, :citet:`Zaheer.Reddi.Sachan.ea.2018` đã đề xuất một bước cập nhật (và cách khởi tạo) cải tiến cho $\mathbf{s}_t$. Để hiểu điều gì đang diễn ra, hãy viết lại bước cập nhật của Adam như sau:

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

Bất cứ khi nào $\mathbf{g}_t^2$ có phương sai cao hoặc các bước cập nhật thưa, $\mathbf{s}_t$ có thể quên các giá trị quá khứ quá nhanh. Một cách khắc phục khả dĩ cho điều này là thay $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ bằng $\mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$. Giờ độ lớn của bước cập nhật không còn phụ thuộc vào mức độ sai lệch. Điều này cho ta các bước cập nhật Yogi

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

Các tác giả còn khuyên nên khởi tạo động lượng trên một lô ban đầu lớn hơn thay vì chỉ một ước lượng điểm ban đầu. Ta bỏ qua chi tiết vì chúng không quan trọng với thảo luận và vì ngay cả không có điều này, sự hội tụ vẫn khá tốt.

```{.python .input}
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## Tóm tắt

* Adam kết hợp đặc điểm của nhiều thuật toán tối ưu thành một quy tắc cập nhật khá bền vững.
* Được xây dựng dựa trên RMSProp, Adam cũng dùng EWMA trên gradient ngẫu nhiên theo minibatch.
* Adam dùng hiệu chỉnh độ chệch để bù cho sự khởi đầu chậm khi ước lượng động lượng và mô-men bậc hai.
* Với các gradient có phương sai đáng kể, ta có thể gặp vấn đề về hội tụ. Chúng có thể được khắc phục bằng cách dùng minibatch lớn hơn hoặc chuyển sang một ước lượng cải tiến cho $\mathbf{s}_t$. Yogi cung cấp một phương án thay thế như vậy.

## Bài tập

1. Điều chỉnh tốc độ học, quan sát và phân tích các kết quả thí nghiệm.
1. Bạn có thể viết lại các bước cập nhật động lượng và mô-men bậc hai sao cho không cần hiệu chỉnh độ chệch không?
1. Vì sao bạn cần giảm tốc độ học $\eta$ khi ta hội tụ?
1. Thử dựng một trường hợp mà Adam phân kỳ còn Yogi hội tụ.

[Discussions](https://discuss.d2l.ai/t/1078)
