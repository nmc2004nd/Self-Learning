# Hạ Gradient Ngẫu nhiên theo Minibatch
:label:`sec_minibatch_sgd`

Cho đến nay ta đã gặp hai thái cực trong cách tiếp cận học dựa trên gradient: :numref:`sec_gd` dùng toàn bộ tập dữ liệu để tính gradient và cập nhật tham số, mỗi lần một lượt duyệt. Ngược lại, :numref:`sec_sgd` xử lý mỗi lần một mẫu huấn luyện để tiến triển. Mỗi cách đều có nhược điểm riêng. Hạ gradient không đặc biệt *hiệu quả về dữ liệu* mỗi khi dữ liệu rất giống nhau. Hạ gradient ngẫu nhiên không đặc biệt *hiệu quả về tính toán* vì CPU và GPU không thể khai thác hết sức mạnh của vector hóa. Điều này gợi ý rằng có thể có thứ gì đó ở giữa, và thực ra đó chính là những gì ta đã dùng cho đến nay trong các ví dụ đã thảo luận.

## Vector hóa và Bộ nhớ đệm

Cốt lõi của quyết định dùng minibatch là hiệu quả tính toán. Điều này dễ hiểu nhất khi xét việc song song hóa trên nhiều GPU và nhiều máy chủ. Trong trường hợp này, ta cần gửi ít nhất một ảnh đến mỗi GPU. Với 8 GPU mỗi máy chủ và 16 máy chủ, ta đã có kích thước minibatch không nhỏ hơn 128.

Mọi thứ tinh tế hơn một chút khi nói đến một GPU đơn lẻ hoặc thậm chí CPU. Những thiết bị này có nhiều loại bộ nhớ, thường có nhiều loại đơn vị tính toán và các ràng buộc băng thông khác nhau giữa chúng. Chẳng hạn, một CPU có một số ít thanh ghi rồi đến bộ nhớ đệm L1, L2, và trong một số trường hợp thậm chí cả L3 (dùng chung giữa các nhân xử lý). Các bộ nhớ đệm này có kích thước và độ trễ tăng dần (đồng thời băng thông giảm dần). Chỉ cần nói rằng bộ xử lý có khả năng thực hiện nhiều phép tính hơn rất nhiều so với những gì giao diện bộ nhớ chính có thể cung cấp.

Thứ nhất, một CPU 2GHz với 16 nhân và vector hóa AVX-512 có thể xử lý đến $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ byte mỗi giây. Năng lực của GPU dễ dàng vượt con số này 100 lần. Mặt khác, một bộ xử lý máy chủ tầm trung có thể không có nhiều hơn 100 GB/s băng thông, tức là chưa đến một phần mười những gì cần thiết để cung cấp đủ dữ liệu cho bộ xử lý. Tệ hơn nữa, không phải mọi lần truy cập bộ nhớ đều như nhau: giao diện bộ nhớ thường rộng 64 bit hoặc hơn (chẳng hạn trên GPU lên tới 384 bit), do đó đọc một byte duy nhất phải chịu chi phí của một lần truy cập rộng hơn nhiều.

Thứ hai, có chi phí đáng kể cho lần truy cập đầu tiên trong khi truy cập tuần tự tương đối rẻ (điều này thường được gọi là đọc theo loạt – burst read). Còn nhiều điều khác cần lưu ý, như bộ nhớ đệm khi ta có nhiều socket, chiplet và các cấu trúc khác. Xem [bài viết Wikipedia này](https://en.wikipedia.org/wiki/Cache_hierarchy) để có thảo luận sâu hơn.

Cách để giảm bớt những ràng buộc này là dùng một hệ thống phân cấp bộ nhớ đệm CPU thực sự đủ nhanh để cung cấp dữ liệu cho bộ xử lý. Đây *chính* là động lực đằng sau việc chia lô trong học sâu. Để giữ mọi thứ đơn giản, hãy xét phép nhân ma trận–ma trận, chẳng hạn $\mathbf{A} = \mathbf{B}\mathbf{C}$. Ta có một số lựa chọn để tính $\mathbf{A}$. Chẳng hạn, ta có thể thử những cách sau:

1. Ta có thể tính $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}$, tức là tính theo từng phần tử bằng các tích vô hướng.
1. Ta có thể tính $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}$, tức là tính mỗi lần một cột. Tương tự, ta có thể tính $\mathbf{A}$ mỗi lần một hàng $\mathbf{A}_{i,:}$.
1. Ta có thể đơn giản tính $\mathbf{A} = \mathbf{B} \mathbf{C}$.
1. Ta có thể chia $\mathbf{B}$ và $\mathbf{C}$ thành các ma trận khối nhỏ hơn và tính $\mathbf{A}$ mỗi lần một khối.

Nếu làm theo lựa chọn thứ nhất, ta sẽ cần sao chép một vector hàng và một vector cột vào CPU mỗi lần muốn tính một phần tử $\mathbf{A}_{ij}$. Tệ hơn nữa, do các phần tử ma trận được sắp xếp tuần tự, ta phải truy cập nhiều vị trí rời rạc đối với một trong hai vector khi đọc chúng từ bộ nhớ. Lựa chọn thứ hai thuận lợi hơn nhiều. Trong đó, ta có thể giữ vector cột $\mathbf{C}_{:,j}$ trong bộ nhớ đệm CPU trong khi tiếp tục duyệt qua $\mathbf{B}$. Điều này giảm một nửa yêu cầu băng thông bộ nhớ, với tốc độ truy cập nhanh hơn tương ứng. Tất nhiên, lựa chọn 3 là đáng mong muốn nhất. Đáng tiếc là hầu hết các ma trận có thể không vừa hoàn toàn trong bộ nhớ đệm (suy cho cùng, đó là điều ta đang thảo luận). Tuy nhiên, lựa chọn 4 đưa ra một phương án thay thế hữu ích trong thực tế: ta có thể chuyển các khối của ma trận vào bộ nhớ đệm và nhân chúng tại chỗ. Các thư viện tối ưu sẽ lo việc này cho ta. Hãy xem những phép toán này hiệu quả đến đâu trong thực tế.

Ngoài hiệu quả tính toán, chi phí do Python và bản thân framework học sâu đưa vào là đáng kể. Hãy nhớ lại rằng mỗi lần ta thực thi một lệnh, trình thông dịch Python gửi một lệnh đến bộ máy MXNet, bộ máy này cần chèn nó vào đồ thị tính toán và xử lý nó khi lập lịch. Chi phí như vậy có thể khá tai hại. Tóm lại, rất nên dùng vector hóa (và ma trận) bất cứ khi nào có thể.

```{.python .input}
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import time
import torch
from torch import nn

A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

Vì ta sẽ thường xuyên đo thời gian chạy trong phần còn lại của cuốn sách, hãy định nghĩa một bộ đếm giờ.

```{.python .input}
class Timer:  #@save
    """Ghi lại nhiều lần thời gian chạy."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Bắt đầu bộ đếm giờ."""
        self.tik = time.time()

    def stop(self):
        """Dừng bộ đếm giờ và ghi thời gian vào một danh sách."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Trả về thời gian trung bình."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Trả về tổng thời gian."""
        return sum(self.times)

    def cumsum(self):
        """Trả về thời gian cộng dồn."""
        return np.array(self.times).cumsum().tolist()

timer = Timer()
```

Gán theo từng phần tử chỉ đơn giản duyệt qua mọi hàng của $\mathbf{B}$ và mọi cột của $\mathbf{C}$ để gán giá trị cho $\mathbf{A}$.

```{.python .input}
# Tính A = BC mỗi lần một phần tử
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

Một chiến lược nhanh hơn là gán theo từng cột.

```{.python .input}
# Tính A = BC mỗi lần một cột
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

Cuối cùng, cách hiệu quả nhất là thực hiện toàn bộ phép toán trong một khối. Lưu ý rằng nhân hai ma trận bất kỳ $\mathbf{B} \in \mathbb{R}^{m \times n}$ và $\mathbf{C} \in \mathbb{R}^{n \times p}$ tốn khoảng $2mnp$ phép tính dấu phẩy động, khi phép nhân và phép cộng vô hướng được đếm như những phép toán riêng biệt (trong thực tế được gộp làm một). Do đó, nhân hai ma trận $256 \times 256$ tốn $0.03$ tỷ phép tính dấu phẩy động. Hãy xem tốc độ tương ứng của các phép toán.

```{.python .input}
# Tính A = BC một lần cho tất cả
timer.start()
A = torch.mm(B, C)
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## Minibatch

:label:`sec_minibatches`

Trước đây ta coi việc đọc các *minibatch* dữ liệu thay vì từng quan sát đơn lẻ để cập nhật tham số là điều hiển nhiên. Giờ ta đưa ra một lời giải thích ngắn gọn cho điều đó. Xử lý từng quan sát đơn lẻ đòi hỏi ta thực hiện nhiều phép nhân ma trận–vector (hoặc thậm chí vector–vector) đơn lẻ, khá tốn kém và gây ra chi phí đáng kể cho framework học sâu bên dưới. Điều này áp dụng cả khi đánh giá một mạng trên dữ liệu (thường được gọi là suy luận) lẫn khi tính gradient để cập nhật tham số. Tức là, điều này áp dụng bất cứ khi nào ta thực hiện $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$, trong đó

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

Ta có thể tăng hiệu quả *tính toán* của phép toán này bằng cách áp dụng nó cho một minibatch quan sát mỗi lần. Tức là, ta thay gradient $\mathbf{g}_t$ trên một quan sát đơn lẻ bằng gradient trên một lô nhỏ

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

Hãy xem điều này tác động thế nào đến các tính chất thống kê của $\mathbf{g}_t$: vì cả $\mathbf{x}_t$ lẫn mọi phần tử của minibatch $\mathcal{B}_t$ đều được lấy ngẫu nhiên đều từ tập huấn luyện, kỳ vọng của gradient giữ nguyên. Mặt khác, phương sai giảm đáng kể. Vì gradient minibatch gồm $b \stackrel{\textrm{def}}{=} |\mathcal{B}_t|$ gradient độc lập được lấy trung bình, độ lệch chuẩn của nó giảm đi $b^{-\frac{1}{2}}$ lần. Bản thân điều này là tốt, vì nó nghĩa là các bước cập nhật khớp với gradient đầy đủ một cách đáng tin cậy hơn.

Một cách ngây thơ, điều này cho thấy chọn một minibatch $\mathcal{B}_t$ lớn sẽ luôn đáng mong muốn. Than ôi, sau một ngưỡng nào đó, mức giảm thêm của độ lệch chuẩn là rất nhỏ so với mức tăng tuyến tính của chi phí tính toán. Trong thực tế, ta chọn một minibatch đủ lớn để mang lại hiệu quả tính toán tốt trong khi vẫn vừa với bộ nhớ của GPU. Để minh họa sự tiết kiệm, hãy xem một đoạn code. Trong đó ta thực hiện cùng phép nhân ma trận–ma trận, nhưng lần này chia thành các "minibatch" gồm 64 cột mỗi lần.

```{.python .input}
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

Như ta có thể thấy, việc tính toán trên minibatch về cơ bản hiệu quả như trên toàn bộ ma trận. Cần một lời cảnh báo. Trong :numref:`sec_batch_norm`, ta đã dùng một loại điều chuẩn phụ thuộc nhiều vào lượng phương sai trong một minibatch. Khi ta tăng kích thước minibatch, phương sai giảm và cùng với đó là lợi ích của việc đưa nhiễu vào nhờ chuẩn hóa theo batch. Xem chẳng hạn :citet:`Ioffe.2017` để biết chi tiết về cách co giãn lại và tính các số hạng phù hợp.

## Đọc Tập dữ liệu

Hãy xem cách các minibatch được tạo ra từ dữ liệu một cách hiệu quả. Dưới đây ta dùng một tập dữ liệu do NASA phát triển để kiểm tra [tiếng ồn cánh của các loại máy bay khác nhau](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) để so sánh các thuật toán tối ưu này. Để tiện lợi, ta chỉ dùng $1500$ mẫu đầu tiên. Dữ liệu được làm trắng (whitening) khi tiền xử lý, tức là ta trừ đi trung bình và co giãn phương sai về $1$ cho mỗi tọa độ.

```{.python .input}
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## Cài đặt từ Đầu

Hãy nhớ lại cách cài đặt hạ gradient ngẫu nhiên theo minibatch từ :numref:`sec_linear_scratch`. Dưới đây ta đưa ra một cách cài đặt tổng quát hơn một chút. Để tiện lợi, nó có cùng chữ ký lời gọi với các thuật toán tối ưu khác được giới thiệu sau trong chương này. Cụ thể, ta thêm đầu vào trạng thái `states` và đặt siêu tham số vào từ điển `hyperparams`. Ngoài ra, ta sẽ lấy trung bình mất mát của mỗi mẫu trong minibatch trong hàm huấn luyện, nên gradient trong thuật toán tối ưu không cần chia cho kích thước batch.

```{.python .input}
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

Tiếp theo, ta cài đặt một hàm huấn luyện tổng quát để tiện dùng với các thuật toán tối ưu khác được giới thiệu sau trong chương này. Nó khởi tạo một mô hình hồi quy tuyến tính và có thể được dùng để huấn luyện mô hình bằng hạ gradient ngẫu nhiên theo minibatch và các thuật toán khác được giới thiệu sau đó.

```{.python .input}
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Khởi tạo
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Huấn luyện
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

Hãy xem quá trình tối ưu diễn ra thế nào với hạ gradient theo toàn bộ lô. Điều này có thể đạt được bằng cách đặt kích thước minibatch bằng 1500 (tức là bằng tổng số mẫu). Kết quả là các tham số mô hình chỉ được cập nhật một lần mỗi epoch. Tiến triển rất ít. Thực ra, sau 6 bước, tiến trình bị đình trệ.

```{.python .input}
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

Khi kích thước batch bằng 1, ta dùng hạ gradient ngẫu nhiên để tối ưu. Để cài đặt đơn giản, ta chọn một tốc độ học không đổi (dù nhỏ). Trong hạ gradient ngẫu nhiên, các tham số mô hình được cập nhật mỗi khi một mẫu được xử lý. Trong trường hợp của ta, điều này tương đương 1500 lần cập nhật mỗi epoch. Như ta có thể thấy, giá trị hàm mục tiêu giảm chậm lại sau một epoch. Dù cả hai quy trình đều xử lý 1500 mẫu trong một epoch, hạ gradient ngẫu nhiên tốn nhiều thời gian hơn hạ gradient trong thí nghiệm của ta. Đó là vì hạ gradient ngẫu nhiên cập nhật tham số thường xuyên hơn và vì xử lý từng quan sát đơn lẻ một kém hiệu quả hơn.

```{.python .input}
sgd_res = train_sgd(0.005, 1)
```

Cuối cùng, khi kích thước batch bằng 100, ta dùng hạ gradient ngẫu nhiên theo minibatch để tối ưu. Thời gian cần cho mỗi epoch ngắn hơn thời gian cần cho hạ gradient ngẫu nhiên và thời gian cho hạ gradient theo toàn bộ lô.

```{.python .input}
mini1_res = train_sgd(.4, 100)
```

Giảm kích thước batch xuống 10, thời gian cho mỗi epoch tăng lên vì khối lượng công việc của mỗi batch được thực thi kém hiệu quả hơn.

```{.python .input}
mini2_res = train_sgd(.05, 10)
```

Giờ ta có thể so sánh thời gian và mất mát của bốn thí nghiệm trước. Như có thể thấy, dù hạ gradient ngẫu nhiên hội tụ nhanh hơn GD xét theo số mẫu đã xử lý, nó tốn nhiều thời gian hơn để đạt cùng mức mất mát so với GD vì tính gradient theo từng mẫu không hiệu quả bằng. Hạ gradient ngẫu nhiên theo minibatch có thể cân bằng giữa tốc độ hội tụ và hiệu quả tính toán. Kích thước minibatch 10 hiệu quả hơn hạ gradient ngẫu nhiên; kích thước minibatch 100 thậm chí còn vượt trội GD về thời gian chạy.

```{.python .input}
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## Cài đặt Súc tích

Trong Gluon, ta có thể dùng lớp `Trainer` để gọi các thuật toán tối ưu. Điều này được dùng để cài đặt một hàm huấn luyện tổng quát. Ta sẽ dùng nó xuyên suốt chương này.

```{.python .input}
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Khởi tạo
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` tính bình phương sai số không có hệ số 1/2
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

Dùng Gluon để lặp lại thí nghiệm cuối cùng cho thấy hành vi giống hệt.

```{.python .input}
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

## Tóm tắt

* Vector hóa làm code hiệu quả hơn nhờ giảm chi phí phát sinh từ framework học sâu và nhờ tính cục bộ bộ nhớ cùng bộ nhớ đệm tốt hơn trên CPU và GPU.
* Có một sự đánh đổi giữa hiệu quả thống kê từ hạ gradient ngẫu nhiên và hiệu quả tính toán từ việc xử lý các lô dữ liệu lớn mỗi lần.
* Hạ gradient ngẫu nhiên theo minibatch mang lại ưu điểm của cả hai: hiệu quả tính toán và hiệu quả thống kê.
* Trong hạ gradient ngẫu nhiên theo minibatch, ta xử lý các lô dữ liệu thu được từ một hoán vị ngẫu nhiên của dữ liệu huấn luyện (tức là mỗi quan sát chỉ được xử lý một lần mỗi epoch, dù theo thứ tự ngẫu nhiên).
* Nên giảm dần tốc độ học trong quá trình huấn luyện.
* Nói chung, hạ gradient ngẫu nhiên theo minibatch nhanh hơn hạ gradient ngẫu nhiên và hạ gradient trong việc hội tụ về rủi ro nhỏ hơn, khi đo bằng thời gian thực.

## Bài tập

1. Thay đổi kích thước batch và tốc độ học, quan sát tốc độ giảm của giá trị hàm mục tiêu và thời gian tiêu tốn cho mỗi epoch.
1. Đọc tài liệu MXNet và dùng hàm `set_learning_rate` của lớp `Trainer` để giảm tốc độ học của hạ gradient ngẫu nhiên theo minibatch xuống 1/10 giá trị trước đó sau mỗi epoch.
1. So sánh hạ gradient ngẫu nhiên theo minibatch với một biến thể thực sự *lấy mẫu có hoàn lại* từ tập huấn luyện. Điều gì xảy ra?
1. Một vị thần đèn độc ác sao chép tập dữ liệu của bạn mà không cho bạn biết (tức là mỗi quan sát xuất hiện hai lần và tập dữ liệu của bạn lớn gấp đôi kích thước ban đầu, nhưng không ai nói với bạn). Hành vi của hạ gradient ngẫu nhiên, hạ gradient ngẫu nhiên theo minibatch và hạ gradient thay đổi như thế nào?

[Discussions](https://discuss.d2l.ai/t/1068)
