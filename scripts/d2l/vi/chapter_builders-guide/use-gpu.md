# GPU
:label:`sec_use_gpu`

Trong :numref:`tab_intro_decade`, chúng tôi đã minh họa sự tăng trưởng nhanh chóng của năng lực tính toán trong hai thập kỷ qua. Nói ngắn gọn, hiệu năng GPU đã tăng 1000 lần mỗi thập kỷ kể từ năm 2000. Điều này mang lại những cơ hội to lớn nhưng cũng cho thấy đã có nhu cầu đáng kể về hiệu năng như vậy.

Trong mục này, ta bắt đầu thảo luận cách khai thác hiệu năng tính toán này cho nghiên cứu của bạn. Trước tiên là dùng một GPU, và sau này là cách dùng nhiều GPU và nhiều máy chủ (mỗi máy có nhiều GPU).

Cụ thể, ta sẽ thảo luận cách dùng một GPU NVIDIA để tính toán. Trước tiên, hãy đảm bảo bạn đã lắp ít nhất một GPU NVIDIA. Sau đó, tải [driver NVIDIA và CUDA](https://developer.nvidia.com/cuda-downloads) và làm theo hướng dẫn để đặt đường dẫn thích hợp. Khi các bước chuẩn bị này hoàn tất, lệnh `nvidia-smi` có thể được dùng để (**xem thông tin card đồ họa**).

Trong PyTorch, mỗi mảng có một thiết bị (device); ta thường gọi nó là một *ngữ cảnh* (context). Cho đến nay, theo mặc định, mọi biến và phép tính liên quan đều được gán cho CPU. Thông thường, các ngữ cảnh khác có thể là các GPU khác nhau. Mọi thứ còn có thể rắc rối hơn khi ta triển khai công việc trên nhiều máy chủ. Bằng cách gán mảng cho các ngữ cảnh một cách thông minh, ta có thể giảm thiểu thời gian truyền dữ liệu giữa các thiết bị. Chẳng hạn, khi huấn luyện mạng neuron trên một máy chủ có GPU, ta thường muốn các tham số của mô hình nằm trên GPU.

Để chạy các chương trình trong mục này, bạn cần ít nhất hai GPU. Lưu ý rằng điều này có thể là xa xỉ với hầu hết máy tính để bàn nhưng lại dễ dàng có được trên đám mây, chẳng hạn bằng cách dùng các máy ảo nhiều GPU của AWS EC2. Hầu như mọi mục khác đều *không* cần nhiều GPU, nhưng ở đây chúng tôi chỉ muốn minh họa luồng dữ liệu giữa các thiết bị khác nhau.

```{.python .input}
from d2l import torch as d2l
import torch
from torch import nn
```

## [**Thiết bị Tính toán**]

Ta có thể chỉ định các thiết bị, như CPU và GPU, để lưu trữ và tính toán. Theo mặc định, tensor được tạo trong bộ nhớ chính và sau đó CPU được dùng để tính toán.

Trong PyTorch, CPU và GPU có thể được biểu thị bằng `torch.device('cpu')` và `torch.device('cuda')`. Cần lưu ý rằng thiết bị `cpu` nghĩa là toàn bộ CPU vật lý và bộ nhớ. Điều này có nghĩa là các phép tính của PyTorch sẽ cố gắng dùng mọi nhân CPU. Tuy nhiên, một thiết bị `gpu` chỉ đại diện cho một card và bộ nhớ tương ứng của nó. Nếu có nhiều GPU, ta dùng `torch.device(f'cuda:{i}')` để biểu diễn GPU thứ $i$ ($i$ bắt đầu từ 0). Ngoài ra, `gpu:0` và `gpu` là tương đương.

```{.python .input}
def cpu():  #@save
    """Lấy thiết bị CPU."""
    return torch.device('cpu')

def gpu(i=0):  #@save
    """Lấy một thiết bị GPU."""
    return torch.device(f'cuda:{i}')

cpu(), gpu(), gpu(1)
```

Ta có thể (**truy vấn số GPU sẵn có.**)

```{.python .input .gpu}
def num_gpus():  #@save
    """Lấy số GPU sẵn có."""
    return torch.cuda.device_count()

num_gpus()
```

Giờ ta [**định nghĩa hai hàm tiện lợi cho phép chạy code ngay cả khi các GPU được yêu cầu không tồn tại.**]

```{.python .input .gpu}
def try_gpu(i=0):  #@save
    """Trả về gpu(i) nếu tồn tại, ngược lại trả về cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Trả về mọi GPU sẵn có, hoặc [cpu(),] nếu không có GPU nào."""
    return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()
```

## Tensor và GPU

Theo mặc định, tensor được tạo trên CPU. Ta có thể [**truy vấn thiết bị nơi tensor nằm.**]

```{.python .input}
x = torch.tensor([1, 2, 3])
x.device
```

Điều quan trọng cần lưu ý là bất cứ khi nào ta muốn thao tác trên nhiều số hạng, chúng cần nằm trên cùng một thiết bị. Chẳng hạn, nếu ta cộng hai tensor, ta cần đảm bảo cả hai đối số nằm trên cùng một thiết bị, nếu không framework sẽ không biết lưu kết quả ở đâu hay thậm chí quyết định thực hiện phép tính ở đâu.

### Lưu trữ trên GPU

Có vài cách để [**lưu một tensor trên GPU.**] Chẳng hạn, ta có thể chỉ định thiết bị lưu trữ khi tạo tensor. Tiếp theo, ta tạo biến tensor `X` trên `gpu` thứ nhất. Tensor được tạo trên một GPU chỉ tiêu tốn bộ nhớ của GPU đó. Ta có thể dùng lệnh `nvidia-smi` để xem mức sử dụng bộ nhớ GPU. Nói chung, ta cần đảm bảo không tạo dữ liệu vượt quá giới hạn bộ nhớ GPU.

```{.python .input .gpu}
X = torch.ones(2, 3, device=try_gpu())
X
```

Giả sử bạn có ít nhất hai GPU, đoạn code sau sẽ (**tạo một tensor ngẫu nhiên `Y` trên GPU thứ hai.**)

```{.python .input .gpu}
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

### Sao chép

[**Nếu muốn tính `X + Y`, ta cần quyết định thực hiện phép toán này ở đâu.**] Chẳng hạn, như minh họa trong :numref:`fig_copyto`, ta có thể chuyển `X` sang GPU thứ hai và thực hiện phép toán ở đó. *Đừng* đơn giản cộng `X` với `Y`, vì điều này sẽ gây ra ngoại lệ. Bộ máy thực thi sẽ không biết phải làm gì: nó không thể tìm thấy dữ liệu trên cùng một thiết bị và sẽ thất bại. Vì `Y` nằm trên GPU thứ hai, ta cần chuyển `X` sang đó trước khi có thể cộng hai tensor.

![Sao chép dữ liệu để thực hiện phép toán trên cùng một thiết bị.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input .norun}
Z = X.cuda(1)
print(X)
print(Z)
```

Giờ khi [**dữ liệu (cả `Z` và `Y`) đã ở trên cùng một GPU, ta có thể cộng chúng lại.**]

```{.python .input .norun}
Y + Z
```

Nhưng nếu biến `Z` của bạn đã nằm sẵn trên GPU thứ hai thì sao? Điều gì xảy ra nếu ta vẫn gọi `Z.cuda(1)`? Nó sẽ trả về `Z` thay vì tạo một bản sao và cấp phát bộ nhớ mới.

```{.python .input .norun}
Z.cuda(1) is Z
```

### Lưu ý Bên lề

Người ta dùng GPU để làm học máy vì họ mong đợi chúng chạy nhanh. Nhưng truyền biến giữa các thiết bị thì chậm: chậm hơn nhiều so với tính toán. Vì vậy chúng tôi muốn bạn chắc chắn 100% rằng bạn muốn làm một việc chậm trước khi cho phép bạn làm việc đó. Nếu framework học sâu cứ tự động sao chép mà không báo lỗi, có thể bạn sẽ không nhận ra rằng mình đã viết một đoạn code chậm.

Truyền dữ liệu không chỉ chậm, nó còn khiến việc song song hóa khó hơn nhiều, vì ta phải chờ dữ liệu được gửi đi (hay đúng hơn là được nhận) trước khi có thể tiếp tục với các phép toán khác. Đây là lý do các thao tác sao chép cần được thực hiện hết sức cẩn thận. Theo kinh nghiệm, nhiều phép toán nhỏ tệ hơn nhiều so với một phép toán lớn. Hơn nữa, thực hiện vài phép toán cùng lúc tốt hơn nhiều so với nhiều phép toán đơn lẻ rải rác trong code, trừ khi bạn biết rõ mình đang làm gì. Đó là vì những phép toán như vậy có thể bị chặn nếu một thiết bị phải chờ thiết bị kia trước khi có thể làm việc khác. Điều này hơi giống việc xếp hàng gọi cà phê thay vì đặt trước qua điện thoại và thấy nó đã sẵn sàng khi bạn đến.

Cuối cùng, khi ta in tensor hoặc chuyển tensor sang định dạng NumPy, nếu dữ liệu không nằm trong bộ nhớ chính, framework sẽ sao chép nó vào bộ nhớ chính trước, gây thêm chi phí truyền dữ liệu. Tệ hơn nữa, lúc này nó phải chịu sự chi phối của khóa thông dịch toàn cục (global interpreter lock) đáng sợ, khiến mọi thứ phải chờ Python hoàn tất.

## [**Mạng Neuron và GPU**]

Tương tự, một mô hình mạng neuron có thể chỉ định thiết bị. Đoạn code sau đặt các tham số mô hình lên GPU.

```{.python .input .gpu}
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())
```

Ta sẽ thấy nhiều ví dụ hơn nữa về cách chạy mô hình trên GPU trong các chương tiếp theo, đơn giản vì các mô hình sẽ trở nên đòi hỏi tính toán nhiều hơn.

Chẳng hạn, khi đầu vào là một tensor trên GPU, mô hình sẽ tính kết quả trên cùng GPU đó.

```{.python .input .gpu}
net(X)
```

Hãy (**xác nhận rằng các tham số mô hình được lưu trên cùng GPU.**)

```{.python .input .gpu}
net[0].weight.data.device
```

Cho bộ huấn luyện hỗ trợ GPU.

```{.python .input}
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        model.to(self.gpus[0])
    self.model = model
```

Nói ngắn gọn, miễn là mọi dữ liệu và tham số nằm trên cùng một thiết bị, ta có thể học mô hình một cách hiệu quả. Trong các chương tiếp theo, ta sẽ thấy vài ví dụ như vậy.

## Tóm tắt

Ta có thể chỉ định thiết bị để lưu trữ và tính toán, như CPU hoặc GPU. Theo mặc định, dữ liệu được tạo trong bộ nhớ chính và sau đó dùng CPU để tính toán. Framework học sâu yêu cầu mọi dữ liệu đầu vào cho phép tính nằm trên cùng một thiết bị, dù là CPU hay cùng một GPU. Bạn có thể mất đáng kể hiệu năng nếu di chuyển dữ liệu một cách thiếu cẩn thận. Một lỗi điển hình như sau: tính mất mát cho mỗi minibatch trên GPU và báo cáo lại cho người dùng trên dòng lệnh (hoặc ghi lại vào một `ndarray` của NumPy) sẽ kích hoạt khóa thông dịch toàn cục, làm đình trệ mọi GPU. Tốt hơn nhiều là cấp phát bộ nhớ để ghi log ngay bên trong GPU và chỉ di chuyển những bản log lớn hơn.

## Bài tập

1. Thử một tác vụ tính toán lớn hơn, chẳng hạn nhân các ma trận lớn, và xem sự khác biệt về tốc độ giữa CPU và GPU. Còn với một tác vụ có ít phép tính thì sao?
1. Ta nên đọc và ghi tham số mô hình trên GPU như thế nào?
1. Đo thời gian để tính 1000 phép nhân ma trận–ma trận với các ma trận $100 \times 100$ và ghi lại chuẩn Frobenius của ma trận đầu ra, mỗi lần một kết quả. So sánh với việc giữ log trên GPU và chỉ truyền kết quả cuối cùng.
1. Đo thời gian thực hiện hai phép nhân ma trận–ma trận trên hai GPU cùng lúc. So sánh với việc tính lần lượt trên một GPU. Gợi ý: bạn sẽ thấy khả năng mở rộng gần như tuyến tính.

[Discussions](https://discuss.d2l.ai/t/63)
