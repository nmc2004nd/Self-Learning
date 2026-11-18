# 6.3 Khởi tạo Tham số

> Ghi chú của người biên soạn: trang này dịch nguyên văn từ sách *Dive into Deep Learning* của **Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola** ([bản gốc](https://d2l.ai/chapter_builders-guide/init-param.html), [mã nguồn](https://github.com/d2l-ai/d2l-en), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); code theo giấy phép MIT-0). Chỉ giữ code PyTorch. Số mục, hình và công thức giữ như bản HTML d2l.ai; thuật ngữ đối chiếu với bản dịch [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn).

Giờ khi đã biết cách truy cập tham số, hãy xem cách khởi tạo chúng cho đúng. Ta đã thảo luận về sự cần thiết của việc khởi tạo đúng cách trong [Mục 5.4](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html). Framework học sâu cung cấp các cách khởi tạo ngẫu nhiên mặc định cho các tầng của nó. Tuy nhiên, ta thường muốn khởi tạo trọng số theo nhiều quy tắc khác nhau. Framework cung cấp những quy tắc thông dụng nhất, và cũng cho phép tạo bộ khởi tạo tùy chỉnh.

<!-- run -->
```python
import torch
from torch import nn
```

Theo mặc định, PyTorch khởi tạo các ma trận trọng số và hệ số điều chỉnh theo phân phối đều, lấy mẫu từ một khoảng được tính dựa trên số chiều đầu vào và đầu ra. Mô-đun `nn.init` của PyTorch cung cấp nhiều phương thức khởi tạo dựng sẵn.

<!-- run -->
```python
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

<!-- output -->
```output
torch.Size([2, 1])
```
<!-- /output -->

## 6.3.1 Khởi tạo Dựng sẵn

Hãy bắt đầu bằng việc gọi các bộ khởi tạo dựng sẵn. Đoạn code dưới đây khởi tạo mọi tham số trọng số như các biến ngẫu nhiên Gauss có độ lệch chuẩn 0.01, trong khi các hệ số điều chỉnh được đặt về không.

<!-- run -->
```python
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

<!-- output -->
```output
(tensor([0.0104, 0.0021, 0.0146, 0.0045]), tensor(0.))
```
<!-- /output -->

Ta cũng có thể khởi tạo mọi tham số bằng một giá trị hằng số cho trước (chẳng hạn 1).

<!-- run -->
```python
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

<!-- output -->
```output
(tensor([1., 1., 1., 1.]), tensor(0.))
```
<!-- /output -->

Ta cũng có thể áp dụng các bộ khởi tạo khác nhau cho những khối nhất định. Chẳng hạn, dưới đây ta khởi tạo tầng thứ nhất bằng bộ khởi tạo Xavier và khởi tạo tầng thứ hai bằng giá trị hằng 42.

<!-- run -->
```python
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

<!-- output -->
```output
tensor([-0.3564, -0.5286,  0.0459, -0.1269])
tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
```
<!-- /output -->

### 6.3.1.1 Khởi tạo Tùy chỉnh

Đôi khi, framework học sâu không cung cấp phương thức khởi tạo mà ta cần. Trong ví dụ dưới đây, ta định nghĩa một bộ khởi tạo cho mọi tham số trọng số $w$ theo phân phối kỳ lạ sau:

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ với xác suất } \frac{1}{4} \\
            0    & \textrm{ với xác suất } \frac{1}{2} \\
        U(-10, -5) & \textrm{ với xác suất } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

Một lần nữa, ta cài đặt một hàm `my_init` để áp dụng cho `net`.

<!-- run -->
```python
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

<!-- output -->
```output
Init weight torch.Size([8, 4])
Init weight torch.Size([1, 8])
tensor([[-0.0000,  0.0000, -0.0000,  8.2826],
        [-0.0000, -8.4362,  6.4450, -0.0000]], grad_fn=<SliceBackward0>)
```
<!-- /output -->

Lưu ý rằng ta luôn có lựa chọn đặt trực tiếp giá trị cho tham số.

<!-- run -->
```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

<!-- output -->
```output
tensor([42.0000,  1.0000,  1.0000,  9.2826])
```
<!-- /output -->

## 6.3.2 Tóm tắt

Ta có thể khởi tạo tham số bằng các bộ khởi tạo dựng sẵn và tùy chỉnh.

## 6.3.3 Bài tập

Tra cứu tài liệu trực tuyến để biết thêm các bộ khởi tạo dựng sẵn.

[Thảo luận (tiếng Anh)](https://discuss.d2l.ai/t/8090)
