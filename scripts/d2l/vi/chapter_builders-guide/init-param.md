# Khởi tạo Tham số

Giờ khi đã biết cách truy cập tham số, hãy xem cách khởi tạo chúng cho đúng. Ta đã thảo luận về sự cần thiết của việc khởi tạo đúng cách trong :numref:`sec_numerical_stability`. Framework học sâu cung cấp các cách khởi tạo ngẫu nhiên mặc định cho các tầng của nó. Tuy nhiên, ta thường muốn khởi tạo trọng số theo nhiều quy tắc khác nhau. Framework cung cấp những quy tắc thông dụng nhất, và cũng cho phép tạo bộ khởi tạo tùy chỉnh.

```{.python .input}
import torch
from torch import nn
```

Theo mặc định, PyTorch khởi tạo các ma trận trọng số và hệ số điều chỉnh theo phân phối đều, lấy mẫu từ một khoảng được tính dựa trên số chiều đầu vào và đầu ra. Mô-đun `nn.init` của PyTorch cung cấp nhiều phương thức khởi tạo dựng sẵn.

```{.python .input}
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

## [**Khởi tạo Dựng sẵn**]

Hãy bắt đầu bằng việc gọi các bộ khởi tạo dựng sẵn. Đoạn code dưới đây khởi tạo mọi tham số trọng số như các biến ngẫu nhiên Gauss có độ lệch chuẩn 0.01, trong khi các hệ số điều chỉnh được đặt về không.

```{.python .input}
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

Ta cũng có thể khởi tạo mọi tham số bằng một giá trị hằng số cho trước (chẳng hạn 1).

```{.python .input}
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

[**Ta cũng có thể áp dụng các bộ khởi tạo khác nhau cho những khối nhất định.**] Chẳng hạn, dưới đây ta khởi tạo tầng thứ nhất bằng bộ khởi tạo Xavier và khởi tạo tầng thứ hai bằng giá trị hằng 42.

```{.python .input}
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

### [**Khởi tạo Tùy chỉnh**]

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

```{.python .input}
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

Lưu ý rằng ta luôn có lựa chọn đặt trực tiếp giá trị cho tham số.

```{.python .input}
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

## Tóm tắt

Ta có thể khởi tạo tham số bằng các bộ khởi tạo dựng sẵn và tùy chỉnh.

## Bài tập

Tra cứu tài liệu trực tuyến để biết thêm các bộ khởi tạo dựng sẵn.

[Discussions](https://discuss.d2l.ai/t/8090)
