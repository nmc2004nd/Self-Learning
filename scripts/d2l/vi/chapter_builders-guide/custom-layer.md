# Tầng Tùy chỉnh

Một yếu tố đằng sau thành công của học sâu là sự sẵn có của nhiều loại tầng có thể được kết hợp theo những cách sáng tạo để thiết kế các kiến trúc phù hợp với nhiều tác vụ khác nhau. Chẳng hạn, các nhà nghiên cứu đã phát minh ra những tầng chuyên dùng để xử lý ảnh, văn bản, lặp qua dữ liệu tuần tự, và thực hiện quy hoạch động. Sớm hay muộn, bạn sẽ cần một tầng chưa có sẵn trong framework học sâu. Trong những trường hợp này, bạn phải tự xây dựng một tầng tùy chỉnh. Trong mục này, chúng tôi sẽ chỉ cho bạn cách làm.

```{.python .input}
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## (**Tầng không có Tham số**)

Để bắt đầu, ta dựng một tầng tùy chỉnh không có tham số nào của riêng nó. Điều này sẽ trông quen thuộc nếu bạn nhớ lại phần giới thiệu về mô-đun trong :numref:`sec_model_construction`. Lớp `CenteredLayer` sau đây chỉ đơn giản trừ đi giá trị trung bình khỏi đầu vào. Để xây dựng nó, ta chỉ cần kế thừa lớp tầng cơ sở và cài đặt hàm lan truyền xuôi.

```{.python .input}
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

Hãy kiểm chứng rằng tầng của ta hoạt động như dự định bằng cách đưa một số dữ liệu qua nó.

```{.python .input}
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

Giờ ta có thể [**đưa tầng này vào làm một thành phần khi xây dựng những mô hình phức tạp hơn.**]

```{.python .input}
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

Để kiểm tra thêm cho chắc, ta có thể đưa dữ liệu ngẫu nhiên qua mạng và kiểm tra rằng giá trị trung bình thực sự bằng 0. Vì đang làm việc với số thực dấu phẩy động, ta vẫn có thể thấy một số khác không rất nhỏ do sai số lượng tử hóa.

```{.python .input}
Y = net(d2l.rand(4, 8))
Y.mean()
```

## [**Tầng có Tham số**]

Giờ khi đã biết cách định nghĩa những tầng đơn giản, hãy chuyển sang định nghĩa các tầng có tham số có thể được điều chỉnh qua quá trình huấn luyện. Ta có thể dùng các hàm dựng sẵn để tạo tham số, chúng cung cấp một số chức năng quản lý cơ bản. Cụ thể, chúng quản lý việc truy cập, khởi tạo, dùng chung, lưu và nạp các tham số mô hình. Bằng cách này, cùng với những lợi ích khác, ta sẽ không cần viết các thủ tục tuần tự hóa riêng cho mỗi tầng tùy chỉnh.

Giờ hãy cài đặt phiên bản tầng kết nối đầy đủ của riêng mình. Hãy nhớ lại rằng tầng này cần hai tham số, một để biểu diễn trọng số và một cho hệ số điều chỉnh. Trong cách cài đặt này, ta gắn sẵn hàm kích hoạt ReLU làm mặc định. Tầng này cần hai đối số đầu vào: `in_units` và `units`, lần lượt biểu thị số đầu vào và số đầu ra.

```{.python .input}
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

Tiếp theo, ta khởi tạo lớp `MyLinear` và truy cập các tham số mô hình của nó.

```{.python .input}
linear = MyLinear(5, 3)
linear.weight
```

Ta có thể [**trực tiếp thực hiện các phép tính lan truyền xuôi bằng tầng tùy chỉnh.**]

```{.python .input}
linear(torch.rand(2, 5))
```

Ta cũng có thể (**xây dựng mô hình bằng các tầng tùy chỉnh.**) Khi đã có tầng đó, ta có thể dùng nó giống như tầng kết nối đầy đủ dựng sẵn.

```{.python .input}
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

## Tóm tắt

Ta có thể thiết kế các tầng tùy chỉnh thông qua lớp tầng cơ sở. Điều này cho phép ta định nghĩa những tầng mới linh hoạt, hoạt động khác với mọi tầng có sẵn trong thư viện. Khi đã được định nghĩa, các tầng tùy chỉnh có thể được gọi trong bất kỳ bối cảnh và kiến trúc nào. Các tầng có thể có tham số cục bộ, được tạo thông qua các hàm dựng sẵn.

## Bài tập

1. Thiết kế một tầng nhận một đầu vào và tính một phép rút gọn tensor, tức là trả về $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Thiết kế một tầng trả về nửa đầu của các hệ số Fourier của dữ liệu.

[Discussions](https://discuss.d2l.ai/t/59)
