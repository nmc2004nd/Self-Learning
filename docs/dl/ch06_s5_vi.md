# 6.5 Tầng Tùy chỉnh

> Ghi chú của người biên soạn: trang này dịch nguyên văn từ sách *Dive into Deep Learning* của **Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola** ([bản gốc](https://d2l.ai/chapter_builders-guide/custom-layer.html), [mã nguồn](https://github.com/d2l-ai/d2l-en), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); code theo giấy phép MIT-0). Chỉ giữ code PyTorch. Số mục, hình và công thức giữ như bản HTML d2l.ai; thuật ngữ đối chiếu với bản dịch [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn).
>
> **Đính chính** *(người biên soạn)*:
>
> - Mục 6.5.2: trong `MyLinear.forward`, code của sách dùng `self.weight.data` và `self.bias.data`. Thuộc tính `.data` trả về tensor đã **tách khỏi đồ thị tính toán**, nên autograd không tính gradient cho hai tham số này và tầng sẽ không học được gì khi huấn luyện. Muốn dùng tầng này để huấn luyện, hãy viết `torch.matmul(X, self.weight) + self.bias`. Trang [Code từ đầu](dl/ch06_code_vi) của chương kiểm chứng điều này.

Một yếu tố đằng sau thành công của học sâu là sự sẵn có của nhiều loại tầng có thể được kết hợp theo những cách sáng tạo để thiết kế các kiến trúc phù hợp với nhiều tác vụ khác nhau. Chẳng hạn, các nhà nghiên cứu đã phát minh ra những tầng chuyên dùng để xử lý ảnh, văn bản, lặp qua dữ liệu tuần tự, và thực hiện quy hoạch động. Sớm hay muộn, bạn sẽ cần một tầng chưa có sẵn trong framework học sâu. Trong những trường hợp này, bạn phải tự xây dựng một tầng tùy chỉnh. Trong mục này, chúng tôi sẽ chỉ cho bạn cách làm.

<!-- run -->
```python
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## 6.5.1 Tầng không có Tham số

Để bắt đầu, ta dựng một tầng tùy chỉnh không có tham số nào của riêng nó. Điều này sẽ trông quen thuộc nếu bạn nhớ lại phần giới thiệu về mô-đun trong [Mục 6.1](dl/ch06_s1_vi). Lớp `CenteredLayer` sau đây chỉ đơn giản trừ đi giá trị trung bình khỏi đầu vào. Để xây dựng nó, ta chỉ cần kế thừa lớp tầng cơ sở và cài đặt hàm lan truyền xuôi.

<!-- run -->
```python
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

Hãy kiểm chứng rằng tầng của ta hoạt động như dự định bằng cách đưa một số dữ liệu qua nó.

<!-- run -->
```python
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

<!-- output -->
```output
tensor([-2., -1.,  0.,  1.,  2.])
```
<!-- /output -->

Giờ ta có thể đưa tầng này vào làm một thành phần khi xây dựng những mô hình phức tạp hơn.

<!-- run -->
```python
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

Để kiểm tra thêm cho chắc, ta có thể đưa dữ liệu ngẫu nhiên qua mạng và kiểm tra rằng giá trị trung bình thực sự bằng 0. Vì đang làm việc với số thực dấu phẩy động, ta vẫn có thể thấy một số khác không rất nhỏ do sai số lượng tử hóa.

<!-- run -->
```python
Y = net(d2l.rand(4, 8))
Y.mean()
```

<!-- output -->
```output
tensor(-5.5879e-09, grad_fn=<MeanBackward0>)
```
<!-- /output -->

## 6.5.2 Tầng có Tham số

Giờ khi đã biết cách định nghĩa những tầng đơn giản, hãy chuyển sang định nghĩa các tầng có tham số có thể được điều chỉnh qua quá trình huấn luyện. Ta có thể dùng các hàm dựng sẵn để tạo tham số, chúng cung cấp một số chức năng quản lý cơ bản. Cụ thể, chúng quản lý việc truy cập, khởi tạo, dùng chung, lưu và nạp các tham số mô hình. Bằng cách này, cùng với những lợi ích khác, ta sẽ không cần viết các thủ tục tuần tự hóa riêng cho mỗi tầng tùy chỉnh.

Giờ hãy cài đặt phiên bản tầng kết nối đầy đủ của riêng mình. Hãy nhớ lại rằng tầng này cần hai tham số, một để biểu diễn trọng số và một cho hệ số điều chỉnh. Trong cách cài đặt này, ta gắn sẵn hàm kích hoạt ReLU làm mặc định. Tầng này cần hai đối số đầu vào: `in_units` và `units`, lần lượt biểu thị số đầu vào và số đầu ra.

<!-- run -->
```python
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

<!-- run -->
```python
linear = MyLinear(5, 3)
linear.weight
```

<!-- output -->
```output
Parameter containing:
tensor([[-0.0779, -1.4050, -0.6110],
        [ 0.1102, -0.1466, -1.1137],
        [ 1.4848, -0.8964,  0.1973],
        [ 0.0380, -0.4680, -0.1763],
        [-0.6293, -0.7359, -0.4859]], requires_grad=True)
```
<!-- /output -->

Ta có thể trực tiếp thực hiện các phép tính lan truyền xuôi bằng tầng tùy chỉnh.

<!-- run -->
```python
linear(torch.rand(2, 5))
```

<!-- output -->
```output
tensor([[1.2739, 0.0000, 0.0000],
        [1.0255, 0.0000, 0.0000]])
```
<!-- /output -->

Ta cũng có thể xây dựng mô hình bằng các tầng tùy chỉnh. Khi đã có tầng đó, ta có thể dùng nó giống như tầng kết nối đầy đủ dựng sẵn.

<!-- run -->
```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

<!-- output -->
```output
tensor([[0.0000],
        [1.8711]])
```
<!-- /output -->

## 6.5.3 Tóm tắt

Ta có thể thiết kế các tầng tùy chỉnh thông qua lớp tầng cơ sở. Điều này cho phép ta định nghĩa những tầng mới linh hoạt, hoạt động khác với mọi tầng có sẵn trong thư viện. Khi đã được định nghĩa, các tầng tùy chỉnh có thể được gọi trong bất kỳ bối cảnh và kiến trúc nào. Các tầng có thể có tham số cục bộ, được tạo thông qua các hàm dựng sẵn.

## 6.5.4 Bài tập

1. Thiết kế một tầng nhận một đầu vào và tính một phép rút gọn tensor, tức là trả về $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Thiết kế một tầng trả về nửa đầu của các hệ số Fourier của dữ liệu.

[Thảo luận (tiếng Anh)](https://discuss.d2l.ai/t/59)
