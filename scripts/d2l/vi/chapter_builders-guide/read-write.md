# Đọc/Ghi Tệp

Cho đến nay ta đã thảo luận cách xử lý dữ liệu và cách xây dựng, huấn luyện, kiểm tra các mô hình học sâu. Tuy nhiên, đến một lúc nào đó, hy vọng là ta sẽ đủ hài lòng với các mô hình đã học để muốn lưu lại kết quả cho việc sử dụng sau này trong nhiều bối cảnh khác nhau (thậm chí có thể để đưa ra dự đoán khi triển khai). Ngoài ra, khi chạy một quá trình huấn luyện dài, cách làm tốt nhất là định kỳ lưu các kết quả trung gian (checkpointing) để đảm bảo không mất công tính toán của vài ngày nếu ta lỡ vấp phải dây nguồn của máy chủ. Vì vậy đã đến lúc học cách nạp và lưu cả từng vector trọng số riêng lẻ lẫn toàn bộ mô hình. Mục này giải quyết cả hai vấn đề.

```{.python .input}
import torch
from torch import nn
from torch.nn import functional as F
```

## (**Nạp và Lưu Tensor**)

Với từng tensor riêng lẻ, ta có thể gọi trực tiếp các hàm `load` và `save` để lần lượt đọc và ghi chúng. Cả hai hàm đều yêu cầu ta cung cấp một cái tên, và `save` yêu cầu đầu vào là biến cần lưu.

```{.python .input}
x = torch.arange(4)
torch.save(x, 'x-file')
```

Giờ ta có thể đọc dữ liệu từ tệp đã lưu trở lại bộ nhớ.

```{.python .input}
x2 = torch.load('x-file')
x2
```

Ta có thể [**lưu một danh sách các tensor và đọc chúng trở lại bộ nhớ.**]

```{.python .input}
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

Ta thậm chí có thể [**ghi và đọc một từ điển ánh xạ từ chuỗi ký tự tới tensor.**] Điều này tiện lợi khi ta muốn đọc hoặc ghi mọi trọng số trong một mô hình.

```{.python .input}
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

## [**Nạp và Lưu Tham số Mô hình**]

Lưu từng vector trọng số (hoặc các tensor khác) là hữu ích, nhưng sẽ rất tẻ nhạt nếu ta muốn lưu (và sau đó nạp) toàn bộ một mô hình. Suy cho cùng, ta có thể có hàng trăm nhóm tham số rải rác khắp nơi. Vì lý do này, framework học sâu cung cấp các chức năng dựng sẵn để nạp và lưu toàn bộ mạng. Một chi tiết quan trọng cần lưu ý là việc này lưu *tham số* của mô hình chứ không phải toàn bộ mô hình. Chẳng hạn, nếu ta có một MLP 3 tầng, ta cần đặc tả kiến trúc riêng. Lý do là bản thân các mô hình có thể chứa code tùy ý, nên chúng không thể được tuần tự hóa một cách tự nhiên như vậy. Do đó, để khôi phục một mô hình, ta cần tạo kiến trúc bằng code rồi nạp các tham số từ ổ đĩa. (**Hãy bắt đầu với MLP quen thuộc của ta.**)

```{.python .input}
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

Tiếp theo, ta [**lưu các tham số của mô hình thành một tệp**] với tên "mlp.params".

```{.python .input}
torch.save(net.state_dict(), 'mlp.params')
```

Để khôi phục mô hình, ta khởi tạo một bản sao của mô hình MLP ban đầu. Thay vì khởi tạo ngẫu nhiên các tham số mô hình, ta [**đọc trực tiếp các tham số được lưu trong tệp**].

```{.python .input}
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

Vì cả hai thực thể có cùng tham số mô hình, kết quả tính toán với cùng đầu vào `X` phải giống nhau. Hãy kiểm chứng điều này.

```{.python .input}
Y_clone = clone(X)
Y_clone == Y
```

## Tóm tắt

Các hàm `save` và `load` có thể được dùng để đọc/ghi tệp cho các đối tượng tensor. Ta có thể lưu và nạp toàn bộ tập tham số của một mạng thông qua một từ điển tham số. Việc lưu kiến trúc phải được thực hiện bằng code chứ không phải bằng tham số.

## Bài tập

1. Ngay cả khi không cần triển khai mô hình đã huấn luyện sang một thiết bị khác, việc lưu tham số mô hình có những lợi ích thực tế gì?
1. Giả sử ta chỉ muốn tái sử dụng một phần của mạng để đưa vào một mạng có kiến trúc khác. Bạn sẽ làm thế nào để dùng, chẳng hạn, hai tầng đầu tiên của một mạng trước đó trong một mạng mới?
1. Bạn sẽ làm thế nào để lưu cả kiến trúc mạng lẫn tham số? Bạn sẽ đặt ra những hạn chế gì đối với kiến trúc?

[Discussions](https://discuss.d2l.ai/t/61)
