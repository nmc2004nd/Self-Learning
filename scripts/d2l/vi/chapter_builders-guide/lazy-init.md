# Khởi tạo Trễ
:label:`sec_lazy_init`

Cho đến nay, có vẻ như ta đã thoát được dù khá cẩu thả khi thiết lập mạng. Cụ thể, ta đã làm những việc phản trực giác sau, có vẻ như lẽ ra không thể chạy được:

* Ta định nghĩa kiến trúc mạng mà không chỉ định số chiều đầu vào.
* Ta thêm các tầng mà không chỉ định số chiều đầu ra của tầng trước.
* Ta thậm chí còn "khởi tạo" các tham số trước khi cung cấp đủ thông tin để xác định mô hình nên có bao nhiêu tham số.

Có thể bạn sẽ ngạc nhiên khi code của ta chạy được. Suy cho cùng, framework học sâu không có cách nào biết được số chiều đầu vào của mạng sẽ là bao nhiêu. Mẹo ở đây là framework *trì hoãn việc khởi tạo* (defers initialization), chờ đến lần đầu tiên ta đưa dữ liệu qua mô hình để suy ra kích thước của từng tầng ngay lúc đó.

Sau này, khi làm việc với mạng neuron tích chập, kỹ thuật này sẽ còn tiện lợi hơn nữa vì số chiều đầu vào (chẳng hạn độ phân giải của ảnh) sẽ ảnh hưởng đến số chiều của mỗi tầng tiếp theo. Do đó, khả năng đặt tham số mà không cần biết giá trị của số chiều vào lúc viết code có thể đơn giản hóa đáng kể việc đặc tả và sau đó chỉnh sửa mô hình. Tiếp theo, ta đi sâu hơn vào cơ chế của việc khởi tạo.

```{.python .input}
from d2l import torch as d2l
import torch
from torch import nn
```

Để bắt đầu, hãy khởi tạo một MLP.

```{.python .input}
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

Lúc này, mạng không thể biết được số chiều của trọng số tầng đầu vào vì số chiều đầu vào vẫn chưa được biết.

Do đó framework chưa khởi tạo bất kỳ tham số nào. Ta xác nhận điều này bằng cách thử truy cập các tham số dưới đây.

```{.python .input}
net[0].weight
```

Tiếp theo, hãy đưa dữ liệu qua mạng để framework cuối cùng cũng khởi tạo các tham số.

```{.python .input}
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

Ngay khi biết số chiều đầu vào là 20, framework có thể xác định kích thước ma trận trọng số của tầng thứ nhất bằng cách thay giá trị 20 vào. Sau khi xác định được kích thước của tầng thứ nhất, framework tiếp tục với tầng thứ hai, và cứ thế đi qua đồ thị tính toán cho đến khi biết mọi kích thước. Lưu ý rằng trong trường hợp này, chỉ tầng thứ nhất cần khởi tạo trễ, nhưng framework khởi tạo một cách tuần tự. Khi đã biết kích thước của mọi tham số, framework cuối cùng có thể khởi tạo các tham số.

Phương thức sau đưa các đầu vào giả qua mạng để chạy thử, nhằm suy ra kích thước của mọi tham số rồi sau đó khởi tạo các tham số. Nó sẽ được dùng sau này khi ta không muốn dùng cách khởi tạo ngẫu nhiên mặc định.

```{.python .input}
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

## Tóm tắt

Khởi tạo trễ có thể tiện lợi, cho phép framework tự động suy ra kích thước tham số, giúp dễ dàng chỉnh sửa kiến trúc và loại bỏ một nguồn lỗi thường gặp. Ta có thể đưa dữ liệu qua mô hình để framework cuối cùng khởi tạo các tham số.

## Bài tập

1. Điều gì xảy ra nếu bạn chỉ định số chiều đầu vào cho tầng thứ nhất nhưng không cho các tầng tiếp theo? Việc khởi tạo có diễn ra ngay lập tức không?
1. Điều gì xảy ra nếu bạn chỉ định các số chiều không khớp nhau?
1. Bạn cần làm gì nếu đầu vào có số chiều thay đổi? Gợi ý: hãy xem lại phần ràng buộc tham số.

[Discussions](https://discuss.d2l.ai/t/8092)
