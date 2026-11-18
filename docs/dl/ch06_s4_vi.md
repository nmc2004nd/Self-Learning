# 6.4 Khởi tạo Trễ

> Ghi chú của người biên soạn: trang này dịch nguyên văn từ sách *Dive into Deep Learning* của **Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola** ([bản gốc](https://d2l.ai/chapter_builders-guide/lazy-init.html), [mã nguồn](https://github.com/d2l-ai/d2l-en), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); code theo giấy phép MIT-0). Chỉ giữ code PyTorch. Số mục, hình và công thức giữ như bản HTML d2l.ai; thuật ngữ đối chiếu với bản dịch [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn).

Cho đến nay, có vẻ như ta đã thoát được dù khá cẩu thả khi thiết lập mạng. Cụ thể, ta đã làm những việc phản trực giác sau, có vẻ như lẽ ra không thể chạy được:

* Ta định nghĩa kiến trúc mạng mà không chỉ định số chiều đầu vào.
* Ta thêm các tầng mà không chỉ định số chiều đầu ra của tầng trước.
* Ta thậm chí còn "khởi tạo" các tham số trước khi cung cấp đủ thông tin để xác định mô hình nên có bao nhiêu tham số.

Có thể bạn sẽ ngạc nhiên khi code của ta chạy được. Suy cho cùng, framework học sâu không có cách nào biết được số chiều đầu vào của mạng sẽ là bao nhiêu. Mẹo ở đây là framework *trì hoãn việc khởi tạo* (defers initialization), chờ đến lần đầu tiên ta đưa dữ liệu qua mô hình để suy ra kích thước của từng tầng ngay lúc đó.

Sau này, khi làm việc với mạng neuron tích chập, kỹ thuật này sẽ còn tiện lợi hơn nữa vì số chiều đầu vào (chẳng hạn độ phân giải của ảnh) sẽ ảnh hưởng đến số chiều của mỗi tầng tiếp theo. Do đó, khả năng đặt tham số mà không cần biết giá trị của số chiều vào lúc viết code có thể đơn giản hóa đáng kể việc đặc tả và sau đó chỉnh sửa mô hình. Tiếp theo, ta đi sâu hơn vào cơ chế của việc khởi tạo.

<!-- run -->
```python
from d2l import torch as d2l
import torch
from torch import nn
```

Để bắt đầu, hãy khởi tạo một MLP.

<!-- run -->
```python
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

Lúc này, mạng không thể biết được số chiều của trọng số tầng đầu vào vì số chiều đầu vào vẫn chưa được biết.

Do đó framework chưa khởi tạo bất kỳ tham số nào. Ta xác nhận điều này bằng cách thử truy cập các tham số dưới đây.

<!-- run -->
```python
net[0].weight
```

<!-- output -->
```output
<UninitializedParameter>
```
<!-- /output -->

Tiếp theo, hãy đưa dữ liệu qua mạng để framework cuối cùng cũng khởi tạo các tham số.

<!-- run -->
```python
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

<!-- output -->
```output
torch.Size([256, 20])
```
<!-- /output -->

Ngay khi biết số chiều đầu vào là 20, framework có thể xác định kích thước ma trận trọng số của tầng thứ nhất bằng cách thay giá trị 20 vào. Sau khi xác định được kích thước của tầng thứ nhất, framework tiếp tục với tầng thứ hai, và cứ thế đi qua đồ thị tính toán cho đến khi biết mọi kích thước. Lưu ý rằng trong trường hợp này, chỉ tầng thứ nhất cần khởi tạo trễ, nhưng framework khởi tạo một cách tuần tự. Khi đã biết kích thước của mọi tham số, framework cuối cùng có thể khởi tạo các tham số.

Phương thức sau đưa các đầu vào giả qua mạng để chạy thử, nhằm suy ra kích thước của mọi tham số rồi sau đó khởi tạo các tham số. Nó sẽ được dùng sau này khi ta không muốn dùng cách khởi tạo ngẫu nhiên mặc định.

<!-- run -->
```python
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

## 6.4.1 Tóm tắt

Khởi tạo trễ có thể tiện lợi, cho phép framework tự động suy ra kích thước tham số, giúp dễ dàng chỉnh sửa kiến trúc và loại bỏ một nguồn lỗi thường gặp. Ta có thể đưa dữ liệu qua mô hình để framework cuối cùng khởi tạo các tham số.

## 6.4.2 Bài tập

1. Điều gì xảy ra nếu bạn chỉ định số chiều đầu vào cho tầng thứ nhất nhưng không cho các tầng tiếp theo? Việc khởi tạo có diễn ra ngay lập tức không?
1. Điều gì xảy ra nếu bạn chỉ định các số chiều không khớp nhau?
1. Bạn cần làm gì nếu đầu vào có số chiều thay đổi? Gợi ý: hãy xem lại phần ràng buộc tham số.

[Thảo luận (tiếng Anh)](https://discuss.d2l.ai/t/8092)
