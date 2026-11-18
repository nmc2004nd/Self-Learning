# Quản lý Tham số

Sau khi đã chọn kiến trúc và đặt các siêu tham số, ta chuyển sang vòng lặp huấn luyện, với mục tiêu tìm các giá trị tham số cực tiểu hóa hàm mất mát. Sau khi huấn luyện, ta sẽ cần các tham số này để đưa ra dự đoán trong tương lai. Ngoài ra, đôi khi ta muốn trích xuất các tham số, có thể để tái sử dụng chúng trong một bối cảnh khác, để lưu mô hình ra ổ đĩa nhằm chạy nó trong phần mềm khác, hoặc để xem xét với hy vọng hiểu biết thêm về mặt khoa học.

Hầu hết thời gian, ta có thể bỏ qua những chi tiết vụn vặt về cách khai báo và thao tác với tham số, để các framework học sâu làm phần việc nặng nhọc. Tuy nhiên, khi rời khỏi các kiến trúc xếp chồng gồm những tầng tiêu chuẩn, đôi khi ta sẽ cần đi sâu vào chi tiết khai báo và thao tác với tham số. Trong mục này, ta đề cập đến những nội dung sau:

* Truy cập tham số để gỡ lỗi, chẩn đoán và trực quan hóa.
* Dùng chung tham số giữa các thành phần khác nhau của mô hình.

```{.python .input}
import torch
from torch import nn
```

(**Ta bắt đầu bằng việc tập trung vào một MLP có một tầng ẩn.**)

```{.python .input}
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4))
net(X).shape
```

## [**Truy cập Tham số**]
:label:`subsec_param-access`

Hãy bắt đầu với cách truy cập tham số từ những mô hình mà bạn đã biết.

Khi một mô hình được định nghĩa qua lớp `Sequential`, trước tiên ta có thể truy cập bất kỳ tầng nào bằng cách lấy chỉ số trên mô hình như thể nó là một danh sách. Tham số của mỗi tầng nằm thuận tiện trong thuộc tính của tầng đó.

Ta có thể xem tham số của tầng kết nối đầy đủ thứ hai như sau.

```{.python .input}
net[2].state_dict()
```

Ta có thể thấy tầng kết nối đầy đủ này chứa hai tham số, lần lượt tương ứng với trọng số và hệ số điều chỉnh của tầng.

### [**Truy cập Tham số Cụ thể**]

Lưu ý rằng mỗi tham số được biểu diễn như một thực thể của lớp tham số. Để làm bất cứ điều gì hữu ích với tham số, trước tiên ta cần truy cập các giá trị số bên dưới. Có vài cách để làm điều này. Một số cách đơn giản hơn, số khác tổng quát hơn. Đoạn code sau trích xuất hệ số điều chỉnh từ tầng thứ hai của mạng neuron, trả về một thực thể của lớp tham số, rồi truy cập tiếp giá trị của tham số đó.

```{.python .input}
type(net[2].bias), net[2].bias.data
```

Tham số là những đối tượng phức tạp, chứa giá trị, gradient và các thông tin bổ sung. Đó là lý do ta cần yêu cầu giá trị một cách tường minh.

Ngoài giá trị, mỗi tham số còn cho phép ta truy cập gradient. Vì ta chưa gọi lan truyền ngược cho mạng này, gradient vẫn ở trạng thái ban đầu.

```{.python .input}
net[2].weight.grad == None
```

### [**Truy cập Mọi Tham số Cùng lúc**]

Khi cần thực hiện các thao tác trên mọi tham số, việc truy cập từng tham số một có thể trở nên tẻ nhạt. Tình huống có thể đặc biệt cồng kềnh khi ta làm việc với những mô-đun phức tạp hơn, chẳng hạn các mô-đun lồng nhau, vì ta sẽ cần đệ quy qua toàn bộ cây để trích xuất tham số của từng mô-đun con. Dưới đây ta minh họa việc truy cập tham số của mọi tầng.

```{.python .input}
[(name, param.shape) for name, param in net.named_parameters()]
```

## [**Tham số Ràng buộc**]

Thường thì ta muốn dùng chung tham số giữa nhiều tầng. Hãy xem cách làm điều này một cách gọn gàng. Dưới đây ta cấp phát một tầng kết nối đầy đủ rồi dùng chính các tham số của nó để đặt tham số cho một tầng khác. Ở đây ta cần chạy lan truyền xuôi `net(X)` trước khi truy cập các tham số.

```{.python .input}
# Ta cần đặt tên cho tầng dùng chung để có thể tham chiếu tới các tham số
# của nó
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Kiểm tra xem các tham số có giống nhau không
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Đảm bảo rằng chúng thực sự là cùng một đối tượng chứ không chỉ có cùng
# giá trị
print(net[2].weight.data[0] == net[4].weight.data[0])
```

Ví dụ này cho thấy tham số của tầng thứ hai và tầng thứ ba được ràng buộc với nhau. Chúng không chỉ bằng nhau, mà được biểu diễn bởi chính xác cùng một tensor. Do đó, nếu ta thay đổi một trong hai tham số, tham số còn lại cũng thay đổi theo.

Bạn có thể thắc mắc, khi các tham số bị ràng buộc thì gradient sẽ ra sao? Vì tham số mô hình chứa gradient, gradient của tầng ẩn thứ hai và tầng ẩn thứ ba được cộng lại với nhau trong quá trình lan truyền ngược.

## Tóm tắt

Ta có vài cách để truy cập và ràng buộc các tham số mô hình.

## Bài tập

1. Dùng mô hình `NestMLP` đã định nghĩa trong :numref:`sec_model_construction` và truy cập tham số của các tầng khác nhau.
1. Dựng một MLP chứa một tầng tham số dùng chung và huấn luyện nó. Trong quá trình huấn luyện, hãy quan sát tham số mô hình và gradient của từng tầng.
1. Vì sao dùng chung tham số là một ý tưởng hay?

[Discussions](https://discuss.d2l.ai/t/57)
