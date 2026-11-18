# Tầng và Mô-đun
:label:`sec_model_construction`

Khi mới giới thiệu mạng neuron, ta tập trung vào các mô hình tuyến tính với một đầu ra duy nhất. Khi đó, toàn bộ mô hình chỉ gồm một neuron duy nhất. Lưu ý rằng một neuron đơn lẻ (i) nhận một tập đầu vào; (ii) tạo ra một đầu ra vô hướng tương ứng; và (iii) có một tập tham số liên quan có thể được cập nhật để tối ưu một hàm mục tiêu nào đó mà ta quan tâm. Sau đó, khi bắt đầu nghĩ đến những mạng có nhiều đầu ra, ta tận dụng các phép toán vector hóa để mô tả cả một tầng neuron. Giống như từng neuron riêng lẻ, các tầng (i) nhận một tập đầu vào, (ii) tạo ra các đầu ra tương ứng, và (iii) được mô tả bởi một tập tham số có thể điều chỉnh. Khi ta làm việc với hồi quy softmax, bản thân một tầng duy nhất đã là mô hình. Tuy nhiên, ngay cả khi sau đó ta giới thiệu MLP, ta vẫn có thể coi mô hình giữ nguyên cấu trúc cơ bản này.

Thú vị là, với MLP, cả toàn bộ mô hình lẫn các tầng cấu thành đều có chung cấu trúc này. Toàn bộ mô hình nhận các đầu vào thô (các đặc trưng), tạo ra các đầu ra (các dự đoán), và sở hữu các tham số (tham số gộp lại từ mọi tầng cấu thành). Tương tự, mỗi tầng riêng lẻ nhận đầu vào (do tầng trước cung cấp), tạo ra đầu ra (đầu vào cho tầng tiếp theo), và sở hữu một tập tham số có thể điều chỉnh, được cập nhật theo tín hiệu truyền ngược về từ tầng tiếp theo.

Dù bạn có thể nghĩ rằng neuron, tầng và mô hình đã cho ta đủ các tầng trừu tượng để làm việc, hóa ra ta thường thấy tiện lợi khi nói về những thành phần lớn hơn một tầng riêng lẻ nhưng nhỏ hơn toàn bộ mô hình. Chẳng hạn, kiến trúc ResNet-152, cực kỳ phổ biến trong thị giác máy tính, có hàng trăm tầng. Các tầng này gồm những khuôn mẫu lặp lại của *các nhóm tầng*. Cài đặt một mạng như vậy theo từng tầng một có thể trở nên tẻ nhạt. Mối lo này không chỉ là giả định: những khuôn mẫu thiết kế như vậy rất phổ biến trong thực tế. Kiến trúc ResNet nói trên đã giành chiến thắng ở các cuộc thi thị giác máy tính ImageNet và COCO năm 2015 cho cả nhận dạng và phát hiện :cite:`He.Zhang.Ren.ea.2016`, và vẫn là một kiến trúc được ưa dùng cho nhiều tác vụ thị giác. Những kiến trúc tương tự, trong đó các tầng được sắp xếp theo nhiều khuôn mẫu lặp lại khác nhau, giờ đây có mặt khắp nơi trong các lĩnh vực khác, gồm xử lý ngôn ngữ tự nhiên và tiếng nói.

Để cài đặt những mạng phức tạp này, ta giới thiệu khái niệm *mô-đun* (module) của mạng neuron. Một mô-đun có thể mô tả một tầng đơn lẻ, một thành phần gồm nhiều tầng, hoặc chính toàn bộ mô hình! Một lợi ích của việc làm việc với tầng trừu tượng mô-đun là chúng có thể được kết hợp thành những cấu phần lớn hơn, thường là theo cách đệ quy. Điều này được minh họa trong :numref:`fig_blocks`. Bằng cách định nghĩa code để sinh ra các mô-đun có độ phức tạp tùy ý theo yêu cầu, ta có thể viết code gọn đến bất ngờ mà vẫn cài đặt được những mạng neuron phức tạp.

![Nhiều tầng được kết hợp thành các mô-đun, tạo thành những khuôn mẫu lặp lại của các mô hình lớn hơn.](../img/blocks.svg)
:label:`fig_blocks`

Từ góc độ lập trình, một mô-đun được biểu diễn bởi một *lớp* (class). Bất kỳ lớp con nào của nó cũng phải định nghĩa một phương thức lan truyền xuôi biến đổi đầu vào thành đầu ra và phải lưu trữ mọi tham số cần thiết. Lưu ý rằng một số mô-đun hoàn toàn không cần tham số nào. Cuối cùng, một mô-đun phải có một phương thức lan truyền ngược nhằm mục đích tính gradient. May mắn thay, nhờ một chút phép màu hậu trường do cơ chế tính vi phân tự động mang lại (đã giới thiệu trong :numref:`sec_autograd`), khi định nghĩa mô-đun của riêng mình, ta chỉ cần quan tâm đến các tham số và phương thức lan truyền xuôi.

```{.python .input}
import torch
from torch import nn
from torch.nn import functional as F
```

[**Để bắt đầu, ta xem lại đoạn code mà ta đã dùng để cài đặt MLP**] (:numref:`sec_mlp`). Đoạn code sau tạo một mạng với một tầng ẩn kết nối đầy đủ gồm 256 nút và hàm kích hoạt ReLU, theo sau là một tầng đầu ra kết nối đầy đủ gồm mười nút (không có hàm kích hoạt).

```{.python .input}
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

X = torch.rand(2, 20)
net(X).shape
```

Trong ví dụ này, ta đã dựng mô hình bằng cách khởi tạo một `nn.Sequential`, với các tầng được truyền vào làm đối số theo đúng thứ tự mà chúng cần được thực thi. Nói ngắn gọn, (**`nn.Sequential` định nghĩa một loại `Module` đặc biệt**), lớp biểu diễn một mô-đun trong PyTorch. Nó duy trì một danh sách có thứ tự các `Module` cấu thành. Lưu ý rằng mỗi tầng trong hai tầng kết nối đầy đủ là một thực thể của lớp `Linear`, bản thân lớp này là một lớp con của `Module`. Phương thức lan truyền xuôi (`forward`) cũng đơn giản đáng kinh ngạc: nó nối chuỗi các mô-đun trong danh sách lại với nhau, truyền đầu ra của mỗi mô-đun làm đầu vào cho mô-đun tiếp theo. Lưu ý rằng cho đến nay, ta vẫn gọi các mô hình thông qua cú pháp `net(X)` để lấy đầu ra của chúng. Đây thực ra chỉ là cách viết tắt của `net.__call__(X)`.

## [**Một Mô-đun Tùy chỉnh**]

Có lẽ cách dễ nhất để có trực giác về cách một mô-đun hoạt động là tự cài đặt một mô-đun. Trước khi làm điều đó, ta tóm tắt ngắn gọn các chức năng cơ bản mà mỗi mô-đun phải cung cấp:

1. Nhận dữ liệu đầu vào làm đối số cho phương thức lan truyền xuôi.
1. Tạo ra đầu ra bằng cách để phương thức lan truyền xuôi trả về một giá trị. Lưu ý rằng đầu ra có thể có kích thước khác với đầu vào. Chẳng hạn, tầng kết nối đầy đủ thứ nhất trong mô hình ở trên nhận đầu vào có số chiều tùy ý nhưng trả về đầu ra có 256 chiều.
1. Tính gradient của đầu ra theo đầu vào, có thể truy cập qua phương thức lan truyền ngược. Thông thường việc này diễn ra tự động.
1. Lưu trữ và cung cấp quyền truy cập tới những tham số cần thiết để thực thi phép tính lan truyền xuôi.
1. Khởi tạo các tham số mô hình khi cần.

Trong đoạn code sau, ta tự viết từ đầu một mô-đun tương ứng với một MLP có một tầng ẩn gồm 256 nút ẩn và một tầng đầu ra 10 chiều. Lưu ý rằng lớp `MLP` dưới đây kế thừa lớp biểu diễn mô-đun. Ta sẽ dựa nhiều vào các phương thức của lớp cha, chỉ cung cấp hàm khởi tạo của riêng mình (phương thức `__init__` trong Python) và phương thức lan truyền xuôi.

```{.python .input}
class MLP(nn.Module):
    def __init__(self):
        # Gọi hàm khởi tạo của lớp cha nn.Module để thực hiện các bước khởi
        # tạo cần thiết
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    # Định nghĩa lan truyền xuôi của mô hình, tức là cách trả về đầu ra cần
    # thiết của mô hình dựa trên đầu vào X
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

Trước hết hãy tập trung vào phương thức lan truyền xuôi. Lưu ý rằng nó nhận `X` làm đầu vào, tính biểu diễn ẩn sau khi áp dụng hàm kích hoạt, và đưa ra các logit của nó. Trong cách cài đặt `MLP` này, cả hai tầng đều là biến thực thể (instance variable). Để thấy vì sao điều này hợp lý, hãy tưởng tượng khởi tạo hai MLP, `net1` và `net2`, rồi huấn luyện chúng trên các dữ liệu khác nhau. Lẽ tự nhiên, ta sẽ mong chúng biểu diễn hai mô hình đã học khác nhau.

Ta [**khởi tạo các tầng của MLP**] trong hàm khởi tạo (**và sau đó gọi các tầng này**) trong mỗi lần gọi phương thức lan truyền xuôi. Hãy chú ý một vài chi tiết quan trọng. Thứ nhất, phương thức `__init__` tùy chỉnh của ta gọi phương thức `__init__` của lớp cha thông qua `super().__init__()`, giúp ta khỏi phải viết lại những đoạn code khuôn mẫu áp dụng cho hầu hết các mô-đun. Sau đó ta khởi tạo hai tầng kết nối đầy đủ, gán chúng cho `self.hidden` và `self.out`. Lưu ý rằng trừ khi cài đặt một tầng mới, ta không cần quan tâm đến phương thức lan truyền ngược hay việc khởi tạo tham số. Hệ thống sẽ tự động sinh ra các phương thức này. Hãy thử nghiệm điều này.

```{.python .input}
net = MLP()
net(X).shape
```

Một ưu điểm then chốt của tầng trừu tượng mô-đun là tính linh hoạt. Ta có thể kế thừa một mô-đun để tạo các tầng (như lớp tầng kết nối đầy đủ), toàn bộ mô hình (như lớp `MLP` ở trên), hoặc các thành phần khác nhau có độ phức tạp trung gian. Ta sẽ khai thác tính linh hoạt này xuyên suốt các chương tiếp theo, chẳng hạn khi làm việc với mạng neuron tích chập.

## [**Mô-đun Tuần tự**]
:label:`subsec_model-construction-sequential`

Giờ ta có thể xem kỹ hơn cách lớp `Sequential` hoạt động. Hãy nhớ lại rằng `Sequential` được thiết kế để nối chuỗi các mô-đun khác lại với nhau. Để xây dựng phiên bản đơn giản hóa `MySequential` của riêng mình, ta chỉ cần định nghĩa hai phương thức then chốt:

1. Một phương thức để lần lượt thêm từng mô-đun vào một danh sách.
1. Một phương thức lan truyền xuôi để đưa đầu vào qua chuỗi các mô-đun, theo đúng thứ tự mà chúng được thêm vào.

Lớp `MySequential` sau đây cung cấp cùng chức năng với lớp `Sequential` mặc định.

```{.python .input}
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():            
            X = module(X)
        return X
```

Trong phương thức `__init__`, ta thêm từng mô-đun bằng cách gọi phương thức `add_modules`. Sau này có thể truy cập các mô-đun đó bằng phương thức `children`. Bằng cách này, hệ thống biết được các mô-đun đã thêm, và nó sẽ khởi tạo đúng cách các tham số của mỗi mô-đun.

Khi phương thức lan truyền xuôi của `MySequential` được gọi, mỗi mô-đun đã thêm được thực thi theo thứ tự mà chúng được thêm vào. Giờ ta có thể cài đặt lại một MLP bằng lớp `MySequential` của mình.

```{.python .input}
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X).shape
```

Lưu ý rằng cách dùng `MySequential` này giống hệt đoạn code ta đã viết trước đây cho lớp `Sequential` (như đã mô tả trong :numref:`sec_mlp`).

## [**Thực thi Code trong Phương thức Lan truyền Xuôi**]

Lớp `Sequential` giúp việc xây dựng mô hình trở nên dễ dàng, cho phép ta lắp ráp các kiến trúc mới mà không cần định nghĩa lớp riêng. Tuy nhiên, không phải kiến trúc nào cũng là chuỗi nối đơn giản. Khi cần sự linh hoạt lớn hơn, ta sẽ muốn định nghĩa các khối của riêng mình. Chẳng hạn, ta có thể muốn thực thi luồng điều khiển của Python bên trong phương thức lan truyền xuôi. Hơn nữa, ta có thể muốn thực hiện các phép toán tùy ý, không chỉ dựa vào các tầng mạng neuron đã định nghĩa sẵn.

Có thể bạn đã để ý rằng cho đến nay, mọi phép toán trong mạng của ta đều tác động lên các kích hoạt và các tham số của mạng. Tuy nhiên, đôi khi ta có thể muốn đưa vào những số hạng không phải là kết quả của các tầng trước cũng không phải là tham số có thể cập nhật. Ta gọi chúng là *tham số hằng* (constant parameter). Chẳng hạn, giả sử ta muốn một tầng tính hàm $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$, trong đó $\mathbf{x}$ là đầu vào, $\mathbf{w}$ là tham số, và $c$ là một hằng số cho trước không được cập nhật trong quá trình tối ưu. Vì vậy ta cài đặt một lớp `FixedHiddenMLP` như sau.

```{.python .input}
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Trọng số ngẫu nhiên không tính gradient nên giữ nguyên trong suốt
        # quá trình huấn luyện
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)        
        X = F.relu(X @ self.rand_weight + 1)
        # Dùng lại tầng kết nối đầy đủ. Điều này tương đương với việc hai tầng
        # kết nối đầy đủ dùng chung tham số
        X = self.linear(X)
        # Luồng điều khiển
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

Trong mô hình này, ta cài đặt một tầng ẩn có trọng số (`self.rand_weight`) được khởi tạo ngẫu nhiên lúc khởi tạo thực thể và sau đó giữ nguyên. Trọng số này không phải tham số mô hình nên nó không bao giờ được cập nhật bởi lan truyền ngược. Sau đó mạng đưa đầu ra của tầng "cố định" này qua một tầng kết nối đầy đủ.

Lưu ý rằng trước khi trả về đầu ra, mô hình của ta đã làm một việc khác thường. Ta chạy một vòng lặp while, kiểm tra điều kiện chuẩn $\ell_1$ của nó lớn hơn $1$, và chia vector đầu ra cho $2$ cho đến khi nó thỏa mãn điều kiện. Cuối cùng, ta trả về tổng các phần tử trong `X`. Theo hiểu biết của chúng tôi, không có mạng neuron tiêu chuẩn nào thực hiện phép toán này. Lưu ý rằng phép toán cụ thể này có thể không hữu ích trong bất kỳ tác vụ thực tế nào. Chúng tôi chỉ muốn cho bạn thấy cách tích hợp code tùy ý vào luồng tính toán của mạng neuron.

```{.python .input}
net = FixedHiddenMLP()
net(X)
```

Ta có thể [**phối hợp nhiều cách lắp ráp các mô-đun với nhau.**] Trong ví dụ sau, ta lồng các mô-đun theo một số cách sáng tạo.

```{.python .input}
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
chimera(X)
```

## Tóm tắt

Từng tầng riêng lẻ có thể là mô-đun. Nhiều tầng có thể tạo thành một mô-đun. Nhiều mô-đun có thể tạo thành một mô-đun.

Một mô-đun có thể chứa code. Mô-đun lo rất nhiều công việc quản lý, gồm khởi tạo tham số và lan truyền ngược. Việc ghép nối tuần tự các tầng và mô-đun được xử lý bởi mô-đun `Sequential`.

## Bài tập

1. Những vấn đề gì sẽ xảy ra nếu bạn sửa `MySequential` để lưu các mô-đun trong một danh sách Python?
1. Cài đặt một mô-đun nhận hai mô-đun làm đối số, chẳng hạn `net1` và `net2`, và trả về đầu ra ghép nối của cả hai mạng trong lan truyền xuôi. Đây còn được gọi là *mô-đun song song*.
1. Giả sử bạn muốn ghép nối nhiều thực thể của cùng một mạng. Hãy cài đặt một hàm nhà máy (factory function) sinh ra nhiều thực thể của cùng một mô-đun và xây dựng một mạng lớn hơn từ đó.

[Discussions](https://discuss.d2l.ai/t/55)
