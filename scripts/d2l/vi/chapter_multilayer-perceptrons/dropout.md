# Dropout
:label:`sec_dropout`

Hãy nghĩ nhanh xem ta mong đợi gì ở một mô hình dự đoán tốt. Ta muốn nó hoạt động tốt trên dữ liệu chưa thấy. Lý thuyết tổng quát hóa cổ điển gợi ý rằng để thu hẹp khoảng cách giữa hiệu năng huấn luyện và kiểm tra, ta nên hướng tới một mô hình đơn giản. Sự đơn giản có thể đến dưới dạng số chiều nhỏ. Ta đã khám phá điều này khi thảo luận về các hàm cơ sở đơn thức của mô hình tuyến tính trong :numref:`sec_generalization_basics`. Ngoài ra, như ta đã thấy khi thảo luận về suy giảm trọng số (điều chuẩn $\ell_2$) trong :numref:`sec_weight_decay`, chuẩn (nghịch đảo) của tham số cũng là một thước đo hữu ích về sự đơn giản. Một khái niệm hữu ích khác về sự đơn giản là độ trơn, tức là hàm không nên nhạy cảm với những thay đổi nhỏ ở đầu vào. Chẳng hạn, khi phân loại ảnh, ta mong rằng thêm một chút nhiễu ngẫu nhiên vào các điểm ảnh phần lớn là vô hại.

:citet:`Bishop.1995` đã hình thức hóa ý tưởng này khi ông chứng minh rằng huấn luyện với nhiễu ở đầu vào tương đương với điều chuẩn Tikhonov. Công trình này đã vạch ra một mối liên hệ toán học rõ ràng giữa yêu cầu một hàm phải trơn (và do đó đơn giản) và yêu cầu nó phải bền vững trước các nhiễu loạn ở đầu vào.

Sau đó, :citet:`Srivastava.Hinton.Krizhevsky.ea.2014` đã phát triển một ý tưởng thông minh về cách áp dụng ý tưởng của Bishop cho cả các tầng bên trong của mạng. Ý tưởng của họ, gọi là *dropout*, gồm việc đưa nhiễu vào khi tính mỗi tầng bên trong trong quá trình lan truyền xuôi, và nó đã trở thành một kỹ thuật tiêu chuẩn để huấn luyện mạng neuron. Phương pháp được gọi là *dropout* vì ta thực sự *loại bỏ* (drop out) một số neuron trong khi huấn luyện. Trong suốt quá trình huấn luyện, ở mỗi vòng lặp, dropout tiêu chuẩn gồm việc đặt về không một tỉ lệ nào đó các nút trong mỗi tầng trước khi tính tầng tiếp theo.

Nói cho rõ, chính chúng tôi đang áp đặt câu chuyện liên hệ với Bishop. Bài báo gốc về dropout đưa ra trực giác thông qua một phép so sánh đáng ngạc nhiên với sinh sản hữu tính. Các tác giả lập luận rằng quá khớp trong mạng neuron được đặc trưng bởi một trạng thái trong đó mỗi tầng dựa vào một khuôn mẫu kích hoạt cụ thể ở tầng trước, gọi điều kiện này là *đồng thích nghi* (co-adaptation). Họ cho rằng dropout phá vỡ sự đồng thích nghi giống như sinh sản hữu tính được cho là phá vỡ các gen đồng thích nghi. Dù một lập luận như vậy cho lý thuyết này chắc chắn còn gây tranh cãi, bản thân kỹ thuật dropout đã chứng tỏ sức sống bền bỉ, và nhiều dạng dropout khác nhau được cài đặt trong hầu hết các thư viện học sâu.

Thách thức then chốt là cách đưa nhiễu này vào. Một ý tưởng là đưa nó vào theo cách *không chệch* (unbiased) để giá trị kỳ vọng của mỗi tầng (khi cố định các tầng khác) bằng giá trị mà nó sẽ nhận nếu không có nhiễu. Trong công trình của Bishop, ông thêm nhiễu Gauss vào đầu vào của một mô hình tuyến tính. Ở mỗi vòng lặp huấn luyện, ông thêm nhiễu lấy mẫu từ một phân phối có trung bình không $\epsilon \sim \mathcal{N}(0,\sigma^2)$ vào đầu vào $\mathbf{x}$, tạo ra một điểm bị nhiễu $\mathbf{x}' = \mathbf{x} + \epsilon$. Về kỳ vọng, $E[\mathbf{x}'] = \mathbf{x}$.

Trong điều chuẩn dropout tiêu chuẩn, người ta đặt về không một tỉ lệ nào đó các nút trong mỗi tầng rồi *khử chệch* (debias) mỗi tầng bằng cách chuẩn hóa theo tỉ lệ các nút được giữ lại (không bị loại bỏ). Nói cách khác, với *xác suất dropout* $p$, mỗi kích hoạt trung gian $h$ được thay bằng một biến ngẫu nhiên $h'$ như sau:

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \textrm{ với xác suất } p \\
    \frac{h}{1-p} & \textrm{ trong trường hợp còn lại}
\end{cases}
\end{aligned}
$$

Theo thiết kế, kỳ vọng giữ nguyên, tức là $E[h'] = h$.

```{.python .input}
from d2l import torch as d2l
import torch
from torch import nn
```

## Dropout trong Thực tế

Hãy nhớ lại MLP có một tầng ẩn với năm nút ẩn từ :numref:`fig_mlp`. Khi áp dụng dropout cho một tầng ẩn, đặt về không mỗi nút ẩn với xác suất $p$, kết quả có thể được xem như một mạng chỉ chứa một tập con các neuron ban đầu. Trong :numref:`fig_dropout2`, $h_2$ và $h_5$ bị loại bỏ. Do đó, việc tính đầu ra không còn phụ thuộc vào $h_2$ hay $h_5$ và gradient tương ứng của chúng cũng biến mất khi thực hiện lan truyền ngược. Bằng cách này, việc tính tầng đầu ra không thể phụ thuộc quá mức vào bất kỳ một phần tử nào trong $h_1, \ldots, h_5$.

![MLP trước và sau dropout.](../img/dropout2.svg)
:label:`fig_dropout2`

Thông thường, ta tắt dropout khi kiểm tra. Với một mô hình đã huấn luyện và một mẫu mới, ta không loại bỏ nút nào và do đó không cần chuẩn hóa. Tuy nhiên, có một số ngoại lệ: một số nhà nghiên cứu dùng dropout lúc kiểm tra như một phương pháp kinh nghiệm để ước lượng *độ bất định* của dự đoán mạng neuron: nếu các dự đoán thống nhất giữa nhiều đầu ra dropout khác nhau, ta có thể nói rằng mạng tự tin hơn.

## Cài đặt từ Đầu

Để cài đặt hàm dropout cho một tầng, ta phải lấy số mẫu từ một biến ngẫu nhiên Bernoulli (nhị phân) bằng với số chiều của tầng, trong đó biến ngẫu nhiên nhận giá trị $1$ (giữ lại) với xác suất $1-p$ và $0$ (loại bỏ) với xác suất $p$. Một cách dễ để cài đặt điều này là trước tiên lấy mẫu từ phân phối đều $U[0, 1]$. Sau đó ta có thể giữ lại những nút có mẫu tương ứng lớn hơn $p$, loại bỏ phần còn lại.

Trong đoạn code sau, ta (**cài đặt một hàm `dropout_layer` loại bỏ các phần tử trong tensor đầu vào `X` với xác suất `dropout`**), co giãn lại phần còn lại như mô tả ở trên: chia các phần tử sống sót cho `1.0-dropout`.

```{.python .input}
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

Ta có thể [**thử hàm `dropout_layer` trên một vài ví dụ**]. Trong các dòng code sau, ta đưa đầu vào `X` qua phép dropout với xác suất lần lượt là 0, 0.5 và 1.

```{.python .input}
X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
```

### Định nghĩa Mô hình

Mô hình dưới đây áp dụng dropout cho đầu ra của mỗi tầng ẩn (sau hàm kích hoạt). Ta có thể đặt xác suất dropout riêng cho từng tầng. Một lựa chọn phổ biến là đặt xác suất dropout thấp hơn cho tầng gần đầu vào hơn. Ta đảm bảo rằng dropout chỉ hoạt động trong khi huấn luyện.

```{.python .input}
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:  
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

### [**Huấn luyện**]

Phần sau tương tự như việc huấn luyện MLP đã mô tả trước đây.

```{.python .input .gpu}
hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## [**Cài đặt Súc tích**]

Với các API cấp cao, tất cả những gì ta cần làm là thêm một tầng `Dropout` sau mỗi tầng kết nối đầy đủ, truyền xác suất dropout làm đối số duy nhất cho hàm khởi tạo của nó. Trong khi huấn luyện, tầng `Dropout` sẽ loại bỏ ngẫu nhiên các đầu ra của tầng trước (hay tương đương, các đầu vào của tầng tiếp theo) theo xác suất dropout đã chỉ định. Khi không ở chế độ huấn luyện, tầng `Dropout` chỉ đơn giản cho dữ liệu đi qua khi kiểm tra.

```{.python .input}
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(), 
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(), 
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))
```

Tiếp theo, ta [**huấn luyện mô hình**].

```{.python .input .gpu}
model = DropoutMLP(**hparams)
trainer.fit(model, data)
```

## Tóm tắt

Ngoài việc kiểm soát số chiều và kích thước của vector trọng số, dropout là thêm một công cụ nữa để tránh quá khớp. Thường thì các công cụ được dùng kết hợp với nhau. Lưu ý rằng dropout chỉ được dùng trong khi huấn luyện: nó thay một kích hoạt $h$ bằng một biến ngẫu nhiên có giá trị kỳ vọng $h$.

## Bài tập

1. Điều gì xảy ra nếu bạn thay đổi xác suất dropout của tầng thứ nhất và thứ hai? Cụ thể, điều gì xảy ra nếu bạn hoán đổi xác suất của hai tầng? Hãy thiết kế một thí nghiệm để trả lời những câu hỏi này, mô tả kết quả một cách định lượng, và tóm tắt những bài học định tính.
1. Tăng số epoch và so sánh kết quả thu được khi dùng dropout với khi không dùng.
1. Phương sai của các kích hoạt trong mỗi tầng ẩn là bao nhiêu khi có và không có dropout? Vẽ một đồ thị cho thấy đại lượng này thay đổi thế nào theo thời gian ở cả hai mô hình.
1. Vì sao dropout thường không được dùng khi kiểm tra?
1. Dùng mô hình trong mục này làm ví dụ, so sánh tác dụng của dropout và suy giảm trọng số. Điều gì xảy ra khi dùng đồng thời dropout và suy giảm trọng số? Kết quả có cộng dồn không? Có hiện tượng lợi ích giảm dần (hoặc tệ hơn) không? Chúng có triệt tiêu lẫn nhau không?
1. Điều gì xảy ra nếu ta áp dụng dropout cho từng trọng số riêng lẻ của ma trận trọng số thay vì cho các kích hoạt?
1. Hãy phát minh một kỹ thuật khác để đưa nhiễu ngẫu nhiên vào mỗi tầng, khác với kỹ thuật dropout tiêu chuẩn. Bạn có thể phát triển một phương pháp vượt trội dropout trên tập dữ liệu Fashion-MNIST (với một kiến trúc cố định) không?

[Discussions](https://discuss.d2l.ai/t/101)
