# Nền tảng học sâu: khởi tạo tham số

Trang [Code](dlnote/dl_basic_code_vi) đi kèm tái hiện bốn hình histogram của bài trên một mạng 6 tầng, kiểm tra cách khởi tạo mặc định của `nn.Linear` trong PyTorch, và cho thấy khởi tạo toàn 0 khiến các neuron mãi giống nhau.

<!-- theory: qa/ch03_vi#3.8 ; qa/ch13_vi#13.5 -->

> Bài này tham khảo hai sách *Cửa ngõ trí tuệ: nhập môn mạng neuron và học sâu* (智能之门, tức giáo trình AI-EDU của Microsoft) và *Giải tích mạng neuron tích chập* (解析卷积神经网络), cùng một số tài liệu trên mạng, kèm cách hiểu và tóm tắt của tác giả. Mọi hình histogram trong bài lấy từ tài liệu tham khảo 3.

## 1 Tổng quan về khởi tạo tham số

Mạng neuron thường được cập nhật tham số bằng thuật toán **stochastic gradient descent**, và học tham số mạng là bài toán không lồi; khi tối ưu bằng gradient descent, **giá trị khởi tạo của trọng số rất quan trọng**.

Trước hết cần rõ: các **chiến lược khởi tạo tham số** hiện nay đơn giản và mang tính kinh nghiệm (heuristic). Thiết kế chiến lược khởi tạo tốt hơn là việc khó, vì quá trình tối ưu mạng neuron đến nay vẫn chưa được hiểu rõ (quá trình huấn luyện vẫn là một hộp đen).

Phần lớn chiến lược khởi tạo nhắm tới việc mạng có một số tính chất tốt *lúc khởi tạo*. Nhưng ta chưa hiểu rõ tính chất nào được giữ lại sau khi việc học bắt đầu, và trong trường hợp nào. Khó khăn nữa là có những điểm khởi đầu tốt về mặt tối ưu nhưng lại bất lợi về mặt tổng quát hoá. Hiểu biết về việc điểm khởi đầu ảnh hưởng thế nào tới tổng quát hoá còn rất sơ khai, gần như không cho chỉ dẫn nào về cách chọn điểm khởi đầu.

### 1.1 Vì sao phải khởi tạo tham số

Thuật toán huấn luyện mô hình học sâu thường là **lặp**, nên người huấn luyện phải chỉ định điểm bắt đầu lặp, tức chọn chiến lược khởi tạo tham số.

### 1.2 Vì sao khởi tạo tham số quan trọng

Huấn luyện mô hình học sâu khó đến mức phần lớn thuật toán chịu ảnh hưởng lớn của cách khởi tạo.

**Điểm khởi đầu có thể quyết định thuật toán có hội tụ hay không.** Có những điểm khởi đầu rất bất ổn, khiến thuật toán gặp khó khăn về số học và có thể thất bại hoàn toàn. Khi việc học hội tụ, điểm khởi đầu quyết định hội tụ nhanh đến đâu, và hội tụ tới điểm có chi phí cao hay thấp. Ngoài ra, các điểm có cùng giá trị chi phí vẫn có thể có sai số tổng quát hoá rất khác nhau, và điểm khởi đầu cũng ảnh hưởng tới tổng quát hoá.

## 2 Các cách khởi tạo trọng số

Trong thực tế, khởi tạo trọng số theo **phân phối Gauss** (`Gaussian distribution`) hoặc **phân phối đều** (`uniform distribution`) đều khá **hiệu quả**. Khác biệt giữa hai lựa chọn này đến nay chưa được nghiên cứu kỹ; điều chắc chắn là **độ lớn** (thang đo) của phân phối khởi tạo ảnh hưởng lớn tới cả kết quả tối ưu lẫn khả năng tổng quát hoá.

> Phân phối của giá trị khởi tạo ngẫu nhiên ảnh hưởng tới hiệu năng mô hình, nhưng cơ chế (nguyên lý tối ưu mạng neuron) chưa được hiểu thấu; huấn luyện mô hình đúng là "luyện đan", nhiều lúc phải dựa vào kinh nghiệm của người luyện.

Các chiến lược khởi tạo khác:

1. **Nạp tham số của mô hình tiền huấn luyện**: dùng thẳng tham số đã huấn luyện trên tập dữ liệu lớn, phần nào cải thiện khả năng tổng quát hoá.
2. **Khởi tạo ngẫu nhiên**: không được khởi tạo mọi tham số bằng `0`. Nếu vậy, ở lần lan truyền xuôi đầu tiên, mọi neuron trong tầng ẩn có cùng kích hoạt, lan truyền ngược cũng cập nhật chúng như nhau, nên các neuron không bao giờ phân biệt được, gọi là hiện tượng **trọng số đối xứng**. Cách tốt là **khởi tạo ngẫu nhiên** từng tham số.
3. **Khởi tạo bằng giá trị cố định**: ví dụ độ lệch (bias) thường khởi tạo bằng `0`; độ lệch của cổng quên trong LSTM thường đặt 1 hoặc 2 để cổng mở lúc đầu, giúp gradient đi xa hơn theo thời gian; với neuron ReLU, có khi đặt độ lệch 0,01 để neuron dễ được kích hoạt lúc đầu.

Mỗi tài liệu phân loại các cách khởi tạo một kiểu; bài này liệt kê thẳng các cách thường dùng và hiệu quả, kèm `API` của `PyTorch`.

### 2.1 Khởi tạo toàn 0

Dù lý tưởng là tham số nên có dấu dương âm cân bằng (**kỳ vọng bằng 0**), điều đó không có nghĩa là có thể khởi tạo mọi tham số bằng `0` ($W=0$)! Dưới đây là trọng số của một mạng hai tầng (3 neuron ẩn, 1 đầu ra) **sau khi huấn luyện** từ khởi tạo toàn 0:

```python
W1= [[-0.82452497 -0.82452497 -0.82452497]]
B1= [[-0.01143752 -0.01143752 -0.01143752]]
W2= [[-0.68583865]
     [-0.68583865]
     [-0.68583865]]
B2= [[0.68359678]]
```

Ba đơn vị trong `W1`, `B1`, `W2` có giá trị giống hệt nhau: vì khởi tạo đều là 0, gradient lan truyền về như nhau, mọi neuron được cập nhật đồng bộ, không khác gì nhau. Ba neuron ẩn thực chất chỉ là một neuron, nên huấn luyện bao lâu kết quả cũng không đúng được.

> **Biên tập:** Bản gốc gọi đoạn in ra là "trọng số khởi tạo bằng 0", dễ gây hiểu nhầm vì các giá trị không bằng 0: đó là trọng số sau khi huấn luyện xong từ khởi tạo bằng 0 (ví dụ của AI-EDU). Trang Code tái hiện hiện tượng này.

### 2.2 Khởi tạo chuẩn

1. **Khởi tạo Gauss**: lấy mỗi tham số ngẫu nhiên theo phân phối Gauss:

$$
W \sim \mathcal N(0, \sigma^2)
$$

với $\mathcal N$ là **phân phối Gauss** (còn gọi là phân phối chuẩn), tham số vị trí $\mu = 0$ (**kỳ vọng**) và tham số tỉ lệ $\sigma^2$ (**phương sai**). Có nơi gọi là **khởi tạo `Normal`**. Khi $\sigma=1$ thì gọi là phân phối Gauss chuẩn tắc.

> **Biên tập:** Bản gốc gọi $\mathcal N(0,\sigma^2)$ là "phân phối Gauss chuẩn tắc"; chuẩn tắc chỉ khi $\sigma=1$.

`API` tương ứng trong `PyTorch`:

```python
torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
```

2. Tương tự là **khởi tạo đều**, lấy giá trị trong khoảng $[-r, r]$:

```python
torch.nn.init.uniform_(tensor, a=-r, b=r)       # mặc định a=0, b=1
```

> **Biên tập:** Bản gốc chú thích "mặc định thường dùng khởi tạo Gauss chuẩn tắc, trung bình 0, phương sai 1". Đó chỉ là giá trị mặc định của hàm `normal_`, không phải cách PyTorch khởi tạo các tầng. `nn.Linear` và `nn.Conv2d` mặc định dùng `kaiming_uniform_(a=√5)`, tức phân phối đều $U(-1/\sqrt{n_{in}},\ 1/\sqrt{n_{in}})$; trang Code kiểm tra điều này. Khởi tạo $\mathcal N(0,1)$ cho một tầng 1000 đầu vào sẽ cho đầu ra có độ lệch chuẩn khoảng $\sqrt{1000}\approx 32$.

Gauss và đều đều là cách khởi tạo với phương sai cố định; điểm then chốt là **chọn phương sai thế nào**:

- quá nhỏ: đầu ra của neuron quá nhỏ, qua nhiều tầng thì tín hiệu (và gradient) tiêu biến;
- quá lớn: đầu vào của `sigmoid` rơi vào vùng bão hoà, gradient gần `0`, cũng tiêu biến gradient. Thường phải dùng kèm tầng `BN`.

Khi bài toán đơn giản và mạng không sâu, khởi tạo chuẩn là đủ. Nhưng với mạng sâu dùng sigmoid, khởi tạo chuẩn gặp vấn đề như hình 1.

![Khởi tạo chuẩn với sigmoid](../images/parameter_init/init_normal_sigmoid.png)

Hình 1: Khởi tạo chuẩn (Normal) cho mạng 6 tầng kết nối đầy đủ dùng sigmoid; mỗi cột là histogram kích hoạt của một tầng.

Hình trên là mạng sâu `6` tầng gồm **tầng kết nối đầy đủ + sigmoid**, vẽ **histogram kích hoạt của từng tầng**. Kích hoạt dồn về hai đầu 0 và 1, nơi đạo hàm sigmoid gần `0`, nên **gradient tiêu biến dần khi lan truyền ngược**. Rất ít giá trị nằm ở giữa, rất bất lợi cho việc học.

> Khởi tạo Gauss với **phương sai cố định** truyền thống khiến mô hình khó hội tụ khi mạng sâu.

### 2.3 Khởi tạo Xavier

Từ quan sát trên, Xavier Glorot và Yoshua Bengio đề xuất cách khởi tạo `Xavier` năm `2010`.

So với khởi tạo Gauss trực tiếp, ưu điểm của `Xavier` là: ở mạng thông thường, phương sai đầu ra của neuron tăng dần khi lan truyền xuôi, còn `Xavier` về lý thuyết giữ được **phương sai của dữ liệu vào và ra ở mỗi tầng bằng nhau** (tác dụng giống `BN`; nguyên lý xem [bài báo Xavier](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)).

Điều kiện: khi lan truyền xuôi, phương sai của kích hoạt không đổi qua các tầng; khi lan truyền ngược, phương sai của gradient theo đầu vào tổng không đổi qua các tầng.

Lập luận (với hàm kích hoạt tuyến tính quanh 0, trọng số và đầu vào độc lập, trung bình 0): đầu ra $a=\sum_{i=1}^{n_{in}}w_ix_i$ có $\operatorname{Var}(a)=n_{in}\operatorname{Var}(w)\operatorname{Var}(x)$. Giữ phương sai xuôi cần $n_{in}\operatorname{Var}(w)=1$, giữ phương sai ngược cần $n_{out}\operatorname{Var}(w)=1$; Xavier lấy trung bình hai điều kiện: $\operatorname{Var}(w)=\frac{2}{n_{in}+n_{out}}$.

**Xavier phân phối Gauss**:

$$
W \sim \mathcal N
\left(0,\ \frac{2}{n_{in} + n_{out}}\right)
$$

`API` tương ứng:

```python
torch.nn.init.xavier_normal_(tensor, gain=1.0)
```

**Xavier phân phối đều** (phân phối đều trên $[-r, r]$ có phương sai $r^2/3$, chọn $r$ để phương sai vẫn bằng $\frac{2}{n_{in}+n_{out}}$):

$$
W \sim U
\left(-\sqrt{\frac{6}{n_{in} + n_{out}}},\ \sqrt{\frac{6}{n_{in} + n_{out}}}\right)
$$

```python
torch.nn.init.xavier_uniform_(tensor, gain=1.0)
```

> **Biên tập:** Bản gốc viết $\mathcal N\big(0, \sqrt{2/(n_{in}+n_{out})}\big)$, tức đặt **độ lệch chuẩn** vào vị trí của **phương sai** trong ký hiệu $\mathcal N(\mu,\sigma^2)$; phương sai đúng là $2/(n_{in}+n_{out})$ (độ lệch chuẩn là căn của nó). Bản gốc cũng ghi Xavier "dùng cho sigmoid" và nhận xét bản phân phối đều "có lẽ ít dùng". Xavier được suy ra cho hàm kích hoạt đối xứng, gần tuyến tính quanh 0 như tanh; và bản phân phối đều (`glorot_uniform`) chính là khởi tạo mặc định của các tầng trong Keras/TensorFlow, nên được dùng rất nhiều.

Hình 2 cho thấy Xavier với sigmoid, cùng mạng 6 tầng như hình 1. Kích hoạt của các tầng không còn dồn về hai đầu mà tập trung ở vùng giữa, nơi sigmoid còn gradient, có lợi cho việc học. (Phân phối hẹp dần ở các tầng sau, vì Xavier giả định hàm kích hoạt có tâm tại 0, còn sigmoid có tâm tại 0,5.)

![Xavier với sigmoid](../images/parameter_init/init_xavier_sigmoid.png)

Hình 2: Khởi tạo Xavier cho cùng mạng sigmoid 6 tầng.

### 2.4 Khởi tạo He

Khi học sâu phát triển, người ta thấy sigmoid hạn chế hiệu quả lan truyền ngược và gây tiêu biến gradient, nên chuyển sang hàm kích hoạt `ReLU`.

Nhưng Xavier không hoạt động tốt với ReLU: càng lên tầng sâu, kích hoạt càng dồn về `0`, cũng dẫn tới tiêu biến gradient.

Hình 3 cho thấy Xavier với ReLU.

![Xavier với ReLU](../images/parameter_init/init_xavier_relu.png)

Hình 3: Khởi tạo Xavier cho mạng 6 tầng dùng ReLU: phân phối kích hoạt co dần về 0 qua các tầng.

Vì vậy Hà Khải Minh (Kaiming He) và cộng sự đề xuất cách khởi tạo He năm 2015 (còn gọi là khởi tạo MSRA).

**Khởi tạo He** giải quyết việc phương sai thay đổi khi dùng ReLU: ReLU đặt nửa số giá trị (phần âm) về 0, nên với đầu vào đối xứng, $\mathbb E[\operatorname{ReLU}(z)^2]=\frac12\operatorname{Var}(z)$; để bù lại, phương sai trọng số phải gấp đôi Xavier (chỉ xét số đầu vào).

Khởi tạo He là phân phối Gauss trung bình 0, **phương sai $2/n$**, hợp với ReLU:

$$
W \sim \mathcal N
\left(0,\ \frac{2}{n}\right)
$$

với $n$ là số đầu vào của tầng ($n_{in}$, hay `fan_in`).

`API` tương ứng:

```python
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='relu')
```

> **Biên tập:** Bản gốc viết $\mathcal N\big(0, \sqrt{2/n}\big)$, cùng lỗi lẫn độ lệch chuẩn với phương sai như ở mục Xavier. Bản gốc ghi `nonlinearity='leaky_relu'` (giá trị mặc định); với `a=0`, Leaky ReLU chính là ReLU nên kết quả như nhau, nhưng ghi `'relu'` rõ hơn.

Hình 4 là khởi tạo He với ReLU, cùng mạng như trên.

![He với ReLU](../images/parameter_init/init_msra_relu.png)

Hình 4: Khởi tạo He (MSRA) cho mạng ReLU 6 tầng: phân phối kích hoạt gần như giữ nguyên qua các tầng.

### 2.5 Tổng kết

- Khởi tạo Xavier và He đều **điều chỉnh phương sai của phân phối khởi tạo theo số kết nối của neuron**, còn gọi là **co giãn phương sai** (variance scaling).
- Cách khởi tạo quan trọng cho ổn định số học, và phải chọn cùng với hàm kích hoạt.
- Cách khởi tạo có thể quyết định thuật toán tối ưu hội tụ nhanh đến đâu. Chọn sai có thể gây bùng nổ hoặc tiêu biến gradient khi huấn luyện.

Ngoài các cách trên, hay dùng còn có "khởi tạo ngẫu nhiên kèm BN" và "khởi tạo từ mô hình tiền huấn luyện". Đặc điểm của các cách:

| Cách khởi tạo | Đặc điểm |
| --- | --- |
| Toàn 0 | Mọi neuron trong một tầng học cùng một thứ; không phá được tính đối xứng |
| Ngẫu nhiên (phương sai cố định) | Phương sai đầu ra thay đổi theo số đầu vào $n^{(l-1)}$ của tầng: $\operatorname{Var}[a^l] = n^{(l-1)}\operatorname{Var}[w^l]\operatorname{Var}[a^{(l-1)}]$ |
| Xavier | Không xét ảnh hưởng của hàm kích hoạt lên phân phối đầu ra |
| He | Xét ảnh hưởng của ReLU lên phân phối đầu ra, giữ phương sai đầu vào và đầu ra bằng nhau |
| Ngẫu nhiên + BN | BN giảm sự phụ thuộc vào thang đo của giá trị khởi tạo; dùng độ lệch chuẩn nhỏ là được |
| Mô hình tiền huấn luyện | Dùng tham số đã huấn luyện cho nhiệm vụ khác làm điểm khởi đầu (fine-tuning) |

Ba cách cuối (He, ngẫu nhiên + BN, tiền huấn luyện) là các cách nên ưu tiên.

> **Biên tập:** Bản gốc dán ảnh chụp bảng tiếng Trung (có công thức phương sai bên cạnh); đã gõ lại thành bảng trên.

## 3 Bổ sung: khởi tạo trong mô hình hiện nay

Mục này do người biên soạn viết thêm.

- **Mặc định của PyTorch** (`nn.Linear`, `nn.Conv*`): `kaiming_uniform_(a=√5)`, tức $U(-1/\sqrt{n_{in}},\ 1/\sqrt{n_{in}})$ với phương sai $\frac{1}{3n_{in}}$, nhỏ hơn phương sai $\frac{2}{n_{in}}$ của He sáu lần. Lựa chọn này giữ từ thời Lua Torch, và PyTorch không đổi vì lý do tương thích; mạng có BN hay residual ít nhạy với khác biệt này, nhưng với mạng ReLU thuần, sâu, gọi `kaiming_normal_` tường minh sẽ ổn định hơn.
- **Mạng residual**: đầu ra cộng dồn qua các khối, phương sai tăng theo độ sâu. Mẹo phổ biến là khởi tạo $\gamma$ của tầng BN cuối trong mỗi khối bằng 0 ("zero-init residual", Goyal et al. 2017), để lúc đầu mỗi khối là hàm đồng nhất. Fixup và T-Fixup đẩy ý này xa hơn để huấn luyện mạng residual không cần chuẩn hoá.
- **Transformer**: thường khởi tạo $\mathcal N(0;\ 0{,}02^2)$ (BERT, GPT-2), và GPT-2 chia thêm cho $\sqrt{2L}$ ở các phép chiếu đổ vào nhánh residual ($L$ là số tầng). Một số mô hình gần đây (μP, các họ Llama/OLMo) chọn thang khởi tạo theo độ rộng để siêu tham số chuyển được từ mô hình nhỏ sang mô hình lớn.
- **Fine-tuning** từ mô hình tiền huấn luyện vẫn là "cách khởi tạo" hiệu quả nhất khi có mô hình phù hợp: chỉ tầng mới thêm vào (như đầu phân loại) mới cần khởi tạo ngẫu nhiên.

## Tài liệu tham khảo

- Goodfellow, Bengio, Courville. *Deep Learning*, mục 8.4 Chiến lược khởi tạo tham số.
- Ngụy Tú Tham. *Giải tích mạng neuron tích chập* (解析卷积神经网络), chương 7.
- [AI-EDU 15.1 Khởi tạo ma trận trọng số](https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC7%E6%AD%A5%20-%20%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/15.1-%E6%9D%83%E9%87%8D%E7%9F%A9%E9%98%B5%E5%88%9D%E5%A7%8B%E5%8C%96.html)
- [神经网络之权重初始化](https://www.cnblogs.com/makefile/p/init-weight.html)
- [神经网络参数初始化小结](https://zhuanlan.zhihu.com/p/133835463)
- Glorot, Bengio. [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf). AISTATS 2010.
- He et al. [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852). ICCV 2015.
