# CNN hiệu quả: RepVGG

Bài này đọc bài báo *RepVGG: Making VGG-style ConvNets Great Again* (Ding và cộng sự, CVPR 2021), cùng ACNet là tiền thân của nó. Trang [Code](dlnote/backbone_code_vi) đi kèm kiểm tra bằng số đẳng thức cộng kernel của tích chập, chuyển một khối ACBlock và một khối RepVGG từ dạng huấn luyện (nhiều nhánh, có BN) sang dạng suy luận (một tích chập 3×3) và so đầu ra, rồi đo độ trễ của hai dạng.

<!-- theory: qa/ch04_vi#4.5 -->

## 1 Kiến thức nền

### 1.1 Nhìn lại VGG và ResNet

1. Họ `VGGNet` có `5` giai đoạn tích chập. Lấy `VGG16` làm ví dụ: mỗi giai đoạn có `2–3` tầng tích chập, cuối mỗi giai đoạn có một tầng max pooling để giảm kích thước `feature map`. Số kernel trong một giai đoạn như nhau, giai đoạn sau nhiều hơn giai đoạn trước: `64-128-256-512-512`. Số tầng tích chập mỗi giai đoạn của `VGG16` là `2-2-3-3-3`, tổng cộng $2+2+3+3+3 = 13$ tầng, thêm ba tầng phân loại kết nối đầy đủ là `16` tầng, nên gọi là `VGG16`. Số kernel của 5 giai đoạn tăng gấp đôi lần lượt: `64-128-256-512-512`. Bảng cấu hình VGG như hình dưới.

![VGG](../images/backbone/VGG.png)

Hình 1.1: Cấu hình VGG (bảng 1 của bài báo VGG).

2. `ResNet18` cũng có `5` giai đoạn tích chập: một tầng tích chập $7 \times 7$ riêng, và bốn `layer` (bốn giai đoạn) do hàm `make_layer` tạo ra trong code; mỗi `layer` có 2 khối residual cơ bản (basic block). Mỗi `basic block` có `2` tầng tích chập nên các khối residual có tổng $(2+2+2+2)\times 2=16$ tầng, cộng tầng tích chập đầu và tầng phân loại cuối là `18` tầng, nên gọi là `ResNet18`. Số kernel của 5 giai đoạn: `64-64-128-256-512`, cũng tăng gấp đôi qua các giai đoạn sau. Bảng tham số ResNet-18 như dưới.

![Bảng tham số ResNet-18](../images/RepVGG/resnet18_parameter.png)

Hình 1.2: Kích thước đầu ra và cấu hình từng giai đoạn của ResNet-18.

**Tóm lại: dùng kernel nhỏ thay kernel lớn, chia tích chập thành các giai đoạn, số kernel tăng gấp đôi và kích thước `Feature Map` giảm một nửa qua mỗi giai đoạn.**

### 1.2 Tính lượng truy cập bộ nhớ (MAC)

`MAC` (memory access cost), lượng truy cập bộ nhớ, còn gọi là lượng sử dụng bộ nhớ. `MAC` của mỗi tầng trong `CNN` gồm ba phần: đọc `feature map` vào, đọc trọng số (đọc `DDR`), và ghi `feature map` ra (ghi `DDR`).

Lấy tầng tích chập làm ví dụ: đầu vào (`Cin, Hin, Win`), đầu ra (`Cout, Hout, Wout`), kernel (`Cout, Cin, K, K`), MAC lý thuyết (thường nhỏ hơn MAC thực tế) là:

> `feature map` thường có dạng (`N, C, H, W`). Chỉ số `MAC` thường dùng cho suy luận trên thiết bị đầu cuối, nơi thường suy luận từng khung hình, tức `N = 1` (`batch_size = 1`), khác với huấn luyện, nơi `batch_size` thường lớn hơn 1.

```python
input = Hin * Win * Cin             # feature map vào
output = Hout * Wout * Cout         # feature map ra
weights = K * K * Cin * Cout + bias # bias là độ lệch của tầng
ddr_read = input + weights
ddr_write = output
MAC = ddr_read + ddr_write
```

Xem thêm [bài độ phức tạp của CNN](dlnote/deploy_flops_vi).

### 1.3 Tích chập và tích ma trận

#### Tích vô hướng

- Trong **toán học**, tích vô hướng (`Dot Product`, còn gọi là Scalar Product) là phép toán đại số nhận hai dãy số cùng độ dài (thường là vector toạ độ) và trả về một số.
- Trong **hình học Euclid**, tích vô hướng của hai vector toạ độ Descartes thường gọi là tích trong (`Inner Product`); xem không gian tích trong.

Phép tích chập có thể xem là một loạt tích trong, tương đương phép nhân ma trận (im2col), nên tích chập là phép **tuyến tính** theo kernel và theo đầu vào.

Một số tính chất thường dùng của tích ma trận:

- **Phân phối**: $A(B+C) = AB+AC$
- **Kết hợp**: $A(BC) = (AB)C$
- **Giao hoán**: phần lớn trường hợp không thoả, tức $AB \neq BA$.

> **Biên tập:** Bản gốc viết luật phân phối là $A(B+C) = AB+BC$ (đúng là $AB + AC$), và kết luận "nên tích chập thoả luật giao hoán, kết hợp…". Điều cần cho tái tham số hoá chỉ là tính **tuyến tính**: $I * K_1 + I * K_2 = I * (K_1 + K_2)$, chính là luật phân phối. (Tích chập theo nghĩa toán học giữa hai hàm thì giao hoán, nhưng tầng tích chập nhiều kênh trong mạng neuron thì không.)

### 1.4 Hiểu ACNet

Trước khi học [ACNet](https://arxiv.org/abs/1908.03930), cần hiểu **đẳng thức cộng của tích chập, cơ sở lý thuyết của ý tưởng "tái tham số hoá cấu trúc"**, như hình 1.3.

> Tái tham số hoá cấu trúc (structural re-parameterization) là: trước hết tạo một loạt cấu trúc (thường dùng khi huấn luyện), rồi chuyển tham số của chúng một cách tương đương sang một bộ tham số khác (thường dùng khi suy luận), nhờ đó chuyển tương đương loạt cấu trúc đó sang một loạt cấu trúc khác.

![Đẳng thức cộng của tích chập](../images/RepVGG/Figure2.png)

Hình 1.3: Tính cộng được của tích chập: hai tích chập kích thước tương thích trên cùng đầu vào cộng lại bằng một tích chập với tổng hai kernel (hình 2 của bài báo ACNet).

Đẳng thức dưới nói rằng: với feature map vào $I$, tích chập $I$ với $K^{(1)}$, tích chập $I$ với $K^{(2)}$ rồi cộng kết quả, cũng bằng cộng từng phần tử $K^{(1)}$ với $K^{(2)}$ trước rồi tích chập với $I$ (hiểu qua luật phân phối của tích ma trận). Đây là cơ sở để `ACNet` **không thêm chút tính toán nào khi suy luận**; đổi lại, lúc huấn luyện tính toán nhiều hơn, thời gian huấn luyện dài hơn, và cần nhiều bộ nhớ GPU hơn.

$$I \ast K^{(1)} + I \ast K^{(2)} = I \ast (K^{(1)} \oplus K^{(2)})$$

Ở đây $\oplus$ là phép cộng kernel sau khi căn tâm (kernel nhỏ được đệm 0 cho bằng kernel lớn), với điều kiện hai tích chập có cùng stride và cửa sổ trượt khớp nhau.

Đóng góp của `ACNet` gồm hai giai đoạn:

- **Huấn luyện**: thay mỗi tầng tích chập $3 \times 3$ của mạng có sẵn bằng ba tầng: tích chập $3 \times 1$ + tích chập $1 \times 3$ + tích chập $3 \times 3$, rồi cộng kết quả ba tầng làm đầu ra. Vì các kernel $1 \times 3$ và $3 \times 1$ thêm vào là bất đối xứng nên gọi là `Asymmetric Convolution`. Thí nghiệm trong bài báo (bảng 4) cho thấy thêm kernel ngang $1 \times 3$ giúp mô hình bền vững hơn với ảnh bị lật trên dưới; kernel dọc $3 \times 1$ tương tự với lật trái phải.
- **Suy luận**: hợp nhất ba kernel; khi cài đặt, chỉ cần dùng tham số của kernel đã hợp nhất để khởi tạo mạng gốc.

Phép hợp nhất tích chập lúc suy luận đi kèm `BN`: hợp nhất sau khi đã gộp BN của từng nhánh; thí nghiệm của bài báo cho thấy đặt BN trên từng nhánh rồi mới hợp nhất cho kết quả tốt hơn. Sơ đồ hợp nhất ở giai đoạn suy luận như dưới (bỏ qua $\varepsilon$ của BN):

![Hợp nhất tích chập khi suy luận](../images/RepVGG/convolutional_layer_fusion_at_the_inference_stage.png)

Hình 1.4: Hợp nhất BN vào từng nhánh rồi cộng các kernel 3×3, 1×3, 3×1 thành một kernel 3×3 (hình 3 của bài báo ACNet).

Code chuyển trọng số của `ACBlock` từ dạng huấn luyện sang dạng suy luận:

```python
def get_equivalent_kernel_bias(self):
    hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
    ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
    square_k, square_b = self._fuse_bn_tensor(self.square_conv, self.square_bn)
    self._add_to_square_kernel(square_k, hor_k)
    self._add_to_square_kernel(square_k, ver_k)
    return square_k, hor_b + ver_b + square_b

def _fuse_bn_tensor(self, conv, bn):
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

def _add_to_square_kernel(self, square_kernel, asym_kernel):
    asym_h = asym_kernel.size(2)
    asym_w = asym_kernel.size(3)
    square_h = square_kernel.size(2)
    square_w = square_kernel.size(3)
    # cộng kernel ngang và dọc vào đúng vị trí giữa của kernel vuông
    square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
            square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel
```

#### Cài đặt ACBlock bằng PyTorch

Tác giả ACNet công bố [code](https://github.com/DingXiaoH/ACNet/blob/master/acb.py). Khối dạng huấn luyện `ACBlock`, thay tích chập $3\times 3$ gốc bằng tích chập $3 \times 3 + 3 \times 1 + 1 \times 3$:

```python
import torch.nn as nn

class CropLayer(nn.Module):
    """Bỏ các hàng hoặc cột thừa ra do padding của tích chập 3x3."""
    # (-1, 0): bỏ hàng đầu và hàng cuối của feature map; (0, -1): bỏ cột đầu và cột cuối
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class ACBlock(nn.Module):
    """Khối 3x3 + 1x3 + 3x1 của bài báo ACNet."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride, padding=ver_conv_padding, dilation=dilation, groups=groups,
                                      bias=False, padding_mode=padding_mode)
            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride, padding=hor_conv_padding, dilation=dilation, groups=groups,
                                      bias=False, padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_bn(self.square_conv(input))      # tích chập 3x3
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_bn(self.ver_conv(vertical_outputs))      # tích chập 3x1
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_bn(self.hor_conv(horizontal_outputs))  # tích chập 1x3
            # cộng thẳng đầu ra của ba nhánh
            return square_outputs + vertical_outputs + horizontal_outputs
```

> **Biên tập:** Bản gốc dẫn tới tệp `custom_layers/crop_layer.py`; đoạn code trên là tệp `acb.py` của kho ACNet. Trang Code chạy `ACBlock` trên đầu vào ngẫu nhiên (BN có thống kê ngẫu nhiên, chế độ `eval`), chuyển sang dạng `deploy` bằng `get_equivalent_kernel_bias` và so đầu ra hai dạng.

## 2 Tóm tắt bài báo RepVGG

Đóng góp chính của bài báo:

- Đề xuất kiến trúc CNN đơn giản mà mạnh là RepVGG; so với các kiến trúc như `EfficientNet`, `RegNet`, `RepVGG` cân bằng độ chính xác và tốc độ tốt hơn;
- Đề xuất dùng kỹ thuật tái tham số hoá để tách rời cấu trúc huấn luyện và cấu trúc suy luận của kiến trúc thẳng (`plain`);
- Kiểm chứng hiệu quả của `RepVGG` trên phân loại ảnh, phân đoạn ngữ nghĩa và các bài toán khác.

## 3 Định nghĩa mô hình RepVGG

Mạng kiểu `VGG` thường được hiểu là:

1. không có nhánh nào, tức kiến trúc thẳng (`plain`, `feed-forward`);
2. chỉ dùng tích chập $3 \times 3$;
3. chỉ dùng `ReLU` làm hàm kích hoạt.

Năm ưu điểm của mạng tối giản kiểu `VGG`:

1. **Tích chập 3×3 rất nhanh.** Trên `GPU`, mật độ tính toán (lượng tính toán lý thuyết chia cho thời gian) của tích chập 3×3 có thể gấp bốn lần tích chập 1×1 hay 5×5 (nhờ thuật toán Winograd trong cuDNN).
2. **Kiến trúc một đường rất nhanh vì mức song song cao.** Cùng lượng tính toán, phép tính "lớn và liền" hiệu quả hơn hẳn phép tính "nhỏ và vụn". Nghiên cứu đã cho thấy mô hình song song cao suy luận nhanh hơn mô hình song song thấp (G3 của [ShuffleNet v2](dlnote/bb_shufflenet_vi)).
3. **Kiến trúc một đường tiết kiệm bộ nhớ.** Ví dụ, kết nối tắt của ResNet tuy không tốn tính toán, nhưng làm bộ nhớ đỉnh tăng gấp đôi (phải giữ đầu vào của khối cho tới phép cộng).
4. **Kiến trúc một đường linh hoạt hơn, dễ đổi độ rộng của từng tầng (như khi cắt tỉa).**
5. **Thân mạng RepVGG chỉ có một loại toán tử**: tích chập `3x3` rồi `ReLU`. Khi thiết kế chip chuyên dụng, với diện tích hay giá thành cho trước, có thể tích hợp rất nhiều đơn vị tính tích chập `3x3`–`ReLU` để đạt hiệu suất cao, và đặc tính tiết kiệm bộ nhớ của kiến trúc một đường cũng giúp giảm đơn vị lưu trữ.

Kiến trúc cơ bản của `RepVGG` tóm gọn là: chồng hơn 20 tầng tích chập $3 \times 3$, chia thành 5 giai đoạn, tầng đầu của mỗi giai đoạn giảm kích thước bằng `stride=2`, mỗi tầng tích chập dùng ReLU làm hàm kích hoạt.

## 4 Cấu trúc khối RepVGG

> Đây là đổi mới về cấu trúc mô hình.

So với các cấu trúc nhiều nhánh (như ResNet, Inception, DenseNet), các kiến trúc thẳng (`plain`) kiểu `VGG` những năm gần đây ít được quan tâm, chủ yếu vì hiệu năng kém. Có nghiên cứu [1] cho rằng một cách giải thích cho hiệu năng tốt của ResNet là cấu trúc nhánh (shortcut) tạo ra một tổ hợp ngầm (implicit ensemble) của rất nhiều mô hình con (mỗi lần gặp một nhánh, tổng số đường đi tăng gấp đôi); kiến trúc một đường rõ ràng không có đặc điểm này.

Thiết kế của `RepVGG` lấy cảm hứng từ `ResNet`: cấu trúc nhiều nhánh không thân thiện với suy luận nhưng thân thiện với huấn luyện, nên tác giả đề xuất ý tưởng mới: **huấn luyện một mô hình nhiều nhánh, khi suy luận chuyển tương đương sang mô hình một đường**. Theo nhánh `identity` và nhánh $1 \times 1$ của `ResNet`, khối tích chập được thiết kế như sau:

$$y = x + g(x) + f(x)$$

trong đó $x$, $g(x)$, $f(x)$ lần lượt là ánh xạ đồng nhất, tích chập $1 \times 1$ và tích chập $3 \times 3$ (mỗi nhánh có BN riêng). Tức là khi huấn luyện, mỗi tầng tích chập `3x3` được thêm một nhánh tích chập `1x1` song song và một nhánh đồng nhất, tạo thành một `RepVGG Block`. Thiết kế này mượn cách làm của `ResNet`, khác ở chỗ `ResNet` cứ hai hoặc ba tầng mới thêm một nhánh, còn `RepVGG` thêm hai nhánh ở mỗi tầng (khi huấn luyện). Nhánh đồng nhất chỉ có khi số kênh vào bằng số kênh ra và stride bằng 1 (tức không có ở tầng đầu mỗi giai đoạn).

![Hình 2 của RepVGG](../images/RepVGG/figure_2.png)

Hình 4.1: (A) ResNet. (B) RepVGG lúc huấn luyện: mỗi tầng có nhánh 1×1 và nhánh đồng nhất. (C) RepVGG lúc suy luận: chỉ còn chồng tích chập 3×3 và ReLU (hình 2 của bài báo).

Khi huấn luyện, mô hình `RepVGG` được tạo bằng cách chồng các `RepVGG Block` trên; khi suy luận, khối được chuyển sang dạng $y=h(x)$, tham số của $h(x)$ tính được bằng tổ hợp tuyến tính từ mô hình đã huấn luyện.

## 5 Tái tham số hoá cấu trúc của khối RepVGG

Huấn luyện dùng cấu trúc tích chập nhiều nhánh, suy luận hợp nhất các nhánh thành một tầng tích chập $3 \times 3$ duy nhất. Nhờ tính tuyến tính (cụ thể là tính cộng được) của tích chập, ba nhánh của mỗi khối RepVGG có thể gộp thành một tầng tích chập $3 \times 3$ (chuyển đổi tương đương); hình 5.1 mô tả chi tiết quá trình này.

Bài báo dùng $W^{(3)} \in \mathbb{R}^{C_2 \times C_1 \times 3 \times 3}$ cho tầng tích chập có kernel $(C_2, C_1, 3, 3)$, $W^{(1)} \in \mathbb {R}^{C_{2} \times C_{1}}$ cho nhánh tích chập $1 \times 1$ với $C_1$ kênh vào và $C_2$ kênh ra; $\mu^{(3)}, \sigma^{(3)}, \gamma^{(3)}, \beta^{(3)}$ là tham số `BatchNorm` sau tích chập $3 \times 3$ (trung bình, độ lệch chuẩn, hệ số co giãn, độ lệch), $\mu^{(1)}, \sigma^{(1)}, \gamma^{(1)}, \beta^{(1)}$ cho BN sau nhánh $1 \times 1$, $\mu^{(0)}, \sigma^{(0)}, \gamma^{(0)}, \beta^{(0)}$ cho BN của nhánh `identity`. Gọi $M^{(1)} \in \mathbb{R}^{N \times C_1 \times H_1 \times W_1}$, $M^{(2)} \in \mathbb{R}^{N \times C_2 \times H_2 \times W_2}$ là đầu vào và đầu ra, $\ast$ là phép tích chập. Khi $C_2 = C_1, H_1 = H_2, W_1 = W_2$:

$$
\begin{aligned}
M^{(2)} &= \mathrm{bn}(M^{(1)} \ast W^{(3)}, \mu^{(3)}, \sigma^{(3)}, \gamma^{(3)}, \beta^{(3)}) \\
&+ \mathrm{bn}(M^{(1)} \ast W^{(1)}, \mu^{(1)}, \sigma^{(1)}, \gamma^{(1)}, \beta^{(1)}) \\
&+ \mathrm{bn}(M^{(1)}, \mu^{(0)}, \sigma^{(0)}, \gamma^{(0)}, \beta^{(0)}).
\end{aligned}\tag{1}
$$

Nếu không có nhánh `identity`, đẳng thức chỉ còn hai số hạng đầu. Ở đây `bn` là hàm BN lúc suy luận; với $1 \leq i \leq C_2$:

$$
\mathrm{bn}(M, \mu, \sigma, \gamma, \beta)_{:,i,:,:} = (M_{:,i,:,:} - \mu_i) \frac{\gamma_i}{\sigma_i} + \beta_i. \tag{2}
$$

Trước hết chuyển mỗi `BN` cùng tầng tích chập trước nó thành một **tích chập có vector bias** ("hấp thụ BN"). Gọi $\{W', b'\}$ là kernel và bias của tích chập sau khi hấp thụ BN:

> Khi suy luận, tầng tích chập và tầng `BN` sau nó có thể chuyển tương đương thành một tầng tích chập có `bias`; nguyên lý xem [bài hợp nhất Conv và BN](dlnote/deploy_fuse_vi).

$$
W_{i,:,:,:}^{'} = \frac{\gamma_i}{\sigma_i} W_{i,:,:,:}, \quad b_{i}^{'} = -\frac{\mu_{i} \gamma_i}{\sigma_i} + \beta_{i}. \tag{3}
$$

Dễ chứng minh rằng với $1 \leq i \leq C_2$:

$$
\mathrm{bn}(M \ast W,\mu,\sigma,\gamma,\beta)_{:,i,:,:} = (M \ast W^{'})_{:,i,:,:} + b_{i}^{'}. \tag{4}
$$

> **Biên tập:** Công thức (2) của bản gốc thiếu chỉ số $i$ ở $\beta$; đã sửa thành $\beta_i$. Ở đây $\sigma_i$ là độ lệch chuẩn đã cộng $\varepsilon$, tức $\sqrt{\mathrm{var}_i + \varepsilon}$.

Công thức (4) cũng đúng cho nhánh `identity`, vì có thể xem `identity` là một tích chập $1\times 1$ có kernel là **ma trận đơn vị**. Tới đây đã xong nguyên lý và công thức hợp nhất tích chập với `BN` cho cả ba nhánh, tương ứng bước thứ hai (hấp thụ BN) trong hình 5.1.

Bước cuối là hợp nhất ba nhánh, tức hợp nhất ba tầng tích chập; đầu ra của mỗi khối `RepVGG` trước và sau khi chuyển hoàn toàn như nhau, nguyên lý như bài `ACNet` trước đó của tác giả. Sau các bước trên, khối `RepVGG` có một kernel $3 \times 3$, hai kernel $1 \times 1$ và ba vector `bias`. Cộng ba vector bias được bias của tầng mới; đệm (`pad`) 0 cho các kernel $1 \times 1$ thành $3 \times 3$ rồi cộng từng phần tử (`element-wise add`) với kernel $3 \times 3$, được kernel $3 \times 3$ của tầng đã hợp nhất.

Tới đây đã xong quá trình hợp nhất ba nhánh, tương ứng bước thứ ba trong hình 5.1.

> Chi tiết về kernel: kernel $3 \times 3$ và $1 \times 1$ có cùng `stride`, còn `padding` của nhánh $1 \times 1$ nhỏ hơn nhánh $3 \times 3$ đúng `1` (như padding 1 và 0), để đầu ra hai nhánh khớp vị trí.

![Hình 4 của RepVGG](../images/RepVGG/figure_4.png)

Hình 5.1: Tái tham số hoá một khối RepVGG: (1) hấp thụ BN vào từng nhánh; (2) coi nhánh đồng nhất là tích chập 1×1 với kernel đơn vị, đệm các kernel 1×1 thành 3×3; (3) cộng ba kernel và ba bias (hình 4 của bài báo).

Từ quá trình chuyển đổi trên có thể hiểu bản chất của **tái tham số hoá cấu trúc**: cấu trúc lúc huấn luyện ứng với một bộ tham số, cấu trúc ta muốn lúc suy luận ứng với một bộ tham số khác; chỉ cần chuyển tương đương được bộ trước sang bộ sau là chuyển tương đương được cấu trúc trước sang cấu trúc sau.

Trang Code chuyển một chồng 8 khối RepVGG (64 kênh, 56×56) sang dạng suy luận: đầu ra khớp tới $5\times10^{-5}$ (trên giá trị cỡ 30), và trên CPU dạng suy luận nhanh hơn dạng huấn luyện khoảng 40%.

## 6 Kết luận

Cuối cùng cần lưu ý, `RepVGG` là mô hình hiệu quả thiết kế cho `GPU` và phần cứng chuyên dụng, nhắm tốc độ cao và tiết kiệm bộ nhớ, ít quan tâm số tham số và lượng tính toán lý thuyết. Trên thiết bị sức tính thấp, nó có thể không phù hợp bằng họ MobileNet và ShuffleNet.

## 7 Vấn đề của RepVGG

Mô hình suy luận của `RepVGG` **rất khó lượng tử hoá sau huấn luyện** (`Post-Training Quantization`, `PTQ`). Ví dụ, chỉ dùng `INT8 PTQ` đơn giản, độ chính xác trên ImageNet của mô hình `RepVGG` tụt xuống `54,55%`.

Bài báo `RepOptimizer` nghiên cứu vấn đề khó lượng tử hoá của cấu trúc tái tham số hoá, và thấy việc hợp nhất nhánh và hấp thụ `BN` làm độ lệch chuẩn của phân phối trọng số tăng đáng kể. **Phân phối trọng số bất thường** này lại sinh ra phân phối kích hoạt có giá trị quá lớn, khiến sai số lượng tử hoá của tầng đó quá lớn, nên mô hình mất nhiều độ chính xác.

> Tham khảo bài của đội kỹ thuật Meituan: [Thực chiến lượng tử hoá và triển khai YOLOv6 tại Meituan](https://tech.meituan.com/2022/09/22/yolov6-quantization-in-meituan.html).

Trong bài báo cải tiến [RepOptimizer](https://arxiv.org/abs/2205.15242), lượng tử hoá trực tiếp mô hình `RepOpt-VGG` chỉ làm giảm `2,5%` độ chính xác trên `ImageNet`.

> `RepOptimizer`: tái tham số hoá optimizer thay vì cấu trúc: kiến trúc kiểu VGG + optimizer đặc biệt (gradient được nhân với các hệ số suy ra từ cấu trúc nhiều nhánh) = huấn luyện nhanh + hiệu năng mạnh, và mô hình huấn luyện ra đã là một đường, không cần chuyển đổi.

> **Biên tập:** Cách khắc phục khác, được YOLOv6 dùng, là huấn luyện có lượng tử hoá (QAT) trên mô hình đã tái tham số hoá, hoặc lượng tử hoá theo kênh (per-channel) cho trọng số; xem [bài lượng tử hoá](dlnote/compress_quant_vi). Tái tham số hoá cấu trúc nay có mặt trong MobileOne, YOLOv6/v7, FastViT và nhiều mạng nhẹ khác.

## Tài liệu tham khảo

1. Veit et al. [Residual Networks Behave Like Ensembles of Relatively Shallow Networks](https://arxiv.org/abs/1605.06431). NeurIPS 2016.
2. Ding et al. [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](https://arxiv.org/abs/1908.03930). ICCV 2019.
3. Ding et al. [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697). CVPR 2021.
4. [Wikipedia: Phép nhân ma trận](https://zh.wikipedia.org/wiki/%E7%9F%A9%E9%99%A3%E4%B9%98%E6%B3%95)
5. [RepVGG: kiến trúc tối giản, hiệu năng SOTA, làm mô hình kiểu VGG vĩ đại trở lại](https://zhuanlan.zhihu.com/p/344324470)
6. [Hợp nhất BN khi suy luận, dễ dàng tăng tốc khoảng 5%](https://mp.weixin.qq.com/s/P94ACKuoA0YapBKlrgZl3A)
7. [[Thiết kế cấu trúc CNN] Mẹo tăng độ chính xác không tốn kém: ACNet](https://zhuanlan.zhihu.com/p/131282789)
8. Ding et al. [Re-parameterizing Your Optimizers rather than Architectures](https://arxiv.org/abs/2205.15242) (RepOptimizer). ICLR 2023.
