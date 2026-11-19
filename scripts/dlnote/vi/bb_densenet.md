# Backbone kinh điển: DenseNet

Trang [Code](dlnote/backbone_code_vi) đi kèm chạy đoạn code DenseNet-BC ở mục 4 (tệp `densenet.py` của bản gốc): đếm tham số của cấu hình CIFAR ($L=100$, $k=12$), kiểm tra công thức số kênh $k_0 + k(l-1)$, và so số tham số, MACs, độ trễ của DenseNet-121 với ResNet-50.

<!-- theory: qa/ch04_vi#Bổ sung: ResNet và DenseNet ; qa/ch09_vi#9.11 -->

## 1 Tóm tắt

`ResNet` cho thấy chỉ cần tạo "kết nối tắt" (shortcut) giữa tầng trước và tầng sau là giúp gradient lan truyền ngược khi huấn luyện, nhờ đó huấn luyện được CNN "sâu" hơn. `DenseNet` có ý tưởng cơ bản như `ResNet`, nhưng tạo **kết nối dày đặc** (dense connection) giữa mọi tầng trước với tầng sau. Mạng tích chập truyền thống $L$ tầng có $L$ kết nối (mỗi tầng nối với tầng ngay sau nó), còn DenseNet có $L(L+1)/2$ kết nối.

Trong DenseNet, mỗi tầng nối trực tiếp với mọi tầng trước nó, để **tái sử dụng đặc trưng**; đồng thời mỗi tầng được thiết kế rất "hẹp" (ít feature map, ít bộ lọc), tức chỉ học rất ít feature map mới (cực đoan nhất là mỗi tầng chỉ học một feature map), nhằm giảm dư thừa.

## 2 Cấu trúc mạng

Mô hình `DenseNet` chủ yếu gồm các `DenseBlock`.

Viết bằng công thức, đầu ra tầng $l$ của mạng thẳng (`plain`) truyền thống là:

$$\mathrm{x}_l = H_l(\mathrm{x}_{l-1})$$

Khối residual thêm một ánh xạ đồng nhất (kết nối `shortcut`):

$$\mathrm{x}_l = H_l(\mathrm{x}_{l-1}) + \mathrm{x}_{l-1}$$

Còn trong khối dày đặc (`DenseBlock`), mỗi tầng nhận đầu vào là kết quả nối (`concat`) của mọi tầng trước:

$$\mathrm{x}_l = H_l([\mathrm{x}_0,\mathrm{x}_1,\dots,\mathrm{x}_{l-1}])$$

$[\mathrm{x}_0,\mathrm{x}_1,\dots,\mathrm{x}_{l-1}]$ là phép nối các feature map đầu ra của các tầng $0,\dots,l-1$. Điều này ngầm đòi hỏi các feature map trong một DenseBlock có cùng kích thước không gian. $H_l(\cdot)$ là hàm biến đổi phi tuyến (non-linear transformation), gồm BN (`Batch Normalization`), ReLU và tầng Conv.

> **Biên tập:** Bản gốc viết chỉ số dưới là $\mathrm{x}_l-1$ (tức $\mathrm{x}_l$ trừ 1); đúng là $\mathrm{x}_{l-1}$. Thứ tự BN → ReLU → Conv chính là tiền kích hoạt của [ResNet v2](dlnote/bb_resnetv2_vi).

Sơ đồ `DenseBlock`:

![Sơ đồ DenseBlock](../images/densenet/densenet_block_structure_diagram.png)

Hình 2.1: Một DenseBlock 5 tầng với tốc độ tăng $k=4$: mỗi tầng nhận mọi feature map trước nó làm đầu vào (hình 1 của bài báo DenseNet).

Trong thiết kế `DenseBlock`, tác giả nhấn mạnh tham số $k$, gọi là **tốc độ tăng trưởng** (`growth rate`) của mạng, thực chất là số bộ lọc (số kênh ra) của mỗi tầng tích chập $3\times 3$ trong `DenseBlock`. Nếu mỗi hàm $H_l(\cdot)$ cho ra $k$ feature map, thì tầng thứ $l$ nhận $k_0 + k\times (l-1)$ feature map, với $k_0$ là số feature map vào `DenseBlock` (số kênh vào của tầng tích chập đầu tiên). Điểm khác biệt rõ nhất giữa `DenseNet` và các mạng khác là $k$ có thể rất nhỏ, như $k=12$, tức mạng rất "hẹp", mà không giảm độ chính xác. Như bảng 2 dưới đây.

![Kết quả so sánh của DenseNet](../images/densenet/densenet_comparison_experimental_results.png)

Hình 2.2: Sai số trên CIFAR và SVHN của DenseNet với các độ sâu $L$ và tốc độ tăng $k$ khác nhau, so với các mạng khác (bảng 2 của bài báo DenseNet).

> **Biên tập:** Bản gốc gọi bảng này là "bảng 4"; trong bài báo đó là bảng 2.

Để giữ kích thước feature map của các tầng trong `DenseBlock` như nhau, tác giả chèn **tầng chuyển tiếp** (`transition`) giữa hai `DenseBlock`, gồm average pooling $2\times 2$ stride 2 và tích chập $1\times 1$, cụ thể là **BN + ReLU + Conv 1×1 + AvgPooling 2×2**. Tầng `transition` giảm kích thước feature map và giảm số kênh.

> `CNN` thường giảm kích thước feature map bằng tầng pooling hoặc tầng tích chập có stride > 1 (như tích chập 3×3 stride 2).

Hình dưới là một mạng `DenseNet` gồm `3` `DenseBlock` (thường dùng `4`), nối với nhau qua các tầng `Transition`.

![Cấu trúc DenseNet](../images/densenet/densenet_network_structure_diagram.png)

Hình 2.3: DenseNet với ba DenseBlock; giữa hai khối là tầng chuyển tiếp (tích chập và pooling) làm giảm kích thước (hình 2 của bài báo DenseNet).

Giống `ResNet`, `DenseNet` cũng có đơn vị `bottleneck` cho các DenseNet sâu hơn. Đơn vị `Bottleneck` có dạng BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3); mạng gồm các đơn vị dày đặc có `bottleneck` được gọi là `DenseNet-B`.

> `Bottleneck` là "nút cổ chai", một đầu lớn một đầu nhỏ: tích chập 1×1 có nhiều kênh ra ($4k$), tích chập 3×3 có ít kênh ra ($k$).

> **Biên tập:** Bản gốc chưa nói tới chữ "C". Nếu tầng chuyển tiếp còn giảm số kênh theo hệ số nén $\theta < 1$ (bài báo dùng $\theta = 0{,}5$, tham số `compression` trong code ở mục 4), mạng được gọi là `DenseNet-C`; có cả bottleneck lẫn nén là `DenseNet-BC`.

Với tập `ImageNet`, ảnh vào có kích thước $224\times 224$, mạng dùng `DenseNet-BC` có `4` `DenseBlock`; tầng đầu là tích chập $7\times 7$ `stride=2`, tiếp theo là MaxPooling $3\times 3$ `stride=2`, rồi tới các `DenseBlock`. Cấu hình cho `ImageNet` như bảng 1:

![Bảng cấu hình DenseNet](../images/densenet/densenet_series_network_parameter_table.png)

Hình 2.4: Cấu hình DenseNet-121, 169, 201, 264 cho ImageNet; mỗi "conv" tương ứng chuỗi BN-ReLU-Conv (bảng 1 của bài báo DenseNet).

Tốc độ tăng của mọi tầng tích chập trong mạng cho ImageNet là $k=32$.

## 3 Ưu điểm

1. **Tiết kiệm tham số**
2. **Tiết kiệm tính toán**
3. **Chống quá khớp**

> Lưu ý: về sau VoVNet chỉ ra rằng dù DenseNet ít tham số, hiệu quả suy luận của nó không cao.

Để đạt cùng độ chính xác trên `ImageNet`, `DenseNet` cần chưa tới một nửa số tham số và lượng tính toán của `ResNet`. Với công nghiệp, mô hình nhỏ (ít tham số) **tiết kiệm đáng kể băng thông và chi phí lưu trữ**.

> Bản gốc viết thêm: "mô hình ít tham số thì chắc chắn cũng ít tính toán".

> **Biên tập:** Câu này sai: số tham số và lượng tính toán không tỉ lệ với nhau. Tầng FC 2048→1000 của ResNet-50 có 2 triệu tham số nhưng chỉ 2 triệu MACs, còn một tầng tích chập 3×3 64→64 trên ảnh 56×56 chỉ có 37 nghìn tham số nhưng tốn 116 triệu MACs. Chính DenseNet là ví dụ: DenseNet-121 có 8 triệu tham số (ít hơn ResNet-50 ba lần) nhưng 2,9 G MACs (khoảng 70% của ResNet-50), và trên CPU của trang Code nó chạy xấp xỉ ResNet-50 (khoảng 61 so với 63 ms), không nhanh hơn tương ứng, vì phép concat liên tục tạo lượng truy cập bộ nhớ lớn.

Tác giả thấy qua thí nghiệm rằng **`DenseNet` khó bị quá khớp, nhất là khi tập dữ liệu không lớn**. Trong một số bài toán phân đoạn ảnh và phát hiện đối tượng, mô hình dựa trên `DenseNet` thường có thể bỏ bước tiền huấn luyện trên `ImageNet`, huấn luyện thẳng từ khởi tạo ngẫu nhiên mà vẫn đạt kết quả bằng hoặc tốt hơn.

Tác giả giải thích trực quan vì sao `DenseNet` chống quá khớp: đặc trưng mỗi tầng trích ra là một biến đổi phi tuyến của dữ liệu vào, và càng sâu thì biến đổi càng phức tạp (hợp của nhiều hàm phi tuyến hơn). Bộ phân loại của mạng thông thường chỉ dựa vào đặc trưng của tầng cuối (phức tạp nhất), còn DenseNet dùng được cả đặc trưng độ phức tạp thấp của các tầng nông, nên dễ có hàm quyết định trơn, tổng quát hoá tốt hơn.

Tính tổng quát hoá tốt hơn của `DenseNet` có cơ sở lý thuyết: bài báo *AdaNet: Adaptive Structural Learning of Artificial Neural Networks*, đăng arXiv gần như cùng lúc với DenseNet, chứng minh (định lý 1) rằng các cấu trúc mạng giống DenseNet có cận sai số tổng quát hoá nhỏ hơn.

## 4 Code

Dưới đây là code DenseNet tiết kiệm bộ nhớ (tệp `densenet.py` của bản gốc, lấy từ kho [efficient_densenet_pytorch](https://github.com/gpleiss/efficient_densenet_pytorch) của Geoff Pleiss, đồng tác giả bài báo *Memory-Efficient Implementation of DenseNets*).

```python
# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:  # thêm dropout để tăng khả năng tổng quát hoá
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):
        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        # tích chập đầu tiên
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))
        # các DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
        # BN cuối
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        # tầng tuyến tính
        self.classifier = nn.Linear(num_features, num_classes)
        # khởi tạo
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
```

Điểm tiết kiệm bộ nhớ nằm ở `efficient=True`: phần BN-ReLU-Conv 1×1 trên các feature map đã nối được bọc trong `torch.utils.checkpoint`, nên khi lan truyền xuôi không lưu kết quả concat và BN (mỗi tầng một bản sao ngày càng lớn), mà tính lại chúng khi lan truyền ngược. Đổi lại, huấn luyện chậm hơn khoảng 15–20%. Khởi tạo trong code là khởi tạo He chuẩn theo `fan_out` (xem [bài khởi tạo tham số](dlnote/dl_init_vi)). Cấu hình mặc định là DenseNet-BC cho CIFAR với $L=100$, $k=12$ (3 khối × 16 tầng × 2 tích chập + 4 tầng), khoảng 0,8 triệu tham số như bảng 2 của bài báo.

> **Biên tập:** Bản gốc gọi đây là "code mã nguồn mở của tác giả"; đó là bản cài đặt của Geoff Pleiss (đồng tác giả bài báo về cài đặt tiết kiệm bộ nhớ), dựa trên code torchvision, không phải code gốc (Lua Torch) của Gao Huang. Với PyTorch mới, nên gọi `cp.checkpoint(..., use_reentrant=False)`.

## 5 Câu hỏi

1. Nhiều kết nối dày đặc như vậy có cần thiết hết không, bỏ bớt một số có ảnh hưởng tới hiệu năng không?

**Trả lời của tác giả DenseNet**: bài báo có một bản đồ nhiệt (`heatmap`) thể hiện trực quan độ mạnh của từng kết nối (trọng số trung bình theo trị tuyệt đối từ mỗi tầng nguồn tới mỗi tầng đích). Có thể thấy các tầng khá sâu cũng dùng tới đặc trưng của các tầng rất nông.

Lưu ý: phiên bản cải tiến về sau là `VoVNet` thiết kế mô-đun `OSA`, bỏ kết nối dày đặc ở các tầng giữa, chỉ tầng cuối gộp đặc trưng của mọi tầng trước, và làm cùng thí nghiệm đó. Bản đồ nhiệt cho thấy bỏ kết nối dày đặc ở các tầng giữa thì kết nối của tầng cuối mạnh hơn. Đồng thời, so sánh với `DenseNet` trên `CIFAR-10`, độ chính xác của `OSA` gần bằng `DenseBlock` nhưng `MAC` giảm nhiều, cho thấy kết nối dày đặc của `DenseBlock` sinh nhiều đặc trưng dư thừa ở các tầng giữa. Xem [bài VoVNet](dlnote/bb_vovnet_vi).

> **Biên tập:** Bản gốc viết tên mô-đun là `OSP`; đúng là `OSA` (One-Shot Aggregation).

## Tài liệu tham khảo

- [Tác giả bài báo xuất sắc nhất CVPR 2017 giải thích DenseNet: "what", "why" và "how"](https://www.leiphone.com/category/ai/0MNOwwfvWiAu43WO.html)
- [efficient_densenet_pytorch](https://github.com/gpleiss/efficient_densenet_pytorch)
- Huang et al. [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993). CVPR 2017.
- Pleiss et al. [Memory-Efficient Implementation of DenseNets](https://arxiv.org/abs/1707.06990). 2017.
