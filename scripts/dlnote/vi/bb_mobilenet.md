# CNN hiệu quả: MobileNet v1

Trang [Code](dlnote/backbone_code_vi) đi kèm kiểm tra công thức tỉ lệ $\frac{1}{N} + \frac{1}{D_K^2}$ của tích chập tách được theo chiều sâu, chạy đoạn code `groups` ở mục 1, đo tỉ lệ MACs và tham số của tích chập 1×1 trong MobileNet (bảng 2 của bài báo), và chạy lại code MobileNet v1 ở mục 5 (bản gốc có lỗi, đã sửa).

<!-- theory: qa/ch17_vi#17.8.2 ; qa/ch17_vi#17.9.2 -->

> Bài này cũng được tác giả đăng trên [Zhihu](https://zhuanlan.zhihu.com/p/359524431).

> Đóng góp chính của bài báo `MobileNet` là đề xuất **kiến trúc tích chập tách được theo chiều sâu (tích chập DW + PW)**: trước hết chứng minh về lý thuyết rằng kiến trúc này có chi phí tính toán (`Mult-Adds`) nhỏ hơn tích chập thường, rồi chứng minh hiệu quả của mô hình qua nhiều thí nghiệm phân loại, phát hiện đối tượng…

## 1 Các công trình liên quan

### Tích chập thường

Một bộ lọc kích thước $h_1\times w_1$ (kernel 2 chiều) trượt từ góc trên trái tới góc dưới phải của `feature map`. Mỗi lần dịch chuyển, nó nhân các tham số của bộ lọc với các điểm ảnh trong vùng tương ứng rồi cộng lại thành một giá trị. Vì có $c_1$ `feature map` đầu vào (số kênh), ta cần một bộ lọc hình dạng $(c_1, h_1, w_1)$ (kernel 3 chiều), cộng các tổng tính trên từng feature map đầu vào ở cùng vị trí để được giá trị của điểm ảnh tương ứng trên feature map đầu ra. Vì có $c_2$ `feature map` đầu ra nên cần $c_2$ bộ lọc. Quá trình tích chập thường được minh hoạ như sau.

![Quá trình tích chập thường](../images/mobilenetv1/standard_convolution_process.png)

Hình 1.1: Tích chập thường: $c_2$ bộ lọc, mỗi bộ lọc phủ đủ $c_1$ kênh đầu vào.

Hình động dưới đây cho thấy trực quan cách kernel trượt và cộng lại để ra $c_2$ `feature map` (nguồn [ở đây](https://blog.csdn.net/v_july_v/article/details/51812459)).

![Quá trình tích chập](../images/mobilenetv1/conv_gif.gif)

Hình 1.2: Input Shape: (3, 7, 7), Output Shape: (2, 3, 3), K: (3, 3), P: (1, 1), S: (2, 2), D: (2, 2), G: 1.

Hình động trên nhấn mạnh quá trình trượt của bộ lọc, còn hình dưới giải thích quan hệ giữa năm khái niệm của tầng tích chập (nguồn: [AI-EDU 17.1](https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC8%E6%AD%A5%20-%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/17.1-%E5%8D%B7%E7%A7%AF%E7%9A%84%E5%89%8D%E5%90%91%E8%AE%A1%E7%AE%97%E5%8E%9F%E7%90%86.html)):

- kênh đầu vào (Input Channel);
- nhóm kernel (WeightsBias);
- bộ lọc (Filter);
- kernel;
- feature map đầu ra (Feature Map).

![Tích chập ba kênh với hai bộ lọc](../images/mobilenetv1/conv3d.png)

Hình 1.3: Đầu vào 3 kênh đi qua 2 bộ lọc, mỗi bộ lọc gồm 3 kernel 5×5 cộng một bias.

Trong ví dụ này, đầu vào là dữ liệu ba chiều $3\times 32 \times32$, đi qua tầng tích chập có trọng số $2\times 3\times 5\times 5$, đầu ra là ba chiều $2\times 28\times 28$. Số chiều không đổi, chỉ kích thước trong mỗi chiều thay đổi, thường là nhỏ đi để đơn giản hoá tính toán.

### Tích chập nhóm

Tích chập nhóm (`Group Convolution`) xuất hiện sớm nhất trong `AlexNet`. Cách nối feature map vào và ra của tích chập thường và tích chập nhóm như hình dưới (nguồn: bài báo [CondenseNet](https://www.researchgate.net/figure/The-transformations-within-a-layer-in-DenseNets-left-and-CondenseNets-at-training-time_fig2_321325862)).

![Tích chập nhóm](../images/mobilenetv1/grouped_convolution.png)

Hình 1.4: Trái: tích chập thường, mỗi kênh ra nối với mọi kênh vào. Phải: tích chập nhóm, mỗi kênh ra chỉ nối với các kênh vào cùng nhóm.

**Định nghĩa tích chập nhóm**: **chia các `feature map` đầu vào thành nhóm, rồi tích chập riêng từng nhóm**. Giả sử feature map vào có kích thước $H \times W \times c_{1}$, có $c_2$ feature map ra; chia đầu vào theo kênh thành $g$ nhóm thì mỗi nhóm có kích thước $H \times W \times \frac{c_1}{g}$, bộ lọc của **mỗi nhóm** có kích thước $h_{1} \times w_{1} \times \frac{c_{1}}{g}$, mỗi nhóm có $\frac{c_{2}}{g}$ bộ lọc, tổng số bộ lọc vẫn là $c_2$; tức kernel của tích chập nhóm có hình dạng $(c_2,\frac{c_1}{g}, h_1,w_1)$. Bộ lọc của mỗi nhóm chỉ tích chập với feature map đầu vào cùng nhóm, mỗi nhóm cho đầu ra $H \times W \times \frac{c_{2}}{g}$; nối (`concatenate`) kết quả của $g$ nhóm được đầu ra cuối cùng $H \times W \times c_2$. Quá trình như hình dưới:

![Quá trình tích chập nhóm](../images/mobilenetv1/grouped_convolution_process_2.png)

Hình 1.5: Tích chập nhóm với $g$ nhóm.

**Ý nghĩa của tích chập nhóm**: tích chập nhóm là thành phần cốt lõi của thiết kế mạng hiện nay. Nó giảm độ phức tạp tính toán nhờ **kết nối thưa** giữa các kênh (chỉ nối với đặc trưng trong cùng nhóm). Một mặt, nó cho phép dùng nhiều kênh hơn để tăng năng lực mạng và độ chính xác; mặt khác, nhiều kênh hơn cũng kéo theo $MAC$ (lượng truy cập bộ nhớ) lớn hơn. Với tích chập nhóm $1 \times 1$, $MAC$ và $FLOPs$ là:

$$
\begin{aligned}
& MACC = H \times W \times 1 \times 1 \times \frac{c_{1}}{g}\frac{c_{2}}{g} \times g = \frac{HWc_{1}c_{2}}{g} \\
& FLOPs = 2 \times MACC \\
& Params = g \times \frac{c_2}{g}\times\frac{c_1}{g} \times 1\times 1 = \frac{c_{1}c_{2}}{g} \\
& MAC = HW(c_1 + c_2) + \frac{c_{1}c_{2}}{g}
\end{aligned}
$$

trong đó MACC là số phép nhân-cộng, MAC là lượng truy cập bộ nhớ (đọc vào, ghi ra, đọc trọng số). Từ đó suy ra số tham số và lượng tính toán của tích chập nhóm $1\times 1$ bằng $\frac{1}{g}$ của tích chập thường; thực ra hiểu kỹ quá trình tích chập nhóm cũng suy ra được ngay điều này.

**Hiểu sâu hơn về tích chập nhóm**: với tích chập $1\times 1$, mỗi điểm ảnh trên feature map đầu ra của tích chập thường được tính từ $c_1$ điểm của đầu vào, còn của tích chập nhóm chỉ từ $\frac{c_1}{g}$ điểm. **Phép tích chập là tuyến tính, nên tự nhiên số tham số và lượng tính toán của tích chập nhóm bằng $\frac{1}{g}$ của tích chập thường.**

Khi số nhóm = số feature map vào = số feature map ra, tức $g=c_1=c_2$, có $c_1$ bộ lọc, mỗi bộ lọc kích thước $1 \times K \times K$, tích chập nhóm trở thành tích chập theo chiều sâu (Depthwise Convolution, tích chập DW), **kernel của tích chập DW có hình dạng** $(c_{1}, 1, K, K)$.

> Kernel của tích chập thường có hình dạng (`C_out, C_in, kernel_height, kernel_width`), của tích chập nhóm là (`C_out, C_in/g, kernel_height, kernel_width`), của tích chập DW là (`C_in, 1, kernel_height, kernel_width`).

Tích chập nhóm có **một ràng buộc**: **số kênh ra của tầng phải là bội số của số nhóm**, tức $c_2$ phải chia hết cho $g$, vì đầu ra là phép nối kết quả của $g$ nhóm. (Tương tự, $c_1$ cũng phải chia hết cho $g$.)

> Ở đây $c_2$ là số kênh ra, $c_1$ là số kênh vào, $g$ là số nhóm.

### Tích chập DW

Khác với tích chập thường, nơi mỗi bộ lọc chứa một kernel cho mỗi kênh đầu vào, tích chập nhóm và tích chập DW có đặc điểm:

- **Tích chập nhóm**: chia kênh vào thành nhiều nhóm, **bộ lọc của mỗi nhóm chỉ tích chập với feature map vào cùng nhóm**, cuối cùng nối kênh ra của các nhóm thành đầu ra.
- **Tích chập DW**: mỗi bộ lọc chỉ có một kernel, ứng với một kênh đầu vào; tức mỗi kênh vào được tích chập riêng bằng một kernel kích thước cố định.

Tích chập nhóm cực đoan nhất là số nhóm bằng số kênh vào, đó chính là tích chập `DW`:

![Tích chập DW](../images/mobilenetv1/dw_conv.png)

Hình 1.6: Tích chập DW: mỗi kênh vào được tích chập với một kernel riêng, số kênh không đổi.

API tầng tích chập của `PyTorch` hỗ trợ cả tích chập thường, tích chập nhóm và tích chập DW. Lưu ý với tích chập nhóm, số kênh vào và ra phải là bội số của số nhóm, nếu không sẽ báo lỗi!

```python
import torch
input = torch.randn([20, 10, 224, 224])  # 10 kênh vào
conv3x3 = torch.nn.Conv2d(in_channels=10, out_channels=5, kernel_size=3, groups=5)
output = conv3x3(input)
print(conv3x3.weight.shape)   # torch.Size([5, 2, 3, 3])
print(output.shape)           # torch.Size([20, 5, 222, 222])
```

Nếu đổi `groups=5` thành `groups=6` hoặc đổi `out_channels=5` thành `6`, PyTorch báo lỗi:

```bash
ValueError: in_channels must be divisible by groups
ValueError: out_channels must be divisible by groups
```

> **Biên tập:** Code của bản gốc dùng tham số `output_channels=5`, không tồn tại trong `nn.Conv2d` (tên đúng là `out_channels`), nên chạy sẽ báo `TypeError` trước cả khi tới lỗi về `groups`; đã sửa và thêm kết quả in ra (trang Code chạy lại cả ba trường hợp).

### Từ mô-đun Inception tới tích chập tách được theo chiều sâu

Tích chập tách được theo chiều sâu (depthwise separable convolution) bắt nguồn từ bài báo `Xception`. Xception viết rằng với tích chập, kernel có thể xem là một bộ lọc ba chiều: chiều kênh cộng hai chiều không gian (rộng, cao của feature map); tích chập thường thực hiện **ánh xạ đồng thời** tương quan giữa các kênh và tương quan không gian. **Giả định đằng sau mô-đun Inception là tương quan giữa các kênh và tương quan không gian có thể tách rời đủ mức, nên ánh xạ riêng chúng sẽ tốt hơn** (*the fundamental hypothesis behind Inception is that cross-channel correlations and spatial correlations are sufficiently decoupled that it is preferable not to map them jointly*).

**Inception dùng tích chập tách được theo chiều sâu được gọi là Xception**, là bản cải tiến của Inception v3, tăng chút ít độ chính xác trên ImageNet và JFT, nhưng số tham số và tốc độ gần như không đổi, vì mục tiêu của Xception không phải nén mô hình. Mô-đun Inception "cực đoan" như hình 4 của bài báo Xception:

![Xception](../images/mobilenetv1/Xception.png)

Hình 1.7: Phiên bản "cực đoan" của mô-đun Inception: một tích chập 1×1, rồi mỗi kênh ra được tích chập 3×3 riêng (hình 4 của bài báo Xception).

Mô-đun Inception "cực đoan" ở hình 1.7 giống mô-đun tích chập tách được theo chiều sâu, nhân vật chính của bài, chỉ khác ở thứ tự: tích chập tách được làm tích chập không gian theo từng kênh (`channel-wise`) trước, rồi mới tích chập kênh $1 \times 1$; còn Inception ở hình trên làm ngược lại.

## 2 Cấu trúc MobileNet

### 2.1 Tích chập tách được theo chiều sâu

`MobileNet` là mạng nhẹ hiệu quả cho thiết bị di động do Google đề xuất năm 2017, cốt lõi là tích chập tách được theo chiều sâu (`depthwise separable convolution`): tách một phép tích chập đầy đủ thành hai bước, tích chập theo chiều sâu (Depthwise Convolution, tích chập `DW`) và tích chập theo điểm (Pointwise Convolution, tích chập `PW`). Các bước tính và kích thước bộ lọc như sau.

![Các bước của tích chập tách được theo chiều sâu](../images/mobilenetv1/computational_steps_of_depthwise_separable_convolution.png)

Hình 2.1: Tích chập thường (a) được tách thành tích chập DW (b) và tích chập PW 1×1 (c) (hình 2 của bài báo MobileNet).

![Kích thước bộ lọc](../images/mobilenetv1/filter_size.png)

Hình 2.2: Kích thước bộ lọc của tích chập DW và PW (nguồn: bài báo [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)).

#### Tích chập theo chiều sâu (Depthwise)

> Lưu ý: cách tính lượng tính toán của DW và PW trong bài này khác bài báo một chút: ở đây feature map đầu ra có kích thước $D_G \times D_G$, còn công thức của bài báo dùng $D_F \times D_F$ (kích thước đầu vào, coi như stride 1 và có padding).

Khác với tích chập thường, trong tích chập DW mỗi kernel chỉ phụ trách một kênh, **mỗi kênh chỉ được tích chập bởi một kernel** (các kênh khác nhau dùng kernel khác nhau); tức là **tích chập nhóm đặc biệt có số kênh vào, số kênh ra và số nhóm bằng nhau**, nên tích chập DW không làm đổi số kênh của feature map. Các bước của tích chập `DW`:

![Các bước tính của tích chập DW](../images/mobilenetv1/dw_convolution_calculation_steps.jpg)

Hình 2.3: Tích chập DW: feature map vào kích thước $(D_F, D_F, M)$, $M$ bộ lọc kích thước $(D_K, D_K, 1)$, cho feature map ra $M$ kênh.

Lượng tính toán của tích chập `DW` là $MACC = M \times D_{G}^{2} \times D_{K}^{2}$.

#### Tích chập theo điểm (Pointwise)

Vấn đề của tích chập Depthwise là mỗi kernel chỉ tính trên một kênh, thông tin giữa các kênh không được trao đổi, nên luồng thông tin phía sau sẽ mất thông tin liên kênh. Vì vậy bài báo thêm tích chập Pointwise để trộn thông tin giữa các kênh. Tích chập PW là một tích chập thường đặc biệt với kernel $1 \times 1$. Quá trình tích chập `PW`:

![Quá trình tích chập PW](../images/mobilenetv1/pw_convolution_process.jpg)

Hình 2.4: Tích chập PW: $N$ bộ lọc $1\times 1\times M$ trộn các kênh tại từng điểm ảnh.

Giả sử feature map vào có kích thước $D_{G} \times D_{G} \times M$, ra $D_{G} \times D_{G} \times N$, thì bộ lọc có kích thước $1 \times 1 \times M$ và có $N$ bộ lọc. Lượng tính toán của tích chập `PW` là $MACC = N \times M \times D_{G}^{2}$.

Tổng hợp: lượng tính toán của hai phần `Depthwise` và `Pointwise` là $MACC_1 = M \times D_{G}^{2} \times D_{K}^{2} + N \times M \times D_{G}^{2}$, còn của tích chập thường là $MACC_2 = N \times D_{G}^{2} \times D_{K}^{2} \times M$. Tỉ lệ giữa hai bên là:

$$
\begin{aligned}
\frac{\text{Depthwise Separable Conv}}{\text{Standard Conv}} &= \frac{M \times D_{G}^{2}(D_{K}^{2} + N)}{N \times D_{G}^{2} \times D_{K}^{2} \times M} \\
&= \frac{D_{K}^{2} + N}{D_{K}^{2} \times N} \\
&= \frac{1}{N} + \frac{1}{D_{K}^{2}}
\end{aligned}
$$

Có thể thấy **lượng tính toán của tích chập `Depthwise + Pointwise` nhỏ hơn tích chập thường khoảng $D_K^2$ lần** (với kernel 3×3 là 8–9 lần), và **số tham số** cũng giảm theo cùng tỉ lệ. Cùng mục đích (tính trên các phần tử lân cận và giữa các kênh), tích chập tách được theo chiều sâu giảm rất nhiều lượng tính toán, nên rất nhiều `backbone` cho di động dùng cấu trúc này, kết hợp với chưng cất, cắt tỉa để suy luận hiệu quả hơn trên thiết bị.

> **Biên tập:** Bản gốc kết luận lượng tính toán "giảm gần $N$ lần" ($N$ là số kênh ra). Từ chính công thức trên, tỉ lệ là $\frac{1}{N} + \frac{1}{D_K^2}$; khi $N$ lớn (như 64–1024 trong MobileNet), số hạng $\frac{1}{N}$ không đáng kể và tỉ lệ xấp xỉ $\frac{1}{D_K^2} = \frac{1}{9}$, tức giảm khoảng **9 lần**, đúng như bài báo viết ("8 to 9 times less computation"). Trang Code tính cả công thức lẫn số MACs đếm bằng công cụ cho $N = 64, 256, 1024$.

### 2.2 Cấu trúc mạng

Khối (`Block`) tích chập tách được theo chiều sâu $3 \times 3$ như hình dưới:

![Khối tích chập tách được 3x3](../images/mobilenetv1/3x3_depth_separable_convolution_block_structure.png)

Hình 2.5: Trái: tầng tích chập thường kèm BN và ReLU. Phải: khối tích chập tách được theo chiều sâu 3×3, mỗi tầng DW và PW đều kèm BN và ReLU (hình 3 của bài báo MobileNet).

Code PyTorch của khối tích chập tách được $3 \times 3$:

```python
class MobilnetV1Block(nn.Module):
    """Depthwise conv + Pointwise conv"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobilnetV1Block, self).__init__()
        # kernel của dw conv có hình dạng (in_channels, 1, ksize, ksize)
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.dw(x)))
        out2 = F.relu(self.bn2(self.pw(out1)))
        return out2
```

Sau khi xuất mô hình `PyTorch` của `MobileNet v1` sang `ONNX`, đồ thị của một `khối tích chập tách được` như sau.

![Đồ thị ONNX của khối tích chập tách được](../images/mobilenetv1/onnx_model_structure_diagram_of_depth_separable_convolution_block.png)

Hình 2.6: Đồ thị ONNX (xem bằng Netron): tích chập thường đầu mạng, rồi tích chập DW (số nhóm = số kênh vào = số kênh ra = 32) và tích chập PW (32 kênh vào, 64 kênh ra).

Chỉ định nghĩa mạng hiệu năng cao bằng việc MobileNet có ít `Mult-Adds` (phép nhân-cộng) hơn là chưa đủ; còn phải bảo đảm các phép tính đó được cài đặt hiệu quả. Ví dụ, phép tính trên ma trận thưa không có cấu trúc (unstructured sparse matrix operations) thường không nhanh hơn phép tính trên ma trận đặc (dense matrix operations), trừ khi độ thưa rất cao.

> Câu này phải chăng cùng quan điểm với `ShuffleNet v2`: không thể chỉ dùng lượng tính toán FLOPs để đánh giá tốc độ chạy của mạng, trừ khi so các mạng cùng kiến trúc?

Kiến trúc MobileNet dồn gần như toàn bộ tính toán vào các tích chập 1×1 đặc (dense 1 × 1 convolutions), loại tích chập cài được bằng các hàm nhân ma trận tổng quát (`GEMM`) đã tối ưu rất kỹ. Tích chập thường cũng được cài bằng `GEMM`, nhưng trước đó phải sắp xếp lại dữ liệu trong bộ nhớ bằng `im2col`; framework Caffe làm theo cách này. Tích chập `1×1` không cần sắp xếp lại, có thể gọi thẳng `GEMM` (một trong những thuật toán đại số tuyến tính số được tối ưu nhất).

Như bảng 2, tích chập `1x1` của MobileNet chiếm `95%` lượng tính toán và `75%` số tham số, phần tham số còn lại gần như đều nằm ở tầng kết nối đầy đủ; tích chập DW 3×3 và tích chập thường chiếm rất ít tính toán (Mult-Adds) và tham số.

![Bảng 2](../images/mobilenetv1/table_2.png)

Hình 2.7: Phân bổ Mult-Adds và tham số theo loại tầng của MobileNet (bảng 2 của bài báo).

> **Biên tập:** Trang Code đếm lại trên mô hình đã sửa ở mục 5 và ra đúng các con số của bảng: tích chập 1×1 chiếm 94,86% Mult-Adds và 74,59% tham số, tích chập DW 3×3 chiếm 3,06% và 1,06%, tầng FC 0,18% và 24,33%. Riêng tầng tích chập 3×3 đầu mạng, bảng ghi 1,19% Mult-Adds, còn tính lại là 1,91% ($112^2 \times 32 \times 27 \approx 10{,}8$ triệu trên 569 triệu); có lẽ hai chữ số bị đảo, vì chỉ với 1,91% tổng bốn dòng mới bằng 100%.

### 2.3 Hệ số độ rộng: mô hình nhỏ hơn

Dù MobileNet cơ bản đã nhỏ và có độ trễ (`latency`) thấp, nhiều trường hợp hay ứng dụng cụ thể có thể cần mô hình còn nhỏ hơn, nhanh hơn. Để tạo các mô hình nhỏ và rẻ hơn, bài báo đưa vào một tham số rất đơn giản $\alpha$, gọi là **hệ số độ rộng** (width multiplier). Hệ số này làm mỏng đều mọi tầng: với một tầng và hệ số $\alpha$, số kênh vào $M$ thành $\alpha M$, số kênh ra $N$ thành $\alpha N$. Chi phí tính toán của tích chập tách được với hệ số độ rộng $\alpha$ (các ký hiệu khác như trên) là:

$$\alpha M \times D_{G}^{2} \times D_{K}^{2} + \alpha N \times \alpha M \times D_{G}^{2}$$

trong đó $\alpha \in (0,1]$, giá trị thường dùng là `1; 0,75; 0,5` và `0,25`. $\alpha = 1$ là MobileNet cơ bản, $\alpha < 1$ là các MobileNet thu nhỏ. **Hệ số độ rộng giảm lượng tính toán và số tham số khoảng $\alpha^2$ lần, nhờ đó giảm chi phí tính toán của mạng.** Có thể áp hệ số này cho bất kỳ kiến trúc nào để có mô hình nhỏ hơn, **với sự đánh đổi hợp lý giữa độ chính xác, độ trễ và kích thước mô hình**. Mô hình thu nhỏ phải huấn luyện lại từ đầu. Cấu trúc tổng thể của MobileNet cơ bản như bảng 1.

![Bảng 1](../images/mobilenetv1/table_1.png)

Hình 2.8: Cấu trúc MobileNet (bảng 1 của bài báo). Chú thích đỏ của tác giả bản gốc: tầng "Conv dw / s2" gần cuối thực ra phải là s1.

> **Biên tập:** Bảng 1 của bài báo ghi tầng DW thứ hai từ cuối là `Conv dw / s2` với đầu vào $7\times7\times1024$; nếu stride 2 thì đầu ra là $4\times 4$, mâu thuẫn với tầng `Avg Pool 7×7` ngay sau. Đó là lỗi in của bài báo (stride đúng là 1), mọi cài đặt đều dùng stride 1; tác giả bản gốc đã đánh dấu trên ảnh.

### 2.4 Hệ số độ phân giải: giảm biểu diễn

Siêu tham số thứ hai để giảm chi phí tính toán là **hệ số độ phân giải** $\rho$: áp lên ảnh đầu vào, khiến feature map ở mọi tầng cũng nhỏ đi theo hệ số $\rho$. Trên thực tế bài báo đặt $\rho$ ngầm qua độ phân giải đầu vào.

Chi phí tính toán của các tầng lõi, với hệ số độ rộng $\alpha$ và hệ số độ phân giải $\rho$:

$$\alpha M \times \rho^2 D_{G}^{2} \times D_{K}^{2} + \alpha N \times \alpha M \times \rho^2 D_{G}^{2}$$

trong đó $\rho \in (0,1]$, thường đặt ngầm để độ phân giải đầu vào là `224, 192, 160` hoặc `128`. $\rho = 1$ là MobileNet cơ bản (`baseline`), $\rho < 1$ là các MobileNet thu nhỏ. **Hệ số độ phân giải giảm lượng tính toán $\rho^2$ lần** (số tham số không đổi).

> **Biên tập:** Bản gốc viết $\rho D_G^2$; kích thước feature map là $\rho D_G \times \rho D_G$ nên số hạng đúng là $\rho^2 D_G^2$, khớp với kết luận "giảm $\rho^2$ lần" ngay sau đó.

### 2.5 Tóm tắt cấu trúc mô hình

- Không tính tầng average pooling và tầng `softmax`, và tính tích chập `DW` và `PW` là hai tầng riêng, thì `MobileNet` có `28` tầng.
- Các tích chập stride 2 trong mạng đáng chú ý: chúng vừa tích chập vừa giảm kích thước (thay cho pooling).
- `26` tầng sau tầng đầu đều là các khối tích chập tách được lặp lại, chia thành `4` giai đoạn (`stage`) tích chập.
- Mọi tầng tích chập (thường, DW, PW) đều theo sau bởi batch normalization và hàm kích hoạt `ReLU`.
- Tầng kết nối đầy đủ cuối cùng không dùng hàm kích hoạt.

## 3 Thí nghiệm

Tác giả làm các thí nghiệm nhận dạng chi tiết (fine-grained) trên tập Stanford Dogs, định vị địa lý quy mô lớn, phân loại thuộc tính khuôn mặt, và phát hiện đối tượng trên COCO, để chứng minh rằng so với các `backbone` như `Inception V3`, `GoogleNet`, `VGG16`, mô hình `MobileNet` giảm hàng chục lần lượng tính toán (`Mult-Adds`) mà độ chính xác gần như không đổi.

> **Biên tập:** Cụ thể trên ImageNet (bảng 8 của bài báo): MobileNet 1.0-224 đạt 70,6% top-1 với 569 triệu Mult-Adds và 4,2 triệu tham số, so với VGG-16 71,5% với 15 300 triệu Mult-Adds và 138 triệu tham số (ít hơn 27 lần về tính toán, 32 lần về tham số), và GoogleNet 69,8% với 1550 triệu Mult-Adds.

## 4 Kết luận

Bài báo đề xuất một kiến trúc mô hình mới **dựa trên tích chập tách được theo chiều sâu**, gọi là `MobileNets`. Ở phần công trình liên quan, tác giả xem xét một số nguyên tắc thiết kế quan trọng giúp mô hình hiệu quả hơn; rồi trình bày cách dùng hệ số độ rộng và hệ số độ phân giải để tạo MobileNet nhỏ hơn, nhanh hơn, đánh đổi hợp lý độ chính xác lấy kích thước và độ trễ. Sau đó so sánh các MobileNet với các mô hình phổ biến, cho thấy ưu thế về kích thước, tốc độ và độ chính xác. Cuối cùng, bài báo cho thấy MobileNet hiệu quả khi áp dụng cho nhiều bài toán khác nhau.

## 5 Code của mô hình cơ bản

Code PyTorch MobileNet v1 cơ bản do tác giả bản gốc tự viết lại (đã sửa lỗi, xem ô Biên tập bên dưới):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten

class MobilnetV1Block(nn.Module):
    """Depthwise conv + Pointwise conv"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(MobilnetV1Block, self).__init__()
        # kernel của dw conv có hình dạng (in_channels, 1, ksize, ksize)
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


def convbn_relu(in_channels, out_channels, stride=2):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                                   padding=1, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


class MobileNetV1(nn.Module):
    # (32, 64, 1): MobilnetV1Block có 32 kênh vào, 64 kênh ra, kích thước không đổi
    stage_cfg = [(32, 64, 1),
           (64, 128, 2), (128, 128, 1),     # stage1
           (128, 256, 2), (256, 256, 1),    # stage2
           (256, 512, 2), (512, 512, 1), (512, 512, 1), (512, 512, 1), (512, 512, 1), (512, 512, 1),  # stage3
           (512, 1024, 2), (1024, 1024, 1)  # stage4
    ]
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.first_conv = convbn_relu(3, 32, 2)   # ảnh vào giảm một nửa
        self.stage_layers = self._make_layers()
        self.linear = nn.Linear(1024, num_classes)  # tầng kết nối đầy đủ

    def _make_layers(self):
        layers = [MobilnetV1Block(cin, cout, s) for cin, cout, s in self.stage_cfg]
        return nn.Sequential(*layers)

    def forward(self, x):
        """Kích thước feature map: 224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 1"""
        x = self.first_conv(x)
        x = self.stage_layers(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = flatten(x, 1)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = MobileNetV1().eval()
    outputs = model(torch.rand(1, 3, 224, 224))
    print("Model output size is", outputs.size())   # torch.Size([1, 1000])
```

> **Biên tập:** Tầng DW trong code gốc viết `nn.Conv2d(in_channels, 64, kernel_size=3, ..., groups=4)`: số kênh ra cố định là 64 và chỉ 4 nhóm, tức không phải tích chập DW; ngay khối đầu (32 kênh vào) đầu ra 64 kênh không khớp với `BatchNorm2d(32)` phía sau, nên mô hình báo lỗi khi chạy, khác với kết quả "Model output size is torch.Size([1, 1000])" mà bản gốc in. Đã sửa thành `nn.Conv2d(in_channels, in_channels, ..., groups=in_channels)`, đúng như khối ở mục 2.2. Trang Code chạy cả hai bản: bản gốc báo lỗi kích thước BN, bản sửa có 4,23 triệu tham số và khoảng 0,57 G MACs, khớp với 4,2 triệu và 569 triệu của bài báo. Cũng đã thay `F.avg_pool2d(x, 7)` bằng `adaptive_avg_pool2d` để chạy được với các độ phân giải khác 224 (hệ số $\rho$).

## Suy nghĩ cá nhân của tác giả

Về giảm lượng tính toán `FLOPs`, kiến trúc `MobileNet` thực sự rất tốt; nhưng trên phần cứng `GPU`, `DSP` và `TPU`, mô hình `MobileNet` chưa chắc đã chạy tốt, vì mỗi phần cứng tính toán theo cách khác nhau, dẫn tới **`FLOPs` ít không có nghĩa là `latency` thấp**.

Giải thích nguyên lý hoạt động của `TPU` và `DSP` thì hơi phức tạp; có thể xem hình dưới để hiểu trực quan khác biệt trong cách chúng chạy. Xét một tích chập (`convolution`) đơn giản: trên `CPU`, `latency` tăng đều theo số kênh vào và ra; còn trên `DSP` thì tăng theo bậc thang, thậm chí ở số kênh cao hơn còn có những "điểm ngọt" có `latency` đặc biệt thấp.

![Hành vi của tích chập trên CPU và DSP](../images/mobilenetv1/convolution_behavior_differences_between_cpu_and_dsp.png)

Hình 5.1: Độ trễ của một tầng tích chập theo số kênh vào và ra trên CPU (trái) và DSP (phải) (nguồn: bài báo MobileDets).

Ở một mức độ nào đó, mạng càng sâu, càng rộng thì càng tốt. Độ rộng là số kênh (`channel`), độ sâu là số tầng (`layer`), như `resnet18` có `18` tầng có trọng số. Lưu ý "độ rộng" ở đây không liên quan tới các mô hình kiểu học rộng (broad learning), mà chỉ độ rộng (số **kênh**) của mạng neuron tích chập sâu.

- **Ý nghĩa của độ sâu**: các tầng của `CNN` trừu tượng hoá dần dữ liệu ảnh, ví dụ tầng đầu học đặc trưng cạnh, tầng hai học hình dạng đơn giản, tầng ba học hình dạng của đối tượng; độ sâu tăng thì năng lực trừu tượng của mô hình tăng.
- **Ý nghĩa của độ rộng**: độ rộng (số kênh) là số bộ lọc (`3` chiều); càng nhiều bộ lọc thì khả năng trích đặc trưng càng mạnh, mỗi tầng học được đặc trưng phong phú hơn, như vân theo các hướng, các tần số khác nhau.

## Cải tiến về sau: MobileDets

Những nhận xét từ bài báo MobileDets (2021):

1. `FLOPs` ít không có nghĩa là `latency` thấp, nhất là trên phần cứng có tăng tốc (`GPU`, `DSP` và `TPU`).
2. Khối tích chập của `MobileNet` (`depthwise separable convolution`) kém hiệu quả trên phần cứng có tăng tốc (chip chuyên dụng `NPU`).
3. Khi số kênh ít (như các tầng đầu mạng), trên phần cứng có tăng tốc, tích chập **thường** thường hiệu quả hơn tích chập tách được.
4. Trên phần lớn phần cứng, số kênh là bội số của `8` có lợi cho tính toán.
5. Trên `DSP` và `TPU`, thường phải tính ở dạng `uint8`, nên **lượng tử hoá** (`quantization`) là kỹ thuật phổ biến.
6. Trên `DSP` và `TPU`, `h-Swish` và `squeeze-and-excitation` không hiệu quả rõ rệt (do thiết kế phần cứng và lượng tử hoá `uint8`).
7. Trên `DSP` và `TPU`, tích chập `5x5` kém hiệu quả.

## Bổ sung: các thế hệ MobileNet sau

Mục này do người biên soạn viết thêm.

- **MobileNet v2** (2018): khối **inverted residual** với **linear bottleneck**: tích chập 1×1 *mở rộng* số kênh (thường 6 lần), tích chập DW 3×3 trên không gian rộng đó, rồi tích chập 1×1 *thu hẹp* lại, không có ReLU sau tầng thu hẹp (vì ReLU trên không gian ít chiều làm mất thông tin), và kết nối tắt giữa hai đầu hẹp. Ngược với bottleneck của ResNet (hẹp ở giữa). 3,4 triệu tham số, 300 triệu MACs, 72% top-1.
- **MobileNet v3** (2019): kiến trúc tìm bằng NAS có ràng buộc độ trễ trên điện thoại, thêm mô-đun SE và hàm kích hoạt h-swish ($x\cdot\mathrm{ReLU6}(x+3)/6$, xấp xỉ rẻ của Swish); bản Large đạt 75,2% top-1.
- **MobileNet v4** (2024): khối Universal Inverted Bottleneck (UIB) gộp nhiều biến thể (có/không DW trước và sau), cùng attention đa truy vấn cho phần cứng di động; nhắm hiệu năng tốt trên nhiều loại phần cứng (CPU, DSP, GPU, Apple Neural Engine) cùng lúc, đúng các bài học của MobileDets ở trên.

Khối inverted residual của MobileNet v2 (MBConv) là nền của EfficientNet và nhiều mạng nhẹ khác; xem [bài tổng kết mạng nhẹ](dlnote/compress_light_vi) ở phần nén mô hình.

## Tài liệu tham khảo

1. [Group Convolution, Depthwise Convolution và Global Depthwise Convolution](https://www.cnblogs.com/shine-lee/p/10243114.html)
2. [Hiểu vì sao tích chập nhóm và tích chập tách được giảm số tham số](https://zhuanlan.zhihu.com/p/65377955)
3. [Tích chập tách được theo chiều sâu (Xception và MobileNet)](https://www.jianshu.com/p/38dc74d12fcf)
4. [Cài đặt MobileNetV1](https://www.cnblogs.com/linzzz98/articles/13453810.html)
5. [Tích chập Depthwise và Pointwise](https://zhuanlan.zhihu.com/p/80041030)
6. [[Thiết kế cấu trúc CNN] Hiểu sâu tích chập tách được theo chiều sâu](https://mp.weixin.qq.com/s/IZ-nbrCL8-9w32RSYeP_bg)
7. [FLOPs và tốc độ suy luận của mô hình](https://zhuanlan.zhihu.com/p/122943688)
8. [MobileDets: FLOPs không bằng latency, kiến trúc hiệu quả xét đến phần cứng khác nhau](https://medium.com/ai-blog-tw/mobiledets-flops%E4%B8%8D%E7%AD%89%E6%96%BClatency-%E8%80%83%E9%87%8F%E4%B8%8D%E5%90%8C%E7%A1%AC%E9%AB%94%E7%9A%84%E9%AB%98%E6%95%88%E6%9E%B6%E6%A7%8B-5bfc27d4c2c8)
9. Howard et al. [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). 2017.
10. Sandler et al. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381). CVPR 2018.
11. Howard et al. [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244). ICCV 2019.
12. Xiong et al. [MobileDets: Searching for Object Detection Architectures for Mobile Accelerators](https://arxiv.org/abs/2004.14525). CVPR 2021.
