# CNN hiệu quả: ShuffleNet v2

Bài này đọc bài báo *ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design* (Ma, Zhang, Zheng, Sun, ECCV 2018). Trang [Code](dlnote/backbone_code_vi) đi kèm làm lại bốn thí nghiệm G1–G4 của bài báo trên CPU (code của bản gốc cho G1 có lỗi, đã sửa), cài phép trộn kênh (channel shuffle), và đo số tham số, MACs, độ trễ của ShuffleNet v2 so với MobileNet v2.

<!-- theory: qa/ch17_vi#17.9.6 ; qa/ch17_vi#17.8.3 -->

## Tóm tắt

Hiện nay thiết kế kiến trúc mạng neuron chủ yếu dựa vào độ phức tạp tính toán gián tiếp như `FLOPs`, nhưng thước đo trực tiếp như tốc độ còn phụ thuộc các yếu tố khác, như chi phí truy cập bộ nhớ và đặc điểm nền tảng. Vì vậy bài báo đề nghị dùng thước đo trực tiếp, chứ không chỉ `FLOPs`, và đo thẳng trên nền tảng đích. Qua một loạt thí nghiệm có kiểm soát, tác giả đưa ra **một số nguyên tắc thực tế để thiết kế kiến trúc mạng hiệu quả**, và từ đó đề xuất kiến trúc mới `ShuffleNet V2`. Các thí nghiệm so sánh cho thấy mô hình đạt cân bằng tốt nhất giữa tốc độ và độ chính xác (`state-of-the-art`).

## 1 Giới thiệu

Để đo độ phức tạp tính toán, thước đo được dùng rộng rãi là số phép tính số thực `FLOPs`; nhưng đó là thước đo gián tiếp, một ước lượng gần đúng cho thứ ta thực sự quan tâm như tốc độ hay độ trễ. Sự không nhất quán này đã được ghi nhận trong các công trình trước, ví dụ MobileNet v2 nhanh hơn NASNet-A nhiều dù FLOPs xấp xỉ nhau.

![Hình 1](../images/shufflenetv2/figure_1.png)

Hình 1.1: Độ chính xác (a, b) và tốc độ (c, d) theo FLOPs của bốn mạng trên GPU và ARM (hình 1 của bài báo).

Hình 1.1 cho thấy trên cả `GPU` và `ARM`, các mô hình cùng `FLOPs` có tốc độ chạy khác nhau đáng kể. Vì vậy chỉ dùng FLOPs để đo độ phức tạp là không đủ, và có thể dẫn tới thiết kế mạng không tối ưu.

Có hai nguyên nhân chính của sự không nhất quán: một là FLOPs không tính tới một số yếu tố quan trọng ảnh hưởng tới tốc độ, như `MAC` (Memory Access Cost, lượng truy cập bộ nhớ) và mức độ song song; hai là cùng FLOPs, mô hình có thể chạy nhanh chậm khác nhau trên các nền tảng khác nhau.

Do đó tác giả đề xuất hai nguyên tắc khi thiết kế mạng hiệu quả: một là dùng thước đo trực tiếp (tốc độ) thay vì gián tiếp (FLOPs); hai là đo trực tiếp trên nền tảng đích.

## 2 Các nguyên tắc thực tế để thiết kế mạng hiệu quả

Trước hết, tác giả phân tích thời gian chạy của hai mạng kinh điển ShuffleNet v1 và MobileNet v2. Môi trường chạy trên ARM và GPU như sau.

![Môi trường ARM và GPU](../images/shufflenetv2/specific_operating_environment_of_arm_and_gpu.png)

Hình 2.1: Cấu hình thí nghiệm: GPU NVIDIA GeForce GTX 1080Ti với cuDNN 7.0, batch 8; ARM Qualcomm Snapdragon 810, một luồng, batch 1.

![Hình 2](../images/shufflenetv2/figure_2.jpg)

Hình 2.2: Phân bổ thời gian chạy theo loại phép tính của ShuffleNet v1 và MobileNet v2 trên GPU và ARM (hình 2 của bài báo).

Hình 2.2 cho thấy dù tích chập (phần chiếm phần lớn `FLOPs`) chiếm phần lớn thời gian, các phép khác cũng tốn đáng kể thời gian chạy, như vào/ra dữ liệu, trộn kênh và các phép từng phần tử (cộng tensor, hàm kích hoạt). Vì vậy FLOPs không phải ước lượng chính xác của thời gian chạy thực tế. Từ đó tác giả rút ra bốn nguyên tắc:

1. **G1**: số kênh vào và ra bằng nhau thì `MAC` nhỏ nhất.
2. **G2**: tích chập nhóm với quá nhiều nhóm làm tăng MAC.
3. **G3**: mạng phân mảnh làm giảm mức độ song song.
4. **G4**: không được bỏ qua các phép từng phần tử.

### G1: số kênh vào và ra bằng nhau thì MAC của tích chập 1×1 nhỏ nhất

Các mạng hiện đại như `Xception`, `MobileNet`, `MobileNet V2`, `ShuffleNet` đều dùng tích chập tách được theo chiều sâu, trong đó tích chập theo điểm (tích chập $1\times 1$) chiếm phần lớn độ phức tạp tính toán (bài báo ShuffleNet có phân tích). Giả sử feature map vào có kích thước $h\times w\times c_1$, kernel có hình dạng $(c_2, c_1, 1, 1)$, kích thước không gian của đầu ra không đổi; khi đó `FLOPs` của tích chập $1 \times 1$ là $B = hwc_{1}c_{2}$.

> Bài báo tính `FLOPs` bằng cách coi một phép nhân-cộng là một phép tính, tức tương đương công thức `MACs` thường dùng.

Đơn giản hoá, giả sử bộ nhớ đệm của thiết bị đủ lớn để chứa toàn bộ feature map và tham số. Khi đó lượng truy cập bộ nhớ (số lần truy cập) của tầng tích chập $1 \times 1$ là $\text{MAC} = hwc_1 + hwc_2 + c_{1}c_{2} = hw(c_{1} + c_{2}) + c_{1}c_{2}$, ba số hạng lần lượt là feature map vào, feature map ra và trọng số. Theo bất đẳng thức AM-GM:

$$
\begin{aligned}
\text{MAC} &= hw(c_{1} + c_{2}) + c_{1}c_{2} \\
&= \sqrt{(hw)^{2}(c_{1} + c_{2})^{2}} + \frac{B}{hw} \\
&\geq \sqrt{(hw)^{2}\cdot 4c_{1}c_{2}} + \frac{B}{hw} \\
&= 2\sqrt{hwB} + \frac{B}{hw}
\end{aligned}
$$

Dấu bằng xảy ra khi $c_1 = c_2$, lúc đó $(c_{1} + c_{2})^{2} = 4c_{1}c_{2}$. Tức là **MAC đạt giá trị nhỏ nhất (với FLOPs cho trước) khi và chỉ khi $c_{1}=c_{2}$ (tích chập $1 \times 1$ có số kênh vào và ra bằng nhau)**.

Nhưng kết luận này chỉ đúng về lý thuyết: thực tế bộ nhớ đệm có thể không đủ lớn, và chiến lược cache khác nhau giữa các nền tảng. Vì vậy tác giả làm thêm thí nghiệm kiểm chứng: mạng mốc gồm 10 khối, mỗi khối hai tầng tích chập 1×1, tầng đầu có $c_{1}$ kênh vào và $c_{2}$ kênh ra, tầng sau ngược lại; giữ tổng FLOPs cố định và thay đổi tỉ lệ $c_{1}:c_{2}$ để đo tốc độ thật. Kết quả ở bảng 1:

![Bảng 1](../images/shufflenetv2/table_1.png)

Hình 2.3: Kiểm chứng G1: tốc độ (ảnh/giây) với các tỉ lệ $c_1:c_2$ khác nhau, cùng FLOPs, trên GPU (batch 1, 2, 4) và ARM (bảng 1 của bài báo).

> Thí nghiệm dùng khối tích chập rất thú vị; một điểm là không thể có quá ít tầng.

Khi tỉ lệ gần `1:1`, `MAC` của mạng nhỏ hơn và tốc độ đo được nhanh nhất. Có thể phân tích thêm bằng mô hình roofline: tích chập $1\times 1$ về bản chất bị giới hạn bởi bộ nhớ, nên chỉ có giảm `MAC` mới giảm được thời gian chạy từ gốc.

Code đo thử dưới đây do tác giả bản gốc viết, dựa theo thiết kế thí nghiệm của bài báo:

```python
import torch
import torch.nn as nn
import time

def benchmark(model, input_tensor, batch_size, device, num_iterations=100):
    model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()                     # chế độ suy luận
    for _ in range(10):              # chạy khởi động
        with torch.no_grad():
            model(input_tensor)
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            output = model(input_tensor)
    elapsed_time = time.time() - start_time
    return batch_size * num_iterations / elapsed_time   # ảnh/giây

# khối tích chập
class ConvBlock(nn.Module):
    def __init__(self, c1, c2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # tầng tích chập thứ nhất
        x = self.relu(self.conv2(x))  # tầng tích chập thứ hai
        return x

# mạng gồm 10 khối
class ConvNet(nn.Module):
    def __init__(self, c1, c2, num_blocks=10):
        super(ConvNet, self).__init__()
        self.blocks = nn.Sequential(*[ConvBlock(c1, c2) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)

input_size = (56, 56)
batch_sizes = [1, 2, 4]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# bốn tỉ lệ c1:c2, cùng FLOPs (c1 * c2 xấp xỉ 128 * 128)
channel_configs = {"1:1": (128, 128), "1:2": (90, 180), "1:6": (52, 312), "1:12": (36, 432)}

for ratio, (c1, c2) in channel_configs.items():
    print(f"Testing ratio {ratio} with channels ({c1}, {c2})")
    model = ConvNet(c1, c2).to(device)
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, c1, input_size[0], input_size[1])
        print(f"Batch size {batch_size}, Images/sec: {benchmark(model, input_tensor, batch_size, device):.2f}")
```

> **Biên tập:** Code gốc đặt `padding=1` cho tích chập 1×1. Với kernel 1×1, padding 1 làm feature map rộng thêm 2 điểm ảnh sau *mỗi* tầng: sau 20 tầng, 56×56 thành 96×96, FLOPs của các tầng sau lớn gần gấp ba tầng đầu, và vùng mở rộng chỉ toàn số 0 (cộng bias). Thí nghiệm vì vậy không còn "cùng FLOPs" ở mọi tầng như bài báo. Đã sửa thành `padding=0`. Tệp `shufflenetv2_expr.py` của bản gốc là cùng thí nghiệm với tích chập 3×3 (`padding=1`, đúng).

Tác giả bản gốc chạy code trên `MacBook M3 Pro` (CPU ARM), kết quả như hai hình dưới. Ngoài ra, với tích chập $3\times 3$ và batch lớn hơn 1, tức trường hợp bị giới hạn bởi tính toán, cho số kênh vào và ra bằng nhau không giảm được thời gian chạy.

![Kết quả thử G1 của tác giả](../images/shufflenetv2/g1test.png)

Hình 2.4: Kết quả thử G1 với tích chập 1×1 trên MacBook M3 Pro (code gốc, còn lỗi padding).

![Kết quả thử G1 của tác giả, 3x3](../images/shufflenetv2/g1test2.png)

Hình 2.5: Kết quả thử cùng thí nghiệm với tích chập 3×3 trên MacBook M3 Pro.

### G2: tích chập nhóm với quá nhiều nhóm làm tăng MAC

Tích chập nhóm là cốt lõi của thiết kế các mạng nhẹ hiện nay (`ShuffleNet/MobileNet/Xception/ResNeXt`). Nó giảm độ phức tạp tính toán nhờ **kết nối thưa** giữa các kênh (chỉ nối với đặc trưng cùng nhóm). Một mặt, nó cho phép dùng nhiều kênh hơn để tăng năng lực mạng và độ chính xác; mặt khác, nhiều kênh hơn kéo theo `MAC` lớn hơn.

Với tích chập nhóm $1 \times 1$:

> Công thức `FLOPs` của tích chập nhóm được suy ra ở [bài MobileNet v1](dlnote/bb_mobilenet_vi).

$$
B = h \cdot w \cdot 1 \cdot 1 \cdot \frac{c_1}{g} \cdot \frac{c_2}{g} \cdot g = \frac{hwc_{1}c_{2}}{g}
$$

$$
\text{MAC} = hw(c_{1} + c_{2}) + \frac{c_{1}c_{2}}{g} = hwc_{1} + \frac{Bg}{c_1}+\frac{B}{hw}
$$

trong đó $B$ là FLOPs của tầng, $g$ là số nhóm. Nếu cố định hình dạng đầu vào $c_{1} \times h \times w$ và chi phí tính toán $B$, thì $MAC$ tăng theo số nhóm $g$.

> Muốn giữ $B$ cố định khi tăng $g$ thì phải tăng $c_2$ tỉ lệ với $g$, nên số hạng $hwc_2 = \frac{Bg}{c_1}$ tăng tuyến tính theo $g$.

Bài báo kiểm chứng bằng mạng gồm 10 tầng tích chập nhóm 1×1 chồng nhau, giữ cùng chi phí tính toán (`FLOPs`) và đo thời gian chạy với các số nhóm khác nhau; kết quả ở bảng 2.

![Bảng 2](../images/shufflenetv2/table_2.png)

Hình 2.6: Kiểm chứng G2: tốc độ với số nhóm $g$ = 1, 2, 4, 8, cùng FLOPs (bảng 2 của bài báo).

Rõ ràng mạng dùng nhiều nhóm chạy chậm hơn: ví dụ 8 nhóm chậm hơn nhiều so với 1 nhóm, nhất là trên GPU, chủ yếu do `MAC` tăng. Vì vậy bài báo khuyên chọn số nhóm cẩn thận theo nền tảng phần cứng và bài toán đích, không nên chọn số nhóm lớn chỉ vì tăng được độ chính xác mà bỏ qua chi phí truy cập bộ nhớ (`MAC`) tăng lên.

### G3: mạng phân mảnh làm giảm mức độ song song

Trong họ `GoogLeNet` và các kiến trúc tìm tự động (NAS), mỗi khối (`block`) của mạng đều có cấu trúc nhiều nhánh (`multi-path`), dùng nhiều phép toán nhỏ (fragmented operators, toán tử phân mảnh) thay vì ít phép toán lớn; mỗi phép tích chập hay pooling trong khối được gọi là một toán tử phân mảnh. Ví dụ NASNet-A có số toán tử phân mảnh (tổng số phép tích chập hay pooling riêng lẻ trong một khối) là `13`. Ngược lại, trong mạng chuẩn như `ResNet`, con số này là 2 hoặc 3.

> Khối residual có hai loại, `basic block` và `bottleneck block`. "Fragment" là "mảnh", có thể hiểu đơn giản là số đơn vị hay số nhánh trong khối.

Dù các bài báo trước đã cho thấy cấu trúc phân mảnh (nhiều nhánh) giúp tăng độ chính xác, nó làm giảm hiệu quả, vì không thân thiện với thiết bị song song mạnh như `GPU`. Nó còn thêm chi phí khởi chạy kernel và đồng bộ.

> Kernel launching and synchronization: đồng bộ giữa các nhánh của cấu trúc nhiều nhánh.

Để đo mức độ ảnh hưởng của phân mảnh (network fragmentation) tới hiệu quả, bài báo đánh giá một loạt khối mạng có mức phân mảnh khác nhau. Cụ thể, mỗi khối gồm 1 tới 4 tích chập 1×1, xếp nối tiếp hoặc song song, như hình 1 phụ lục.

![Hình 1 phụ lục](../images/shufflenetv2/appendix_figure_1.png)

Hình 2.7: Các khối dùng để kiểm chứng G3: (a) 1 mảnh; (b) 2 mảnh nối tiếp; (c) 4 mảnh nối tiếp; (d) 2 mảnh song song; (e) 4 mảnh song song (hình 1 phụ lục của bài báo).

Mỗi khối được chồng 10 lần. Kết quả ở bảng 3 cho thấy phân mảnh làm giảm đáng kể tốc độ trên GPU, ví dụ cấu trúc 4 mảnh song song chậm hơn 1 mảnh khoảng 3 lần. Nhưng trên `ARM`, ảnh hưởng của phân mảnh tới tốc độ nhỏ hơn tương đối so với `GPU`.

![Bảng 3](../images/shufflenetv2/table_3.png)

Hình 2.8: Kiểm chứng G3 (bảng 3 của bài báo). Chú thích của tác giả bản gốc: số kênh của 4 khối kia được chỉnh để FLOPs bằng với khối 1 mảnh.

### G4: không được bỏ qua các phép từng phần tử

Như hình 2.2, trong các mô hình nhẹ như `MobileNet v2` và `ShuffleNet v1`, các phép từng phần tử (`element-wise`) chiếm nhiều thời gian, nhất là trên `GPU`.

Trong bài báo, các toán tử từng phần tử gồm `ReLU`, `AddTensor`, `AddBias`…; chúng có `FLOPs` nhỏ nhưng `MAC` tương đối lớn. Đặc biệt, bài báo coi `depthwise convolution` cũng là một toán tử từng phần tử, vì tỉ lệ `MAC/FLOPs` của nó cũng cao.

> Phép `shortcut` (cộng residual) cũng thuộc nhóm phép từng phần tử (`AddTensor`).

> **Biên tập:** Bản gốc ghi chú "phép shortcut không được tính là toán tử từng phần tử". Ngược lại: bài báo coi phép cộng của kết nối tắt là `AddTensor`, một phép từng phần tử, và thí nghiệm ngay dưới đây đo chính tác động của việc bỏ nó.

Bài báo thí nghiệm với khối "bottleneck" của `ResNet`, gồm tích chập $1 \times 1$, rồi $3 \times 3$, rồi $1 \times 1$, có `ReLU` và kết nối `shortcut`, cấu trúc như hình dưới. Bài báo bỏ dần `ReLU` và `shortcut`; bảng 4 là thời gian chạy của các biến thể. Khi bỏ cả ReLU và shortcut, tốc độ tăng khoảng `20%` trên cả GPU và ARM.

![Bottleneck](../images/shufflenetv2/bottleneck.png)

Hình 2.9: Khối bottleneck của ResNet: giảm chiều bằng 1×1, tích chập 3×3, rồi tăng chiều bằng 1×1.

![Bảng 4](../images/shufflenetv2/table_4.png)

Hình 2.10: Kiểm chứng G4: tốc độ của khối bottleneck khi bỏ ReLU và/hoặc shortcut, mỗi biến thể chồng 10 lần (bảng 4 của bài báo). Chú thích của tác giả bản gốc: bỏ ReLU dường như tăng tốc nhiều hơn bỏ shortcut.

**Kết luận và thảo luận.** Từ 4 nguyên tắc và các nghiên cứu thực nghiệm trên, một kiến trúc mạng hiệu quả nên:

1. dùng tích chập "cân bằng", tức số kênh vào và ra bằng nhau;
2. dùng tích chập nhóm hợp lý;
3. giảm mức phân mảnh;
4. giảm các phép từng phần tử.

**Mức tác dụng của 4 tính chất lý tưởng này phụ thuộc vào đặc điểm nền tảng** (thao tác bộ nhớ, tối ưu code…), nằm ngoài phạm vi lý thuyết của bài báo, nhưng khi thiết kế mạng thực tế nên cố gắng tuân theo.

Các tiến bộ trước đây về kiến trúc mạng nhẹ chủ yếu dựa trên thước đo `FLOPs` và không xét tới 4 tính chất trên. Ví dụ `ShuffleNet v1` dựa nhiều vào tích chập nhóm, vi phạm `G2`; `MobileNet v2` dùng cấu trúc bottleneck đảo ngược (inverted bottleneck), vi phạm `G1`, và dùng `ReLU` cùng tích chập DW trên các tầng mở rộng có nhiều kênh, vi phạm `G4`; các cấu trúc do `NAS` sinh ra rất phân mảnh, vi phạm `G3`.

## 3 ShuffleNet V2: một kiến trúc hiệu quả

**Xem lại ShuffleNet v1.** `ShuffleNet` là mạng tốt nhất thời đó, được dùng rộng rãi trên thiết bị cấu hình thấp (như điện thoại). Nó gợi cảm hứng cho bài báo này, nên được xem xét và phân tích trước.

Theo `ShuffleNet v1`, thách thức chính của mạng nhẹ là với ngân sách tính toán (`FLOPs`) cho trước, số kênh của feature map bị hạn chế. Để tăng số kênh mà không tăng đáng kể `FLOPs`, `ShuffleNet v1` dùng hai kỹ thuật: **tích chập nhóm theo điểm và cấu trúc kiểu bottleneck** (pointwise group convolutions and bottleneck-like structures); rồi thêm phép "channel shuffle" (trộn kênh) để các kênh của các nhóm khác nhau trao đổi thông tin, tăng độ chính xác. Khối của nó như hình 3(a)(b).

Từ phân tích ở mục 2, **tích chập nhóm theo điểm và cấu trúc bottleneck** đều làm tăng `MAC` (`G2` và `G1`). Chi phí này không thể bỏ qua, nhất là với mô hình nhẹ. Dùng quá nhiều nhóm cũng vi phạm `G3`. **Phép cộng từng phần tử** (element-wise "Add") trong `shortcut connection` cũng không tốt (`G4`). Vì vậy, để đạt **năng lực mô hình và hiệu quả** cao, vấn đề then chốt là **làm sao giữ được nhiều kênh và các kênh rộng bằng nhau, mà không dùng tích chập đặc cũng không dùng quá nhiều nhóm**.

> Câu trên khá khó hiểu. Cách hiểu của tác giả bản gốc: (1) các tầng tích chập trong khối có nhiều kênh và rộng bằng nhau, tức hai tầng nối nhau có số kênh lớn và bằng nhau; (2) "tích chập đặc" ở đây là tích chập $1\times 1$ thường (không chia nhóm); (3) nếu dùng tích chập nhóm thì số nhóm không nên quá nhiều.

**Phép tách kênh của ShuffleNet v2.** Trên nền khối của ShuffleNet v1, **khối ShuffleNet v2 thêm một toán tử đơn giản là tách kênh (`Channel Split`)** để đạt mục tiêu trên, như hình 3(c). Ở đầu mỗi khối, $c$ kênh của feature map vào được tách (`split`) thành hai nhánh (`branches`): $c-c^{'}$ kênh và $c^{'}$ kênh. Theo **G3** (càng ít mảnh càng tốt), một nhánh giữ nguyên (kết nối tắt), nhánh kia gồm ba tầng tích chập có cùng số kênh vào và ra, thoả **G1**. Khác với v1, hai tích chập $1 \times 1$ trong khối v2 không còn chia nhóm, một phần để thoả **G2**, một phần vì phép tách kênh ở đầu đã có tác dụng như chia nhóm.

Cuối cùng, kết quả hai nhánh được nối (`concatenate`), nên số kênh vào và ra của khối bằng nhau, thoả nguyên tắc **G1**. Giống `ShuffleNet v1`, **cả hai đều dùng phép trộn kênh (`channel shuffle`) để bảo đảm thông tin giữa hai nhánh được trao đổi**.

> Basic block và bottleneck block của ResNet cũng được thiết kế có số kênh vào, ra của khối bằng nhau, thoả **G1**.

![Hình 3](../images/shufflenetv2/figure_3.png)

Hình 3.1: (a) Khối ShuffleNet v1; (b) khối v1 giảm kích thước; (c) khối ShuffleNet v2; (d) khối v2 giảm kích thước 2 lần (hình 3 của bài báo).

Đầu ra sau khi trộn kênh là đầu vào của khối tiếp theo. Phép "Add" của ShuffleNet v1 không còn dùng; các toán tử từng phần tử như `ReLU` và tích chập DW chỉ còn ở nhánh bên phải. Đồng thời, ba toán tử liên tiếp: nối (`"Concat"`), trộn kênh (`"Channel Shuffle"`) và tách kênh (`"Channel Split"`) của khối kế tiếp được **gộp thành một toán tử từng phần tử**. Theo **G4**, các thay đổi này có lợi.

Với khối cần giảm kích thước không gian, khối được sửa lại: bỏ toán tử tách kênh, và số kênh ra của khối tăng gấp đôi; chi tiết như hình 3(d).

Chồng các khối ở hình 3(c)(d) tạo thành mô hình `ShuffleNet v2` cuối cùng. Để đơn giản, đặt $c^{'} = c/2$; mạng sau khi chồng giống ShuffleNet v1, cấu trúc chi tiết như bảng 5. Khác biệt giữa v1 và v2 là v2 thêm một tích chập $1 \times 1$ trước tầng global average pooling để trộn đặc trưng (mix up features), v1 không có. Giống v1, **số kênh của các khối v2 được nhân theo tỉ lệ `0.5x`, `1x`… để tạo các ShuffleNet v2 có độ phức tạp khác nhau**, gọi là ShuffleNet v2 0.5×, ShuffleNet v2 1×…

> Lưu ý: số kênh ở bảng 5 được chọn để điều khiển `FLOPs`, chỉnh cho FLOPs khớp với các công trình trước để so sánh công bằng; không dùng số kênh dạng `2^n` vì điều đó không liên quan tới độ chính xác.

![Bảng 5](../images/shufflenetv2/table_5.png)

Hình 3.2: Cấu trúc tổng thể của ShuffleNet v2 với bốn mức độ phức tạp (bảng 5 của bài báo). Chú thích của tác giả bản gốc: khối đầu của mỗi giai đoạn có tích chập 3×3 stride 2 và chỉ có 1 khối; các khối sau có stride 1 và được lặp lại.

> Theo phân tích ở trên, kiến trúc này tuân theo mọi nguyên tắc nên rất hiệu quả.

**Phân tích độ chính xác.** `ShuffleNet v2` không chỉ hiệu quả mà còn chính xác. Có hai lý do chính: một là khối hiệu quả cho phép dùng nhiều kênh đặc trưng hơn, năng lực mạng lớn hơn; hai là khi $c^{'} = c/2$, một nửa feature map đi thẳng qua khối hiện tại vào khối kế tiếp, giống việc tái sử dụng đặc trưng của `DenseNet` và `CondenseNet`.

> DenseNet là mạng tích chập có kết nối dày đặc: giữa hai tầng bất kỳ có kết nối trực tiếp, đầu vào mỗi tầng là hợp của đầu ra mọi tầng trước, và feature map tầng đó học được cũng đưa thẳng tới mọi tầng sau. Xem [bài DenseNet](dlnote/bb_densenet_vi).

Bài báo DenseNet phân tích kiểu tái sử dụng đặc trưng bằng cách vẽ chuẩn L1 của trọng số giữa các tầng, như hình 4(a). Có thể thấy **liên hệ giữa các tầng kề nhau mạnh hơn hẳn các tầng khác, tức kết nối dày đặc giữa mọi tầng có thể là thừa**; bài báo CondenseNet gần đây cũng ủng hộ quan điểm này.

![Hình 4](../images/shufflenetv2/figure_4.png)

Hình 3.3: Minh hoạ kiểu tái sử dụng đặc trưng của DenseNet (a) và ShuffleNet v2 (b) (hình 4 của bài báo).

Trong ShuffleNet V2, dễ chứng minh số kênh feature map được nối trực tiếp giữa khối thứ $i$ và khối thứ $i+j$ là $r^{j}c$, với $r=(c-c^{'})/c$. Nói cách khác, lượng đặc trưng tái sử dụng giữa hai khối giảm theo cấp số mũ khi khoảng cách giữa chúng tăng; các khối cách xa nhau tái sử dụng rất ít đặc trưng.

> **Biên tập:** Bản gốc viết $r=(1−c^{'})/c$; đúng là $r = (c - c')/c = 1 - c'/c$, phần kênh đi thẳng qua mỗi khối. Với $c' = c/2$ thì $r = 1/2$: sau $j$ khối, còn $c/2^j$ kênh của khối $i$ được chuyển nguyên vẹn (phép trộn kênh xen kẽ hai nửa, nên mỗi lần chỉ một nửa số kênh cũ rơi vào nhánh giữ nguyên).

> Tác giả bản gốc thấy cách hiểu liên hệ giữa hai khối ở hình 4 hơi khó.

Vì vậy, giống DenseNet, cấu trúc của ShuffleNet v2 tạo ra kiểu tái sử dụng đặc trưng theo thiết kế, nên đạt độ chính xác cao mà hiệu quả hơn; thí nghiệm ở bảng 8 đã chứng minh.

![Bảng 8](../images/shufflenetv2/table_8.png)

Hình 3.4: So sánh nhiều kiến trúc ở bốn mức độ phức tạp (khoảng 40, 140, 300, 500 MFLOPs): sai số top-1, FLOPs và tốc độ trên GPU và ARM (bảng 8 của bài báo).

## 4 Thí nghiệm

**Độ chính xác so với FLOPs.** Rõ ràng mô hình ShuffleNet v2 đề xuất vượt xa các mạng khác, nhất là với ngân sách tính toán nhỏ. Ngoài ra, MobileNet v2 làm việc kém ở ảnh $224 \times 224$ và ngân sách khoảng `40 MFLOPs`, có lẽ do quá ít kênh. Ngược lại, mô hình hiệu quả của bài báo dùng được nhiều kênh hơn nên không có nhược điểm đó. Như thảo luận ở mục 3, dù cả mô hình này và DenseNet đều tái sử dụng đặc trưng, mô hình này hiệu quả hơn nhiều.

**Tốc độ suy luận so với FLOPs và độ chính xác.** Bài báo so sánh tốc độ thực và `FLOPs` của bốn mô hình `ShuffleNet v2, MobileNet v2, ShuffleNet v1 và Xception`, như hình 1(c)(d); kết quả ở các độ phân giải khác trong bảng 1 phụ lục.

Dù độ chính xác của MobileNet v1 kém hơn, nó lại chạy nhanh hơn ShuffleNet v2 và các mạng khác. Bài báo cho rằng đó là vì MobileNet v1 thoả các nguyên tắc đề xuất (chẳng hạn **G3**: MobileNet v1 ít phân mảnh hơn ShuffleNet v2).

**Kết hợp với các phương pháp khác.** ShuffleNet v2 kết hợp với các phương pháp khác có thể tốt hơn nữa. Khi thêm mô-đun `SE`, mô hình chậm đi một chút nhưng độ chính xác phân loại tăng 0,5%. Cấu trúc khối như hình 2(b) phụ lục, kết quả ở bảng 8.

![Hình 2 phụ lục](../images/shufflenetv2/appendix_figure_2.png)

Hình 4.1: (a) Khối ShuffleNet v2 có thêm SE; (b) khối có thêm SE và kết nối residual, dùng cho mạng rất sâu (hình 2 phụ lục của bài báo).

Khối `Squeeze-and-Excitation (SE)` không phải một mạng hoàn chỉnh mà là một cấu trúc con, **tính tổng quát cao, cắm vào là dùng được**: có thể gắn vào các mô hình phân loại, phát hiện khác, kết hợp với `ResNeXt`, `ShuffleNet v2`… `SENet` học tương quan giữa các `channel`, tạo ra trọng số chú ý theo kênh; tốn thêm rất ít tính toán mà hiệu quả khá tốt. Cấu trúc khối `SE`:

![SE](../images/shufflenetv2/SE.png)

Hình 4.2: Khối Squeeze-and-Excitation: global average pooling ("squeeze") cho mỗi kênh một số, hai tầng FC với sigmoid ("excitation") cho trọng số từng kênh, rồi nhân lại vào feature map (hình 1 của bài báo SENet).

**Mở rộng cho mô hình lớn** (Generalization to Large Models). Dù các thí nghiệm loại bỏ (`ablation`) chủ yếu cho mạng nhẹ, `ShuffleNet v2` cũng không thua kém với mô hình lớn ($FLOPs \geq 2G$). **Bảng 6** so sánh độ chính xác phân loại ImageNet của `ShuffleNet v2` 50 tầng, `ShuffleNet v1` và `ResNet-50`: cùng `FLOPs = 2,3G`, ShuffleNet v2 chính xác hơn v1, và so với ResNet-50 thì ít hơn khoảng `40%` FLOPs mà chính xác hơn. Chi tiết mạng ở bảng 2 phụ lục.

![Bảng 6](../images/shufflenetv2/table_6.png)

Hình 4.3: So sánh với các mô hình lớn trên ImageNet (bảng 6 của bài báo).

Với ShuffleNet v2 rất sâu (như hơn `100` tầng), bài báo sửa nhẹ khối cơ bản bằng cách thêm một đường `residual` để mô hình hội tụ nhanh hơn. Bảng 6 có ShuffleNet v2 `164` tầng kèm mô-đun `SE`, chính xác hơn mô hình tốt nhất lúc đó là `SENet` mà ít `FLOPs` hơn.

**Đánh giá trên phát hiện đối tượng.** Để đánh giá khả năng tổng quát hoá, bài báo dùng `Light-Head RCNN` làm framework phát hiện và so sánh trên COCO. Bảng 7 cho thấy ở 4 mức độ phức tạp, mô hình dùng backbone ShuffleNet v2 chính xác hơn và nhanh hơn các mạng khác.

![Bảng 7](../images/shufflenetv2/table_7.png)

Hình 4.4: Kết quả phát hiện đối tượng trên COCO (bảng 7 của bài báo). Ảnh vào 800×1200; hàng FLOPs ghi độ phức tạp ở đầu vào 224×224; tốc độ GPU đo với batch 4; không đo trên ARM vì phép PSRoI Pooling mà Light-Head RCNN cần chưa có trên ARM.

So sánh kết quả phát hiện (bảng 7), về độ chính xác: `ShuffleNet v2 > Xception ≥ ShuffleNet v1 ≥ MobileNet v2`.

So sánh kết quả phân loại (bảng 8), về độ chính xác: `ShuffleNet v2 ≥ MobileNet v2 > ShuffleNet v1 > Xception`.

## 5 Kết luận

Bài báo khuyên khi thiết kế mạng nhẹ nên xét thước đo **trực tiếp** (như tốc độ `speed`) thay vì thước đo gián tiếp (như `FLOPs`). Bài báo đưa ra các nguyên tắc thực tế và một kiến trúc mới, ShuffleNet v2. Các thí nghiệm toàn diện đã kiểm chứng hiệu quả của mô hình. Tác giả hy vọng công trình gợi ý cho các thiết kế kiến trúc về sau chú trọng hơn tới đặc điểm nền tảng và tính thực dụng.

> Thước đo trực tiếp ở đây có thể là thời gian suy luận, độ trễ, hay tốc độ suy luận của mô hình; ý nghĩa như nhau.

## Bổ sung: làm lại G1–G4 trên CPU

Mục này do người biên soạn viết thêm. Trang Code làm lại bốn thí nghiệm trên CPU laptop (4 lõi hiệu năng cao, batch 1, feature map 56×56, mỗi mạng chồng 10 lần, cùng MACs trong mỗi thí nghiệm):

| Thí nghiệm | Cấu hình | Tốc độ (ảnh/s) |
| --- | --- | --- |
| G1, tích chập 1×1, $c_1:c_2$ | 1:1 / 1:2 / 1:6 / 1:12 | 69,8 / 56,6 / 54,6 / 47,0 |
| G2, tích chập nhóm 1×1, số nhóm $g$ | 1 / 2 / 4 / 8 | 145 / 128 / 89 / 71 |
| G3, số mảnh | 1 mảnh / 4 nối tiếp / 4 song song | 148 / 93 / 74 |
| G4, khối bottleneck | gốc / bỏ ReLU và shortcut | 30,3 ms / 26,3 ms (nhanh hơn 15%) |

Cả bốn nguyên tắc đều đúng chiều trên CPU này. Hai lưu ý: trên CPU có lõi lớn và lõi nhỏ, phải ghim tiến trình vào một loại lõi, nếu không nhiễu đo lớn hơn chính các hiệu ứng trên; và ShuffleNet v2 1.0× (145 triệu MACs) chỉ nhanh hơn MobileNet v2 (301 triệu MACs) khoảng 10% trên CPU này, một ví dụ nữa cho thông điệp chính của bài: FLOPs không phải tốc độ.

## 6 Suy nghĩ cá nhân của tác giả

Phân tích hiệu năng suy luận phải gắn với nền tảng suy luận cụ thể (thường gặp: `GPU` NVIDIA, `CPU ARM` di động, chip `NPU` nhúng…). Các yếu tố đã biết ảnh hưởng tới **hiệu năng suy luận** gồm: lượng tính toán của toán tử `FLOPs` (số tham số `Params`) và lượng truy cập bộ nhớ của toán tử (băng thông bộ nhớ). Nhưng với cùng phần cứng và cùng kiến trúc mạng, tỉ lệ giảm `FLOPs` tỉ lệ thuận với tỉ lệ giảm thời gian suy luận.

Ví dụ trên `GPU`, toán tử tích chập `Depthwise` thực chất gồm nhiều thao tác ít `FLOPs` nhưng đọc ghi dữ liệu nhiều. Các thao tác đọc ghi nhiều này, cộng với giới hạn băng thông bộ nhớ GPU, khiến mô hình mất nhiều thời gian đọc ghi dữ liệu trong bộ nhớ GPU, và sức tính của GPU không được "dùng hết". Kết luận lấy từ bài viết trên Zhihu [FLOPs và tốc độ suy luận của mô hình](https://zhuanlan.zhihu.com/p/122943688).

Cuối cùng, hệ thống huấn luyện `AI` đã có một chuẩn đánh giá được công nhận là `MLPerf`, nhưng hệ thống suy luận thì tác giả cho rằng chưa có chỉ số được công nhận chung. Hiệu năng của hệ thống `Training` đo được bằng một tiêu chí đơn giản, "thời gian để đạt một độ chính xác cho trước"; còn với `Inference`, khó tìm một chỉ số đơn giản: `Latency`, `Throughput`, `Power`, `Cost`…, chỉ số nào phù hợp?

> **Biên tập:** Hai điểm. (1) "Cùng phần cứng, cùng kiến trúc thì tỉ lệ FLOPs bằng tỉ lệ thời gian" chỉ gần đúng khi các tầng vẫn bị giới hạn bởi tính toán; khi thu nhỏ quá (như hệ số độ rộng nhỏ, độ phân giải nhỏ), các tầng chuyển sang bị giới hạn bởi bộ nhớ hoặc chi phí gọi kernel và thời gian giảm chậm hơn FLOPs. (2) MLPerf có bộ đo **suy luận** (MLPerf Inference, từ 2019), với các kịch bản chuẩn: một luồng (đo độ trễ), nhiều luồng, máy chủ (thông lượng với ràng buộc độ trễ), ngoại tuyến (thông lượng), và có hạng mục đo điện năng; chính các kịch bản này trả lời câu hỏi "chỉ số nào" bằng cách đo nhiều chỉ số theo từng tình huống sử dụng.

## Tài liệu tham khảo

1. [Group Convolution, Depthwise Convolution và Global Depthwise Convolution](https://www.cnblogs.com/shine-lee/p/10243114.html)
2. [Tích chập nhóm và tích chập tách được theo chiều sâu](https://linzhenyuyuchen.github.io/2020/05/09/%E5%88%86%E7%BB%84%E5%8D%B7%E7%A7%AF%E5%92%8C%E6%B7%B1%E5%BA%A6%E5%8F%AF%E5%88%86%E7%A6%BB%E5%8D%B7%E7%A7%AF/)
3. [Hiểu vì sao tích chập nhóm và tích chập tách được giảm số tham số](https://zhuanlan.zhihu.com/p/65377955)
4. [Tích chập tách được theo chiều sâu (Xception và MobileNet)](https://www.jianshu.com/p/38dc74d12fcf)
5. [Cài đặt MobileNetV1](https://www.cnblogs.com/linzzz98/articles/13453810.html)
6. [Mạng neuron nhẹ: giải thích ShuffleNetV2](https://www.jiqizhixin.com/articles/2019-06-03-14)
7. [ShuffleNetV2: 4 nguyên tắc thực tế cho mạng hiệu quả](https://zhuanlan.zhihu.com/p/42288448)
8. Zhang et al. [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083). CVPR 2018.
9. Sandler et al. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381). CVPR 2018.
10. Howard et al. [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). 2017.
11. Hu et al. [Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf). CVPR 2018.
12. Ma et al. [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164). ECCV 2018.
