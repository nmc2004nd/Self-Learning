# Triển khai mô hình: phân tích độ phức tạp của CNN

Trang [Code](dlnote/deploy_code_vi) đi kèm (mục 1) kiểm tra các công thức FLOPs, MACs, số tham số bằng `thop`, `torchinfo` và `torch.utils.flop_counter`, lập lại bảng độ phức tạp của các backbone kèm độ trễ đo thật, vẽ mô hình roofline cho CPU chạy trang, và in các giới hạn của FP16, BF16, FP8.

<!-- theory: qa/ch17_vi#17.6 ; qa/ch17_vi#17.8.3 -->

## Lời nói đầu

Các mô hình nhẹ hiện nay như họ MobileNet, ShuffleNet, CSPNet, RepVGG, VoVNet đều phải dựa vào nền tảng tính toán cụ thể (CPU, GPU, ASIC…) mới phát huy hết ưu điểm của kiến trúc.

1. Nền tảng tính toán có hai chỉ số chính: **khả năng tính toán** $\pi$ và **băng thông** $\beta$.

- **Khả năng tính toán** (sức tính): số phép tính số thực lớn nhất nền tảng làm được mỗi giây, đơn vị `FLOPS`.
- **Băng thông**: lượng dữ liệu lớn nhất nền tảng chuyển được mỗi giây (lượng trao đổi bộ nhớ mỗi giây), đơn vị `Byte/s`.

Chia hai chỉ số cho nhau được **cận trên cường độ tính toán** $I_{max}$ của nền tảng: số phép tính mà mỗi byte nạp từ (hay ghi vào) bộ nhớ phải "gánh" để nền tảng dùng hết sức tính.

$$I_{max} = \frac {\pi }{\beta}$$

> "Bộ nhớ" ở đây hiểu theo nghĩa rộng: với `CPU` là bộ nhớ chính (`RAM`), với `GPU` là bộ nhớ của GPU (VRAM, HBM).

2. Tương ứng với hai chỉ số của nền tảng, mô hình có hai **chỉ số gián tiếp** phản ánh tốc độ: lượng tính toán `FLOPs` và lượng truy cập bộ nhớ `MAC`.

- **Lượng tính toán (FLOPs)**: số phép tính số thực khi đưa vào một mẫu (một ảnh) và chạy một lần lan truyền xuôi, tức độ phức tạp thời gian của mô hình, đơn vị `FLOPs`.
- **Lượng truy cập bộ nhớ (MAC)**: tổng lượng dữ liệu trao đổi với bộ nhớ khi đưa vào một mẫu và chạy một lần lan truyền xuôi, đơn vị `Byte`. Vì trọng số CNN thường ở kiểu `float32`, số phần tử phải nhân `4`. `MAC` của mỗi tầng gồm ba phần: đọc `feature map` đầu vào, đọc trọng số (đọc `DDR`) và ghi `feature map` đầu ra (ghi `DDR`).
- **Cường độ tính toán** của mô hình $I = \frac{FLOPs}{MAC}$: **mỗi byte trao đổi với bộ nhớ được dùng cho bao nhiêu phép tính**, đơn vị `FLOPs/Byte`. Cường độ càng lớn thì mô hình dùng bộ nhớ càng hiệu quả.
- **Hiệu năng lý thuyết** $P$ của mô hình: chỉ số ta quan tâm nhất, tức số phép tính mỗi giây (lý thuyết) mà mô hình đạt được trên nền tảng, đơn vị `FLOPS` hay `FLOP/s`. Mô hình roofline (`Roofline Model`) cho cách tính chỉ số này (mục 4).

> **Biên tập:** Bản gốc đặt tên cho lượng truy cập bộ nhớ là "độ phức tạp không gian". Độ phức tạp không gian thường chỉ bộ nhớ *chiếm giữ* (tham số và kích hoạt cần giữ cùng lúc), còn MAC là lượng dữ liệu *di chuyển*; một tầng có thể chiếm ít bộ nhớ nhưng đọc lại cùng dữ liệu nhiều lần. Bản gốc cũng dùng chữ viết tắt `AI` (arithmetic intensity) mà không giải thích; đã thay bằng "cường độ tính toán". Lưu ý chữ `MAC` ở bài này mang hai nghĩa: *memory access cost* (lượng truy cập bộ nhớ, như trong bài ShuffleNet v2) và *multiply-accumulate* (phép nhân-cộng, mục 1).

## 1 Phân tích lượng tính toán

> Chạy học sâu trên thiết bị đầu cuối phải cân nhắc yêu cầu về bộ nhớ và sức tính, nên cần phân tích độ phức tạp của mô hình: lượng tính toán (độ phức tạp thời gian) và số tham số (độ phức tạp không gian).

Để phân tích độ phức tạp tính toán, thước đo được dùng rộng rãi là số phép tính số thực khi suy luận (`FLOPs`), tức lượng tính toán lý thuyết. Nhưng đó là thước đo **gián tiếp**, một ước lượng gần đúng cho thứ ta thực sự quan tâm là tốc độ hay độ trễ.

Bài này giả định kernel tích chập vuông, cạnh `K`.

- `FLOPs` (floating point operations): số phép tính số thực, **hiểu là lượng tính toán**, dùng đo độ phức tạp thời gian của thuật toán, mô hình.
- `FLOPS` (viết hoa toàn bộ, floating-point operations per second): số phép tính số thực mỗi giây, hiểu là tốc độ tính toán, dùng đo hiệu năng phần cứng hay tốc độ của mô hình.
- `MACCs` (multiply-accumulate operations): số phép nhân-cộng; `MACCs` xấp xỉ một nửa FLOPs. Mỗi số hạng $w[0]\cdot x[0] + \dots$ được tính là một phép nhân-cộng, tức `1` `MACC`.

> Lưu ý: nhiều bài viết trên mạng nhầm lẫn MACCs và MACC, có lẽ do dịch máy các bài tiếng Anh không chuẩn; xem thêm [liên kết này](https://machinethink.net/blog/how-fast-is-my-model/). Cũng cần nói thêm rằng nhiều phần cứng hiện nay coi **phép nhân-cộng là một lệnh duy nhất** (FMA).

**FLOPs của tầng tích chập**:

> Tích chập bản chất là phép toán tuyến tính, giả sử kernel vuông cạnh $K$. Công thức dưới viết theo cách dễ hiểu; để dễ nhớ, thường viết gọn thành $MACCs = H \times W \times K^2 \times C_i \times C_o$.

- $FLOPs=(2\times C_i\times K^2-1)\times H\times W\times C_o$ (không tính bias)
- $FLOPs=(2\times C_i\times K^2)\times H\times W\times C_o$ (có bias)
- $MACCs=(C_i\times K^2)\times H\times W\times C_o$

**$C_i$ là số kênh của feature map đầu vào, $K$ là cạnh kernel, $H, W, C_o$ là chiều cao, chiều rộng và số kênh của feature map đầu ra.** Quá trình tích chập hai chiều như hình dưới:

> Tích chập hai chiều là phép toán khá đơn giản: bắt đầu từ kernel, một ma trận trọng số nhỏ. Kernel "trượt" trên dữ liệu vào hai chiều, nhân từng phần tử với phần đầu vào nó đang phủ rồi cộng lại thành một điểm ảnh đầu ra.

![Quá trình tích chập](../images/flops/conv_dynamic_visual.gif)

Hình 1.1: Kernel trượt trên đầu vào (nguồn: [Multi-Label Classification and Class Activation Map on Fashion-MNIST](https://towardsdatascience.com/multi-label-classification-and-class-activation-map-on-fashion-mnist-1454f09f5925)).

Giải thích công thức:

**Hiểu công thức `FLOPs` theo hai bước.** Phần trong ngoặc là bước một: **tính một `pixel` của `output feature map`, rồi nhân với $H\times W\times C_o$** để mở rộng ra toàn bộ output feature map. Phần trong ngoặc lại chia làm hai: $(2\times C_i\times K^2-1)=(C_i\times K^2) + (C_i\times K^2-1)$. Số hạng đầu là số phép nhân, số hạng sau là số phép cộng, vì cộng $n$ số cần $n-1$ phép cộng; nên khi không có `bias` thì có $-1$, có `bias` thì phép cộng bias bù đúng vào, phần trong ngoặc thành $(2\times C_i\times K^2)$.

> **Biên tập:** Bản gốc ghi "(có bias)" ở cả công thức MACCs; số phép nhân-cộng không phụ thuộc bias (cộng bias là $H W C_o$ phép cộng riêng, không phải phép nhân-cộng). Trang Code kiểm tra trên một tầng 64→128 kênh, kernel 3, đầu ra 56×56: công thức cho $231{,}2$ triệu MACs, khớp `thop`; còn `torch.utils.flop_counter` luôn báo $2\times$MACs bất kể có bias hay không. Mỗi công cụ một quy ước, nên khi so FLOPs giữa các bài báo phải kiểm tra họ đếm FLOPs hay MACs.

**FLOPs của tầng kết nối đầy đủ**:

Gọi $I$ là số chiều đầu vào, $O$ là số chiều đầu ra:

- không có bias: $FLOPs = (I + I -1) \times O = (2I − 1)O$;
- có bias: $FLOPs = (I + I -1) \times O + O = 2IO$.

### 1.1 Tỉ lệ sử dụng tính toán (Utilization)

Tỉ lệ sử dụng (utilization) là phần trăm sức tính thô của chip được dùng hiệu quả cho khối lượng công việc thực. Học sâu dùng tương đối ít phép toán nguyên thuỷ (computational primitives), nhưng chính những phép này chiếm phần lớn thời gian tính. Nhân ma trận (MM) và chuyển vị là các phép cơ bản. MM gồm các phép nhân-cộng (MAC). Chỉ số OPs/s (số phép toán mỗi giây) tính từ số MAC làm được mỗi giây (phép nhân và phép cộng mỗi phép tính là 1 operation, nên một MAC là 2 OP). Vì vậy có thể định nghĩa tỉ lệ sử dụng là tỉ số giữa sức tính thực dùng và sức tính thô:

$$ \text{mac utilization} = \frac {\text{used OPs/s}}{\text{raw OPs/s}} = \frac {FLOPs/\text{time (s)}}{\text{Raw FLOPs}} \qquad (\text{Raw FLOPs} = 1{,}7\text{ T với chip 3519})$$

> **Biên tập:** Trang Code đo con số này cho từng tầng trên CPU chạy trang (cột "% trần" ở mục 1.4): tích chập 3×3 thường đạt 90–100% đỉnh, nhưng tích chập depthwise chỉ 25–40% *trần roofline* của nó, và trần đó vốn đã thấp vì cường độ tính toán nhỏ.

## 2 Tính số tham số

Số tham số (params) của mô hình quyết định trực tiếp kích thước mô hình, và ảnh hưởng tới bộ nhớ chiếm khi suy luận; đơn vị thường là `M` (triệu). Trên `GPU`, tham số thường lưu ở `float32`, nên kích thước mô hình (byte) bằng `4` lần số tham số. Ở đây vẫn xét kernel vuông cạnh `K`.

**Số tham số của tầng tích chập** $= C_i\times K^2\times C_o + C_o$.

$C_i$ là số kênh vào, $K$ là cạnh kernel, $C_o$ là số kênh ra (cũng là số bộ lọc); số hạng thứ hai là số tham số của bias. (Thường không ghi bias vì nó không ảnh hưởng tới bậc độ lớn của tổng tham số; ở đây ghi vào cho chính xác.)

Ví dụ đầu vào 96×96×3, tầng tích chập đầu tiên dùng 16 kernel 5×5: số tham số của tầng là $5\times5\times3\times16+16=1216$.

**Số tham số của tầng `BN`** $= 2\times C_i$.

trong đó $C_i$ là số kênh vào (tầng BN có hai tham số học: hệ số co giãn $\gamma$ và hệ số dịch $\beta$).

> **Biên tập:** Ngoài hai tham số học, tầng BN còn lưu $2C_i$ giá trị thống kê chạy (running mean, running var) dưới dạng *buffer*; chúng không được tính trong `model.parameters()` nhưng có trong `state_dict` và trong tệp mô hình. Khi suy luận, BN thường được hợp nhất vào tầng tích chập trước nó nên không còn tham số nào; xem [bài hợp nhất Conv và BN](dlnote/deploy_fuse_vi).

**Số tham số của tầng kết nối đầy đủ** $= T_i\times T_o + T_o$.

$T_i$ là độ dài vector vào, $T_o$ là độ dài vector ra, số hạng thứ hai là bias. (Tầng kết nối đầy đủ hiện đã dần được thay bằng `Global Average Pooling`.) Lưu ý số tham số (bộ nhớ chiếm) của tầng kết nối đầy đủ lớn hơn rất nhiều so với tầng tích chập.

> **Biên tập:** Câu cuối đúng với mạng kiểu AlexNet, VGG (ba tầng FC của VGG-16 chiếm 124 trong 138 triệu tham số), nhưng không đúng với các mạng dùng global average pooling: tầng FC cuối của ResNet-50 có 2,05 triệu trong 25,6 triệu tham số (8%).

### 2.1 Tính lượng truy cập bộ nhớ

`MAC` (`memory access cost`), lượng truy cập bộ nhớ, còn gọi là lượng sử dụng bộ nhớ: tổng lượng dữ liệu trao đổi với bộ nhớ khi đưa vào một mẫu (một ảnh) và chạy một lần lan truyền xuôi của mô hình hay một tầng, đơn vị `Byte`.

Phân tích số tham số giúp biết bộ nhớ chiếm; còn băng thông bộ nhớ trong nhiều trường hợp quan trọng hơn `FLOPs`, vì với kiến trúc máy tính hiện nay, một lần truy cập bộ nhớ chậm hơn một phép tính rất nhiều. `MAC` của mỗi tầng `CNN` gồm:

- đọc `feature map` đầu vào (đọc `DDR`);
- đọc trọng số (đọc `DDR`);
- ghi `feature map` đầu ra (ghi `DDR`).

Lấy tầng tích chập làm ví dụ: đầu vào `feature map` (`Cin, Hin, Win`), đầu ra (`Cout, Hout, Wout`), kernel (`Cout, Cin, K, K`), MAC lý thuyết (thường nhỏ hơn MAC thực tế) là:

```python
# Mô hình đã lượng tử hoá INT8 để suy luận trên thiết bị: mỗi phần tử 1 byte
input = Hin * Win * Cin            # feature map đầu vào
output = Hout * Wout * Cout        # feature map đầu ra
weights = K * K * Cin * Cout + bias  # bias là độ lệch của tầng
ddr_read = input + weights
ddr_write = output
MAC = ddr_read + ddr_write
```

> `feature map` thường có dạng (`N, C, H, W`). Chỉ số `MAC` thường dùng cho suy luận trên thiết bị đầu cuối, nơi mỗi lần thường chỉ suy luận một khung hình, tức `N = 1` (`batch_size = 1`), khác với huấn luyện, nơi `batch_size` thường lớn hơn 1.

## 3 Đơn vị của lượng tính toán và số tham số

### 3.1 Khả năng tính toán số thực

`FLOPS`: số phép tính số thực mỗi giây. Phép tính số thực gồm mọi phép tính có phần thập phân, tốn thời gian hơn phép tính số nguyên. Dưới đây là các đơn vị đo khả năng tính toán số thực. Thường dùng `TFLOPS` (với số nguyên là `TOPS`) để đo sức tính của `NPU/GPU`, chẳng hạn chip HiSilicon `3519AV100` có sức tính mạng neuron `1,7 TOPS`.

- `MFLOPS` (megaFLOPS): $10^6$ phép tính số thực mỗi giây.
- `GFLOPS` (gigaFLOPS): $10^9$ phép tính mỗi giây.
- `TFLOPS` (teraFLOPS): $10^{12}$ phép tính mỗi giây.
- `PFLOPS` (petaFLOPS): $10^{15}$ phép tính mỗi giây.
- `EFLOPS` (exaFLOPS): $10^{18}$ phép tính mỗi giây.

> **Biên tập:** Bản gốc viết `TFLOPS (Tops)` như hai tên của cùng một đơn vị. `TOPS` (tera operations per second) thường dùng cho phép tính số nguyên (INT8), `TFLOPS` cho số thực; sức tính của NPU nhúng như 3519 được công bố bằng TOPS ở INT8, nên không so trực tiếp với TFLOPS FP32 của GPU.

`params`: số tham số của mô hình, quyết định kích thước mô hình, thường tính bằng "triệu" (million) hoặc "tỉ" (billion):

- `M` là million (triệu): mô hình 100M tham số có $10^8$ tham số.
- `B` là billion (tỉ): mô hình 2B tham số có $2\times10^9$ tham số.

### 3.2 Độ chính xác kép, đơn và nửa

Sức tính số thực của `CPU/GPU` phải phân biệt theo độ chính xác: độ chính xác kép `FP64`, đơn `FP32` và nửa `FP16`. Số bit khác nhau cho độ chính xác biểu diễn khác nhau, nên sai số tính toán khác nhau. Tính toán khoa học cần dải số rộng và tính chính xác thì phải dùng độ chính xác kép; xử lý đa phương tiện và đồ hoạ thông thường thì số thực 32 bit là đủ; với các ứng dụng chấp nhận độ chính xác thấp hơn như học máy, số thực 16 bit, thậm chí 8 bit, đã dùng được.

`CPU` có thể cùng lúc hỗ trợ phép tính với các độ chính xác khác nhau, còn `GPU` cần các đơn vị tính riêng cho độ chính xác đơn và kép.

Bộ nhớ mà tham số chiếm phụ thuộc vào **kiểu dữ liệu và độ chính xác**, thường gặp:

- `FP64`: số thực độ chính xác kép (64 bit, 8 byte).
- `FP32`: số thực độ chính xác đơn (32 bit, 4 byte), gồm 1 bit dấu, 8 bit mũ và 23 bit phần định trị.
- `FP16`: số thực nửa độ chính xác (16 bit, 2 byte), gồm 1 bit dấu, 5 bit mũ và 10 bit phần định trị.
- `BF16`: dạng rút gọn của `FP32` theo IEEE 754 (16 bit, 2 byte), gồm 1 bit dấu, 8 bit mũ và 7 bit phần định trị.

Tóm lại: so với FP32, dùng BF16/FP16 có thể tăng gần gấp đôi thông lượng (throughput) và giảm một nửa bộ nhớ. Nhưng độ chính xác của hai kiểu khác nhau: BF16 có **dải động** rộng, tương đương `float32`, nhưng phần định trị ngắn; FP16 có dải động hẹp, nhưng phần định trị dài hơn.

> **Biên tập:** Bản gốc nói BF16 "biểu diễn được dải số nguyên rộng hơn" và FP16 "dải số nguyên hẹp hơn". Cái khác nhau là **dải động** (do số bit mũ quyết định), không phải dải số nguyên; về số nguyên, FP16 biểu diễn chính xác mọi số tới 2048 còn BF16 chỉ tới 256 (trang Code: `257` thành `256` khi đổi sang BF16, vẫn giữ nguyên ở FP16). Với FP16, giá trị lớn nhất là 65504 (`70000` thành `inf`) và số chuẩn nhỏ nhất là $6{,}1\times10^{-5}$; với BF16 là $3{,}4\times10^{38}$ và $1{,}2\times10^{-38}$, như FP32.

![Định dạng bfloat16, float32, float16](../images/flops/fp32_fp16_bf16.png)

Hình 3.1: Bố cục bit và dải biểu diễn của bfloat16, float32 và float16.

### 3.3 Độ chính xác bfloat16

`bfloat16` là định dạng 16 bit khác do Google phát triển, tên đầy đủ "Brain Floating Point Format". Định dạng FP16 gốc của IEEE không được thiết kế cho học sâu, **dải động quá hẹp**. BFLOAT16 ra đời để giải quyết vấn đề đó, có dải động như FP32.

Định dạng bfloat16:

- bit dấu: 1 bit;
- độ rộng phần mũ: 8 bit;
- độ chính xác phần định trị: 7 bit (lưu tường minh; cộng bit ẩn là 8 bit), thay vì 23 bit (24 bit tính cả bit ẩn) của số thực độ chính xác đơn.

![Định dạng bfloat16](../images/flops/bfloat16_format.png)

Hình 3.2: Định dạng bfloat16 (nguồn: [Wikipedia: bfloat16 floating-point format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)).

bfloat16 thực chất là FP32 IEEE 754 bị cắt bớt, nên chuyển đổi qua lại với FP32 rất nhanh. Khi chuyển sang bfloat16, **phần mũ được giữ nguyên, còn phần định trị được rút ngắn**.

> **Biên tập:** Bản gốc viết phần định trị được rút ngắn "bằng cách cắt trực tiếp, bỏ qua trường hợp NaN". Cắt (truncation) là cách rẻ nhất, nhưng PyTorch, TensorFlow và các lệnh phần cứng (như `VCVTNEPS2BF16` của x86) **làm tròn tới số gần nhất, chẵn khi bằng nhau** (round-to-nearest-even), và giữ NaN là NaN. Trang Code cho ví dụ một số FP32 mà 16 bit thấp bị bỏ lớn hơn nửa đơn vị cuối của BF16: cắt cho `0100000001001001`, còn PyTorch cho `0100000001001010`.

Mạng neuron nhạy với độ lớn của phần mũ hơn là của phần định trị, và khác với `float16` thường cần xử lý đặc biệt (như loss scaling, nhân loss lên để gradient nhỏ không bị làm tròn về 0), `bfloat16` có thể thay thẳng `float32` khi huấn luyện và chạy mạng neuron sâu. Vì vậy đội phần cứng của Google chọn `bfloat16` cho Cloud TPU, để tăng hiệu quả phần cứng mà vẫn huấn luyện chính xác được, và giảm tối thiểu chi phí chuyển đổi với `float32`. Xem thêm [bài chip AI](dlnote/deploy_chips_vi).

### 3.4 Công cụ phân tích số tham số và lượng tính toán

1. [torchinfo](https://github.com/TylerYep/torchinfo): bản thay thế cho torchsummary (đã ngừng cập nhật), in ra một lần kích thước `feature map` và số tham số của từng tầng, cùng tổng số tham số `params` và lượng tính toán `MACs` của mô hình; **hỗ trợ nhiều đầu vào**.
2. [thop: pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter): in ra một lần tổng lượng tính toán `MACs` và số tham số `params`, hỗ trợ khai báo cách đếm cho toán tử tự định nghĩa.

Cả hai đều cài đơn giản, như `pip install torchinfo`. Ví dụ đếm số tham số và lượng tính toán của `resnet50`:

```python
import torchvision, torch

model = torchvision.models.resnet50()

# 1. torchinfo
from torchinfo import summary
summary(model, (1, 3, 224, 224), depth=3)   # resnet50: 25,557M tham số, 4,09 G mult-adds

# 2. thop
from thop import profile, clever_format
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print("The resnet50 model info: ", macs, params)  # resnet50: 4.134G 25.557M

# 3. PyTorch có sẵn (không cần cài thêm): đếm FLOPs theo toán tử ATen
from torch.utils.flop_counter import FlopCounterMode
with FlopCounterMode(display=False) as fc:
    model(input)
print(fc.get_total_flops() / 1e9)   # 8.178 GFLOPs = 2 x 4.089 G MACs
```

Kết quả `torchinfo` (rút gọn; bản gốc in đủ 150 dòng, một dòng cho mỗi tầng):

```bash
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   [1, 1000]                 --
├─Conv2d: 1-1                            [1, 64, 112, 112]         9,408
├─BatchNorm2d: 1-2                       [1, 64, 112, 112]         128
├─ReLU: 1-3                              [1, 64, 112, 112]         --
├─MaxPool2d: 1-4                         [1, 64, 56, 56]           --
├─Sequential: 1-5                        [1, 256, 56, 56]          --
│    └─Bottleneck: 2-1                   [1, 256, 56, 56]          --
│    │    └─Conv2d: 3-1                  [1, 64, 56, 56]           4,096
│    │    └─BatchNorm2d: 3-2             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-3                    [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-4                  [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-5             [1, 64, 56, 56]           128
│    │    └─ReLU: 3-6                    [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-7                  [1, 256, 56, 56]          16,384
│    │    └─BatchNorm2d: 3-8             [1, 256, 56, 56]          512
│    │    └─Sequential: 3-9              [1, 256, 56, 56]          16,896
│    │    └─ReLU: 3-10                   [1, 256, 56, 56]          --
...
├─Sequential: 1-8                        [1, 2048, 7, 7]           --
│    └─Bottleneck: 2-16                  [1, 2048, 7, 7]           --
│    │    └─Conv2d: 3-140                [1, 512, 7, 7]            1,048,576
│    │    └─BatchNorm2d: 3-141           [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-142                  [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-143                [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-144           [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-145                  [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-146                [1, 2048, 7, 7]           1,048,576
│    │    └─BatchNorm2d: 3-147           [1, 2048, 7, 7]           4,096
│    │    └─ReLU: 3-148                  [1, 2048, 7, 7]           --
├─AdaptiveAvgPool2d: 1-9                 [1, 2048, 1, 1]           --
├─Linear: 1-10                           [1, 1000]                 2,049,000
==========================================================================================
Total params: 25,557,032
Trainable params: 25,557,032
Non-trainable params: 0
Total mult-adds (G): 4.09
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 177.83
Params size (MB): 102.23
Estimated Total Size (MB): 280.66
==========================================================================================
```

Lưu ý các công cụ có thể cho lượng tính toán và số tham số khác nhau vì cách đếm khác nhau, nhưng đều khá chính xác. Với ResNet-50, `torchinfo` và `flop_counter` cho 4,089 G MACs (chỉ đếm tích chập và tầng FC), còn `thop` cho 4,134 G vì đếm thêm phép tính của các tầng BN và pooling (ReLU được tính là 0, phép cộng residual không phải mô-đun nên không được đếm).

> **Biên tập:** Bài báo ResNet ghi ResNet-50 có "3,8×10⁹ FLOPs"; con số đó thực ra là số phép nhân-cộng (MACs) và đếm theo một quy ước hơi khác. Tính theo nghĩa FLOPs (nhân và cộng là hai phép), ResNet-50 khoảng 8,2 GFLOPs. Nhiều bài báo về mạng nhẹ dùng chữ "FLOPs" cho MACs; khi so cần đọc kỹ.

Bảng độ phức tạp của các backbone kinh điển, kèm độ trễ batch 1 đo trên CPU chạy trang Code (laptop Intel Core i5-1340P, 4 luồng, PyTorch 2.14 eager):

| Mô hình | Tham số (M) | MACs (G) | Độ trễ CPU (ms) | ms / GMAC |
| --- | --- | --- | --- | --- |
| AlexNet | 61,10 | 0,714 | 12,6 | 17,7 |
| VGG-16 | 138,36 | 15,470 | 121,6 | 7,9 |
| ResNet-18 | 11,69 | 1,814 | 25,8 | 14,2 |
| ResNet-50 | 25,56 | 4,089 | 46,3 | 11,3 |
| DenseNet-121 | 7,98 | 2,834 | 49,0 | 17,3 |
| SqueezeNet 1.1 | 1,24 | 0,349 | 11,0 | 31,4 |
| MobileNet v2 | 3,50 | 0,301 | 12,3 | 40,7 |
| MobileNet v3 large | 5,48 | 0,217 | 9,4 | 43,3 |
| ShuffleNet v2 x1.0 | 2,28 | 0,145 | 7,1 | 48,9 |
| EfficientNet-B0 | 5,29 | 0,386 | 16,7 | 43,3 |
| RegNetY-400MF | 4,34 | 0,402 | 10,4 | 25,8 |
| ConvNeXt-T | 28,59 | 4,456 | 58,7 | 13,2 |

> **Biên tập:** Bản gốc dán ảnh một bảng `thop` cho 35 mô hình của torchvision; đã thay bằng bảng trên, đếm lại và thêm các mô hình sau 2019 cùng cột độ trễ. Cột cuối cho thấy điều quan trọng nhất của bài: **FLOPs ít không có nghĩa là nhanh tương ứng**. ShuffleNet v2 có số MACs bằng 1/28 ResNet-50 nhưng chỉ nhanh hơn khoảng 6,5 lần; trên CPU này, mỗi GMAC của ShuffleNet tốn thời gian gấp 3,4 lần của ResNet-18 và 6 lần của VGG-16, vì tích chập depthwise, phép trộn kênh (channel shuffle) và các phép từng phần tử có cường độ tính toán thấp, bị giới hạn bởi băng thông bộ nhớ (mục 4). Trên CPU laptop này, cùng một phép đo có thể chênh tới 30% giữa các lần chạy (xung nhịp turbo thay đổi theo nhiệt độ), nên chỉ nên đọc tỉ lệ.

## 4 Bổ sung: mô hình roofline

Mục này do người biên soạn viết thêm, hoàn thiện ý ở lời nói đầu của bản gốc.

Mô hình roofline (Williams, Waterman, Patterson, 2009) nói rằng hiệu năng đạt được của một tầng (hay một chương trình) có cường độ tính toán $I$ trên nền tảng có sức tính $\pi$ và băng thông $\beta$ bị chặn trên bởi

$$
P \le \min(\pi,\ \beta \cdot I).
$$

- Khi $I < I_{max} = \pi/\beta$: tầng **bị giới hạn bởi bộ nhớ** (memory-bound); hiệu năng tăng tuyến tính theo $I$, phần cứng tính nhanh đến đâu cũng phải chờ dữ liệu.
- Khi $I > I_{max}$: tầng **bị giới hạn bởi tính toán** (compute-bound); hiệu năng chạm trần $\pi$.

Trang Code đo $\pi \approx 400$ GFLOP/s (mức cao nhất đạt được, ở tích chập 3×3; nhân hai ma trận 4096×4096 đạt khoảng 330) và $\beta \approx 75$ GB/s (chép mảng 64 MiB), nên $I_{max} \approx 5$ FLOP/byte, rồi đo từng loại tầng:

<div align=center><img src="../img/code/deploy_code/05_1.png" alt="Mô hình roofline trên CPU"/></div>

Hình 4.1: Mô hình roofline của CPU chạy trang Code và hiệu năng đo được của một số tầng (trang Code).

- Tích chập 3×3 thường và tầng FC với batch 256 có cường độ 80–130 FLOP/byte, nằm ở vùng compute-bound và đạt 90–100% đỉnh. Tích chập 1×1 (25 FLOP/byte) cũng compute-bound nhưng đạt từ 30% tới 80% tuỳ lần chạy.
- Tích chập depthwise 3×3 chỉ có cường độ khoảng 2,3 FLOP/byte, nằm ở vùng memory-bound, và chỉ đạt 25–40% trần của nó. Đây là lý do mạng nhẹ nhanh hơn ít hơn nhiều so với tỉ lệ FLOPs.
- Tầng FC với batch 1 (0,5 FLOP/byte: mỗi trọng số chỉ dùng một lần) và ReLU (0,1 FLOP/byte) vượt "trần" đo được, vì dữ liệu của chúng (8 MB và 3 MB) nằm gọn trong bộ nhớ đệm L2/L3, có băng thông cao hơn RAM. Mô hình roofline đầy đủ vẽ một đường băng thông cho mỗi cấp bộ nhớ.

Với GPU, $I_{max}$ lớn hơn nhiều: A100 có 312 TFLOPS BF16 và 2 TB/s HBM, $I_{max} \approx 150$; H100 SXM khoảng 990 TFLOPS BF16 dày và 3,35 TB/s, $I_{max} \approx 300$. Vì vậy trên GPU, gần như mọi phép từng phần tử, chuẩn hoá, softmax, và cả nhân ma trận–vector khi sinh token của mô hình ngôn ngữ lớn (batch nhỏ), đều memory-bound; hợp nhất kernel (kernel fusion), FlashAttention, lượng tử hoá trọng số và batching là các cách tăng cường độ tính toán hoặc giảm lượng dữ liệu phải di chuyển.

## Tài liệu tham khảo

- Molchanov et al. [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440). ICLR 2017.
- [Tính số tham số của mạng neuron: ví dụ UNet](https://zhuanlan.zhihu.com/p/57437131)
- [How fast is my model?](https://machinethink.net/blog/how-fast-is-my-model/)
- [Giới thiệu MobileNetV1 & MobileNetV2](https://blog.csdn.net/mzpmzk/article/details/82976871)
- [Độ chính xác kép, đơn và nửa](https://blog.csdn.net/sinat_24143931/article/details/78557852)
- [Giải thích Computational Capacity của phần cứng AI](https://zhuanlan.zhihu.com/p/27836831)
- [Roofline Model và phân tích hiệu năng mô hình học sâu](https://zhuanlan.zhihu.com/p/34204282)
- [Các chỉ số đánh giá hiệu năng phát hiện đối tượng](https://zhuanlan.zhihu.com/p/70306015)
- [FP64, FP32, FP16, BFLOAT16, TF32, and other members of the ZOO](https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407)
- Williams, Waterman, Patterson. [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785). CACM 2009.
- Ma et al. [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164). ECCV 2018.
