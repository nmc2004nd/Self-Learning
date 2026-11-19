# Triển khai mô hình: hợp nhất tầng tích chập và tầng BN

> Bản gốc của bài này chỉ có một câu ở lời nói đầu và bốn tài liệu tham khảo. Câu đó được dịch nguyên văn; phần còn lại do người biên soạn viết, dựa trên các tài liệu bản gốc đã chọn. Trang [Code](dlnote/deploy_code_vi) đi kèm (mục 2) cài công thức, kiểm tra với PyTorch và đo độ trễ trên CPU.

<!-- theory: qa/ch17_vi#17.5.2 -->

## Lời nói đầu

**Hai thách thức chính của suy luận mô hình là độ trễ và chi phí.**

*Từ đây trở xuống do người biên soạn viết.*

Mạng CNN hiện đại gần như luôn đặt tầng Batch Normalization (BN) ngay sau tầng tích chập (bộ `cbr` ở [bài siêu tham số](dlnote/alch_hyper_vi), mục 1.2). Khi huấn luyện, BN dùng thống kê của từng batch; nhưng khi suy luận, nó chỉ là một phép biến đổi affine cố định trên từng kênh. Một phép affine đi sau một phép tuyến tính (tích chập) có thể gộp vào phép tuyến tính đó, nên tầng BN có thể biến mất hoàn toàn khỏi mô hình suy luận: ít một lần đọc, ghi cả feature map, ít một lần gọi kernel, và tham số BN không còn. Đây là phép tối ưu đồ thị phổ biến nhất, có mặt trong mọi engine suy luận (TensorRT, ONNX Runtime, OpenVINO, ncnn, MNN…).

## 1 BN lúc suy luận là một phép affine

Với kênh $c$, BN lúc suy luận tính

$$
\mathrm{BN}(x_c) = \gamma_c \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c,
$$

trong đó $\mu_c, \sigma_c^2$ là trung bình và phương sai *chạy* (running mean/var) tích luỹ trong lúc huấn luyện, $\gamma_c, \beta_c$ là tham số học, và $\epsilon$ là hằng số nhỏ (PyTorch mặc định $10^{-5}$). Mọi đại lượng đều cố định, nên đặt

$$
s_c = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}, \qquad t_c = \beta_c - s_c\,\mu_c,
$$

thì $\mathrm{BN}(x_c) = s_c\,x_c + t_c$: nhân một hệ số và cộng một hằng số theo từng kênh. Chi tiết về BN xem [bài Batch Normalization](dlnote/dl_bn_vi).

## 2 Gộp vào tầng tích chập

Tầng tích chập tính đầu ra kênh $c$ là $y_c = W_c * x + b_c$, với $W_c$ là kernel thứ $c$ (kích thước $C_{in}\times K\times K$) và $b_c$ là bias (bằng 0 nếu tầng không có bias, như thường gặp khi theo sau là BN). Thay vào:

$$
\mathrm{BN}(y_c) = s_c (W_c * x + b_c) + t_c = (s_c W_c) * x + \big(s_c (b_c - \mu_c) + \beta_c\big).
$$

Vậy tầng tích chập sau khi hợp nhất có

$$
W'_c = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}\, W_c, \tag{2.3}
$$

$$
b'_c = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}\,(b_c - \mu_c) + \beta_c. \tag{2.4}
$$

Viết dưới dạng ma trận như tài liệu 1: nếu trải mỗi vùng đầu vào thành vector $\mathbf{f}$, tầng tích chập là $W_{conv}\mathbf f + b_{conv}$, tầng BN là $W_{bn}\mathbf z + b_{bn}$ với $W_{bn} = \mathrm{diag}(s)$, nên hợp nhất là $W' = W_{bn}W_{conv}$, $b' = W_{bn} b_{conv} + b_{bn}$.

Cài đặt trong PyTorch:

```python
def fuse_conv_bn(conv, bn):
    """W' = γ/√(σ²+ε) · W ;  b' = γ/√(σ²+ε) · (b − μ) + β"""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True)
    b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        fused.bias.copy_((b - bn.running_mean) * scale + bn.bias)
    return fused
```

PyTorch có sẵn hàm tương đương `torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)`, và `torch.ao.quantization.fuse_modules(model, [["conv1", "bn1", "relu"]])` để gộp theo tên mô-đun. Trang Code kiểm tra: sai khác lớn nhất giữa `bn(conv(x))` và tầng đã hợp nhất là $3\times10^{-6}$, đúng bằng sai số làm tròn của fp32, và hàm tự cài cho kết quả giống hệt hàm của thư viện.

## 3 Điều kiện và giới hạn

- **Chỉ đúng ở chế độ suy luận.** Khi huấn luyện, BN dùng trung bình và phương sai của batch hiện tại, không phải hằng số, nên không gộp được. Trang Code cho thấy nếu để BN ở chế độ `train()` thì kết quả lệch hàng đơn vị so với tầng đã hợp nhất. Luôn gọi `model.eval()` trước khi hợp nhất hay xuất mô hình.
- **BN phải theo ngay sau tích chập** (hoặc tầng tuyến tính), không có phép phi tuyến ở giữa. Với thứ tự tích chập → ReLU → BN (ít gặp), không gộp BN vào tích chập trước được, nhưng có thể gộp phép affine vào *tầng tích chập phía sau*, cẩn thận với padding (phần đệm 0 của tầng sau không còn là 0 sau phép affine).
- **Tích chập nhóm, depthwise** gộp được như thường: hệ số $s_c$ nhân vào kernel của kênh ra $c$, không phụ thuộc số nhóm.
- **Lượng tử hoá**: nên hợp nhất *trước* khi lượng tử hoá (cả PTQ lẫn QAT), vì nhân $s_c$ làm đổi phân phối trọng số theo từng kênh; đây là một lý do lượng tử hoá theo kênh (per-channel) cho trọng số tốt hơn theo tầng. Xem [bài lượng tử hoá](dlnote/compress_quant_vi).
- **Số tham số** gần như không đổi: mất $2C$ tham số BN và $2C$ buffer, thêm $C$ bias. ResNet-18 giảm từ 11 689 512 xuống 11 684 712 tham số.

## 4 Lợi ích thực tế

Tầng BN không có phép nhân ma trận nào, chỉ có $2$ phép tính cho mỗi phần tử, nên FLOPs gần như không đổi. Lợi ích nằm ở **lượng truy cập bộ nhớ**: BN là phép từng phần tử, phải đọc và ghi lại toàn bộ feature map, cường độ tính toán rất thấp (xem mô hình roofline ở [bài độ phức tạp](dlnote/deploy_flops_vi), mục 4). Bỏ được nó là bỏ được một lần đọc-ghi feature map và một lần gọi kernel cho mỗi tầng tích chập.

Trang Code hợp nhất cả 20 tầng BN của ResNet-18 (sai khác logit lớn nhất $4\times10^{-6}$) và đo trên CPU laptop, PyTorch eager, đo xen kẽ hai mô hình. Qua sáu lần chạy, mô hình đã hợp nhất nhanh hơn 0–4% ở batch 1 và 5–14% ở batch 8 (lần chạy cuối: 18,6 → 18,5 ms và 171 → 161 ms). Lợi ích có thật nhưng nhỏ, cùng cỡ nhiễu đo trên máy này, và khớp với con số "khoảng 5%" của tài liệu 2. Trong engine suy luận như TensorRT hay ONNX Runtime, BN thường đã được tự hợp nhất, và lợi ích lớn hơn đến từ việc gộp tiếp ReLU (và cả phép cộng residual) vào cùng kernel tích chập: tầng `Convolution` trong tệp ncnn ở trang Code có tham số `9=1`, tức hàm kích hoạt ReLU đã nằm sẵn trong tầng tích chập.

## 5 Từ hợp nhất BN tới tái tham số hoá cấu trúc

Ý tưởng "huấn luyện một cấu trúc, suy luận bằng một cấu trúc tương đương đơn giản hơn" được đẩy xa trong **RepVGG** (Ding và cộng sự, CVPR 2021): khi huấn luyện, mỗi khối có ba nhánh song song (tích chập 3×3 + BN, tích chập 1×1 + BN, và nhánh đồng nhất + BN); khi suy luận, mỗi nhánh được hợp nhất BN như trên, kernel 1×1 và nhánh đồng nhất được đệm thành kernel 3×3, rồi cộng ba kernel lại thành **một tầng tích chập 3×3 duy nhất**. Mạng suy luận chỉ còn chồng tích chập 3×3 + ReLU kiểu VGG, rất hợp với GPU và NPU. Kỹ thuật này (structural re-parameterization) được dùng trong MobileOne, YOLOv6, YOLOv7 và nhiều mạng nhẹ khác. Xem bài RepVGG ở phần 3 của khoá.

## Tài liệu tham khảo

1. [Speeding up model with fusing batch normalization and convolution](https://learnml.today/speeding-up-model-with-fusing-batch-normalization-and-convolution-3)
2. [Hợp nhất BN khi suy luận, dễ dàng tăng tốc khoảng 5%](https://mp.weixin.qq.com/s/PXsyy-WBqxA5FTMFseC9Rw) (深度学习推理时融合BN，轻松获得约5%的提速)
3. [Bản dịch ghi chú CS231n: mạng neuron tích chập](https://zhuanlan.zhihu.com/p/22038289)
4. [Mẹo tăng tốc suy luận: hợp nhất BN và Conv](https://zhuanlan.zhihu.com/p/110552861)
5. Ioffe, Szegedy. [Batch Normalization](https://arxiv.org/abs/1502.03167). ICML 2015.
6. Ding et al. [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697). CVPR 2021.
7. [PyTorch: torch.nn.utils.fusion](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.fusion.fuse_conv_bn_eval.html)
