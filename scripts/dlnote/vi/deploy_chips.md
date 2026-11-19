# Triển khai mô hình: lướt qua các chip AI

> Bản gốc của bài này chỉ có mục về TPU của Google (tới TPU v4) và một bảng thông số. Phần đó được dịch đầy đủ, có sửa lỗi; mục 2 và 3 do người biên soạn viết thêm, cập nhật tới năm 2026. Trang [Code](dlnote/deploy_code_vi) đi kèm (mục 6) tính lại các con số đỉnh của TPU v4 từ kích thước mảng MXU và xung nhịp.

<!-- theory: qa/ch15_vi#15.8.5 -->

## 1 TPU

Bộ xử lý tensor (Tensor Processing Unit, TPU) là **bộ tăng tốc học máy** do Google thiết kế, hỗ trợ các framework [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://cloud.google.com/tpu/docs/tutorials/pytorch-pod) (qua PyTorch/XLA) và [JAX](https://jax.readthedocs.io/en/latest/).

Mỗi `TensorCore` trên chip TPU gồm một hoặc nhiều đơn vị nhân ma trận (`MXU`), một đơn vị vector và một đơn vị vô hướng.

`MXU` gồm $128 \times 128$ bộ nhân-cộng xếp thành [mảng tâm thu](https://en.wikipedia.org/wiki/Systolic_array) (systolic array). MXU cung cấp phần lớn sức tính của TensorCore. Mỗi MXU làm được 16K ($128\times128 = 16384$) phép nhân-cộng mỗi chu kỳ. Mọi phép nhân nhận đầu vào [bfloat16](https://cloud.google.com/tpu/docs/bfloat16), nhưng mọi phép cộng dồn đều làm ở định dạng FP32.

> Trong mảng tâm thu, trọng số được nạp sẵn vào lưới $128\times128$ ô tính; dữ liệu đi vào từ một cạnh và chảy qua lưới, mỗi ô nhân giá trị đi qua với trọng số của nó, cộng vào tổng từng phần rồi chuyển sang ô kế bên. Mỗi giá trị đọc từ bộ nhớ được dùng lại 128 lần mà không phải đọc lại, nên cường độ tính toán rất cao (xem mô hình roofline ở [bài độ phức tạp](dlnote/deploy_flops_vi), mục 4).

Đơn vị vector dùng cho các phép tính thông thường như hàm kích hoạt và `softmax`. Đơn vị vô hướng dùng cho điều khiển luồng, tính địa chỉ bộ nhớ và các thao tác quản lý khác.

### 1.1 TPU v4

TPU v2 ra mắt năm 2018 (trên Google Cloud), TPU v3 năm 2019, TPU v4 năm 2020 (dùng nội bộ; có trên Google Cloud từ 2022). Khi bản gốc được viết, TPU v4 là thế hệ bộ tăng tốc học máy **mới nhất** của Google.

Mỗi chip TPU v4 có hai TensorCore. Mỗi TensorCore có bốn MXU, một đơn vị vector và một đơn vị vô hướng. Bảng dưới là các thông số chính của một pod TPU v4.

| **Thông số chính** | **Giá trị của pod v4** |
| :--- | :--- |
| Đỉnh tính toán mỗi chip | 275 TFLOPS (bf16 hoặc int8) |
| Dung lượng và băng thông HBM2 | 32 GiB, 1200 GB/s |
| Công suất đo được (nhỏ nhất / trung bình / lớn nhất) | 90 / 170 / 192 W |
| Kích thước pod | 4096 chip |
| Cấu trúc liên kết (topology) | Hình xuyến 3 chiều (3D torus) |
| Đỉnh tính toán mỗi pod | 1,1 exaFLOPS (bf16 hoặc int8) |
| Băng thông all-reduce mỗi pod | 1,1 PB/s |
| Băng thông chia đôi (bisection) mỗi pod | 24 TB/s |

> **Biên tập:** Bản gốc (dịch máy từ tài liệu Google Cloud tiếng Trung) có hai lỗi trong bảng: kích thước pod ghi "4096 条状标签" ("4096 nhãn dạng thanh", dịch nhầm chữ *chips*), và đỉnh tính toán mỗi pod ghi "1,1 万亿次" ($1{,}1\times10^{12}$, tức 1,1 TFLOPS, còn nhỏ hơn một chip). Đúng là **1,1 exaFLOPS** ($1{,}1\times10^{18}$): $4096 \times 275$ TFLOPS $\approx 1{,}13$ EFLOPS. Trang Code cũng tính lại đỉnh mỗi chip: $2$ TensorCore $\times\ 4$ MXU $\times\ 16384$ MAC $\times\ 2$ FLOP $\times\ 1{,}05$ GHz $= 275{,}3$ TFLOPS, khớp bảng. Hàng "băng thông toàn phần" của bản gốc là băng thông all-reduce. Bản gốc cũng nhận xét "275 nghìn tỉ phép tính mỗi giây, sức tính này hơi khoa trương"; để so sánh, GPU A100 cùng thời có 312 TFLOPS BF16.

![Bố cục TPU v4](../images/ai_chips/tpu-v4-layout.png)

Hình 1.1: Bố cục chip TPU v4: hai TensorCore (mỗi cái có đơn vị vô hướng, đơn vị vector và bốn MXU) dùng chung bộ nhớ băng thông cao (HBM), được trình bày như một lõi ảo (Virtual Core).

Cường độ tính toán cần để dùng hết sức tính của một chip TPU v4 là $275\text{ TFLOPS} / 1200\text{ GB/s} \approx 229$ FLOP/byte: chỉ các phép nhân ma trận lớn mới đạt được, mọi phép khác đều bị giới hạn bởi băng thông HBM.

### 1.2 Định dạng số thực IEEE và Brain float

![Định dạng số thực IEEE và Brain float](../images/ai_chips/ieee_fp_bfp.png)

Hình 1.2: Bố cục bit của IEEE fp32, IEEE fp16 và bf16; mọi định dạng đều có một bit định trị ẩn (bằng 1) ở số chuẩn (từ bài báo của Jouppi và cộng sự, tài liệu 2).

bf16 giữ nguyên 8 bit mũ của fp32 nên có cùng dải động, đổi lại chỉ có 7 bit định trị. Chi tiết và ví dụ số xem [bài độ phức tạp](dlnote/deploy_flops_vi), mục 3.2–3.3.

## 2 Bổ sung: các thế hệ TPU sau v4

Mục này do người biên soạn viết thêm.

| Thế hệ | Năm | Đỉnh mỗi chip (BF16) | HBM mỗi chip | Ghi chú |
| --- | --- | --- | --- | --- |
| TPU v4 | 2020 (cloud 2022) | 275 TFLOPS | 32 GB, 1,2 TB/s | Pod 4096 chip, dùng chuyển mạch quang (OCS) để đổi topo |
| TPU v5e | 2023 | 197 TFLOPS | 16 GB, 0,8 TB/s | Bản tiết kiệm chi phí, cho suy luận và huấn luyện vừa |
| TPU v5p | 2023 | 459 TFLOPS | 95 GB, 2,8 TB/s | Pod 8960 chip |
| Trillium (v6e) | 2024 | 918 TFLOPS | 32 GB, 1,6 TB/s | MXU 256×256 |
| Ironwood (v7) | 2025 | khoảng 4,6 PFLOPS FP8 | 192 GB, 7,4 TB/s | Thiết kế cho suy luận, pod 9216 chip |

> Số liệu theo công bố của Google; các thế hệ không cùng đo một định dạng (Ironwood công bố ở FP8), nên chỉ so được xu hướng.

## 3 Bổ sung: bức tranh chip AI năm 2026

Mục này do người biên soạn viết thêm.

**GPU.** NVIDIA vẫn chiếm phần lớn thị trường huấn luyện và suy luận trên đám mây: A100 (2020, Ampere) → H100/H200 (2022–2024, Hopper, thêm FP8 và Transformer Engine) → B200/GB200 (2024–2025, Blackwell, thêm FP4) → dòng Rubin (2026). Lợi thế lớn nhất nằm ở hệ sinh thái phần mềm (CUDA, cuDNN, TensorRT, NCCL) và kết nối NVLink giữa các GPU. AMD có dòng Instinct MI300X/MI325X/MI355X với bộ nhớ HBM lớn và phần mềm ROCm.

**ASIC trên đám mây.** Ngoài TPU của Google: AWS Trainium (huấn luyện) và Inferentia (suy luận), Microsoft Maia, Meta MTIA; các công ty khởi nghiệp như Cerebras (chip cỡ cả tấm wafer), Groq (LPU, suy luận độ trễ thấp bằng SRAM trên chip), SambaNova. Ở Trung Quốc: Huawei Ascend (dòng 910), Cambricon, Hygon và các bộ tăng tốc của Alibaba, Baidu (Kunlun).

**NPU trong thiết bị.** Hầu hết SoC điện thoại và máy tính cá nhân nay có NPU riêng: Apple Neural Engine, Qualcomm Hexagon, MediaTek APU, Google Tensor, cùng NPU trong CPU Intel Core Ultra và AMD Ryzen AI (chuẩn "AI PC" của Microsoft yêu cầu khoảng 40 TOPS INT8). Các chip nhúng cho camera và xe như HiSilicon, Ambarella, Horizon Robotics, Rockchip mà [bài tổng quan](dlnote/deploy_overview_vi) nhắc tới vẫn là các NPU INT8 chuyên cho CNN, nay thêm hỗ trợ Transformer.

**Xu hướng chung.**

- **Độ chính xác thấp hơn**: từ FP32 → FP16/BF16 → INT8 → FP8 → FP4 (các định dạng MX theo khối của OCP: MXFP8, MXFP6, MXFP4, cùng NVFP4 của NVIDIA). Mỗi lần giảm một nửa số bit, sức tính trên cùng diện tích chip gần gấp đôi.
- **Bộ nhớ là nút thắt**: dung lượng và băng thông HBM tăng chậm hơn sức tính, nên $I_{max}$ của chip ngày càng lớn (vài trăm FLOP/byte) và suy luận mô hình ngôn ngữ lớn bị giới hạn bởi bộ nhớ; nhiều chip mới tăng HBM nhiều hơn tăng FLOPS.
- **Kết nối quy mô lớn**: huấn luyện mô hình lớn chạy trên hàng chục nghìn chip, nên băng thông giữa các chip (NVLink, ICI của TPU, chuyển mạch quang) quan trọng ngang sức tính của từng chip.
- **Phần mềm quyết định**: chip chỉ dùng được khi có trình biên dịch và runtime tốt; các tầng trung gian như XLA, Triton (ngôn ngữ kernel của OpenAI), MLIR, và ONNX giúp đưa cùng một mô hình lên nhiều loại phần cứng.

## Tài liệu tham khảo

1. [Google Cloud: TPU system architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)
2. Jouppi et al. [A Domain-Specific Supercomputer for Training Deep Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3360307). Communications of the ACM, 2020.
3. Jouppi et al. [TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings](https://arxiv.org/abs/2304.01433). ISCA 2023.
4. [Google Cloud: TPU versions](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#versions)
5. [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
