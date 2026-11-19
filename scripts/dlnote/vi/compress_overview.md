# Tổng quan các phương pháp nén mô hình học sâu

Bài này là cửa vào phần **Nén mô hình**: giới thiệu ngắn bốn nhóm phương pháp chính (chưng cất tri thức, mạng nhẹ, cắt tỉa, lượng tử hóa). Mỗi nhóm có bài riêng đi sâu hơn:

| Phương pháp | Bài chi tiết |
| --- | --- |
| Mạng nhẹ | [Tổng kết mạng nhẹ](dlnote/compress_light_vi) |
| Cắt tỉa | [Thuật toán cắt tỉa](dlnote/compress_prune_vi), [Cắt tỉa bằng PyTorch](dlnote/compress_prune_torch_vi) |
| Lượng tử hóa | [Cơ bản về lượng tử hóa](dlnote/compress_quant_vi) |
| Chưng cất tri thức | [Chưng cất tri thức](dlnote/compress_kd_vi) |

<!-- theory: qa/ch17_vi -->

## 1 Tổng quan về kỹ thuật nén mô hình

Ta biết rằng, ở một mức độ nào đó, mạng càng sâu, càng nhiều tham số thì mô hình càng phức tạp và cho kết quả cuối càng tốt. Thuật toán nén mô hình nhằm biến một mô hình lớn và phức tạp thành một mô hình nhỏ gọn.

Phải nén mô hình là vì thiết bị nhúng có **năng lực tính toán và bộ nhớ hạn chế**; mô hình phải được nén thì mới triển khai được lên thiết bị nhúng.

Bài toán nén mô hình có thể định nghĩa từ `3` góc độ:

1. Lợi ích của nén mô hình:
   - **Tính toán**: giảm số phép tính dấu phẩy động (`FLOPs`), giảm độ trễ (`Latency`).
   - **Lưu trữ**: giảm bộ nhớ sử dụng, tăng mức tận dụng tính toán của `GPU/NPU`.
2. Bài toán nén mô hình viết dưới dạng công thức: $\min_{Policy_i} Model\_Size(Policy_i)$
3. Ràng buộc của bài toán: $accuracy(Policy_i) \ge accuracy\_sla$, tức độ chính xác sau nén không được thấp hơn mức yêu cầu (SLA, service-level agreement).

### 1.1 Phân loại phương pháp nén mô hình

Theo mức độ phá vỡ cấu trúc mạng trong quá trình nén, sách *Phân tích mạng neuron tích chập* (解析卷积神经网络) chia kỹ thuật nén mô hình thành hai phần, "nén phía trước" (front-end) và "nén phía sau" (back-end):

- Nén phía trước: các kỹ thuật nén không thay đổi cấu trúc mạng gốc, chủ yếu gồm `chưng cất tri thức`, mạng nhẹ (thiết kế cấu trúc mô hình gọn) và `cắt tỉa ở mức bộ lọc (filter), tức cắt tỉa có cấu trúc`;
- Nén phía sau: gồm `xấp xỉ hạng thấp`, cắt tỉa không ràng buộc (cắt tỉa không cấu trúc, làm thưa), `lượng tử hóa tham số` và mạng nhị phân. Mục tiêu là giảm kích thước mô hình nhiều nhất có thể, nên cấu trúc mạng gốc bị thay đổi rất nhiều.

Tóm lại: nén phía trước gần như không đổi cấu trúc mạng gốc (chỉ giảm số lớp hoặc số bộ lọc trên mô hình gốc); nén phía sau thay đổi lớn và không đảo ngược được cấu trúc mạng, khiến thư viện học sâu hiện có, thậm chí cả phần cứng, có thể không tương thích với mạng sau khi thay đổi. Chi phí bảo trì rất cao.

### 1.2 Ví dụ về các phương pháp nén mô hình

Các phương pháp nén mô hình chủ đạo trong công nghiệp gồm: chưng cất tri thức (Knowledge Distillation, KD), kiến trúc mô hình nhẹ (còn gọi là thiết kế mô hình gọn), cắt tỉa (Pruning) và lượng tử hóa (Quantization). Tóm tắt từng phương pháp như sau:

| Phương pháp nén | Mô tả | Lớp mạng liên quan | Ví dụ |
| --- | --- | --- | --- |
| Chưng cất tri thức | Một dạng học chuyển giao; ý chính là chuyển "tri thức" từ mô hình giáo viên phức tạp, có năng lực học mạnh sang mô hình học sinh đơn giản. | Tích chập và kết nối đầy đủ | Bài báo KD kinh điển thuộc loại chưng cất "logits": dùng nhãn mềm (soft label) do mạng giáo viên xuất ra làm nhãn để huấn luyện mạng học sinh. Phải huấn luyện lại mô hình. |
| Kiến trúc mô hình nhẹ | Cốt lõi của mạng nhẹ là cải tạo mạng cho nhẹ hơn cả về kích thước lẫn tốc độ, trong khi cố giữ độ chính xác. | Lớp/khối tích chập | MobileNet đề xuất tích chập tách được theo chiều sâu; [bài báo ShuffleNet v2](https://arxiv.org/pdf/1807.11164.pdf) đề xuất **bốn nguyên tắc thực tế để thiết kế mạng hiệu quả**; RepVGG đề xuất ý tưởng tái tham số hóa. Đều phải thiết kế lại `backbone` và huấn luyện lại mô hình. |
| Cắt tỉa | Xoá khỏi mạng mọi kết nối có trọng số nhỏ hơn một ngưỡng. | Tích chập và kết nối đầy đủ | Deep Compression của Song Han (2016) là công trình mở đường; các bước cắt tỉa: huấn luyện bình thường, xoá các kết nối có trọng số dưới ngưỡng, huấn luyện lại. Phải huấn luyện lại mô hình. |
| Lượng tử hóa | Chuyển phép tính số thực dấu phẩy động của mạng neuron thành phép tính dấu phẩy tĩnh. | Tích chập, kết nối đầy đủ, kích hoạt, BN, … | Chiến lược lượng tử hóa INT8 dựa trên divergence KL của TensorRT là kỹ thuật chủ đạo. Lượng tử hóa sau huấn luyện `PTQ` không cần huấn luyện lại mô hình. |

> **Biên tập:** Ý tưởng cắt tỉa có từ trước Song Han rất lâu, ví dụ *Optimal Brain Damage* (LeCun và cộng sự, 1989) đã dùng đạo hàm bậc hai để chọn trọng số cần bỏ. Công trình của Song Han (*Learning both Weights and Connections*, NIPS 2015, rồi *Deep Compression*, ICLR 2016) là công trình mở đường cho **cắt tỉa mạng sâu hiện đại**. Ngoài ra, KL chỉ là một trong các cách hiệu chuẩn của TensorRT; xem [bài lượng tử hóa](dlnote/compress_quant_vi), mục 3.4.

## 2 Chưng cất tri thức

Chưng cất tri thức (knowledge distillation) thực ra cũng là một dạng học chuyển giao (transfer learning). Hiểu nôm na là huấn luyện một mô hình lớn (mô hình `teacher`, giáo viên) và một mô hình nhỏ (mô hình `student`, học sinh), rồi dùng kỹ thuật nào đó chuyển tri thức mà mô hình lớn và phức tạp đã học sang mô hình nhỏ gọn, để mô hình nhỏ đạt hiệu năng gần với mô hình lớn. Cũng có thể nói là cho mô hình nhỏ khớp theo mô hình lớn, để **mô hình nhỏ học được ánh xạ hàm số giống mô hình lớn**. Nhờ vậy mô hình nhỏ vừa giữ được tốc độ tính nhanh, vừa có hiệu năng của mô hình phức tạp, đạt mục đích nén mô hình.

Mấu chốt của chưng cất tri thức là thiết kế đặc trưng dùng để giám sát. Công trình mở đầu của lĩnh vực, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), dùng mức độ giống nhau giữa các lớp chứa trong `Soft Target` (nhãn mềm) làm căn cứ để hướng dẫn mô hình nhỏ học (`chưng cất nhãn mềm, KD`). Các công trình sau còn dùng feature map hoặc attention map ở lớp giữa của mô hình lớn (phương pháp `feature KD`) làm đặc trưng giám sát để hướng dẫn mô hình nhỏ.

Lấy thí nghiệm chưng cất tri thức kinh điển làm ví dụ: ta huấn luyện xong một mạng `teacher`, rồi lấy đầu ra $q$ của mạng `teacher` làm mục tiêu cho mạng `student`, huấn luyện mạng `student` để kết quả $p$ của nó gần với $q$. Do đó hàm mất mát của mạng `student` là $L = CE(y,p)+\alpha\, CE(q,p)$. Ở đây `CE` là entropy chéo (Cross Entropy), $y$ là mã hóa `onehot` của nhãn thật, $q$ là đầu ra của mạng `teacher`, $p$ là đầu ra của mạng `student`.

Tuy nhiên, dùng thẳng đầu ra softmax $q$ của mạng `teacher` có thể không hợp lắm. Vì sau khi một mạng đã được huấn luyện tốt, nó sẽ rất tự tin với đáp án đúng và rất ít tự tin với đáp án sai. Ví dụ với dữ liệu MNIST, với một ảnh số 2, xác suất dự đoán là 2 sẽ rất cao, còn với các số giống 2, như 3 và 7, xác suất chỉ là $10^{-6}$ và $10^{-9}$. Như vậy, **thông tin về sự giống nhau giữa dữ liệu** mà mạng `teacher` học được (ví dụ số 2 khá giống 3 và 7) rất khó truyền sang mạng `student`, vì các xác suất này gần bằng `0`. Vì thế bài báo đề xuất `softmax-T` (công thức tính nhãn mềm) như sau:

$$q_{i} = \frac{\exp(z_{i}/T)}{\sum_{j}\exp(z_{j}/T)}$$

Ở đây $q_i$ là mục tiêu mà mạng $student$ học theo (soft targets), $z_i$ là đầu ra `logit` của lớp ngay trước `softmax` trong mô hình `teacher`. Nếu lấy $T = 1$, công thức trên **chính là softmax**, cho xác suất mỗi lớp từ logit. Nếu $T$ tiến về 0, giá trị lớn nhất sẽ tiến về 1, các giá trị khác tiến về 0, gần với mã hóa `onehot`.

> **Biên tập:** Bản gốc viết công thức là $q_i = \frac{z_i/T}{\sum_j z_j/T}$, thiếu hàm mũ (khi đó $T$ triệt tiêu và công thức không còn là softmax). Đã sửa thành $\exp(z_i/T)/\sum_j \exp(z_j/T)$ như bài báo của Hinton.

Như vậy hàm mất mát cuối cùng của mô hình `student` gồm hai phần:

+ Phần thứ nhất là entropy chéo (cross entropy) giữa dự đoán của mô hình nhỏ và "nhãn mềm" của mô hình lớn;
+ Phần thứ hai là entropy chéo giữa dự đoán và nhãn lớp thông thường.

Tầm quan trọng của hai hàm mất mát này điều chỉnh được bằng trọng số. Trong thực tế, giá trị `T` ảnh hưởng tới kết quả cuối. Nói chung, `T` lớn hơn cho độ chính xác cao hơn; T (nhiệt độ chưng cất) là một siêu tham số khi huấn luyện mô hình chưng cất. **T là siêu tham số điều chỉnh được; T càng lớn thì phân bố xác suất càng "mềm" (theo mô tả của bài báo), đường cong càng mượt.** Điều này giống như thêm nhiễu trong quá trình học chuyển giao, giúp mạng học sinh học hiệu quả hơn và tổng quát hóa tốt hơn; thực chất đây là một chiến lược chống quá khớp.

> **Biên tập:** Câu "T lớn hơn cho độ chính xác cao hơn" không đúng trong mọi trường hợp. Chính bài báo của Hinton ghi rằng khi mạng học sinh rất nhỏ (30 đơn vị mỗi lớp ẩn), nhiệt độ trong khoảng 2,5–4 tốt hơn hẳn nhiệt độ cao hơn hoặc thấp hơn. Trên trang [Code của bài chưng cất](dlnote/compress_kd_code_vi), với một học sinh 9 nghìn tham số, $T=1$ cho kết quả tốt nhất, còn $T=4$ và $T=10$ không hơn huấn luyện từ đầu một cách rõ rệt. Ngoài ra, khi dùng $T>1$, phần mất mát mềm thường được nhân thêm $T^2$ để độ lớn gradient không đổi khi thay $T$; chi tiết ở bài [Chưng cất tri thức](dlnote/compress_kd_vi).

Hình dưới là khung tổng thể của thuật toán chưng cất tri thức.

![knowledge_distillation](../images/model_compression/knowledge_distillation.png)

Hình 1: Khung chưng cất tri thức. Giáo viên và học sinh nhận cùng đầu vào $x$. Đầu ra mềm của hai mạng (softmax với cùng nhiệt độ $T = t$) tạo thành mất mát chưng cất (distillation loss); đầu ra thường của học sinh ($T = 1$) so với nhãn thật $y$ tạo thành mất mát học sinh (student loss). Nguồn ảnh: [Intel Distiller](https://intellabs.github.io/distiller/knowledge_distillation.html).

## 3 Kiến trúc mô hình nhẹ

**Cốt lõi của mạng nhẹ là cải tạo mạng cho nhẹ hơn cả về kích thước lẫn tốc độ, trong khi cố giữ độ chính xác.** Về cách tự tay thiết kế mạng nhẹ, hiện chưa có quy tắc chung được dùng rộng rãi, mà chỉ có một số tư tưởng chỉ đạo và các tổng kết thiết kế cho từng nền tảng chip (từng kiến trúc chip). Nên rút ra tư tưởng và gợi ý từ các bài báo kinh điển, rồi tự triển khai và đo hiệu năng mô hình trên từng nền tảng phần cứng.

Với mạng neuron tích chập, các công trình tiêu biểu là dòng `MobileNet`, dòng `ShuffleNet`, `RepVGG`, `CSPNet`, `VoVNet`, v.v.

> **Biên tập:** Hai mục con của bản gốc, "3.1 Thiết kế kiến trúc CNN hiệu quả" và "3.2 Tổng kết triển khai mô hình nhẹ", trùng gần như từng chữ với bài [Tổng kết mạng nhẹ](dlnote/compress_light_vi). Để không lặp lại, nội dung này chỉ giữ ở bài đó: các kết luận (FLOPs thấp không có nghĩa là độ trễ thấp, tích chập depthwise kém hiệu quả trên GPU/NPU, …), các gợi ý thiết kế (số kênh là bội của 16, bốn nguyên tắc của ShuffleNet v2, …) và bảng đo MobileNetV2/ResNet50 trên các NPU.

## 4 Cắt tỉa mô hình

> Cắt tỉa mô hình (model pruning) còn gọi là làm thưa mô hình (model sparsity).

Mô hình học sâu thường có rất nhiều tham số dư thừa. Loại bỏ (tức đặt bằng `0`) những trọng số tương đối "không quan trọng" trong ma trận trọng số giúp giảm tài nguyên tính toán và tăng tính thời gian thực; kỹ thuật tương ứng gọi là cắt tỉa mô hình.

![Minh hoạ cắt tỉa mô hình điển hình](../images/model_compression/model_pruning_pipeline.png)

Hình 2: Trái: quy trình huấn luyện ba bước (huấn luyện kết nối → cắt tỉa kết nối → huấn luyện trọng số, lặp lại). Phải: khớp nối (synapse) và neuron trước và sau khi cắt tỉa. Nguồn: [Han et al., Learning both Weights and Connections for Efficient Neural Networks, NIPS 2015](https://arxiv.org/pdf/1506.02626.pdf).

Hình trên là quy trình (`pipeline`) cắt tỉa ba bước điển hình, gồm 3 bước chính:

1. Huấn luyện mô hình bình thường;
2. Cắt tỉa mô hình;
3. Huấn luyện lại mô hình.

Ba bước trên được lặp lại cho tới khi độ chính xác của mô hình đạt mục tiêu thì dừng.

Theo **độ mịn** (granularity), thuật toán cắt tỉa có thể chia thô thành **cắt tỉa mịn** và **cắt tỉa thô**, như sau:

1. **Cắt tỉa mịn (fine-grained)**: cắt tỉa kết nối hoặc neuron; đây là mức cắt tỉa nhỏ nhất.
2. **Cắt tỉa vector (vector-level)**: thô hơn cắt tỉa mịn, là cắt tỉa bên trong một kernel tích chập (intra-kernel).
3. **Cắt tỉa kernel (kernel-level)**: bỏ một kernel tích chập, tức bỏ phản hồi ứng với một kênh đầu vào.
4. **Cắt tỉa bộ lọc (filter-level)**: còn gọi là cắt tỉa kênh (`Channel Pruning`), cắt tỉa cả một nhóm kernel (một bộ lọc), làm số kênh đặc trưng đầu ra thay đổi khi suy luận. Cắt tỉa bộ lọc là hướng được nghiên cứu nhiều nhất hiện nay.

Theo cách cắt có đều đặn hay không, thuật toán cắt tỉa cũng có thể chia thành:

- **Cắt tỉa không cấu trúc**, chính là loại 1 ở trên; không thân thiện với phần cứng, cần bộ tăng tốc phần cứng chuyên dụng. Công trình tiêu biểu là bài báo của Song Han năm `2016`.
- **Cắt tỉa có cấu trúc**, như loại 2, 3, 4; thân thiện với phần cứng. Công trình tiêu biểu như [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/pdf/1708.06519.pdf).

### 4.1 So sánh thưa có cấu trúc và cắt tỉa không cấu trúc

> **Biên tập:** Mục này của bản gốc trùng từng chữ với mục 3.2 của bài [Thuật toán cắt tỉa](dlnote/compress_prune_vi); xem ở đó. Tóm tắt một câu: thưa không cấu trúc giữ độ chính xác và tỉ lệ nén tốt hơn nhưng khó tăng tốc trên phần cứng thông thường; thưa có cấu trúc thì ngược lại. Trang [Code cắt tỉa](dlnote/compress_prune_torch_code_vi) đo cả hai.

## 5 Lượng tử hóa mô hình

So với cắt tỉa, lượng tử hóa tham số là một kỹ thuật nén phía sau hay được dùng. Cái gọi là **lượng tử hóa** thực ra có thể xem là tương đương với khái niệm tính toán **độ chính xác thấp** (Low precision). Mô hình độ chính xác thông thường lưu trọng số bằng FP32 (số thực 32 bit, độ chính xác đơn), còn độ chính xác thấp là dùng các định dạng trọng số `INT8`, `FP16`, v.v.

Quá trình lượng tử hóa mô hình (`Model Quantization`, còn gọi là lượng tử hóa mạng) gồm hai phần: chuyển **tham số độ chính xác đơn** của mô hình (thường là tham số **dấu phẩy động** `FP32` 32 bit) thành **tham số độ chính xác thấp** (thường là tham số **dấu phẩy tĩnh** `INT8` 8 bit), và chuyển các phép tính dấu phẩy động khi suy luận thành phép tính dấu phẩy tĩnh. Phần sau cần framework suy luận hỗ trợ.

Lượng tử hóa mô hình giúp giảm dung lượng lưu trữ, bộ nhớ sử dụng và tài nguyên tính toán, từ đó tăng tốc suy luận, và cũng để hợp hơn với bộ tăng tốc NPU trên thiết bị di động/đầu cuối. Tóm lại: mô hình nhỏ hơn, nhanh hơn, dùng được ở nhiều tình huống hơn.

Cuối cùng, hướng chủ đạo trong công nghiệp hiện nay là huấn luyện bằng mô hình tham số độ chính xác cao FP32, suy luận bằng mô hình tham số độ chính xác thấp INT8: chuyển mô hình từ FP32 sang INT8 (quá trình số học lượng tử hóa), rồi suy luận bằng INT8.

> **Biên tập:** Hai mục con của bản gốc, "5.1 Các phương án lượng tử hóa" và "5.2 Phân loại lượng tử hóa", trùng từng chữ với mục 1.2 và 1.3 của bài [Cơ bản về lượng tử hóa](dlnote/compress_quant_vi); xem ở đó. Bài đó cũng có mục bổ sung về lượng tử hóa từ 2023 (FP8, INT4, lượng tử hóa LLM), vì câu "công nghiệp dùng INT8" ở trên nay chỉ còn đúng với CNN trên thiết bị đầu cuối.

## 6 Tổng kết các phương pháp nén

1. Theo cách cắt có đều đặn hay không, thuật toán cắt tỉa chia thành **cắt tỉa không cấu trúc** và **cắt tỉa có cấu trúc**. Loại trước giữ được tỉ lệ nén và độ chính xác cao nhưng khó tăng tốc; loại sau dễ tăng tốc bằng phần cứng hơn nhưng phải hy sinh tỉ lệ nén hoặc độ chính xác.
2. Nếu cần huấn luyện đầu cuối một lần để có ngay mô hình đã nén và tăng tốc, có thể dùng mô hình dựa trên thiết kế mạng nhẹ.
3. Có `4` yếu tố chính ảnh hưởng tới tốc độ suy luận của mạng neuron: lượng tính toán `FLOPs`, chi phí truy cập bộ nhớ `MAC`, mức song song của phép tính, và kiến trúc cùng đặc tính của nền tảng phần cứng (năng lực tính toán, băng thông bộ nhớ GPU).

## 7 Bổ sung: kết hợp các phương pháp nén

Mục này do người biên soạn viết thêm. Bốn nhóm phương pháp trên không loại trừ nhau, và trong thực tế thường được **xếp chồng**:

- *Deep Compression* (Han và cộng sự, 2016) đã nối ba bước: cắt tỉa → lượng tử hóa có huấn luyện (chia sẻ trọng số) → mã hóa Huffman, nén AlexNet 35 lần và VGG-16 49 lần mà không mất độ chính xác.
- Bài lượng tử hóa (mục 3.5) nhắc rằng chưng cất giúp mô hình lượng tử hóa lấy lại độ chính xác; cách làm phổ biến là dùng chính mô hình FP32 làm giáo viên khi huấn luyện QAT.
- Với mô hình ngôn ngữ lớn, quy trình "cắt tỉa có cấu trúc rồi chưng cất để phục hồi" đã thành chuẩn: ví dụ Minitron (NVIDIA, 2024) cắt bớt chiều rộng và chiều sâu của Nemotron 15B rồi chưng cất từ mô hình gốc, được các mô hình 8B và 4B với lượng token huấn luyện ít hơn tới 40 lần so với huấn luyện từ đầu. Sau đó mô hình thường được lượng tử hóa thêm xuống 4–8 bit để triển khai.
- Thứ tự thường gặp khi triển khai một mô hình đã có sẵn: chọn kiến trúc nhẹ (nếu còn được chọn) → cắt tỉa (nếu phần cứng tận dụng được) → chưng cất để phục hồi → lượng tử hóa sau cùng, vì lượng tử hóa gần như "miễn phí" và được hầu hết phần cứng hỗ trợ.

## Tài liệu tham khảo

1. [Nén và tăng tốc mô hình học sâu](https://www.cnblogs.com/LXP-Never/p/14833772.html) (深度学习模型压缩与加速)
2. [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/pdf/1510.00149.pdf)
3. *Phân tích mạng neuron tích chập* (《解析卷积神经网络》), Wei Xiu-Shen.
4. [4-model_compression](https://github.com/HarleysZhang/deep_learning_system/tree/main/4-model_compression), kho ghi chú trước đây của tác giả.

Tài liệu tham khảo chung của phần nén mô hình (từ `README.md` của thư mục gốc):

- [AI-System](https://github.com/microsoft/AI-System/tree/main/Textbook): giáo trình *Hệ thống trí tuệ nhân tạo* của Microsoft.
- [ncnn](https://github.com/Tencent/ncnn): framework suy luận cho thiết bị di động của Tencent.
- [Neural Network Intelligence (NNI)](https://nni.readthedocs.io/en/stable/compression/overview.html): bộ công cụ nén mô hình của Microsoft.

Tài liệu cho mục 7 (người biên soạn bổ sung):

- LeCun et al., [Optimal Brain Damage](https://proceedings.neurips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf), NIPS 1989.
- Muralidharan et al., [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679) (Minitron), 2024.
