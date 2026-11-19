# Nén mô hình: Chưng cất tri thức

> Ghi chú của người biên soạn: tệp tương ứng trong *dl_note* của **Harley Zhang** ([kho mã](https://github.com/harleyszhang/dl_note)) để trống, nên toàn bộ bài này do người biên soạn viết, dựa trên mục chưng cất tri thức của bài tổng quan và các bài báo gốc ghi ở cuối trang.

Bài này mở rộng mục 2 của [bài tổng quan](dlnote/compress_overview_vi): đi từ chưng cất theo logit của Hinton, qua chưng cất theo đặc trưng, đến cách chưng cất mô hình ngôn ngữ lớn hiện nay. Trang [Code](dlnote/compress_kd_code_vi) đi kèm cài đặt hàm mất mát chưng cất, kiểm tra hệ số $T^2$ bằng số, và chưng cất một CNN 1,7 triệu tham số xuống một học sinh 9 nghìn tham số trên FashionMNIST.

<div class="qa-theory">

**Lý thuyết:** [Ôn tập · Chương 17 Nén, tăng tốc mô hình và triển khai trên thiết bị di động](qa/ch17_vi)

</div>

## 1 Ý tưởng: "tri thức tối" trong nhãn mềm

Một mạng phân loại đã huấn luyện tốt cho ra phân bố xác suất trên mọi lớp, không chỉ lớp đúng. Với một ảnh "áo sơ mi" trong FashionMNIST, mạng giáo viên của trang Code cho 0,88 cho "áo sơ mi", 0,07 cho "áo phông", 0,03 cho "áo len" và gần 0 cho "giày". Các xác suất nhỏ này cho biết mạng thấy những lớp nào **giống nhau**: áo sơ mi giống áo phông hơn giống giày. Hinton gọi thông tin này là **tri thức tối** (dark knowledge). Nhãn cứng (one-hot) không chứa nó: mọi lớp sai đều bằng 0 như nhau.

**Chưng cất tri thức** (knowledge distillation, KD) huấn luyện một mô hình nhỏ (học sinh, student) bắt chước đầu ra của một mô hình lớn đã huấn luyện (giáo viên, teacher), thay vì chỉ học từ nhãn cứng. Học sinh nhận được nhiều thông tin hơn trên mỗi mẫu: với $K$ lớp, nhãn cứng chỉ cho $\log_2 K$ bit, còn phân bố của giáo viên cho cả một vector $K$ chiều. Vì vậy chưng cất thường có lợi nhất khi **ít dữ liệu** hoặc khi học sinh khó tự học.

Ý tưởng gốc có từ trước: Buciluǎ, Caruana và Niculescu-Mizil (2006) nén một tổ hợp mô hình lớn thành một mạng neuron bằng cách huấn luyện mạng này khớp với đầu ra của tổ hợp; Ba và Caruana (2014) cho mạng nông học theo logit của mạng sâu. Hinton, Vinyals và Dean (2015) tổng quát hoá thành phương pháp dùng nhiệt độ, và đặt tên "chưng cất".

## 2 Chưng cất theo logit (Hinton và cộng sự, 2015)

### 2.1 Softmax có nhiệt độ

Giáo viên thường rất tự tin, nên các xác suất nhỏ gần như bằng 0 và không truyền được gì. Cách giải quyết là **làm mềm** phân bố bằng nhiệt độ $T$:

$$p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} \tag{2.1}$$

với $z_i$ là logit (đầu vào của softmax). $T = 1$ là softmax thường; $T$ càng lớn thì phân bố càng phẳng, các lớp sai nhận xác suất lớn hơn và học sinh "thấy" được thứ tự giữa chúng; $T \to \infty$ cho phân bố đều; $T \to 0$ cho one-hot.

### 2.2 Hàm mất mát

Gọi $v_i$ là logit của giáo viên, $z_i$ là logit của học sinh, $p^{t}(T)$ và $p^{s}(T)$ là phân bố mềm của hai mạng theo (2.1). Hàm mất mát của học sinh là tổng có trọng số của hai phần:

$$\mathcal{L} = \alpha\, T^2 \, \mathrm{KL}\!\left(p^{t}(T)\,\|\,p^{s}(T)\right) + (1-\alpha)\, \mathrm{CE}\!\left(y,\, p^{s}(1)\right) \tag{2.2}$$

- Phần thứ nhất (**mất mát chưng cất**) kéo phân bố mềm của học sinh về phía giáo viên. Dùng KL hay entropy chéo $\mathrm{CE}(p^t, p^s)$ đều được: hai cách chỉ khác nhau một hằng số (entropy của $p^t$), nên có cùng gradient theo học sinh.
- Phần thứ hai (**mất mát học sinh**) là entropy chéo thường với nhãn thật $y$, ở $T = 1$.
- $\alpha$ cân bằng hai phần; bài báo gốc thấy đặt trọng số nhỏ cho phần nhãn cứng thường tốt hơn, nhiều cài đặt dùng $\alpha \approx 0{,}5$–$0{,}9$.

### 2.3 Vì sao nhân $T^2$

Đạo hàm của phần chưng cất (dạng entropy chéo) theo logit của học sinh là

$$\frac{\partial}{\partial z_i}\, \mathrm{CE}\!\left(p^t(T), p^s(T)\right) = \frac{1}{T}\left(p^s_i(T) - p^t_i(T)\right) \tag{2.3}$$

Khi $T$ lớn so với độ lớn của logit, dùng $e^x \approx 1 + x$ và giả sử logit của mỗi mạng có trung bình 0 trên $K$ lớp, ta có $p_i(T) \approx \frac{1}{K}\left(1 + \frac{z_i}{T}\right)$, nên

$$\frac{\partial}{\partial z_i} \approx \frac{1}{K T^2}\left(z_i - v_i\right) \tag{2.4}$$

Hai hệ quả:

1. Gradient của phần chưng cất co lại theo $1/T^2$. Nếu không nhân bù $T^2$, đổi $T$ sẽ làm thay đổi cân bằng giữa hai phần của (2.2); nhân $T^2$ giữ độ lớn tương đối của chúng gần như không đổi khi thử các nhiệt độ khác nhau. Trang Code (mục 2) kiểm tra điều này bằng số.
2. Ở nhiệt độ cao, chưng cất tương đương với **khớp logit** bằng sai số bình phương, $\frac{1}{2}\sum_i (z_i - v_i)^2$, đúng là cách của Ba và Caruana (2014). Ở nhiệt độ thấp hơn, học sinh ít chú ý tới các logit rất âm của giáo viên; đó là điều tốt nếu các logit này nhiễu.

### 2.4 Chọn nhiệt độ

Không có $T$ tốt nhất chung. Bài báo của Hinton trên MNIST thấy khi học sinh đủ lớn (300 đơn vị ẩn mỗi lớp trở lên), mọi $T \ge 8$ cho kết quả gần như nhau; nhưng khi học sinh chỉ có 30 đơn vị mỗi lớp, $T$ trong khoảng 2,5–4 tốt hơn hẳn. Lý do có thể hiểu qua (2.4): học sinh quá nhỏ không đủ sức khớp mọi logit của giáo viên, nên nhiệt độ thấp hơn, chỉ tập trung vào các lớp có xác suất đáng kể, lại có lợi.

Trang Code thấy đúng xu hướng này, ở mức rõ hơn: với học sinh 9 nghìn tham số (nhỏ hơn giáo viên 187 lần), $T = 1$ tốt nhất, còn $T = 4$ và $T = 10$ không hơn huấn luyện từ đầu một cách rõ rệt. Trong thực tế, $T$ thường được chọn trong khoảng 1–10 bằng tập kiểm định.

## 3 Chưng cất theo đặc trưng

Logit chỉ là đầu ra cuối. Nhiều phương pháp cho học sinh học cả **biểu diễn trung gian** của giáo viên:

- **FitNets** (Romero và cộng sự, 2015): chọn một lớp giữa của giáo viên làm "gợi ý" (hint) và một lớp giữa của học sinh làm "lớp được dẫn" (guided layer); thêm một bộ hồi quy nhỏ (thường là tích chập $1 \times 1$) để đưa đặc trưng của học sinh về cùng số kênh với giáo viên, rồi tối thiểu hóa sai số bình phương giữa chúng. Cách này cho phép huấn luyện học sinh **sâu và hẹp** hơn giáo viên.
- **Chuyển giao attention** (Zagoruyko và Komodakis, 2017): thay vì khớp cả feature map, chỉ khớp "bản đồ chú ý" không gian $A = \sum_c |F_c|^2$ (tổng theo kênh, rồi chuẩn hóa), nên không cần học sinh và giáo viên có cùng số kênh.
- **Chưng cất quan hệ** (Park và cộng sự, 2019): học sinh khớp **khoảng cách và góc** giữa các mẫu trong không gian đặc trưng của giáo viên, không khớp từng đặc trưng riêng lẻ.

Chưng cất đặc trưng mạnh hơn khi kiến trúc hai mạng giống nhau, và là thành phần chính của nhiều phương pháp chưng cất cho phát hiện đối tượng và phân vùng ảnh, nơi logit cuối không chứa đủ thông tin.

## 4 Các biến thể

- **Tự chưng cất** (self-distillation, *Born-Again Networks*, Furlanello và cộng sự, 2018): học sinh có **cùng kiến trúc** với giáo viên, và lại thường vượt giáo viên. Điều này cho thấy chưng cất không chỉ là nén mà còn là một dạng chính quy hóa: nhãn mềm giống làm mượt nhãn (label smoothing) nhưng có thông tin về lớp.
- **Học tương hỗ** (Deep Mutual Learning, Zhang và cộng sự, 2018): không có giáo viên huấn luyện sẵn; nhiều học sinh học cùng lúc và chưng cất lẫn nhau.
- **Trợ giảng** (Teacher Assistant, Mirzadeh và cộng sự, 2020): khi khoảng cách năng lực giữa giáo viên và học sinh quá lớn, chưng cất kém đi; chèn một mạng trung gian (trợ giảng) giúp học sinh học tốt hơn. Đây là một cách giải thích kết quả của trang Code, nơi học sinh nhỏ hơn giáo viên 187 lần.
- **Chưng cất không dữ liệu** (data-free): khi không còn dữ liệu huấn luyện, sinh dữ liệu giả từ giáo viên (ví dụ tối ưu ảnh đầu vào để khớp thống kê BN của giáo viên) rồi chưng cất trên đó.
- **Chưng cất cho lượng tử hóa và cắt tỉa**: dùng mô hình FP32 hoặc mô hình chưa cắt làm giáo viên khi huấn luyện QAT hoặc khi tinh chỉnh sau cắt tỉa (xem bài [tổng quan](dlnote/compress_overview_vi), mục 7).

## 5 Chưng cất trong thực tế

- **DistilBERT** (Sanh và cộng sự, 2019): học sinh 6 lớp từ BERT-base 12 lớp, khởi tạo bằng một nửa số lớp của giáo viên, mất mát gồm chưng cất logit, mô hình hóa ngôn ngữ có che và khớp cosin giữa trạng thái ẩn. Nhỏ hơn 40%, nhanh hơn 60%, giữ 97% năng lực hiểu ngôn ngữ.
- **DeiT** (Touvron và cộng sự, 2021): thêm một "token chưng cất" vào Vision Transformer, học theo nhãn cứng do một CNN giáo viên dự đoán. Nhờ vậy ViT huấn luyện được tốt chỉ với ImageNet-1k.
- **Mô hình ngôn ngữ lớn.** Với LLM, phân bố trên từ vựng (hàng chục nghìn token) ở mỗi vị trí đóng vai trò nhãn mềm:
  - *Chưng cất mức chuỗi* (Kim và Rush, 2016): đơn giản cho học sinh huấn luyện có giám sát trên **văn bản do giáo viên sinh ra**. Cách này không cần truy cập logit của giáo viên, nên dùng được với giáo viên chỉ có API. Ví dụ, DeepSeek (2025) tạo khoảng 800 nghìn mẫu bằng DeepSeek-R1, rồi tinh chỉnh có giám sát các mô hình Qwen2.5 và Llama 3 (1,5B–70B) trên đó, được các mô hình nhỏ có khả năng suy luận tốt hơn hẳn so với huấn luyện học tăng cường trực tiếp trên mô hình nhỏ.
  - *Chưng cất mức token*: khớp phân bố từng token như (2.2). MiniLLM (Gu và cộng sự, 2024) dùng KL **ngược** $\mathrm{KL}(p^s\|p^t)$ để học sinh tập trung vào các vùng xác suất cao của giáo viên thay vì cố phủ hết, và lấy mẫu từ chính học sinh (chưng cất on-policy) để tránh lệch giữa huấn luyện và sinh văn bản.
  - *Cắt tỉa + chưng cất*: Minitron (NVIDIA, 2024), xem bài [tổng quan](dlnote/compress_overview_vi), mục 7.

## 6 Tóm tắt kết quả trên trang Code

| Thí nghiệm | Kết quả |
| --- | --- |
| Nhãn mềm của giáo viên | Ở $T = 1$ lớp đúng chiếm 0,88; ở $T = 4$ các loại áo khác (áo phông, áo len, áo khoác) nhận 0,12–0,17, giày dép vẫn gần 0 |
| Hệ số $T^2$ | Không nhân $T^2$, độ lớn gradient phần chưng cất giảm gần đúng theo $1/T^2$; có nhân thì gần như không đổi |
| Học sinh 9 nghìn tham số, 1000 ảnh | Từ đầu 79,7%; KD $T=1$ 81,1% (+1,4 điểm) |
| 5000 ảnh | Từ đầu 84,5%; KD $T=1$ 85,6% (+1,1 điểm) |
| Toàn bộ 60.000 ảnh | Từ đầu 87,5%; KD $T=1$ 87,6% (gần như bằng nhau) |
| $T = 4$, $T = 10$ | Ở 1000 và 5000 ảnh chỉ chênh trong phạm vi độ lệch giữa các lần chạy; kém 0,8 điểm khi đủ dữ liệu |

Chưng cất có ích rõ nhất khi thiếu dữ liệu, và với học sinh rất nhỏ thì nhiệt độ thấp tốt hơn, đúng như mục 2.4 và mục "trợ giảng" ở trên dự đoán.

## Tài liệu tham khảo

- Buciluǎ, Caruana and Niculescu-Mizil, [Model Compression](https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf), KDD 2006.
- Ba and Caruana, [Do Deep Nets Really Need to be Deep?](https://arxiv.org/abs/1312.6184), NIPS 2014.
- Hinton, Vinyals and Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), 2015.
- Romero et al., [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550), ICLR 2015.
- Zagoruyko and Komodakis, [Paying More Attention to Attention](https://arxiv.org/abs/1612.03928), ICLR 2017.
- Park et al., [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068), CVPR 2019.
- Furlanello et al., [Born-Again Neural Networks](https://arxiv.org/abs/1805.04770), ICML 2018.
- Zhang et al., [Deep Mutual Learning](https://arxiv.org/abs/1706.00384), CVPR 2018.
- Mirzadeh et al., [Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/abs/1902.03393), AAAI 2020.
- Sanh et al., [DistilBERT](https://arxiv.org/abs/1910.01108), 2019.
- Touvron et al., [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) (DeiT), ICML 2021.
- Kim and Rush, [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947), EMNLP 2016.
- Gu et al., [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543), ICLR 2024.
- DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948), 2025.
- [Intel Distiller: Knowledge Distillation](https://intellabs.github.io/distiller/knowledge_distillation.html).
