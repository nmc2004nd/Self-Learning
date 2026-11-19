# Mẹo huấn luyện: xử lý mẫu mất cân bằng

Trang [Code](dlnote/alchemy_code_vi) đi kèm (mục 3) cài phương pháp tổ hợp lại lớp, `WeightedRandomSampler` và focal loss, kiểm tra ý nghĩa của `pos_weight`, rồi so sáu cách xử lý trên một phiên bản đuôi dài của FashionMNIST.

<!-- theory: qa/ch02_vi#2.16.21 -->

## Lời nói đầu

Các giả định kinh điển của học máy thường coi số mẫu của các lớp trong tập huấn luyện là cân bằng, nhưng trong thực tế dữ liệu huấn luyện thường mất cân bằng. Ví dụ cực đoan trong bài toán phân loại ảnh hai lớp: tập huấn luyện có `95` mẫu dương nhưng chỉ `5` mẫu âm. Nếu không xử lý, mô hình sẽ "học thiếu" ở lớp ít mẫu, có thể mất hẳn khả năng dự đoán mẫu âm trên tập kiểm tra.

Ngoài phân loại và hồi quy, các bài toán ở mức điểm ảnh như phân đoạn ngữ nghĩa, ước lượng độ sâu cũng gặp vấn đề mất cân bằng.

Có hai hướng xử lý:

1. Ở **mức dữ liệu**: lấy mẫu lại (resampling) và lấy mẫu cân bằng theo lớp.
2. Ở **mức thuật toán**: các phương pháp nhạy chi phí (cost-sensitive).

Bản gốc chỉ trình bày ý tưởng và chiến lược, không có code cụ thể; trong dự án thực tế cần kết hợp các chiến lược này với bài toán để viết code xử lý dữ liệu hay hàm mất mát phù hợp. Bản dịch này dẫn sang trang Code cho từng phương pháp.

> **Biên tập:** Câu "mô hình mất hẳn khả năng dự đoán mẫu âm" là trường hợp xấu nhất. Điều thường xảy ra hơn là bộ phân loại thiên về lớp nhiều mẫu, và độ chính xác (accuracy) gộp chung che mất điều đó: với 95/5, đoán mọi mẫu là dương cũng đạt 95%. Vì vậy với dữ liệu mất cân bằng nên đánh giá bằng độ chính xác theo từng lớp, balanced accuracy, F1 hay PR-AUC; xem [mục 2.16.21 của khoá Ôn tập](qa/ch02_vi).

## 1 Các phương pháp ở mức dữ liệu

Ở mức dữ liệu có hai nhóm: **mở rộng dữ liệu và lấy mẫu**. Mở rộng dữ liệu thay đổi trực tiếp số lượng và độ phong phú của mẫu; lấy mẫu thì làm cho tập huấn luyện đưa vào mô hình gần cân bằng hơn, tức số mẫu các lớp gần bằng nhau.

Có hai chiến lược lấy mẫu:

1. **Lấy mẫu lại dữ liệu**, làm ở bước **tiền xử lý**, thay đổi số lượng và phân phối của toàn bộ tập huấn luyện.
2. **Lấy mẫu cân bằng theo lớp**, làm ở bước **nạp dữ liệu** (nạp vào mô hình, không phải đọc tệp từ ổ đĩa): đặt chiến lược lấy mẫu để tổng số lần mẫu của các lớp được đưa vào huấn luyện gần bằng nhau.

### 1.1 Mở rộng dữ liệu

Mất cân bằng thực chất là một số lớp có quá ít dữ liệu, vậy thì thêm dữ liệu cho chúng, đơn giản và trực tiếp. Nếu có thể, ưu tiên thu thập thêm dữ liệu thật; nhưng phần lớn thời gian ta không có lựa chọn đó, khi ấy cách hiệu quả nhất là **mở rộng dữ liệu bằng tăng cường dữ liệu**.

Các cách tăng cường thường gặp:

- lật ngang / lật dọc;
- xoay 90°, 180°, 270°;
- lật kết hợp xoay (xoay và lật giúp mô hình học được tính bất biến với phép xoay; cách làm ở mức tầng tích chập có thể tham khảo bài báo `ACNet`);
- thay đổi ngẫu nhiên độ sáng, độ bão hoà, độ tương phản;
- cắt ngẫu nhiên (Random Crop);
- co giãn ngẫu nhiên (Random Resize);
- làm mờ (Blurring);
- thêm nhiễu Gauss (Gaussian Noise).

Lưu ý: **cách tăng cường phải chọn theo bài toán cụ thể**; ngoài ba cách đầu, các cách khác cần cân nhắc kỹ vì mỗi bài toán phụ thuộc vào những đặc trưng khác nhau. Ví dụ trong cuộc thi phát hiện khuyết tật nhôm của Thiên Trì (Tianchi), có những ảnh vốn đã bị camera lấy nét không chuẩn, chi tiết khó nhìn; thêm làm mờ nữa thì nhiều mẫu coi như hỏng.

> Tham khảo bài [如何针对数据不平衡做处理](http://spytensor.com/index.php/archives/45/) (Xử lý dữ liệu mất cân bằng thế nào).

Các framework học sâu đều có sẵn một số hàm tăng cường dữ liệu, nhưng muốn nhiều phép mạnh hơn thì có thể dùng thư viện tăng cường ảnh. Bản gốc gợi ý thư viện `imgaug`; hiện nay nên dùng `torchvision.transforms.v2` hoặc `Albumentations` (xem [bài tăng cường dữ liệu](dlnote/alch_augment_vi), mục 4).

### 1.2 Lấy mẫu lại dữ liệu

Cách lấy mẫu lại đơn giản gồm **lấy mẫu thêm** (`over-sampling`, `up-sampling`) và **lấy mẫu bớt** (`under-sampling`, `down-sampling`).

1. **Với lớp có ít mẫu, có thể lấy mẫu thêm** (`over-sampling`): sao chép ảnh để số mẫu của lớp đó bằng số mẫu của lớp nhiều nhất.

2. **Với lớp có nhiều mẫu, có thể lấy mẫu bớt** (`under-sampling`). Với học sâu và thị giác máy tính, lấy mẫu bớt không phải là vứt ngẫu nhiên một phần ảnh; chiến lược đúng là: khi nạp dữ liệu theo batch (`dataloader`), với lớp nhiều mẫu, khống chế số ảnh rút vào mỗi batch, **sao cho mỗi batch có số mẫu dương và âm cân bằng**. Lấy bài toán hai lớp làm ví dụ: nếu theo phân phối gốc mỗi batch trung bình có tỉ lệ dương:âm là `9:1`, thì khi chỉ dùng lấy mẫu bớt, cứ `9` mẫu dương chỉ lấy `1` làm mẫu dương của batch, còn mẫu âm giữ nguyên, nhờ đó mỗi batch cân bằng.

Hình dưới minh hoạ lấy mẫu thêm và lấy mẫu bớt.

![Lấy mẫu bớt và lấy mẫu thêm](../images/imbalance/under_over.png)

Hình 1.1: Trái: lấy mẫu bớt (undersampling) lớp nhiều mẫu. Phải: lấy mẫu thêm (oversampling) bằng cách sao chép lớp ít mẫu. *Original dataset*: dữ liệu gốc.

#### Tổng kết các phương pháp lấy mẫu

Hiểu đơn giản, lấy mẫu thêm và lấy mẫu bớt là "thêm ảnh" và "bớt ảnh":

- **Lấy mẫu thêm**: lặp lại dữ liệu của lớp ít mẫu, thực ra không đưa thêm dạng dữ liệu mới nào cho mô hình; nhấn mạnh quá mức lớp ít mẫu sẽ khuếch đại ảnh hưởng của nhiễu trong lớp đó và dễ gây quá khớp.
- **Lấy mẫu bớt**: bỏ một phần dữ liệu của lớp nhiều mẫu, nên mất thông tin, mô hình có thể học thiếu ở lớp đó.

**Cả hai cách lấy mẫu lại đều thay đổi phân phối gốc của dữ liệu**; mô hình học trên phân phối cân bằng giả tạo này sẽ bị lệch so với phân phối thật khi triển khai.

> **Biên tập:** Bản gốc ghi lấy mẫu bớt "cũng gây quá khớp như lấy mẫu thêm". Rủi ro chính của lấy mẫu bớt là mất thông tin của lớp nhiều mẫu; quá khớp là vấn đề của lấy mẫu thêm (lặp đi lặp lại cùng vài ảnh). Trang Code minh hoạ: với `WeightedRandomSampler`, lớp ít nhất (60 ảnh) được rút khoảng 1500 lần mỗi epoch, nghĩa là mỗi ảnh lặp trung bình 25 lần.

Trên đây là **mô tả chiến lược** của các cách lấy mẫu; muốn giải quyết vấn đề trong bài toán thực tế còn cần phân tích sâu bài toán (`task`), hiểu dữ liệu, và có kinh nghiệm viết code xử lý, lấy mẫu dữ liệu: chiến lược tốt cộng công cụ thành thạo.

> Vì chỉ lấy mẫu thêm có thể gây quá khớp, trong thực tế chiến lược an toàn hơn thường là kết hợp lấy mẫu thêm và lấy mẫu bớt.

### 1.3 Lấy mẫu cân bằng theo lớp

Chiến lược lấy mẫu lại ở trên nhắm vào số lượng mẫu của từng lớp; một nhóm chiến lược khác **nhắm thẳng vào lớp và không đổi tổng số mẫu, gọi là lấy mẫu cân bằng theo lớp**. Cách đơn giản là nhóm mẫu theo lớp, mỗi lớp một danh sách; khi huấn luyện, chọn ngẫu nhiên một hoặc vài lớp rồi chọn ngẫu nhiên mẫu từ danh sách của lớp đó, nhờ vậy cơ hội tham gia huấn luyện của các lớp gần bằng nhau.

Cách trên quá đơn giản và có nhiều hạn chế trong thực tế, chẳng hạn khi có rất nhiều lớp (như `ImageNet`). **Dựa trên lấy mẫu cân bằng theo lớp, Viện nghiên cứu Hikvision (Hải Khang Uy Thị) đề xuất phương pháp "tổ hợp lại lớp" (class-aware sampling / label shuffling).**

> Tác giả bản gốc biết đến phương pháp này qua sách *Giải tích mạng neuron tích chập* (解析卷积神经网络) và không tìm được bài báo, code gốc.

> **Biên tập:** Phương pháp này là kỹ thuật "label shuffling" do đội Hikvision trình bày trong slide tại hội thảo ILSVRC 2016, không có bài báo riêng; sách của Ngụy Tú Tham dẫn nó ở chương 12. Ý tưởng gần giống "class-aware sampling" của Shen và cộng sự (ECCV 2016, Relay Backpropagation).

Như hình dưới, **phương pháp tổ hợp lại lớp** gồm các bước:

![Phương pháp tổ hợp lại lớp](../images/imbalance/class_balance1.png)

Hình 1.2: Phương pháp tổ hợp lại lớp xử lý mẫu mất cân bằng (từ sách *Giải tích mạng neuron tích chập*, hình 12.2).

1. Xếp các mẫu theo lớp, đếm số mẫu của mỗi lớp và ghi lại số mẫu của lớp lớn nhất, `max_num`.
2. Với mỗi lớp, sinh một danh sách ngẫu nhiên độ dài `max_num` (một hoán vị của $0, 1, \dots, \text{max\_num}-1$), rồi lấy từng số chia dư cho số mẫu của lớp đó, được danh sách chỉ số `index_list`.
3. Theo `index_list`, lấy ảnh tương ứng của lớp, được danh sách ảnh ngẫu nhiên của lớp đó.
4. Nối danh sách của mọi lớp rồi xáo trộn, được danh sách ảnh cuối cùng; trong danh sách này mỗi lớp có cùng số mẫu (ảnh của lớp ít mẫu bị lấy nhiều lần). Mỗi epoch duyệt danh sách này để huấn luyện, rồi lặp lại.

Phương pháp này khá dễ cài, chỉ cần viết code theo bài toán. Ví dụ sinh một danh sách số nguyên ngẫu nhiên trong khoảng $[0, 9]$:

```python
import random
# sinh danh sách số nguyên ngẫu nhiên trong khoảng [0, 9], không lặp
# sample(L, n): rút ngẫu nhiên n phần tử từ dãy L, trả về list
# cũng có thể dùng random.shuffle(L) để xáo trộn list tại chỗ
random_list = random.sample(range(0, 10), 10)
print(random_list)
```

```output
[2, 1, 4, 0, 3, 5, 7, 9, 8, 6]
```

> **Biên tập:** Chú thích trong bản gốc ghi khoảng `[1, 10]` (ảnh chụp màn hình ghi `[1,100]`) trong khi code sinh các số từ 0 đến 9; kết quả ở trên chạy với `random.seed(1)`. Trang Code cài trọn phương pháp tổ hợp lại lớp (khoảng 10 dòng) và chạy trên đúng ví dụ của hình 1.2: lớp a 3 ảnh, b 2 ảnh, c 5 ảnh cho danh sách 15 ảnh, mỗi lớp 5 lần.

Ưu điểm của phương pháp tổ hợp lại lớp rất rõ: viết xong hàm tổ hợp, chỉ cần danh sách ảnh gốc, mọi thao tác làm trực tiếp trong bộ nhớ, dễ cài và dùng chung được. Nghĩ kỹ thì phương pháp của Hikvision rất giống các cách lấy mẫu ở trên: **bản chất đều là dùng chiến lược lấy mẫu (sampler) để các lớp mất cân bằng xuất hiện cùng số lần trong mỗi epoch**. Trong PyTorch, cách làm gọn nhất là `torch.utils.data.WeightedRandomSampler` với trọng số mỗi mẫu bằng $1/n_c$ ($n_c$ là số mẫu của lớp chứa nó).

## 2 Các phương pháp ở mức thuật toán (hàm mất mát)

Bản chất của mất cân bằng lớp là lớp ít mẫu bị "học thiếu", biểu hiện trực tiếp là phần đóng góp của lớp đó vào hàm mất mát nhỏ. Cách giải quyết tự nhiên là tăng **chi phí phạt khi phân loại sai lớp ít mẫu** và đưa chi phí đó thẳng vào hàm mục tiêu (hàm mất mát); đó là phương pháp "nhạy chi phí". Bản chất của phương pháp nhạy chi phí có thể hiểu là điều chỉnh mức chú ý của mô hình tới lớp ít mẫu.

### 2.1 Focal Loss

Focal Loss xây dựng trên hàm mất mát cross-entropy (CE) của bài toán hai lớp, nhằm giải quyết việc tỉ lệ mẫu dương/âm mất cân bằng nặng trong bộ phát hiện đối tượng một giai đoạn (`one-stage`). Hàm này **giảm trọng số của số lượng lớn mẫu âm dễ trong quá trình huấn luyện**, cũng có thể hiểu là một dạng khai thác mẫu khó (hard example mining). Thực tế cho thấy `Focal Loss` hiệu quả trong phát hiện đối tượng một giai đoạn, nhưng với phân loại nhiều lớp thì chưa chắc.

Tác giả `Focal Loss` thêm vào cross-entropy một hệ số điều biến (`modulating factor`) $(1-p_t)^\gamma$ để giảm loss của các mẫu có độ tin cậy cao (mẫu dễ). Định nghĩa:

$$
FL(p_t) = -(1-p_t)^\gamma \log(p_t) = \begin{cases}
-(1-p)^\gamma \log(p), & y=1 \\
-p^\gamma \log(1-p), & y=0
\end{cases}
$$

trong đó $p$ là xác suất mô hình dự đoán cho lớp dương và $p_t = p$ nếu $y=1$, $p_t = 1-p$ nếu $y=0$.

`Focal Loss` có hai tính chất:

- Khi mẫu bị phân loại sai và $p_t$ nhỏ, hệ số điều biến gần `1` nên loss gần như không đổi; khi $p_t$ gần `1`, hệ số gần `0`, **loss của mẫu dễ bị giảm trọng số**. Nhìn tổng thể, điều này tương đương tăng trọng số của các mẫu phân loại chưa đúng trong hàm mất mát.
- Tham số $\gamma$ điều chỉnh mượt tốc độ giảm trọng số của mẫu dễ: $\gamma = 0$ thì `Focal Loss` chính là `CE Loss`; $\gamma$ càng lớn thì hệ số điều biến càng mạnh. Trong thí nghiệm của tác giả, $\gamma = 2$ cho kết quả tốt nhất.

Trực quan, **hệ số điều biến giảm phần đóng góp của mẫu dễ vào loss, và mở rộng vùng mà mẫu nhận loss thấp**. Ví dụ với $\gamma = 2$, so với CE, mẫu có $p_t = 0{,}9$ có loss giảm `100` lần, mẫu có $p_t \approx 0{,}968$ giảm khoảng `1000` lần. Điều này lại làm tăng tầm quan trọng của mẫu bị phân loại sai (với $p_t \le 0{,}5$ và $\gamma = 2$, loss giảm nhiều nhất `4` lần).

| $p_t$ | CE | FL ($\gamma=2$) | CE / FL |
| --- | --- | --- | --- |
| 0,1 | 2,303 | 1,865 | 1,2 |
| 0,5 | 0,693 | 0,173 | 4 |
| 0,9 | 0,105 | 0,00105 | 100 |
| 0,968 | 0,0325 | 0,000033 | 977 |
| 0,99 | 0,0101 | 0,000001 | 10000 |

Trong thực tế thường dùng `Focal Loss` có thêm hệ số $\alpha$:

$$
FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t), \qquad \alpha_t = \begin{cases} \alpha, & y=1 \\ 1-\alpha, & y=0 \end{cases}
$$

Tác giả dùng dạng này trong thí nghiệm và thấy độ chính xác nhích lên một chút so với dạng không có $\alpha$. Thí nghiệm cho thấy $\gamma = 2$, $\alpha = 0{,}25$ là tốt nhất.

> Xem thêm [bài báo focal loss](https://arxiv.org/abs/1708.02002) (Lin và cộng sự, ICCV 2017).

> **Biên tập:** Bản gốc có một bảng xếp thứ tự ưu tiên "dương khó > âm khó > dương dễ > âm dễ", trong đó mẫu âm bị giảm trọng số bởi $\alpha$, và công thức $FL$ có $\alpha$ viết thiếu chỉ số $t$. Theo bài báo, $\alpha$ là trọng số của lớp **dương**, $1-\alpha$ của lớp âm; với $\alpha = 0{,}25$, chính **mẫu dương** bị giảm trọng số (0,25 so với 0,75). Bài báo giải thích: số hạng $(1-p_t)^\gamma$ đã giảm rất mạnh phần đóng góp của số lượng lớn mẫu âm dễ, nên khi tăng $\gamma$ thì $\alpha$ tốt nhất lại giảm. Trang Code kiểm tra với `torchvision.ops.sigmoid_focal_loss`: mẫu dương và mẫu âm cùng $p_t = 0{,}9$ có loss $0{,}00026$ và $0{,}00079$, đúng tỉ lệ 1:3. Bảng xếp thứ tự đã bỏ.

### 2.2 Đánh trọng số cho hàm mất mát

Ngoài chiến lược hàm mất mát tinh tế như `Focal Loss`, với bài toán phân loại ảnh còn một cách đơn giản, trực tiếp: **đánh trọng số cho loss của từng lớp khi tính hàm mất mát**. Trong `PyTorch`, với bài toán một nhãn nhiều lớp dùng tham số `weight` của `CrossEntropyLoss`:

```python
weights = torch.tensor([1, 1, 8, 8, 4.])   # trọng số các lớp 1:1:8:8:4
# weight: tensor 1 chiều, n phần tử, trọng số của n lớp;
# loss của mẫu thuộc lớp c được nhân với weights[c],
# rồi lấy trung bình có trọng số (chia cho tổng trọng số của các mẫu trong batch).
criterion = nn.CrossEntropyLoss(weight=weights)
```

Còn với bài toán **đa nhãn** (mỗi ảnh có thể thuộc nhiều lớp, mỗi lớp là một bài toán hai lớp riêng), `BCEWithLogitsLoss` có tham số `pos_weight`: hệ số nhân vào số hạng **mẫu dương** ($y=1$) của từng lớp, thường đặt bằng $n_{âm}/n_{dương}$ của lớp đó:

```python
pos_weight = torch.tensor([1, 1, 8, 8, 4.])  # 5 nhãn
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss = criterion(logits, multi_hot_target)   # cả hai có hình dạng (N, 5)
```

> **Biên tập:** Bản gốc dùng `BCEWithLogitsLoss(pos_weight=weights)` với chú thích "trọng số của n lớp, hệ số co giãn thủ công cho loss của mỗi phần tử trong batch" cho bài toán phân loại ảnh nhiều lớp. Chú thích này trộn hai tham số khác nhau: `weight` của `BCEWithLogitsLoss` là hệ số cho từng phần tử, còn `pos_weight` chỉ nhân vào số hạng $y=1$, không đổi số hạng $y=0$. Và `BCEWithLogitsLoss` cần target cùng hình dạng với đầu ra (đa nhãn); đưa nhãn lớp dạng `(N,)` vào thì báo lỗi `Target size must be the same as input size` (trang Code kiểm tra cả ba điều). Với phân loại nhiều lớp, dùng `CrossEntropyLoss(weight=...)` như trên. Lệnh `.cuda()` cuối dòng của bản gốc cũng đã bỏ: hàm mất mát không có tham số, không cần chuyển sang GPU.

## 3 Bổ sung: so sánh trên dữ liệu đuôi dài và các phương pháp mới

Mục này do người biên soạn viết thêm.

Trang Code tạo một phiên bản **đuôi dài** (long-tailed) của FashionMNIST: lớp 0 giữ 6000 ảnh, các lớp sau giảm theo cấp số nhân tới lớp 9 chỉ còn 60 ảnh (tỉ lệ mất cân bằng 100, tổng 14886 ảnh), huấn luyện một MLP 10 epoch rồi đánh giá trên tập kiểm tra cân bằng (1000 ảnh mỗi lớp), trung bình 3 seed:

| Phương pháp | Độ chính xác (cân bằng) | 3 lớp nhiều mẫu | 3 lớp ít mẫu |
| --- | --- | --- | --- |
| CE thường | 0,798 | 0,930 | 0,816 |
| CE có trọng số $1/n_c$ | 0,822 | 0,866 | 0,861 |
| Lấy mẫu cân bằng (`WeightedRandomSampler`) | 0,831 | 0,892 | 0,881 |
| Focal loss, $\gamma=2$ | 0,796 | 0,934 | 0,828 |
| Logit adjustment trong hàm mất mát | 0,833 | 0,871 | 0,886 |
| CE + chỉnh logit khi suy luận | 0,830 | 0,862 | 0,884 |

Mọi cách tái cân bằng đều đổi một phần độ chính xác của lớp nhiều mẫu lấy độ chính xác của lớp ít mẫu, và tăng độ chính xác cân bằng 2,4–3,5 điểm. Focal loss không giúp gì: nó nhắm vào mẫu dễ và mẫu khó chứ không nhắm vào lớp, đúng như nhận xét "với phân loại nhiều lớp thì chưa chắc" ở mục 2.1.

Các phương pháp đáng biết từ sau bài gốc:

- **Logit adjustment** (Menon và cộng sự, ICLR 2021): mô hình huấn luyện bằng CE thường học xác suất có điều kiện theo phân phối huấn luyện, $p(y\mid x) \propto p(x\mid y)\,\pi_y$. Để có bộ phân loại tối ưu cho phân phối cân bằng, chỉ cần trừ $\tau \log \pi_y$ khỏi logit khi suy luận ($\pi_y$ là tần suất lớp trong tập huấn luyện), hoặc cộng $\tau\log\pi_y$ vào logit khi tính loss lúc huấn luyện. Không phải đổi dữ liệu, không thêm siêu tham số nào ngoài $\tau$ (thường bằng 1). Bảng trên cho thấy cách chỉnh khi suy luận, áp lên mô hình CE đã huấn luyện sẵn, đạt gần như ngang các cách phải huấn luyện lại.
- **Class-balanced loss** (Cui và cộng sự, CVPR 2019): thay trọng số $1/n_c$ bằng $\frac{1-\beta}{1-\beta^{n_c}}$ dựa trên "số mẫu hiệu dụng", tránh trọng số quá lớn cho lớp rất ít mẫu.
- **Tách rời biểu diễn và bộ phân loại** (Kang và cộng sự, ICLR 2020, decoupling): huấn luyện toàn mạng với lấy mẫu thường, sau đó chỉ huấn luyện lại tầng phân loại với lấy mẫu cân bằng (cRT) hoặc chuẩn hoá chuẩn của trọng số từng lớp ($\tau$-normalization).
- **LDAM-DRW** (Cao và cộng sự, NeurIPS 2019): biên (margin) lớn hơn cho lớp ít mẫu, kết hợp đánh trọng số lại từ giữa quá trình huấn luyện.
- **Với phát hiện đối tượng**, các bộ phát hiện hiện nay (FCOS, YOLOv8, DETR) vẫn dùng focal loss hoặc các biến thể (Quality Focal Loss, Varifocal Loss) cho nhánh phân loại.
- **Đánh giá**: dùng balanced accuracy, macro-F1 hoặc PR-AUC; với bài toán hai lớp, có thể giữ mô hình huấn luyện bình thường và chỉ dời ngưỡng quyết định theo đường precision–recall trên tập kiểm định.

## Tài liệu tham khảo

- Ngụy Tú Tham. *Giải tích mạng neuron tích chập* (解析卷积神经网络), chương 12.
- [如何针对数据不平衡做处理](http://spytensor.com/index.php/archives/45/) (Xử lý dữ liệu mất cân bằng thế nào)
- [10 Techniques to deal with Imbalanced Classes in Machine Learning](https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/)
- Lin et al. [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). ICCV 2017.
- Shen et al. [Relay Backpropagation for Effective Learning of Deep Convolutional Neural Networks](https://arxiv.org/abs/1512.05830). ECCV 2016.
- Cui et al. [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555). CVPR 2019.
- Cao et al. [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://arxiv.org/abs/1906.07413). NeurIPS 2019.
- Kang et al. [Decoupling Representation and Classifier for Long-Tailed Recognition](https://arxiv.org/abs/1910.09217). ICLR 2020.
- Menon et al. [Long-tail learning via logit adjustment](https://arxiv.org/abs/2007.07314). ICLR 2021.
