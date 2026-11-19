# Backbone kinh điển: ResNet v2

Bài này đọc bài báo *Identity Mappings in Deep Residual Networks* (He, Zhang, Ren, Sun, ECCV 2016), thường gọi là ResNet v2. Trang [Code](dlnote/backbone_code_vi) đi kèm kiểm tra bằng số phân tích lan truyền ngược ở mục 2 và 3: gradient qua kết nối tắt đồng nhất giữ nguyên độ lớn, còn shortcut có hệ số $\lambda$ làm gradient nổ hoặc tắt theo cấp số mũ.

<!-- theory: qa/ch04_vi#Bổ sung: ResNet và DenseNet -->

## Lời nói đầu

> Đóng góp chính của bài báo là chứng minh bằng phân tích lý thuyết và nhiều thí nghiệm rằng dùng ánh xạ đồng nhất (`identity mapping`) làm kết nối tắt (`skip connection`) là rất quan trọng với khối residual. Đồng thời, bài báo dời các phép kích hoạt `BN/ReLU` lên **trước** tầng `Conv` (tầng trọng số thực sự), gọi là "tiền kích hoạt" (pre-activation), và qua thí nghiệm so sánh với "hậu kích hoạt" (post-activation) cho thấy với mạng nhiều tầng, ResNet dùng đơn vị residual tiền kích hoạt (`pre-activation residual unit`), tức `ResNet v2`, đều cho kết quả tốt hơn `ResNet v1` (hoặc ResNet v1.5).

## Tóm tắt

Gần đây xuất hiện nhiều kiến trúc mạng cực sâu dựa trên mạng residual sâu (deep residual network), cho kết quả rất ấn tượng về độ chính xác và khả năng hội tụ. Bài báo phân tích sự lan truyền tín hiệu trong khối xây dựng residual (`residual building block`), và phát hiện rằng khi dùng ánh xạ đồng nhất làm kết nối tắt và đặt hàm kích hoạt sau phép cộng thành ánh xạ đồng nhất, tín hiệu lan truyền xuôi và ngược có thể đi thẳng từ một khối tới bất kỳ khối nào khác mà không qua phép biến đổi nào. **Nhiều thí nghiệm khẳng định tầm quan trọng của ánh xạ đồng nhất.** Dựa trên phát hiện này, bài báo thiết kế lại đơn vị residual, giúp mạng dễ huấn luyện hơn và tổng quát hoá tốt hơn.

> Lưu ý các thí nghiệm ở đây là trên ResNet sâu ($\geq 110$ tầng), nên theo tác giả bản gốc, kết luận đúng nhất là: với ResNet sâu, mạng dùng đơn vị residual tiền kích hoạt (ResNet v2) dễ huấn luyện hơn và chính xác hơn.

## 1 Giới thiệu

Mạng residual sâu (`ResNets`) được tạo bằng cách chồng các đơn vị residual (`Residual Units`). Mỗi đơn vị (hình 1(a)) có thể viết là:

$$
y_l = h(x_l) + F(x_l, W_l), \qquad x_{l+1} = f(y_l) \tag{1}
$$

trong đó $x_l$ và $x_{l+1}$ là đầu vào và đầu ra của đơn vị residual thứ $l$, $F$ là hàm residual. Trong ResNet, $h(x_{l})= x_{l}$ là ánh xạ đồng nhất (`identity`), $f$ là hàm kích hoạt `ReLU`. ResNet với hơn 100 tầng đạt độ chính xác tốt nhất thời điểm đó trên ImageNet và COCO. Ý tưởng cốt lõi của mạng residual là học thêm hàm residual $F$ trên nền $h(x_{l})$, và lựa chọn then chốt là dùng ánh xạ đồng nhất $h(x_{l})= x_{l}$, thực hiện bằng một kết nối tắt (`shortcut`) đồng nhất.

> **Biên tập:** Bản gốc viết "mạng residual hơn 1000 tầng đạt độ chính xác tốt nhất trên ImageNet và COCO". Bài báo viết ResNet "hơn 100 tầng" đạt kết quả tốt nhất trên ImageNet và COCO; mạng hơn 1000 tầng chỉ được huấn luyện trên CIFAR (bài báo ResNet gốc thử 1202 tầng và thấy kém hơn 110 tầng do quá khớp).

Bài báo tập trung phân tích việc tạo một đường lan truyền thông tin "trực tiếp" trong mạng residual sâu: không chỉ trong một đơn vị residual mà trên toàn mạng. Nếu cả $h(x_{l})$ và $f(y_{l})$ đều là ánh xạ đồng nhất, **tín hiệu có thể lan truyền xuôi và ngược trực tiếp giữa các đơn vị**. Thí nghiệm cho thấy kiến trúc gần thoả điều kiện này thường dễ huấn luyện hơn. Bài báo thử các dạng $h(x_{l})$ khác nhau và thấy ánh xạ đồng nhất cho kết quả tốt nhất: sai số giảm nhanh nhất và loss huấn luyện thấp nhất. Các thí nghiệm này cho thấy một kênh thông tin "sạch" giúp tối ưu dễ hơn. Các dạng $h(x_{l})$ khác nhau là các mũi tên xám trong hình 1, 2 và 4 của bài báo.

![Đơn vị residual của ResNet và ResNet v2](../images/resnetv2/residual_unit_structure_diagram_of_resnet_and_resnetv2.png)

Hình 1: (a) Đơn vị residual gốc. (b) Đơn vị residual tiền kích hoạt của ResNet v2. Bên phải: sai số huấn luyện (nét đứt) và kiểm tra (nét liền) của ResNet-1001 trên CIFAR-10 với hai loại đơn vị (hình 1 của bài báo).

Để có $f(y_l)=y_l$ là ánh xạ đồng nhất, bài báo dời các hàm kích hoạt (`ReLU` và `BN`) lên trước tầng trọng số (`Conv`), thành cách "tiền kích hoạt" (`pre-activation`) thay vì "hậu kích hoạt" (`post-activation`) thông thường, và thiết kế ra đơn vị residual mới (hình `1(b)`). Với đơn vị mới, bài báo huấn luyện mạng residual `1001` tầng trên `CIFAR-10/100`, và thấy nó dễ huấn luyện và tổng quát hoá tốt hơn `ResNet` gốc. Bài báo cũng thử mạng 200 tầng trên `ImageNet`, độ sâu mà ResNet gốc bắt đầu quá khớp. Các kết quả cho thấy **chiều sâu của mạng** còn nhiều chỗ để khai thác, vì độ sâu là chìa khoá thành công của mạng neuron hiện đại.

## 2 Phân tích mạng residual sâu

Đơn vị residual của ResNet gốc có thể viết:

$$
y_l = h(x_l) + F(x_l, W_l), \tag{1}
$$

$$
x_{l+1} = f(y_l). \tag{2}
$$

Trong đó $x_l$ là đặc trưng đầu vào của đơn vị thứ $l$, $W_l = \{W_{l,k}\}_{1\le k\le K}$ là tập trọng số (và bias) của đơn vị, $K$ là số tầng trong một đơn vị (2 hoặc 3 trong ResNet), $F$ là hàm residual (ví dụ hai tầng tích chập 3×3 chồng nhau), $f$ là phép sau khi cộng (ReLU trong ResNet). Trong `ResNet`, $h$ là ánh xạ đồng nhất, $h(x_{l}) = x_{l}$.

> **Biên tập:** Bản gốc dán các công thức và phần giải thích ký hiệu dưới dạng ảnh chụp bài báo; đã gõ lại toàn bộ bằng LaTeX.

Nếu $f$ cũng là ánh xạ đồng nhất, $x_{l+1} \equiv y_{l}$, gộp hai công thức `(1)(2)` được:

$$
x_{l+1} = x_l + F(x_l, W_l). \tag{3}
$$

Áp dụng đệ quy, quan hệ giữa một đơn vị sâu **bất kỳ** $L$ và một đơn vị nông hơn $l$ là:

$$
x_L = x_l + \sum_{i=l}^{L-1} F(x_i, W_i). \tag{4}
$$

> **Biên tập:** Bản gốc viết điều kiện là "$y_{l}\equiv y_{l}$"; đúng là $x_{l+1} \equiv y_l$.

Công thức `(4)` có hai tính chất:

1. Đặc trưng của đơn vị sâu bằng đặc trưng của đơn vị nông cộng với các hàm residual;
2. Đặc trưng của một đơn vị sâu bất kỳ bằng đặc trưng ban đầu $x_0$ cộng tổng mọi hàm residual trước nó. Điều này khác với mạng thẳng (`plain`), nơi đặc trưng ở tầng sâu là kết quả của một chuỗi phép nhân ma trận với vector (bỏ qua BN và ReLU): $x_L = \prod_{i=0}^{L-1} W_i x_0$. **Mạng residual là tổng liên tiếp, mạng thẳng là tích liên tiếp.**

Công thức `(4)` còn cho tính chất lan truyền ngược tốt. Gọi $\varepsilon$ là hàm mất mát, theo quy tắc chuỗi:

$$
\frac{\partial \varepsilon}{\partial x_l} = \frac{\partial \varepsilon}{\partial x_L}\frac{\partial x_L}{\partial x_l} = \frac{\partial \varepsilon}{\partial x_L}\left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F(x_i, W_i)\right). \tag{5}
$$

Công thức `(5)` cho thấy gradient cũng đi theo hai đường: số hạng $\frac{\partial \varepsilon}{\partial x_L}$ truyền thông tin trực tiếp, không qua tầng trọng số nào; số hạng còn lại đi qua mọi tầng có trọng số. Số hạng đầu bảo đảm thông tin truyền ngược trực tiếp về bất kỳ đơn vị nông nào. Ngoài ra, số hạng trong ngoặc khó có thể luôn bằng $-1$ với mọi mẫu trong một `mini-batch`, nên gradient không biến mất, kể cả khi trọng số rất nhỏ.

## 3 Tầm quan trọng của kết nối tắt đồng nhất

Xét tầm quan trọng của ánh xạ đồng nhất. Giả sử thay ánh xạ đồng nhất bằng một phép nhân hệ số, $h(x_{l}) = \lambda_{l}x_{l}$:

$$
x_{l+1} = \lambda_l x_l + F(x_l, W_l). \tag{6}
$$

Áp dụng đệ quy như công thức `(4)`:

$$
x_L = \left(\prod_{i=l}^{L-1}\lambda_i\right) x_l + \sum_{i=l}^{L-1}\left(\prod_{j=i+1}^{L-1}\lambda_j\right) F(x_i, W_i) = \left(\prod_{i=l}^{L-1}\lambda_i\right) x_l + \sum_{i=l}^{L-1}\hat F(x_i, W_i), \tag{7}
$$

trong đó $\hat{F}$ là hàm residual đã gộp các hệ số vô hướng vào. Tương tự công thức `(5)`, lan truyền ngược cho:

$$
\frac{\partial \varepsilon}{\partial x_l} = \frac{\partial \varepsilon}{\partial x_L}\left(\left(\prod_{i=l}^{L-1}\lambda_i\right) + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1}\hat F(x_i, W_i)\right). \tag{8}
$$

Khác với công thức `(5)`, số hạng đầu của công thức `(8)` bị nhân với $\prod_{i=l}^{L-1}\lambda_{i}$. Với mạng cực sâu ($L$ rất lớn): nếu mọi $\lambda_i > 1$, tích này tăng theo cấp số mũ; nếu mọi $\lambda_i < 1$, tích này rất nhỏ, thậm chí biến mất, chặn tín hiệu lan truyền ngược qua `shortcut` và buộc nó đi qua các tầng trọng số. Thí nghiệm của bài báo cho thấy điều này gây khó khăn cho tối ưu.

> Ví dụ bằng số: với 100 đơn vị, $\lambda = 0{,}9$ cho hệ số $0{,}9^{100} \approx 2,7\times10^{-5}$, còn $\lambda = 1{,}1$ cho $1{,}1^{100} \approx 1,4\times10^{4}$. Trang Code đo gradient thật qua một chồng 100 đơn vị residual và cho kết quả khớp.

Các dạng biến đổi khác trên shortcut (cổng, tích chập 1×1, dropout) cũng cản trở tín hiệu lan truyền và ảnh hưởng quá trình huấn luyện.

![Các dạng shortcut khác nhau](../images/resnetv2/different_forms_of_shortcuts.png)

Hình 2: Các dạng shortcut được thử: (a) đồng nhất, (b) nhân hằng số 0,5, (c) cổng độc quyền, (d) cổng chỉ trên shortcut, (e) tích chập 1×1, (f) dropout (hình 2 của bài báo).

## 4 Cách dùng hàm kích hoạt

Mục `3` xét các dạng ánh xạ khác nhau (hình `2`) để kiểm chứng tầm quan trọng của việc $h$ là ánh xạ đồng nhất. Mục này bàn về $f$ trong công thức `(2)`: nếu $f$ cũng là ánh xạ đồng nhất, hiệu năng có tốt hơn không? Bằng cách đổi vị trí của các hàm kích hoạt (`ReLU` và/hoặc `BN`) để $f$ thành ánh xạ đồng nhất. Hình `4` là các đơn vị residual với hàm kích hoạt ở các vị trí khác nhau.

> Hình `4(e)`, "tiền kích hoạt đầy đủ", là cấu trúc mà bài báo đề xuất để huấn luyện mạng residual sâu hiệu quả hơn (`ResNet v2`).

![Các vị trí của hàm kích hoạt](../images/resnetv2/different_forms_of_relu_activation.png)

Hình 3: Các cách đặt hàm kích hoạt: (a) gốc, (b) BN sau phép cộng, (c) ReLU trước phép cộng, (d) chỉ ReLU tiền kích hoạt, (e) tiền kích hoạt đầy đủ (hình 4 của bài báo).

### 4.1 Thí nghiệm về hàm kích hoạt

Mục này dùng `ResNet-110` và mạng bottleneck `164` tầng (gọi là `ResNet-164`). Đơn vị residual bottleneck gồm một tầng $1\times 1$ để giảm số chiều, một tầng $3\times 3$, và một tầng $1\times 1$ để khôi phục số chiều. Như bài báo ResNet mô tả, độ phức tạp tính toán của nó tương đương đơn vị residual có hai tầng $3\times 3$.

**BN sau phép cộng (hình 4(b))**: kết quả kém hơn mốc so sánh. Đặt BN sau phép cộng cản trở lan truyền tín hiệu; dấu hiệu rõ là sai số giảm chậm ở đầu quá trình huấn luyện.

**ReLU trước phép cộng (hình 4(c))**: đầu ra của nhánh residual luôn không âm, làm giảm năng lực biểu diễn của mô hình (đặc trưng chỉ có thể tăng dần qua các đơn vị); thí nghiệm cũng cho kết quả kém hơn mốc so sánh.

**Hậu kích hoạt hay tiền kích hoạt**: thiết kế gốc có ReLU sau phép cộng, ảnh hưởng tới cả hai nhánh của đơn vị residual. Nay chuyển nó vào nhánh residual, để nhánh shortcut không còn bị ảnh hưởng. Cách làm như hình `5`.

![Hậu kích hoạt và tiền kích hoạt](../images/resnetv2/Figure5.png)

Hình 4: Dùng hàm kích hoạt bất đối xứng: hậu kích hoạt của đơn vị trước tương đương tiền kích hoạt của đơn vị sau (hình 5 của bài báo).

Theo vị trí của hàm kích hoạt so với phép cộng, cách cũ gọi là "hậu kích hoạt" (`post-activation`), cách mới gọi là "tiền kích hoạt" (`pre-activation`). So sánh giữa thiết kế gốc và đơn vị tiền kích hoạt ở bảng `3`. Tiền kích hoạt lại có hai dạng: chỉ đưa `ReLU` lên trước, hoặc đưa cả `ReLU` và `BN` lên trước; kết quả ở bảng `2` cho thấy tiền kích hoạt đầy đủ (`full pre-activation`) tốt hơn.

![Bảng 2](../images/resnetv2/table_2.png)

Hình 5: Sai số phân loại trên CIFAR-10 của ResNet-110 và ResNet-164 với các cách đặt hàm kích hoạt (bảng 2 của bài báo).

![Bảng 3](../images/resnetv2/table_3.png)

Hình 6: Sai số phân loại trên CIFAR-10/100 của các đơn vị gốc và tiền kích hoạt (bảng 3 của bài báo).

### 4.2 Phân tích

Tiền kích hoạt có hai lợi ích: (1) $f$ thành ánh xạ đồng nhất, mạng dễ tối ưu hơn; (2) BN làm tiền kích hoạt tăng hiệu ứng regularization cho mô hình.

**Dễ tối ưu.** Điều này thấy rõ nhất khi huấn luyện mạng residual `1001` tầng (hình 1). Với thiết kế gốc, sai số giảm rất chậm lúc đầu, vì $f$ là `ReLU`, tín hiệu âm bị **cắt** (về 0), khiến mô hình khó xấp xỉ hàm mong muốn; còn trong mạng tiền kích hoạt, $f$ là ánh xạ đồng nhất, tín hiệu đi thẳng giữa các đơn vị. Mạng `1001` tầng của bài báo tối ưu rất nhanh và đạt sai số thấp nhất.

**Với mạng residual nông hơn (như 164 tầng ở hình 6 phải), việc $f$ là `ReLU` ảnh hưởng không đáng kể. Bài báo cho rằng sau một thời gian huấn luyện, trọng số được điều chỉnh sao cho đầu ra các đơn vị phần lớn không âm, khi đó $f$ không còn cắt tín hiệu. Nhưng hiện tượng cắt xảy ra thường xuyên ở mạng hơn `1000` tầng.**

![Hình 6](../images/resnetv2/Figure6.png)

Hình 7: Đường huấn luyện trên CIFAR-10. Trái: ResNet-110 với BN sau phép cộng (hình 4(b)) so với gốc. Phải: ResNet-164 tiền kích hoạt so với gốc (hình 6 của bài báo).

**Giảm quá khớp.** Ở hình `6` phải, mạng tiền kích hoạt có sai số huấn luyện cao hơn một chút nhưng sai số kiểm tra thấp hơn; bài báo cho rằng đó là nhờ hiệu ứng regularization của `BN`. Trong đơn vị residual gốc, dù BN chuẩn hoá tín hiệu, tín hiệu đó lập tức được cộng vào shortcut nên tín hiệu tổng **không** được chuẩn hoá, và **tín hiệu chưa chuẩn hoá này lại là đầu vào của tầng trọng số tiếp theo**. Ngược lại, trong phiên bản tiền kích hoạt, đầu vào của mọi tầng trọng số đều đã được chuẩn hoá.

## 5 Kết quả

Bảng `4` và `5` là kết quả của các mạng sâu trên các tập dữ liệu khác nhau. Mạng residual **sâu hơn** dùng đơn vị tiền kích hoạt (`ResNet v2`) đạt độ chính xác tốt nhất.

![Bảng 4](../images/resnetv2/table_4.png)

Hình 8: So sánh với các phương pháp khác trên CIFAR-10/100; ResNet-1001 tiền kích hoạt đạt sai số 4,62% trên CIFAR-10 (bảng 4 của bài báo).

![Bảng 5](../images/resnetv2/table_5.png)

Hình 9: Sai số trên tập kiểm định ImageNet của ResNet-152 và ResNet-200 gốc và tiền kích hoạt (bảng 5 của bài báo).

## 6 Kết luận

**Kết nối tắt dạng ánh xạ đồng nhất và tiền kích hoạt là yếu tố then chốt để tín hiệu lan truyền thông suốt trong mạng.**

## Bổ sung: tiền kích hoạt trong các kiến trúc về sau

Mục này do người biên soạn viết thêm.

Ý tưởng "giữ đường residual hoàn toàn sạch, mọi chuẩn hoá và phi tuyến chỉ nằm trên nhánh" của ResNet v2 chính là thiết kế **Pre-LN** của Transformer: $x_{l+1} = x_l + \mathrm{Attn}(\mathrm{LN}(x_l))$, thay vì Post-LN $x_{l+1} = \mathrm{LN}(x_l + \mathrm{Attn}(x_l))$ của Transformer gốc. Pre-LN huấn luyện ổn định hơn nhiều với mô hình sâu (không cần warmup dài), và là lựa chọn của GPT-2, GPT-3, Llama, ViT. Trong CNN thị giác, ResNet v1/v1.5 (hậu kích hoạt) vẫn phổ biến hơn v2 vì ở độ sâu 50–152 tầng khác biệt nhỏ; ConvNeXt dùng một khối gần với tiền kích hoạt (chuẩn hoá đặt trong nhánh, không có kích hoạt sau phép cộng).

## Tài liệu tham khảo

1. [[DL-Kiến trúc-Họ ResNet] 002 ResNet-v2](https://zhuanlan.zhihu.com/p/29678910)
2. [Identity Mappings in Deep Residual Networks (bản dịch)](https://blog.csdn.net/wspba/article/details/60750007)
3. He et al. [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027). ECCV 2016.
4. Xiong et al. [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745). ICML 2020.
