# Nền tảng học sâu: thuật toán tối ưu

Trang [Code](dlnote/dl_basic_code_vi) đi kèm tự cài SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam theo đúng công thức của bài, kiểm tra chúng khớp từng bước với `torch.optim`, và chạy cả nhóm trên một hàm có điều kiện xấu để thấy khác biệt.

<!-- theory: dl/ch12_vi ; qa/ch13_vi#13.19.1 -->

## Lời nói đầu

Thuật toán tối ưu cho mạng neuron sâu **là thuật toán cập nhật tham số của mạng để cực tiểu hoá hàm mất mát**. Chúng rất quan trọng: nếu khởi tạo tham số (điểm bắt đầu lặp) quyết định mô hình có hội tụ hay không, thì hiệu năng của thuật toán tối ưu ảnh hưởng **trực tiếp** tới hiệu quả huấn luyện.

Hiểu nguyên lý của các thuật toán và vai trò của siêu tham số giúp ta chỉnh bộ tối ưu hiệu quả hơn, nhờ đó cải thiện mô hình.

Thuật toán tối ưu trong bài này cụ thể là: tìm một bộ tham số $\theta$ của mạng làm giảm đáng kể hàm mất mát $J(\theta)$, thường gồm đánh giá hiệu năng trên toàn tập huấn luyện cộng thêm số hạng regularization.

> Bài này không phân biệt chặt chẽ hàm mất mát, hàm mục tiêu và hàm chi phí.

## 1 Thuật toán gradient descent

### 1.1 Stochastic gradient descent (SGD)

Gradient descent là nhóm bộ tối ưu cơ bản nhất, có ba dạng: gradient descent theo cả tập (batch gradient descent, BGD, mỗi bước dùng toàn bộ tập huấn luyện), stochastic gradient descent (SGD, mỗi bước một mẫu) và minibatch gradient descent (mỗi bước một batch nhỏ).

Trong các dự án học sâu, "`SGD`" thường mặc định là **minibatch** SGD. Thuật toán:

- Đầu vào và siêu tham số: $\eta$, tốc độ học toàn cục.
- Tính gradient (trên minibatch hiện tại): $g_t = \nabla_\theta J(\theta_{t-1})$.
- Cập nhật tham số: $\theta_t = \theta_{t-1} - \eta \cdot g_t$.

> **Biên tập:** Bản gốc gọi ba dạng là "gradient descent chuẩn (GD), stochastic (SGD) và batch gradient descent (BGD)", rồi nói SGD trong dự án "mặc định là batch gradient descent". BGD thường chỉ dạng dùng **cả** tập dữ liệu (chính là "GD chuẩn"); dạng dùng mặc định trong dự án là minibatch (MBGD), xem bài [Lan truyền ngược và gradient descent](dlnote/dl_bp_vi).

SGD là phương pháp tối ưu kinh điển nhất cho mạng neuron; **hội tụ chậm hơn nhưng khá ổn định**.

Hình 1 minh hoạ quỹ đạo tìm kiếm của SGD. Vì gradient mang tính ngẫu nhiên, quỹ đạo rất nhiễu (**hiện tượng dao động**).

![Quỹ đạo của SGD](../images/optimizers/sgd_algorithm.png)

Hình 1: Quỹ đạo SGD trên đường đồng mức của một hàm bậc hai dẹt: dao động mạnh theo phương dốc, tiến chậm theo phương thoải.

Vì vậy trong thực tế, SGD thường dùng cùng **lịch tốc độ học giảm dần**: với tốc độ học cố định, nhiễu của gradient ngẫu nhiên khiến tham số dao động mãi quanh cực tiểu.

### 1.2 Momentum

Dù **stochastic gradient descent** vẫn rất phổ biến, đôi khi nó học chậm, và hướng cập nhật hoàn toàn phụ thuộc gradient của `batch` hiện tại nên rất bất ổn, vì dữ liệu có nhiễu.

Lấy cảm hứng từ vật lý, SGD với **động lượng** (momentum, Polyak 1964) được dùng **để giảm dao động khi cập nhật tham số**. Momentum nhằm tăng tốc việc học, đặc biệt khi gặp độ cong lớn, gradient nhỏ nhưng nhất quán, hoặc gradient nhiễu. So sánh hai thuật toán ở hình 2.

> Sách *Deep Learning* giải thích mục đích của momentum mang tính học thuật hơn: giải quyết hai vấn đề là ma trận Hessian có điều kiện xấu và phương sai của gradient ngẫu nhiên.

![Hiệu quả của momentum](../images/optimizers/momentum_algorithm.png)

Hình 2: Trên một hàm bậc hai có điều kiện xấu, SGD thuần (mũi tên đen) nhảy qua lại giữa hai sườn thung lũng; SGD có momentum (đường đỏ) triệt tiêu dao động và đi thẳng hơn về cực tiểu (hình theo sách *Deep Learning*, hình 8.5).

Hiểu nôm na, momentum mô phỏng quán tính của vật chuyển động: khi cập nhật, kết hợp cả gradient trong quá khứ và gradient của batch hiện tại. Thuật toán giữ lại một phần hướng cập nhật trước, đồng thời dùng gradient của batch hiện tại để tinh chỉnh hướng cuối cùng. Nhờ vậy tăng độ ổn định, học nhanh hơn, và có phần nào khả năng thoát cực tiểu địa phương.

Hình 3 minh hoạ hướng đi của momentum.

![Hướng cập nhật của momentum](../images/optimizers/momentum_algorithm_update.png)

Hình 3: Hướng cập nhật của momentum.

Sau lần cập nhật đầu tiên, động lượng $v_1$ được ghi lại. Ở lần tính gradient thứ hai tại "điểm tính gradient", ta có hướng 2; **kết hợp** với $v_1$, hướng cập nhật cuối cùng là 2'. Nhờ có $v_1$, hướng cập nhật mang "quán tính", giảm dao động do mẫu ngẫu nhiên gây ra.

**Thuật toán momentum**:

1. **Đầu vào và tham số**:

   - $\eta$: tốc độ học toàn cục;
   - $\alpha$: hệ số động lượng, thường 0,5; 0,9 hay 0,99; bằng `0` thì thành SGD thường. Nó **điều khiển mức ảnh hưởng của động lượng tới cập nhật**;
   - $v_t$: động lượng tại thời điểm $t$, khởi đầu bằng 0.

2. **Các bước tính**:

   - Tính gradient: $g_t = \nabla_\theta J(\theta_{t-1})$
   - Cập nhật vận tốc: $v_t = \alpha \cdot v_{t-1} + \eta \cdot g_t$ (công thức 1)
   - Cập nhật tham số: $\theta_t = \theta_{t-1} - v_t$ (công thức 2)

> Lưu ý, công thức (1) và (2) hơi khác hình thức với sách *Deep Learning* (sách viết $v\leftarrow\alpha v-\eta g$, $\theta\leftarrow\theta+v$, tức đổi dấu $v$), nhưng kết quả như nhau. Khai triển dưới đây theo [AI-EDU 15.2 Thuật toán tối ưu gradient descent](https://github.com/microsoft/ai-edu/blob/master/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC7%E6%AD%A5%20-%20%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/15.2-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95.md).

Khai triển công thức lặp sẽ dễ hiểu hơn. Theo (1), (2), với tham số $W$ và $dW_k$ là gradient tại $W_k$:

1. $v_0 = 0$
2. $dW_0 = \nabla J(W_0)$
3. $v_1 = \alpha v_0 + \eta \cdot dW_0 = \eta \cdot dW_0$
4. $W_1 = W_0 - v_1=W_0 - \eta \cdot dW_0$
5. $dW_1 = \nabla J(W_1)$
6. $v_2 = \alpha v_1 + \eta\, dW_1$
7. $W_2 = W_1 - v_2 = W_1 - (\alpha v_1 +\eta\, dW_1) = W_1 - \alpha \eta\, dW_0 - \eta\, dW_1$
8. $dW_2 = \nabla J(W_2)$
9. $v_3=\alpha v_2 + \eta\, dW_2$
10. $W_3 = W_2 - v_3=W_2-(\alpha v_2 + \eta\, dW_2) = W_2 - \alpha^2 \eta\, dW_0 - \alpha \eta\, dW_1 - \eta\, dW_2$

So với SGD thường $W_3 = W_2 - \eta\, dW_2$, momentum không chỉ trừ gradient hiện tại mà còn trừ các gradient trước ($dW_0$, $dW_1$) nhân với hệ số giảm dần $\alpha^2$, $\alpha$. Vì $\alpha<1$ nên $\alpha^2<\alpha$, $\alpha^3<\alpha^2$... Cách cộng dồn này gọi là **trung bình trượt có trọng số mũ**. (Đúng ra là tổng có trọng số mũ: các hệ số $1+\alpha+\alpha^2+\dots\to\frac{1}{1-\alpha}$, nên với $\alpha=0{,}9$, bước đi ổn định gấp khoảng 10 lần $\eta g$ khi gradient không đổi.)

> **Biên tập:** Bản gốc ký hiệu mọi gradient là $\nabla J(w)$, không ghi tính tại điểm nào, và viết "trừ các gradient lịch sử $W_0$, $W_1$" (đó là tham số, không phải gradient); đã viết lại thành $dW_k=\nabla J(W_k)$.

Khi huấn luyện thật, SGD và momentum so sánh như bảng dưới.

> Hình so sánh lấy từ tài liệu tham khảo 1 (AI-EDU): cùng một mạng hồi quy, batch 10, $\eta=0{,}1$, khởi tạo Xavier.

| Thuật toán | Mất mát và độ chính xác |
| --- | --- |
| SGD | ![Mất mát khi huấn luyện bằng SGD](../images/optimizers/op_sgd_train_loss_log.png) |
| Momentum | ![Mất mát khi huấn luyện bằng momentum](../images/optimizers/op_momentum_train_loss_log.png) |

Từ bảng có thể thấy, với cùng một mô hình, đường mất mát của SGD thường dao động mạnh, và sau 10 000 epoch vẫn chưa đạt mức mất mát mục tiêu 0,001; còn momentum đạt mục tiêu và dừng sau khoảng 2 000 epoch. ("Độ chính xác" trong hình của AI-EDU là một chỉ số cho bài toán hồi quy, nên có thể âm.)

### 1.3 Momentum Nesterov

Lấy cảm hứng từ thuật toán gradient tăng tốc của Nesterov (Nesterov, 1983, 2004), Sutskever và cộng sự (2013) đề xuất một biến thể của momentum: momentum Nesterov (`NAG`, Nesterov Accelerated Gradient, hay Nesterov Momentum).

NAG thêm một bước hiệu chỉnh khi tính gradient: thay vì tính gradient tại vị trí hiện tại, nó tính tại vị trí mà động lượng **sắp** đưa tham số tới. Nói đơn giản là **thêm một hệ số hiệu chỉnh vào momentum chuẩn**.

Thuật toán NAG:

1. **Đầu vào và tham số**:

   - $\eta$: tốc độ học toàn cục;
   - $\alpha$: hệ số động lượng, mặc định 0,9;
   - $v$: động lượng, khởi đầu bằng 0.

2. **Các bước tính**:

   - Cập nhật tạm tham số: $\hat \theta = \theta_{t-1} - \alpha \cdot v_{t-1}$
   - Lan truyền xuôi tại điểm tạm: $f(\hat \theta)$
   - Tính gradient tại điểm tạm: $g_t = \nabla_{\hat\theta} J(\hat \theta)$
   - Cập nhật vận tốc: $v_t = \alpha \cdot v_{t-1} + \eta \cdot g_t$
   - Cập nhật tham số: $\theta_t = \theta_{t-1}  - v_t$

### 1.4 Thực hành với PyTorch

PyTorch gộp SGD thường, momentum và momentum Nesterov vào một lớp `torch.optim.SGD`. Công thức cập nhật của nó hơi khác các framework khác (đặt tốc độ học ở bước cập nhật tham số thay vì trong vận tốc); với $p$, $g$, $v$, $\mu$ lần lượt là tham số, gradient, vận tốc và hệ số động lượng:

$$
\begin{aligned}
v_{t+1} &= \mu \cdot v_{t} + g_{t+1} \\
p_{t+1} &= p_{t} - \text{lr} \cdot v_{t+1} \\
&= p_{t} - \text{lr} \cdot \mu \cdot v_{t} - \text{lr} \cdot g_{t+1}
\end{aligned}
$$

Khi tốc độ học không đổi, hai cách viết tương đương (vận tốc của PyTorch bằng vận tốc ở mục 1.2 chia cho lr); khi đổi tốc độ học giữa chừng thì chúng khác nhau một chút.

```python
# đã bỏ bớt các tham số ít dùng
class torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

1. **Chức năng**: cài đặt SGD, SGD có momentum, SGD có momentum Nesterov, tất cả đều có thể thêm weight decay.

2. **Tham số**:

   - `params` (iterable): nhóm tham số (xem khái niệm param group ở lớp cơ sở `Optimizer`), tức phần tham số mà bộ tối ưu quản lý.
   - `lr` (float): tốc độ học ban đầu; có thể thay đổi trong quá trình huấn luyện bằng lịch tốc độ học.
   - `momentum` (float): hệ số động lượng, thường 0,9 hay 0,8.
   - `weight_decay` (float): hệ số suy giảm trọng số, tức hệ số của số hạng regularization L2.
   - `nesterov` (bool): có dùng NAG hay không.

Cấu hình thường dùng khi huấn luyện:

```python
torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)
```

> **Biên tập:** Ví dụ gốc gọi `torch.optim.SGD(lr=0.02, momentum=0.9, weight_decay=0.0001)` không truyền tham số mô hình (báo `TypeError`). Trong chữ ký gốc `lr=required`; từ PyTorch 2.x giá trị mặc định là `1e-3`.

## 2 Thuật toán tốc độ học thích nghi

Từ lâu người nghiên cứu mạng neuron đã nhận ra tốc độ học là một trong những siêu tham số khó đặt nhất, vì nó ảnh hưởng rõ rệt tới hiệu năng mô hình. Các thuật toán dưới đây cho **mỗi tham số một tốc độ học riêng**, tự điều chỉnh theo lịch sử gradient của tham số đó.

### 2.1 AdaGrad

Trước AdaGrad (Duchi và cộng sự, 2011), mọi tham số dùng chung một tốc độ học. AdaGrad là thuật toán tốc độ học thích nghi đầu tiên: chia tốc độ học cho **căn bậc hai của tổng bình phương mọi gradient trong lịch sử**, nên độ dài bước giảm đơn điệu. Nó chỉnh tốc độ học trên từng chiều theo độ lớn gradient trên chiều đó, tránh việc một tốc độ học chung không hợp với mọi chiều.

Tham số nào nhận gradient lớn hoặc thường xuyên thì tổng tích luỹ lớn, tốc độ học giảm nhanh; tham số nào ít khi có gradient (ví dụ embedding của từ hiếm) thì tổng nhỏ, tốc độ học giữ lớn và mỗi lần có gradient được cập nhật mạnh. Vì vậy AdaGrad rất hợp với dữ liệu thưa.

> **Biên tập:** Bản gốc viết "tham số ít xuất hiện được cập nhật lớn (tốc độ học giảm nhanh)" và "tham số xuất hiện nhiều được cập nhật nhỏ (tốc độ học giảm tương đối chậm)": hai ngoặc đơn bị đảo. Tham số ít có gradient thì tốc độ học giảm **chậm**, nên cập nhật lớn.

Thuật toán AdaGrad:

1. **Đầu vào và tham số**:

   - $\eta$: tốc độ học toàn cục;
   - $\epsilon$: hằng số nhỏ để ổn định số học (PyTorch mặc định `1e-10`);
   - $r$: tổng tích luỹ, khởi đầu bằng 0.

2. **Các bước tính**:

   - Tính gradient: $g_t = \nabla_\theta J(\theta_{t-1})$
   - Tích luỹ bình phương gradient: $r_t = r_{t-1} + g_t \odot g_t$
   - Tính lượng cập nhật: $\Delta \theta = \dfrac{\eta}{\epsilon + \sqrt{r_t}} \odot g_t$ (khác hình thức với công thức trong *Dive into Deep Learning*, nơi $\epsilon$ đặt trong căn)
   - Cập nhật tham số: $\theta_t=\theta_{t-1} - \Delta \theta$

$\odot$ là nhân từng phần tử; phép khai căn, chia và nhân đều theo từng phần tử. Nhờ vậy mỗi phần tử của tham số có tốc độ học riêng.

**Tổng kết AdaGrad**: trong tối ưu lồi, AdaGrad có một số tính chất lý thuyết tốt. Nhưng thực nghiệm cho thấy, với mạng neuron sâu, tích luỹ bình phương gradient từ đầu huấn luyện làm tốc độ học hiệu dụng giảm quá sớm và quá nhiều. **AdaGrad tốt với một số mô hình học sâu, nhưng không phải tất cả.**

Bộ tối ưu AdaGrad trong PyTorch:

```python
class torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
```

### 2.2 RMSProp

`RMSProp` (Root Mean Square Propagation).

RMSProp (Hinton, 2012) khác AdaGrad ở chỗ: **thay tổng tích luỹ bằng trung bình trượt có trọng số mũ của bình phương gradient**, nên gradient cũ dần bị quên và tốc độ học không giảm mãi về 0.

Thuật toán RMSProp:

1. **Đầu vào và tham số**:

   - $\eta$: tốc độ học toàn cục, nên đặt 0,001;
   - $\epsilon$: hằng số nhỏ để ổn định số học, mặc định 1e-8;
   - $\alpha$: tốc độ phân rã, Hinton đề xuất 0,9 (PyTorch mặc định 0,99);
   - $r$: biến tích luỹ, cùng kích thước với $\theta$, khởi đầu bằng 0.

2. **Các bước tính** (tính gradient và cập nhật tham số như AdaGrad):

   - Trung bình trượt của bình phương gradient: $r_t = \alpha \cdot r_{t-1} + (1-\alpha)(g_t \odot g_t)$
   - Tính lượng cập nhật: $\Delta \theta = \dfrac{\eta}{\epsilon+\sqrt{r_t}} \odot g_t$

**Tổng kết RMSProp**: thực nghiệm cho thấy RMSProp là thuật toán tối ưu hiệu quả và thực dụng cho mạng sâu, và là một trong các bộ tối ưu hay được dùng.

> **Biên tập:** Bản gốc khuyên tốc độ học 0,001 ở phần tham số nhưng lại kết luận "tốc độ học ban đầu 0,01 là lý tưởng"; giá trị phổ biến là $10^{-3}$ (0,01 là mặc định của PyTorch nhưng thường quá lớn với mạng sâu). Công thức gốc đặt $\epsilon$ trong căn $\sqrt{r+\epsilon}$; PyTorch đặt ngoài căn như ở trên.

Bộ tối ưu RMSprop trong PyTorch:

```python
class torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```

### 2.3 AdaDelta

AdaDelta cũng là một cải tiến của AdaGrad, **nhằm giải quyết việc tốc độ học quá nhỏ ở giai đoạn sau của huấn luyện**. Thay vì cộng mọi bình phương gradient từ trước tới nay, AdaDelta chỉ xét gradient trong một cửa sổ thời gian gần đây (cài đặt bằng trung bình trượt có trọng số mũ, giống RMSProp).

Thuật toán AdaDelta:

1. **Tham số**:

   - **AdaDelta không có tốc độ học** theo nghĩa thông thường: nó dùng chính độ lớn của các bước cập nhật gần đây để đặt thang đo cho bước tiếp theo (`lr` trong PyTorch chỉ là hệ số nhân thêm, mặc định 1,0);
   - $\alpha$: tốc độ phân rã (`rho` trong PyTorch, mặc định 0,9);
   - $s$, $r$: hai biến tích luỹ, khởi đầu bằng 0.

2. **Các bước tính**:

   - Tính gradient: $g_t = \nabla_\theta J(\theta_{t-1})$
   - Trung bình trượt của bình phương gradient: $s_t = \alpha \cdot s_{t-1} + (1-\alpha) \cdot g_t \odot g_t$
   - Tính lượng cập nhật: $\Delta \theta = \sqrt{\dfrac{r_{t-1} + \epsilon}{s_t + \epsilon}} \odot g_t$
   - Cập nhật tham số: $\theta_t = \theta_{t-1} - \Delta \theta$
   - Cập nhật trung bình trượt của bình phương lượng cập nhật: $r_t = \alpha \cdot r_{t-1} + (1-\alpha) \cdot \Delta \theta \odot \Delta \theta$

> **Biên tập:** Bản gốc mở đầu phần các bước bằng "Thuật toán RMSProp như sau" (chép nhầm tên), ghi bước cập nhật tham số là "cập nhật gradient", và không khai báo biến $r$.

Bộ tối ưu Adadelta trong PyTorch:

```python
class torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
```

### 2.4 Adam

**Adam** (Adaptive Moment Estimation, Kingma và Ba, 2014) là một thuật toán tốc độ học thích nghi khác, có tác dụng như `RMSProp + Momentum`, tức **RMSProp có thêm động lượng**.

Trên nền RMSProp, Adam lấy thêm trung bình trượt có trọng số mũ của chính gradient. **Giống AdaGrad, RMSProp và AdaDelta, mỗi phần tử của tham số có tốc độ học riêng.**

Thuật toán Adam:

1. **Tham số**:

   - $t$: số bước hiện tại;
   - $\eta$: tốc độ học toàn cục, mặc định 0,001;
   - $\epsilon$: hằng số nhỏ để ổn định số học, mặc định 1e-8;
   - $\beta_1, \beta_2$: tốc độ phân rã của ước lượng mô-men, thuộc $[0,1)$, mặc định 0,9 và 0,999.

2. **Các bước tính**:

   - Tính gradient: $g_t = \nabla_\theta J(\theta_{t-1})$
   - Tăng bộ đếm: $t=t+1$
   - Cập nhật ước lượng mô-men bậc một (có chệch): $m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t$
   - Cập nhật ước lượng mô-men bậc hai (có chệch): $v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2)(g_t \odot g_t)$
   - Hiệu chỉnh độ chệch của mô-men bậc một: $\hat m_t = m_t / (1-\beta_1^t)$
   - Hiệu chỉnh độ chệch của mô-men bậc hai: $\hat v_t = v_t / (1-\beta_2^t)$
   - Tính lượng cập nhật: $\Delta \theta = \eta \cdot \hat m_t /(\epsilon + \sqrt{\hat v_t})$
   - Cập nhật tham số: $\theta_t=\theta_{t-1} - \Delta \theta$

Từ công thức có thể thấy Adam dùng **trung bình trượt có trọng số mũ** để ước lượng mô-men bậc một (động lượng) và bậc hai của gradient, qua hai biến trạng thái $m_t$, $v_t$.

**Hiểu Adam thế nào?**

Thứ nhất, trong Adam, động lượng được đưa thẳng vào ước lượng mô-men bậc một (có trọng số mũ) của gradient. Cách trực quan nhất để thêm động lượng vào RMSProp là áp động lượng lên gradient đã được co giãn; nhưng việc kết hợp động lượng với co giãn như vậy không có động cơ lý thuyết rõ ràng. Thứ hai, Adam có bước hiệu chỉnh độ chệch cho các ước lượng mô-men bậc một (động lượng) và bậc hai (không trừ trung bình), vốn được khởi tạo từ 0. RMSProp cũng dùng ước lượng mô-men bậc hai nhưng thiếu hệ số hiệu chỉnh, nên khác với Adam, ước lượng của RMSProp có thể chệch nhiều ở đầu quá trình huấn luyện. Adam thường được xem là khá bền vững với lựa chọn siêu tham số, dù đôi khi phải đổi tốc độ học khỏi giá trị gợi ý.

> Người mới chỉ cần xem công thức; đoạn trên tác giả chép từ sách *Deep Learning* (mục 8.5.3). Ví dụ cho hiệu chỉnh độ chệch: ở bước 1, $m_1=(1-\beta_1)g_1=0{,}1g_1$, nhỏ hơn gradient thật 10 lần; chia cho $1-\beta_1^1=0{,}1$ thì $\hat m_1=g_1$.

Thuật toán Adam trong sách *Deep Learning*:

```text
Thuật toán 8.7  Adam
Input:  ε                  # bước, gợi ý 0.001
        ρ1, ρ2 ∈ [0, 1)    # tốc độ phân rã của ước lượng mô-men, gợi ý 0.9 và 0.999
        δ                  # hằng số nhỏ cho ổn định số, gợi ý 1e-8
        θ                  # tham số khởi đầu
s ← 0, r ← 0               # biến mô-men bậc một và bậc hai
t ← 0
while chưa đạt điều kiện dừng do
    lấy minibatch {x^(1), ..., x^(m)} với nhãn y^(i)
    g ← (1/m) ∇θ Σ_i L(f(x^(i); θ), y^(i))
    t ← t + 1
    s ← ρ1·s + (1 − ρ1)·g           # ước lượng mô-men bậc một (có chệch)
    r ← ρ2·r + (1 − ρ2)·g ⊙ g       # ước lượng mô-men bậc hai (có chệch)
    ŝ ← s / (1 − ρ1^t)              # hiệu chỉnh độ chệch
    r̂ ← r / (1 − ρ2^t)
    Δθ ← −ε · ŝ / (√r̂ + δ)          # tính theo từng phần tử
    θ ← θ + Δθ
end while
```

> **Biên tập:** Bản gốc dán ảnh chụp thuật toán 8.7 từ sách *Deep Learning* bản tiếng Trung; đã gõ lại thành giả mã (ký hiệu theo sách: $\epsilon$ là bước, $\delta$ là hằng số ổn định).

**Tổng kết Adam**: Adam kế thừa RMSProp nên tốc độ học cũng không nên đặt quá cao; giá trị khởi đầu thường dùng là $10^{-3}$ (với Transformer cỡ lớn thường nhỏ hơn, $10^{-4}$ đến $3\times10^{-4}$, kèm warmup).

> **Biên tập:** Bản gốc viết "tốc độ học ban đầu 0,01 là lý tưởng" cho Adam, mâu thuẫn với giá trị mặc định 0,001 ghi ngay trong bài; 0,01 thường làm Adam mất ổn định với mạng sâu.

Bộ tối ưu Adam trong PyTorch:

```python
class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

## 3 Tổng kết

### 3.1 Các bộ tối ưu trong PyTorch

Mọi bộ tối ưu trong PyTorch kế thừa lớp cơ sở `Optimizer`, với 5 phương thức cơ bản:

- `zero_grad()`: đặt gradient về 0 (từ PyTorch 2.0 mặc định đặt thành `None` để tiết kiệm bộ nhớ).
- `step(closure)`: thực hiện một bước cập nhật tham số; có thể truyền một closure (bắt buộc với L-BFGS).
- `state_dict()`: trả về trạng thái **của bộ tối ưu** dưới dạng dict gồm `state` (bộ đệm động lượng, các mô-men của Adam... cho từng tham số) và `param_groups` (siêu tham số của từng nhóm).
- `load_state_dict(state_dict)`: nạp lại trạng thái đó, dùng khi tiếp tục huấn luyện từ checkpoint.
- `add_param_group(param_group)`: thêm một nhóm tham số mới, có thể đặt `lr`, `momentum`, `weight_decay` riêng; hay dùng khi fine-tuning (ví dụ tốc độ học nhỏ cho thân mạng, lớn cho đầu mới).

> **Biên tập:** Bản gốc mô tả `state_dict()` là "lấy tham số hiện tại của mô hình, key là tên tham số, value là tham số" và `load_state_dict` là "nạp tham số vào mạng hiện tại, hay dùng khi finetune". Đó là `state_dict()` của **mô hình** (`nn.Module`). `state_dict()` của bộ tối ưu chứa trạng thái nội bộ của nó; khi tiếp tục huấn luyện phải lưu cả hai, nếu không Adam sẽ bắt đầu lại với mô-men bằng 0.

Bản gốc (2021) viết PyTorch có mười bộ tối ưu; PyTorch 2.14 có 15: `SGD`, `ASGD`, `Rprop`, `RMSprop`, `Adagrad`, `Adadelta`, `Adam`, `AdamW`, `Adamax`, `NAdam`, `RAdam`, `SparseAdam`, `LBFGS`, `Adafactor`, `Muon`. Lưu ý, bộ tối ưu trong PyTorch có thể khác chút ít so với thuật toán trong bài báo gốc (như vị trí của $\epsilon$, cách tính weight decay); muốn chính xác thì đọc mã nguồn hoặc tài liệu của từng lớp.

### 3.2 Tổng kết các thuật toán tối ưu

- Các thuật toán tốc độ học thích nghi như Adam có lợi với dữ liệu thưa và hội tụ nhanh; nhưng SGD (+ momentum) được chỉnh kỹ thường cho kết quả cuối tốt hơn (với CNN cho thị giác máy tính).
- Với cùng số siêu tham số được chỉnh, đôi khi bộ tối ưu thích nghi có mất mát huấn luyện nhỏ hơn nhưng mất mát kiểm tra lại cao hơn họ SGD.
- Bộ tối ưu thích nghi hội tụ nhanh hơn trên tập huấn luyện ở giai đoạn đầu, nhưng có thể tổng quát hoá kém hơn trên tập kiểm tra.
- Các thuật toán phổ biến nhất gồm SGD, SGD có momentum, RMSProp, RMSProp có momentum, AdaDelta và Adam. Chọn thuật toán nào phần lớn phụ thuộc vào việc người dùng quen thuộc thuật toán nào (để dễ chỉnh siêu tham số) (theo sách *Deep Learning*, mục 8.5.4).

## 4 Bổ sung: bộ tối ưu sau Adam

Mục này do người biên soạn viết thêm; chi tiết hơn ở mục 13.19.1 của [chương 13 khoá Ôn tập](qa/ch13_vi).

- **AdamW** (Loshchilov và Hutter, 2019): với Adam, cộng $\lambda\theta$ vào gradient (tham số `weight_decay` của `torch.optim.Adam`) **không** tương đương weight decay, vì số hạng này bị chia cho $\sqrt{\hat v}$. AdamW trừ thẳng $\eta\lambda\theta$ vào tham số. Đây là bộ tối ưu mặc định để huấn luyện Transformer và LLM.
- **Lịch tốc độ học**: warmup tuyến tính vài trăm đến vài nghìn bước đầu (tránh bước quá lớn khi ước lượng mô-men còn nhiễu), rồi giảm theo hàm cosine; xem phần mẹo huấn luyện của khoá.
- **Lion** (Chen và cộng sự, 2023): chỉ dùng **dấu** của tổ hợp động lượng, mỗi tham số cập nhật đúng $\pm\eta$; tiết kiệm bộ nhớ hơn Adam (một biến trạng thái thay vì hai), cần tốc độ học nhỏ hơn khoảng 3–10 lần.
- **Muon** (Jordan và cộng sự, 2024): với ma trận trọng số 2D, trực giao hoá động lượng (lặp Newton–Schulz) trước khi cập nhật; huấn luyện LLM nhanh hơn AdamW ở nhiều thí nghiệm, có sẵn trong PyTorch (`torch.optim.Muon`, có trong bản 2.14 dùng ở trang Code).
- **Tiết kiệm bộ nhớ**: Adam giữ hai bản trạng thái bằng kích thước mô hình. Adafactor phân tích mô-men bậc hai thành tích hàng và cột; Adam 8 bit (bitsandbytes) lượng tử hoá trạng thái; ZeRO chia trạng thái bộ tối ưu giữa các GPU.

## Tài liệu tham khảo

1. [AI-EDU 15.2 Thuật toán tối ưu gradient descent](https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC7%E6%AD%A5%20-%20%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/15.2-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95.html) (sách *Cửa ngõ trí tuệ*)
2. Goodfellow, Bengio, Courville. *Deep Learning*, chương 8 Tối ưu cho huấn luyện mô hình sâu.
3. *Dive into Deep Learning*, chương 12 Thuật toán tối ưu; trên site: [Học sâu, chương 12](dl/ch12_vi).
4. Loshchilov, Hutter. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). ICLR 2019.
5. Chen et al. [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675) (Lion). 2023.
