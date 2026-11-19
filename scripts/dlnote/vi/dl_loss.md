# Nền tảng học sâu: hàm mất mát

Trang [Code](dlnote/dl_basic_code_vi) đi kèm tính lại mọi con số của bài (entropy, hai mô hình ở mục 2.3, softmax), so `nn.CrossEntropyLoss` với `LogSoftmax + NLLLoss`, cho thấy vì sao không nên dùng `sigmoid + BCELoss` và vì sao MSE học chậm khi phân loại.

<!-- theory: qa/ch13_vi#13.6 ; qa/ch03_vi#3.4.11 -->

## 1 Tổng quan về hàm mất mát

Phần lớn thuật toán học sâu có một dạng tối ưu nào đó. **Tối ưu là việc thay đổi $x$ để cực tiểu hoá hoặc cực đại hoá một hàm $f(x)$**; ta thường quy mọi bài toán tối ưu về cực tiểu hoá $f(x)$.

Trong học máy, hàm mất mát là một phần của hàm chi phí, còn hàm chi phí là một loại hàm mục tiêu:

- **Hàm mất mát** (`loss function`): sai số giữa dự đoán và giá trị thật trên **một** mẫu huấn luyện.
- **Hàm chi phí** (`cost function`): sai số tích luỹ (thường là trung bình) trên một batch hay cả tập huấn luyện.
- **Hàm mục tiêu** (`objective function`): bất kỳ hàm nào được tối ưu (có thể gồm cả hàm chi phí và số hạng regularization).

**Định nghĩa**: hàm mất mát là hàm thực không âm, định lượng khác biệt giữa dự đoán của mô hình và nhãn thật; nó gắn chặt với thuật toán tối ưu. Bước đầu tiên khi tối ưu một thuật toán học sâu là chọn dạng hàm mất mát.

Hàm mất mát chia đại khái thành hai loại: mất mát hồi quy (cho biến liên tục) và mất mát phân loại (cho biến rời rạc). Vị trí của nó trong quy trình thực nghiệm học sâu như hình dưới.

![Quy trình thực nghiệm học sâu](../images/loss/define_loss.png)

Hình 1: Định nghĩa mạng, hàm mất mát và thuật toán tối ưu, rồi huấn luyện, kiểm định, kiểm tra (nguồn: khoá Machine Learning 2022 của Lý Hoằng Nghị, [Machine Learning Pytorch Tutorial](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2022-course-data/Pytorch%20Tutorial%201.pdf)).

Thuật toán tối ưu hay dùng để giảm hàm mất mát là gradient descent.

## 2 Cross-entropy: mất mát cho phân loại

Mất mát cross-entropy (`Cross-Entropy Loss`) còn gọi là mất mát log-likelihood (log-likelihood loss), log loss; với phân loại nhị phân còn gọi là mất mát logistic (logistic loss).

### 2.1 Nguồn gốc của cross-entropy

> Phần này dựa theo [AI-EDU: Hàm mất mát cross-entropy](https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/). Các khái niệm này được trình bày đầy đủ hơn ở mục 3.13 bài [Xác suất và lý thuyết thông tin](dlnote/math_prob_vi).

**1. Lượng tin**

Trong lý thuyết thông tin, lượng tin của một sự kiện là:

> Sách *Deep Learning* gọi là lượng tin riêng (self-information). Trong bài này $\log$ luôn là logarit tự nhiên, **cơ số** $e$.

$$
I(x_j) = -\log p(x_j)
$$

- $x_j$: một sự kiện;
- $p(x_j)$: xác suất xảy ra sự kiện $x_j$;
- $I(x_j)$: lượng tin; $x_j$ càng khó xảy ra thì khi nó xảy ra, lượng tin càng lớn.

**2. Entropy**

Lượng tin chỉ xét một kết quả. Có thể dùng entropy (entropy Shannon, `Shannon entropy`) để định lượng tổng độ không chắc chắn của cả phân phối:

$$
H(p) = - \sum_{j=1}^n p(x_j) \log p(x_j)
$$

Ví dụ, một biến có ba kết quả với xác suất $p = (0{,}7;\ 0{,}2;\ 0{,}1)$ có entropy:

$$
\begin{aligned} H(p)&=-[p(x_1) \ln p(x_1) + p(x_2) \ln p(x_2) + p(x_3) \ln p(x_3)] \\ &=0{,}7 \times 0{,}357 + 0{,}2 \times 1{,}609 + 0{,}1 \times 2{,}303 \\ &\approx 0{,}802 \end{aligned}
$$

> **Biên tập:** Bản gốc viết "entropy của bài toán ở trên" nhưng không có bài toán nào ở trên (ví dụ bị cắt khi chép từ AI-EDU); đã thêm phân phối $p$. Bản gốc làm tròn $-\ln 0{,}7$ thành 0,36 nên được 0,804; giá trị chính xác là 0,8018.

**3. Entropy tương đối (KL divergence)**

Entropy tương đối còn gọi là KL divergence. Nếu cùng một biến ngẫu nhiên $x$ có hai phân phối $P(x)$ và $Q(x)$, có thể dùng KL divergence (Kullback-Leibler divergence) để **đo khác biệt giữa hai phân phối**:

$$
D_{KL}(p\parallel q)=\sum_{j=1}^m p(x_j) \log {p(x_j) \over q(x_j)}
$$

$m$ là số kết quả có thể (trong phân loại là số lớp). **$D_{KL}$ càng nhỏ, $q$ càng gần $p$**; nó bằng 0 khi và chỉ khi hai phân phối trùng nhau.

> **Biên tập:** Bản gốc ví KL divergence "tương đương sai số bình phương trung bình trong lý thuyết thông tin". So sánh này chỉ đúng theo nghĩa "đều đo khác biệt"; KL không đối xứng và không phải khoảng cách.

**4. Cross-entropy**

Biến đổi công thức KL:

$$
\begin{aligned} D_{KL}(p\parallel q)&=\sum_{j=1}^m p(x_j) \log {p(x_j)} - \sum_{j=1}^m p(x_j) \log q(x_j) \\ &=- H(p) + H(p,q) \end{aligned}
$$

Phần trước chính là (số đối của) entropy của $p$; phần sau là **cross-entropy** (trong học máy, $p$ là phân phối thật/mục tiêu, $q$ là phân phối dự đoán):

$$
H(p,q) =- \sum_{j=1}^m p(x_j) \log q(x_j)
$$

Trong học máy, ta cần đánh giá khác biệt giữa **nhãn $y$ và dự đoán $a$** (tức độ giống nhau giữa hai phân phối); dùng KL divergence $D_{KL}(y\parallel a)$ là được. Nhưng phân phối của nhãn cố định, tức $H(y)$ không đổi, nên khi tối ưu chỉ cần quan tâm cross-entropy. Vì vậy **học máy thường dùng thẳng cross-entropy làm hàm mất mát**:

$$
\text{loss} = -\sum_{j = 1}^{m}y_{j}\log a_{j}
$$

> **Biên tập:** Bản gốc viết "$H(a)$ không đổi"; phần cố định là entropy của **nhãn** $H(y)$, còn $a$ là dự đoán thay đổi khi huấn luyện. Công thức loss của bản gốc thiếu dấu trừ (khi đó loss luôn âm và cực tiểu hoá nó là sai hướng).

Công thức trên cho một mẫu; $m$ **không phải số mẫu mà là số lớp**. **Cross-entropy cho một batch** (rất quan trọng!):

$$
J = -\frac{1}{n}\sum_{i=1}^n \sum_{j=1}^{m} y_{ij} \log a_{ij}
$$

với $n$ là số mẫu và $m$ là số lớp.

> Công thức theo [AI-EDU: Hàm mất mát cross-entropy](https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC1%E6%AD%A5%20-%20%E5%9F%BA%E6%9C%AC%E7%9F%A5%E8%AF%86/03.2-%E4%BA%A4%E5%8F%89%E7%86%B5%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.html), đổi ký hiệu số mẫu thành $n$, số lớp thành $m$.

Một trường hợp đặc biệt là sự kiện chỉ có hai khả năng, như "là chó" và "không phải chó", gọi là phân loại $0/1$ hay **phân loại nhị phân**. Khi đó $m=2$, $y_1=1-y_2$, $a_1=1-a_2$, nên **cross-entropy của một mẫu trong phân loại nhị phân** rút gọn thành:

$$
\text{loss} =-[y \log a + (1-y) \log (1-a)]
$$

và **cho một batch**:

$$
J= -\frac{1}{n} \sum_{i=1}^n [y_i \log a_i + (1-y_i) \log (1-a_i)]
$$

> Vì sao hàm chi phí cross-entropy lấy trung bình chứ không lấy tổng? Cross-entropy được định nghĩa là một **kỳ vọng** theo phân phối của biến ngẫu nhiên, và trung bình mẫu là ước lượng của kỳ vọng (xem [d2l, mục Information Theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html#cross-entropy)). Về mặt thực hành, lấy trung bình giữ độ lớn của gradient không đổi khi đổi batch size, nên không phải chỉnh lại tốc độ học.

#### 2.1.1 Tóm tắt entropy, entropy tương đối và cross-entropy

> Cross-entropy $H(p, q)$ còn được viết $CE(p, q)$ hay $H(P, Q)$, và có một cách viết khác (cùng ý nghĩa):
> $$H(P, Q)  = -\mathbb{E}_{\textrm{x}\sim P}\log Q(x)$$

Cross-entropy thường dùng cho hồi quy logistic (`logistic regression`), tức bài toán phân loại.

Gộp ba công thức:

$$
\begin{aligned}
H(p) &= -\sum_{j}p(x_j) \log p(x_j) \\
D_{KL}(p \parallel q) &= \sum_{j}p(x_j)\log \frac{p(x_j)}{q(x_j)} = \sum_j \big(p(x_j)\log p(x_j) - p(x_j) \log q(x_j)\big) \\
H(p,q) &=  -\sum_j p(x_j)\log q(x_j) = H(p) + D_{KL}(p\parallel q)
\end{aligned}
$$

### 2.2 Cross-entropy cho phân loại nhị phân

Tách công thức cross-entropy nhị phân thành hai trường hợp:

- Khi $y=1$ (mẫu dương), số hạng sau dấu cộng bằng 0: $\text{loss} = -\log a$.
- Khi $y=0$ (mẫu âm), số hạng trước dấu cộng bằng 0: $\text{loss} = -\log (1-a)$.

Hình dưới, trục hoành là đầu ra dự đoán $a$, trục tung là giá trị mất mát. Với $y=1$, dự đoán càng gần 1 thì mất mát càng nhỏ; càng gần 0 thì mất mát càng lớn (tiến tới vô cùng). Với $y=0$ thì ngược lại.

![Cross-entropy nhị phân](../images/loss/binary_cross_entropy_loss_function_diagram.png)

Hình 2: Mất mát cross-entropy nhị phân theo dự đoán $a$, với nhãn $y=1$ (cam) và $y=0$ (xanh).

### 2.3 Cross-entropy cho phân loại nhiều lớp

Khi nhãn không chỉ là 0 hay 1, đó là bài toán nhiều lớp.

Giả sử ta muốn dựa vào đường viền, màu sắc... trong ảnh để đoán loài vật, với ba lớp: mèo, chó, lợn. Hai mô hình đã huấn luyện cho các dự đoán sau:

**Mô hình 1**:

| Dự đoán (mèo, chó, lợn) | Nhãn | Đúng không |
| --- | --- | --- |
| 0,3  0,3  0,4 | 0 0 1 (lợn) | Đúng |
| 0,3  0,4  0,3 | 0 1 0 (chó) | Đúng |
| 0,1  0,2  0,7 | 1 0 0 (mèo) | Sai |

Mỗi hàng là một mẫu, tổng cộng 3 mẫu. Mô hình 1 đoán đúng mẫu 1 và 2 với ưu thế rất mỏng, và sai hẳn mẫu 3.

**Mô hình 2**:

| Dự đoán (mèo, chó, lợn) | Nhãn | Đúng không |
| --- | --- | --- |
| 0,1  0,2  0,7 | 0 0 1 (lợn) | Đúng |
| 0,1  0,7  0,2 | 0 1 0 (chó) | Đúng |
| 0,3  0,4  0,3 | 1 0 0 (mèo) | Sai |

Mô hình 2 đoán mẫu 1 và 2 rất chắc chắn (xác suất gần 1 hơn), còn mẫu 3 tuy sai nhưng không sai quá xa.

> **Biên tập:** Bản gốc ghi dự đoán thứ hai của mô hình 1 và thứ ba của mô hình 2 là "0,3 0,4 0,4": tổng bằng 1,1 (không phải phân phối xác suất) và có hai lớp bằng nhau nên không xác định được "đúng". Bảng gốc của bài Zhihu được trích là "0,3 0,4 0,3"; các giá trị mất mát bên dưới không đổi.

Theo công thức cross-entropy nhiều lớp, mất mát của mô hình 1:

$$
\begin{aligned}
\text{mẫu 1:}\quad & -(0\times \ln 0{,}3 + 0\times \ln 0{,}3 + 1\times \ln 0{,}4) = 0{,}91 \\
\text{mẫu 2:}\quad & -(0\times \ln 0{,}3 + 1\times \ln 0{,}4 + 0\times \ln 0{,}3) = 0{,}91 \\
\text{mẫu 3:}\quad & -(1\times \ln 0{,}1 + 0\times \ln 0{,}2 + 0\times \ln 0{,}7) = 2{,}30
\end{aligned}
$$

Lấy trung bình trên các mẫu:

$$
L = \frac{0{,}91 + 0{,}91 + 2{,}30}{3} = 1{,}37
$$

Mất mát của mô hình 2:

$$
\begin{aligned}
\text{mẫu 1:}\quad & -(0\times \ln 0{,}1 + 0\times \ln 0{,}2 + 1\times \ln 0{,}7) = 0{,}36 \\
\text{mẫu 2:}\quad & -(0\times \ln 0{,}1 + 1\times \ln 0{,}7 + 0\times \ln 0{,}2) = 0{,}36 \\
\text{mẫu 3:}\quad & -(1\times \ln 0{,}3 + 0\times \ln 0{,}4 + 0\times \ln 0{,}3) = 1{,}20
\end{aligned}
$$

Lấy trung bình:

$$
L = \frac{0{,}36 + 0{,}36 + 1{,}20}{3} = 0{,}64
$$

> **Biên tập:** Bản gốc đặt nhãn "sample 1" cho cả ba dòng của mỗi mô hình, và làm tròn $-\ln 0{,}7=0{,}357$ xuống 0,35 nên được trung bình 0,63; tính chính xác là 0,639.

0,64 nhỏ hơn nhiều so với 1,37: dự đoán càng gần nhãn thật thì mất mát càng nhỏ, tức cross-entropy nắm bắt được khác biệt về chất lượng giữa hai mô hình, dù cả hai đều có độ chính xác 2/3. **Mất mát cross-entropy càng nhỏ thì tín hiệu lan truyền ngược càng nhỏ**: với softmax + cross-entropy, gradient theo logit đúng bằng $a - y$.

> Ví dụ tính cross-entropy nhiều lớp lấy từ bài Zhihu [损失函数｜交叉熵损失函数](https://zhuanlan.zhihu.com/p/35709485).

### 2.4 Cross-entropy trong PyTorch

Hàm mất mát cross-entropy hay dùng trong PyTorch là `torch.nn.CrossEntropyLoss`:

```python
class torch.nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)
```

**1. Chức năng**:

Nhận **logit** (đầu ra chưa qua softmax), áp `log_softmax` rồi tính cross-entropy với `target`; tức gộp `nn.LogSoftmax()` và `nn.NLLLoss()`. Nhãn `target` có thể là chỉ số lớp (kiểu `long`, kích thước $(N)$) hoặc xác suất cho từng lớp (kiểu `float`, kích thước $(N, C)$, ví dụ nhãn mềm).

**2. Tham số**:

- `weight` (Tensor): trọng số cho mất mát của từng lớp, hay dùng khi lớp mất cân bằng. Phải là tensor `float` có độ dài bằng số lớp `C`.
- `ignore_index` (int): bỏ qua một giá trị nhãn: mẫu có nhãn đó có mất mát 0 và không được tính vào mẫu số khi lấy trung bình (hay dùng cho token đệm trong NLP).
- `reduction` (str): `'mean'` (mặc định) lấy trung bình, `'sum'` lấy tổng, `'none'` trả về mất mát của từng mẫu.
- `label_smoothing` (float): làm mượt nhãn, trộn nhãn one-hot với phân phối đều theo tỉ lệ này (thêm từ PyTorch 1.10).

> **Biên tập:** Bản gốc liệt kê `size_average` và `reduce` và mặc định `reduction='elementwise_mean'`; đây là API của PyTorch 0.4 (2018). `size_average` và `reduce` đã bị đánh dấu lỗi thời từ lâu, chỉ còn giữ để tương thích, thay bằng `reduction`; giá trị `'elementwise_mean'` nay là `'mean'`. Bản gốc cũng nói "cross-entropy đúng nghĩa là `nn.NLLLoss()`": `NLLLoss` chỉ lấy $-\log q$ tại lớp đúng từ **log-xác suất** đã tính sẵn, phải đi sau `LogSoftmax` mới thành cross-entropy.

#### 2.4.1 Hàm softmax cho nhiều lớp

> Lưu ý: softmax ở tầng cuối thường đi kèm hàm mất mát cross-entropy, dùng cho phân loại nhiều lớp.

Với phân loại nhị phân, ta dùng hàm `Logistic` để tính xác suất và chia mẫu thành hai lớp dương, âm. Với nhiều lớp, dùng `Softmax` ở tầng cuối để **biến đầu ra thành phân phối xác suất, mỗi giá trị trong $[0, 1]$ và tổng bằng 1**.

Tên Softmax gồm hai phần soft và max. Max là lấy giá trị lớn nhất. Cốt lõi của Softmax là "soft" (mềm), đối lập với "hard" (cứng), tức hardmax. Dưới đây so sánh lấy max và dùng softmax trên đầu ra của mô hình.

**Lấy max (hardmax)**

Giả sử mô hình cho $\boldsymbol z = [3, 1, -3]$. Lấy max được $[1, 0, 0]$, đúng nhu cầu phân loại: tổng bằng 1 và mẫu thuộc lớp thứ nhất. Nhưng có hai nhược điểm:

1. Kết quả $[1,0,0]$ chỉ giữ thông tin 0 hay 1, trắng đen rõ ràng, mất thông tin các phần tử chênh nhau bao nhiêu; có thể hiểu là "hard max";
2. Phép max không khả vi (theo nghĩa hữu ích: đạo hàm bằng 0 hầu khắp nơi), không dùng được cho lan truyền ngược.

**Dùng softmax**

`Softmax` thêm chữ "soft" để mô phỏng hành vi của max, nhưng vẫn giữ thông tin về độ lớn tương đối:

$$
a_j = \operatorname{Softmax}(\boldsymbol z)_j = \frac{e^{z_j}}{\sum_{i=1}^m e^{z_i}}=\frac{e^{z_j}}{e^{z_1}+e^{z_2}+\dots+e^{z_m}}
$$

trong đó:

- $z_j$ là giá trị gốc (logit) của lớp thứ $j$, tức kết quả phép nhân ma trận;
- $z_i$ là giá trị gốc của từng lớp tham gia;
- $m$ là số lớp;
- $a_j$ là kết quả cho lớp thứ $j$.

Khác với hardmax, softmax không chọn duy nhất một giá trị lớn nhất mà gán cho mỗi lớp một xác suất (độ tin cậy), thể hiện khả năng thuộc lớp đó.

Hình dưới minh hoạ cách tính softmax.

![Cách softmax tính](../images/loss/softmax_process.png)

Với đầu vào $[z_1,z_2,z_3] = [3, 1, -3]$, tính theo hình được phân phối $[0{,}879;\ 0{,}119;\ 0{,}002]$. So sánh max và softmax:

| Giá trị gốc | Max | Softmax |
| --- | --- | --- |
| $[3, 1, -3]$ | $[1, 0, 0]$ | $[0{,}879;\ 0{,}119;\ 0{,}002]$ |

Kết quả softmax có hai đặc điểm:

1. Xác suất của ba lớp cộng lại bằng 1;
2. Xác suất của mỗi lớp đều lớn hơn 0.

Code hardmax và softmax:

```python
import numpy as np

data = np.array([3, 1, -3])

def hardmax(z):
    """Chỉ số của phần tử lớn nhất."""
    return np.argmax(z)

def softmax(z):
    e = np.exp(z - np.max(z))     # trừ max để tránh tràn số, kết quả không đổi
    return e / e.sum()

print(hardmax(data))              # 0
p = softmax(data)
print(p)                          # [0.87887824 0.11894324 0.00217852]
print(p.sum())                    # 1.0
```

Kết quả khớp với tính tay. Trừ $\max_i z_i$ trước khi lấy mũ không đổi kết quả (nhân cả tử và mẫu với $e^{-\max z}$) nhưng tránh tràn số: với $\boldsymbol z=[1000, 0]$, `np.exp(1000)` là `inf` và softmax không trừ max cho `nan`.

Trong PyTorch, softmax theo từng hàng cho batch có thể viết:

```python
def softmax(x):
    e = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return e / e.sum(dim=1, keepdim=True)      # keepdim giữ kích thước (N, 1) để chia theo hàng
```

nhưng thực tế nên dùng `torch.softmax(x, dim=1)`, hoặc tốt hơn là đưa thẳng logit vào `nn.CrossEntropyLoss`.

> **Biên tập:** Code softmax của bản gốc không trừ max nên tràn số với logit lớn. Bản PyTorch gốc dùng `.view(-1, 1)` "để tránh broadcast"; thực ra chính `.view(-1, 1)` (hay `keepdim=True`) là để **broadcast đúng** khi chia mỗi hàng cho tổng của nó.

### 2.5 Vì sao không dùng sai số bình phương cho bài toán phân loại?

Bài toán hồi quy thường dùng sai số bình phương; với mô hình tuyến tính, hàm mất mát khi đó lồi theo tham số, nên tìm được nghiệm tối ưu. Còn với phân loại, đầu ra phải đi qua `Sigmoid` hay `Softmax`; ghép với sai số bình phương thì hàm mất mát **không còn lồi** theo tham số (ngay cả với hồi quy logistic), nên khó tìm nghiệm tối ưu. Ghép với cross-entropy thì hồi quy logistic có hàm mất mát lồi.

Lý do quan trọng hơn là gradient. Với sigmoid $a=\sigma(z)$:

- MSE: $\frac{\partial}{\partial z}\frac12(a-y)^2 = (a-y)\,\sigma'(z)$. Khi mô hình sai rất xa (ví dụ $y=1$ mà $z\ll0$), $\sigma'(z)\approx 0$ nên gradient gần như bằng 0: mô hình sai nhiều nhất lại học chậm nhất.
- Cross-entropy: $\frac{\partial}{\partial z}\big[-y\log a-(1-y)\log(1-a)\big] = a-y$. Sai càng nhiều, gradient càng lớn; chỉ một phép trừ là có sai số lan truyền ngược.

> **Biên tập:** Bản gốc nói "cross-entropy đảm bảo đơn điệu trong khoảng", một cách nói mơ hồ; phần trên thay bằng hai lý do cụ thể (tính lồi và độ lớn gradient), trang Code minh hoạ lý do thứ hai.

## 3 Mất mát cho hồi quy

Khác với phân loại, hồi quy là **dự đoán một giá trị số cụ thể**. Mạng giải bài toán hồi quy thường chỉ có một nút đầu ra (hoặc $M$ nút nếu nhãn là vector $M$ chiều), và đầu ra chính là giá trị dự đoán.

Một khái niệm cơ bản của hồi quy là **phần dư** hay **sai số dự đoán**, đo độ gần giữa dự đoán và nhãn thật. Giả sử với đặc trưng đầu vào $x_i$ của mẫu thứ $i$, **nhãn** là $\boldsymbol y^i = (y_1,y_2,\dots,y_M)^{\top}$ với $M$ là số chiều của nhãn, và dự đoán của mạng là $\hat{\boldsymbol y}^i$. Sai số dự đoán (phần dư) ở chiều thứ $t$ là:

$$
l_{t}^{i} = y_{t}^{i} - \hat{y}_{t}^{i}
$$

Hai hàm mất mát hay dùng là $\text{MAE}$ (còn gọi là mất mát `L1`) và $\text{MSE}$ (còn gọi là mất mát `L2`).

### 3.1 Mất mát MAE

Sai số tuyệt đối trung bình (Mean Absolute Error, `MAE`) là một trong những hàm mất mát hồi quy đơn giản nhưng mạnh nhất.

Dữ liệu hồi quy có thể có ngoại lai (giá trị khác hẳn phần còn lại), khiến biến mục tiêu không thực sự theo phân phối Gauss. Khi đó MAE là lựa chọn tốt, vì mất mát của nó chỉ tăng **tuyến tính** theo sai số, nên một vài điểm ngoại lai không chi phối được toàn bộ hàm mất mát như với bình phương.

> **Biên tập:** Bản gốc giải thích MAE hợp với dữ liệu có ngoại lai "vì nó không xét hướng của ngoại lai (giá trị dương hay âm cao bất thường)". MSE cũng không xét hướng (bình phương bỏ dấu); lý do đúng là MAE tăng tuyến tính, MSE tăng bậc hai.

Đúng như tên gọi, MAE là **trung bình trị tuyệt đối của hiệu giữa nhãn và dự đoán**:

$$
\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{M} |y_{t}^{i} - \hat{y}_{t}^{i}|
$$

với $N$ là số mẫu.

### 3.2 Mất mát MSE

Sai số bình phương trung bình (Mean Square Error, `MSE`) gần như là lựa chọn mặc định cho hồi quy, vì **nhiều biến có thể mô hình hoá bằng phân phối Gauss**: cực tiểu hoá MSE tương đương ước lượng hợp lý cực đại với nhiễu Gauss (mục 5.5 bài [Nguyên lý cơ bản của học máy](dlnote/math_ml_vi)).

MSE là **trung bình bình phương khoảng cách giữa dự đoán và nhãn thật**; dự đoán càng gần nhãn, MSE càng nhỏ:

$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{M} (y_{t}^{i} - \hat{y}_{t}^{i})^2
$$

### 3.3 Mất mát Huber

So sánh MAE và MSE:

1. **MAE bền vững hơn MSE với ngoại lai.** Nhìn công thức: khi chênh lệch giữa dự đoán và nhãn lớn, MSE khuếch đại nó (bình phương). Vì MSE bị ngoại lai chi phối nên MAE bền vững hơn.
2. **MAE kém ổn định hơn MSE.** Nghiệm tối ưu của MAE là trung vị, có thể nhảy khi dữ liệu thay đổi một chút; qua nhiều vòng lặp, độ dốc của đường hồi quy có thể dao động. MSE cho phép đường hồi quy chỉnh nhẹ theo từng điểm dữ liệu.
3. **Gradient của MAE luôn có độ lớn không đổi**, kể cả khi mất mát đã rất nhỏ, bất lợi cho việc học. Có thể khắc phục bằng tốc độ học giảm dần khi gần cực tiểu.
4. **Gradient của MSE tăng theo mất mát và giảm về 0 khi mất mát về 0**, nên dùng tốc độ học cố định cũng hội tụ tốt.

Mất mát Huber kết hợp tính bền vững của MAE và tính ổn định của MSE, lấy cái tốt của cả hai: **tuyến tính với sai số lớn, bậc hai với sai số nhỏ**.

Huber có tham số $\delta$. Khi $\vert y - \hat{y} \vert \le \delta$ nó là mất mát bình phương, khi lớn hơn $\delta$ thì giống mất mát tuyệt đối, nên khá bền vững:

$$
\text{Huber} = \begin{cases}
\frac12\big(y_{t}^{i} - \hat{y}_{t}^{i}\big)^2 & |y_{t}^{i} - \hat{y}_{t}^{i}| \leq \delta \\
\delta|y_{t}^{i} - \hat{y}_{t}^{i}| - \frac12\delta^2 & |y_{t}^{i} - \hat{y}_{t}^{i}| > \delta
\end{cases}
$$

Hai nhánh khớp nhau cả về giá trị lẫn đạo hàm tại $|r|=\delta$, nên Huber khả vi liên tục. Trong PyTorch là `nn.HuberLoss(delta=1.0)`; `nn.SmoothL1Loss(beta)` bằng Huber chia cho $\beta$.

So sánh ba hàm mất mát hồi quy:

![Ba hàm mất mát hồi quy](../images/loss/loss_for_regression.png)

Hình 3: Mất mát bình phương $(y-f(x))^2$, tuyệt đối $|y-f(x)|$ và Huber với $\delta=3$, $\delta=1{,}5$ theo phần dư $y-f(x)$ (code vẽ: [Loss Function Plot.ipynb](https://nbviewer.org/github/massquantity/Loss-Functions/blob/master/Loss%20Function%20Plot.ipynb)).

Với $y$ là giá trị thật và $f(x)$ là dự đoán, ba hàm viết gọn:

- mất mát bình phương (squared loss): $(y - f(x))^2$;
- mất mát tuyệt đối (absolute loss): $|y - f(x)|$;
- mất mát Huber: $\frac12(y-f(x))^2$ khi $|y-f(x)|\le\delta$, và $\delta|y-f(x)|-\frac12\delta^2$ khi $|y-f(x)|>\delta$.

> **Biên tập:** Bản gốc dán ảnh chụp ba công thức này có nhãn tiếng Trung; đã gõ lại. Công thức Huber của bản gốc thừa một dấu ngoặc ở điều kiện thứ hai.

### 3.4 Cài đặt

Code ba hàm mất mát hồi quy bằng Python, và hàm có sẵn trong `sklearn`:

```python
import numpy as np
# true: mảng nhãn thật; pred: mảng dự đoán
def mse(true, pred):
    return np.mean((true - pred) ** 2)

def mae(true, pred):
    return np.mean(np.abs(true - pred))

def huber(true, pred, delta):
    r = np.abs(true - pred)
    loss = np.where(r <= delta, 0.5 * r ** 2, delta * r - 0.5 * delta ** 2)
    return np.mean(loss)

# sklearn cũng có sẵn
from sklearn.metrics import mean_squared_error, mean_absolute_error
```

> **Biên tập:** Code gốc dùng `np.sum` nên trả về tổng chứ không phải trung bình như tên MSE/MAE (và khác với `sklearn.metrics.mean_squared_error` import ngay bên dưới); đã đổi thành `np.mean`.

## 4 Bổ sung: hàm mất mát hay gặp ngoài bài

Mục này do người biên soạn viết thêm.

- **Focal loss** (Lin et al., 2017): $-(1-a_t)^\gamma\log a_t$, giảm trọng số các mẫu đã dễ để mô hình tập trung vào mẫu khó; dùng cho phát hiện vật thể một giai đoạn, nơi số mẫu nền (dễ) áp đảo.
- **Label smoothing**: thay nhãn one-hot bằng $(1-\varepsilon)\,\text{one-hot}+\varepsilon/m$, giảm mô hình quá tự tin; có sẵn qua `label_smoothing` của `nn.CrossEntropyLoss`.
- **Mất mát cho mô hình ngôn ngữ**: chính là cross-entropy trên token tiếp theo, trung bình trên mọi vị trí; $\exp$ của nó là **perplexity**.
- **Mất mát tương phản** (contrastive, InfoNCE): cross-entropy trên các cặp, coi cặp đúng là "lớp đúng" giữa nhiều cặp sai; nền của CLIP và các mô hình embedding.
- **KL divergence** dùng trực tiếp khi mục tiêu là một phân phối: chưng cất tri thức ([bài chưng cất](dlnote/compress_kd_vi)), VAE, và ràng buộc chính sách trong RLHF.

## Tài liệu tham khảo

1. [*Dive into Deep Learning*, mục 22.11 Information Theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html#cross-entropy)
2. [损失函数｜交叉熵损失函数](https://zhuanlan.zhihu.com/p/35709485)
3. [AI-EDU: Hàm mất mát cross-entropy](https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC1%E6%AD%A5%20-%20%E5%9F%BA%E6%9C%AC%E7%9F%A5%E8%AF%86/03.2-%E4%BA%A4%E5%8F%89%E7%86%B5%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.html)
4. [常见回归和分类损失函数比较](https://www.cnblogs.com/massquantity/p/8964029.html)
5. Dư Đình Tung (余霆嵩). *PyTorch tutorial 0.0.5*.
6. https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
7. [一文详解Softmax函数](https://zhuanlan.zhihu.com/p/105722023)
8. [AI-EDU: Hàm phân loại nhiều lớp](https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/en-us/Step3%20-%20LinearClassification/07.1-%E5%A4%9A%E5%88%86%E7%B1%BB%E5%87%BD%E6%95%B0.html#711-softmax)
9. Lin et al. [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). ICCV 2017.
