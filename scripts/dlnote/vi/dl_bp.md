# Nền tảng học sâu: lan truyền ngược và gradient descent

Trang [Code](dlnote/dl_basic_code_vi) đi kèm tự viết lan truyền xuôi và ngược cho mạng hai tầng ở hình 1 rồi so với autograd, và đo bộ nhớ mà các kết quả trung gian của lan truyền xuôi chiếm theo batch size. Ví dụ gradient descent hai biến ở mục 2.3 được chạy lại (sau khi sửa lỗi vẽ hình) ở [trang Code nền tảng toán](dlnote/math_code_vi).

<!-- theory: qa/ch03_vi#3.2.1 ; ml/ch12_vi ; ml/ch16_vi -->

## 1 Lan truyền xuôi và lan truyền ngược

### 1.1 Quá trình huấn luyện mạng neuron

**Huấn luyện mạng neuron** gồm các bước:

1. Dùng tham số ngẫu nhiên "đoán" một kết quả (lan truyền xuôi), gọi là dự đoán $a$;
2. Tính khoảng cách giữa $a$ và nhãn $y$ (tính hàm mất mát);
3. Dùng thuật toán lan truyền ngược để có gradient rồi cập nhật tham số, thử lại với tham số mới. Lần này không còn là "đoán" nữa mà có căn cứ để tiến về hướng đúng, vì việc chỉnh tham số theo chiến lược (gradient descent).

Lặp lại các bước trên nhiều lần cho tới khi dự đoán và kết quả thật gần như không khác nhau, tức $|a-y|\rightarrow 0$, thì kết thúc huấn luyện. (Trong thực tế, ta dừng khi mất mát trên tập kiểm định không giảm nữa, vì đưa sai số huấn luyện về 0 thường là quá khớp.)

Tóm lại: **huấn luyện mô hình là quá trình dùng thuật toán tối ưu như `SGD` để dẫn dắt việc cập nhật tham số**.

### 1.2 Lan truyền xuôi

Lan truyền xuôi (forward propagation, forward pass) là: lần lượt (từ tầng vào tới tầng ra) tính và lưu kết quả của từng tầng trong mạng.

Để hiểu sâu hơn quá trình tính xuôi, có thể vẽ đồ thị tính toán xuôi theo cấu trúc mạng. Dưới đây là một mạng đơn giản và đồ thị tính toán tương ứng:

![Cấu trúc mạng và đồ thị tính xuôi](../images/bp/network_structure_and_forward_computation_graph.png)

Hình 1: Trên: mạng hai tầng phân loại ba lớp (tầng ẩn dùng sigmoid, tầng ra dùng softmax). Dưới: đồ thị tính xuôi tương ứng, chữ hoa là ma trận (theo AI-EDU).

Trong đồ thị, ô vuông là biến, ô tròn là phép toán; dữ liệu được tính lần lượt từ trái sang phải.

### 1.3 Lan truyền ngược

Lan truyền ngược (`backward propagation`, gọi tắt `BP`) là **phương pháp tính gradient theo tham số của mạng**. Nó dựa trên **quy tắc chuỗi** của giải tích, duyệt mạng theo thứ tự ngược lại, từ tầng ra về tầng vào, lần lượt tính gradient theo từng biến trung gian và tham số.

Với mạng ở hình 1 (dùng softmax + cross-entropy), ví dụ hai bước đầu của chiều ngược là:

$$
\frac{\partial L}{\partial Z_2}=A_2-Y,\qquad
\frac{\partial L}{\partial W_2}=A_1^\top\frac{\partial L}{\partial Z_2},\qquad
\frac{\partial L}{\partial A_1}=\frac{\partial L}{\partial Z_2}W_2^\top,\qquad
\frac{\partial L}{\partial Z_1}=\frac{\partial L}{\partial A_1}\odot A_1\odot(1-A_1),
$$

mỗi bước dùng lại gradient của bước trước và một giá trị ($A_1$, $A_2$) đã tính ở chiều xuôi.

> Tính gradient tự động (vi phân tự động, autograd) giúp đơn giản hoá rất nhiều việc cài đặt thuật toán học sâu: chỉ cần viết chiều xuôi, framework tự dựng đồ thị và tính chiều ngược.

Lưu ý, lan truyền ngược dùng lại các giá trị trung gian đã lưu khi lan truyền xuôi để tránh tính lại, nên phải giữ các kết quả trung gian đó. Vì vậy **huấn luyện tốn bộ nhớ (bộ nhớ GPU) hơn nhiều so với chỉ dự đoán**. Bộ nhớ dành cho các kết quả trung gian tỉ lệ với số tầng và với kích thước batch (`batch_size`), nên huấn luyện mạng sâu hơn với batch lớn hơn dễ gây lỗi hết bộ nhớ (out of memory)!

### 1.4 Tóm tắt

- Lan truyền xuôi tính và lưu các biến trung gian theo đồ thị tính toán của mạng, từ tầng vào tới tầng ra.
- Lan truyền ngược tính và lưu gradient theo các biến trung gian và tham số, theo thứ tự ngược lại (từ tầng ra về tầng vào).
- Khi huấn luyện, sau khi khởi tạo tham số, ta luân phiên lan truyền xuôi và ngược, dùng gradient từ lan truyền ngược và thuật toán tối ưu (SGD, `Adam`...) để cập nhật tham số.
- Huấn luyện mô hình học sâu cần nhiều bộ nhớ hơn suy luận.

## 2 Gradient descent

### 2.1 Tối ưu trong học sâu

Phần lớn thuật toán học sâu có một dạng tối ưu nào đó. **Mục đích của bộ tối ưu là cập nhật trọng số để đi tới điểm có mất mát nhỏ nhất trên mặt mất mát.**

Tối ưu trong học sâu có nhiều thách thức, phiền nhất là cực tiểu địa phương, điểm yên ngựa và tiêu biến gradient.

- **Cực tiểu địa phương** (`local minimum`): với hàm mục tiêu $f(x)$, nếu giá trị $f(x)$ tại $x$ nhỏ hơn giá trị tại mọi điểm lân cận $x$, thì $f(x)$ là một cực tiểu địa phương. Nếu $f(x)$ tại $x$ là giá trị nhỏ nhất trên toàn miền, thì đó là cực tiểu toàn cục.
- **Điểm yên ngựa** (`saddle point`): điểm mà mọi thành phần gradient đều bằng 0 nhưng không phải cực tiểu (cũng không phải cực đại): theo một số hướng hàm tăng, theo hướng khác hàm giảm. Trong không gian nhiều chiều, điểm yên ngựa phổ biến hơn cực tiểu địa phương nhiều.
- **Tiêu biến gradient** (`vanishing gradient`): vì một lý do nào đó, gradient của hàm mục tiêu $f$ theo các tham số ở tầng thấp gần bằng 0. Đây là một nguyên nhân khiến mô hình sâu rất khó huấn luyện trước khi có hàm kích hoạt `ReLU` và kiến trúc `ResNet`.

Trong học sâu, phần lớn hàm mục tiêu rất phức tạp, không có nghiệm giải tích, nên phải dùng thuật toán tối ưu số; `SGD` và `Adam` thuộc loại này.

### 2.2 Hiểu gradient descent thế nào

Gradient descent (`GD`) là bộ tối ưu phổ biến nhất khi huấn luyện mạng neuron. Dù gradient descent nguyên bản ít khi được dùng trực tiếp trong học sâu, hiểu nó là nền tảng để hiểu **stochastic gradient descent và minibatch stochastic gradient descent**.

Nhiều bài dùng ví dụ "một người bị kẹt trên núi, cần xuống thung lũng nhanh nhất" để giải thích gradient descent, nhưng không hoàn toàn chính xác. Ví dụ tốt nhất trong tự nhiên là dòng suối chảy xuống núi:

1. Nước chịu tác dụng của trọng lực, tại vị trí hiện tại chảy theo hướng **dốc nhất**, đôi khi thành thác (**ngược hướng gradient là hướng hàm giảm nhanh nhất**);
2. Đường nước chảy không duy nhất: tại một điểm có thể có nhiều hướng dốc như nhau, khiến dòng nước tách nhánh (có thể có nhiều nghiệm);
3. Gặp chỗ trũng, nước có thể đọng thành hồ và dừng lại (chỉ được nghiệm tối ưu địa phương, không phải toàn cục).

> Ví dụ theo [AI-EDU: Gradient descent](https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC1%E6%AD%A5%20-%20%E5%9F%BA%E6%9C%AC%E7%9F%A5%E8%AF%86/02.3-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.html).

### 2.3 Nguyên lý gradient descent

Công thức gradient descent:

$$
\theta_{n+1} = \theta_{n} - \eta \cdot \nabla J(\theta_n) \tag{1}
$$

trong đó:

- $\theta_{n+1}$: giá trị tiếp theo (tham số sau khi cập nhật);
- $\theta_n$: giá trị hiện tại (tham số hiện tại);
- dấu $-$: đi ngược hướng gradient (ngược hướng gradient là hướng hàm giảm nhanh nhất);
- $\eta$: tốc độ học hay độ dài bước, điều khiển mỗi bước đi bao xa: không quá nhanh để khỏi vượt qua điểm tốt nhất, không quá chậm để khỏi mất quá nhiều thời gian (siêu tham số phải chỉnh tay);
- $\nabla$: **gradient**, chỉ hướng hàm tăng nhanh nhất tại vị trí hiện tại (vector gradient chỉ lên dốc, vector gradient âm chỉ xuống dốc);
- $J(\theta)$: hàm mục tiêu cần tối ưu.

Hình dưới minh hoạ một bước gradient descent; mục đích là đưa $x$ về gần điểm cực trị.

![Các bước gradient descent](../images/bp/gd_concept.png)

Hình 2: Một bước gradient descent với $\eta=0{,}1$ trên $Y=(X-2)^2+1$, xuất phát từ hai phía của cực tiểu: bước đi luôn ngược hướng gradient nên luôn tiến về cực tiểu (hình vẽ lại theo AI-EDU; hình gốc chú thích tiếng Trung).

Lấy ví dụ một hàm lồi hai biến đơn giản $J(x, y) = x^2 + 2y^2$ để mô tả quá trình gradient descent.

Muốn tìm giá trị nhỏ nhất bằng gradient descent, trước hết tính gradient:

$$
\frac{\partial J(x,y)}{\partial x} = 2x, \qquad
\frac{\partial J(x,y)}{\partial y} = 4y
$$

Đặt điểm khởi đầu $(x_0, y_0) = (-3, -3)$, tốc độ học $\eta = 0{,}1$. Theo công thức (1), công thức lặp là:

$$
\begin{aligned}
(x_{n+1}, y_{n+1}) &= (x_n, y_n) - \eta \cdot \nabla J(x_n, y_n) \\
&= (x_n, y_n) - \eta \cdot (2x_n, 4y_n)
\end{aligned} \tag{2}
$$

Tính tay điểm tiếp theo:

$$
\begin{aligned}
(x_1, y_1) &= (-3, -3) - 0{,}1\times(2\times(-3),\ 4\times(-3)) \\
&= (-3 + 0{,}6,\ -3 + 1{,}2) \\
&= (-2{,}4;\ -1{,}8)
\end{aligned}
$$

Theo công thức (2), với điều kiện dừng $J(x,y) < 0{,}005$, quá trình lặp như bảng 1.

Bảng 1: **Quá trình lặp** của gradient descent hai biến

| Bước $n$ | $x$ | $y$ | $J(x,y)$ |
| --- | --- | --- | --- |
| 0 | -3 | -3 | 27 |
| 1 | -2,4 | -1,8 | 12,24 |
| ... | ... | ... | ... |
| 16 | -0,084442 | -0,000846 | 0,007132 |
| 17 | -0,067554 | -0,000508 | 0,004564 |

Sau $17$ bước, $J(x,y) = 0{,}004564$, thoả điều kiện nhỏ hơn $0{,}005$ nên dừng.

> **Biên tập:** Bảng gốc đánh số dòng đầu (điểm khởi đầu, chưa lặp) là 1 nhưng dòng cuối lại là 17, lệch một bước so với code in ra (`0:` là điểm khởi đầu); ở đây đánh số theo số bước đã đi. Một số ô của bảng gốc còn sót tiền tố `y=`.

Vì là hàm hai biến, quá trình gradient descent phải vẽ trong không gian ba chiều:

<img src="../img/code/math_code/03_1.png" alt="Gradient descent trên J(x, y) = x² + 2y²">

Hình 3: Quỹ đạo gradient descent trên mặt $J=x^2+2y^2$ và trên đường đồng mức. Theo trục $y$ hàm cong gấp đôi, nên $y$ co nhanh hơn và quỹ đạo về trục $x$ trước, rồi trượt dọc theo nó (hình từ [trang Code nền tảng toán](dlnote/math_code_vi), mục 2).

> **Biên tập:** Bản gốc dán hai ảnh 3D cùng toàn bộ đoạn code tạo ra chúng (script `gd_double_variable.py` của AI-EDU). Đoạn code đó vẽ mặt cong $x^2+4y^2$ chứ không phải hàm $x^2+2y^2$ đang tối ưu, nên quỹ đạo không nằm trên mặt vẽ; và `Axes3D(fig)` không còn hiện hình từ matplotlib 3.6. Code đã sửa và chạy ở trang Code nền tảng toán; hình trên lấy từ đó.

Lưu ý! Với các độ dài bước $\eta$ khác nhau, giá trị của hàm $J$ thay đổi rất khác nhau theo số vòng lặp:

![Ảnh hưởng của tốc độ học](../images/bp/different_lr_loss.png)

Hình 4: Gradient descent trên $f(x)=x^2$ với bốn tốc độ học. $\eta$ quá nhỏ: hội tụ rất chậm; vừa phải: hội tụ nhanh; khá lớn: dao động qua lại quanh cực tiểu nhưng vẫn hội tụ; quá lớn ($\eta>1$ với hàm này): phân kỳ (hình vẽ lại theo bài [如何理解梯度下降法？](https://mp.weixin.qq.com/s/SlTV6lbPnauf36bZLXglCw); hình gốc chú thích tiếng Trung).

## 3 Stochastic gradient descent và minibatch stochastic gradient descent

### 3.1 Stochastic gradient descent

Trong học sâu, hàm mục tiêu thường là trung bình của mất mát trên từng mẫu của tập huấn luyện: $J(\theta)=\frac1n\sum_{i=1}^nJ_i(\theta)$. Dùng gradient descent thì mỗi bước lặp tốn $O(n)$, tăng tuyến tính theo số mẫu $n$; tập huấn luyện lớn thì mỗi bước rất tốn kém.

Stochastic gradient descent (`SGD`) giảm chi phí mỗi bước lặp. Ở mỗi bước, ta lấy **ngẫu nhiên đều** một chỉ số $i \in \{1, \dots, n\}$, rồi tính gradient $\nabla J_i(\theta)$ để cập nhật tham số $\theta$:

$$
\theta_{n+1} = \theta_{n} - \eta \cdot \nabla J_i(\theta_n) \tag{3}
$$

Chi phí mỗi bước giảm từ $O(n)$ của gradient descent xuống hằng số $O(1)$. Cần nhấn mạnh: gradient ngẫu nhiên $\nabla J_i(\theta)$ là **ước lượng không chệch** của gradient đầy đủ $\nabla J(\theta)$, vì $\mathbb E_i[\nabla J_i(\theta)]=\nabla J(\theta)$.

> Ước lượng không chệch: dùng thống kê từ mẫu để ước lượng tham số của tổng thể mà kỳ vọng của ước lượng đúng bằng tham số đó.

Trong thực tế, **SGD thường được dùng cùng tốc độ học giảm dần** (lịch tốc độ học). Với tốc độ học cố định, nhiễu của gradient ngẫu nhiên khiến tham số dao động mãi quanh cực tiểu thay vì hội tụ hẳn; giảm tốc độ học dần giúp dao động nhỏ lại. Cách chỉnh tốc độ học xem bài *Luyện đan học sâu: thiết lập siêu tham số và huấn luyện mô hình* (phần 4 của khoá này).

> **Biên tập:** Bản gốc viết SGD "bắt buộc phải kết hợp với tốc độ học động, nếu không quá trình hội tụ sẽ phức tạp hơn". Lý do cụ thể là nhiễu của gradient ngẫu nhiên không giảm về 0 tại cực tiểu, như giải thích ở trên; trong thực tế vẫn có nhiều trường hợp huấn luyện với tốc độ học cố định và lấy mô hình tốt nhất trên tập kiểm định.

### 3.2 Minibatch stochastic gradient descent

Gradient descent (`GD`) và stochastic gradient descent (`SGD`) ở trên đều quá cực đoan: hoặc dùng cả tập dữ liệu để tính gradient, hoặc mỗi lần chỉ dùng một mẫu. Trong thực tế, ta dùng cách dung hoà là **minibatch stochastic gradient descent** (`minibatch stochastic gradient descent`), với các ưu điểm:

- Gradient của mất mát trên minibatch là ước lượng của gradient trên tập huấn luyện, và chất lượng ước lượng tăng khi batch lớn hơn (phương sai giảm tỉ lệ nghịch với kích thước batch).
- Nhờ khả năng **tính song song** của phần cứng hiện đại, tính trên một batch $m$ mẫu cùng lúc hiệu quả hơn nhiều so với tính $m$ mẫu riêng lẻ.

Các mẫu trong minibatch đều được **lấy ngẫu nhiên** từ tập huấn luyện. Gọi kích thước batch là $m$ (`batch_size = m`), tức ở mỗi bước ta xét minibatch $\boldsymbol x_{1\ldots m}$:

$$
\theta_{n+1} = \theta_{n} - \eta \cdot \frac{1}{m}\sum_{i=1}^{m}\nabla J_i(\theta_n) \tag{4}
$$

Khi nói dùng thuật toán `SGD`, các dự án thường ngầm hiểu là minibatch SGD, tức $m > 1$; chỉ khi bộ nhớ GPU không đủ mới đặt $m = 1$.

> Ký hiệu gradient cũng có thể viết $\frac{\partial\ell(\boldsymbol x_i, \theta)}{\partial\theta}$, với $\boldsymbol x_i$ là một mẫu huấn luyện, $\theta$ là tham số, và $\ell(\boldsymbol x_i, \theta)$ là mất mát trên mẫu đó.

> **Biên tập:** Bản gốc đánh số công thức minibatch là (3), trùng với công thức SGD, và đặt $\nabla$ trước dấu tổng ($\nabla\sum$, vẫn đúng do tính tuyến tính nhưng khó đọc); ở đây đánh số (4) và đưa $\nabla$ vào trong tổng.

## 4 Tổng kết

Stochastic gradient descent (`SGD`) đơn giản và hiệu quả, nhưng cần chỉnh cẩn thận các siêu tham số, nhất là **tốc độ học**, cùng **giá trị khởi tạo của tham số** (xem bài [Khởi tạo tham số](dlnote/dl_init_vi)). Vì đầu vào của mỗi tầng chịu ảnh hưởng của tham số mọi tầng trước, huấn luyện trở nên phức tạp: mạng càng sâu, thay đổi nhỏ của tham số càng bị tích luỹ và khuếch đại. Các bộ tối ưu cải tiến (momentum, Adam...) ở bài [Thuật toán tối ưu](dlnote/dl_optim_vi) giải quyết một phần các khó khăn này.

## Tài liệu tham khảo

1. [如何理解梯度下降法？](https://mp.weixin.qq.com/s/SlTV6lbPnauf36bZLXglCw)
2. [AI-EDU: Gradient descent](https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC1%E6%AD%A5%20-%20%E5%9F%BA%E6%9C%AC%E7%9F%A5%E8%AF%86/02.3-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.html)
3. *Dive into Deep Learning*, chương 12 Thuật toán tối ưu; trên site: [Học sâu, chương 12](dl/ch12_vi).
