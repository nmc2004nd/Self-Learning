# Hạ Gradient
:label:`sec_gd`

Trong mục này, ta sẽ giới thiệu các khái niệm cơ bản làm nền tảng cho *hạ gradient* (gradient descent). Dù hiếm khi được dùng trực tiếp trong học sâu, hiểu về hạ gradient là chìa khóa để hiểu các thuật toán hạ gradient ngẫu nhiên. Chẳng hạn, bài toán tối ưu có thể phân kỳ do tốc độ học quá lớn. Hiện tượng này đã có thể thấy ngay trong hạ gradient. Tương tự, tiền điều kiện (preconditioning) là một kỹ thuật phổ biến trong hạ gradient và được chuyển sang các thuật toán nâng cao hơn. Hãy bắt đầu với một trường hợp đặc biệt đơn giản.

## Hạ Gradient Một chiều

Hạ gradient một chiều là một ví dụ tuyệt vời để giải thích vì sao thuật toán hạ gradient có thể làm giảm giá trị của hàm mục tiêu. Xét một hàm giá trị thực khả vi liên tục $f: \mathbb{R} \rightarrow \mathbb{R}$ nào đó. Dùng khai triển Taylor, ta có

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

Tức là, trong xấp xỉ bậc nhất, $f(x+\epsilon)$ được cho bởi giá trị hàm $f(x)$ và đạo hàm bậc nhất $f'(x)$ tại $x$. Không phải vô lý khi giả định rằng với $\epsilon$ nhỏ, di chuyển theo hướng ngược gradient sẽ làm giảm $f$. Để giữ mọi thứ đơn giản, ta chọn một kích thước bước cố định $\eta > 0$ và chọn $\epsilon = -\eta f'(x)$. Thay vào khai triển Taylor ở trên, ta được

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

Nếu đạo hàm $f'(x) \neq 0$ không triệt tiêu, ta có tiến triển vì $\eta f'^2(x)>0$. Hơn nữa, ta luôn có thể chọn $\eta$ đủ nhỏ để các số hạng bậc cao trở nên không đáng kể. Do đó ta đi đến

$$f(x - \eta f'(x)) \lessapprox f(x).$$

Điều này nghĩa là, nếu ta dùng

$$x \leftarrow x - \eta f'(x)$$

để cập nhật lặp $x$, giá trị của hàm $f(x)$ có thể giảm. Do đó, trong hạ gradient, trước tiên ta chọn một giá trị ban đầu $x$ và một hằng số $\eta > 0$, rồi dùng chúng để liên tục cập nhật lặp $x$ cho đến khi đạt điều kiện dừng, chẳng hạn khi độ lớn của gradient $|f'(x)|$ đủ nhỏ hoặc số vòng lặp đã đạt một giá trị nhất định.

Để đơn giản, ta chọn hàm mục tiêu $f(x)=x^2$ để minh họa cách cài đặt hạ gradient. Dù biết rằng $x=0$ là nghiệm cực tiểu hóa $f(x)$, ta vẫn dùng hàm đơn giản này để quan sát $x$ thay đổi thế nào.

```{.python .input}
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
def f(x):  # Hàm mục tiêu
    return x ** 2

def f_grad(x):  # Gradient (đạo hàm) của hàm mục tiêu
    return 2 * x
```

Tiếp theo, ta dùng $x=10$ làm giá trị ban đầu và giả sử $\eta=0.2$. Dùng hạ gradient để cập nhật $x$ 10 lần, ta có thể thấy cuối cùng giá trị của $x$ tiến gần đến nghiệm tối ưu.

```{.python .input}
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

Tiến trình tối ưu theo $x$ có thể được vẽ như sau.

```{.python .input}
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### Tốc độ học
:label:`subsec_gd-learningrate`

Tốc độ học $\eta$ có thể do người thiết kế thuật toán đặt. Nếu dùng tốc độ học quá nhỏ, nó sẽ khiến $x$ cập nhật rất chậm, cần nhiều vòng lặp hơn để có nghiệm tốt hơn. Để thấy điều gì xảy ra trong trường hợp như vậy, hãy xét tiến trình của cùng bài toán tối ưu với $\eta = 0.05$. Như ta có thể thấy, ngay cả sau 10 bước, ta vẫn còn rất xa nghiệm tối ưu.

```{.python .input}
show_trace(gd(0.05, f_grad), f)
```

Ngược lại, nếu dùng tốc độ học quá cao, $\left|\eta f'(x)\right|$ có thể quá lớn đối với công thức khai triển Taylor bậc nhất. Tức là, số hạng $\mathcal{O}(\eta^2 f'^2(x))$ trong :eqref:`gd-taylor-2` có thể trở nên đáng kể. Trong trường hợp này, ta không thể đảm bảo rằng việc cập nhật $x$ sẽ làm giảm giá trị của $f(x)$. Chẳng hạn, khi đặt tốc độ học $\eta=1.1$, $x$ vượt quá nghiệm tối ưu $x=0$ và dần dần phân kỳ.

```{.python .input}
show_trace(gd(1.1, f_grad), f)
```

### Cực tiểu Địa phương

Để minh họa điều gì xảy ra với các hàm không lồi, hãy xét trường hợp $f(x) = x \cdot \cos(cx)$ với một hằng số $c$ nào đó. Hàm này có vô số cực tiểu địa phương. Tùy thuộc vào lựa chọn tốc độ học và mức độ điều kiện tốt của bài toán, ta có thể đi đến một trong nhiều nghiệm. Ví dụ dưới đây minh họa việc một tốc độ học cao (một cách phi thực tế) sẽ dẫn đến một cực tiểu địa phương kém như thế nào.

```{.python .input}
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Hàm mục tiêu
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient của hàm mục tiêu
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## Hạ Gradient Nhiều biến

Giờ khi đã có trực giác tốt hơn về trường hợp một biến, hãy xét tình huống $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$. Tức là, hàm mục tiêu $f: \mathbb{R}^d \to \mathbb{R}$ ánh xạ vector thành số vô hướng. Tương ứng, gradient của nó cũng nhiều biến. Nó là một vector gồm $d$ đạo hàm riêng:

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

Mỗi phần tử đạo hàm riêng $\partial f(\mathbf{x})/\partial x_i$ trong gradient cho biết tốc độ thay đổi của $f$ tại $\mathbf{x}$ theo đầu vào $x_i$. Như trước đây trong trường hợp một biến, ta có thể dùng xấp xỉ Taylor tương ứng cho hàm nhiều biến để có ý tưởng về những gì nên làm. Cụ thể, ta có

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

Nói cách khác, bỏ qua các số hạng bậc hai trong $\boldsymbol{\epsilon}$, hướng giảm dốc nhất được cho bởi gradient âm $-\nabla f(\mathbf{x})$. Chọn một tốc độ học phù hợp $\eta > 0$ cho ta thuật toán hạ gradient nguyên mẫu:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

Để thấy thuật toán hoạt động thế nào trong thực tế, hãy dựng một hàm mục tiêu $f(\mathbf{x})=x_1^2+2x_2^2$ với đầu vào là một vector hai chiều $\mathbf{x} = [x_1, x_2]^\top$ và đầu ra là một số vô hướng. Gradient được cho bởi $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$. Ta sẽ quan sát quỹ đạo của $\mathbf{x}$ theo hạ gradient từ vị trí ban đầu $[-5, -2]$.

Để bắt đầu, ta cần thêm hai hàm hỗ trợ. Hàm thứ nhất dùng một hàm cập nhật và áp dụng nó 20 lần lên giá trị ban đầu. Hàm hỗ trợ thứ hai trực quan hóa quỹ đạo của $\mathbf{x}$.

```{.python .input}
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Tối ưu một hàm mục tiêu hai chiều với bộ huấn luyện tùy chỉnh."""
    # `s1` và `s2` là các biến trạng thái bên trong sẽ được dùng trong Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
```

```{.python .input}
def show_trace_2d(f, results):  #@save
    """Hiển thị quỹ đạo của các biến hai chiều trong quá trình tối ưu."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

Tiếp theo, ta quan sát quỹ đạo của biến tối ưu $\mathbf{x}$ với tốc độ học $\eta = 0.1$. Ta có thể thấy sau 20 bước, giá trị của $\mathbf{x}$ tiến gần đến cực tiểu tại $[0, 0]$. Tiến trình khá ổn định dù hơi chậm.

```{.python .input}
def f_2d(x1, x2):  # Hàm mục tiêu
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient của hàm mục tiêu
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## Các Phương pháp Thích ứng

Như ta đã thấy trong :numref:`subsec_gd-learningrate`, chọn tốc độ học $\eta$ "vừa đúng" là việc khó. Nếu chọn quá nhỏ, ta tiến triển rất ít. Nếu chọn quá lớn, nghiệm dao động và trong trường hợp tệ nhất thậm chí có thể phân kỳ. Sẽ thế nào nếu ta có thể xác định $\eta$ một cách tự động, hoặc hoàn toàn không phải chọn tốc độ học? Các phương pháp bậc hai, vốn không chỉ xem xét giá trị và gradient của hàm mục tiêu mà còn cả *độ cong* của nó, có thể giúp ích trong trường hợp này. Dù những phương pháp này không thể áp dụng trực tiếp vào học sâu do chi phí tính toán, chúng cung cấp trực giác hữu ích về cách thiết kế các thuật toán tối ưu nâng cao mô phỏng nhiều tính chất đáng mong muốn của các thuật toán được trình bày dưới đây.

### Phương pháp Newton

Xem lại khai triển Taylor của một hàm $f: \mathbb{R}^d \rightarrow \mathbb{R}$ nào đó, không cần phải dừng sau số hạng đầu tiên. Thực ra, ta có thể viết nó như sau

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

Để tránh ký hiệu rườm rà, ta định nghĩa $\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2 f(\mathbf{x})$ là ma trận Hessian của $f$, một ma trận $d \times d$. Với $d$ nhỏ và các bài toán đơn giản, $\mathbf{H}$ dễ tính. Mặt khác, với mạng neuron sâu, $\mathbf{H}$ có thể lớn đến mức không thể chấp nhận được, do chi phí lưu trữ $\mathcal{O}(d^2)$ phần tử. Hơn nữa, nó có thể quá tốn kém để tính bằng lan truyền ngược. Tạm thời hãy bỏ qua những cân nhắc như vậy và xem ta sẽ có được thuật toán gì.

Suy cho cùng, cực tiểu của $f$ thỏa mãn $\nabla f = 0$. Theo các quy tắc giải tích trong :numref:`subsec_calculus-grad`, bằng cách lấy đạo hàm của :eqref:`gd-hot-taylor` theo $\boldsymbol{\epsilon}$ và bỏ qua các số hạng bậc cao, ta đi đến

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \textrm{ và do đó }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

Tức là, ta cần nghịch đảo ma trận Hessian $\mathbf{H}$ như một phần của bài toán tối ưu.

Lấy một ví dụ đơn giản, với $f(x) = \frac{1}{2} x^2$ ta có $\nabla f(x) = x$ và $\mathbf{H} = 1$. Do đó với mọi $x$ ta được $\epsilon = -x$. Nói cách khác, chỉ *một* bước là đủ để hội tụ hoàn hảo mà không cần bất kỳ điều chỉnh nào! Than ôi, ở đây ta hơi may mắn: khai triển Taylor là chính xác vì $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$.

Hãy xem điều gì xảy ra với những bài toán khác. Cho một hàm cosin hyperbolic lồi $f(x) = \cosh(cx)$ với một hằng số $c$ nào đó, ta có thể thấy cực tiểu toàn cục tại $x=0$ đạt được sau vài vòng lặp.

```{.python .input}
c = d2l.tensor(0.5)

def f(x):  # Hàm mục tiêu
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient của hàm mục tiêu
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian của hàm mục tiêu
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

Giờ hãy xét một hàm *không lồi*, như $f(x) = x \cos(c x)$ với một hằng số $c$ nào đó. Suy cho cùng, lưu ý rằng trong phương pháp Newton, cuối cùng ta chia cho Hessian. Điều này nghĩa là nếu đạo hàm bậc hai *âm*, ta có thể đi theo hướng *làm tăng* giá trị của $f$. Đó là một khiếm khuyết chết người của thuật toán. Hãy xem điều gì xảy ra trong thực tế.

```{.python .input}
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Hàm mục tiêu
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient của hàm mục tiêu
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian của hàm mục tiêu
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

Kết quả sai một cách ngoạn mục. Làm sao ta có thể khắc phục nó? Một cách là "sửa" Hessian bằng cách lấy giá trị tuyệt đối của nó. Một chiến lược khác là đưa tốc độ học trở lại. Điều này có vẻ đi ngược mục đích, nhưng không hẳn. Có thông tin bậc hai cho phép ta thận trọng bất cứ khi nào độ cong lớn và bước dài hơn bất cứ khi nào hàm mục tiêu phẳng hơn. Hãy xem điều này hoạt động thế nào với một tốc độ học nhỏ hơn một chút, chẳng hạn $\eta = 0.5$. Như ta có thể thấy, ta có một thuật toán khá hiệu quả.

```{.python .input}
show_trace(newton(0.5), f)
```

### Phân tích Hội tụ

Ta chỉ phân tích tốc độ hội tụ của phương pháp Newton cho một hàm mục tiêu $f$ lồi và khả vi ba lần, trong đó đạo hàm bậc hai khác không, tức là $f'' > 0$. Chứng minh cho trường hợp nhiều biến là sự mở rộng trực tiếp của lập luận một chiều dưới đây và được bỏ qua vì nó không giúp ta nhiều về mặt trực giác.

Ký hiệu $x^{(k)}$ là giá trị của $x$ ở vòng lặp thứ $k$ và đặt $e^{(k)} \stackrel{\textrm{def}}{=} x^{(k)} - x^*$ là khoảng cách đến điểm tối ưu ở vòng lặp thứ $k$. Theo khai triển Taylor, điều kiện $f'(x^*) = 0$ có thể được viết là

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

đúng với một $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$ nào đó. Chia khai triển trên cho $f''(x^{(k)})$ ta được

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

Hãy nhớ lại rằng ta có phép cập nhật $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$. Thay phương trình cập nhật này vào và lấy giá trị tuyệt đối hai vế, ta có

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

Do đó, bất cứ khi nào ta ở trong một vùng mà $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$ bị chặn, ta có sai số giảm theo bậc hai

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$

Nói thêm, các nhà nghiên cứu tối ưu gọi đây là hội tụ *tuyến tính*, trong khi một điều kiện như $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$ sẽ được gọi là tốc độ hội tụ *hằng số*. Lưu ý rằng phân tích này đi kèm một số lưu ý. Thứ nhất, ta thực sự không có nhiều đảm bảo về việc khi nào sẽ đến được vùng hội tụ nhanh. Thay vào đó, ta chỉ biết rằng một khi đến được đó, sự hội tụ sẽ rất nhanh. Thứ hai, phân tích này đòi hỏi $f$ hoạt động tốt đến các đạo hàm bậc cao. Điều đó quy về việc đảm bảo $f$ không có bất kỳ tính chất "bất ngờ" nào về cách nó có thể thay đổi giá trị.

### Tiền điều kiện

Không có gì ngạc nhiên khi tính và lưu trữ toàn bộ ma trận Hessian là rất tốn kém. Do đó, việc tìm những phương án thay thế là đáng mong muốn. Một cách để cải thiện tình hình là *tiền điều kiện* (preconditioning). Nó tránh việc tính toàn bộ Hessian mà chỉ tính các phần tử trên *đường chéo*. Điều này dẫn đến các thuật toán cập nhật có dạng

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$

Dù cách này không tốt bằng phương pháp Newton đầy đủ, nó vẫn tốt hơn nhiều so với không dùng. Để thấy vì sao đây có thể là một ý tưởng hay, hãy xét một tình huống trong đó một biến biểu thị chiều cao tính bằng milimét và biến kia biểu thị chiều cao tính bằng kilômét. Giả sử với cả hai, thang đo tự nhiên là mét, ta có một sự chênh lệch khủng khiếp trong cách tham số hóa. May mắn thay, dùng tiền điều kiện sẽ loại bỏ điều này. Về thực chất, tiền điều kiện kết hợp với hạ gradient tương đương với việc chọn một tốc độ học khác nhau cho mỗi biến (mỗi tọa độ của vector $\mathbf{x}$). Như ta sẽ thấy sau này, tiền điều kiện thúc đẩy một phần những đổi mới trong các thuật toán tối ưu hạ gradient ngẫu nhiên.

### Hạ Gradient với Tìm kiếm theo Đường

Một trong những vấn đề then chốt của hạ gradient là ta có thể vượt quá mục tiêu hoặc tiến triển không đủ. Một cách khắc phục đơn giản là dùng tìm kiếm theo đường (line search) kết hợp với hạ gradient. Tức là, ta dùng hướng cho bởi $\nabla f(\mathbf{x})$ rồi thực hiện tìm kiếm nhị phân xem tốc độ học $\eta$ nào cực tiểu hóa $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$.

Thuật toán này hội tụ nhanh (để có phân tích và chứng minh, xem chẳng hạn :citet:`Boyd.Vandenberghe.2004`). Tuy nhiên, với mục đích học sâu, cách này không khả thi lắm, vì mỗi bước tìm kiếm theo đường sẽ đòi hỏi ta đánh giá hàm mục tiêu trên toàn bộ tập dữ liệu. Điều này quá tốn kém để thực hiện.

## Tóm tắt

* Tốc độ học rất quan trọng. Quá lớn thì ta phân kỳ, quá nhỏ thì ta không tiến triển.
* Hạ gradient có thể bị mắc kẹt ở cực tiểu địa phương.
* Trong không gian nhiều chiều, điều chỉnh tốc độ học rất phức tạp.
* Tiền điều kiện có thể giúp điều chỉnh thang đo.
* Phương pháp Newton nhanh hơn rất nhiều một khi nó đã bắt đầu hoạt động đúng trong các bài toán lồi.
* Hãy cẩn thận khi dùng phương pháp Newton mà không có bất kỳ điều chỉnh nào cho các bài toán không lồi.

## Bài tập

1. Thử nghiệm với các tốc độ học và hàm mục tiêu khác nhau cho hạ gradient.
1. Cài đặt tìm kiếm theo đường để cực tiểu hóa một hàm lồi trên khoảng $[a, b]$.
    1. Bạn có cần đạo hàm cho tìm kiếm nhị phân không, tức là để quyết định chọn $[a, (a+b)/2]$ hay $[(a+b)/2, b]$?
    1. Tốc độ hội tụ của thuật toán nhanh đến đâu?
    1. Cài đặt thuật toán và áp dụng nó để cực tiểu hóa $\log (\exp(x) + \exp(-2x -3))$.
1. Thiết kế một hàm mục tiêu định nghĩa trên $\mathbb{R}^2$ mà hạ gradient chậm vô cùng. Gợi ý: co giãn các tọa độ khác nhau theo những mức khác nhau.
1. Cài đặt phiên bản gọn nhẹ của phương pháp Newton dùng tiền điều kiện:
    1. Dùng đường chéo của Hessian làm bộ tiền điều kiện.
    1. Dùng giá trị tuyệt đối của nó thay vì giá trị thực (có thể mang dấu).
    1. Áp dụng điều này cho bài toán ở trên.
1. Áp dụng thuật toán trên cho một số hàm mục tiêu (lồi hoặc không). Điều gì xảy ra nếu bạn xoay hệ tọa độ đi $45$ độ?

[Discussions](https://discuss.d2l.ai/t/351)
