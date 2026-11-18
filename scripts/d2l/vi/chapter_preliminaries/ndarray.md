# Thao tác với Dữ liệu
:label:`sec_ndarray`

Để làm được bất cứ việc gì, ta cần có cách lưu trữ và thao tác với dữ liệu. Nhìn chung, có hai việc quan trọng ta cần làm với dữ liệu: (i) thu thập chúng; và (ii) xử lý chúng khi đã nằm trong máy tính. Thu thập dữ liệu mà không có cách nào lưu trữ thì cũng vô ích, vì vậy để bắt đầu, hãy bắt tay vào làm việc với các mảng $n$ chiều, mà ta cũng gọi là *tensor*. Nếu bạn đã biết gói tính toán khoa học NumPy, phần này sẽ rất dễ dàng. Với mọi framework học sâu hiện đại, *lớp tensor* (`ndarray` trong MXNet, `Tensor` trong PyTorch và TensorFlow) giống với `ndarray` của NumPy, có thêm một vài tính năng đắt giá. Thứ nhất, lớp tensor hỗ trợ tính vi phân tự động. Thứ hai, nó tận dụng GPU để tăng tốc tính toán số học, trong khi NumPy chỉ chạy trên CPU. Những đặc tính này khiến mạng neuron vừa dễ lập trình vừa chạy nhanh.

## Bắt đầu

(**Để bắt đầu, ta nhập thư viện PyTorch. Lưu ý rằng tên gói là `torch`.**)

```{.python .input}
import torch
```

[**Một tensor biểu diễn một mảng (có thể nhiều chiều) các giá trị số.**] Trong trường hợp một chiều, tức là khi dữ liệu chỉ cần một trục, tensor được gọi là một *vector*. Với hai trục, tensor được gọi là một *ma trận*. Với $k > 2$ trục, ta không dùng tên gọi riêng nữa mà chỉ gọi đối tượng đó là *tensor bậc $k$*.

PyTorch cung cấp nhiều hàm để tạo các tensor mới đã được điền sẵn giá trị. Chẳng hạn, bằng cách gọi `arange(n)`, ta có thể tạo một vector gồm các giá trị cách đều nhau, bắt đầu từ 0 (có lấy) và kết thúc ở `n` (không lấy). Theo mặc định, khoảng cách giữa các giá trị là $1$. Trừ khi được chỉ định khác, các tensor mới được lưu trong bộ nhớ chính và được dùng cho tính toán trên CPU.

```{.python .input}
x = torch.arange(12, dtype=torch.float32)
x
```

Mỗi giá trị này được gọi là một *phần tử* của tensor. Tensor `x` chứa 12 phần tử. Ta có thể kiểm tra tổng số phần tử của một tensor bằng phương thức `numel`.

```{.python .input}
x.numel()
```

(**Ta có thể truy cập *kích thước* của một tensor**) (độ dài theo mỗi trục) bằng cách xem thuộc tính `shape` của nó. Vì ở đây ta đang làm việc với một vector, `shape` chỉ chứa một phần tử duy nhất và trùng với số phần tử.

```{.python .input}
x.shape
```

Ta có thể [**thay đổi kích thước của một tensor mà không làm thay đổi số phần tử hay giá trị của nó**] bằng cách gọi `reshape`. Chẳng hạn, ta có thể biến đổi vector `x` có kích thước (12,) thành một ma trận `X` có kích thước (3, 4). Tensor mới này giữ lại mọi phần tử nhưng sắp xếp lại chúng thành một ma trận. Lưu ý rằng các phần tử của vector được xếp lần lượt từng hàng một, do đó `x[3] == X[0, 3]`.

```{.python .input}
X = x.reshape(3, 4)
X
```

Lưu ý rằng việc chỉ định mọi thành phần kích thước cho `reshape` là thừa. Vì đã biết số phần tử của tensor, ta có thể tính ra một thành phần của kích thước khi biết các thành phần còn lại. Chẳng hạn, với một tensor có $n$ phần tử và kích thước đích ($h$, $w$), ta biết rằng $w = n/h$. Để tự động suy ra một thành phần của kích thước, ta có thể đặt `-1` vào vị trí thành phần cần được suy ra tự động. Trong trường hợp của ta, thay vì gọi `x.reshape(3, 4)`, ta có thể gọi một cách tương đương `x.reshape(-1, 4)` hoặc `x.reshape(3, -1)`.

Người làm thực tế thường cần làm việc với các tensor được khởi tạo chứa toàn số 0 hoặc toàn số 1. [**Ta có thể dựng một tensor có mọi phần tử bằng 0**] (~~hoặc bằng 1~~) và kích thước (2, 3, 4) bằng hàm `zeros`.

```{.python .input}
torch.zeros((2, 3, 4))
```

Tương tự, ta có thể tạo một tensor chứa toàn số 1 bằng cách gọi `ones`.

```{.python .input}
torch.ones((2, 3, 4))
```

Ta thường muốn [**lấy mẫu mỗi phần tử một cách ngẫu nhiên (và độc lập)**] từ một phân phối xác suất cho trước. Chẳng hạn, các tham số của mạng neuron thường được khởi tạo ngẫu nhiên. Đoạn code sau tạo một tensor với các phần tử được lấy từ phân phối Gauss (phân phối chuẩn) tiêu chuẩn có trung bình 0 và độ lệch chuẩn 1.

```{.python .input}
torch.randn(3, 4)
```

Cuối cùng, ta có thể dựng tensor bằng cách [**cung cấp giá trị chính xác cho từng phần tử**] thông qua (các) danh sách Python (có thể lồng nhau) chứa các giá trị số. Ở đây, ta dựng một ma trận từ một danh sách các danh sách, trong đó danh sách ngoài cùng tương ứng với trục 0, và danh sách bên trong tương ứng với trục 1.

```{.python .input}
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Chỉ số và Cắt lát

Giống như với danh sách Python, ta có thể truy cập các phần tử của tensor bằng chỉ số (bắt đầu từ 0). Để truy cập một phần tử theo vị trí tính từ cuối danh sách, ta có thể dùng chỉ số âm. Ngoài ra, ta có thể truy cập cả một khoảng chỉ số bằng cách cắt lát (slicing, chẳng hạn `X[start:stop]`), trong đó giá trị trả về gồm chỉ số đầu (`start`) *nhưng không gồm chỉ số cuối* (`stop`). Cuối cùng, khi chỉ có một chỉ số (hoặc một lát cắt) được chỉ định cho một tensor bậc $k$, nó được áp dụng theo trục 0. Do đó, trong đoạn code sau, [**`[-1]` chọn hàng cuối cùng và `[1:3]` chọn hàng thứ hai và thứ ba**].

```{.python .input}
X[-1], X[1:3]
```

Ngoài việc đọc, (**ta cũng có thể *ghi* các phần tử của ma trận bằng cách chỉ định chỉ số.**)

```{.python .input}
X[1, 2] = 17
X
```

Nếu muốn [**gán cùng một giá trị cho nhiều phần tử, ta đặt phép lấy chỉ số ở vế trái của phép gán.**] Chẳng hạn, `[:2, :]` truy cập hàng thứ nhất và thứ hai, trong đó `:` lấy mọi phần tử theo trục 1 (cột). Dù ta đã thảo luận việc lấy chỉ số cho ma trận, cách này cũng áp dụng được cho vector và cho tensor nhiều hơn hai chiều.

```{.python .input}
X[:2, :] = 12
X
```

## Các Phép toán

Giờ ta đã biết cách dựng tensor và cách đọc, ghi các phần tử của chúng, ta có thể bắt đầu thao tác với chúng bằng nhiều phép toán. Một trong những phép toán hữu ích nhất là các phép toán *theo từng phần tử* (elementwise). Chúng áp dụng một phép toán vô hướng tiêu chuẩn lên từng phần tử của tensor. Với những hàm nhận hai tensor làm đầu vào, phép toán theo từng phần tử áp dụng một toán tử hai ngôi tiêu chuẩn lên từng cặp phần tử tương ứng. Ta có thể tạo một hàm theo từng phần tử từ bất kỳ hàm nào ánh xạ một số vô hướng thành một số vô hướng.

Theo ký hiệu toán học, ta ký hiệu các toán tử vô hướng *một ngôi* (nhận một đầu vào) như vậy bằng $f: \mathbb{R} \rightarrow \mathbb{R}$. Điều này đơn giản nghĩa là hàm ánh xạ bất kỳ số thực nào thành một số thực khác. Hầu hết các toán tử tiêu chuẩn, kể cả các toán tử một ngôi như $e^x$, đều có thể áp dụng theo từng phần tử.

```{.python .input}
torch.exp(x)
```

Tương tự, ta ký hiệu các toán tử vô hướng *hai ngôi*, ánh xạ cặp số thực thành một số thực (duy nhất), bằng $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$. Cho hai vector bất kỳ $\mathbf{u}$ và $\mathbf{v}$ *có cùng kích thước*, và một toán tử hai ngôi $f$, ta có thể tạo ra một vector $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ bằng cách đặt $c_i \gets f(u_i, v_i)$ với mọi $i$, trong đó $c_i, u_i$ và $v_i$ là phần tử thứ $i$ của các vector $\mathbf{c}, \mathbf{u}$ và $\mathbf{v}$. Ở đây, ta tạo ra hàm giá trị vector $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ bằng cách *nâng* hàm vô hướng thành một phép toán vector theo từng phần tử. Các toán tử số học tiêu chuẩn thường dùng là cộng (`+`), trừ (`-`), nhân (`*`), chia (`/`) và lũy thừa (`**`) đều đã được *nâng* thành phép toán theo từng phần tử cho các tensor cùng kích thước, với kích thước bất kỳ.

```{.python .input}
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

Ngoài các phép tính theo từng phần tử, ta cũng có thể thực hiện các phép toán đại số tuyến tính, như tích vô hướng và nhân ma trận. Ta sẽ trình bày kỹ hơn về chúng trong :numref:`sec_linear-algebra`.

Ta cũng có thể [***ghép nối* nhiều tensor**], xếp chúng nối đuôi nhau để tạo thành một tensor lớn hơn. Ta chỉ cần cung cấp một danh sách các tensor và cho hệ thống biết ghép nối theo trục nào. Ví dụ dưới đây cho thấy điều gì xảy ra khi ta ghép nối hai ma trận theo hàng (trục 0) thay vì theo cột (trục 1). Ta có thể thấy độ dài theo trục 0 của đầu ra thứ nhất ($6$) bằng tổng độ dài theo trục 0 của hai tensor đầu vào ($3 + 3$); còn độ dài theo trục 1 của đầu ra thứ hai ($8$) bằng tổng độ dài theo trục 1 của hai tensor đầu vào ($4 + 4$).

```{.python .input}
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

Đôi khi, ta muốn [**dựng một tensor nhị phân bằng các *mệnh đề logic*.**] Lấy `X == Y` làm ví dụ. Với mỗi vị trí `i, j`, nếu `X[i, j]` và `Y[i, j]` bằng nhau thì mục tương ứng trong kết quả nhận giá trị `1`, ngược lại nhận giá trị `0`.

```{.python .input}
X == Y
```

[**Lấy tổng mọi phần tử trong tensor**] cho ra một tensor chỉ có một phần tử.

```{.python .input}
X.sum()
```

## Cơ chế Lan truyền
:label:`subsec_broadcasting`

Đến đây, bạn đã biết cách thực hiện các phép toán hai ngôi theo từng phần tử trên hai tensor cùng kích thước. Trong những điều kiện nhất định, ngay cả khi kích thước khác nhau, ta vẫn có thể [**thực hiện các phép toán hai ngôi theo từng phần tử bằng cách dùng *cơ chế lan truyền* (broadcasting).**] Cơ chế lan truyền hoạt động theo quy trình hai bước sau: (i) mở rộng một hoặc cả hai mảng bằng cách sao chép các phần tử dọc theo những trục có độ dài 1, để sau phép biến đổi này hai tensor có cùng kích thước; (ii) thực hiện phép toán theo từng phần tử trên các mảng thu được.

```{.python .input}
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

Vì `a` và `b` lần lượt là các ma trận $3\times1$ và $1\times2$, kích thước của chúng không khớp nhau. Cơ chế lan truyền tạo ra một ma trận lớn hơn kích thước $3\times2$ bằng cách sao chép ma trận `a` theo các cột và ma trận `b` theo các hàng trước khi cộng chúng theo từng phần tử.

```{.python .input}
a + b
```

## Tiết kiệm Bộ nhớ

[**Việc thực hiện các phép toán có thể khiến bộ nhớ mới được cấp phát để chứa kết quả.**] Chẳng hạn, nếu ta viết `Y = X + Y`, ta bỏ tham chiếu tới tensor mà `Y` từng trỏ tới và thay vào đó cho `Y` trỏ tới vùng nhớ mới được cấp phát. Ta có thể minh họa vấn đề này bằng hàm `id()` của Python, hàm cho biết địa chỉ chính xác của đối tượng được tham chiếu trong bộ nhớ. Lưu ý rằng sau khi chạy `Y = Y + X`, `id(Y)` trỏ tới một vị trí khác. Đó là vì Python trước tiên tính `Y + X`, cấp phát bộ nhớ mới cho kết quả, rồi cho `Y` trỏ tới vị trí mới này trong bộ nhớ.

```{.python .input}
before = id(Y)
Y = Y + X
id(Y) == before
```

Điều này có thể không mong muốn vì hai lý do. Thứ nhất, ta không muốn lúc nào cũng cấp phát bộ nhớ một cách không cần thiết. Trong học máy, ta thường có hàng trăm megabyte tham số và cập nhật tất cả chúng nhiều lần mỗi giây. Bất cứ khi nào có thể, ta muốn thực hiện các cập nhật này *tại chỗ* (in place). Thứ hai, ta có thể trỏ tới cùng các tham số từ nhiều biến. Nếu không cập nhật tại chỗ, ta phải cẩn thận cập nhật mọi tham chiếu này, kẻo gây rò rỉ bộ nhớ hoặc vô tình tham chiếu tới các tham số đã cũ.

May mắn thay, (**thực hiện các phép toán tại chỗ**) rất dễ. Ta có thể gán kết quả của một phép toán cho một mảng `Y` đã được cấp phát trước bằng ký hiệu cắt lát: `Y[:] = <biểu thức>`. Để minh họa khái niệm này, ta ghi đè các giá trị của tensor `Z`, sau khi khởi tạo nó bằng `zeros_like` để có cùng kích thước với `Y`.

```{.python .input}
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

[**Nếu giá trị của `X` không được dùng lại trong các tính toán tiếp theo, ta cũng có thể dùng `X[:] = X + Y` hoặc `X += Y` để giảm chi phí bộ nhớ của phép toán.**]

```{.python .input}
before = id(X)
X += Y
id(X) == before
```

## Chuyển đổi sang các Đối tượng Python khác

[**Chuyển đổi sang một tensor NumPy (`ndarray`)**], hoặc ngược lại, rất dễ dàng. Tensor của torch và mảng NumPy sẽ dùng chung vùng nhớ bên dưới, và việc thay đổi một bên thông qua một phép toán tại chỗ cũng sẽ làm thay đổi bên kia.

```{.python .input}
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

Để (**chuyển một tensor có kích thước 1 thành một số vô hướng Python**), ta có thể gọi hàm `item` hoặc các hàm có sẵn của Python.

```{.python .input}
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

## Tóm tắt

Lớp tensor là giao diện chính để lưu trữ và thao tác với dữ liệu trong các thư viện học sâu. Tensor cung cấp nhiều chức năng, gồm các hàm khởi tạo; lấy chỉ số và cắt lát; các phép toán cơ bản; cơ chế lan truyền; phép gán tiết kiệm bộ nhớ; và chuyển đổi qua lại với các đối tượng Python khác.

## Bài tập

1. Chạy code trong mục này. Thay mệnh đề điều kiện `X == Y` bằng `X < Y` hoặc `X > Y`, rồi xem bạn thu được loại tensor nào.
1. Thay hai tensor được tính theo từng phần tử trong cơ chế lan truyền bằng các tensor có kích thước khác, chẳng hạn tensor 3 chiều. Kết quả có giống như mong đợi không?

[Discussions](https://discuss.d2l.ai/t/27)
