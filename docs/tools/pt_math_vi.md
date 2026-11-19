# PyTorch cơ bản: phép toán trên tensor

> Ghi chú của người biên soạn: trang này dịch từ ghi chú *Pytorch基础-张量数学运算* trong *dl_note* của **Harley Zhang** ([bản gốc](https://github.com/harleyszhang/dl_note/blob/main/2-deep_learning_basic/pytorch_basic/Pytorch基础-张量数学运算.md), giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)). Mọi đoạn code chạy lại với **PyTorch 2.14**; script `tensor_math.py` trong thư mục `src` của bản gốc được dùng ở mục 3.4. Chỗ sai ghi ở ô **Biên tập**; mục 4 do người biên soạn viết thêm.

Phép toán trên tensor là tính toán trên giá trị các phần tử; hay gặp gồm phép toán vô hướng (từng phần tử), phép toán vector (rút gọn theo chiều) và phép toán ma trận.

<!-- run -->
```python
import torch
torch.manual_seed(0)
```

## 1 Hiểu số chiều của tensor

Trong PyTorch, số chiều (còn gọi là "hạng", rank) của tensor quyết định cấu trúc và hình dạng dữ liệu:

- tensor 1D: vector, ví dụ vector độ dài 5 `[1, 2, 3, 4, 5]`;
- tensor 2D: ma trận, ví dụ kích thước `(3, 4)`;
- tensor 3D: hay dùng trong NLP, kích thước `(batch_size, sequence_length, hidden_size)`;
- tensor 4D: hay dùng trong CV, kích thước `(batch_size, channels, height, width)`.

Với mảng hai chiều $M$ hàng $N$ cột, $M$ là chiều 0 (số hàng), $N$ là chiều 1 (số cột). Với tensor phức tạp hơn, đọc số chiều bằng mắt thế nào?

**Quy tắc**: số tầng dấu `[` lồng nhau là số chiều. Số phần tử trong cặp ngoặc ngoài cùng là kích thước chiều 0, rồi đi dần vào trong.

<!-- run -->
```python
tensor = torch.tensor([[[0.6238, -0.9315, 0.2173, 0.1954, -1.1565],
                        [0.4559, 0.1531, 0.4178, 1.0225, 0.5923],
                        [0.0499, 0.4024, -1.2547, -0.5042, -0.0231],
                        [-1.1253, 0.3145, 0.8796, 0.4516, -0.0915]],
                       [[1.5794, -0.6367, -0.2559, 0.1237, -0.1951],
                        [0.1012, 0.0357, -0.5699, 1.0983, -0.2084],
                        [-0.7019, 0.5872, 0.7736, 0.7423, -0.7894],
                        [-0.3248, -0.5316, 1.2029, 0.2852, -0.4565]],
                       [[-0.0073, 1.4143, -0.1859, -0.7211, -0.8652],
                        [-0.3173, -0.4816, 0.1174, -0.1554, 0.9385],
                        [0.1283, -0.6547, 0.3687, -0.1948, 0.7754],
                        [-0.2185, -1.0437, 1.5963, -0.3284, -0.3654]]])
print(tensor.shape)
```

<!-- output -->
```output
torch.Size([3, 4, 5])
```
<!-- /output -->

- `[` ngoài cùng chứa 3 danh sách con → chiều 0 có kích thước 3;
- `[` tầng thứ hai chứa 4 danh sách con → chiều 1 có kích thước 4;
- `[` tầng thứ ba chứa 5 phần tử → chiều 2 có kích thước 5.

Vậy đây là tensor 3 chiều, kích thước `[3, 4, 5]`.

## 2 Hiểu tham số dim

**Tham số `dim` của các hàm toán học trong PyTorch thường nghĩa là thực hiện phép toán dọc theo chiều `dim`**: tính tổng, trung bình, tích luỹ..., hoặc xoá, thêm chiều `dim`. Với `x` kích thước `[2, 5, 3]`:

- `dim = 0`: dọc theo chiều có 2 phần tử;
- `dim = 1`: dọc theo chiều có 5 phần tử;
- `dim = 2`: dọc theo chiều có 3 phần tử.

Chiều được rút gọn sẽ **biến mất** khỏi kích thước kết quả: `mean(dim=0)` của tensor `[2, 5, 3]` có kích thước `[5, 3]`.

<!-- run -->
```python
x = torch.tensor([[[8., 1., 6.], [1., 5., 9.], [5., 7., 1.], [5., 1., 2.], [8., 4., 4.]],
                  [[3., 3., 7.], [7., 4., 3.], [7., 7., 5.], [8., 3., 9.], [1., 1., 8.]]])
print(x.shape)
y = torch.mean(x, dim=0)          # y[0,0] = (x[0,0,0] + x[1,0,0]) / 2 = (8 + 3) / 2 = 5.5
print(y)                          # y[2,2] = (x[0,2,2] + x[1,2,2]) / 2 = (1 + 5) / 2 = 3.0
y = torch.mean(x, dim=2)          # y[1,4] = (x[1,4,0] + x[1,4,1] + x[1,4,2]) / 3 = (1 + 1 + 8) / 3
print(y)
```

<!-- output -->
```output
torch.Size([2, 5, 3])
tensor([[5.5000, 2.0000, 6.5000],
        [4.0000, 4.5000, 6.0000],
        [6.0000, 7.0000, 3.0000],
        [6.5000, 2.0000, 5.5000],
        [4.5000, 2.5000, 6.0000]])
tensor([[5.0000, 5.0000, 4.3333, 2.6667, 5.3333],
        [4.3333, 4.6667, 6.3333, 6.6667, 3.3333]])
```
<!-- /output -->

> **Biên tập:** Chú thích gốc tính `y[2, 2] = (x[0,2,2] + x[1,2,2])` thiếu phép chia 2 (kết quả 3,0 thì đúng). Bản gốc dùng `torch.randint` không đặt seed nên đoạn in kết quả không tái lập được; ở đây ghi thẳng giá trị của `x` như trong bài.

## 3 Phép rút gọn

Phép rút gọn (reduction) là tổng hợp theo nhóm; kết quả bị giảm chiều.

### 3.1 torch.mean

`torch.mean` tính trung bình của tensor theo chiều chỉ định:

```python
torch.mean(input, dim, keepdim=False, *, dtype=None) -> Tensor
```

- `input`: tensor vào;
- `dim`: chiều (một số hoặc tuple) cần lấy trung bình;
- `keepdim`: có giữ chiều đã rút gọn (với kích thước 1) không, mặc định `False`;
- `dtype`: kiểu dữ liệu của kết quả.

1. Hiểu theo cách tính: **tính trung bình dọc (băng qua) chiều `dim`**.
   - Với ma trận 2D, `dim = 0` là băng qua các hàng, tức tính trung bình mọi phần tử trong mỗi cột.
   - Với tensor 3D của NLP `(batch_size, sequence_length, embedding_size)`, `dim = 2` là trung bình theo chiều embedding: với mỗi vị trí (batch, token), lấy trung bình các phần tử embedding. Ví dụ tensor `(4, 16, 4)`, tại vị trí $(0, 0, :)$: $\operatorname{mean}(x[0, 0, :]) = \frac{x[0,0,0] + x[0,0,1] + x[0,0,2] + x[0,0,3]}{4}$.
   - Với tensor 4D của CV `(batch_size, channels, height, width)`, `dim = 0` là trung bình theo batch, tức lấy trung bình mọi mẫu trong batch. Ví dụ tensor `(2, 3, 3, 3)`, tại vị trí $(0, 0, 0)$: $\operatorname{mean}(x[:, 0, 0, 0]) = \frac{x[0, 0, 0, 0] + x[1, 0, 0, 0]}{2}$.
2. Hiểu theo kích thước kết quả:
   - tensor 3D của NLP, `dim = 2`: bỏ chiều đó, kết quả có kích thước `(batch_size, sequence_length)`;
   - tensor 4D của CV, `dim = 0`: kết quả có kích thước `(channels, height, width)`.
3. Tham số `keepdim`:
   - `keepdim=False` (mặc định): chiều đã rút gọn bị bỏ, kết quả ít chiều hơn;
   - `keepdim=True`: chiều đó được giữ với kích thước 1, rất tiện cho các phép tiếp theo cần broadcast (như chuẩn hoá `x - x.mean(dim, keepdim=True)`).

<!-- run -->
```python
x = torch.randn(4, 16, 8)
print(x.mean(dim=2).shape, x.mean(dim=2, keepdim=True).shape, x.mean(dim=(0, 1)).shape)
xc = x - x.mean(dim=2, keepdim=True)          # keepdim để broadcast đúng
print("trung bình sau khi trừ:", float(xc.mean(dim=2).abs().max()))
```

<!-- output -->
```output
torch.Size([4, 16]) torch.Size([4, 16, 1]) torch.Size([8])
trung bình sau khi trừ: 7.450580596923828e-08
```
<!-- /output -->

### 3.2 torch.sum

`torch.sum` tính tổng theo chiều chỉ định:

<!-- run -->
```python
x = torch.randn(4, 5)
print(x)
print(x.sum(dim=0), x.sum(dim=0).shape)
print(x.sum(dim=1), x.sum(dim=1).shape)
```

<!-- output -->
```output
tensor([[-0.5461, -0.6302, -0.6347,  0.9747,  0.0362],
        [-0.3422, -0.3817, -0.0569, -0.8770,  0.5223],
        [ 0.4620, -1.3293,  0.5109, -0.2860,  0.3351],
        [ 1.1719, -0.4039, -0.3263,  0.7482, -0.1953]])
tensor([ 0.7457, -2.7450, -0.5069,  0.5599,  0.6982]) torch.Size([5])
tensor([-0.8001, -1.1355, -0.3073,  0.9947]) torch.Size([4])
```
<!-- /output -->

Với `dim = 0`, ta cộng dọc theo chiều 0 (cộng các hàng lại với nhau); `sum` là phép rút gọn nên chiều đó biến mất, kết quả có kích thước `[5]`.

### 3.3 torch.max

`torch.max()` lấy giá trị lớn nhất, có ba cách dùng:

- lấy giá trị lớn nhất của cả tensor;
- lấy giá trị lớn nhất theo một chiều, kèm chỉ số;
- so từng phần tử của hai tensor và lấy giá trị lớn hơn (giống `torch.maximum()`).

```python
torch.max(input) -> Tensor
torch.max(input, dim, keepdim=False, *, out=None) -> (values, indices)
torch.max(input, other, *, out=None) -> Tensor        # other cũng là tensor
```

- Cách thứ nhất: trả về **giá trị lớn nhất toàn cục** của tensor.
- Cách thứ hai: dùng khi cần giá trị lớn nhất theo một chiều; kết quả gồm hai phần: các giá trị lớn nhất (`values`) và chỉ số của chúng (`indices`). Phép này dùng nhiều trong lượng tử hoá (tìm khoảng giá trị theo từng kênh) và để lấy lớp dự đoán (`argmax`).
- Cách thứ ba: so từng phần tử của `input` và `other`, trả về giá trị **lớn hơn** ở mỗi vị trí.

> **Biên tập:** Bản gốc ghi cách thứ ba "giống `torch.minimum()`", chữ ký `torch.min(input, other)` và "trả về giá trị nhỏ hơn ở mỗi vị trí"; với `torch.max` thì là giá trị lớn hơn, tương đương `torch.maximum()`, như kết quả ví dụ của chính bản gốc. Bản gốc cũng viết cách thứ hai trả về "chỉ số rồi giá trị"; thứ tự đúng là `values` trước, `indices` sau.

<!-- run -->
```python
x = torch.tensor([[1, 6, 5, 7, 7], [5, 3, 2, 2, 6]])
y = torch.tensor([[8, 3, 8, 7, 4], [11, 6, 2, 5, 3]])
print(torch.max(x, y))
print(torch.equal(torch.max(x, y), torch.maximum(x, y)))
print(torch.max(x))
max_x = torch.max(x, dim=0)
print(max_x)
print(max_x[0], max_x.values, max_x.indices)
```

<!-- output -->
```output
tensor([[ 8,  6,  8,  7,  7],
        [11,  6,  2,  5,  6]])
True
tensor(7)
torch.return_types.max(
values=tensor([5, 6, 5, 7, 7]),
indices=tensor([1, 0, 0, 0, 0]))
tensor([5, 6, 5, 7, 7]) tensor([5, 6, 5, 7, 7]) tensor([1, 0, 0, 0, 0])
```
<!-- /output -->

`torch.min` tương tự `torch.max`, nhưng trả về giá trị nhỏ nhất.

### 3.4 torch.cumprod

`torch.cumprod` tính tích luỹ theo chiều `dim`: trả về tensor cùng kích thước với `input`, mỗi phần tử là tích của chính nó và mọi phần tử đứng trước nó theo chiều đó.

```python
torch.cumprod(input, dim, *, dtype=None, out=None) -> Tensor
```

Ví dụ trong script `tensor_math.py` của bản gốc:

<!-- run -->
```python
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("tensor gốc:\n", b)
print("tích luỹ theo chiều 0 (dọc xuống, giữa các hàng):\n", torch.cumprod(b, dim=0))
print("tích luỹ theo chiều 1 (sang ngang, trong mỗi hàng):\n", torch.cumprod(b, dim=1))
y = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("ma trận 3x3, chiều 0:\n", torch.cumprod(y, dim=0))
```

<!-- output -->
```output
tensor gốc:
 tensor([[1, 2, 3],
        [4, 5, 6]])
tích luỹ theo chiều 0 (dọc xuống, giữa các hàng):
 tensor([[ 1,  2,  3],
        [ 4, 10, 18]])
tích luỹ theo chiều 1 (sang ngang, trong mỗi hàng):
 tensor([[  1,   2,   6],
        [  4,  20, 120]])
ma trận 3x3, chiều 0:
 tensor([[  1,   2,   3],
        [  4,  10,  18],
        [ 28,  80, 162]])
```
<!-- /output -->

Với ma trận $3\times3$ và `dim=0`: hàng đầu giữ nguyên; hàng hai bằng hàng đầu nhân hàng hai $[1\cdot4, 2\cdot5, 3\cdot6] = [4, 10, 18]$; hàng ba bằng kết quả hàng hai nhân hàng ba $[4\cdot7, 10\cdot8, 18\cdot9] = [28, 80, 162]$.

> **Biên tập:** Bản gốc giải thích ví dụ $3\times3$ bằng một ảnh chụp có chú thích tiếng Trung; đã thay bằng kết quả chạy và đoạn giải thích trên.

**Tổng kết**: so sánh kích thước kết quả của các hàm trên:

<!-- run -->
```python
x = torch.tensor([[2., 3., 4., 5., 6.], [9., 8., 7., 6., 5.]])   # kích thước [2, 5]
print(torch.cumprod(x, dim=0))                       # [2, 5]
print(torch.cumprod(x, dim=1))                       # [2, 5]
print(torch.min(x, dim=0))                           # [5]
print(torch.max(x, dim=1))                           # [2]
print(torch.mean(x, dim=0, keepdim=True))            # [1, 5]
```

<!-- output -->
```output
tensor([[ 2.,  3.,  4.,  5.,  6.],
        [18., 24., 28., 30., 30.]])
tensor([[2.0000e+00, 6.0000e+00, 2.4000e+01, 1.2000e+02, 7.2000e+02],
        [9.0000e+00, 7.2000e+01, 5.0400e+02, 3.0240e+03, 1.5120e+04]])
torch.return_types.min(
values=tensor([2., 3., 4., 5., 5.]),
indices=tensor([0, 0, 0, 0, 1]))
torch.return_types.max(
values=tensor([6., 9.]),
indices=tensor([4, 0]))
tensor([[5.5000, 5.5000, 5.5000, 5.5000, 5.5000]])
```
<!-- /output -->

## 4 Bổ sung: phép toán ma trận và broadcast

Mục này do người biên soạn viết thêm (bản gốc nêu "phép toán ma trận" ở đầu bài nhưng không trình bày).

**Broadcast**: khi hai tensor khác kích thước, PyTorch so kích thước từ chiều **cuối** ngược lên; hai chiều tương thích nếu bằng nhau hoặc một trong hai bằng 1 (hoặc không có); chiều bằng 1 được "kéo giãn" mà không chép dữ liệu.

**Nhân ma trận**: `A @ B` (hay `torch.matmul`) với tensor nhiều chiều coi hai chiều cuối là ma trận và broadcast các chiều phía trước (nhân theo batch). `torch.einsum` viết phép toán bằng ký hiệu chỉ số, tiện cho attention.

<!-- run -->
```python
a = torch.ones(4, 1, 3); b = torch.arange(5.).view(5, 1)
print("broadcast (4,1,3) + (5,1) ->", tuple((a + b).shape))
try:
    torch.ones(4, 3) + torch.ones(4)
except RuntimeError as e:
    print("(4,3) + (4,) lỗi:", str(e)[:70])

A = torch.randn(8, 2, 16, 32); B = torch.randn(32, 10)
print("matmul (8,2,16,32) @ (32,10) ->", tuple((A @ B).shape))

q = torch.randn(2, 4, 6, 8); k = torch.randn(2, 4, 6, 8)        # (batch, head, token, dim)
scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / 8 ** 0.5
print("einsum attention:", tuple(scores.shape), torch.allclose(scores, q @ k.transpose(-1, -2) / 8 ** 0.5, atol=1e-6))
```

<!-- output -->
```output
broadcast (4,1,3) + (5,1) -> (4, 5, 3)
(4,3) + (4,) lỗi: The size of tensor a (3) must match the size of tensor b (4) at non-si
matmul (8,2,16,32) @ (32,10) -> (8, 2, 16, 10)
einsum attention: (2, 4, 6, 6) True
```
<!-- /output -->

Phép toán theo từng phần tử (`+`, `*`, `torch.exp`...) và phép rút gọn có hậu tố `_` (như `x.add_(1)`) là phiên bản **tại chỗ**, sửa thẳng tensor; tiết kiệm bộ nhớ nhưng có thể làm hỏng đồ thị autograd nếu tensor đó cần cho lan truyền ngược.
