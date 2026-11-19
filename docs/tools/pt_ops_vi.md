# PyTorch cơ bản: thao tác cấu trúc tensor

> Ghi chú của người biên soạn: trang này dịch từ ghi chú *Pytorch基础-张量结构操作* trong *dl_note* của **Harley Zhang** ([bản gốc](https://github.com/harleyszhang/dl_note/blob/main/2-deep_learning_basic/pytorch_basic/Pytorch基础-张量结构操作.md), giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)). Bản gốc thử trên PyTorch 1.7; mọi đoạn code ở đây chạy lại với **PyTorch 2.14** và in kết quả thật. Chỗ sai hoặc lỗi thời ghi ở ô **Biên tập**; mục 5.3 (bản gốc để "chờ cập nhật") do người biên soạn viết.

> "Cho cá không bằng cho cần câu": kiến thức nguyên bản mới tinh tuý nhất. Bài này chỉ là ghi chép về các thao tác cơ bản trên tensor; muốn hiểu sâu, nên đọc [tài liệu chính thức của PyTorch](https://pytorch.org/docs/stable/torch.html).

<!-- run -->
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
```

## 1 Thao tác cơ bản

Trong PyTorch, các phép trên tensor chia thành **thao tác cấu trúc** và **phép toán**, đúng như tên gọi: thao tác cấu trúc thay đổi cấu trúc của tensor, phép toán tính trên giá trị các phần tử.

+ Thao tác cấu trúc hay dùng: đổi chiều (`transpose`, `view`...), ghép và tách (`split`, `chunk`...), chỉ số và cắt lát (`index_select`, `gather`...).
+ Phép toán hay dùng: phép toán vô hướng, vector, ma trận ([bài phép toán](tools/pt_math_vi)).

### 1.1 Đổi kích thước: view và reshape

Cả hai đổi `shape` của tensor. **`view()` chỉ dùng được khi bố cục bộ nhớ của tensor tương thích với kích thước mới** (trường hợp hay gặp nhất là tensor liên tục, `contiguous`), còn `reshape()` dùng được cả khi không tương thích. Khi tương thích, `a.reshape()` trả về kết quả giống `a.view()`: một khung nhìn dùng chung bộ nhớ, không cấp phát mới. Khi không tương thích, `view()` báo lỗi, còn `reshape()` vẫn chạy nhưng chép dữ liệu sang vùng nhớ mới, tức trả về **bản sao** (tương đương gọi `contiguous()` rồi `view()`).

<!-- run -->
```python
a = torch.arange(6).reshape(2, 3)
v = a.view(3, 2); r = a.reshape(3, 2)
print("a liên tục:", a.is_contiguous(), "| view, reshape dùng chung bộ nhớ với a:",
      v.data_ptr() == a.data_ptr(), r.data_ptr() == a.data_ptr())
t = a.t()                                           # chuyển vị: không liên tục
print("a.t() liên tục:", t.is_contiguous(), "stride", t.stride())
try:
    t.view(6)
except RuntimeError as e:
    print("t.view(6) lỗi:", str(e)[:75])
r2 = t.reshape(6)
print("t.reshape(6) =", r2.tolist(), "| dùng chung bộ nhớ:", r2.data_ptr() == a.data_ptr())
r[0, 0] = 99                                        # reshape trả về khung nhìn: sửa r là sửa a
print("sau khi sửa r[0, 0]:", a.tolist())
```

<!-- output -->
```output
a liên tục: True | view, reshape dùng chung bộ nhớ với a: True True
a.t() liên tục: False stride (1, 3)
t.view(6) lỗi: view size is not compatible with input tensor's size and stride (at least o
t.reshape(6) = [0, 3, 1, 4, 2, 5] | dùng chung bộ nhớ: False
sau khi sửa r[0, 0]: [[99, 1, 2], [3, 4, 5]]
```
<!-- /output -->

| Tính chất | view | reshape |
| --- | --- | --- |
| **Phụ thuộc bố cục bộ nhớ** | Cần bố cục tương thích (thường là liên tục); không thì báo lỗi | Không phụ thuộc; chép sang vùng nhớ mới khi cần |
| **Hiệu năng** | Không bao giờ di chuyển dữ liệu | Như `view` khi được; chép khi không được |
| **Ngữ nghĩa** | Luôn là khung nhìn: sửa tensor mới là sửa tensor gốc | Có thể là khung nhìn hoặc bản sao, nên không được dựa vào việc sửa hay không sửa tensor gốc |

> **Biên tập:** Bảng gốc ghi `reshape` "tạo tensor mới, tensor gốc không bị ảnh hưởng". Như kết quả ở trên, khi bố cục cho phép, `reshape` trả về khung nhìn và sửa nó **có** làm đổi tensor gốc; tài liệu PyTorch khuyên không dựa vào việc nó trả về khung nhìn hay bản sao. Điều kiện của `view` cũng không chỉ là "liên tục": ví dụ `a.t().view(3, 2)` vẫn lỗi, nhưng tách một chiều liên tục bên trong một tensor không liên tục có thể vẫn `view` được.

> Đọc thêm [bài này](https://blog.csdn.net/Flag_ing/article/details/109129752) (tiếng Trung).

### 1.2 Nối tensor

`torch.concat` (bí danh của `torch.cat`) nối các tensor theo một chiều: `torch.cat(tensors, dim=0)`, với `tensors` là một dãy (list, tuple) tensor cần nối và `dim` là chiều nối.

<!-- run -->
```python
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
print(torch.concat([A, B], dim=0))
print(torch.concat([A, B], dim=1))
```

<!-- output -->
```output
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])
```
<!-- /output -->

## 2 Đổi chiều

### 2.1 unsqueeze và squeeze: thêm, bớt chiều

- `torch.squeeze(input, dim=None)`: bỏ các chiều có kích thước 1 (mặc định bỏ tất cả; chỉ định `dim` thì chỉ bỏ chiều đó nếu nó có kích thước 1).
- `torch.unsqueeze(input, dim)`: chèn một chiều kích thước 1 tại vị trí `dim`.

> "Squeeze" nghĩa là "ép, bóp".

<!-- run -->
```python
a = torch.rand(1, 1, 3, 3)
print(torch.squeeze(a).shape, a.squeeze(1).shape, a.squeeze(2).shape)   # squeeze(2): chiều 2 có kích thước 3, giữ nguyên
x = torch.rand(3, 3)
print(torch.unsqueeze(x, 0).shape, x.unsqueeze(-1).shape, x[None, :, None].shape)
```

<!-- output -->
```output
torch.Size([3, 3]) torch.Size([1, 3, 3]) torch.Size([1, 1, 3, 3])
torch.Size([1, 3, 3]) torch.Size([3, 3, 1]) torch.Size([1, 3, 1, 3])
```
<!-- /output -->

### 2.2 transpose và permute: hoán đổi chiều

`torch.transpose()` chỉ đổi chỗ hai chiều, còn `.permute()` sắp lại mọi chiều tuỳ ý. Cả hai trả về khung nhìn.

```python
transpose(dim0, dim1) → Tensor
permute(*dims) → Tensor
```

Trong CNN hay phải đổi chiều. Ví dụ tensor bốn chiều `[batch, channel, h, w]` (`NCHW`) muốn đưa `channel` ra cuối thành `[batch, h, w, channel]` (`NHWC`): với `transpose()` phải đổi ít nhất hai lần (đổi chiều 1 với 3, rồi 1 với 2), còn với `permute()` chỉ cần một lần:

<!-- run -->
```python
x = torch.rand(1, 3, 28, 32)
print(x.transpose(1, 3).shape)                    # [1, 32, 28, 3]
print(x.transpose(1, 3).transpose(1, 2).shape)    # [1, 28, 32, 3]
print(x.permute(0, 2, 3, 1).shape)                # [1, 28, 32, 3]
print(torch.equal(x.transpose(1, 3).transpose(1, 2), x.permute(0, 2, 3, 1)))
```

<!-- output -->
```output
torch.Size([1, 32, 28, 3])
torch.Size([1, 28, 32, 3])
torch.Size([1, 28, 32, 3])
True
```
<!-- /output -->

> **Biên tập:** Code gốc tạo biến `input` nhưng lại dùng `b` (báo `NameError`), và ghi kết quả của `permute(0, 2, 3, 1)` là `[1, 28, 28, 3]`; đúng là `[1, 28, 32, 3]`, trùng với cách đổi hai lần bằng `transpose`.

## 3 Chỉ số và cắt lát

### 3.1 Chỉ số, cắt lát thông thường

Cách đánh chỉ số và cắt lát tensor gần như giống hệt NumPy và list nhiều chiều của Python; có thể dùng chúng để sửa một phần phần tử. Cắt lát hỗ trợ bỏ trống tham số và dấu ba chấm (`...`).

<!-- run -->
```python
t = torch.randint(1, 10, [3, 3])
print(t)
print(t[0], t[2][2])          # hàng đầu; phần tử (2, 2)
print(t[0:2, :])              # hàng 0 tới 1, mọi cột
print(t[1:, -1])              # từ hàng 1 tới cuối, cột cuối
print(t[1:, ::2])             # từ hàng 1 tới cuối, các cột 0, 2, 4...
print(torch.rand(2, 3, 4, 5)[..., 0].shape)   # dấu ba chấm thay cho mọi chiều phía trước
```

<!-- output -->
```output
tensor([[7, 4, 2],
        [8, 1, 4],
        [5, 5, 6]])
tensor([7, 4, 2]) tensor(6)
tensor([[7, 4, 2],
        [8, 1, 4]])
tensor([4, 6])
tensor([[8, 4],
        [5, 6]])
torch.Size([2, 3, 4])
```
<!-- /output -->

Với cách lấy không theo quy luật, dùng `torch.index_select`, `torch.take`, `torch.gather`, `torch.masked_select` (hoặc chỉ số bằng tensor bool).

### 3.2 gather và index_select

> Cách dùng `gather` khá khó hiểu; sau khi đọc tài liệu chính thức và tài liệu trên mạng, tác giả có một số cách hiểu riêng.

1. `gather`: lấy dữ liệu theo chỉ số `index` dọc một chiều `dim` (Gathers values along an axis specified by dim):

```python
torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor
```

+ `input` (Tensor): tensor nguồn;
+ `dim` (int): chiều để đánh chỉ số;
+ `index` (LongTensor): chỉ số các phần tử cần lấy.

Lưu ý khi dùng `gather`:

+ `input` và `index` phải có **cùng số chiều**;
+ với mọi chiều $d \ne$ `dim`, phải có `index.size(d) <= input.size(d)`;
+ đầu ra có cùng kích thước với `index`; `input` và `index` không broadcast với nhau.

> **Biên tập:** Bản gốc viết điều kiện đầu là "`input.shape = index.shape`"; chỉ cần cùng **số chiều**, kích thước có thể nhỏ hơn (như ví dụ thứ nhất dưới đây, `index` có kích thước $1\times4$ còn `input` là $4\times4$).

Với tensor 3 chiều, định nghĩa chính thức của đầu ra:

```python
out[i][j][k] = input[index[i][j][k]][j][k]  # nếu dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # nếu dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # nếu dim == 2
```

Ví dụ với tensor 2 chiều:

<!-- run -->
```python
a = torch.arange(0, 16).view(4, 4)
index = torch.tensor([[0, 1, 2, 3]])
print(torch.gather(a, 0, index))            # out[0][j] = a[index[0][j]][j] -> a[0][0], a[1][1], a[2][2], a[3][3]: đường chéo
t = torch.tensor([[1, 2], [3, 4]])
print(torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]])))
```

<!-- output -->
```output
tensor([[ 0,  5, 10, 15]])
tensor([[1, 1],
        [4, 3]])
```
<!-- /output -->

Cách tính ví dụ thứ hai (`dim = 1`, nên `out[i][j] = input[i][index[i][j]]`):

```text
out[0][0] = input[0][index[0][0]] = input[0][0] = 1
out[0][1] = input[0][index[0][1]] = input[0][0] = 1
out[1][0] = input[1][index[1][0]] = input[1][1] = 4
out[1][1] = input[1][index[1][1]] = input[1][0] = 3
```

Tóm lại: **`gather` lấy dữ liệu bằng cách thay chỉ số ở chiều `dim` bằng giá trị trong `index`, giữ nguyên chỉ số ở các chiều khác.** Trực quan, nó sắp xếp lại phần tử trong từng hàng (với `dim=1`). Ứng dụng quen thuộc nhất: lấy xác suất của lớp đúng cho từng mẫu, `probs.gather(1, labels[:, None])`, chính là bước bên trong `nll_loss`:

<!-- run -->
```python
logits = torch.randn(4, 3); labels = torch.tensor([2, 0, 1, 2])
logp = F.log_softmax(logits, 1)
print(-logp.gather(1, labels[:, None]).mean(), F.nll_loss(logp, labels))
```

<!-- output -->
```output
tensor(1.6900) tensor(1.6900)
```
<!-- /output -->

2. Hiểu `gather` rồi thì `index_select` đơn giản hơn: trả về tập con của tensor theo các chỉ số `index` dọc chiều `dim`. Tensor kết quả có cùng số chiều với tensor gốc; chiều `dim` có kích thước bằng độ dài `index`, các chiều khác giữ nguyên.

```python
torch.index_select(input, dim, index, *, out=None) → Tensor
```

<!-- run -->
```python
x = torch.randn(3, 4)
indices = torch.tensor([0, 2])
print(x)
print(torch.index_select(x, 0, indices))     # hàng 0 và 2, tương đương x[indices]
print(torch.index_select(x, 1, indices))     # cột 0 và 2, tương đương x[:, indices]
```

<!-- output -->
```output
tensor([[ 0.2754,  0.0356,  0.1588, -0.5529],
        [ 0.2879,  1.4130, -1.3518, -0.7016],
        [-0.6087, -0.6644,  1.2476, -1.3823]])
tensor([[ 0.2754,  0.0356,  0.1588, -0.5529],
        [-0.6087, -0.6644,  1.2476, -1.3823]])
tensor([[ 0.2754,  0.1588],
        [ 0.2879, -1.3518],
        [-0.6087,  1.2476]])
```
<!-- /output -->

## 4 Ghép và tách

### 4.1 torch.cat và torch.stack

`torch.cat` và `torch.stack` ghép nhiều tensor thành một; `torch.split` tách một tensor thành nhiều. Khác biệt giữa hai hàm ghép: `torch.cat` **nối** theo một chiều có sẵn, không thêm chiều; `torch.stack` **xếp chồng** theo một chiều **mới**.

```python
# Nối các tensor theo chiều dim; mọi tensor phải cùng kích thước (trừ chiều nối) hoặc rỗng.
torch.cat(tensors, dim=0, *, out=None) → Tensor
# Xếp chồng các tensor theo một chiều mới; mọi tensor phải cùng kích thước.
torch.stack(tensors, dim=0, *, out=None) → Tensor
```

<!-- run -->
```python
a = torch.arange(0, 9).view(3, 3)
b = torch.arange(10, 19).view(3, 3)
c = torch.arange(20, 29).view(3, 3)
cat_abc = torch.cat([a, b, c], dim=0)
print(cat_abc.shape)
stack_abc = torch.stack([a, b, c], axis=0)   # trong torch, tên tham số dim và axis dùng thay nhau được
print(stack_abc.shape, stack_abc[1])
print(torch.chunk(cat_abc, 3, dim=0)[2])
```

<!-- output -->
```output
torch.Size([9, 3])
torch.Size([3, 3, 3]) tensor([[10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]])
tensor([[20, 21, 22],
        [23, 24, 25],
        [26, 27, 28]])
```
<!-- /output -->

Ngoài ra `torch.hstack` xếp các tensor theo chiều ngang: với tensor một chiều là nối thẳng các phần tử; với tensor hai chiều trở lên là nối theo chiều thứ hai (chiều cột), tức dạng tắt của `torch.cat(tensors, dim=1)`. `torch.vstack` thì nối theo chiều đầu.

<!-- run -->
```python
a = torch.tensor([[1], [2], [3]]); b = torch.tensor([[4], [5], [6]])
print(torch.hstack((a, b)))
print(torch.hstack((torch.tensor([1, 2]), torch.tensor([3]))), torch.vstack((a, b)).shape)
```

<!-- output -->
```output
tensor([[1, 4],
        [2, 5],
        [3, 6]])
tensor([1, 2, 3]) torch.Size([6, 1])
```
<!-- /output -->

> **Biên tập:** Bản gốc minh hoạ `hstack` bằng một ảnh chụp có chú thích tiếng Trung; đã thay bằng kết quả chạy ở trên.

### 4.2 torch.split và torch.chunk

`torch.split()` và `torch.chunk()` là phép ngược của `torch.cat()`. `split()` tách tensor thành các khối, mỗi khối là một khung nhìn của tensor gốc:

```python
torch.split(tensor, split_size_or_sections, dim=0)
```

- `split_size_or_sections` là số nguyên: tách thành các khối bằng nhau có kích thước đó (khối cuối nhỏ hơn nếu không chia hết);
- là list: tách thành `len(list)` khối với kích thước theo list.

`chunk()` tách tensor theo chiều `dim` thành **số khối** `chunks` cho trước, trả về tuple:

```python
torch.chunk(input, chunks, dim=0) → tuple of Tensors
```

<!-- run -->
```python
a = torch.arange(10).reshape(5, 2)
print([t.tolist() for t in torch.split(a, 2)])          # khối kích thước 2: [2, 2, 1] hàng
print([t.shape[0] for t in torch.split(a, [1, 4])])     # khối 1 hàng và 4 hàng
print([t.tolist() for t in torch.chunk(a, 2, dim=1)])   # 2 khối theo cột
print([t.shape[0] for t in torch.chunk(torch.arange(5), 3)])  # 5 phần tử chia 3 khối: [2, 2, 1]
```

<!-- output -->
```output
[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9]]]
[1, 4]
[[[0], [2], [4], [6], [8]], [[1], [3], [5], [7], [9]]]
[2, 2, 1]
```
<!-- /output -->

## 5 Các phép liên quan tới tích chập

### 5.1 Các cách lấy mẫu lên

Lấy mẫu lên có thể chia thành ba loại:

1. Dựa trên nội suy: láng giềng gần nhất (`nearest`), song tuyến (`bilinear`), song lập phương (`bicubic`)..., các phương pháp xử lý ảnh truyền thống.
2. Dựa trên học sâu: tích chập chuyển vị (còn gọi là giải chập, `ConvTranspose2d`), pixel shuffle...
3. `Unpooling` (điền số 0 hoặc sao chép đơn giản).

> Chất lượng: láng giềng gần nhất < song tuyến < song lập phương. Tốc độ: láng giềng gần nhất > song tuyến > song lập phương.

### 5.2 Hàm lấy mẫu F.interpolate

> `nn.Upsample` vẫn còn trong PyTorch (là một lớp `nn.Module` bọc quanh `F.interpolate`); `torch.nn.functional.interpolate` là dạng hàm, một hàm làm được cả lấy mẫu lên lẫn lấy mẫu xuống theo nhu cầu.

```python
F.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None,
              recompute_scale_factor=None, antialias=False)
```

+ `input` (Tensor): tensor vào;
+ `size`: kích thước ra, dạng `([D_out], [H_out], W_out)`; chọn một trong hai với `scale_factor`;
+ `scale_factor`: hệ số phóng theo chiều sâu, cao, rộng; một số cho mọi chiều hoặc tuple cho từng chiều;
+ `mode`: cách lấy mẫu: `nearest` (mặc định), `linear`, `bilinear`, `bicubic`, `trilinear`, `area`, `nearest-exact`;
+ `align_corners`: nếu `True`, điểm ảnh ở các góc của ảnh vào và ảnh ra được căn thẳng nhau; chỉ có tác dụng với `linear`, `bilinear`, `bicubic`, `trilinear`; mặc định `False`.

<!-- run -->
```python
x = torch.rand(1, 3, 224, 224)
y = F.interpolate(x, scale_factor=(2, 2), mode='bilinear').squeeze(0)
print(y.shape)
small = torch.tensor([[[[1., 2.], [3., 4.]]]])
print(F.interpolate(small, scale_factor=2, mode='nearest')[0, 0])
print("align_corners=True :", F.interpolate(small, size=(4, 4), mode='bilinear', align_corners=True)[0, 0, 0].tolist())
print("align_corners=False:", F.interpolate(small, size=(4, 4), mode='bilinear', align_corners=False)[0, 0, 0].tolist())
```

<!-- output -->
```output
torch.Size([3, 448, 448])
tensor([[1., 1., 2., 2.],
        [1., 1., 2., 2.],
        [3., 3., 4., 4.],
        [3., 3., 4., 4.]])
align_corners=True : [1.0, 1.3333332538604736, 1.6666667461395264, 2.0]
align_corners=False: [1.0, 1.25, 1.75, 2.0]
```
<!-- /output -->

> **Biên tập:** Code gốc nhân đầu vào với 2 (`x * 2`, không cần thiết) và ghi kết quả là `torch.Size([3, 224, 224])`; phóng 2 lần thì đúng là `[3, 448, 448]`. Hai dòng cuối (hàng đầu của ảnh $2\times2$ phóng thành $4\times4$) cho thấy tác dụng của `align_corners`: với `True`, tâm các điểm ảnh ở góc được căn thẳng nên các điểm ở giữa nội suy đều trên đoạn $[1, 2]$; với `False`, mỗi điểm ảnh được coi là một ô vuông, các điểm ra nằm ở tâm các ô nhỏ, nên giá trị là 1; 1,25; 1,75; 2.

### 5.3 Tích chập chuyển vị nn.ConvTranspose2d

Tích chập chuyển vị (có khi gọi là "giải chập", deconvolution; tác giả cho rằng cách gọi này không chuẩn, vì nó không phải phép ngược của tích chập) là một loại tích chập đặc biệt: về mặt tính toán, nó tương đương **chèn $s-1$ số 0 giữa các điểm của đầu vào, đệm $k-1-p$ ở biên**, rồi tính tích chập thông thường (bước trượt 1) với kernel **xoay 180°**. Kích thước ra: $(\text{in}-1)s-2p+k$.

> **Biên tập:** Bản gốc chỉ có một câu mô tả rồi ghi "【chờ cập nhật】"; phần dưới do người biên soạn viết.

Tên "chuyển vị" đến từ cách viết tích chập thành phép nhân ma trận: nếu tích chập là $\boldsymbol y = C\boldsymbol x$ với $C$ là ma trận thưa, thì tích chập chuyển vị là $\boldsymbol x' = C^\top\boldsymbol y$. Nó đưa kích thước về như đầu vào của tích chập, nhưng **không** khôi phục giá trị; chính vì vậy chiều ngược của tích chập (gradient theo đầu vào) là một tích chập chuyển vị.

<!-- run -->
```python
torch.manual_seed(0)
x = torch.randn(1, 1, 3, 3)
up = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
y = up(x)
print("kích thước:", tuple(y.shape), "| công thức (3-1)*2 - 2*1 + 3 =", (3 - 1) * 2 - 2 + 3)

# 1) chèn số 0 + đệm + tích chập thường với kernel xoay 180°
s, k, p = 2, 3, 1
z = torch.zeros(1, 1, (3 - 1) * s + 1, (3 - 1) * s + 1); z[..., ::s, ::s] = x
w = up.weight.detach()                                  # ConvTranspose2d lưu trọng số dạng (C_in, C_out, k, k)
y1 = F.conv2d(F.pad(z, [k - 1 - p] * 4), torch.rot90(w, 2, (2, 3)).transpose(0, 1))
print("cách 1 (chèn 0 + tích chập):", torch.allclose(y, y1, atol=1e-6))

# 2) nhân với ma trận chuyển vị của tích chập cùng tham số
conv = nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False)
with torch.no_grad():
    conv.weight.copy_(w)
C = torch.stack([conv(e.view(1, 1, 5, 5)).flatten() for e in torch.eye(25)], 1)   # C: (9, 25), y = C x
y2 = (C.T @ x.flatten()).view(1, 1, 5, 5)
print("cách 2 (C^T x):", torch.allclose(y, y2, atol=1e-6), "| C có", int((C != 0).sum()), "phần tử khác 0 trên", C.numel())
```

<!-- output -->
```output
kích thước: (1, 1, 5, 5) | công thức (3-1)*2 - 2*1 + 3 = 5
cách 1 (chèn 0 + tích chập): True
cách 2 (C^T x): True | C có 49 phần tử khác 0 trên 225
```
<!-- /output -->

## Tài liệu tham khảo

+ [pytorch演示卷积和反卷积运算](https://blog.csdn.net/qq_37879432/article/details/80297263)
+ [torch.Tensor](https://pytorch.org/docs/stable/tensors.html)
+ [PyTorch学习笔记(10)——上采样和PixelShuffle](https://blog.csdn.net/g11d111/article/details/82855946)
+ [反卷积 Transposed convolution](https://zhuanlan.zhihu.com/p/124626648)
+ [PyTorch中的转置卷积详解——全网最细](https://blog.csdn.net/w55100/article/details/106467776)
+ [4-1,张量的结构操作](https://github.com/lyhue1991/eat_pytorch_in_20_days/blob/master/4-1,%E5%BC%A0%E9%87%8F%E7%9A%84%E7%BB%93%E6%9E%84%E6%93%8D%E4%BD%9C.md) (*eat pytorch in 20 days*)
+ Dumoulin, Visin. [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285). 2016.
