# Nén mô hình: Cắt tỉa mô hình bằng PyTorch

> Ghi chú của người biên soạn: trang này dịch từ ghi chú *dl_note* của **Harley Zhang** ([bản gốc](https://github.com/harleyszhang/dl_note/blob/main/5-model_compression/基于pytorch实现模型剪枝.md), [kho mã](https://github.com/harleyszhang/dl_note), giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0); phần dịch có chỉnh sửa). Số mục giữ như bản gốc. Chỗ bản gốc sai, thiếu hoặc đã lỗi thời được sửa ngay trong bài và ghi ở ô **Biên tập**. Các mục có chữ **Bổ sung** do người biên soạn viết thêm.

Trang [Code](dlnote/compress_prune_torch_code_vi) đi kèm chạy lại mọi ví dụ của bài (thay cho các ảnh chụp notebook của bản gốc), rồi cắt tỉa một CNN thật trên FashionMNIST.

## 1 Phân loại cắt tỉa

Cắt tỉa mô hình thực ra là một kỹ thuật nén mô hình loại bỏ các trọng số hoặc bias (weights/bias) "không cần thiết" khỏi mạng neuron. Tham số nào là "không cần thiết" thì hiện vẫn là một lĩnh vực đang được nghiên cứu.

### 1.1 Cắt tỉa không cấu trúc

Cắt tỉa không cấu trúc (Unstructured Pruning) là cắt từng phần tử của tham số, ví dụ một trọng số trong lớp kết nối đầy đủ, một phần tử của kernel trong lớp tích chập, hay một số thực (scaling float) trong một lớp tự định nghĩa. Điểm chính là **các trọng số bị cắt có thể nằm ở bất kỳ vị trí nào, không theo cấu trúc cụ thể nào, nên gọi là cắt tỉa không cấu trúc**.

> **Biên tập:** Bản gốc viết "đối tượng trọng số bị cắt là ngẫu nhiên". Vị trí bị cắt không có cấu trúc, nhưng cách chọn không nhất thiết ngẫu nhiên: `l1_unstructured` chọn theo độ lớn, chỉ `random_unstructured` mới chọn ngẫu nhiên.

### 1.2 Cắt tỉa có cấu trúc

Ngược với cắt tỉa không cấu trúc, cắt tỉa có cấu trúc cắt cả một cấu trúc tham số, ví dụ bỏ cả hàng hay cả cột trọng số, hoặc bỏ cả một bộ lọc (`Filter`) trong lớp tích chập.

### 1.3 Cắt tỉa cục bộ và toàn cục

Cắt tỉa có thể làm trên từng lớp (cục bộ, local) hoặc trên nhiều lớp/tất cả các lớp (toàn cục, global).

## 2 Cắt tỉa trong PyTorch

Hiện framework PyTorch hỗ trợ các cách cắt tỉa trọng số sau:

- **Random**: cắt ngẫu nhiên các tham số.
- **Magnitude**: cắt các tham số có độ lớn nhỏ nhất (ví dụ theo chuẩn L1 hoặc L2 của chúng).

Hai cách trên dễ cài đặt, tính toán đơn giản và áp dụng được mà không cần dữ liệu nào.

### 2.1 Cách PyTorch thực hiện cắt tỉa

Chức năng cắt tỉa nằm trong mô-đun `torch.nn.utils.prune`, mã ở tệp `torch/nn/utils/prune.py`. Các lớp cắt tỉa chính trong tệp là: `BasePruningMethod` (lớp cơ sở trừu tượng), `PruningContainer` (gộp nhiều lần cắt liên tiếp trên cùng một tham số), `Identity`, `RandomUnstructured`, `L1Unstructured`, `RandomStructured`, `LnStructured` và `CustomFromMask`.

Nguyên lý cắt tỉa dựa trên mặt nạ (Mask) của tensor. Mặt nạ là một tensor cùng hình dạng với tensor tham số; giá trị 1 (True) ở vị trí nào nghĩa là giữ trọng số ở vị trí đó, giá trị 0 (False) nghĩa là trọng số đó bị bỏ.

PyTorch chép tham số gốc `<param>` sang một tham số tên `<param>_orig`, và tạo một **buffer** để lưu mặt nạ cắt tỉa `<param>_mask`. Đồng thời, nó tạo một hàm callback `forward_pre_hook` ở mức mô-đun (hàm được gọi trước mỗi lần lan truyền xuôi của mô hình) để áp mặt nạ lên trọng số gốc, tức tính lại `<param> = <param>_orig * <param>_mask`.

> **Biên tập:** Bản gốc viết tên tham số là `<param>_original`; tên đúng là `<param>_orig` (ví dụ `weight_orig`), như trang Code in ra ở mục 1.

API và hướng dẫn cắt tỉa của PyTorch khá rối, nên tác giả lập bảng dưới để tổng kết API theo cách cắt và cách phân loại:

| | Cắt tỉa có cấu trúc | Cắt tỉa không cấu trúc |
| --- | --- | --- |
| **Toàn cục (global)** | `CustomFromMask` | \ |
| **Cục bộ (local)** | `RandomStructured`, `LnStructured` | `RandomUnstructured`, `L1Unstructured` |

> **Biên tập:** Bảng gốc là ảnh tiếng Trung, được gõ lại ở trên. Cách xếp của bảng gốc dễ gây hiểu lầm: hàm toàn cục duy nhất là `prune.global_unstructured`, và nó nhận các phương pháp **không cấu trúc** (`L1Unstructured`, `RandomUnstructured`). `CustomFromMask` không phải cắt tỉa toàn cục mà là áp một mặt nạ do người dùng tự tạo (`prune.custom_from_mask`) lên một tham số; với mặt nạ tự tạo, ta làm được bất kỳ kiểu cắt nào, kể cả có cấu trúc trên nhiều lớp.

Quy trình cắt tỉa mô hình trong PyTorch như sau:

1. Chọn phương pháp cắt tỉa (hoặc kế thừa `BasePruningMethod` để tự cài đặt phương pháp riêng).
2. Chỉ định mô-đun và tên tham số cần cắt.
3. Đặt tham số cho phương pháp cắt tỉa, ví dụ tỉ lệ cắt.

### 2.2 Cắt tỉa cục bộ

Cắt tỉa cục bộ trong PyTorch có hai loại, không cấu trúc và có cấu trúc. Đáng chú ý là cắt tỉa có cấu trúc chỉ hỗ trợ cục bộ, không hỗ trợ toàn cục.

#### 2.2.1 Cắt tỉa cục bộ không cấu trúc

1. **Cắt tỉa cục bộ không cấu trúc** (Local Unstructured Pruning) có hàm như sau:

```python
def random_unstructured(module, name, amount)
```

**Chức năng**:

Cắt tỉa **không cấu trúc** một tensor tham số trọng số. Hàm chọn **ngẫu nhiên** một số trọng số hoặc kết nối trong tensor để cắt, tỉ lệ cắt do người dùng chỉ định.

**Tham số**:

- `module` (nn.Module): lớp/mô-đun cần cắt tỉa, ví dụ `nn.Conv2d()` và `nn.Linear()`.
- `name` (str): tên tham số cần cắt, ví dụ `"weight"` hoặc `"bias"`.
- `amount` (int hoặc float): lượng cần cắt. Nếu là số thực trong khoảng 0–1 thì là tỉ lệ cắt; nếu là số nguyên thì là số tham số tuyệt đối cần cắt. Ví dụ `amount=0.2` nghĩa là chọn ngẫu nhiên 20% phần tử để cắt.

Ví dụ dùng hàm `random_unstructured`:

```python
import torch
import torch.nn.utils.prune as prune
conv = torch.nn.Conv2d(1, 1, 4)
prune.random_unstructured(conv, name="weight", amount=0.5)
conv.weight
"""
tensor([[[[-0.1703,  0.0000, -0.0000,  0.0690],
          [ 0.1411,  0.0000, -0.0000, -0.1031],
          [-0.0527,  0.0000,  0.0640,  0.1666],
          [ 0.0000, -0.0000, -0.0000,  0.2281]]]], grad_fn=<MulBackward0>)
"""
```

Có thể thấy một nửa số trọng số trong lớp conv đầu ra bằng `0`. (Cột `grad_fn=<MulBackward0>` cho thấy `weight` không còn là tham số mà là kết quả của phép nhân `weight_orig * weight_mask`.)

#### 2.2.2 Cắt tỉa cục bộ có cấu trúc

**Cắt tỉa cục bộ có cấu trúc** (Local Structured Pruning) có hai hàm:

```python
def random_structured(module, name, amount, dim)
def ln_structured(module, name, amount, n, dim, importance_scores=None)
```

**Chức năng**: khác với cắt tỉa không cấu trúc là bỏ từng trọng số kết nối, cắt tỉa có cấu trúc bỏ cả một lát trọng số theo một chiều (ví dụ cả một kênh).

**Tham số**: rất giống hàm cắt tỉa cục bộ không cấu trúc, khác duy nhất là phải chỉ định tham số `dim` (hàm `ln_structured` có thêm tham số `n`). `n` là bậc của chuẩn dùng để xếp hạng (ví dụ `n=2` là chuẩn L2), `dim` là chiều cắt tỉa.

Với `torch.nn.Linear` (trọng số có hình dạng `(out_features, in_features)`):

- `dim = 0`: bỏ một neuron (một hàng: mọi kết nối đi ra một đầu ra).
- `dim = 1`: bỏ mọi kết nối tới một đầu vào (một cột).

Với `torch.nn.Conv2d`:

- `dim = 0` (kênh): cắt tỉa kênh/bộ lọc.
- `dim = 1`: cắt tỉa kernel 2 chiều, tức các kernel nối với một kênh đầu vào.

#### 2.2.3 Mã ví dụ cắt tỉa cục bộ có cấu trúc

Trước khi viết mã, ta cần hiểu quan hệ giữa tham số của `Conv2d`, hình dạng kernel, các trục và tensor.

Hàm khởi tạo của `Conv2d`:

```python
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

Trọng số của tích chập thường trong PyTorch luôn có hình dạng `(C_out, C_in, kernel_height, kernel_width)`. Vì vậy với `Conv2d(2, 3, 3)` trong ví dụ dưới, trọng số có hình dạng `[3, 2, 3, 3]`: `dim = 0` ứng với số `3` đầu tiên (3 bộ lọc), `dim = 1` ứng với số `2` (2 kênh đầu vào). Chọn `dim` là trục nào thì sau khi cắt, các lát theo trục đó bị đặt bằng 0.

Hiểu các khái niệm trên rồi thì có thể dùng thực tế. Ví dụ với `dim=0`:

```python
conv = torch.nn.Conv2d(2, 3, 3)
norm1 = torch.norm(conv.weight, p=1, dim=[1,2,3])
print(norm1)
"""
tensor([1.9384, 2.3780, 1.8638], grad_fn=<NormBackward1>)
"""
prune.ln_structured(conv, name="weight", amount=1, n=2, dim=0)
print(conv.weight)
"""
tensor([[[[-0.0005,  0.1039,  0.0306],
          [ 0.1233,  0.1517,  0.0628],
          [ 0.1075, -0.0606,  0.1140]],

         [[ 0.2263, -0.0199,  0.1275],
          [-0.0455, -0.0639, -0.2153],
          [ 0.1587, -0.1928,  0.1338]]],


        [[[-0.2023,  0.0012,  0.1617],
          [-0.1089,  0.2102, -0.2222],
          [ 0.0645, -0.2333, -0.1211]],

         [[ 0.2138, -0.0325,  0.0246],
          [-0.0507,  0.1812, -0.2268],
          [-0.1902,  0.0798,  0.0531]]],


        [[[ 0.0000, -0.0000, -0.0000],
          [ 0.0000, -0.0000, -0.0000],
          [ 0.0000, -0.0000,  0.0000]],

         [[ 0.0000,  0.0000,  0.0000],
          [-0.0000,  0.0000,  0.0000],
          [-0.0000, -0.0000, -0.0000]]]], grad_fn=<MulBackward0>)
"""
```

Từ kết quả chạy có thể thấy rõ tham số của kênh cuối cùng (bộ lọc thứ ba) trong lớp tích chập đã bị bỏ (thành tensor `0`), vì bộ lọc này có chuẩn nhỏ nhất.

> **Biên tập:** Đoạn mã gốc in chuẩn **L1** (`p=1`) nhưng lại cắt theo chuẩn **L2** (`n=2`). Ở ví dụ này hai chuẩn tình cờ cùng chọn bộ lọc thứ ba, nhưng nói chung chúng có thể chọn khác nhau. Trang Code (mục 2) in cả chuẩn L2 để so cho đúng.

Trường hợp `dim = 1`:

```python
conv = torch.nn.Conv2d(2, 3, 3)
norm1 = torch.norm(conv.weight, p=1, dim=[0, 2, 3])
print(norm1)
"""
tensor([3.1487, 3.9088], grad_fn=<NormBackward1>)
"""
prune.ln_structured(conv, name="weight", amount=1, n=2, dim=1)
print(conv.weight)
"""
tensor([[[[ 0.0000, -0.0000, -0.0000],
          [-0.0000,  0.0000,  0.0000],
          [-0.0000,  0.0000, -0.0000]],

         [[-0.2140,  0.1038,  0.1660],
          [ 0.1265, -0.1650, -0.2183],
          [-0.0680,  0.2280,  0.2128]]],


        [[[-0.0000,  0.0000,  0.0000],
          [ 0.0000,  0.0000, -0.0000],
          [-0.0000, -0.0000, -0.0000]],

         [[-0.2087,  0.1275,  0.0228],
          [-0.1888, -0.1345,  0.1826],
          [-0.2312, -0.1456, -0.1085]]],


        [[[-0.0000,  0.0000,  0.0000],
          [ 0.0000, -0.0000,  0.0000],
          [ 0.0000, -0.0000,  0.0000]],

         [[-0.0891,  0.0946, -0.1724],
          [-0.2068,  0.0823,  0.0272],
          [-0.2256, -0.1260, -0.0323]]]], grad_fn=<MulBackward0>)
"""
```

Rõ ràng, theo chiều `dim=1`, lát thứ nhất có chuẩn nhỏ hơn, nên trong mỗi tensor hình dạng [2, 3, 3] (mỗi bộ lọc), ma trận [3, 3] thứ nhất bị bỏ (thành ma trận 0). Nói cách khác, mọi kernel nối với kênh đầu vào thứ nhất đều bị bỏ.

### 2.3 Cắt tỉa toàn cục không cấu trúc

Cắt tỉa cục bộ ở trên nhắm vào một lớp cụ thể; còn cắt tỉa toàn cục xem cả mô hình là một khối và bỏ một tỉ lệ (hoặc số lượng) tham số đã chỉ định. Vì vậy sau cắt tỉa toàn cục, độ thưa của từng lớp trong mô hình sẽ khác nhau.

Hàm cắt tỉa toàn cục không cấu trúc:

```python
# phiên bản v1.4.0
def global_unstructured(parameters, pruning_method, **kwargs)
# phiên bản v2.0.0-rc2 trở đi
def global_unstructured(parameters, pruning_method, importance_scores=None, **kwargs):
```

**Chức năng**: gộp mọi tham số được liệt kê (trên nhiều lớp) lại, rồi cắt một phần trong số đó theo phương pháp đã chọn, bất kể chúng thuộc lớp nào. Với `L1Unstructured`, đó là các tham số có trị tuyệt đối nhỏ nhất trong toàn bộ tập.

**Tham số**:

- `parameters` (Iterable các tuple (module, name)): danh sách tham số cần cắt, mỗi phần tử là một cặp (module, name).
- `pruning_method` (function): phương pháp cắt không cấu trúc, như `prune.L1Unstructured`, `prune.RandomUnstructured`, hoặc một phương pháp không cấu trúc tự cài đặt.
- `importance_scores`: điểm tầm quan trọng của từng tham số; nếu là None thì dùng chính giá trị tham số.
- `**kwargs`: tham số bổ sung truyền cho phương pháp cắt cụ thể, ví dụ `amount` là lượng cần cắt.

> **Biên tập:** Bản gốc mô tả hàm này là "chọn **ngẫu nhiên** một phần mọi tham số (cả trọng số lẫn bias)" và "hình như chính thức chỉ hỗ trợ `L1Unstructured`". Thực tế hàm chỉ cắt các tham số được liệt kê (ví dụ dưới chỉ liệt kê trọng số, không có bias), chọn theo phương pháp truyền vào (với `L1Unstructured` là theo độ lớn, không ngẫu nhiên), và nhận được mọi phương pháp **không cấu trúc**, kể cả `RandomUnstructured`.

Mã ví dụ dùng `global_unstructured`:

```python
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 kênh ảnh vào, 6 kênh ra, kernel vuông 3x3
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # ảnh 32x32 còn 6x6 sau hai lần conv 3x3 + gộp
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet()

parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
# Tính độ thưa của lớp tích chập và của cả mô hình.
# nelement() chính là Tensor.numel(): tổng số phần tử của tensor.
print(
    "Sparsity in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0))
        / float(model.conv1.weight.nelement())
    )
)
print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.conv1.weight == 0)
            + torch.sum(model.conv2.weight == 0)
            + torch.sum(model.fc1.weight == 0)
            + torch.sum(model.fc2.weight == 0)
            + torch.sum(model.fc3.weight == 0)
        )
        / float(
            model.conv1.weight.nelement()
            + model.conv2.weight.nelement()
            + model.fc1.weight.nelement()
            + model.fc2.weight.nelement()
            + model.fc3.weight.nelement()
        )
    )
)
# Kết quả chạy
"""
Sparsity in conv1.weight: 3.70%
Global sparsity: 20.00%
"""
```

Kết quả cho thấy dù độ thưa toàn cục của mô hình là `20%`, độ thưa của mỗi lớp không nhất thiết là 20%.

> **Biên tập:** Mã gốc khai báo `fc1 = nn.Linear(16 * 5 * 5, 120)` với chú thích "ảnh 5x5", lấy từ phiên bản LeNet dùng kernel 5×5. Với hai lớp conv 3×3 như ở đây, ảnh 32×32 còn $6 \times 6$ khi vào `fc1`, nên đã sửa thành `16 * 6 * 6` (giống hướng dẫn chính thức của PyTorch); mã gốc báo lỗi kích thước ngay khi chạy lượt xuôi. Mã gốc còn có hai dòng thừa (`device = ...` và `model = LeNet().to(device=device)` rồi lại `model = LeNet()`), đã bỏ. Con số 3,70% phụ thuộc trọng số khởi tạo ngẫu nhiên; trang Code in kết quả của lần chạy với hạt giống cố định.

## 3 Tổng kết

PyTorch còn cung cấp một số hàm hỗ trợ:

1. `torch.nn.utils.prune.is_pruned(module)`: kiểm tra mô-đun đã bị cắt tỉa chưa.
2. `torch.nn.utils.prune.remove(module, name)`: **biến việc cắt tỉa thành vĩnh viễn**: gán `<param> = <param>_orig * <param>_mask` thành tham số thường, rồi xoá `<param>_orig`, `<param>_mask` và hook. Các trọng số đã bị cắt vẫn bằng 0.

> **Biên tập:** Bản gốc viết `prune.remove` "gỡ thao tác cắt tỉa, khôi phục hình dạng và giá trị ban đầu của tham số". Điều này **sai**: `remove` không khôi phục trọng số đã cắt mà làm ngược lại, gộp mặt nạ vào trọng số vĩnh viễn (tài liệu PyTorch: *"make the pruning permanent"*). Trang Code (mục 1) kiểm tra: sau `remove`, số phần tử bằng 0 không đổi. Muốn lưu một mô hình đã cắt tỉa gọn gàng (không kèm bản sao `weight_orig` và mặt nạ, vốn làm tệp lưu to gấp đôi), phải gọi `remove` trước khi lưu.

Dù PyTorch có sẵn `API` cắt tỉa và hỗ trợ một số phương pháp cắt tỉa không cấu trúc và có cấu trúc, `API` khá rối và tài liệu mô tả chưa rõ, nên sau này tác giả sẽ dùng thêm công cụ mã nguồn mở `nni` của Microsoft để cắt tỉa mô hình.

Có thể tham khảo thêm các phương pháp cắt tỉa trong kho `github`: [Model-Compression](https://github.com/tangchen2/Model-Compression).

> **Biên tập (bổ sung):** Lưu ý quan trọng nhất khi dùng `torch.nn.utils.prune`: nó **chỉ đặt trọng số bằng 0**, không làm mô hình nhỏ hơn hay nhanh hơn. Tensor vẫn dày đặc, phép tính vẫn đủ. Muốn nhanh hơn thật sự, với cắt tỉa có cấu trúc phải tạo lại mô hình với ít kênh hơn và chép trọng số còn lại sang (trang Code, mục 6, làm tay việc này). Thư viện [Torch-Pruning](https://github.com/VainF/Torch-Pruning) tự động làm việc đó, kể cả theo dõi các lớp phụ thuộc nhau (residual, concat). Với cắt tỉa không cấu trúc, phải dùng định dạng thưa và kernel thưa: thưa 2:4 trên GPU NVIDIA (trang Code, mục 7) là cách phổ biến nhất hiện nay.

## Tài liệu tham khảo

1. [How to Prune Neural Networks with PyTorch](https://towardsdatascience.com/how-to-prune-neural-networks-with-pytorch-ebef60316b91)
2. [Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html), PyTorch.
3. [PyTorch Pruning](https://leimao.github.io/blog/PyTorch-Pruning/), Lei Mao.
4. *(Bổ sung)* [Torch-Pruning](https://github.com/VainF/Torch-Pruning): cắt tỉa có cấu trúc tự động theo đồ thị phụ thuộc.
5. *(Bổ sung)* [Accelerating BERT with semi-structured (2:4) sparsity](https://pytorch.org/tutorials/advanced/semi_structured_sparse.html), PyTorch.
