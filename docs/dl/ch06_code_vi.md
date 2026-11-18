# Chương 6 · Code từ đầu

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Chương 6 dùng `nn.Module` như một hộp đen biết tự tìm tham số, khởi tạo, lưu và nạp. Trang này tự viết một phiên bản nhỏ của `nn.Module` (khoảng 60 dòng) để thấy bên trong nó làm gì, rồi kiểm chứng hai chi tiết của chương: gradient của tham số dùng chung được cộng lại (mục 6.2.2), và lỗi `.data` trong `MyLinear` (mục 6.5.2).

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `torch`. Gradient vẫn do autograd của PyTorch tính; phần tự viết chỉ là cách tổ chức tham số và mô-đun.

## 1. `Parameter` và `Module` tự viết

<small class="viz-ref">Mục 6.1, 6.2</small>

Bí mật của `nn.Module` nằm ở phương thức `__setattr__`: mỗi khi ta viết `self.hidden = ...` trong `__init__`, mô-đun xem giá trị được gán là gì. Nếu là tham số thì ghi vào danh sách tham số, nếu là mô-đun con thì ghi vào danh sách mô-đun con. Nhờ vậy, `parameters()` chỉ cần duyệt đệ quy cây mô-đun, và đó là lý do sách nói ta "không cần lo về tham số" khi tự viết mô-đun.

<!-- run -->
```python
import math
import torch

class Parameter:
    """Một tensor cần học, được đánh dấu để Module nhận ra."""
    def __init__(self, data):
        self.data = data.detach().clone().requires_grad_(True)

class Module:
    def __init__(self):
        object.__setattr__(self, '_params', {})
        object.__setattr__(self, '_modules', {})

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def add_module(self, name, module):          # như MySequential của mục 6.1.2
        self._modules[name] = module

    def children(self):
        return self._modules.values()

    def named_parameters(self, prefix=''):
        for name, p in self._params.items():
            yield prefix + name, p
        for name, m in self._modules.items():
            yield from m.named_parameters(prefix + name + '.')

    def parameters(self):
        seen = set()
        for _, p in self.named_parameters():
            if id(p) not in seen:                # tham số dùng chung chỉ trả về một lần
                seen.add(id(p))
                yield p

    def apply(self, fn):                         # như net.apply(init_normal) của mục 6.3
        for m in self._modules.values():
            m.apply(fn)
        fn(self)
        return self

    def state_dict(self):                        # mục 6.6: chỉ lưu tham số, không lưu kiến trúc
        return {name: p.data.detach().clone() for name, p in self.named_parameters()}

    def load_state_dict(self, sd):
        for name, p in self.named_parameters():
            with torch.no_grad():
                p.data.copy_(sd[name])

    def zero_grad(self):
        for p in self.parameters():
            p.data.grad = None

    def __call__(self, *args):
        return self.forward(*args)
```

Ba "tầng" tối thiểu và một `Sequential`:

<!-- run -->
```python
class Linear(Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        bound = 1 / math.sqrt(d_in)              # cùng khoảng phân phối đều với nn.Linear mặc định
        self.weight = Parameter(torch.empty(d_out, d_in).uniform_(-bound, bound))
        self.bias = Parameter(torch.empty(d_out).uniform_(-bound, bound))
    def forward(self, X):
        return X @ self.weight.data.T + self.bias.data   # .data ở đây là tensor cần học, KHÔNG phải tensor.data của PyTorch

class ReLU(Module):
    def forward(self, X):
        return torch.relu(X)

class Sequential(Module):
    def __init__(self, *mods):
        super().__init__()
        for i, m in enumerate(mods):
            self.add_module(str(i), m)
    def forward(self, X):
        for m in self.children():
            X = m(X)
        return X
    def __getitem__(self, i):
        return self._modules[str(i)]

torch.manual_seed(0)
net = Sequential(Linear(20, 256), ReLU(), Linear(256, 10))
X = torch.rand(2, 20)
print("đầu ra:", tuple(net(X).shape))
for name, p in net.named_parameters():
    print(f"  {name:9s} {tuple(p.data.shape)}")
```

<!-- output -->
```output
đầu ra: (2, 10)
  0.weight  (256, 20)
  0.bias    (256,)
  2.weight  (10, 256)
  2.bias    (10,)
```
<!-- /output -->

Tên tham số `0.weight`, `0.bias`, `2.weight`, `2.bias` giống hệt cách `nn.Sequential` đặt tên ([mục 6.2.1](dl/ch06_s2_vi)), vì cùng một quy tắc: tên mô-đun con, dấu chấm, rồi tên tham số.

## 2. Khởi tạo, lưu và nạp

<small class="viz-ref">Mục 6.3, 6.6</small>

<!-- run -->
```python
def init_normal(m):
    if isinstance(m, Linear):
        with torch.no_grad():
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

net.apply(init_normal)
print("sau init_normal: độ lệch chuẩn trọng số tầng đầu =", round(net[0].weight.data.std().item(), 4),
      "| bias =", net[0].bias.data.abs().max().item())

sd = net.state_dict()
clone = Sequential(Linear(20, 256), ReLU(), Linear(256, 10))   # kiến trúc phải dựng lại bằng code
clone.load_state_dict(sd)
print("bản sao cho cùng đầu ra:", torch.equal(clone(X), net(X)))
```

<!-- output -->
```output
sau init_normal: độ lệch chuẩn trọng số tầng đầu = 0.01 | bias = 0.0
bản sao cho cùng đầu ra: True
```
<!-- /output -->

## 3. Tham số dùng chung: gradient được cộng lại

<small class="viz-ref">Mục 6.2.2</small>

Sách nói khi hai tầng dùng chung tham số, "gradient của tầng ẩn thứ hai và thứ ba được cộng lại". Kiểm chứng: so gradient của tầng dùng chung với tổng gradient của hai tầng **riêng biệt** mang cùng giá trị trọng số.

<!-- run -->
```python
torch.manual_seed(1)
shared = Linear(8, 8)
tied = Sequential(Linear(4, 8), ReLU(), shared, ReLU(), shared, ReLU(), Linear(8, 1))
print("số tham số khác nhau:", len(list(tied.parameters())),
      "| số tên tham số:", len(list(tied.named_parameters())))

X4 = torch.rand(5, 4)
tied(X4).sum().backward()
g_shared = shared.weight.data.grad.clone()

# Cùng mạng nhưng tầng 2 và 4 là hai bản sao độc lập
untied = Sequential(*[tied[i] for i in range(2)], Linear(8, 8), ReLU(), Linear(8, 8), ReLU(), tied[6])
untied[2].load_state_dict(shared.state_dict()); untied[4].load_state_dict(shared.state_dict())
tied.zero_grad()
untied(X4).sum().backward()
g_sum = untied[2].weight.data.grad + untied[4].weight.data.grad
print("gradient tầng dùng chung = tổng gradient hai bản sao:", torch.allclose(g_shared, g_sum))
```

<!-- output -->
```output
số tham số khác nhau: 6 | số tên tham số: 8
gradient tầng dùng chung = tổng gradient hai bản sao: True
```
<!-- /output -->

## 4. Lỗi `.data` trong `MyLinear` của mục 6.5.2

<small class="viz-ref">Mục 6.5.2 · đính chính</small>

`MyLinear` của sách viết `torch.matmul(X, self.weight.data) + self.bias.data`. Trong PyTorch, `tensor.data` trả về tensor **đã tách khỏi đồ thị**, nên autograd không thấy hai tham số này. Thử huấn luyện cả bản của sách và bản đã sửa trên một bài hồi quy tuyến tính nhỏ:

<!-- run -->
```python
from torch import nn
from torch.nn import functional as F

class MyLinear(nn.Module):                      # đúng như sách
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

class MyLinearFixed(MyLinear):                  # chỉ bỏ .data
    def forward(self, X):
        return F.relu(torch.matmul(X, self.weight) + self.bias)

torch.manual_seed(0)
Xr = torch.randn(256, 3)
yr = F.relu(Xr @ torch.tensor([[2.0], [-1.0], [0.5]]) + 1.0)
for cls in (MyLinear, MyLinearFixed):
    torch.manual_seed(0)
    layer = cls(3, 1)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    first = F.mse_loss(layer(Xr), yr).item()
    for _ in range(200):
        loss = F.mse_loss(layer(Xr), yr)
        opt.zero_grad()
        if loss.requires_grad:
            loss.backward()
        opt.step()
    print(f"{cls.__name__:14s} loss.requires_grad={str(loss.requires_grad):5s} "
          f"mất mát: {first:.3f} → {loss.item():.3f}, weight.grad là None: {layer.weight.grad is None}")
```

<!-- output -->
```output
MyLinear       loss.requires_grad=False mất mát: 4.215 → 4.215, weight.grad là None: True
MyLinearFixed  loss.requires_grad=True  mất mát: 4.215 → 0.000, weight.grad là None: False
```
<!-- /output -->

Bản của sách: mất mát không có gradient (`requires_grad=False`), trọng số không bao giờ nhận gradient và mất mát không đổi. Bản sửa học được. Khi tự viết tầng, hãy dùng thẳng tham số (`self.weight`), không dùng `.data`.

## Tóm tắt

- `nn.Module` nhận ra tham số và mô-đun con nhờ chặn phép gán thuộc tính (`__setattr__`); mọi tiện ích khác (`parameters`, `apply`, `state_dict`) chỉ là duyệt đệ quy cây mô-đun.
- `state_dict` chỉ chứa tensor tham số theo tên, nên phải dựng lại kiến trúc bằng code trước khi nạp.
- Một tham số dùng ở hai nơi nhận tổng gradient từ cả hai nơi.
- `tensor.data` cắt đứt đồ thị tính toán: dùng nó trong `forward` khiến tham số không học được.
