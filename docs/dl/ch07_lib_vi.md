# Chương 7 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang dựng lại LeNet bằng các lớp có sẵn của `torch.nn`, huấn luyện bằng một vòng lặp PyTorch thông thường (không dùng `d2l.Trainer`), rồi nhìn vào bên trong mạng đã học: các bộ lọc và ánh xạ đặc trưng.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `torch`, `torchvision`, `matplotlib`. Các khối huấn luyện chạy trên GPU nếu có.

| Trong sách | PyTorch |
| --- | --- |
| Tầng tích chập $k_\textrm{h} \times k_\textrm{w}$, $c_\textrm{i} \to c_\textrm{o}$ kênh (mục 7.2, 7.4) | `nn.Conv2d(c_i, c_o, kernel_size=(k_h, k_w))` |
| Đệm $(p_\textrm{h}, p_\textrm{w})$, sải bước $(s_\textrm{h}, s_\textrm{w})$ (mục 7.3) | `padding=(p_h, p_w), stride=(s_h, s_w)` |
| Gộp cực đại, gộp trung bình (mục 7.5) | `nn.MaxPool2d`, `nn.AvgPool2d` (sải bước mặc định bằng cửa sổ) |
| Trải phẳng trước tầng kết nối đầy đủ (mục 7.6) | `nn.Flatten()` |
| Tầng có số kênh vào tự suy ra | `nn.LazyConv2d`, `nn.LazyLinear` (sách dùng) |

## 1. Vì sao dùng thư viện

<small class="viz-ref">Mục 7.2 · hàm corr2d</small>

Hàm `corr2d` của sách dùng hai vòng lặp Python trên mọi vị trí đầu ra. So thời gian với `F.conv2d` trên một minibatch 128 ảnh $28 \times 28$ (cỡ minibatch của sách), kernel $5 \times 5$:

<!-- run -->
```python
import time
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt

def corr2d(X, K):
    """Hàm của mục 7.2."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

torch.manual_seed(0)
X, K = torch.randn(128, 28, 28), torch.randn(5, 5)

def bench(f, n):
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t0) / n

t_loop = bench(lambda: [corr2d(x, K) for x in X], 2)
t_lib = bench(lambda: F.conv2d(X[:, None], K[None, None]), 50)
print(f"corr2d, lặp qua 128 ảnh : {t_loop * 1e3:7.1f} ms")
print(f"F.conv2d, cả minibatch  : {t_lib * 1e3:7.1f} ms  (nhanh hơn khoảng {t_loop / t_lib:.0f} lần)")
print("Kết quả giống nhau:", torch.allclose(corr2d(X[0], K), F.conv2d(X[:1, None], K[None, None])[0, 0], atol=1e-5))
```

<!-- output -->
```output
corr2d, lặp qua 128 ảnh :   416.8 ms
F.conv2d, cả minibatch  :     0.4 ms  (nhanh hơn khoảng 976 lần)
Kết quả giống nhau: True
```
<!-- /output -->

Trên GPU, `F.conv2d` còn gọi thư viện cuDNN của NVIDIA, nhanh hơn nữa (sách nhắc tới ở phần mở đầu chương, trích dẫn Chetlur *et al.*, 2014).

## 2. Tầng tích chập và tầng gộp của `torch.nn`

<small class="viz-ref">Mục 7.3, 7.4</small>

Trọng số của `nn.Conv2d` có kích thước ($c_\textrm{o}$, $c_\textrm{i}$, $k_\textrm{h}$, $k_\textrm{w}$), đúng thứ tự của [mục 7.4.2](dl/ch07_s4_vi). Hàm `out_size` dưới đây là công thức kích thước đầu ra của [mục 7.3.2](dl/ch07_s3_vi), dùng để kiểm tra:

<!-- run -->
```python
def out_size(n, k, p, s):
    """Kích thước đầu ra theo một trục; p là tổng số hàng (cột) đệm."""
    return (n - k + p + s) // s

conv = nn.Conv2d(3, 8, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print("weight:", tuple(conv.weight.shape), " bias:", tuple(conv.bias.shape))
X = torch.randn(2, 3, 32, 32)
print("đầu ra thực tế :", tuple(conv(X).shape))
print("theo công thức :", (2, 8, out_size(32, 3, 0, 3), out_size(32, 5, 2, 4)))

pool = nn.MaxPool2d(3, stride=2, padding=1)
print("gộp cực đại 3x3, sải bước 2, đệm 1:", tuple(pool(X).shape),
      "| theo công thức:", out_size(32, 3, 2, 2))
print("số tham số của tầng gộp:", sum(p.numel() for p in pool.parameters()))
```

<!-- output -->
```output
weight: (8, 3, 3, 5)  bias: (8,)
đầu ra thực tế : (2, 8, 10, 8)
theo công thức : (2, 8, 10, 8)
gộp cực đại 3x3, sải bước 2, đệm 1: (2, 3, 16, 16) | theo công thức: 16
số tham số của tầng gộp: 0
```
<!-- /output -->

## 3. LeNet bằng `nn.Sequential`

<small class="viz-ref">Mục 7.6.1 · hình 7.6.1</small>

Viết rõ số kênh vào thay vì dùng các lớp `Lazy` của sách, để dễ đếm tham số. So với MLP một tầng ẩn 256 nút của [mục 5.2](https://d2l.ai/chapter_multilayer-perceptrons/mlp-implementation.html):

<!-- run -->
```python
def lenet(modern=False):
    act = nn.ReLU if modern else nn.Sigmoid
    pool = (lambda: nn.MaxPool2d(2)) if modern else (lambda: nn.AvgPool2d(2))
    return nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2), act(), pool(),
        nn.Conv2d(6, 16, 5), act(), pool(),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), act(),
        nn.Linear(120, 84), act(),
        nn.Linear(84, 10))

mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
count = lambda net: sum(p.numel() for p in net.parameters())
net = lenet()
print(f"LeNet: {count(net):,} tham số   MLP 784-256-10: {count(mlp):,} tham số")
for name, p in net.named_parameters():
    print(f"  {name:10s} {str(tuple(p.shape)):18s} {p.numel():6,d}")
```

<!-- output -->
```output
LeNet: 61,706 tham số   MLP 784-256-10: 203,530 tham số
  0.weight   (6, 1, 5, 5)          150
  0.bias     (6,)                    6
  3.weight   (16, 6, 5, 5)       2,400
  3.bias     (16,)                  16
  7.weight   (120, 400)         48,000
  7.bias     (120,)                120
  9.weight   (84, 120)          10,080
  9.bias     (84,)                  84
  11.weight  (10, 84)              840
  11.bias    (10,)                  10
```
<!-- /output -->

Phần lớn tham số của LeNet nằm ở ba tầng kết nối đầy đủ; hai tầng tích chập chỉ có khoảng 2 600 tham số dù xử lý toàn bộ ảnh. Đây là hệ quả trực tiếp của tính cục bộ và việc dùng chung trọng số ([mục 7.1](dl/ch07_s1_vi)).

## 4. Huấn luyện trên Fashion-MNIST

<small class="viz-ref">Mục 7.6.2</small>

Vòng lặp chuẩn của PyTorch: `DataLoader` chia minibatch, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`. Giống sách: khởi tạo Xavier, SGD, tốc độ học 0.1, minibatch 128, 10 epoch.

<!-- run -->
```python
import os
import torchvision
from torchvision import transforms

root = os.path.expanduser('~/.cache/d2l-data')
train = torchvision.datasets.FashionMNIST(root, train=True, transform=transforms.ToTensor(), download=True)
test = torchvision.datasets.FashionMNIST(root, train=False, transform=transforms.ToTensor(), download=True)
labels = ['áo phông', 'quần dài', 'áo len', 'váy', 'áo khoác', 'xăng-đan', 'sơ mi', 'giày thể thao', 'túi', 'bốt']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("thiết bị:", device)

def init_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)

def evaluate(net, loader):
    net.eval()
    correct = n = 0
    with torch.no_grad():
        for X, y in loader:
            correct += (net(X.to(device)).argmax(1) == y.to(device)).sum().item()
            n += len(y)
    return correct / n

def train_net(net, epochs=10, lr=0.1, batch_size=128):
    torch.manual_seed(0)
    net.apply(init_xavier).to(device)
    loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, 512, num_workers=2)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = []
    for epoch in range(epochs):
        net.train()
        total = n = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            loss = loss_fn(net(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y); n += len(y)
        hist.append((total / n, evaluate(net, test_loader)))
        print(f"  epoch {epoch + 1}: mất mát {hist[-1][0]:.3f}, độ chính xác kiểm tra {hist[-1][1]:.3f}")
    return hist
```

<!-- output -->
```output
thiết bị: cpu
```
<!-- /output -->

<!-- run gpu -->
```python
t0 = time.time()
print("LeNet của sách (sigmoid, gộp trung bình):")
net_book = lenet(); hist_book = train_net(net_book)
print("LeNet hiện đại (ReLU, gộp cực đại):")
net_modern = lenet(modern=True); hist_modern = train_net(net_modern)
print("MLP 784-256-10:")
hist_mlp = train_net(mlp)
print(f"Tổng thời gian: {time.time() - t0:.0f} s")

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
for h, name in [(hist_book, 'LeNet (sách)'), (hist_modern, 'LeNet hiện đại'), (hist_mlp, 'MLP')]:
    ax[0].plot(range(1, 11), [l for l, _ in h], 'o-', label=name)
    ax[1].plot(range(1, 11), [a for _, a in h], 'o-', label=name)
ax[0].set(xlabel='epoch', ylabel='mất mát huấn luyện'); ax[1].set(xlabel='epoch', ylabel='độ chính xác kiểm tra')
ax[1].legend(); plt.tight_layout()
torch.save(net_modern.state_dict(), os.path.join(root, 'lenet_modern.pt'))
```

<!-- output -->
<div class="gpu-pending">Chưa có kết quả: khối này huấn luyện mô hình nên sẽ được chạy trên máy có GPU. Bạn có thể tự chạy code trên Colab hoặc máy của mình.</div>
<!-- /output -->

## 5. Nhìn vào bên trong mạng đã học

<small class="viz-ref">Mục 7.2.7 · hình 7.2.2</small>

[Mục 7.2.7](dl/ch07_s2_vi) nói các tầng tích chập đầu tiên học được những bộ phát hiện cạnh, giống vùng tiếp nhận trong vỏ não thị giác. Vẽ 6 kernel $5 \times 5$ của tầng đầu tiên trong LeNet hiện đại vừa huấn luyện, và 6 ánh xạ đặc trưng mà chúng tạo ra từ một ảnh kiểm tra:

<!-- run gpu -->
```python
net_modern.eval().cpu()
W = net_modern[0].weight.detach()[:, 0]               # (6, 5, 5)
x, y = test[0]
with torch.no_grad():
    maps = net_modern[1](net_modern[0](x[None]))[0]    # sau ReLU: (6, 28, 28)

fig, ax = plt.subplots(3, 6, figsize=(9, 4.8))
for k in range(6):
    v = W[k].abs().max()
    ax[0, k].imshow(W[k], cmap='RdBu_r', vmin=-v, vmax=v); ax[0, k].set_title(f'kernel {k}', fontsize=8)
    ax[1, k].imshow(maps[k], cmap='gray'); ax[1, k].set_title(f'ánh xạ {k}', fontsize=8)
    ax[2, k].axis('off')
ax[2, 0].imshow(x[0], cmap='gray'); ax[2, 0].set_title(f'ảnh vào: {labels[y]}', fontsize=8); ax[2, 0].axis('on')
for a in ax.flat:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
```

<!-- output -->
<div class="gpu-pending">Chưa có kết quả: khối này huấn luyện mô hình nên sẽ được chạy trên máy có GPU. Bạn có thể tự chạy code trên Colab hoặc máy của mình.</div>
<!-- /output -->

Ma trận nhầm lẫn cho biết LeNet hiện đại còn nhầm những loại quần áo nào:

<!-- run gpu -->
```python
conf = torch.zeros(10, 10, dtype=torch.long)
net_modern.to(device).eval()
with torch.no_grad():
    for X, y in torch.utils.data.DataLoader(test, 512):
        pred = net_modern(X.to(device)).argmax(1).cpu()
        conf += torch.bincount(y * 10 + pred, minlength=100).reshape(10, 10)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(conf, cmap='Blues')
ax.set_xticks(range(10), labels, rotation=60, fontsize=8); ax.set_yticks(range(10), labels, fontsize=8)
ax.set(xlabel='dự đoán', ylabel='nhãn thật')
for i in range(10):
    for j in range(10):
        ax.text(j, i, int(conf[i, j]), ha='center', va='center', fontsize=6,
                color='white' if conf[i, j] > conf.max() / 2 else 'black')
plt.colorbar(im); plt.tight_layout()
per_class = conf.diag() / conf.sum(1)
for k in per_class.argsort()[:3]:
    print(f"lớp khó nhất: {labels[k]:12s} {per_class[k]:.3f}")
```

<!-- output -->
<div class="gpu-pending">Chưa có kết quả: khối này huấn luyện mô hình nên sẽ được chạy trên máy có GPU. Bạn có thể tự chạy code trên Colab hoặc máy của mình.</div>
<!-- /output -->

## Tóm tắt

- `nn.Conv2d`, `nn.MaxPool2d`, `nn.AvgPool2d` nhận đúng các tham số của chương: kích thước kernel, đệm, sải bước, số kênh. Công thức kích thước đầu ra của mục 7.3 dùng để kiểm tra mạng trước khi chạy.
- Trên một minibatch, `F.conv2d` nhanh hơn vòng lặp Python của `corr2d` nhiều bậc độ lớn, ngay cả trên CPU.
- Tầng tích chập có rất ít tham số so với tầng kết nối đầy đủ; phần lớn tham số của LeNet nằm ở phần kết nối đầy đủ.
