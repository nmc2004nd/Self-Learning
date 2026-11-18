# Chương 7 · Code từ đầu

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang tự viết lan truyền xuôi của tầng tích chập và tầng gộp cho cả batch nhiều kênh, có đệm và sải bước, rồi ghép thành LeNet. Gradient do autograd của PyTorch tính. Muốn xem lan truyền ngược viết bằng tay, xem trang [Bổ sung: lan truyền ngược qua tích chập](dl/ch07_bwd_vi).

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Chỉ cần `torch`, `torchvision` (để tải Fashion-MNIST) và `matplotlib`. Trang không dùng thư viện `d2l` của sách.

Hàm `corr2d` của [mục 7.2](dl/ch07_s2_vi) dùng hai vòng lặp trên **mỗi vị trí đầu ra**, nên rất chậm với ảnh thật. Trang này dùng một cách viết khác, bám sát công thức [(7.1.1)](dl/ch07_s1_vi):

$$[\mathbf{H}]_{i, j} = u + \sum_{a} \sum_{b} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$

Thay vì lặp trên $(i, j)$, ta lặp trên **độ lệch** $(a, b)$ của kernel. Với mỗi $(a, b)$, các giá trị $[\mathbf{X}]_{i+a, j+b}$ ứng với mọi $(i, j)$ tạo thành một lát cắt của $\mathbf{X}$ (dịch đi $a$ hàng, $b$ cột). Nhân lát cắt đó với $[\mathbf{V}]_{a, b}$ rồi cộng dồn. Số vòng lặp chỉ còn $k_\textrm{h} \times k_\textrm{w}$ (25 với kernel $5 \times 5$), không phụ thuộc kích thước ảnh hay kích thước batch.

## 1. Tầng tích chập cho batch nhiều kênh

<small class="viz-ref">Mục 7.2–7.4 · công thức (7.1.1), (7.1.3)</small>

Đầu vào `X` có kích thước (batch, $c_\textrm{i}$, $h$, $w$), kernel `W` có kích thước ($c_\textrm{o}$, $c_\textrm{i}$, $k_\textrm{h}$, $k_\textrm{w}$) như [mục 7.4.2](dl/ch07_s4_vi). Với mỗi độ lệch $(a, b)$, phép `einsum` lấy tổng theo kênh vào $c$ cho từng kênh ra $o$, đúng như công thức [(7.1.3)](dl/ch07_s1_vi):

<!-- run -->
```python
import time
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt

torch.manual_seed(0)

def pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)

def pad2d(X, padding, value=0.0):
    """Đệm hai trục cuối của X (kích thước (n, c, h, w))."""
    ph, pw = pair(padding)
    n, c, h, w = X.shape
    Xp = torch.full((n, c, h + 2 * ph, w + 2 * pw), value, dtype=X.dtype)
    Xp[:, :, ph:ph + h, pw:pw + w] = X
    return Xp

def conv2d_scratch(X, W, b=None, padding=0, stride=1):
    """Tích chập (thực ra là tương quan chéo) cho batch nhiều kênh."""
    sh, sw = pair(stride)
    Xp = pad2d(X, padding)
    c_o, c_i, kh, kw = W.shape
    oh = (Xp.shape[2] - kh) // sh + 1      # công thức kích thước đầu ra ở mục 7.3.2
    ow = (Xp.shape[3] - kw) // sw + 1
    Y = 0
    for a in range(kh):
        for b_ in range(kw):
            # [X]_{i+a, j+b} cho mọi vị trí đầu ra (i, j), có tính sải bước
            patch = Xp[:, :, a:a + sh * (oh - 1) + 1:sh, b_:b_ + sw * (ow - 1) + 1:sw]
            Y = Y + torch.einsum('ncij,oc->noij', patch, W[:, :, a, b_])
    if b is not None:
        Y = Y + b.view(1, -1, 1, 1)
    return Y

# Kiểm tra với ví dụ hình 7.2.1: kết quả phải là [[19, 25], [37, 43]]
X0 = torch.arange(9.0).reshape(1, 1, 3, 3)
K0 = torch.tensor([[0.0, 1.0], [2.0, 3.0]]).reshape(1, 1, 2, 2)
print(conv2d_scratch(X0, K0)[0, 0])
```

<!-- output -->
```output
tensor([[19., 25.],
        [37., 43.]])
```
<!-- /output -->

So với `F.conv2d` của PyTorch trên dữ liệu ngẫu nhiên, với nhiều cách đệm và sải bước (gồm cả ví dụ cuối của [mục 7.3](dl/ch07_s3_vi)):

<!-- run -->
```python
X = torch.randn(4, 3, 8, 8)
for k, p, s in [(3, 1, 1), ((5, 3), (2, 1), 1), (3, 1, 2), ((3, 5), (0, 1), (3, 4))]:
    W = torch.randn(2, 3, *pair(k))
    b = torch.randn(2)
    mine = conv2d_scratch(X, W, b, padding=p, stride=s)
    ref = F.conv2d(X, W, b, padding=p, stride=s)
    print(f"kernel {k}, đệm {p}, sải bước {s}: đầu ra {tuple(mine.shape)}, "
          f"sai khác lớn nhất {(mine - ref).abs().max():.1e}")
```

<!-- output -->
```output
kernel 3, đệm 1, sải bước 1: đầu ra (4, 2, 8, 8), sai khác lớn nhất 2.4e-06
kernel (5, 3), đệm (2, 1), sải bước 1: đầu ra (4, 2, 8, 8), sai khác lớn nhất 4.8e-06
kernel 3, đệm 1, sải bước 2: đầu ra (4, 2, 4, 4), sai khác lớn nhất 2.1e-06
kernel (3, 5), đệm (0, 1), sải bước (3, 4): đầu ra (4, 2, 2, 2), sai khác lớn nhất 6.7e-06
```
<!-- /output -->

## 2. Tầng gộp

<small class="viz-ref">Mục 7.5</small>

Cùng ý tưởng: với mỗi độ lệch trong cửa sổ gộp, lấy một lát cắt, xếp các lát cắt thành một trục mới rồi lấy giá trị lớn nhất hoặc trung bình theo trục đó. Gộp xử lý từng kênh riêng, nên không có tổng theo kênh. Hai chi tiết cần khớp với PyTorch: sải bước mặc định bằng kích thước cửa sổ ([mục 7.5.2](dl/ch07_s5_vi)), và khi gộp cực đại thì phần đệm mang giá trị $-\infty$ để không bao giờ được chọn.

<!-- run -->
```python
def pool2d_scratch(X, pool_size, stride=None, padding=0, mode='max'):
    kh, kw = pair(pool_size)
    sh, sw = pair(stride if stride is not None else pool_size)
    Xp = pad2d(X, padding, value=float('-inf') if mode == 'max' else 0.0)
    oh = (Xp.shape[2] - kh) // sh + 1
    ow = (Xp.shape[3] - kw) // sw + 1
    patches = torch.stack([Xp[:, :, a:a + sh * (oh - 1) + 1:sh, b:b + sw * (ow - 1) + 1:sw]
                           for a in range(kh) for b in range(kw)])
    return patches.amax(0) if mode == 'max' else patches.mean(0)

X = torch.arange(16.0).reshape(1, 1, 4, 4)
X2 = torch.cat((X, X + 1), 1)                  # hai kênh, như mục 7.5.3
print(pool2d_scratch(X2, 3, padding=1, stride=2))
for k, p, s in [(2, 0, None), (3, 1, 2), ((2, 3), (0, 1), (2, 3))]:
    Xr = torch.randn(4, 3, 9, 9)
    d_max = (pool2d_scratch(Xr, k, s, p, 'max') - F.max_pool2d(Xr, k, s, p)).abs().max()
    d_avg = (pool2d_scratch(Xr, k, s, p, 'avg') - F.avg_pool2d(Xr, k, s, p)).abs().max()
    print(f"cửa sổ {k}, đệm {p}, sải bước {s}: sai khác max {d_max:.1e}, avg {d_avg:.1e}")
```

<!-- output -->
```output
tensor([[[[ 5.,  7.],
          [13., 15.]],

         [[ 6.,  8.],
          [14., 16.]]]])
cửa sổ 2, đệm 0, sải bước None: sai khác max 0.0e+00, avg 0.0e+00
cửa sổ 3, đệm 1, sải bước 2: sai khác max 0.0e+00, avg 3.0e-08
cửa sổ (2, 3), đệm (0, 1), sải bước (2, 3): sai khác max 0.0e+00, avg 1.2e-07
```
<!-- /output -->

## 3. LeNet ghép từ các tầng tự viết

<small class="viz-ref">Mục 7.6.1 · hình 7.6.1, 7.6.2</small>

Bọc hai hàm trên thành `nn.Module` để PyTorch quản lý tham số. Trọng số khởi tạo Xavier (đều) như hàm `init_cnn` của sách:

<!-- run -->
```python
class Conv2dScratch(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, padding=0, stride=1):
        super().__init__()
        kh, kw = pair(kernel_size)
        self.padding, self.stride = padding, stride
        fan_in, fan_out = c_in * kh * kw, c_out * kh * kw
        bound = (6 / (fan_in + fan_out)) ** 0.5           # Xavier đều
        self.weight = nn.Parameter(torch.empty(c_out, c_in, kh, kw).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, X):
        return conv2d_scratch(X, self.weight, self.bias, self.padding, self.stride)

class Pool2dScratch(nn.Module):
    def __init__(self, pool_size, stride=None, mode='avg'):
        super().__init__()
        self.pool_size, self.stride, self.mode = pool_size, stride, mode

    def forward(self, X):
        return pool2d_scratch(X, self.pool_size, self.stride, mode=self.mode)

def lenet_scratch():
    return nn.Sequential(
        Conv2dScratch(1, 6, 5, padding=2), nn.Sigmoid(), Pool2dScratch(2, 2),
        Conv2dScratch(6, 16, 5), nn.Sigmoid(), Pool2dScratch(2, 2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))

net = lenet_scratch()
X = torch.randn(1, 1, 28, 28)
for layer in net:
    X = layer(X)
    print(f"{layer.__class__.__name__:14s} {tuple(X.shape)}")
print("Số tham số:", sum(p.numel() for p in net.parameters()))
```

<!-- output -->
```output
Conv2dScratch  (1, 6, 28, 28)
Sigmoid        (1, 6, 28, 28)
Pool2dScratch  (1, 6, 14, 14)
Conv2dScratch  (1, 16, 10, 10)
Sigmoid        (1, 16, 10, 10)
Pool2dScratch  (1, 16, 5, 5)
Flatten        (1, 400)
Linear         (1, 120)
Sigmoid        (1, 120)
Linear         (1, 84)
Sigmoid        (1, 84)
Linear         (1, 10)
Số tham số: 61706
```
<!-- /output -->

Kích thước từng tầng khớp với bảng in ra ở [mục 7.6.1](dl/ch07_s6_vi). Để chắc chắn tầng tự viết đúng cả khi lan truyền ngược, chép cùng trọng số sang một LeNet dựng từ `nn.Conv2d`, `nn.AvgPool2d`, rồi so đầu ra và gradient của mọi tham số:

<!-- run -->
```python
ref = nn.Sequential(
    nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(), nn.AvgPool2d(2, 2),
    nn.Conv2d(6, 16, 5), nn.Sigmoid(), nn.AvgPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(400, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
ref.load_state_dict(net.state_dict())     # tên tham số trùng nhau: 0.weight, 0.bias, 3.weight, ...

X = torch.randn(32, 1, 28, 28)
y = torch.randint(0, 10, (32,))
out_mine, out_ref = net(X), ref(X)
F.cross_entropy(out_mine, y).backward()
F.cross_entropy(out_ref, y).backward()
print(f"đầu ra: sai khác lớn nhất {(out_mine - out_ref).abs().max():.1e}")
for (name, p), q in zip(net.named_parameters(), ref.parameters()):
    print(f"gradient {name:9s}: sai khác lớn nhất {(p.grad - q.grad).abs().max():.1e}")
```

<!-- output -->
```output
đầu ra: sai khác lớn nhất 7.5e-08
gradient 0.weight : sai khác lớn nhất 2.2e-11
gradient 0.bias   : sai khác lớn nhất 4.8e-11
gradient 3.weight : sai khác lớn nhất 1.2e-10
gradient 3.bias   : sai khác lớn nhất 1.3e-10
gradient 7.weight : sai khác lớn nhất 1.5e-10
gradient 7.bias   : sai khác lớn nhất 1.2e-10
gradient 9.weight : sai khác lớn nhất 9.3e-10
gradient 9.bias   : sai khác lớn nhất 5.2e-10
gradient 11.weight: sai khác lớn nhất 7.5e-09
gradient 11.bias  : sai khác lớn nhất 3.7e-09
```
<!-- /output -->

## 4. Vòng lặp huấn luyện tự viết

<small class="viz-ref">Mục 7.6.2</small>

Hàm mất mát entropy chéo và bước cập nhật hạ gradient ngẫu nhiên theo minibatch đều viết tay, giống [chương 15](ml/ch15_vi) và [chương 16](ml/ch16_vi) của khoá Học máy. `d2l.Trainer` của sách cũng chỉ dùng SGD thường (`torch.optim.SGD`), nên vòng lặp dưới đây làm đúng những việc mà `trainer.fit` làm ở [mục 7.6.2](dl/ch07_s6_vi).

<!-- run -->
```python
import os
import torchvision
from torchvision import transforms

root = os.path.expanduser('~/.cache/d2l-data')
to_tensor = transforms.ToTensor()
train = torchvision.datasets.FashionMNIST(root, train=True, transform=to_tensor, download=True)
test = torchvision.datasets.FashionMNIST(root, train=False, transform=to_tensor, download=True)
print(len(train), "ảnh huấn luyện,", len(test), "ảnh kiểm tra, mỗi ảnh", tuple(train[0][0].shape))

def cross_entropy(logits, y):
    logits = logits - logits.max(1, keepdim=True).values          # ổn định số
    log_p = logits - logits.exp().sum(1, keepdim=True).log()      # log softmax
    return -log_p[torch.arange(len(y)), y].mean()

def sgd_step(params, lr):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
            p.grad = None

def accuracy(net, loader, device):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            correct += (net(X.to(device)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    net.train()
    return correct / total
```

<!-- output -->
```output
60000 ảnh huấn luyện, 10000 ảnh kiểm tra, mỗi ảnh (1, 28, 28)
```
<!-- /output -->

Trước khi huấn luyện thật, kiểm tra nhanh trên CPU: một mạng đúng phải *học thuộc* được một tập nhỏ. Ở đây là 256 ảnh, 300 bước hạ gradient trên cả 256 ảnh, tốc độ học 0.1 như sách. Ta thử cả LeNet của sách lẫn một bản "hiện đại hoá" (ReLU và gộp cực đại, bài tập 1 của [mục 7.6](dl/ch07_s6_vi)):

<!-- run -->
```python
def lenet_modern_scratch():
    return nn.Sequential(
        Conv2dScratch(1, 6, 5, padding=2), nn.ReLU(), Pool2dScratch(2, 2, mode='max'),
        Conv2dScratch(6, 16, 5), nn.ReLU(), Pool2dScratch(2, 2, mode='max'),
        nn.Flatten(),
        nn.Linear(400, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

small = torch.utils.data.Subset(train, range(256))
Xs = torch.stack([x for x, _ in small]); ys = torch.tensor([y for _, y in small])
for name, make in [("LeNet (sigmoid, gộp trung bình)", lenet_scratch),
                   ("LeNet hiện đại (ReLU, gộp cực đại)", lenet_modern_scratch)]:
    print(name)
    torch.manual_seed(0)
    net = make()
    for step in range(1, 301):
        loss = cross_entropy(net(Xs), ys)
        loss.backward()
        sgd_step(net.parameters(), lr=0.1)
        if step % 100 == 0:
            acc = (net(Xs).argmax(1) == ys).float().mean()
            print(f"  bước {step}: mất mát {loss.item():.3f}, độ chính xác trên 256 ảnh {acc:.2f}")
```

<!-- output -->
```output
LeNet (sigmoid, gộp trung bình)
  bước 100: mất mát 2.296, độ chính xác trên 256 ảnh 0.12
  bước 200: mất mát 2.296, độ chính xác trên 256 ảnh 0.12
  bước 300: mất mát 2.296, độ chính xác trên 256 ảnh 0.12
LeNet hiện đại (ReLU, gộp cực đại)
  bước 100: mất mát 2.017, độ chính xác trên 256 ảnh 0.37
  bước 200: mất mát 0.689, độ chính xác trên 256 ảnh 0.73
  bước 300: mất mát 0.415, độ chính xác trên 256 ảnh 0.83
```
<!-- /output -->

Bản ReLU học thuộc dần, chứng tỏ các tầng tự viết và vòng lặp huấn luyện đều đúng. Bản sigmoid thì đứng yên ở mất mát $\approx \log 10 \approx 2.30$, tức là vẫn đoán đều cả 10 lớp. Lý do nằm ở độ lớn gradient lúc khởi tạo của từng tầng:

<!-- run -->
```python
for name, make in [("sigmoid", lenet_scratch), ("ReLU", lenet_modern_scratch)]:
    torch.manual_seed(0)
    net = make()
    cross_entropy(net(Xs), ys).backward()
    norms = [f"{p.grad.norm():.1e}" for n, p in net.named_parameters() if n.endswith('weight')]
    print(f"{name:8s} chuẩn gradient của trọng số, từ tầng đầu đến tầng cuối:", ", ".join(norms))
```

<!-- output -->
```output
sigmoid  chuẩn gradient của trọng số, từ tầng đầu đến tầng cuối: 1.7e-04, 6.5e-04, 8.2e-03, 3.6e-02, 2.1e-01
ReLU     chuẩn gradient của trọng số, từ tầng đầu đến tầng cuối: 7.4e-03, 2.2e-02, 4.5e-02, 2.5e-02, 2.0e-02
```
<!-- /output -->

Với sigmoid, gradient của tầng tích chập đầu tiên nhỏ hơn gradient của tầng cuối khoảng một nghìn lần: mỗi lần đi qua một hàm sigmoid, gradient bị nhân với đạo hàm $\sigma'(x) \le 0.25$. Với ReLU, đạo hàm bằng 1 ở phần dương nên gradient ở các tầng đều cùng cỡ. Hiện tượng *tiêu biến gradient* (vanishing gradient) này được phân tích ở [mục 5.4](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) của sách (ổn định số và khởi tạo) và là một lý do ReLU thay thế sigmoid trong các mạng hiện đại ([chương 8](https://d2l.ai/chapter_convolutional-modern/index.html)). Biểu đồ huấn luyện in trong sách cho thấy LeNet sigmoid vẫn thoát được vùng bằng phẳng này nếu chạy đủ lâu. Khối huấn luyện đầy đủ dưới đây chạy 4 690 bước (10 epoch × 469 minibatch) để kiểm chứng.

Huấn luyện đầy đủ 10 epoch trên 60 000 ảnh, theo minibatch 128 ảnh như sách, cho cả hai mạng. Khối này tự dùng GPU nếu có:

<!-- run gpu -->
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test, batch_size=512, num_workers=2)

def fit(net, lr, epochs=10):
    torch.manual_seed(0)
    net = net.to(device)
    hist = []
    t0 = time.time()
    for epoch in range(epochs):
        total = n = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            loss = cross_entropy(net(X), y)
            loss.backward()
            sgd_step(net.parameters(), lr)
            total += loss.item() * len(y); n += len(y)
        hist.append((total / n, accuracy(net, test_loader, device)))
        print(f"  epoch {epoch + 1}: mất mát {hist[-1][0]:.3f}, độ chính xác kiểm tra {hist[-1][1]:.3f}")
    print(f"  {time.time() - t0:.0f} s trên {device}")
    return hist

print("LeNet (sigmoid, gộp trung bình):")
h_sig = fit(lenet_scratch(), lr=0.1)
print("LeNet hiện đại (ReLU, gộp cực đại):")
h_relu = fit(lenet_modern_scratch(), lr=0.1)

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
for h, name in [(h_sig, 'sigmoid + gộp TB'), (h_relu, 'ReLU + gộp cực đại')]:
    ax[0].plot(range(1, 11), [l for l, _ in h], 'o-', label=name)
    ax[1].plot(range(1, 11), [a for _, a in h], 'o-', label=name)
ax[0].set(xlabel='epoch', ylabel='mất mát huấn luyện'); ax[1].set(xlabel='epoch', ylabel='độ chính xác kiểm tra')
ax[1].legend()
plt.tight_layout()
```

<!-- output -->
<div class="gpu-pending">Chưa có kết quả: khối này huấn luyện mô hình nên sẽ được chạy trên máy có GPU. Bạn có thể tự chạy code trên Colab hoặc máy của mình.</div>
<!-- /output -->

## Tóm tắt

- Tích chập và gộp đều có thể viết bằng một vòng lặp trên các độ lệch trong cửa sổ, mỗi lần xử lý cả batch, mọi kênh và mọi vị trí. Đây chính là công thức (7.1.1) đọc theo chiều ngược lại.
- Tầng tự viết cho cùng đầu ra và cùng gradient với `nn.Conv2d`, `nn.AvgPool2d` (sai khác chỉ ở mức sai số làm tròn của số thực dấu phẩy động).
- Học thuộc một tập nhỏ là phép thử nhanh, rẻ để bắt lỗi trước khi huấn luyện lâu.
