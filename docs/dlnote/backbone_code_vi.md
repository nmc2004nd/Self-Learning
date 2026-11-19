# Backbone kinh điển · Code

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm cho phần 3 của khoá: [tổng kết backbone](dlnote/bb_summary_vi), [ResNet](dlnote/bb_resnet_vi), [ResNet v2](dlnote/bb_resnetv2_vi), [DenseNet](dlnote/bb_densenet_vi), [MobileNet v1](dlnote/bb_mobilenet_vi), [ShuffleNet v2](dlnote/bb_shufflenet_vi), [RepVGG](dlnote/bb_repvgg_vi), [CSPNet](dlnote/bb_cspnet_vi) và [VoVNet](dlnote/bb_vovnet_vi). Mỗi mục kiểm tra bằng số một công thức hay một nhận xét của các bài đó, và chạy lại code của bản gốc (đã sửa các lỗi).

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `torch`, `torchvision`, `matplotlib`. Mọi khối chạy trên CPU, cả trang mất khoảng ba phút. Số đo thời gian lấy trên CPU laptop Intel Core i5-1340P, 4 luồng; xung nhịp turbo thay đổi theo nhiệt độ nên cùng một phép đo có thể chênh tới 30% giữa các lần chạy, chỉ nên đọc tỉ lệ và xu hướng. Mọi phép đo lấy lượt nhanh nhất trong nhiều lượt, các phép so sánh trực tiếp được đo xen kẽ, và trang được chạy bằng `taskset -c 0,2,4,6 python3 …` để chỉ dùng 4 lõi hiệu năng cao (CPU này có thêm 8 lõi tiết kiệm điện chậm hơn; để hệ điều hành tự chuyển luồng qua lại giữa hai loại lõi làm kết quả các thí nghiệm G1–G4 đảo ngược giữa các lần chạy).

<!-- run -->
```python
import time, warnings, contextlib, io, math, copy
warnings.filterwarnings('ignore')
from collections import OrderedDict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.flop_counter import FlopCounterMode
torch.manual_seed(0); np.set_printoptions(precision=4, suppress=True)
torch.set_num_threads(4)
torch.set_grad_enabled(False)

def bench(fn, reps=10, rounds=7):
    fn(); fn(); ts = []
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(reps): fn()
        ts.append((time.perf_counter() - t) / reps * 1e3)
    return float(np.min(ts))

def bench_many(fns, reps=10, rounds=15):
    """Đo xen kẽ nhiều hàm; trả về ms nhanh nhất của từng hàm."""
    for f in fns: f()
    acc = [[] for _ in fns]
    for _ in range(rounds):
        for f, a in zip(fns, acc):
            t = time.perf_counter()
            for _ in range(reps): f()
            a.append((time.perf_counter() - t) / reps * 1e3)
    return [float(np.min(a)) for a in acc]

def macs(model, x):
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2

def nparams(m):
    return sum(p.numel() for p in m.parameters())

print("torch", torch.__version__, "| torchvision", torchvision.__version__)
```

<!-- output -->
```output
torch 2.14.0+cpu | torchvision 0.29.0+cpu
```
<!-- /output -->

## 1. ResNet

<small class="viz-ref">Bài ResNet: mục 1.2, 2</small>

Dựng ResNet-18 từ khối BasicBlock, so với `torchvision`, và in kích thước feature map qua từng giai đoạn với ảnh vào $1024\times 2048$:

<!-- run -->
```python
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:     # shortcut tích chập 1x1 (phương án B)
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + (x if self.down is None else self.down(x)))

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        cfg, cin, layers = [(64, 1), (128, 2), (256, 2), (512, 2)], 64, []
        for cout, s in cfg:
            layers.append(nn.Sequential(BasicBlock(cin, cout, s), BasicBlock(cout, cout))); cin = cout
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x, trace=False):
        x = self.stem(x)
        if trace: print("  sau conv1 + maxpool:", tuple(x.shape[-2:]))
        for i, l in enumerate(self.layers):
            x = l(x)
            if trace: print(f"  sau conv{i+2}_x:", tuple(x.shape[-2:]), x.shape[1], "kênh")
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))

mine, ref = ResNet18().eval(), torchvision.models.resnet18().eval()
print("tham số: tự dựng", f"{nparams(mine):,}", "| torchvision", f"{nparams(ref):,}")
n_conv = sum(isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3) for m in mine.modules())
print("số tích chập 3x3 trong các khối:", n_conv, "+ conv1 7x7 + fc = 18 tầng có trọng số (không tính 3 shortcut 1x1)")
print("ảnh vào (1024, 2048):")
_ = mine(torch.zeros(1, 3, 1024, 2048), trace=True)
```

<!-- output -->
```output
tham số: tự dựng 11,689,512 | torchvision 11,689,512
số tích chập 3x3 trong các khối: 16 + conv1 7x7 + fc = 18 tầng có trọng số (không tính 3 shortcut 1x1)
ảnh vào (1024, 2048):
  sau conv1 + maxpool: (256, 512)
  sau conv2_x: (256, 512) 64 kênh
  sau conv3_x: (128, 256) 128 kênh
  sau conv4_x: (64, 128) 256 kênh
  sau conv5_x: (32, 64) 512 kênh
```
<!-- /output -->

## 2. ResNet v2: gradient qua kết nối tắt

<small class="viz-ref">Bài ResNet v2: công thức (5), (8)</small>

Chồng 100 đơn vị $x_{l+1} = \lambda x_l + F(x_l)$ với $F$ là một tầng tuyến tính nhỏ (trọng số $\sim 10^{-3}$), đo chuẩn của gradient $\partial \varepsilon/\partial x_0$ so với $\partial \varepsilon/\partial x_L$:

<!-- run -->
```python
torch.set_grad_enabled(True)
torch.manual_seed(0)
L, d = 100, 64
Ws = [torch.randn(d, d) * 1e-3 for _ in range(L)]
for lam in (0.9, 1.0, 1.1):
    x0 = torch.randn(1, d, requires_grad=True); x = x0
    for W in Ws:
        x = lam * x + torch.tanh(x @ W)
    g = torch.randn_like(x)                      # ∂ε/∂x_L
    (x * g).sum().backward()
    print(f"λ = {lam}:  ||∂ε/∂x_0|| / ||∂ε/∂x_L|| = {x0.grad.norm() / g.norm():.3e}   (λ^100 = {lam ** L:.3e})")
torch.set_grad_enabled(False)
```

<!-- output -->
```output
λ = 0.9:  ||∂ε/∂x_0|| / ||∂ε/∂x_L|| = 2.679e-05   (λ^100 = 2.656e-05)
λ = 1.0:  ||∂ε/∂x_0|| / ||∂ε/∂x_L|| = 9.977e-01   (λ^100 = 1.000e+00)
λ = 1.1:  ||∂ε/∂x_0|| / ||∂ε/∂x_L|| = 1.384e+04   (λ^100 = 1.378e+04)
```
<!-- /output -->

Với $\lambda = 1$ (kết nối tắt đồng nhất), gradient về tới tầng đầu gần như nguyên độ lớn; với $\lambda = 0{,}9$ hay $1{,}1$, nó tắt hoặc nổ đúng theo $\lambda^{100}$, như công thức (8) của bài.

## 3. DenseNet

<small class="viz-ref">Bài DenseNet: mục 2, 3, 4</small>

Code `densenet.py` của bản gốc (rút gọn, bỏ phần checkpoint), cấu hình mặc định DenseNet-BC cho CIFAR với $L=100$, $k=12$:

<!-- run -->
```python
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features); self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate); self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
        self.drop_rate = drop_rate
    def forward(self, *prev):
        out = self.conv1(self.relu1(self.norm1(torch.cat(prev, 1))))
        return self.conv2(self.relu2(self.norm2(out)))

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0):
        super().__init__()
        self.layers = nn.ModuleList([_DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
                                     for i in range(num_layers)])
    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            feats.append(layer(*feats))
        return torch.cat(feats, 1)

class DenseNetBC(nn.Module):
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5, num_init_features=24, bn_size=4, num_classes=10):
        super().__init__()
        mods = [nn.Conv2d(3, num_init_features, 3, padding=1, bias=False)]
        nf = num_init_features
        for i, n in enumerate(block_config):
            mods.append(_DenseBlock(n, nf, bn_size, growth_rate)); nf += n * growth_rate
            if i != len(block_config) - 1:
                out = int(nf * compression)
                mods += [nn.BatchNorm2d(nf), nn.ReLU(), nn.Conv2d(nf, out, 1, bias=False), nn.AvgPool2d(2)]; nf = out
        mods.append(nn.BatchNorm2d(nf))
        self.features = nn.Sequential(*mods); self.classifier = nn.Linear(nf, num_classes)
    def forward(self, x):
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(F.relu(self.features(x)), 1), 1))

net = DenseNetBC().eval()
print(f"DenseNet-BC (L=100, k=12): {nparams(net)/1e6:.3f} triệu tham số; đầu ra cho ảnh 32x32:", tuple(net(torch.zeros(2, 3, 32, 32)).shape))
blk = net.features[1]
print("số kênh vào các tầng của khối 1:", [l.norm1.num_features for l in blk.layers][:6], "... = k0 + k(l-1) với k0 = 24, k = 12")
```

<!-- output -->
```output
DenseNet-BC (L=100, k=12): 0.769 triệu tham số; đầu ra cho ảnh 32x32: (2, 10)
số kênh vào các tầng của khối 1: [24, 36, 48, 60, 72, 84] ... = k0 + k(l-1) với k0 = 24, k = 12
```
<!-- /output -->

DenseNet-121 so với ResNet-50 trên ImageNet ($224\times224$, batch 1):

<!-- run -->
```python
x = torch.randn(1, 3, 224, 224)
d121, r50 = torchvision.models.densenet121().eval(), torchvision.models.resnet50().eval()
t_d, t_r = bench_many([lambda: d121(x), lambda: r50(x)], reps=3)
for name, m, t in [("DenseNet-121", d121, t_d), ("ResNet-50", r50, t_r)]:
    print(f"{name:13s} tham số {nparams(m)/1e6:6.2f} M | MACs {macs(m, x)/1e9:5.2f} G | độ trễ {t:6.1f} ms")
```

<!-- output -->
```output
DenseNet-121  tham số   7.98 M | MACs  2.83 G | độ trễ   61.5 ms
ResNet-50     tham số  25.56 M | MACs  4.09 G | độ trễ   63.1 ms
```
<!-- /output -->

DenseNet-121 có chưa tới một phần ba số tham số và khoảng 70% MACs của ResNet-50, nhưng độ trễ trên CPU xấp xỉ nhau.

## 4. MobileNet v1

<small class="viz-ref">Bài MobileNet v1: mục 1, 2.1, 2.2, 5</small>

Tỉ lệ lượng tính toán của tích chập tách được so với tích chập thường, theo công thức và đếm bằng công cụ ($D_K = 3$, $M = N$, $D_G = 28$):

<!-- run -->
```python
for N in (64, 256, 1024):
    x = torch.randn(1, N, 28, 28)
    std = nn.Conv2d(N, N, 3, padding=1, bias=False)
    sep = nn.Sequential(nn.Conv2d(N, N, 3, padding=1, groups=N, bias=False), nn.Conv2d(N, N, 1, bias=False))
    print(f"N = {N:4d}: đếm {macs(sep, x)/macs(std, x):.4f} | công thức 1/N + 1/9 = {1/N + 1/9:.4f} | tham số {nparams(sep)/nparams(std):.4f}")

inp = torch.randn(20, 10, 224, 224)
conv = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=3, groups=5)
print("groups=5:", tuple(conv.weight.shape), tuple(conv(inp).shape))
for kw in [dict(in_channels=10, out_channels=5, kernel_size=3, groups=6), dict(in_channels=10, out_channels=6, kernel_size=3, groups=5),
           dict(in_channels=10, output_channels=5, kernel_size=3, groups=5)]:
    try:
        nn.Conv2d(**kw)
    except Exception as e:
        print(f"{kw} -> {type(e).__name__}: {str(e)[:60]}")
```

<!-- output -->
```output
N =   64: đếm 0.1267 | công thức 1/N + 1/9 = 0.1267 | tham số 0.1267
N =  256: đếm 0.1150 | công thức 1/N + 1/9 = 0.1150 | tham số 0.1150
N = 1024: đếm 0.1121 | công thức 1/N + 1/9 = 0.1121 | tham số 0.1121
groups=5: (5, 2, 3, 3) (20, 5, 222, 222)
{'in_channels': 10, 'out_channels': 5, 'kernel_size': 3, 'groups': 6} -> ValueError: in_channels must be divisible by groups
{'in_channels': 10, 'out_channels': 6, 'kernel_size': 3, 'groups': 5} -> ValueError: out_channels must be divisible by groups
{'in_channels': 10, 'output_channels': 5, 'kernel_size': 3, 'groups': 5} -> TypeError: Conv2d.__init__() got an unexpected keyword argument 'output
```
<!-- /output -->

Code MobileNet v1 của bản gốc (tầng DW viết sai) và bản đã sửa:

<!-- run -->
```python
def convbn_relu(cin, cout, stride=2):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class MobilnetV1Block(nn.Module):
    def __init__(self, cin, cout, stride=1, original_bug=False):
        super().__init__()
        dw = nn.Conv2d(cin, 64, 3, stride, 1, groups=4, bias=False) if original_bug else \
             nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.dw = nn.Sequential(dw, nn.BatchNorm2d(cin), nn.ReLU(inplace=True))
        self.pw = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.pw(self.dw(x))

class MobileNetV1(nn.Module):
    stage_cfg = [(32, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 512, 2)] + \
                [(512, 512, 1)] * 5 + [(512, 1024, 2), (1024, 1024, 1)]
    def __init__(self, num_classes=1000, original_bug=False):
        super().__init__()
        self.first_conv = convbn_relu(3, 32, 2)
        self.stage_layers = nn.Sequential(*[MobilnetV1Block(a, b, s, original_bug) for a, b, s in self.stage_cfg])
        self.linear = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = self.stage_layers(self.first_conv(x))
        return self.linear(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))

x = torch.rand(1, 3, 224, 224)
try:
    MobileNetV1(original_bug=True).eval()(x)
except Exception as e:
    print("bản gốc:", type(e).__name__, "-", str(e).splitlines()[0][:100])
mb = MobileNetV1().eval()
print(f"bản sửa: đầu ra {tuple(mb(x).shape)}, {nparams(mb)/1e6:.3f} triệu tham số, {macs(mb, x)/1e6:.0f} triệu MACs")

# phân bổ MACs và tham số theo loại tầng (bảng 2 của bài báo)
groups = {"Conv 1x1": [0, 0], "Conv DW 3x3": [0, 0], "Conv 3x3 đầu": [0, 0], "FC": [0, 0]}
def kind(m):
    if isinstance(m, nn.Linear): return "FC"
    if m.kernel_size == (1, 1): return "Conv 1x1"
    return "Conv DW 3x3" if m.groups > 1 else "Conv 3x3 đầu"
for m in mb.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        out = {}
        h = m.register_forward_hook(lambda mod, i, o: out.__setitem__('o', o))
        mb(x); h.remove()
        k = kind(m); groups[k][1] += m.weight.numel()
        groups[k][0] += out['o'].numel() * (m.weight[0].numel() if isinstance(m, nn.Conv2d) else m.in_features)
tm, tp = sum(v[0] for v in groups.values()), sum(v[1] for v in groups.values())
for k, (mm, pp) in groups.items():
    print(f"  {k:13s} MACs {mm/tm:7.2%}   tham số {pp/tp:7.2%}")
```

<!-- output -->
```output
bản gốc: RuntimeError - running_mean should contain 64 elements not 32
bản sửa: đầu ra (1, 1000), 4.232 triệu tham số, 569 triệu MACs
  Conv 1x1      MACs  94.86%   tham số  74.59%
  Conv DW 3x3   MACs   3.06%   tham số   1.06%
  Conv 3x3 đầu  MACs   1.91%   tham số   0.02%
  FC            MACs   0.18%   tham số  24.33%
```
<!-- /output -->

## 5. ShuffleNet v2

<small class="viz-ref">Bài ShuffleNet v2: mục 2 (G1–G4), 3</small>

**G1**: 10 khối, mỗi khối hai tích chập 1×1 ($c_1 \to c_2 \to c_1$), feature map 56×56, cùng FLOPs; code của bản gốc đã sửa `padding=0`. In thêm MAC lý thuyết (số phần tử đọc ghi) của một khối:

<!-- run -->
```python
def conv_net(c1, c2, k=1, n=10):
    blk = lambda: nn.Sequential(nn.Conv2d(c1, c2, k, padding=k // 2), nn.ReLU(), nn.Conv2d(c2, c1, k, padding=k // 2), nn.ReLU())
    return nn.Sequential(*[blk() for _ in range(n)]).eval()

cfgs = {"1:1": (128, 128), "1:2": (90, 180), "1:6": (52, 312), "1:12": (36, 432)}
for k in (1, 3):
    nets = {r: conv_net(c1, c2, k) for r, (c1, c2) in cfgs.items()}
    xs = {r: torch.randn(1, c1, 56, 56) for r, (c1, c2) in cfgs.items()}
    ts = bench_many([lambda r=r: nets[r](xs[r]) for r in cfgs], reps=5)
    print(f"tích chập {k}x{k}, batch 1:")
    for (r, (c1, c2)), t in zip(cfgs.items(), ts):
        hw = 56 * 56
        mac = 2 * (hw * (c1 + c2) + c1 * c2 * k * k)
        print(f"  {r:5s} ({c1:3d},{c2:3d})  MACs {macs(nets[r], xs[r])/1e6:7.1f} M  MAC lý thuyết/khối {mac/1e6:5.2f} M phần tử  {1000/t:7.1f} ảnh/s")
```

<!-- output -->
```output
tích chập 1x1, batch 1:
  1:1   (128,128)  MACs  1027.6 M  MAC lý thuyết/khối  1.64 M phần tử     69.8 ảnh/s
  1:2   ( 90,180)  MACs  1016.1 M  MAC lý thuyết/khối  1.73 M phần tử     56.6 ảnh/s
  1:6   ( 52,312)  MACs  1017.6 M  MAC lý thuyết/khối  2.32 M phần tử     54.6 ảnh/s
  1:12  ( 36,432)  MACs   975.4 M  MAC lý thuyết/khối  2.97 M phần tử     47.0 ảnh/s
tích chập 3x3, batch 1:
  1:1   (128,128)  MACs  9248.4 M  MAC lý thuyết/khối  1.90 M phần tử     12.0 ảnh/s
  1:2   ( 90,180)  MACs  9144.6 M  MAC lý thuyết/khối  1.99 M phần tử     10.6 ảnh/s
  1:6   ( 52,312)  MACs  9158.1 M  MAC lý thuyết/khối  2.58 M phần tử     10.0 ảnh/s
  1:12  ( 36,432)  MACs  8778.8 M  MAC lý thuyết/khối  3.22 M phần tử      7.9 ảnh/s
```
<!-- /output -->

Với tích chập 1×1, tỉ lệ 1:1 nhanh nhất và tốc độ giảm dần khi MAC lý thuyết tăng, đúng G1. Với tích chập 3×3 (bị giới hạn bởi tính toán hơn), xu hướng cùng chiều nhưng yếu hơn.

**G2**: 10 tầng tích chập nhóm 1×1, giữ FLOPs bằng cách tăng số kênh theo $\sqrt{g}$:

<!-- run -->
```python
base = 128
nets, xs = {}, {}
for g in (1, 2, 4, 8):
    c = int(round(base * math.sqrt(g) / g)) * g           # c^2 / g ≈ base^2, c chia hết cho g
    nets[g] = nn.Sequential(*[nn.Sequential(nn.Conv2d(c, c, 1, groups=g), nn.ReLU()) for _ in range(10)]).eval()
    xs[g] = torch.randn(1, c, 56, 56)
ts = bench_many([lambda g=g: nets[g](xs[g]) for g in nets], reps=5)
for (g, net), t in zip(nets.items(), ts):
    c = xs[g].shape[1]
    print(f"g = {g}: c = {c:3d}, MACs {macs(net, xs[g])/1e6:6.1f} M, MAC/tầng {(2*56*56*c + c*c/g)/1e6:5.2f} M phần tử, {1000/t:7.1f} ảnh/s")
```

<!-- output -->
```output
g = 1: c = 128, MACs  513.8 M, MAC/tầng  0.82 M phần tử,   145.4 ảnh/s
g = 2: c = 182, MACs  519.4 M, MAC/tầng  1.16 M phần tử,   127.5 ảnh/s
g = 4: c = 256, MACs  513.8 M, MAC/tầng  1.62 M phần tử,    89.0 ảnh/s
g = 8: c = 360, MACs  508.0 M, MAC/tầng  2.27 M phần tử,    71.1 ảnh/s
```
<!-- /output -->

Cùng MACs, tăng số nhóm từ 1 lên 8 làm tốc độ giảm khoảng một nửa, cùng chiều với MAC tăng, đúng G2.

**G3** (khối gồm 1, 2, 4 tích chập 1×1 nối tiếp hoặc song song, cùng FLOPs) và **G4** (khối bottleneck của ResNet khi bỏ ReLU và/hoặc shortcut):

<!-- run -->
```python
class Frag(nn.Module):
    def __init__(self, c, n, parallel):
        super().__init__()
        self.parallel = parallel
        cc = c if not parallel else c          # mỗi nhánh song song giữ c kênh, số nhánh n
        self.convs = nn.ModuleList([nn.Conv2d(c, c // n if parallel else c, 1) for _ in range(n)]) if parallel else \
                     nn.ModuleList([nn.Conv2d(c, c, 1) for _ in range(n)])
        self.scale = None
    def forward(self, x):
        if self.parallel:
            return torch.cat([F.relu(cv(x)) for cv in self.convs], 1)
        for cv in self.convs:
            x = F.relu(cv(x))
        return x

def frag_net(c, n, parallel):
    return nn.Sequential(*[Frag(c, n, parallel) for _ in range(10)]).eval()

c1 = 128   # 1 mảnh: c=128; n mảnh nối tiếp: chỉnh c để FLOPs như nhau; n mảnh song song: mỗi nhánh c/n kênh ra
cfg = {"1 mảnh": (128, 1, False), "2 nối tiếp": (90, 2, False), "4 nối tiếp": (64, 4, False),
       "2 song song": (128, 2, True), "4 song song": (128, 4, True)}
nets = {k: frag_net(*v) for k, v in cfg.items()}
xs = {k: torch.randn(1, v[0], 56, 56) for k, v in cfg.items()}
ts = bench_many([lambda k=k: nets[k](xs[k]) for k in cfg], reps=5)
print("G3:")
for (k, net), t in zip(nets.items(), ts):
    print(f"  {k:12s} MACs {macs(net, xs[k])/1e6:6.1f} M  {1000/t:7.1f} ảnh/s")

class Bottleneck(nn.Module):
    def __init__(self, c=256, relu=True, shortcut=True):
        super().__init__()
        self.c1, self.c2, self.c3 = nn.Conv2d(c, c // 4, 1), nn.Conv2d(c // 4, c // 4, 3, padding=1), nn.Conv2d(c // 4, c, 1)
        self.relu, self.shortcut = relu, shortcut
    def forward(self, x):
        a = (lambda t: F.relu(t)) if self.relu else (lambda t: t)
        y = self.c3(a(self.c2(a(self.c1(x)))))
        if self.shortcut: y = y + x
        return a(y)

variants = {"ReLU + shortcut": (True, True), "bỏ ReLU": (False, True), "bỏ shortcut": (True, False), "bỏ cả hai": (False, False)}
nets = {k: nn.Sequential(*[Bottleneck(256, *v) for _ in range(10)]).eval() for k, v in variants.items()}
x = torch.randn(1, 256, 56, 56)
ts = bench_many([lambda k=k: nets[k](x) for k in variants], reps=5)
print("G4 (khối bottleneck 256 kênh, 56x56, chồng 10 lần):")
for (k, _), t in zip(variants.items(), ts):
    print(f"  {k:16s} {t:6.2f} ms   nhanh hơn {ts[0]/t - 1:6.1%}")
```

<!-- output -->
```output
G3:
  1 mảnh       MACs  513.8 M    148.0 ảnh/s
  2 nối tiếp   MACs  508.0 M     91.1 ảnh/s
  4 nối tiếp   MACs  513.8 M     92.6 ảnh/s
  2 song song  MACs  513.8 M    108.0 ảnh/s
  4 song song  MACs  513.8 M     74.1 ảnh/s
G4 (khối bottleneck 256 kênh, 56x56, chồng 10 lần):
  ReLU + shortcut   30.29 ms   nhanh hơn   0.0%
  bỏ ReLU           29.62 ms   nhanh hơn   2.2%
  bỏ shortcut       28.31 ms   nhanh hơn   7.0%
  bỏ cả hai         26.31 ms   nhanh hơn  15.1%
```
<!-- /output -->

Trên CPU, khối một mảnh nhanh nhất, khối 4 nhánh song song chậm nhất (khoảng một nửa), đúng G3. Bỏ ReLU và shortcut làm khối bottleneck nhanh hơn khoảng 15%, cùng cỡ con số 20% của bài báo (G4). Lưu ý các con số ảnh/s trong G2, G3 dao động 10–20% giữa các lần chạy.

Phép trộn kênh, và ShuffleNet v2 so với MobileNet v2:

<!-- run -->
```python
def channel_shuffle(x, groups=2):
    n, c, h, w = x.shape
    return x.view(n, groups, c // groups, h, w).transpose(1, 2).reshape(n, c, h, w)

x = torch.arange(8.).view(1, 8, 1, 1)
print("trộn kênh 2 nhóm:", channel_shuffle(x).flatten().tolist())
x = torch.randn(1, 3, 224, 224)
s2, m2 = torchvision.models.shufflenet_v2_x1_0().eval(), torchvision.models.mobilenet_v2().eval()
t_s, t_m = bench_many([lambda: s2(x), lambda: m2(x)], reps=10)
for name, m, t in [("ShuffleNet v2 1.0x", s2, t_s), ("MobileNet v2", m2, t_m)]:
    print(f"{name:19s} tham số {nparams(m)/1e6:5.2f} M | MACs {macs(m, x)/1e6:6.1f} M | độ trễ {t:5.2f} ms")
```

<!-- output -->
```output
trộn kênh 2 nhóm: [0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]
ShuffleNet v2 1.0x  tham số  2.28 M | MACs  144.9 M | độ trễ 13.33 ms
MobileNet v2        tham số  3.50 M | MACs  300.8 M | độ trễ 14.81 ms
```
<!-- /output -->

ShuffleNet v2 có một nửa MACs của MobileNet v2 nhưng chỉ nhanh hơn khoảng 10% trên CPU này.

## 6. RepVGG và ACNet

<small class="viz-ref">Bài RepVGG: mục 1.4, 5</small>

Tính cộng được của tích chập, rồi chuyển một khối ACBlock (3×3 + 1×3 + 3×1, mỗi nhánh có BN) sang một tích chập 3×3:

<!-- run -->
```python
torch.manual_seed(0)
I = torch.randn(1, 8, 16, 16); K1 = torch.randn(4, 8, 3, 3); K2 = torch.randn(4, 8, 1, 1)
lhs = F.conv2d(I, K1, padding=1) + F.conv2d(I, K2, padding=0)
rhs = F.conv2d(I, K1 + F.pad(K2, (1, 1, 1, 1)), padding=1)
print("I*K1 + I*K2 == I*(K1 ⊕ K2):", torch.allclose(lhs, rhs, atol=1e-5), "| sai khác", (lhs - rhs).abs().max().item())

def rand_bn(c):
    bn = nn.BatchNorm2d(c)
    bn.running_mean.uniform_(-1, 1); bn.running_var.uniform_(0.5, 2); bn.weight.uniform_(0.5, 1.5); bn.bias.uniform_(-0.5, 0.5)
    return bn.eval()

def fuse_bn(kernel, bn):
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * bn.weight / std

C = 16
sq, ver, hor = nn.Conv2d(C, C, 3, padding=1, bias=False), nn.Conv2d(C, C, (3, 1), padding=(1, 0), bias=False), nn.Conv2d(C, C, (1, 3), padding=(0, 1), bias=False)
bns = [rand_bn(C) for _ in range(3)]
x = torch.randn(2, C, 20, 20)
train_out = bns[0](sq(x)) + bns[1](ver(x)) + bns[2](hor(x))
k0, b0 = fuse_bn(sq.weight, bns[0]); k1, b1 = fuse_bn(ver.weight, bns[1]); k2, b2 = fuse_bn(hor.weight, bns[2])
k = k0.clone(); k[:, :, :, 1:2] += k1; k[:, :, 1:2, :] += k2
deploy_out = F.conv2d(x, k, b0 + b1 + b2, padding=1)
print("ACBlock: dạng huấn luyện vs dạng suy luận, sai khác", (train_out - deploy_out).abs().max().item())
```

<!-- output -->
```output
I*K1 + I*K2 == I*(K1 ⊕ K2): True | sai khác 1.049041748046875e-05
ACBlock: dạng huấn luyện vs dạng suy luận, sai khác 2.6226043701171875e-06
```
<!-- /output -->

Khối RepVGG (3×3 + 1×1 + đồng nhất, mỗi nhánh có BN), và độ trễ của một chồng 8 khối ở hai dạng:

<!-- run -->
```python
class RepVGGBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c3, self.bn3 = nn.Conv2d(c, c, 3, padding=1, bias=False), rand_bn(c)
        self.c1, self.bn1 = nn.Conv2d(c, c, 1, bias=False), rand_bn(c)
        self.bn0 = rand_bn(c)
        self.fused = None
    def forward(self, x):
        if self.fused is not None:
            return F.relu(self.fused(x))
        return F.relu(self.bn3(self.c3(x)) + self.bn1(self.c1(x)) + self.bn0(x))
    def reparam(self):
        c = self.c3.in_channels
        k3, b3 = fuse_bn(self.c3.weight, self.bn3)
        k1, b1 = fuse_bn(F.pad(self.c1.weight, (1, 1, 1, 1)), self.bn1)
        ident = torch.zeros(c, c, 3, 3); ident[range(c), range(c), 1, 1] = 1     # đồng nhất = tích chập 1x1 kernel đơn vị
        k0, b0 = fuse_bn(ident, self.bn0)
        self.fused = nn.Conv2d(c, c, 3, padding=1)
        self.fused.weight.copy_(k3 + k1 + k0); self.fused.bias.copy_(b3 + b1 + b0)

torch.manual_seed(1)
train = nn.Sequential(*[RepVGGBlock(64) for _ in range(8)]).eval()
deploy = copy.deepcopy(train)
for b in deploy: b.reparam()
x = torch.randn(1, 64, 56, 56)
print("RepVGG 8 khối: sai khác đầu ra", (train(x) - deploy(x)).abs().max().item(), "| |đầu ra| lớn nhất", train(x).abs().max().item())
t_tr, t_de = bench_many([lambda: train(x), lambda: deploy(x)], reps=10)
print(f"độ trễ: dạng huấn luyện {t_tr:.2f} ms, dạng suy luận {t_de:.2f} ms (nhanh hơn {t_tr/t_de - 1:.0%})")
```

<!-- output -->
```output
RepVGG 8 khối: sai khác đầu ra 5.340576171875e-05 | |đầu ra| lớn nhất 30.993921279907227
độ trễ: dạng huấn luyện 13.16 ms, dạng suy luận 9.17 ms (nhanh hơn 43%)
```
<!-- /output -->

Sau tái tham số hoá, đầu ra khớp tới sai số làm tròn (5×10⁻⁵ trên giá trị cỡ 30), và chồng 8 khối chạy nhanh hơn rõ rệt vì bỏ được nhánh 1×1, các phép BN và phép cộng.

## 7. CSPNet

<small class="viz-ref">Bài CSPNet: mục 3.1, 6</small>

Code ở mục 6 của bài (đã sửa), so DenseBlock đầy đủ với CSP DenseBlock (64 kênh vào, 4 tầng, $k=32$), và đếm lượng vào/ra (CIO) của các tầng tích chập:

<!-- run -->
```python
class conv2d_bn_relu(nn.Module):
    def __init__(self, cin, cout, k, s, p):
        super().__init__(); self.seq = nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU())
    def forward(self, x): return self.seq(x)

class bn_relu_conv2d(nn.Module):
    def __init__(self, cin, cout, k, s, p):
        super().__init__(); self.seq = nn.Sequential(nn.BatchNorm2d(cin), nn.ReLU(), nn.Conv2d(cin, cout, k, s, p, bias=False))
    def forward(self, x): return self.seq(x)

class DenseBlock(nn.Module):
    def __init__(self, cin, num_layers, k):
        super().__init__()
        self.layers = nn.ModuleList([nn.Sequential(bn_relu_conv2d(cin + i * k, 4 * k, 1, 1, 0), bn_relu_conv2d(4 * k, k, 3, 1, 1))
                                     for i in range(num_layers)])
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = torch.cat((out, layer(out)), 1)
        return out

class CSP_DenseBlock(nn.Module):
    def __init__(self, cin, num_layers, k, part_ratio=0.5):
        super().__init__()
        self.p1 = int(cin * part_ratio); self.p2 = cin - self.p1
        self.dense = DenseBlock(self.p2, num_layers, k)
        t = self.p2 + k * num_layers
        self.transition = conv2d_bn_relu(t, t, 1, 1, 0)
    def forward(self, x):
        part1, part2 = x[:, :self.p1], x[:, self.p1:]
        return torch.cat((part1, self.transition(self.dense(part2))), 1)

def conv_cio(model, x):
    tot = [0]
    hs = [m.register_forward_hook(lambda mod, i, o: tot.__setitem__(0, tot[0] + i[0].shape[1] + o.shape[1]))
          for m in model.modules() if isinstance(m, nn.Conv2d)]
    model(x)
    for h in hs: h.remove()
    return tot[0]

for cin in (64, 256):
    x = torch.randn(1, cin, 56, 56)
    dense = nn.Sequential(DenseBlock(cin, 4, 32), conv2d_bn_relu(cin + 4 * 32, cin + 4 * 32, 1, 1, 0)).eval()   # khối + tầng chuyển tiếp
    csp = CSP_DenseBlock(cin, 4, 32).eval()
    print(f"{cin} kênh vào, 4 tầng, k = 32:")
    r = []
    for name, m in [("DenseBlock + transition", dense), ("CSP DenseBlock", csp)]:
        y = m(x); r.append((nparams(m), macs(m, x), conv_cio(m, x)))
        print(f"  {name:24s} ra {y.shape[1]:3d} kênh | tham số {r[-1][0]:9,} | MACs {r[-1][1]/1e6:7.1f} M | tổng số kênh vào + ra các tầng conv {r[-1][2]}")
    print(f"  CSP / Dense: tham số {r[1][0]/r[0][0]:.2f}, MACs {r[1][1]/r[0][1]:.2f}, kênh vào+ra {r[1][2]/r[0][2]:.2f}")
```

<!-- output -->
```output
64 kênh vào, 4 tầng, k = 32:
  DenseBlock + transition  ra 192 kênh | tham số   243,968 | MACs   757.9 M | tổng số kênh vào + ra các tầng conv 1984
  CSP DenseBlock           ra 192 kênh | tham số   216,000 | MACs   671.2 M | tổng số kênh vào + ra các tầng conv 1792
  CSP / Dense: tham số 0.89, MACs 0.89, kênh vào+ra 0.90
256 kênh vào, 4 tầng, k = 32:
  DenseBlock + transition  ra 384 kênh | tham số   454,784 | MACs  1413.0 M | tổng số kênh vào + ra các tầng conv 3136
  CSP DenseBlock           ra 384 kênh | tham số   306,048 | MACs   950.5 M | tổng số kênh vào + ra các tầng conv 2368
  CSP / Dense: tham số 0.67, MACs 0.67, kênh vào+ra 0.76
```
<!-- /output -->

Mức tiết kiệm của CSP phụ thuộc tỉ lệ giữa số kênh vào và tốc độ tăng trưởng: với 64 kênh vào, phần lớn tính toán nằm ở các tầng 3×3 có số kênh không đổi nên chỉ giảm khoảng 11%; với 256 kênh vào giảm khoảng một phần ba.

## 8. VoVNet

<small class="viz-ref">Bài VoVNet: mục 3, 5</small>

Code `vovnet.py` của bản gốc (đã sửa thụt lề), dựng ba phiên bản và đo trên CPU cùng ResNet-50 và DenseNet-121:

<!-- run -->
```python
def conv3x3(cin, cout, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    return [(f'{module_name}_{postfix}/conv', nn.Conv2d(cin, cout, kernel_size, stride, padding, groups=groups, bias=False)),
            (f'{module_name}_{postfix}/norm', nn.BatchNorm2d(cout)), (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))]

def conv1x1(cin, cout, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    return conv3x3(cin, cout, module_name, postfix, stride, groups, kernel_size, padding)

class _OSA_module(nn.Module):
    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, identity=False):
        super().__init__()
        self.identity = identity
        self.layers = nn.ModuleList()
        c = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(OrderedDict(conv3x3(c, stage_ch, module_name, i)))); c = stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_ch + layer_per_block * stage_ch, concat_ch, module_name, 'concat')))
    def forward(self, x):
        identity_feat, output = x, [x]
        for layer in self.layers:
            x = layer(x); output.append(x)
        xt = self.concat(torch.cat(output, dim=1))
        return xt + identity_feat if self.identity else xt

class _OSA_stage(nn.Sequential):
    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num):
        super().__init__()
        if stage_num != 2:
            self.add_module('Pooling', nn.MaxPool2d(3, 2, ceil_mode=True))
        self.add_module(f'OSA{stage_num}_1', _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, f'OSA{stage_num}_1'))
        for i in range(block_per_stage - 1):
            self.add_module(f'OSA{stage_num}_{i+2}', _OSA_module(concat_ch, stage_ch, concat_ch, layer_per_block, f'OSA{stage_num}_{i+2}', identity=True))

class VoVNet(nn.Module):
    def __init__(self, config_stage_ch, config_concat_ch, block_per_stage, layer_per_block, num_classes=1000):
        super().__init__()
        stem = conv3x3(3, 64, 'stem', '1', 2) + conv3x3(64, 64, 'stem', '2', 1) + conv3x3(64, 128, 'stem', '3', 2)
        self.stem = nn.Sequential(OrderedDict(stem))
        in_ch_list = [128] + config_concat_ch[:-1]
        self.stages = nn.Sequential(*[_OSA_stage(in_ch_list[i], config_stage_ch[i], config_concat_ch[i], block_per_stage[i], layer_per_block, i + 2) for i in range(4)])
        self.classifier = nn.Linear(config_concat_ch[-1], num_classes)
    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))

vovs = {"VoVNet-27-slim": VoVNet([64, 80, 96, 112], [128, 256, 384, 512], [1, 1, 1, 1], 5),
        "VoVNet-39": VoVNet([128, 160, 192, 224], [256, 512, 768, 1024], [1, 1, 2, 2], 5),
        "VoVNet-57": VoVNet([128, 160, 192, 224], [256, 512, 768, 1024], [1, 1, 4, 3], 5),
        "ResNet-50": torchvision.models.resnet50(), "DenseNet-121": torchvision.models.densenet121()}
for m in vovs.values(): m.eval()
x = torch.randn(1, 3, 224, 224)
ts = bench_many([lambda m=m: m(x) for m in vovs.values()], reps=3)
for (name, m), t in zip(vovs.items(), ts):
    n_w = sum(isinstance(k, (nn.Conv2d, nn.Linear)) for k in m.modules())
    print(f"{name:15s} {n_w:3d} tầng có trọng số | tham số {nparams(m)/1e6:6.2f} M | MACs {macs(m, x)/1e9:5.2f} G | độ trễ {t:6.1f} ms")
```

<!-- output -->
```output
VoVNet-27-slim   28 tầng có trọng số | tham số   3.53 M | MACs  2.18 G | độ trễ   29.9 ms
VoVNet-39        40 tầng có trọng số | tham số  22.60 M | MACs  7.07 G | độ trễ   84.1 ms
VoVNet-57        58 tầng có trọng số | tham số  36.64 M | MACs  8.93 G | độ trễ  109.0 ms
ResNet-50        54 tầng có trọng số | tham số  25.56 M | MACs  4.09 G | độ trễ   62.8 ms
DenseNet-121    121 tầng có trọng số | tham số   7.98 M | MACs  2.83 G | độ trễ   61.1 ms
```
<!-- /output -->

Lượng truy cập bộ nhớ của một DenseBlock và một mô-đun OSA cùng số tầng (5 tầng tích chập 3×3, 43 kênh mỗi tầng, như ví dụ DenseNet-40 trên CIFAR của bài), đếm số phần tử đọc (đầu vào, trọng số) và ghi (đầu ra) của mọi tầng tích chập:

<!-- run -->
```python
def conv_mac(model, x):
    tot = [0]
    def hook(mod, i, o):
        tot[0] += i[0].numel() + o.numel() + mod.weight.numel()
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    model(x)
    for h in hs: h.remove()
    return tot[0]

class PlainDense(nn.Module):       # DenseBlock không bottleneck: tầng i nhận concat mọi tầng trước
    def __init__(self, cin, n, k):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(cin + i * k, k, 3, padding=1, bias=False) for i in range(n)])
    def forward(self, x):
        for l in self.layers:
            x = torch.cat([x, l(x)], 1)
        return x

cin, k, n = 43, 43, 5
x = torch.randn(1, cin, 32, 32)
# cả hai kết thúc bằng một tích chập 1x1 gộp (tầng chuyển tiếp của DenseNet, tầng concat của OSA)
dense = nn.Sequential(PlainDense(cin, n, k), nn.Conv2d(cin + n * k, cin + n * k, 1, bias=False)).eval()
osa = _OSA_module(cin, k, cin + n * k, n, 'osa').eval()
for name, m in [("DenseBlock", dense), ("OSA", osa)]:
    print(f"{name:10s} tham số {nparams(m):7,} | MACs {macs(m, x)/1e6:6.1f} M | MAC các tầng conv {conv_mac(m, x)/1e6:5.2f} M phần tử")
```

<!-- output -->
```output
DenseBlock tham số 316,179 | MACs  323.8 M | MAC các tầng conv  1.73 M phần tử
OSA        tham số 150,715 | MACs  153.4 M | MAC các tầng conv  1.12 M phần tử
```
<!-- /output -->

Cùng số tầng và cùng số kênh ra, mô-đun OSA có số tham số và MACs bằng khoảng một nửa, và lượng truy cập bộ nhớ của các tầng tích chập ít hơn khoảng 35% so với DenseBlock (kèm tầng chuyển tiếp), cùng cỡ mức giảm 30% (3,7M → 2,5M) của bài báo.
