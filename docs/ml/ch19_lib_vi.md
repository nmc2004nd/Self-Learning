# Chương 19 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. scikit-learn không có sẵn một mô hình phân tích ma trận cho dữ liệu có ô bị thiếu như ở [chương 19](ml/ch19_vi). Trang so sánh những gì làm được bằng các công cụ có sẵn: SVD cắt ngọn của ma trận đã điền giá trị thiếu ([chương 20](ml/ch20_vi)), `NMF`, và mô hình viết bằng PyTorch với đạo hàm tự động.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`. Dữ liệu MovieLens 100k được tải ở lần chạy đầu.

| Cách làm | Công cụ | Xử lý ô thiếu |
| --- | --- | --- |
| SVD cắt ngọn | `scipy.sparse.linalg.svds`, `TruncatedSVD` | coi ô thiếu là 0 (sau khi trừ trung bình) |
| Phân tích ma trận không âm | `sklearn.decomposition.NMF` | coi ô thiếu là 0 |
| Hàm mất mát chỉ trên ô đã biết (như sách) | PyTorch + `torch.optim` | đúng |

## 1. SVD cắt ngọn của ma trận đã điền

Điền 0 vào ô thiếu sau khi trừ trung bình của mỗi người dùng, rồi lấy SVD cắt ngọn. Đây là xấp xỉ hạng thấp tốt nhất (mục 20.2.6) của ma trận *đã điền*, không phải của các ô đã biết.

<!-- run -->
```python
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import urllib.request, zipfile
from pathlib import Path

root = Path.home() / ".cache"
if not (root / "ml-100k" / "ua.base").exists():
    root.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-100k.zip", root / "ml-100k.zip")
    zipfile.ZipFile(root / "ml-100k.zip").extractall(root)
path = root / "ml-100k"
cols = ["u", "i", "r", "t"]
tr = pd.read_csv(path / "ua.base", sep="\t", names=cols); te = pd.read_csv(path / "ua.test", sep="\t", names=cols)
for d in (tr, te): d[["u", "i"]] -= 1
n_users, n_items = 943, 1682
rmse = lambda p: np.sqrt(np.mean((np.clip(p, 1, 5) - te.r.to_numpy()) ** 2))

mu_u = tr.groupby("u").r.mean().to_numpy()
Rc = sp.csr_matrix((tr.r - mu_u[tr.u], (tr.u, tr.i)), shape=(n_users, n_items))
print(f"chỉ dùng trung bình người dùng: RMSE {rmse(mu_u[te.u]):.4f}")
for k in [2, 10, 20, 50]:
    U, s, Vt = svds(Rc, k=k)
    pred = mu_u[te.u] + np.sum((U[te.u] * s) * Vt.T[te.i], axis=1)
    print(f"SVD cắt ngọn, k = {k:2d}: RMSE {rmse(pred):.4f}")
```

<!-- output -->
```output
chỉ dùng trung bình người dùng: RMSE 1.0431
SVD cắt ngọn, k =  2: RMSE 1.0167
SVD cắt ngọn, k = 10: RMSE 1.0031
SVD cắt ngọn, k = 20: RMSE 1.0022
SVD cắt ngọn, k = 50: RMSE 1.0078
```
<!-- /output -->

Ô thiếu được coi là "người dùng đánh giá đúng bằng trung bình của họ", một giả sử sai với 94% số ô. Với $k$ nhỏ, SVD vẫn cải thiện so với mốc; với $k$ lớn, nó bắt đầu khớp cả các số 0 giả và kết quả xấu đi.

## 2. `NMF`: phân tích ma trận không âm

`NMF` tìm $\mathbf{R} \approx \mathbf{W}\mathbf{H}$ với $\mathbf{W}, \mathbf{H}$ không âm. Áp dụng trực tiếp lên ma trận đánh giá thô (ô thiếu bằng 0) cho dự đoán bị kéo về 0, vì đa số ô bằng 0:

<!-- run -->
```python
from sklearn.decomposition import NMF

R = sp.csr_matrix((tr.r.astype(float), (tr.u, tr.i)), shape=(n_users, n_items))
nmf = NMF(n_components=10, init="nndsvda", max_iter=400, random_state=0)
Wn = nmf.fit_transform(R); Hn = nmf.components_
pred_nmf = np.sum(Wn[te.u] * Hn.T[te.i], axis=1)
print(f"NMF trên ma trận thô: dự đoán trung bình {pred_nmf.mean():.2f} (điểm thật trung bình {te.r.mean():.2f}); RMSE {rmse(pred_nmf):.4f}")
```

<!-- output -->
```output
NMF trên ma trận thô: dự đoán trung bình 1.02 (điểm thật trung bình 3.59); RMSE 2.5303
```
<!-- /output -->

Kết quả cho thấy vì sao cách điền 0 không dùng được để dự đoán điểm: mô hình học rằng hầu hết các ô có giá trị 0. (NMF trên dữ liệu "đã xem hay chưa" lại là một cách tốt để xếp hạng phim, như ở [ví dụ áp dụng chương 18](ml/ch18_app_vi).)

## 3. Hàm mất mát chỉ trên ô đã biết, với PyTorch

<small class="viz-ref">Mục 19.2.2</small>

Viết đúng mô hình của sách, $\hat{y}_{mn} = \mu + b_m + d_n + \mathbf{x}_m^T\mathbf{w}_n$, với hàm mất mát chỉ tính trên các ô đã biết. PyTorch tự tính gradient; bộ tối ưu Adam ([trang Thư viện chương 12](ml/ch12_lib_vi)) không cần chọn tốc độ học cẩn thận như GD.

<!-- run -->
```python
import torch
from torch import nn

torch.manual_seed(0)
class MFModel(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.P = nn.Embedding(n_users, K); self.Q = nn.Embedding(n_items, K)
        self.bu = nn.Embedding(n_users, 1); self.bi = nn.Embedding(n_items, 1)
        for e in (self.P, self.Q): nn.init.normal_(e.weight, std=0.1)
        for e in (self.bu, self.bi): nn.init.zeros_(e.weight)
        self.mu = float(tr.r.mean())
    def forward(self, u, i):
        return self.mu + self.bu(u).squeeze(1) + self.bi(i).squeeze(1) + (self.P(u) * self.Q(i)).sum(1)

u_t, i_t, r_t = (torch.tensor(tr[c].to_numpy()) for c in ["u", "i", "r"]); r_t = r_t.float()
u_v, i_v = torch.tensor(te.u.to_numpy()), torch.tensor(te.i.to_numpy())

def train(K, wd, wd_bias=0.01, epochs=20):
    m = MFModel(K)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    for ep in range(epochs):
        perm = torch.randperm(len(r_t))
        for b in perm.split(1024):
            opt.zero_grad()
            pu, pi = m.P(u_t[b]), m.Q(i_t[b])
            bu, bi = m.bu(u_t[b]), m.bi(i_t[b])
            loss = ((m(u_t[b], i_t[b]) - r_t[b]) ** 2).mean() + wd * ((pu ** 2).sum(1) + (pi ** 2).sum(1)).mean() \
                   + wd_bias * (bu ** 2 + bi ** 2).mean()
            loss.backward(); opt.step()
    with torch.no_grad():
        return rmse(m(u_v, i_v).numpy()), m

for K, wd in [(10, 0.0), (50, 0.05), (50, 0.1)]:
    score, mdl = train(K, wd)
    print(f"PyTorch, K = {K:2d}, phạt {wd}: RMSE kiểm tra {score:.4f}")
```

<!-- output -->
```output
PyTorch, K = 10, phạt 0.0: RMSE kiểm tra 1.0797
PyTorch, K = 50, phạt 0.05: RMSE kiểm tra 0.9762
PyTorch, K = 50, phạt 0.1: RMSE kiểm tra 0.9368
```
<!-- /output -->

Không có hệ số phạt, mô hình quá khớp nhanh. Hệ số phạt quyết định kết quả: với 0.1, RMSE gần mức của ALS tự viết ở [trang Code](ml/ch19_code_vi) (0.933). Hệ số này cũng cần được chọn bằng tập xác thực. Cách viết này mở rộng dễ dàng: thêm đặc trưng nội dung ([chương 17](ml/ch17_vi)), thêm thời gian, hay thay hàm mất mát.

## Tóm tắt

- SVD cắt ngọn và `NMF` của thư viện coi ô thiếu là 0: chỉ dùng được như mốc so sánh cho bài toán dự đoán điểm.
- Mô hình của sách (hàm mất mát chỉ trên ô đã biết, có độ lệch, có phạt) viết gọn bằng PyTorch với `nn.Embedding`.
