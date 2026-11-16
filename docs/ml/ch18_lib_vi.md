# Chương 18 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Lớp `uuCF` ở [trang Code](ml/ch18_code_vi) dự đoán từng ô một bằng vòng lặp Python. Trang này viết lại lọc cộng tác lân cận ([chương 18](ml/ch18_vi)) bằng các phép toán trên ma trận thưa của SciPy và scikit-learn, nhanh hơn nhiều, và giới thiệu `NearestNeighbors` để tìm láng giềng.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `pandas`, `scipy`, `scikit-learn`. Dữ liệu MovieLens 100k được tải ở lần chạy đầu.

| Việc cần làm | Công cụ |
| --- | --- |
| Ma trận tiện ích | `scipy.sparse.csr_matrix` |
| Tương tự cos giữa mọi cặp cột | `sklearn.metrics.pairwise.cosine_similarity` (nhận ma trận thưa) |
| $k$ láng giềng gần nhất | `sklearn.neighbors.NearestNeighbors(metric="cosine")` |

## 1. Chuẩn hoá và tương tự cos trên ma trận thưa

<small class="viz-ref">Mục 18.2.1</small>

<!-- run -->
```python
import numpy as np
import pandas as pd
import scipy.sparse as sp
import urllib.request, zipfile
from pathlib import Path
from time import perf_counter
from sklearn.metrics.pairwise import cosine_similarity

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

# Ma trận tiện ích (sản phẩm x người dùng) và bản chuẩn hoá (trừ trung bình của mỗi người dùng)
mu = tr.groupby("u").r.mean().reindex(range(n_users)).to_numpy()
Ybar = sp.csr_matrix((tr.r - mu[tr.u], (tr.i, tr.u)), shape=(n_items, n_users))    # số thực: không bị cắt phần thập phân
t = perf_counter()
S_user = cosine_similarity(Ybar.T, dense_output=True)             # 943 x 943
S_item = cosine_similarity(Ybar, dense_output=True)               # 1682 x 1682
print(f"hai ma trận tương tự: {S_user.shape}, {S_item.shape}; {perf_counter() - t:.2f} s")
```

<!-- output -->
```output
hai ma trận tương tự: (943, 943), (1682, 1682); 0.04 s
```
<!-- /output -->

## 2. Dự đoán mọi ô kiểm tra cùng lúc

<small class="viz-ref">Mục 18.2.2, 18.3</small>

Với mỗi cặp (sản phẩm $i$, người dùng $u$) cần dự đoán, ta cần $k$ người dùng giống $u$ nhất trong số những người đã đánh giá $i$. Thay vì vòng lặp lồng nhau, gom các cặp theo sản phẩm: mọi người dùng cần dự đoán cho sản phẩm $i$ dùng chung tập "người đã đánh giá $i$".

<!-- run -->
```python
def predict_cf(Ybar_rows, S, mu_cols, pairs_row, pairs_col, k=40):
    """Ybar_rows: ma trận thưa (hàng = đối tượng được đánh giá). Dự đoán cho các cặp (hàng, cột)."""
    pred = np.empty(len(pairs_row))
    Ycsr = Ybar_rows.tocsr()
    order = np.argsort(pairs_row)
    for row in np.unique(pairs_row):
        sel = order[np.searchsorted(pairs_row[order], row, "left"): np.searchsorted(pairs_row[order], row, "right")]
        start, end = Ycsr.indptr[row], Ycsr.indptr[row + 1]
        raters, vals = Ycsr.indices[start:end], Ycsr.data[start:end]
        cols_q = pairs_col[sel]
        if len(raters) == 0:
            pred[sel] = mu_cols[cols_q]; continue
        sims = S[np.ix_(cols_q, raters)]                          # (số cột cần đoán) x (số người đã đánh giá)
        kk = min(k, len(raters))
        top = np.argpartition(sims, -kk, axis=1)[:, -kk:]
        s_top = np.take_along_axis(sims, top, 1)
        pred[sel] = (vals[top] * s_top).sum(1) / (np.abs(s_top).sum(1) + 1e-8) + mu_cols[cols_q]
    return pred

rmse = lambda p: np.sqrt(np.mean((p - te.r.to_numpy()) ** 2))
t = perf_counter()
p_uu = predict_cf(Ybar, S_user, mu, te.i.to_numpy(), te.u.to_numpy())
print(f"user-user, k = 40: RMSE {rmse(p_uu):.4f} ({perf_counter() - t:.2f} s)")

# item-item: đổi vai, chuẩn hoá theo trung bình của mỗi sản phẩm
mu_i = tr.groupby("i").r.mean().reindex(range(n_items)).fillna(tr.r.mean()).to_numpy()
Ybar_T = sp.csr_matrix((tr.r - mu_i[tr.i], (tr.u, tr.i)), shape=(n_users, n_items))
S_item2 = cosine_similarity(Ybar_T.T, dense_output=True)
t = perf_counter()
p_ii = predict_cf(Ybar_T, S_item2, mu_i, te.u.to_numpy(), te.i.to_numpy())
print(f"item-item, k = 40: RMSE {rmse(p_ii):.4f} ({perf_counter() - t:.2f} s)")
print(f"cắt dự đoán về [1, 5]: user-user {rmse(np.clip(p_uu, 1, 5)):.4f}, item-item {rmse(np.clip(p_ii, 1, 5)):.4f}")
```

<!-- output -->
```output
user-user, k = 40: RMSE 0.9580 (0.06 s)
item-item, k = 40: RMSE 0.9349 (0.05 s)
cắt dự đoán về [1, 5]: user-user 0.9568, item-item 0.9345
```
<!-- /output -->

User-user trùng với bản đã sửa ở [trang Code](ml/ch18_code_vi) (0.9580). Item-item lệch 0.001 (0.9349 so với 0.9359) vì cách xử lý hai đánh giá kiểm tra của những phim không xuất hiện trong tập huấn luyện: code của sách dự đoán 0 cho chúng, còn ở đây dùng điểm trung bình chung. Thời gian chạy giảm từ vài giây xuống dưới 0.1 giây. Cắt dự đoán về khoảng điểm hợp lệ $[1, 5]$ cải thiện thêm một chút.

## 3. `NearestNeighbors`: những phim giống nhau nhất

`NearestNeighbors` với `metric="cosine"` tìm các hàng gần nhất theo khoảng cách cos ($1 - \text{cos}$). Dùng trên các hàng của ma trận đã chuẩn hoá theo sản phẩm:

<!-- run -->
```python
from sklearn.neighbors import NearestNeighbors

titles = pd.read_csv(path / "u.item", sep="|", header=None, encoding="latin-1")[1].to_numpy()
item_rows = Ybar_T.T.tocsr()                                    # hàng = phim
nn = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute").fit(item_rows)
for q in [49, 0, 55]:                                           # Star Wars, Toy Story, Pulp Fiction
    dist, idx = nn.kneighbors(item_rows[q])
    print(f"{titles[q]}:\n   " + "\n   ".join(f"{titles[j][:40]:40s} cos = {1 - d:.3f}" for d, j in zip(dist[0][1:], idx[0][1:])))
```

<!-- output -->
```output
Star Wars (1977):
   Return of the Jedi (1983)                cos = 0.548
   Empire Strikes Back, The (1980)          cos = 0.522
   Raiders of the Lost Ark (1981)           cos = 0.348
   Indiana Jones and the Last Crusade (1989 cos = 0.236
   Sting, The (1973)                        cos = 0.212
Toy Story (1995):
   Beauty and the Beast (1991)              cos = 0.243
   Aladdin (1992)                           cos = 0.228
   That Thing You Do! (1996)                cos = 0.219
   Transformers: The Movie, The (1986)      cos = 0.212
   Lion King, The (1994)                    cos = 0.203
Pulp Fiction (1994):
   Cable Guy, The (1996)                    cos = 0.218
   Raising Arizona (1987)                   cos = 0.216
   GoodFellas (1990)                        cos = 0.202
   Trainspotting (1996)                     cos = 0.187
   From Dusk Till Dawn (1996)               cos = 0.181
```
<!-- /output -->

Láng giềng của *Star Wars* là hai phần tiếp theo của nó và hai phim Indiana Jones (cùng đạo diễn, cùng thời kỳ). Láng giềng của *Toy Story* là các phim hoạt hình Disney. Hệ thống không biết gì về nội dung phim; những nhóm này hình thành hoàn toàn từ việc cùng những người xem chấm điểm giống nhau.

## Tóm tắt

- Dùng ma trận thưa kiểu **số thực** cho ma trận tiện ích đã chuẩn hoá.
- `cosine_similarity` tính mọi cặp cùng lúc; gom các cặp cần dự đoán theo hàng để tránh vòng lặp lồng nhau.
- `NearestNeighbors(metric="cosine", algorithm="brute")` cho $k$ láng giềng gần nhất trực tiếp trên ma trận thưa.
