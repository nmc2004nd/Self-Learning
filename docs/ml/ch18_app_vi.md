# Chương 18 · Ví dụ áp dụng: từ dự đoán điểm tới danh sách gợi ý

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. [Chương 18](ml/ch18_vi) đánh giá lọc cộng tác bằng RMSE: dự đoán điểm cho các cặp (người dùng, phim) **đã biết trước**. Nhưng một hệ thống gợi ý thật phải chọn ra vài phim trong hàng nghìn phim chưa xem. Trang này đánh giá theo cách đó trên MovieLens 100k, và cho thấy một mô hình tốt về RMSE có thể rất tệ khi lập danh sách gợi ý.

## Bài toán

Với mỗi người dùng, hệ thống gợi ý 10 phim họ chưa đánh giá trong tập huấn luyện. Một gợi ý được tính là "trúng" nếu trong tập kiểm tra người dùng đó đã xem phim này và chấm từ 4 điểm trở lên. Hai thước đo:

- **Precision@10:** tỉ lệ phim trúng trong 10 phim gợi ý.
- **Recall@10:** tỉ lệ phim người dùng thích (trong tập kiểm tra) xuất hiện trong danh sách gợi ý.

## 1. Dữ liệu

<!-- run -->
```python
import numpy as np
import pandas as pd
import scipy.sparse as sp
import urllib.request, zipfile
from pathlib import Path
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
titles = pd.read_csv(path / "u.item", sep="|", header=None, encoding="latin-1")[1].to_numpy()

R = sp.csr_matrix((tr.r.astype(float), (tr.u, tr.i)), shape=(n_users, n_items))
B = (R > 0).astype(float)                                 # "đã xem hay chưa"
seen = B.toarray().astype(bool)
liked = te[te.r >= 4].groupby("u").i.apply(set)
print(f"số người dùng có ít nhất một phim thích trong tập kiểm tra: {len(liked)}")

def evaluate(score, N=10):
    s = np.array(score, dtype=float); s[seen] = -np.inf     # không gợi ý phim đã xem
    top = np.argsort(-s, axis=1)[:, :N]
    prec = np.mean([len(set(top[u]) & L) / N for u, L in liked.items()])
    rec = np.mean([len(set(top[u]) & L) / len(L) for u, L in liked.items()])
    return prec, rec, top
```

<!-- output -->
```output
số người dùng có ít nhất một phim thích trong tập kiểm tra: 934
```
<!-- /output -->

## 2. Bốn cách lập danh sách

1. **Phim phổ biến nhất:** gợi ý cho mọi người những phim có nhiều lượt đánh giá nhất. Không cá nhân hoá.
2. **Phim có điểm trung bình cao nhất.**
3. **Lọc cộng tác sản phẩm, xếp theo điểm dự đoán:** dự đoán điểm cho mọi phim chưa xem như ở [chương 18](ml/ch18_vi), rồi chọn 10 phim điểm cao nhất.
4. **Lọc cộng tác sản phẩm trên dữ liệu "đã xem":** bỏ qua số sao, chỉ dùng việc người dùng đã đánh giá phim nào. Điểm của một phim là tổng độ tương tự của nó với các phim người dùng đã xem.

<!-- run -->
```python
pop = np.asarray(B.sum(0)).ravel()
item_mean = np.asarray(R.sum(0)).ravel() / np.maximum(pop, 1)

# 3: dự đoán điểm (chuẩn hoá theo người dùng, như sách)
mu = np.asarray(R.sum(1)).ravel() / np.asarray(B.sum(1)).ravel()
Ybar = sp.csr_matrix((tr.r - mu[tr.u], (tr.u, tr.i)), shape=(n_users, n_items))
S = cosine_similarity(Ybar.T, dense_output=True); np.fill_diagonal(S, 0)
pred_rating = np.asarray(Ybar @ S) / (np.asarray(B @ np.abs(S)) + 1e-8) + mu[:, None]

# 4: dữ liệu "đã xem"
Sb = cosine_similarity(B.T, dense_output=True); np.fill_diagonal(Sb, 0)
score_implicit = np.asarray(B @ Sb)

results = {}
for name, sc in [("phổ biến nhất", np.tile(pop, (n_users, 1))),
                 ("điểm trung bình cao nhất", np.tile(item_mean, (n_users, 1))),
                 ("CF sản phẩm, điểm dự đoán", pred_rating),
                 ("CF sản phẩm, 'đã xem'", score_implicit)]:
    p, r, top = evaluate(sc)
    results[name] = top
    print(f"{name:28s}: precision@10 = {p:.3f}, recall@10 = {r:.3f}")
```

<!-- output -->
```output
phổ biến nhất               : precision@10 = 0.084, recall@10 = 0.150
điểm trung bình cao nhất    : precision@10 = 0.000, recall@10 = 0.000
CF sản phẩm, điểm dự đoán   : precision@10 = 0.016, recall@10 = 0.031
CF sản phẩm, 'đã xem'       : precision@10 = 0.134, recall@10 = 0.240
```
<!-- /output -->

## 3. Vì sao xếp theo điểm dự đoán lại thất bại

<!-- run -->
```python
for name in ["điểm trung bình cao nhất", "CF sản phẩm, điểm dự đoán", "CF sản phẩm, 'đã xem'"]:
    top = results[name]
    print(f"{name:28s}: số lượt đánh giá trung vị của các phim được gợi ý = {np.median(pop[top]):.0f}")
print(f"{'(toàn bộ phim)':28s}: số lượt đánh giá trung vị = {np.median(pop):.0f}")
u = 0
print("\nngười dùng 0, gợi ý theo điểm dự đoán:", [f"{titles[j][:25]} ({int(pop[j])} lượt)" for j in results["CF sản phẩm, điểm dự đoán"][u][:4]])
print("người dùng 0, gợi ý theo 'đã xem':    ", [f"{titles[j][:25]} ({int(pop[j])} lượt)" for j in results["CF sản phẩm, 'đã xem'"][u][:4]])
```

<!-- output -->
```output
điểm trung bình cao nhất    : số lượt đánh giá trung vị của các phim được gợi ý = 1
CF sản phẩm, điểm dự đoán   : số lượt đánh giá trung vị của các phim được gợi ý = 2
CF sản phẩm, 'đã xem'       : số lượt đánh giá trung vị của các phim được gợi ý = 282
(toàn bộ phim)              : số lượt đánh giá trung vị = 24

người dùng 0, gợi ý theo điểm dự đoán: ['Letter From Death Row, A  (3 lượt)', 'Night Flier (1997) (7 lượt)', 'Twisted (1996) (5 lượt)', 'Close Shave, A (1995) (104 lượt)']
người dùng 0, gợi ý theo 'đã xem':     ['E.T. the Extra-Terrestria (277 lượt)', 'Groundhog Day (1993) (258 lượt)', 'Batman (1989) (190 lượt)', 'True Lies (1994) (195 lượt)']
```
<!-- /output -->

Các phim có điểm trung bình hay điểm dự đoán cao nhất thường là những phim chỉ vài người xem: một phim có đúng 1 lượt đánh giá 5 sao có trung bình bằng 5. Điểm dự đoán của lọc cộng tác cũng bị kéo về những phim này, vì nó dựa trên rất ít láng giềng. Trong khi đó, người dùng gần như không bao giờ xem những phim đó, nên chúng hiếm khi "trúng".

Dữ liệu "đã xem" trả lời một câu hỏi khác và đúng hơn với mục tiêu: *người dùng có khả năng xem phim nào?* Việc một người chọn đánh giá phim nào đã chứa nhiều thông tin, không kém bản thân số sao.

Các con số xác nhận điều này: phim được gợi ý theo điểm trung bình có trung vị 1 lượt đánh giá, theo điểm dự đoán là 2 lượt. Ngược lại, lọc cộng tác trên dữ liệu "đã xem" gợi ý những phim có trung vị 282 lượt, phổ biến hơn hẳn phim thông thường (24 lượt). Cách này có một nhược điểm riêng: nó thiên về phim nổi tiếng, nên ít khi giúp người dùng khám phá phim lạ.

## 4. Gợi ý cho một người dùng

<!-- run -->
```python
u = 5
hist = tr[tr.u == u].sort_values("r", ascending=False)
print("vài phim người dùng 5 chấm cao:", [titles[i][:30] for i in hist.i[:6]])
top = results["CF sản phẩm, 'đã xem'"][u]
liked_u = liked.get(u, set())
print("\n10 phim gợi ý:")
for j in top:
    print(f"  {titles[j][:45]:45s}" + ("  <- trúng: người dùng chấm >= 4 trong tập kiểm tra" if j in liked_u else ""))
```

<!-- output -->
```output
vài phim người dùng 5 chấm cao: ['Citizen Kane (1941)', 'Lone Star (1996)', 'Wizard of Oz, The (1939)', "Breakfast at Tiffany's (1961)", 'Godfather, The (1972)', 'Big Night (1996)']

10 phim gợi ý:
  Empire Strikes Back, The (1980)              
  Silence of the Lambs, The (1991)               <- trúng: người dùng chấm >= 4 trong tập kiểm tra
  Forrest Gump (1994)                          
  Indiana Jones and the Last Crusade (1989)    
  Return of the Jedi (1983)                    
  Dances with Wolves (1990)                    
  Jaws (1975)                                  
  Dead Poets Society (1989)                    
  Terminator 2: Judgment Day (1991)            
  Stand by Me (1986)                           
```
<!-- /output -->

## Rút ra được gì

- RMSE đo khả năng dự đoán điểm cho những cặp *đã biết*; một danh sách gợi ý cần chọn đúng trong *mọi* phim chưa xem. Hai mục tiêu này khác nhau.
- Xếp hạng theo điểm dự đoán dễ dồn về các phim hiếm, ít người xem. Cần thêm ràng buộc (số lượt đánh giá tối thiểu) hoặc đổi cách chấm điểm.
- Một mốc đơn giản như "phim phổ biến nhất" thường khó vượt qua. Lọc cộng tác trên dữ liệu "đã xem" vượt được nó rõ rệt.
