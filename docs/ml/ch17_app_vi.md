# Chương 17 · Ví dụ áp dụng: gợi ý phim mới ra mắt

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Ở [trang Thư viện](ml/ch17_lib_vi), mô hình dựa trên nội dung ([chương 17](ml/ch17_vi)) chỉ tốt hơn việc đoán điểm trung bình một chút. Trang này xét tình huống mà nó thật sự có giá trị: phim **mới**, chưa ai đánh giá. Các phương pháp lọc cộng tác ([chương 18](ml/ch18_vi), [chương 19](ml/ch19_vi)) không làm gì được với phim như vậy.

## Bài toán

Đây là bài toán *khởi đầu lạnh* (cold start) cho sản phẩm. Ta mô phỏng nó trên MovieLens 100k: chọn ngẫu nhiên 20% số phim làm "phim mới", bỏ **mọi** đánh giá của chúng khỏi tập huấn luyện, rồi dự đoán những đánh giá đó. Với phim mới, ta chỉ biết thông tin nội dung: thể loại và năm phát hành.

## 1. Dữ liệu và cách chia

<!-- run -->
```python
import numpy as np
import pandas as pd
import urllib.request, zipfile
from pathlib import Path
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge

root = Path.home() / ".cache"
if not (root / "ml-100k" / "u.data").exists():
    root.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-100k.zip", root / "ml-100k.zip")
    zipfile.ZipFile(root / "ml-100k.zip").extractall(root)
path = root / "ml-100k"
R = pd.read_csv(path / "u.data", sep="\t", names=["user", "item", "rating", "ts"])
R[["user", "item"]] -= 1
items = pd.read_csv(path / "u.item", sep="|", header=None, encoding="latin-1")
n_users, n_items = 943, 1682

rng = np.random.default_rng(0)
new_items = rng.choice(n_items, n_items // 5, replace=False)
is_new = np.isin(R.item, new_items)
train, test = R[~is_new], R[is_new]
print(f"phim mới: {len(new_items)}; đánh giá huấn luyện: {len(train):,}; đánh giá của phim mới cần đoán: {len(test):,}")
```

<!-- output -->
```output
phim mới: 336; đánh giá huấn luyện: 80,641; đánh giá của phim mới cần đoán: 19,359
```
<!-- /output -->

## 2. Đặc trưng nội dung

Ngoài 19 cột thể loại như sách, thêm thập niên phát hành (ví dụ một người dùng có thể thích phim cũ). Thập niên được mã hoá one-hot ([chương 6](ml/ch06_vi)).

<!-- run -->
```python
genres = TfidfTransformer(smooth_idf=True, norm="l2").fit_transform(items.iloc[:, -19:].to_numpy()).toarray()
year = pd.to_datetime(items[2], format="%d-%b-%Y", errors="coerce").dt.year.fillna(1995).to_numpy()
decade = np.clip((year // 10) * 10, 1930, 1990)
dec_onehot = (decade[:, None] == np.arange(1930, 2000, 10)[None]).astype(float)
X_genre = genres
X_full = np.hstack([genres, 0.5 * dec_onehot])
print("số phim theo thập niên:", {int(k): int(v) for k, v in zip(*np.unique(decade, return_counts=True))})
```

<!-- output -->
```output
số phim theo thập niên: {1930: 31, 1940: 45, 1950: 54, 1960: 43, 1970: 53, 1980: 107, 1990: 1349}
```
<!-- /output -->

## 3. Các cách dự đoán cho phim mới

- **Trung bình toàn bộ:** cùng một con số cho mọi cặp (người dùng, phim).
- **Trung bình của người dùng:** người dễ tính chấm cao mọi phim.
- **Dựa trên nội dung:** mô hình ridge của sách cho mỗi người dùng, với $\alpha$ chọn trên một tập xác thực (cũng gồm các phim bị giấu đi khỏi phần huấn luyện còn lại).

<!-- run -->
```python
def fit_users(train_df, X, alpha):
    W = np.zeros((X.shape[1], n_users)); b = np.full(n_users, train_df.rating.mean())
    for u, g in train_df.groupby("user"):
        m = Ridge(alpha=alpha).fit(X[g.item], g.rating)
        W[:, u], b[u] = m.coef_, m.intercept_
    return W, b

def predict(W, b, df, X):
    return np.clip(np.einsum("ij,ji->i", X[df.item], W[:, df.user]) + b[df.user], 1, 5)

rmse = lambda p, df: np.sqrt(np.mean((p - df.rating.to_numpy()) ** 2))

# chọn alpha: giấu thêm 20% số phim còn lại làm tập xác thực
old = np.setdiff1d(np.arange(n_items), new_items)
val_items = rng.choice(old, len(old) // 5, replace=False)
tr2, va2 = train[~np.isin(train.item, val_items)], train[np.isin(train.item, val_items)]
best = {}
for name, X in [("thể loại", X_genre), ("thể loại + thập niên", X_full)]:
    scores = {a: rmse(predict(*fit_users(tr2, X, a), va2, X), va2) for a in [1, 3, 10, 30]}
    best[name] = min(scores, key=scores.get)
    print(f"{name:22s}: RMSE xác thực theo alpha {', '.join(f'{a}: {v:.4f}' for a, v in scores.items())} -> chọn {best[name]}")

mu = train.rating.mean()
user_mean = train.groupby("user").rating.mean().reindex(range(n_users)).fillna(mu).to_numpy()
print(f"\nRMSE trên các đánh giá của phim mới:")
print(f"  trung bình toàn bộ:             {rmse(np.full(len(test), mu), test):.4f}")
print(f"  trung bình của người dùng:      {rmse(user_mean[test.user], test):.4f}")
for name, X in [("thể loại", X_genre), ("thể loại + thập niên", X_full)]:
    print(f"  dựa trên nội dung ({name}): {rmse(predict(*fit_users(train, X, best[name]), test, X), test):.4f}")
```

<!-- output -->
```output
thể loại              : RMSE xác thực theo alpha 1: 1.0408, 3: 1.0272, 10: 1.0264, 30: 1.0345 -> chọn 10
thể loại + thập niên  : RMSE xác thực theo alpha 1: 1.0242, 3: 1.0104, 10: 1.0134, 30: 1.0266 -> chọn 3

RMSE trên các đánh giá của phim mới:
  trung bình toàn bộ:             1.1368
  trung bình của người dùng:      1.0518
  dựa trên nội dung (thể loại): 1.0291
  dựa trên nội dung (thể loại + thập niên): 1.0115
```
<!-- /output -->

Với phim mới, mô hình dựa trên nội dung (thể loại và thập niên) có RMSE 1.012, tốt hơn trung bình của người dùng (1.052) và tốt hơn hẳn trung bình toàn bộ (1.137). Thêm thập niên phát hành giúp rõ rệt: nhiều người có sở thích về phim cũ hay phim mới.

## 4. Gợi ý cụ thể cho một người dùng

Với một người dùng, xếp hạng các phim mới theo điểm dự đoán và so với điểm thật (nếu người đó đã xem):

<!-- run -->
```python
W, b = fit_users(train, X_full, best["thể loại + thập niên"])
u = 12
cand = pd.DataFrame({"user": u, "item": new_items})
cand["pred"] = predict(W, b, cand, X_full)
truth = test[test.user == u].set_index("item").rating
top = cand.sort_values("pred", ascending=False).head(8)
for _, r in top.iterrows():
    t = truth.get(r["item"], None)
    print(f"{items.iloc[int(r['item']), 1][:38]:38s} dự đoán {r['pred']:.2f}" + (f", người dùng đã chấm {t}" if t is not None else ""))
genre_names = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
wg = W[:19, u]
print("\nthể loại người dùng này thích nhất / ít thích nhất theo mô hình:", [genre_names[i] for i in np.argsort(-wg)[:3]], "/", [genre_names[i] for i in np.argsort(wg)[:3]])
```

<!-- output -->
```output
Lost Horizon (1937)                    dự đoán 4.74, người dùng đã chấm 4
Of Human Bondage (1934)                dự đoán 4.74
Citizen Kane (1941)                    dự đoán 4.69
Symphonie pastorale, La (1946)         dự đoán 4.69
Meet John Doe (1941)                   dự đoán 4.69, người dùng đã chấm 4
Outlaw, The (1943)                     dự đoán 4.63
Walkabout (1971)                       dự đoán 4.51
Big Sleep, The (1946)                  dự đoán 4.40, người dùng đã chấm 5

thể loại người dùng này thích nhất / ít thích nhất theo mô hình: ['Animation', 'Drama', 'Western'] / ['Horror', 'Musical', 'Children']
```
<!-- /output -->

Người dùng số 12 được gợi ý toàn phim của thập niên 1930–1940: mô hình đã học được người này thích phim cũ. Trong 8 phim gợi ý, người dùng đã thực sự xem 3 phim và chấm 4, 4 và 5 điểm.

## Rút ra được gì

- Với phim đã có nhiều đánh giá, thông tin thể loại thêm được rất ít. Với phim mới hoàn toàn, nó là thông tin duy nhất về phim, và mô hình dựa trên nội dung là một trong số ít cách có thể dùng.
- Mô hình của mỗi người dùng dễ diễn giải: các hệ số cho biết người đó thích thể loại nào.
- Thực tế, các hệ thống gợi ý thường kết hợp cả hai: dựa trên nội dung cho sản phẩm mới, lọc cộng tác khi đã có đủ đánh giá.
