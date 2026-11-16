# Chương 5 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Các khái niệm của [chương 5](ml/ch05_vi) được scikit-learn biến thành một giao diện chung cho mọi thuật toán. Hiểu giao diện này một lần là dùng được mọi mô hình ở các chương sau.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `scikit-learn`.

| Khái niệm trong sách | scikit-learn |
| --- | --- |
| Tham số mô hình $\theta$ (5.5) | thuộc tính có dấu gạch dưới ở cuối, như `coef_`, `cluster_centers_` |
| Học từ kinh nghiệm $E$ | `fit(X, y)` (có giám sát) hoặc `fit(X)` (không giám sát) |
| Nhiệm vụ $T$ | `predict`, `transform`, `fit_predict` |
| Phép đánh giá $P$ | `score`, và module `sklearn.metrics` |
| Tập huấn luyện, xác thực, kiểm tra (5.2) | `train_test_split`, `cross_val_score`, `GridSearchCV` |
| Hoàn thiện dữ liệu (5.3.5) | `SimpleImputer`, `KNNImputer` |
| Học bán giám sát (5.4.3) | `LabelSpreading`, `SelfTrainingClassifier` |

## 1. Một giao diện cho mọi loại bài toán

<small class="viz-ref">Mục 5.3, 5.4</small>

<!-- run -->
```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

clf = LogisticRegression(max_iter=5000).fit(Xtr, ytr)                 # phân loại: có giám sát
print("phân loại:  ", type(clf).__name__, "; tham số coef_", clf.coef_.shape, "; score =", round(clf.score(Xte, yte), 3))

reg = LinearRegression().fit(Xtr[:, :32], Xtr[:, 32:].sum(1))          # hồi quy: có giám sát
print("hồi quy:    ", type(reg).__name__, "; tham số coef_", reg.coef_.shape, "; score (R^2) =", round(reg.score(Xte[:, :32], Xte[:, 32:].sum(1)), 3))

km = KMeans(10, n_init=10, random_state=0).fit(Xtr)                   # phân cụm: không giám sát, fit không nhận y
print("phân cụm:   ", type(km).__name__, "; tham số cluster_centers_", km.cluster_centers_.shape, "; nhãn cụm của 5 ảnh kiểm tra:", km.predict(Xte[:5]))

pca = PCA(2).fit(Xtr)                                                 # giảm chiều: không giám sát, có transform
print("giảm chiều: ", type(pca).__name__, "; tham số components_", pca.components_.shape, "; 64 chiều ->", pca.transform(Xte).shape[1])
```

<!-- output -->
```output
phân loại:   LogisticRegression ; tham số coef_ (10, 64) ; score = 0.958
hồi quy:     LinearRegression ; tham số coef_ (32,) ; score (R^2) = 0.327
phân cụm:    KMeans ; tham số cluster_centers_ (10, 64) ; nhãn cụm của 5 ảnh kiểm tra: [4 2 3 4 7]
giảm chiều:  PCA ; tham số components_ (2, 64) ; 64 chiều -> 2
```
<!-- /output -->

Quy ước: mọi tham số học được từ dữ liệu có tên kết thúc bằng `_` và chỉ xuất hiện sau khi gọi `fit`. Các *siêu tham số* (như số cụm, `max_iter`) được truyền vào khi tạo đối tượng.

## 2. Phép đánh giá

<small class="viz-ref">Mục 5.1</small>

`score` trả về một phép đánh giá mặc định (độ chính xác cho phân loại, $R^2$ cho hồi quy). Module `sklearn.metrics` có nhiều phép đánh giá khác; chọn phép đánh giá nào phụ thuộc vào nhiệm vụ.

<!-- run -->
```python
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, adjusted_rand_score

pred = clf.predict(Xte)
print("độ chính xác:", round(accuracy_score(yte, pred), 4))
print("ma trận nhầm lẫn (hàng: nhãn thật, cột: dự đoán), 4 chữ số đầu:\n", confusion_matrix(yte, pred)[:4, :4])
print("hồi quy, sai số tuyệt đối trung bình:", round(mean_absolute_error(Xte[:, 32:].sum(1), reg.predict(Xte[:, :32])), 2))
print("phân cụm, chỉ số Rand điều chỉnh so với nhãn thật:", round(adjusted_rand_score(yte, km.predict(Xte)), 3))
```

<!-- output -->
```output
độ chính xác: 0.9583
ma trận nhầm lẫn (hàng: nhãn thật, cột: dự đoán), 4 chữ số đầu:
 [[35  0  0  0]
 [ 0 35  0  0]
 [ 0  1 34  0]
 [ 0  0  0 37]]
hồi quy, sai số tuyệt đối trung bình: 22.13
phân cụm, chỉ số Rand điều chỉnh so với nhãn thật: 0.646
```
<!-- /output -->

## 3. Hoàn thiện dữ liệu

<small class="viz-ref">Mục 5.3.5</small>

<!-- run -->
```python
from sklearn.impute import SimpleImputer, KNNImputer

rng = np.random.default_rng(0)
Xmiss = Xte.astype(float).copy()
mask = rng.random(Xmiss.shape) < 0.25
Xmiss[mask] = np.nan
for imp in [SimpleImputer(strategy="mean"), KNNImputer(n_neighbors=5)]:
    filled = imp.fit(np.vstack([Xtr, Xmiss])).transform(Xmiss)
    print(f"{type(imp).__name__:14s}: sai số trung bình ở các ô bị thiếu {np.abs(filled - Xte)[mask].mean():.2f}")
```

<!-- output -->
```output
SimpleImputer : sai số trung bình ở các ô bị thiếu 3.05
KNNImputer    : sai số trung bình ở các ô bị thiếu 1.27
```
<!-- /output -->

`KNNImputer` làm đúng việc mà [trang Code](ml/ch05_code_vi) tự viết: điền giá trị thiếu bằng trung bình của các điểm gần nhất (đo trên những đặc trưng còn lại).

## 4. Học bán giám sát

<small class="viz-ref">Mục 5.4.3</small>

Giả sử chỉ 30 trong số hơn 1400 ảnh huấn luyện có nhãn. scikit-learn quy ước dùng nhãn `-1` cho dữ liệu chưa có nhãn.

<!-- run -->
```python
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

y_partial = np.full_like(ytr, -1)
labeled = np.concatenate([rng.choice(np.flatnonzero(ytr == c), 3, replace=False) for c in range(10)])
y_partial[labeled] = ytr[labeled]
print("số ảnh có nhãn:", np.sum(y_partial != -1), "trên", len(ytr))

sup = LogisticRegression(max_iter=5000).fit(Xtr[labeled], ytr[labeled])
print("chỉ học từ 30 ảnh có nhãn:      ", round(sup.score(Xte, yte), 3))
ls = LabelSpreading(kernel="knn", n_neighbors=7).fit(Xtr, y_partial)
print("LabelSpreading (dùng cả ảnh chưa nhãn):", round(ls.score(Xte, yte), 3))
st = SelfTrainingClassifier(LogisticRegression(max_iter=5000), threshold=0.9).fit(Xtr, y_partial)
print("SelfTraining:                   ", round(st.score(Xte, yte), 3))
```

<!-- output -->
```output
số ảnh có nhãn: 30 trên 1437
chỉ học từ 30 ảnh có nhãn:       0.775
LabelSpreading (dùng cả ảnh chưa nhãn): 0.842
SelfTraining:                    0.808
```
<!-- /output -->

Hai ý tưởng khác nhau:

- `LabelSpreading` lan nhãn trên đồ thị nối mỗi điểm với các láng giềng gần nhất: điểm chưa có nhãn nhận nhãn của vùng nó thuộc về.
- `SelfTrainingClassifier` lặp lại: huấn luyện trên dữ liệu có nhãn, gán nhãn giả cho những điểm mô hình rất chắc chắn, rồi huấn luyện lại.

[Trang Ví dụ áp dụng](ml/ch05_app_vi) xem xét kỹ hơn khi nào cách nào có ích.

## Tóm tắt

- Mọi mô hình: tạo đối tượng với siêu tham số → `fit` → `predict`/`transform` → đánh giá.
- Tham số học được nằm trong các thuộc tính kết thúc bằng `_`.
- Đánh giá luôn trên dữ liệu mô hình chưa thấy; chọn phép đánh giá phù hợp với nhiệm vụ.
