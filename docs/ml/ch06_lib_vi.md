# Chương 6 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang giới thiệu các công cụ xây dựng đặc trưng của scikit-learn cho những kỹ thuật ở [chương 6](ml/ch06_vi): vector hoá văn bản, mã hoá dữ liệu dạng hạng mục, chuẩn hoá, lựa chọn đặc trưng, và cách ghép tất cả thành một `Pipeline` để tránh rò rỉ thông tin từ tập kiểm tra.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `pandas`, `scikit-learn`. Mục 3 tải dữ liệu Titanic từ OpenML ở lần chạy đầu.

| Kỹ thuật trong sách | scikit-learn |
| --- | --- |
| Túi từ (6.3.4) | `CountVectorizer`, `TfidfVectorizer` |
| Dữ liệu dạng hạng mục (one-hot) | `OneHotEncoder` |
| Chuyển khoảng giá trị (6.5.1) | `MinMaxScaler` |
| Chuẩn hoá theo phân phối chuẩn (6.5.2) | `StandardScaler` |
| Chuẩn hoá về cùng norm (6.5.3) | `Normalizer` |
| Lựa chọn đặc trưng (6.3.2) | `VarianceThreshold`, `SelectKBest` |
| Giảm chiều dữ liệu (6.3.3) | `PCA` (chương 21) |

## 1. Vector hoá văn bản

<small class="viz-ref">Mục 6.3.4</small>

<!-- run -->
```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["John likes to watch movies. Mary likes movies too.",
        "John also likes to watch football games."]
cv = CountVectorizer(lowercase=False)
X = cv.fit_transform(docs)
print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out()).to_string())

# Cặp từ liên tiếp (bigram) giữ lại một phần thông tin về thứ tự từ, nhược điểm lớn nhất của BoW theo sách
cv2 = CountVectorizer(ngram_range=(1, 2))
print("\nsố đặc trưng với từ đơn + cặp từ:", len(cv2.fit(docs).get_feature_names_out()), "; ví dụ:", list(cv2.get_feature_names_out()[:6]))

tf = TfidfVectorizer()
print("\nTF-IDF (đã chuẩn hoá độ dài 1):")
print(pd.DataFrame(tf.fit_transform(docs).toarray(), columns=tf.get_feature_names_out()).round(3).to_string())
```

<!-- output -->
```output
   John  Mary  also  football  games  likes  movies  to  too  watch
0     1     1     0         0      0      2       2   1    1      1
1     1     0     1         1      1      1       0   1    0      1

số đặc trưng với từ đơn + cặp từ: 22 ; ví dụ: ['also', 'also likes', 'football', 'football games', 'games', 'john']

TF-IDF (đã chuẩn hoá độ dài 1):
    also  football  games   john  likes   mary  movies     to    too  watch
0  0.000     0.000  0.000  0.230  0.461  0.324   0.647  0.230  0.324  0.230
1  0.446     0.446  0.446  0.317  0.317  0.000   0.000  0.317  0.000  0.317
```
<!-- /output -->

`CountVectorizer` xếp từ điển theo thứ tự chữ cái thay vì thứ tự xuất hiện như trong sách, nhưng các số đếm giống nhau. `TfidfVectorizer` dùng phiên bản IDF có làm trơn $\log\frac{1 + N}{1 + n_t} + 1$, nên các từ chung của hai văn bản vẫn có trọng số khác 0 (khác với bản không làm trơn ở [trang Code](ml/ch06_code_vi)).

## 2. Chuẩn hoá: ba lớp của scikit-learn

<small class="viz-ref">Mục 6.5</small>

<!-- run -->
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

Xw, yw = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xw, yw, test_size=0.3, random_state=0, stratify=yw)
for sc in [MinMaxScaler(), StandardScaler(), Normalizer()]:
    Z = sc.fit(Xtr).transform(Xte)                     # fit trên tập huấn luyện, transform tập kiểm tra
    print(f"{type(sc).__name__:15s}: tập kiểm tra sau biến đổi, min {Z.min():7.3f}, max {Z.max():7.3f}, độ dài hàng đầu {np.linalg.norm(Z[0]):.3f}")
```

<!-- output -->
```output
MinMaxScaler   : tập kiểm tra sau biến đổi, min  -0.222, max   1.309, độ dài hàng đầu 2.037
StandardScaler : tập kiểm tra sau biến đổi, min  -3.823, max   4.476, độ dài hàng đầu 3.253
Normalizer     : tập kiểm tra sau biến đổi, min   0.000, max   0.998, độ dài hàng đầu 1.000
```
<!-- /output -->

`Normalizer` làm việc theo **hàng** (mỗi vector dữ liệu), còn hai lớp kia làm việc theo **cột** (mỗi đặc trưng). `Normalizer` vì vậy không cần học gì từ tập huấn luyện.

## 3. Dữ liệu hỗn hợp: `ColumnTransformer`

Dữ liệu thật thường có cả cột số lẫn cột chữ, và có giá trị bị thiếu. Ví dụ: danh sách hành khách tàu Titanic, với câu hỏi "hành khách này có sống sót không?".

<!-- run -->
```python
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

df = fetch_openml("titanic", version=1, as_frame=True).frame
cv5 = StratifiedKFold(5, shuffle=True, random_state=0)        # xáo trộn trước khi chia (xem ghi chú bên dưới)
y = df["survived"].astype(int)
print(df.shape, "\nsố giá trị thiếu:", df.isna().sum()[lambda s: s > 0].to_dict())

num = ["age", "fare", "sibsp", "parch"]
cat = ["pclass", "sex", "embarked"]
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat),
])
model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
print("\nđộ chính xác, kiểm định chéo KHÔNG xáo trộn:", cross_val_score(model, df, y, cv=5).mean().round(4))
print("độ chính xác, kiểm định chéo có xáo trộn: ", cross_val_score(model, df, y, cv=cv5).mean().round(4))
print("hạng vé của 10 dòng đầu và 10 dòng cuối:", list(df["pclass"][:10]), list(df["pclass"][-10:]))
print("số đặc trưng sau khi biến đổi:", model.fit(df, y).named_steps["prep"].transform(df.head()).shape[1])
print("tên các đặc trưng:", list(model.named_steps["prep"].get_feature_names_out()))
```

<!-- output -->
```output
(1309, 14) 
số giá trị thiếu: {'age': 263, 'fare': 1, 'cabin': 1014, 'embarked': 2, 'boat': 823, 'body': 1188, 'home.dest': 564}

độ chính xác, kiểm định chéo KHÔNG xáo trộn: 0.6494
độ chính xác, kiểm định chéo có xáo trộn:  0.7838
hạng vé của 10 dòng đầu và 10 dòng cuối: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
số đặc trưng sau khi biến đổi: 12
tên các đặc trưng: ['num__age', 'num__fare', 'num__sibsp', 'num__parch', 'cat__pclass_1', 'cat__pclass_2', 'cat__pclass_3', 'cat__sex_female', 'cat__sex_male', 'cat__embarked_C', 'cat__embarked_Q', 'cat__embarked_S']
```
<!-- /output -->

Không xáo trộn, độ chính xác chỉ khoảng 65%; có xáo trộn, nó tăng lên khoảng 78%. Dữ liệu từ OpenML được sắp xếp theo hạng vé, nên `cv=5` (chia theo thứ tự) tạo ra những phần kiểm tra toàn hành khách hạng 3 trong khi mô hình học chủ yếu trên hạng 1 và 2. Với dữ liệu có thứ tự, luôn xáo trộn trước khi chia (trừ khi thứ tự là thời gian và ta muốn dự đoán tương lai).

## 4. Tự tạo đặc trưng mới

Xây dựng đặc trưng (feature engineering) không chỉ là biến đổi các cột có sẵn. Cột `name` có danh xưng (Mr, Mrs, Miss, Master...), cho biết cả giới tính, tuổi tác lẫn địa vị:

<!-- run -->
```python
df2 = df.copy()
df2["title"] = df2["name"].str.extract(r",\s*([^\.]+)\.")[0].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
df2["title"] = df2["title"].where(df2["title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Hiếm")
df2["family"] = df2["sibsp"] + df2["parch"]
print(df2.groupby("title")["survived"].agg(lambda s: s.astype(int).mean()).round(3).to_dict())

prep2 = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num + ["family"]),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat + ["title"]),
])
model2 = Pipeline([("prep", prep2), ("clf", LogisticRegression(max_iter=1000))])
print("thêm danh xưng và cỡ gia đình:", cross_val_score(model2, df2, y, cv=cv5).mean().round(4))
```

<!-- output -->
```output
{'Hiếm': 0.379, 'Master': 0.508, 'Miss': 0.678, 'Mr': 0.162, 'Mrs': 0.788}
thêm danh xưng và cỡ gia đình: 0.8105
```
<!-- /output -->

Tỉ lệ sống sót khác nhau rất xa theo danh xưng: 16% với "Mr" nhưng 51% với "Master" (bé trai), dù cả hai đều là nam. Đặc trưng giới tính không phân biệt được điều này, còn danh xưng thì có. Thêm danh xưng và cỡ gia đình nâng độ chính xác từ 78.4% lên 81.1%.

## 5. Rò rỉ thông tin

Dữ liệu Titanic có cột `boat` (số hiệu xuồng cứu sinh). Đưa cột này vào làm đặc trưng sẽ cho kết quả "tuyệt vời":

<!-- run -->
```python
df3 = df2.copy(); df3["has_boat"] = df3["boat"].notna().astype(int)
prep3 = ColumnTransformer([("num", "passthrough", ["has_boat"])])
leak = Pipeline([("prep", prep3), ("clf", LogisticRegression())])
print("chỉ dùng 'có lên xuồng cứu sinh hay không':", cross_val_score(leak, df3, y, cv=cv5).mean().round(4))
```

<!-- output -->
```output
chỉ dùng 'có lên xuồng cứu sinh hay không': 0.9755
```
<!-- /output -->

Độ chính xác gần 98% chỉ với một đặc trưng, nhưng mô hình vô dụng: thông tin hành khách có lên xuồng hay không chỉ có *sau* khi biết kết quả. Đây là **rò rỉ thông tin** (data leakage): một đặc trưng chứa (gián tiếp) chính nhãn cần đoán. Khi xây dựng đặc trưng, luôn tự hỏi: lúc cần dự đoán, ta có biết giá trị này chưa?

Một dạng rò rỉ khác khó thấy hơn: tính các thống kê chuẩn hoá trên *toàn bộ* dữ liệu trước khi chia tập huấn luyện và kiểm tra. Đặt bộ chuẩn hoá bên trong `Pipeline` (như ở mục 3) đảm bảo `cross_val_score` tính lại các thống kê trên đúng phần huấn luyện của mỗi lần chia.

## 6. Lựa chọn đặc trưng

<small class="viz-ref">Mục 6.3.2</small>

<!-- run -->
```python
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.datasets import load_digits

Xd, yd = load_digits(return_X_y=True)
print("VarianceThreshold: giữ", VarianceThreshold(1e-3).fit(Xd).get_support().sum(), "trên 64 điểm ảnh")
for k in [8, 16, 32, 64]:
    pipe = make_pipeline(SelectKBest(f_classif, k=k), StandardScaler(), LogisticRegression(max_iter=3000))
    print(f"SelectKBest, k = {k:2d}: độ chính xác kiểm định chéo {cross_val_score(pipe, Xd, yd, cv=5).mean():.4f}")
```

<!-- output -->
```output
VarianceThreshold: giữ 60 trên 64 điểm ảnh
SelectKBest, k =  8: độ chính xác kiểm định chéo 0.7613
SelectKBest, k = 16: độ chính xác kiểm định chéo 0.8715
SelectKBest, k = 32: độ chính xác kiểm định chéo 0.9088
SelectKBest, k = 64: độ chính xác kiểm định chéo 0.9204
```
<!-- /output -->

## Tóm tắt

- Văn bản: `CountVectorizer`/`TfidfVectorizer`; thêm `ngram_range=(1, 2)` để giữ một phần thứ tự từ.
- Dữ liệu hỗn hợp: `ColumnTransformer` gồm xử lý giá trị thiếu, chuẩn hoá cột số, one-hot cột chữ.
- Đặt mọi bước tiền xử lý trong `Pipeline` để chúng chỉ học từ tập huấn luyện.
- Cẩn thận với những đặc trưng "quá tốt": chúng thường là rò rỉ thông tin.
