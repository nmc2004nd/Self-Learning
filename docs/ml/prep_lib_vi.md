# Data preprocessing · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm. Trang dùng scikit-learn cho các ý trong bài [Tiền xử lý dữ liệu](ml/prep_vi): so sánh các bộ điền giá trị thiếu, các bộ co giãn, các cách mã hoá biến phân loại và các phương pháp chọn đặc trưng; rồi ghép tất cả bằng `ColumnTransformer` và `Pipeline`, và tinh chỉnh cả các bước tiền xử lý bằng `GridSearchCV`.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `pandas`, `matplotlib`, `scikit-learn` (từ 1.3 trở lên, để có `TargetEncoder`). Bộ dữ liệu Adult được tải từ OpenML ở lần chạy đầu.

| Trong bài | Trong thư viện |
| --- | --- |
| Điền giá trị thiếu | `SimpleImputer(strategy, add_indicator)`, `KNNImputer(n_neighbors)`, `IterativeImputer` (cần `from sklearn.experimental import enable_iterative_imputer`) |
| Mã hoá biến phân loại | `OneHotEncoder(handle_unknown, min_frequency, drop)`, `OrdinalEncoder(categories)`, `TargetEncoder(smooth, cv)` |
| Co giãn | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer` |
| Biến đổi phân phối | `PowerTransformer("yeo-johnson" / "box-cox")`, `QuantileTransformer`, `FunctionTransformer(np.log1p)`, `TransformedTargetRegressor` |
| Tạo đặc trưng | `KBinsDiscretizer`, `SplineTransformer`, `PolynomialFeatures` |
| Chọn đặc trưng | `SelectKBest(f_regression / mutual_info_regression / chi2)`, `RFE`, `RFECV`, `SequentialFeatureSelector`, `SelectFromModel` |
| Ghép các bước | `Pipeline`, `make_pipeline`, `ColumnTransformer`, `make_column_selector`; `set_output(transform="pandas")` |

## 1. So sánh các bộ điền giá trị thiếu

<small class="viz-ref">Mục D.3 · hình D.1</small>

Lấy 3000 khu dân cư California, xoá ngẫu nhiên 20% số ô ở mọi cột đặc trưng (MCAR), rồi đo $R^2$ kiểm định chéo của `HistGradientBoostingRegressor` sau mỗi cách điền. Bộ điền nằm **trong** Pipeline, nên nó chỉ học từ fold huấn luyện. Mô hình này cũng nhận ô trống trực tiếp, nên ta so thêm cách "không điền gì".

<!-- run -->
```python
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold

cal = fetch_california_housing()
rng = np.random.default_rng(0)
idx = rng.choice(len(cal.target), 3000, replace=False)
Xc, yc = cal.data[idx], cal.target[idx]
Xm = Xc.copy()
Xm[rng.random(Xm.shape) < 0.2] = np.nan
print(f"tỉ lệ ô bị xoá: {np.isnan(Xm).mean():.3f}; số dòng còn đủ mọi cột: {int((~np.isnan(Xm).any(1)).sum())} / 3000")

cv = KFold(5, shuffle=True, random_state=0)
def score(X, *steps):
    return cross_val_score(make_pipeline(*steps, HistGradientBoostingRegressor(random_state=0)), X, yc, cv=cv, scoring="r2").mean()

print(f"{'dữ liệu đầy đủ (mốc so sánh)':38s} R² = {score(Xc):.3f}")
rows = [("điền trung bình", SimpleImputer(strategy="mean")),
        ("điền trung vị + cột chỉ báo", SimpleImputer(strategy="median", add_indicator=True)),
        ("KNN, k = 10 (sau khi co giãn)", (StandardScaler(), KNNImputer(n_neighbors=10))),
        ("lặp bằng hồi quy (IterativeImputer)", IterativeImputer(max_iter=10, random_state=0)),
        ("không điền, để mô hình tự xử lý NaN", ())]
for name, imp in rows:
    t = time.perf_counter()
    s = score(Xm, *(imp if isinstance(imp, tuple) else (imp,)))
    print(f"{name:38s} R² = {s:.3f}   ({time.perf_counter() - t:.1f} s)")
```

<!-- output -->
```output
tỉ lệ ô bị xoá: 0.198; số dòng còn đủ mọi cột: 505 / 3000
dữ liệu đầy đủ (mốc so sánh)           R² = 0.789
điền trung bình                        R² = 0.659   (2.1 s)
điền trung vị + cột chỉ báo            R² = 0.668   (2.3 s)
KNN, k = 10 (sau khi co giãn)          R² = 0.645   (4.7 s)
lặp bằng hồi quy (IterativeImputer)    R² = 0.650   (2.9 s)
không điền, để mô hình tự xử lý NaN    R² = 0.668   (2.3 s)
```
<!-- /output -->

- Xoá 20% số ô làm $R^2$ giảm từ 0.79 xuống khoảng 0.65–0.67. Nếu bỏ các dòng có ô thiếu, chỉ còn 505 trong 3000 dòng.
- Với dữ liệu thiếu **hoàn toàn ngẫu nhiên** và một mô hình cây mạnh, các cách điền phức tạp **không** hơn cách đơn giản: KNN (0.645) và IterativeImputer (0.650) còn thua điền trung vị kèm cột chỉ báo (0.668). Để mô hình tự xử lý ô trống cũng cho 0.668.
- Kết quả này không có nghĩa là KNN hay IterativeImputer vô dụng. Chúng hữu ích hơn với mô hình tuyến tính, khi các cột tương quan mạnh với nhau, hoặc khi cần chính các giá trị điền (như Hình D.1). Bài học là: **so sánh bằng kiểm định chéo** thay vì đoán.

## 2. Co giãn và mô hình dựa trên khoảng cách

<small class="viz-ref">Mục D.6 · hình D.4, D.5</small>

Bộ dữ liệu rượu vang (`load_wine`) có 13 đặc trưng với thang đo rất khác nhau: cột proline lên tới hàng nghìn, cột hue chỉ khoảng 1. Ta so kNN, SVM với kernel RBF, hồi quy logistic và rừng ngẫu nhiên, với từng bộ co giãn.

<!-- run -->
```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

Xw, yw = load_wine(return_X_y=True)
print("độ lệch chuẩn nhỏ nhất / lớn nhất của các cột:", round(float(Xw.std(0).min()), 3), "/", round(float(Xw.std(0).max()), 1))
scalers = {"không": None, "MinMax": MinMaxScaler(), "Standard": StandardScaler(), "Robust": RobustScaler(),
           "Quantile": QuantileTransformer(n_quantiles=100), "Yeo–Johnson": PowerTransformer()}
models = {"kNN (k=5)": KNeighborsClassifier(5), "SVM RBF": SVC(), "Logistic": LogisticRegression(max_iter=5000),
          "Rừng ngẫu nhiên": RandomForestClassifier(200, random_state=0)}
cvw = StratifiedKFold(5, shuffle=True, random_state=0)
table = pd.DataFrame({m: {s: cross_val_score(make_pipeline(*([sc] if sc is not None else []), mdl), Xw, yw, cv=cvw).mean()
                          for s, sc in scalers.items()} for m, mdl in models.items()})
print(table.round(3).to_string())
```

<!-- output -->
```output
độ lệch chuẩn nhỏ nhất / lớn nhất của các cột: 0.124 / 314.0
             kNN (k=5)  SVM RBF  Logistic  Rừng ngẫu nhiên
không            0.663    0.657     0.955            0.983
MinMax           0.972    0.989     0.983            0.983
Standard         0.961    0.983     0.983            0.978
Robust           0.955    0.977     0.983            0.978
Quantile         0.955    0.972     0.972            0.972
Yeo–Johnson      0.949    0.983     0.983            0.983
```
<!-- /output -->

- **kNN** và **SVM RBF** dựa trên khoảng cách nên phụ thuộc hoàn toàn vào co giãn: không co giãn, cột proline (độ lệch chuẩn 314) át mọi cột khác, và accuracy chỉ khoảng 0.66. Với bất kỳ bộ co giãn nào, accuracy lên 0.95–0.99.
- **Hồi quy logistic** tăng từ 0.955 lên 0.983. Không co giãn, mô hình vẫn đúng về lý thuyết, nhưng bộ tối ưu hội tụ chậm và chính quy hoá $\ell_2$ mặc định phạt các hệ số không công bằng.
- **Rừng ngẫu nhiên** gần như không đổi (0.972–0.983). Về lý thuyết cây bất biến với mọi phép co giãn đơn điệu; chênh lệch nhỏ đến từ ngưỡng chia đặt ở trung điểm giữa hai giá trị, nên phép biến đổi phi tuyến làm dịch ngưỡng một chút.
- Không bộ co giãn nào thắng ở mọi nơi. Bộ dữ liệu này ít ngoại lai, nên `RobustScaler` không có lợi thế.

## 3. Mã hoá biến phân loại trên dữ liệu Adult

<small class="viz-ref">Mục D.5 · thuật toán D.2 · hình D.3</small>

Bộ dữ liệu Adult (điều tra dân số Mỹ 1994) gồm 48 842 người, dự đoán thu nhập trên 50 nghìn USD/năm. Có 8 cột phân loại, trong đó `native-country` có 41 nhóm, và 3 cột có giá trị thiếu.

<!-- run -->
```python
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

adult = fetch_openml(data_id=1590, as_frame=True)
Xa = adult.data.drop(columns=["fnlwgt", "education-num"])      # education-num trùng thông tin với education
ya = (adult.target == ">50K").astype(int).to_numpy()
cat_cols = Xa.select_dtypes("category").columns.tolist()
num_cols = Xa.select_dtypes("number").columns.tolist()
print("cột phân loại và số nhóm:", {c: Xa[c].nunique() for c in cat_cols})
print("số ô thiếu:", {c: int(v) for c, v in Xa.isna().sum().items() if v})
print(f"tỉ lệ thu nhập > 50K: {ya.mean():.3f}")
```

<!-- output -->
```output
cột phân loại và số nhóm: {'workclass': 8, 'education': 16, 'marital-status': 7, 'occupation': 14, 'relationship': 6, 'race': 5, 'sex': 2, 'native-country': 41}
số ô thiếu: {'workclass': 2799, 'occupation': 2809, 'native-country': 857}
tỉ lệ thu nhập > 50K: 0.239
```
<!-- /output -->

Mỗi cách mã hoá là một nhánh của `ColumnTransformer`: cột phân loại điền "thiếu" rồi mã hoá, cột số điền trung vị rồi co giãn. Mô hình là hồi quy logistic, đo bằng AUC kiểm định chéo ([Đánh giá mô hình](ml/eval_vi)).

<!-- run -->
```python
from sklearn.model_selection import cross_validate

def prep(encoder, scale=True):
    cat = make_pipeline(SimpleImputer(strategy="constant", fill_value="thiếu"), encoder)
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()) if scale else SimpleImputer(strategy="median")
    return ColumnTransformer([("cat", cat, cat_cols), ("num", num, num_cols)])

encoders = {
    "mã thứ tự": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "one-hot": OneHotEncoder(handle_unknown="ignore"),
    "one-hot, gộp nhóm < 50 dòng": OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50),
    "mã hoá theo mục tiêu": TargetEncoder(cv=KFold(5, shuffle=True, random_state=0)),
}
cva = StratifiedKFold(5, shuffle=True, random_state=0)
for name, enc in encoders.items():
    pipe = make_pipeline(prep(enc), LogisticRegression(max_iter=2000))
    r = cross_validate(pipe, Xa, ya, cv=cva, scoring="roc_auc")
    ncol = prep(enc).fit(Xa, ya).transform(Xa.iloc[:5]).shape[1]
    print(f"{name:28s} {ncol:3d} cột   AUC {r['test_score'].mean():.4f}   ({r['fit_time'].mean():.2f} s/fold)")
```

<!-- output -->
```output
mã thứ tự                     12 cột   AUC 0.8156   (3.49 s/fold)
one-hot                      106 cột   AUC 0.9063   (0.44 s/fold)
one-hot, gộp nhóm < 50 dòng   89 cột   AUC 0.9063   (0.50 s/fold)
mã hoá theo mục tiêu          12 cột   AUC 0.9003   (0.94 s/fold)
```
<!-- /output -->

- **Mã thứ tự** tệ nhất với hồi quy logistic (AUC 0.816): mô hình tuyến tính buộc phải coi `education` hay `occupation` là một trục số theo thứ tự bảng chữ cái. Nó cũng chậm nhất, vì bộ tối ưu khó hội tụ.
- **One-hot** đạt 0.906 với 106 cột. Gộp các nhóm có ít hơn 50 dòng bỏ được 17 cột mà không mất gì.
- **Mã hoá theo mục tiêu** dùng 12 cột (mỗi cột phân loại thành một cột) và đạt 0.900, chỉ kém one-hot một chút. Lợi thế của nó lớn hơn khi số nhóm lên tới hàng nghìn.

Với mô hình cây, mã thứ tự thường là đủ. `HistGradientBoostingClassifier` còn xử lý **trực tiếp** cột phân loại và ô thiếu, không cần bước nào:

<!-- run -->
```python
for name, pipe in [
    ("HGB + mã thứ tự", make_pipeline(prep(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), scale=False),
                                      HistGradientBoostingClassifier(random_state=0))),
    ("HGB, phân loại gốc", HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0)),
]:
    r = cross_validate(pipe, Xa, ya, cv=cva, scoring="roc_auc")
    print(f"{name:20s} AUC {r['test_score'].mean():.4f}   ({r['fit_time'].mean():.2f} s/fold)")
```

<!-- output -->
```output
HGB + mã thứ tự      AUC 0.9283   (0.72 s/fold)
HGB, phân loại gốc   AUC 0.9291   (0.85 s/fold)
```
<!-- /output -->

Gradient boosting đạt AUC 0.929, cao hơn mọi phiên bản hồi quy logistic, vì nó tự học được tương tác và quan hệ phi tuyến. Mã thứ tự (0.928) và xử lý phân loại gốc (0.929) gần như bằng nhau, đúng với nhận xét ở mục D.5: cây không bị thứ tự giả làm hại nhiều như mô hình tuyến tính.

## 4. Các phương pháp chọn đặc trưng

<small class="viz-ref">Mục D.9 · thuật toán D.3 · hình D.8</small>

Trên dữ liệu Friedman #1 của hình D.8 (5 đặc trưng có ích $x_1, \dots, x_5$, 5 đặc trưng nhiễu), mỗi phương pháp chọn 5 đặc trưng.

<!-- run -->
```python
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import (SelectKBest, f_regression, mutual_info_regression, RFE,
                                       SequentialFeatureSelector, SelectFromModel)
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from functools import partial

Xf, yf = make_friedman1(n_samples=800, n_features=10, noise=1.0, random_state=0)
Xfs = StandardScaler().fit_transform(Xf)
selectors = {
    "Lọc: F-score": SelectKBest(f_regression, k=5),
    "Lọc: thông tin tương hỗ": SelectKBest(partial(mutual_info_regression, random_state=0), k=5),
    "Bọc: RFE + hồi quy tuyến tính": RFE(LinearRegression(), n_features_to_select=5),
    "Bọc: RFE + rừng ngẫu nhiên": RFE(RandomForestRegressor(100, random_state=0), n_features_to_select=5),
    "Bọc: chọn tiến + kNN": SequentialFeatureSelector(KNeighborsRegressor(10), n_features_to_select=5, cv=5),
    "Nhúng: Lasso": SelectFromModel(LassoCV(cv=5), max_features=5, threshold=-np.inf),
}
for name, sel in selectors.items():
    t = time.perf_counter()
    sup = sel.fit(Xfs, yf).get_support()
    picked = [f"x{j + 1}" for j in np.where(sup)[0]]
    r2 = cross_val_score(RandomForestRegressor(100, random_state=0), Xfs[:, sup], yf, cv=cv).mean()
    print(f"{name:30s} {' '.join(picked):20s} R² rừng ngẫu nhiên {r2:.3f}   ({time.perf_counter() - t:.1f} s)")
```

<!-- output -->
```output
Lọc: F-score                   x1 x2 x4 x5 x6       R² rừng ngẫu nhiên 0.809   (1.3 s)
Lọc: thông tin tương hỗ        x1 x2 x3 x4 x5       R² rừng ngẫu nhiên 0.862   (1.4 s)
Bọc: RFE + hồi quy tuyến tính  x1 x2 x4 x5 x10      R² rừng ngẫu nhiên 0.809   (1.3 s)
Bọc: RFE + rừng ngẫu nhiên     x1 x2 x3 x4 x5       R² rừng ngẫu nhiên 0.862   (3.8 s)
Bọc: chọn tiến + kNN           x1 x2 x3 x4 x5       R² rừng ngẫu nhiên 0.862   (1.9 s)
Nhúng: Lasso                   x1 x2 x4 x5 x10      R² rừng ngẫu nhiên 0.809   (1.4 s)
```
<!-- /output -->

Kết quả lặp lại Hình D.8. Ba phương pháp dựa trên giả định **tuyến tính** (F-score, RFE với hồi quy tuyến tính, Lasso) đều bỏ sót $x_3$ và nhặt một đặc trưng nhiễu thay vào ($x_6$ hoặc $x_{10}$). Ba phương pháp **phi tuyến** (thông tin tương hỗ, RFE với rừng ngẫu nhiên, chọn tiến với kNN) đều tìm đúng $x_1, \dots, x_5$, và rừng ngẫu nhiên trên tập này đạt $R^2$ 0.862 thay vì 0.809.

## 5. ColumnTransformer, Pipeline và tinh chỉnh cả tiền xử lý

<small class="viz-ref">Mục D.10 · thuật toán D.4 · hình D.10</small>

Pipeline hoàn chỉnh cho Adult. Vì mọi bước nằm trong một đối tượng, `GridSearchCV` có thể tinh chỉnh **cả** tham số của các bước tiền xử lý, với cú pháp `<tên bước>__<tham số>`. Mỗi tổ hợp được đánh giá đúng theo thuật toán D.4.

<!-- run -->
```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.25, stratify=ya, random_state=0)
pre = ColumnTransformer([
    ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="thiếu")),
                      ("enc", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False))]),
     make_column_selector(dtype_include="category")),
    ("num", Pipeline([("tf", PowerTransformer()), ("imp", SimpleImputer(strategy="median"))]),
     make_column_selector(dtype_include="number")),
])
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])
grid = {
    "pre__num__tf": [StandardScaler(), PowerTransformer(), QuantileTransformer(n_quantiles=200)],
    "pre__cat__enc__min_frequency": [1, 50, 500],
    "clf__C": [0.1, 1.0],
}
gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
res = pd.DataFrame(gs.cv_results_)
res["tf"] = res["param_pre__num__tf"].map(lambda t: type(t).__name__)
print(res.pivot_table(index="tf", columns="param_pre__cat__enc__min_frequency", values="mean_test_score", aggfunc="max").round(4).to_string())
print("tốt nhất:", {k: (type(v).__name__ if hasattr(v, "fit") else v) for k, v in gs.best_params_.items()})
print(f"AUC trên tập kiểm tra: {roc_auc_score(yte, gs.predict_proba(Xte)[:, 1]):.4f}")
```

<!-- output -->
```output
param_pre__cat__enc__min_frequency     1       50      500
tf                                                        
PowerTransformer                    0.8994  0.8995  0.8926
QuantileTransformer                 0.9016  0.9015  0.8950
StandardScaler                      0.9067  0.9067  0.9006
tốt nhất: {'clf__C': 1.0, 'pre__cat__enc__min_frequency': 1, 'pre__num__tf': 'StandardScaler'}
AUC trên tập kiểm tra: 0.9055
```
<!-- /output -->

- `min_frequency = 500` gộp quá tay, làm mất các nhóm có ích và giảm AUC khoảng 0.006. Giữa 1 và 50 không có khác biệt.
- Ở đây `StandardScaler` thắng `PowerTransformer` và `QuantileTransformer`. Các cột `capital-gain` và `capital-loss` bằng 0 ở hơn 90% số dòng; biến đổi phi tuyến dồn khối số 0 đó lại, còn phép co giãn tuyến tính giữ được thông tin "lãi vốn rất lớn".
- AUC trên tập kiểm tra (0.906) khớp với AUC kiểm định chéo (0.907): vì mọi bước học tham số đều nằm trong Pipeline, ước lượng kiểm định chéo không bị rò rỉ.

Pipeline đã huấn luyện biết tên của từng cột nó tạo ra. `set_output(transform="pandas")` cho kết quả là `DataFrame`, tiện để đọc hệ số của mô hình:

<!-- run -->
```python
best = gs.best_estimator_
names = best.named_steps["pre"].get_feature_names_out()
coef = pd.Series(best.named_steps["clf"].coef_[0], index=names).sort_values()
print(f"tổng số cột sau tiền xử lý: {len(names)}")
print("5 hệ số âm nhất:\n" + coef.head(5).round(3).to_string())
print("5 hệ số dương nhất:\n" + coef.tail(5).round(3).to_string())

one = best.named_steps["pre"].set_output(transform="pandas").transform(Xte.iloc[:1])
print("\nmột dòng kiểm tra sau tiền xử lý (các cột khác 0):")
print(one.loc[:, (one != 0).iloc[0]].T.round(3).to_string())
```

<!-- output -->
```output
tổng số cột sau tiền xử lý: 106
5 hệ số âm nhất:
cat__education_7th-8th            -1.484
cat__occupation_Priv-house-serv   -1.453
cat__native-country_Columbia      -1.434
cat__education_Preschool          -1.226
cat__education_1st-4th            -1.224
5 hệ số dương nhất:
cat__education_Masters                    1.316
cat__marital-status_Married-civ-spouse    1.572
cat__education_Prof-school                1.844
cat__education_Doctorate                  1.885
num__capital-gain                         2.388

một dòng kiểm tra sau tiền xử lý (các cột khác 0):
                                   26433
cat__workclass_Private             1.000
cat__education_HS-grad             1.000
cat__marital-status_Divorced       1.000
cat__occupation_Sales              1.000
cat__relationship_Not-in-family    1.000
cat__race_Amer-Indian-Eskimo       1.000
cat__sex_Female                    1.000
cat__native-country_United-States  1.000
num__age                          -0.702
num__capital-gain                 -0.145
num__capital-loss                 -0.218
num__hours-per-week                0.773
```
<!-- /output -->

Các hệ số dễ đọc nhờ tên cột: học vấn cao (`Doctorate`, `Prof-school`, `Masters`) và đã kết hôn làm tăng xác suất thu nhập cao; học vấn thấp làm giảm. Hệ số lớn nhất là `capital-gain` đã chuẩn hoá. Dòng kiểm tra cho thấy đúng định dạng mà mô hình nhận: mỗi cột phân loại thành một cột 1 duy nhất, các cột số đã được chuẩn hoá theo trung bình và độ lệch chuẩn **của tập huấn luyện**.
