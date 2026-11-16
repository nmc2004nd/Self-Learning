# Chương 11 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang chạy lại các ví dụ scikit-learn của [mục 11.3.2](ml/ch11_vi), so với bản tự viết ở [trang Code](ml/ch11_code_vi), rồi giới thiệu những tham số quan trọng: `alpha`, xác suất đầu ra, và cách ghép naive Bayes với bộ trích chọn đặc trưng văn bản.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `scikit-learn`.

| Trong sách | scikit-learn |
| --- | --- |
| Gaussian naive Bayes (11.2.1) | `GaussianNB` |
| Multinomial naive Bayes (11.2.2) | `MultinomialNB(alpha=1.0)` |
| Bernoulli naive Bayes (11.2.3) | `BernoulliNB(alpha=1.0, binarize=0.0)` |
| Làm mềm Laplace | tham số `alpha` |
| Vector đếm từ (BoW, chương 6) | `CountVectorizer` |

## 1. Ví dụ *Bắc hay Nam* của sách

<small class="viz-ref">Mục 11.3.2</small>

<!-- run -->
```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer

docs = ["hanoi pho chaolong hanoi", "hanoi buncha pho omai", "pho banhgio omai", "saigon hutiu banhbo pho"]
labels = ["B", "B", "B", "N"]
vec = CountVectorizer()                                   # tự dựng từ điển và đếm từ
X = vec.fit_transform(docs)
print("từ điển:", list(vec.get_feature_names_out()))
print("ma trận đếm (thưa):\n", X.toarray())

clf = MultinomialNB(alpha=1).fit(X, labels)
for d in ["hanoi hanoi buncha hutiu", "pho hutiu banhbo"]:
    print(f"'{d}': { {str(c): float(v) for c, v in zip(clf.classes_, clf.predict_proba(vec.transform([d]))[0].round(4))} }")

# Bản BernoulliNB của sách
clfb = BernoulliNB().fit(X, labels)
print("BernoulliNB, d5:", {str(c): float(v) for c, v in zip(clfb.classes_, clfb.predict_proba(vec.transform(["hanoi hanoi buncha hutiu"]))[0].round(4))})
```

<!-- output -->
```output
từ điển: ['banhbo', 'banhgio', 'buncha', 'chaolong', 'hanoi', 'hutiu', 'omai', 'pho', 'saigon']
ma trận đếm (thưa):
 [[0 0 0 1 2 0 0 1 0]
 [0 0 1 0 1 0 1 1 0]
 [0 1 0 0 0 0 1 1 0]
 [1 0 0 0 0 1 0 1 1]]
'hanoi hanoi buncha hutiu': {'B': 0.8955, 'N': 0.1045}
'pho hutiu banhbo': {'B': 0.2918, 'N': 0.7082}
BernoulliNB, d5: {'B': 0.7655, 'N': 0.2345}
```
<!-- /output -->

Thay vì tự liệt kê từ điển như sách, `CountVectorizer` dựng từ điển từ dữ liệu huấn luyện (xếp theo thứ tự chữ cái) và trả về ma trận thưa. Xác suất của `MultinomialNB` trùng với bản tự viết: 0.8955 và 0.2918.

## 2. Lọc email rác bằng scikit-learn

<small class="viz-ref">Mục 11.3.3</small>

Dùng lại dữ liệu đã tải ở [trang Code](ml/ch11_code_vi):

<!-- run -->
```python
from pathlib import Path
from scipy.sparse import coo_matrix
import urllib.request

URL = "https://raw.githubusercontent.com/tiepvupsu/ebookMLCB/master/Chapters/03_SimpleML/32_nbc/python/ex6DataPrepared/"
path = Path.home() / ".cache" / "ebookMLCB" / "ex6DataPrepared"
path.mkdir(parents=True, exist_ok=True)

def read_data(data_fn, label_fn, nwords=2500):
    for fn in (data_fn, label_fn):
        if not (path / fn).exists():
            urllib.request.urlretrieve(URL + fn, path / fn)
    label = np.loadtxt(path / label_fn, dtype=int)
    dat = np.loadtxt(path / data_fn, dtype=int)
    return coo_matrix((dat[:, 2], (dat[:, 0] - 1, dat[:, 1] - 1)), shape=(len(label), nwords)).tocsr(), label

X_te, y_te = read_data("test-features.txt", "test-labels.txt")
print(f"{'tập huấn luyện':>15s} {'Multinomial':>12s} {'Bernoulli':>10s} {'Complement':>11s}")
for size in ["", "-100", "-50"]:
    X_tr, y_tr = read_data(f"train-features{size}.txt", f"train-labels{size}.txt")
    accs = [m.fit(X_tr, y_tr).score(X_te, y_te) for m in (MultinomialNB(), BernoulliNB(binarize=.5), ComplementNB())]
    print(f"{X_tr.shape[0]:15d} " + " ".join(f"{100 * a:10.2f}%" for a in accs))
```

<!-- output -->
```output
 tập huấn luyện  Multinomial  Bernoulli  Complement
            700      98.08%      85.38%      98.08%
            100      97.69%      78.08%      97.69%
             50      97.31%      69.62%      97.31%
```
<!-- /output -->

`MultinomialNB` và `BernoulliNB` cho đúng các con số của sách (98.08%, 97.69%, 97.31%; 69.62% với 50 email). `ComplementNB` (Rennie và cộng sự, 2003) là một biến thể ước lượng tham số từ các lớp *còn lại*, thường ổn định hơn khi các lớp mất cân bằng.

## 3. Chọn `alpha` bằng kiểm định chéo

Mặc định `alpha=1` (làm mềm Laplace). Giá trị này không phải lúc nào cũng tốt nhất; ta chọn nó như mọi siêu tham số khác ([chương 8](ml/ch08_vi)):

<!-- run -->
```python
from sklearn.model_selection import GridSearchCV

X_tr, y_tr = read_data("train-features.txt", "train-labels.txt")
grid = GridSearchCV(MultinomialNB(), {"alpha": [1e-3, 1e-2, 0.1, 0.3, 1, 3, 10]}, cv=5).fit(X_tr, y_tr)
for a, m in zip(grid.cv_results_["param_alpha"], grid.cv_results_["mean_test_score"]):
    print(f"alpha = {a:6}: độ chính xác kiểm định chéo {m:.4f}")
print("alpha tốt nhất:", grid.best_params_["alpha"], "; độ chính xác trên tập kiểm tra:", round(grid.score(X_te, y_te), 4))
```

<!-- output -->
```output
alpha =  0.001: độ chính xác kiểm định chéo 0.9914
alpha =   0.01: độ chính xác kiểm định chéo 0.9871
alpha =    0.1: độ chính xác kiểm định chéo 0.9843
alpha =    0.3: độ chính xác kiểm định chéo 0.9843
alpha =    1.0: độ chính xác kiểm định chéo 0.9800
alpha =    3.0: độ chính xác kiểm định chéo 0.9786
alpha =   10.0: độ chính xác kiểm định chéo 0.9771
alpha tốt nhất: 0.001 ; độ chính xác trên tập kiểm tra: 0.9846
```
<!-- /output -->

Trên dữ liệu này, làm mềm càng ít càng tốt: với 700 email, gần như mọi từ quan trọng đã xuất hiện đủ nhiều, và $\alpha$ lớn chỉ làm "nhoè" các tần suất. Chọn $\alpha = 0.001$ nâng độ chính xác kiểm tra từ 98.08% lên 98.46% (thêm 1 email đúng trên 260). Nhưng $\alpha$ phải dương: $\alpha = 0$ sẽ gặp lại vấn đề ở [trang Code](ml/ch11_code_vi), mục 2.

## 4. Xác suất đầu ra của naive Bayes

Giả thiết các đặc trưng độc lập thường sai trong thực tế: các từ trong một email liên quan chặt chẽ với nhau. Hệ quả là naive Bayes "đếm trùng" bằng chứng, và xác suất nó đưa ra thường quá cực đoan (rất gần 0 hoặc 1), dù nhãn dự đoán vẫn đúng.

<!-- run -->
```python
P = MultinomialNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
print("tỉ lệ email có xác suất < 0.001 hoặc > 0.999:", np.mean((P < 1e-3) | (P > 1 - 1e-3)).round(3))
wrong = (P > 0.5) != y_te
print("các email bị phân loại sai và xác suất mô hình đưa ra:", P[wrong].round(6))
```

<!-- output -->
```output
tỉ lệ email có xác suất < 0.001 hoặc > 0.999: 0.969
các email bị phân loại sai và xác suất mô hình đưa ra: [9.99783e-01 1.00000e+00 5.74784e-01 1.00000e+00 1.00000e-06]
```
<!-- /output -->

Mô hình rất "tự tin" ngay cả khi sai. Nếu cần xác suất đáng tin (chẳng hạn để đặt ngưỡng lọc thư), có thể hiệu chỉnh bằng `sklearn.calibration.CalibratedClassifierCV`, như với hồi quy logistic ở [ví dụ áp dụng chương 14](ml/ch14_app_vi).

## 5. Gaussian naive Bayes và ghép thành pipeline

<!-- run -->
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

Xi, yi = load_iris(return_X_y=True)
print("GaussianNB trên Iris, kiểm định chéo 5 phần:", cross_val_score(GaussianNB(), Xi, yi, cv=5).mean().round(4))

# Với văn bản thô: ghép bộ đếm từ và bộ phân loại thành một mô hình duy nhất
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(docs, labels)
print("pipeline dự đoán:", model.predict(["hanoi hanoi buncha hutiu", "pho hutiu banhbo", "saigon saigon hanoi"]))
```

<!-- output -->
```output
GaussianNB trên Iris, kiểm định chéo 5 phần: 0.9533
pipeline dự đoán: ['B' 'N' 'N']
```
<!-- /output -->

## Tóm tắt

- Ba lớp `GaussianNB`, `MultinomialNB`, `BernoulliNB` ứng với ba biến thể trong sách; `alpha` là tham số làm mềm.
- Với văn bản, ghép `CountVectorizer` (hoặc `TfidfVectorizer`, chương 6) với `MultinomialNB` trong một `Pipeline`.
- Nhãn dự đoán của naive Bayes thường tốt, nhưng xác suất của nó thường quá cực đoan.
