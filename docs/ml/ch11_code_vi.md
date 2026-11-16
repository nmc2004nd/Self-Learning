# Chương 11 · Code từ đầu

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang cài đặt ba biến thể naive Bayes của [chương 11](ml/ch11_vi) chỉ với NumPy, chạy lại ví dụ *Bắc hay Nam* và ví dụ lọc email rác của sách, rồi so với số liệu in trong sách.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy` và `scipy` (chỉ để đọc ma trận thưa). Mục 3 tải dữ liệu email (khoảng 1.6 MB) từ kho mã nguồn của sách ở lần chạy đầu.

Cả ba biến thể có chung một khuôn:

$$
c^* = \operatorname*{argmax}_{c}\ \Big(\log p(c) + \sum_{i=1}^d \log p(x_i \mid c)\Big),
$$

chỉ khác nhau ở cách mô hình hoá $p(x_i \mid c)$. Ta luôn tính tổng các log thay vì tích các xác suất, để tránh tràn số dưới ([trang Thư viện chương 3](math/ch03_lib_vi)).

## 1. Multinomial naive Bayes: ví dụ *Bắc hay Nam*

<small class="viz-ref">Mục 11.2.2, 11.3.1 · Bảng 11.1 · Hình 11.1</small>

Với mỗi lớp $c$, $\lambda_{ci}$ là xác suất từ thứ $i$ xuất hiện, ước lượng với làm mềm Laplace:

$$
\hat\lambda_{ci} = \frac{N_{ci} + \alpha}{N_c + d\alpha},
$$

trong đó $N_{ci}$ là tổng số lần từ $i$ xuất hiện trong các văn bản của lớp $c$, $N_c$ là tổng số từ của lớp $c$, $d$ là số từ trong từ điển.

<!-- run -->
```python
import numpy as np

class MultinomialNBScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        self.classes = np.unique(y)
        counts = np.array([X[y == c].sum(axis=0) for c in self.classes])            # N_ci
        self.log_lam = np.log((counts + self.alpha) / (counts.sum(1, keepdims=True) + X.shape[1] * self.alpha))
        self.log_prior = np.log(np.array([np.mean(y == c) for c in self.classes]))
        return self
    def joint_log(self, X):
        X = np.asarray(X, float)
        L = self.log_lam
        # log p(c) + sum_i x_i log lam_ci; với alpha = 0 có thể có lam = 0 (log = -inf): 0 x (-inf) phải tính là 0
        J = X @ np.where(np.isinf(L), 0, L).T + self.log_prior
        J[(X > 0) @ np.isinf(L).T > 0] = -np.inf
        return J
    def predict_proba(self, X):
        J = self.joint_log(X)
        J -= J.max(axis=1, keepdims=True)                                            # tránh tràn số khi lấy mũ
        P = np.exp(J)
        return P / P.sum(axis=1, keepdims=True)
    def predict(self, X):
        return self.classes[np.argmax(self.joint_log(X), axis=1)]

V = ["hanoi", "pho", "chaolong", "buncha", "omai", "banhgio", "saigon", "hutiu", "banhbo"]
def bow(text):
    return [text.split().count(w) for w in V]

docs = ["hanoi pho chaolong hanoi", "hanoi buncha pho omai", "pho banhgio omai", "saigon hutiu banhbo pho"]
labels = ["B", "B", "B", "N"]
nb = MultinomialNBScratch(alpha=1).fit([bow(d) for d in docs], labels)
print("lambda_B:", np.exp(nb.log_lam[0]).round(4))
print("lambda_N:", np.exp(nb.log_lam[1]).round(4))
for name, d in [("d5", "hanoi hanoi buncha hutiu"), ("d6", "pho hutiu banhbo")]:
    p = nb.predict_proba([bow(d)])[0]
    print(f"{name} = '{d}': p(B | {name}) = {p[0]:.4f}, p(N | {name}) = {p[1]:.4f} -> nhãn {nb.predict([bow(d)])[0]}")
```

<!-- output -->
```output
lambda_B: [0.2  0.2  0.1  0.1  0.15 0.1  0.05 0.05 0.05]
lambda_N: [0.0769 0.1538 0.0769 0.0769 0.0769 0.0769 0.1538 0.1538 0.1538]
d5 = 'hanoi hanoi buncha hutiu': p(B | d5) = 0.8955, p(N | d5) = 0.1045 -> nhãn B
d6 = 'pho hutiu banhbo': p(B | d6) = 0.2918, p(N | d6) = 0.7082 -> nhãn N
```
<!-- /output -->

Kết quả khớp với sách: $p(\text{B} \mid \text{d5}) \approx 0.8955$ và $p(\text{B} \mid \text{d6}) \approx 0.29$. Sách cho các giá trị trung gian $1.5 \times 10^{-4}$ và $1.75 \times 10^{-5}$; đây là $p(c)\prod_i \lambda_{ci}^{x_i}$, ta kiểm tra luôn:

<!-- run -->
```python
print("p(c) prod lambda^x:", np.exp(nb.joint_log([bow("hanoi hanoi buncha hutiu")])[0]))
```

<!-- output -->
```output
p(c) prod lambda^x: [1.50000000e-04 1.75063898e-05]
```
<!-- /output -->

## 2. Vì sao cần làm mềm Laplace

<small class="viz-ref">Mục 11.2.2</small>

Không làm mềm ($\alpha = 0$), "hutiu" chưa từng xuất hiện trong văn bản lớp B nên $\lambda_{\text{B, hutiu}} = 0$. Mọi văn bản chứa "hutiu" đều bị loại khỏi lớp B, bất kể nó có bao nhiêu từ của miền Bắc. Tương tự, "hanoi" chưa từng xuất hiện trong lớp N. Đây là cùng hiện tượng với MLE trên ảnh MNIST ở [ví dụ áp dụng chương 4](math/ch04_app_vi); làm mềm Laplace chính là MAP với tiên nghiệm Dirichlet.

<!-- run -->
```python
with np.errstate(divide="ignore"):
    nb0 = MultinomialNBScratch(alpha=0).fit([bow(d) for d in docs], labels)
    print("alpha = 0, lambda_B:", np.exp(nb0.log_lam[0]).round(3))
    for name, d in [("d7", "hanoi hanoi pho"), ("d8", "pho hutiu"), ("d5", "hanoi hanoi buncha hutiu")]:
        J = nb0.joint_log([bow(d)])[0]
        print(f"alpha = 0, {name} = '{d}': log p(B, {name}) = {J[0]:.2f}, log p(N, {name}) = {J[1]:.2f}")
```

<!-- output -->
```output
alpha = 0, lambda_B: [0.273 0.273 0.091 0.091 0.182 0.091 0.    0.    0.   ]
alpha = 0, d7 = 'hanoi hanoi pho': log p(B, d7) = -4.19, log p(N, d7) = -inf
alpha = 0, d8 = 'pho hutiu': log p(B, d8) = -inf, log p(N, d8) = -4.16
alpha = 0, d5 = 'hanoi hanoi buncha hutiu': log p(B, d5) = -inf, log p(N, d5) = -inf
```
<!-- /output -->

Văn bản d5 chứa cả "hanoi" (chưa từng có trong lớp N) lẫn "hutiu" (chưa từng có trong lớp B): cả hai lớp đều có xác suất bằng 0, và mô hình không đưa ra được quyết định nào. Với $\alpha = 1$, d5 được xếp vào lớp B với xác suất 0.8955 như ở mục 1.

## 3. Lọc email rác: chạy lại thí nghiệm của sách

<small class="viz-ref">Mục 11.3.3</small>

Dữ liệu gồm 700 email huấn luyện và 260 email kiểm tra, mỗi email là một vector đếm số lần xuất hiện của 2500 từ (dữ liệu từ bài tập 6 của khoá Machine Learning của Andrew Ng, sách đã đóng gói sẵn). Hàm `read_data` giữ như sách; chỉ thêm phần tải tệp.

<!-- run -->
```python
import urllib.request
from pathlib import Path
from scipy.sparse import coo_matrix

URL = "https://raw.githubusercontent.com/tiepvupsu/ebookMLCB/master/Chapters/03_SimpleML/32_nbc/python/ex6DataPrepared/"
path = Path.home() / ".cache" / "ebookMLCB" / "ex6DataPrepared"
path.mkdir(parents=True, exist_ok=True)
for fn in ["train-features.txt", "train-labels.txt", "train-features-100.txt", "train-labels-100.txt",
           "train-features-50.txt", "train-labels-50.txt", "test-features.txt", "test-labels.txt"]:
    if not (path / fn).exists():
        urllib.request.urlretrieve(URL + fn, path / fn)

nwords = 2500
def read_data(data_fn, label_fn):
    with open(path / label_fn) as f:
        label = [int(x.strip()) for x in f.readlines()]
    with open(path / data_fn) as f:
        content = [x.strip() for x in f.readlines()]
    dat = np.zeros((len(content), 3), dtype=int)
    for i, line in enumerate(content):
        a = line.split(' ')
        dat[i, :] = np.array([int(a[0]), int(a[1]), int(a[2])])
    data = coo_matrix((dat[:, 2], (dat[:, 0] - 1, dat[:, 1] - 1)), shape=(len(label), nwords))
    return data, np.array(label)

X_te, y_te = read_data("test-features.txt", "test-labels.txt")
X_te = X_te.toarray()
for size in ["", "-100", "-50"]:
    X_tr, y_tr = read_data(f"train-features{size}.txt", f"train-labels{size}.txt")
    clf = MultinomialNBScratch(alpha=1).fit(X_tr.toarray(), y_tr)
    acc = np.mean(clf.predict(X_te) == y_te)
    print(f"Training size = {X_tr.shape[0]}, accuracy = {100 * acc:.2f}%")
```

<!-- output -->
```output
Training size = 700, accuracy = 98.08%
Training size = 100, accuracy = 97.69%
Training size = 50, accuracy = 97.31%
```
<!-- /output -->

Ba con số trùng khớp với sách: 98.08%, 97.69%, 97.31%. Mô hình chỉ có $2 \times 2500$ tham số, được "huấn luyện" bằng một lần đếm, nên vẫn tốt khi chỉ có 50 email.

## 4. Bernoulli naive Bayes

<small class="viz-ref">Mục 11.2.3</small>

Bernoulli naive Bayes chỉ quan tâm một từ *có* hay *không* xuất hiện, và (khác với multinomial) phạt cả việc một từ đặc trưng của lớp *không* xuất hiện:

$$
p(\mathbf{x} \mid c) = \prod_{i=1}^d \lambda_{ci}^{x_i}(1 - \lambda_{ci})^{1 - x_i}.
$$

<!-- run -->
```python
class BernoulliNBScratch:
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha, self.binarize = alpha, binarize
    def fit(self, X, y):
        X = (np.asarray(X) > self.binarize).astype(float); self.classes = np.unique(y)
        self.lam = np.array([(X[y == c].sum(0) + self.alpha) / (np.sum(y == c) + 2 * self.alpha) for c in self.classes])
        self.log_prior = np.log([np.mean(y == c) for c in self.classes])
        return self
    def predict(self, X):
        X = (np.asarray(X) > self.binarize).astype(float)
        J = X @ np.log(self.lam).T + (1 - X) @ np.log(1 - self.lam).T + self.log_prior
        return self.classes[np.argmax(J, axis=1)]

X50, y50 = read_data("train-features-50.txt", "train-labels-50.txt")
bnb = BernoulliNBScratch(alpha=1, binarize=0.5).fit(X50.toarray(), y50)
print(f"Bernoulli, Training size = 50, accuracy = {100 * np.mean(bnb.predict(X_te) == y_te):.2f}%")
X700, y700 = read_data("train-features.txt", "train-labels.txt")
bnb7 = BernoulliNBScratch(alpha=1, binarize=0.5).fit(X700.toarray(), y700)
print(f"Bernoulli, Training size = 700, accuracy = {100 * np.mean(bnb7.predict(X_te) == y_te):.2f}%")
```

<!-- output -->
```output
Bernoulli, Training size = 50, accuracy = 69.62%
Bernoulli, Training size = 700, accuracy = 85.38%
```
<!-- /output -->

Với 50 email, kết quả trùng với con số 69.62% của sách. Với 700 email, Bernoulli tốt hơn (85.38%) nhưng vẫn kém xa multinomial (98.08%). Sách chỉ nhận xét rằng `MultinomialNB` "hoạt động hiệu quả hơn"; ta thử tìm hiểu tại sao:

<!-- run -->
```python
Xb = (X_te > 0.5)
print("số từ khác nhau trung bình trong một email kiểm tra:", Xb.sum(1).mean().round(1), "trên", nwords)
pred = bnb.predict(X_te)
print("tỉ lệ dự đoán là thư rác:", pred.mean().round(3), "; tỉ lệ thật:", y_te.mean().round(3))
lam = bnb.lam
print("số từ có lambda_spam > lambda_ham:", int(np.sum(lam[1] > lam[0])), "; ngược lại:", int(np.sum(lam[0] > lam[1])))
```

<!-- output -->
```output
số từ khác nhau trung bình trong một email kiểm tra: 107.6 trên 2500
tỉ lệ dự đoán là thư rác: 0.235 ; tỉ lệ thật: 0.5
số từ có lambda_spam > lambda_ham: 985 ; ngược lại: 377
```
<!-- /output -->

Với chỉ 25 email mỗi lớp, rất nhiều từ chưa xuất hiện ở lớp này nhưng có ở lớp kia. Bernoulli naive Bayes tính cả $(1 - \lambda_{ci})$ cho **mọi** từ không xuất hiện trong email (khoảng 2400 trên 2500 từ), nên những chênh lệch nhỏ của 2400 thừa số này lấn át thông tin từ vài chục từ thật sự có trong email. Kết quả là mô hình lệch hẳn về lớp email thường: chỉ 23.5% email kiểm tra bị đoán là thư rác, trong khi tỉ lệ thật là 50%. Multinomial naive Bayes chỉ dùng những từ thật sự có trong email (với số lần xuất hiện), nên không gặp vấn đề này.

## 5. Gaussian naive Bayes

<small class="viz-ref">Mục 11.2.1</small>

Với đặc trưng liên tục, mỗi $p(x_i \mid c)$ là một phân phối chuẩn một chiều với kỳ vọng và phương sai ước lượng bằng MLE ([chương 4](math/ch04_vi)) trên dữ liệu của lớp $c$:

<!-- run -->
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class GaussianNBScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mu = np.array([X[y == c].mean(0) for c in self.classes])
        self.var = np.array([X[y == c].var(0) for c in self.classes]) + 1e-9 * X.var(0).max()
        self.log_prior = np.log([np.mean(y == c) for c in self.classes])
        return self
    def predict(self, X):
        J = -0.5 * (np.log(2 * np.pi * self.var)[None] + (X[:, None, :] - self.mu[None]) ** 2 / self.var[None]).sum(-1) + self.log_prior
        return self.classes[np.argmax(J, axis=1)]

Xi, yi = load_iris(return_X_y=True)
Xa, Xb_, ya, yb = train_test_split(Xi, yi, test_size=0.4, random_state=0, stratify=yi)
g = GaussianNBScratch().fit(Xa, ya)
print("Iris, độ chính xác kiểm tra:", np.mean(g.predict(Xb_) == yb).round(4))
print("kỳ vọng của lớp setosa:", g.mu[0].round(3), "; phương sai:", g.var[0].round(4))
```

<!-- output -->
```output
Iris, độ chính xác kiểm tra: 0.9667
kỳ vọng của lớp setosa: [5.017 3.42  1.433 0.243] ; phương sai: [0.1667 0.1869 0.0282 0.0105]
```
<!-- /output -->

## Tóm tắt

| Biến thể | Mô hình cho $p(x_i \mid c)$ | Dùng khi | Kết quả trên trang này |
| --- | --- | --- | --- |
| Multinomial | tần suất từ, có làm mềm Laplace | đếm từ trong văn bản | khớp sách: 0.8955, 98.08%, 97.69%, 97.31% |
| Bernoulli | từ có hay không xuất hiện | vector nhị phân | khớp sách: 69.62% với 50 email |
| Gaussian | phân phối chuẩn một chiều | đặc trưng liên tục | Iris |
