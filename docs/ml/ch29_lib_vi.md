# Chương 29 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. scikit-learn không có đúng hàm mất mát (29.5) của [chương 29](ml/ch29_vi). Trang so sánh hàm mất mát này với hai hàm gần nó (Crammer–Singer và entropy chéo của hồi quy softmax), điểm qua các cách scikit-learn giải bài toán SVM nhiều lớp, và kiểm tra bằng số nhận xét của mục 29.2.3 rằng $\Delta$ có thể cố định bằng 1.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `scikit-learn`.

| Cách làm | Lớp trong scikit-learn | Hàm mất mát |
| --- | --- | --- |
| SVM đa lớp của sách (Weston–Watkins) | không có; tự viết như [trang Code](ml/ch29_code_vi) | $\sum_{j \neq y}\max(0, 1 - z_y + z_j)$ |
| SVM đa lớp Crammer–Singer | `LinearSVC(multi_class="crammer_singer")` | $\max(0, 1 - z_y + \max_{j \neq y} z_j)$ |
| one-vs-rest | `LinearSVC()`, `SGDClassifier(loss="hinge")` | $C$ bộ phân loại nhị phân |
| one-vs-one | `SVC`, `OneVsOneClassifier(LinearSVC())` | $C(C-1)/2$ bộ phân loại nhị phân |
| hồi quy softmax | `LogisticRegression()` | $-\log \frac{e^{z_y}}{\sum_j e^{z_j}}$ |

## 1. Ba hàm mất mát trên cùng một vector điểm số

<small class="viz-ref">Mục 29.2.1 · công thức (29.2), (29.3) · Hình 29.5</small>

Lấy cột thứ nhất của ví dụ trong Hình 29.5: lớp thật là lớp 1 với điểm số 2, ba lớp còn lại có điểm số 1.5, $-0.2$, 1.7.

<!-- run -->
```python
import numpy as np
import time

def ww(z, y):      # sách, (29.3) với Delta = 1
    m = np.maximum(0, 1 - z[y] + z); m[y] = 0; return m.sum()
def cs(z, y):      # Crammer–Singer
    return max(0, 1 - z[y] + np.max(np.delete(z, y)))
def ce(z, y):      # entropy chéo của hồi quy softmax
    return np.log(np.sum(np.exp(z - z.max()))) - (z[y] - z.max())

for z in [np.array([2, 1.5, -0.2, 1.7]), np.array([5, 1.5, -0.2, 1.7]), np.array([50, 1.5, -0.2, 1.7])]:
    print(f"z = {z}: Weston–Watkins {ww(z, 0):.3f}, Crammer–Singer {cs(z, 0):.3f}, entropy chéo {ce(z, 0):.5f}")
```

<!-- output -->
```output
z = [ 2.   1.5 -0.2  1.7]: Weston–Watkins 1.200, Crammer–Singer 0.700, entropy chéo 0.89941
z = [ 5.   1.5 -0.2  1.7]: Weston–Watkins 0.000, Crammer–Singer 0.000, entropy chéo 0.07008
z = [50.   1.5 -0.2  1.7]: Weston–Watkins 0.000, Crammer–Singer 0.000, entropy chéo 0.00000
```
<!-- /output -->

Với vector điểm số đầu tiên, mất mát của sách là $0.5 + 0 + 0.7 = 1.2$, đúng như cột thứ nhất của ma trận giữa trong Hình 29.5. Crammer–Singer chỉ phạt lớp vi phạm nhiều nhất (0.7). Khi điểm số của lớp thật vượt các lớp khác ít nhất 1 đơn vị, cả hai mất mát bản lề đều bằng 0 và không quan tâm thêm; entropy chéo thì vẫn dương và tiếp tục đẩy $z_y$ lên, dù rất ít. Đây là khác biệt giữa SVM đa lớp và hồi quy softmax mà mục 29.2.1 mô tả.

## 2. Các cách giải trong scikit-learn

<small class="viz-ref">Mục 29.1.1, 29.4</small>

So sánh trên bộ chữ số viết tay $8 \times 8$ của scikit-learn (1797 ảnh, 10 lớp), với SVM đa lớp tự viết (hàm mất mát và gradient của sách, gradient descent theo cả tập).

<!-- run -->
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

def svm_loss_vectorized(W, X, y, reg, delta=1.0):
    N = X.shape[0]; Z = X @ W; i = np.arange(N)
    m = np.maximum(0, Z - Z[i, y][:, None] + delta); m[i, y] = 0
    F = (m > 0).astype(float); F[i, y] = -F.sum(1)
    return m.sum() / N + 0.5 * reg * np.sum(W * W), X.T @ F / N + reg * W

def ww_fit(X, y, reg=1e-3, lr=0.1, iters=3000, delta=1.0, W0=None):
    Xb = np.c_[X, np.ones(len(X))]
    W = np.zeros((Xb.shape[1], 10)) if W0 is None else W0.copy()
    for _ in range(iters):
        W -= lr * svm_loss_vectorized(W, Xb, y, reg, delta)[1]
    return W

models = {
    "Weston–Watkins (sách, tự viết)": None,
    "LinearSVC one-vs-rest": LinearSVC(C=0.1),
    "LinearSVC Crammer–Singer": LinearSVC(C=0.1, multi_class="crammer_singer"),
    "one-vs-one (LinearSVC)": OneVsOneClassifier(LinearSVC(C=0.1)),
    "SGDClassifier(hinge)": SGDClassifier(loss="hinge", alpha=1e-3, random_state=0),
    "hồi quy softmax": LogisticRegression(C=0.1, max_iter=5000),
}
for name, m in models.items():
    t = time.perf_counter()
    if m is None:
        W = ww_fit(Xtr, ytr)
        acc = np.mean(np.argmax(np.c_[Xte, np.ones(len(Xte))] @ W, 1) == yte)
    else:
        acc = m.fit(Xtr, ytr).score(Xte, yte)
    print(f"{name:32s}: {100 * acc:.2f} %, {time.perf_counter() - t:.2f} s")
```

<!-- output -->
```output
Weston–Watkins (sách, tự viết)  : 95.69 %, 0.47 s
LinearSVC one-vs-rest           : 95.27 %, 0.03 s
LinearSVC Crammer–Singer        : 96.52 %, 0.03 s
one-vs-one (LinearSVC)          : 98.05 %, 0.06 s
SGDClassifier(hinge)            : 94.16 %, 0.06 s
hồi quy softmax                 : 96.80 %, 0.02 s
```
<!-- /output -->

Các cách cho độ chính xác gần nhau (tập kiểm tra có 719 ảnh, nên 1% chỉ là 7 ảnh), và chưa cách nào được chọn tham số cẩn thận, nên không nên xếp hạng chúng từ bảng này. Hai điều đáng chú ý: Crammer–Singer, với hàm mất mát gần với của sách nhất, cho kết quả gần với bản tự viết; và one-vs-one, với 45 đường phân chia tuyến tính cho từng cặp lớp, linh hoạt hơn một bộ phân loại đa lớp tuyến tính, nên ở đây cho kết quả tốt nhất.

Bản tự viết chậm hơn nhiều vì dùng gradient descent với số vòng lặp cố định; các lớp của scikit-learn dùng phương pháp toạ độ trên bài toán đối ngẫu (liblinear) hoặc L-BFGS.

## 3. $\Delta$ và $\lambda$ không độc lập

<small class="viz-ref">Mục 29.2.3 · công thức (29.4), (29.5)</small>

Nếu $\mathbf{W}$ là nghiệm với lề $\Delta = 1$ và tham số kiểm soát $\lambda$, thì $10\mathbf{W}$ là nghiệm với $\Delta = 10$ và $\lambda / 10$: thay $\mathbf{W}$ bằng $10\mathbf{W}$, mất mát dữ liệu nhân 10 lần, phần kiểm soát $\frac{\lambda/10}{2}\|10\mathbf{W}\|_F^2$ cũng nhân 10 lần. Hai mô hình cho cùng dự đoán. Vì vậy chỉ cần chọn $\lambda$, còn $\Delta$ cố định bằng 1.

<!-- run -->
```python
W1 = ww_fit(Xtr, ytr, reg=1e-2, lr=0.1, delta=1.0)
W10 = ww_fit(Xtr, ytr, reg=1e-3, lr=1.0, delta=10.0)       # tốc độ học nhân 10 để đi cùng quỹ đạo
Xteb = np.c_[Xte, np.ones(len(Xte))]
print("max |W10 - 10 W1| / max |10 W1| =", np.abs(W10 - 10 * W1).max() / np.abs(10 * W1).max())
print("tỉ lệ dự đoán giống nhau:", np.mean(np.argmax(Xteb @ W1, 1) == np.argmax(Xteb @ W10, 1)))
```

<!-- output -->
```output
max |W10 - 10 W1| / max |10 W1| = 2.6375516026527327e-15
tỉ lệ dự đoán giống nhau: 1.0
```
<!-- /output -->

$\mathbf{W}$ với $\Delta = 10$ đúng bằng 10 lần $\mathbf{W}$ với $\Delta = 1$ (tới sai số làm tròn), và mọi dự đoán trùng nhau. Đổi $\Delta$ chỉ là đổi thang đo của $\mathbf{W}$; chính $\lambda$ quyết định sự cân bằng giữa khớp dữ liệu và kiểm soát.

## Tóm tắt

- Hàm mất mát của sách (Weston–Watkins) cộng mọi vi phạm; Crammer–Singer (có trong `LinearSVC`) chỉ lấy vi phạm lớn nhất; entropy chéo không bao giờ bằng 0.
- Trong scikit-learn, SVM nhiều lớp thường được làm bằng one-vs-rest (`LinearSVC`, `SGDClassifier`) hoặc one-vs-one (`SVC`).
- $\Delta$ có thể cố định bằng 1; chỉ cần chọn $\lambda$ (hoặc $C$) bằng kiểm tra chéo.
