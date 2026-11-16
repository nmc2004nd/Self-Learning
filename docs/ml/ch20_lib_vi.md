# Chương 20 · Dùng thư viện

> Ghi chú của người biên soạn: trang này do người biên soạn viết thêm, không có trong sách gốc. Trang so sánh các hàm tính SVD của NumPy, SciPy và scikit-learn cho [chương 20](ml/ch20_vi): SVD đầy đủ, SVD giản lược, chỉ tính $k$ thành phần đầu cho ma trận thưa, và SVD ngẫu nhiên cho ma trận lớn. Cuối trang là một ứng dụng của SVD cắt ngọn cho văn bản.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `numpy`, `scipy`, `scikit-learn`.

| Trong sách | Hàm |
| --- | --- |
| SVD đầy đủ (20.2.1) | `np.linalg.svd(A)` |
| SVD giản lược (20.2.4) | `np.linalg.svd(A, full_matrices=False)` |
| SVD cắt ngọn (20.2.5), ma trận thưa | `scipy.sparse.linalg.svds(A, k)` |
| SVD cắt ngọn, ma trận lớn | `sklearn.utils.extmath.randomized_svd`, `sklearn.decomposition.TruncatedSVD` |

## 1. Đầy đủ hay giản lược

<small class="viz-ref">Mục 20.2.1, 20.2.4</small>

<!-- run -->
```python
import numpy as np
from time import perf_counter

rng = np.random.default_rng(0)
A = rng.standard_normal((1000, 50))
U, s, Vt = np.linalg.svd(A)                         # đầy đủ: U là 1000 x 1000
Ur, sr, Vtr = np.linalg.svd(A, full_matrices=False) # giản lược: U là 1000 x 50
print("đầy đủ:   U", U.shape, " s", s.shape, " Vt", Vt.shape)
print("giản lược: U", Ur.shape, " s", sr.shape, " Vt", Vtr.shape)
print("hai cách cho cùng A:", np.allclose(Ur * sr @ Vtr, A))
print(f"bộ nhớ của U: {U.nbytes / 1e6:.1f} MB so với {Ur.nbytes / 1e6:.2f} MB")
```

<!-- output -->
```output
đầy đủ:   U (1000, 1000)  s (50,)  Vt (50, 50)
giản lược: U (1000, 50)  s (50,)  Vt (50, 50)
hai cách cho cùng A: True
bộ nhớ của U: 8.0 MB so với 0.40 MB
```
<!-- /output -->

Với ma trận "cao" ($m \gg n$), $m - n$ cột cuối của $\mathbf{U}$ trong SVD đầy đủ chỉ nhân với các số 0 của $\mathbf{\Sigma}$. Gần như mọi ứng dụng chỉ cần SVD giản lược.

## 2. Chỉ tính $k$ thành phần đầu

<small class="viz-ref">Mục 20.2.5</small>

<!-- run -->
```python
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

B = rng.standard_normal((3000, 40)) @ rng.standard_normal((40, 2000)) + 0.1 * rng.standard_normal((3000, 2000))   # gần hạng 40
k = 10
s_full = np.linalg.svd(B, compute_uv=False)[:k]
_, s_svds, _ = svds(B, k=k)                          # trả về theo thứ tự TĂNG dần
_, s_rand, _ = randomized_svd(B, n_components=k, random_state=0)
print("np.linalg.svd:   ", s_full.round(3))
print("svds (đảo thứ tự):", s_svds[::-1].round(3))
print("randomized_svd:  ", s_rand.round(3))

M = sp.random(20000, 5000, density=0.001, random_state=0, format="csr")   # ma trận thưa 20000 x 5000
_, s_sp, _ = svds(M, k=5)
print(f"\nma trận thưa {M.shape}, {M.nnz:,} phần tử khác 0 ({M.data.nbytes / 1e6:.1f} MB); dạng đầy đủ sẽ cần {M.shape[0] * M.shape[1] * 8 / 1e9:.1f} GB")
print("5 giá trị suy biến lớn nhất:", np.sort(s_sp)[::-1].round(4))
```

<!-- output -->
```output
np.linalg.svd:    [2912.841 2866.656 2816.76  2779.052 2752.255 2725.009 2695.596 2693.023
 2656.898 2627.216]
svds (đảo thứ tự): [2912.841 2866.656 2816.76  2779.052 2752.255 2725.009 2695.596 2693.023
 2656.898 2627.216]
randomized_svd:   [2911.773 2864.907 2808.826 2773.175 2746.243 2713.993 2687.525 2685.861
 2648.261 2617.609]

ma trận thưa (20000, 5000), 100,000 phần tử khác 0 (0.8 MB); dạng đầy đủ sẽ cần 0.8 GB
5 giá trị suy biến lớn nhất: [5.9341 4.3589 4.2862 4.2741 4.2558]
```
<!-- /output -->

`svds` cho kết quả chính xác như `np.linalg.svd`. `randomized_svd` chỉ là xấp xỉ: các giá trị của nó hơi nhỏ hơn (sai lệch dưới 0.5%), và có thể làm chính xác hơn bằng cách tăng `n_iter` hoặc `n_oversamples`, đổi lại thời gian tính. Cả hai không cần dựng ma trận đầy đủ. Với ma trận thưa (như ma trận tiện ích trong hệ thống gợi ý ở [chương 19](ml/ch19_vi), hay ma trận từ–văn bản), đây là cách duy nhất khả thi. Lưu ý: `svds` trả các giá trị suy biến theo thứ tự tăng dần, ngược với `np.linalg.svd`.

## 3. Ứng dụng: phân tích ngữ nghĩa ẩn

SVD cắt ngọn của ma trận văn bản–từ (TF-IDF, [chương 6](ml/ch06_vi)) biểu diễn mỗi văn bản và mỗi từ bằng một vector ngắn, gọi là *phân tích ngữ nghĩa ẩn* (latent semantic analysis). Các từ hay xuất hiện cùng nhau có vector gần nhau, kể cả khi chúng hiếm khi xuất hiện trong cùng một câu.

<!-- run -->
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

cats = ["sci.space", "sci.med", "rec.autos", "comp.graphics"]
news = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
tfidf = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.5)
X = tfidf.fit_transform(news.data)
svd = TruncatedSVD(n_components=100, random_state=0).fit(X)
words = np.array(tfidf.get_feature_names_out())
print("ma trận văn bản–từ:", X.shape, "-> mỗi văn bản còn", svd.n_components, "chiều")

W = svd.components_.T * svd.singular_values_          # vector của từ trong không gian ẩn
W = W / np.linalg.norm(W, axis=1, keepdims=True)
for w in ["orbit", "doctor", "engine", "image"]:
    sims = W @ W[list(words).index(w)]
    print(f"gần '{w}' nhất: {', '.join(words[np.argsort(-sims)[1:8]])}")
```

<!-- output -->
```output
ma trận văn bản–từ: (2365, 5668) -> mỗi văn bản còn 100 chiều
gần 'orbit' nhất: circular, hiten, polar, inclination, flyby, gravitational, maneuvering
gần 'doctor' nhất: clinic, upset, emergency, pissed, pm, delayed, realize
gần 'engine' nhất: rebuilt, diesel, compartment, ins, household, 350, cam
gần 'image' nhất: processing, v1, buffer, intro, performs, printed, spectra
```
<!-- /output -->

Phần lớn các từ lân cận hợp lý ("orbit" gần "inclination", "flyby"; "engine" gần "diesel", "cam"; "image" gần "processing"), xen lẫn vài từ nhiễu. Không có quy tắc ngữ pháp nào được dùng: toàn bộ thông tin đến từ việc các từ hay xuất hiện trong cùng văn bản.

## Tóm tắt

- Dùng `full_matrices=False` trừ khi thật sự cần phần bù trực giao của $\mathbf{U}$.
- Chỉ cần vài thành phần đầu: `svds` (ma trận thưa) hoặc `randomized_svd`/`TruncatedSVD` (ma trận lớn).
- `TruncatedSVD` làm việc trực tiếp với ma trận thưa, không trừ trung bình (khác với PCA ở [chương 21](ml/ch21_vi)), nên không phá vỡ tính thưa.
