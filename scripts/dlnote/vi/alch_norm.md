# Mẹo huấn luyện: chuẩn hoá dữ liệu

Trang [Code](dlnote/alchemy_code_vi) đi kèm (mục 2) chạy lại các ví dụ của bài, kiểm tra z-score có biến dữ liệu thành phân phối chuẩn không, đếm số bước gradient descent trước và sau chuẩn hoá, và xem `ToTensor`, `Normalize` của torchvision thực sự làm gì.

<!-- theory: ml/prep_vi#D.6 ; qa/ch03_vi#3.6.3 -->

## Lời nói đầu

Quy trình (`pipeline`) của một bài toán học máy thường gồm các bước như hình dưới.

![Pipeline của bài toán học máy](../images/data_normalization/ml_pipeline.png)

Hình 0.1: Các bước của một bài toán học máy: thu thập dữ liệu, tiền xử lý, khám phá và trực quan hoá dữ liệu, xây dựng và huấn luyện mô hình, đánh giá và triển khai.

Trong công nghiệp, bước **tiền xử lý dữ liệu** đóng vai trò quan trọng trong việc nâng độ chính xác của mô hình. Với bài toán học máy nói chung, tiền xử lý dữ liệu theo nghĩa rộng gồm bốn giai đoạn (**bài toán thị giác thường chỉ cần `Data Transformation`**): làm sạch dữ liệu (Data Cleaning), tích hợp dữ liệu (Data Integration), biến đổi dữ liệu (Data Transformation) và rút gọn dữ liệu (Data Reduction).

![Bốn giai đoạn tiền xử lý dữ liệu](../images/data_normalization/steps_in_data_preprocessing.png)

Hình 0.2: Bốn giai đoạn của tiền xử lý dữ liệu.

1. `Data Cleaning`: làm sạch dữ liệu bằng cách điền giá trị thiếu, làm trơn dữ liệu nhiễu, giải quyết chỗ không nhất quán và loại bỏ ngoại lai.

2. `Data Integration`: gộp dữ liệu từ nhiều nguồn vào một kho lưu trữ lớn hơn, như kho dữ liệu (data warehouse). Ví dụ gộp ảnh từ nhiều cơ sở y tế thành một cơ sở dữ liệu lớn hơn.

3. `Data Transformation`: sau khi làm sạch, ta dùng các **chiến lược biến đổi dữ liệu** sau để đổi giá trị, cấu trúc hay định dạng dữ liệu.

   - `Generalization`: dùng phân cấp khái niệm để biến dữ liệu mức thấp, chi tiết thành thông tin mức cao hơn. Ví dụ đổi địa chỉ thành phố thô thành thông tin cấp quốc gia.
   - `Normalization`: **co giãn các thuộc tính dữ liệu để chúng rơi vào một khoảng nhỏ hơn (biến đổi về một khoảng cố định)**. Các cách thường gặp:
     - `Min-max normalization`: ánh xạ dữ liệu vào khoảng $[0,1]$.
     - `Z-Score normalization`: biến dữ liệu của mỗi đặc trưng thành dữ liệu có trung bình 0, độ lệch chuẩn 1.
     - `Decimal scaling normalization`: dời dấu thập phân của mọi giá trị (chia cho $10^k$) để đưa dữ liệu về một khoảng nhất định, thường là $[-1, 1]$.

4. `Data Reduction`: tập dữ liệu trong kho có thể quá lớn để các thuật toán phân tích, khai phá xử lý được. Một cách giải quyết là lấy một biểu diễn rút gọn của tập dữ liệu, nhỏ hơn nhiều nhưng cho kết quả phân tích cùng chất lượng. Các chiến lược rút gọn thường gặp:
   - `Data cube aggregation`: tổng hợp khối dữ liệu là kỹ thuật phân tích dữ liệu nhiều chiều, gồm cắt lát (Slice), cắt khối (Dice), xoay (Pivot), đi sâu (Drill), dùng chủ yếu để tổng hợp và phân tích dữ liệu nhiều chiều trong kho dữ liệu hoặc OLAP (xử lý phân tích trực tuyến).
   - `Dimensionality reduction`: **giảm chiều để trích xuất đặc trưng**. Số chiều của tập dữ liệu là số thuộc tính (đặc trưng). Kỹ thuật này nhằm giảm số đặc trưng thừa mà thuật toán học máy phải xét; có thể làm bằng phân tích thành phần chính (`PCA`)…
   - `Data compression`: dùng kỹ thuật mã hoá để giảm đáng kể kích thước dữ liệu.
   - `Discretization`: rời rạc hoá, **chia thuộc tính liên tục thành các khoảng**. Ví dụ thuộc tính tuổi có thể chia thành dưới 18, 18–44, 44–60, trên 60…

> **Biên tập:** Bản gốc nêu lý do rời rạc hoá là "đặc trưng liên tục thường ít liên quan tới biến mục tiêu"; câu này không đúng nói chung và đã bỏ. Rời rạc hoá thường dùng để mô hình tuyến tính bắt được quan hệ phi tuyến, giảm ảnh hưởng của ngoại lai hoặc cho dễ diễn giải, và thường *làm mất* thông tin; xem [bài Tiền xử lý dữ liệu](ml/prep_vi) của khoá Học máy.

Với bài toán thị giác máy tính, trước khi huấn luyện mô hình `CNN`, **chuẩn hoá** (`normalization`) dữ liệu đầu vào là bước **tiền xử lý** (`data preprocessing`) phổ biến nhất.

## 1 Tổng quan về Normalization

> Bản gốc giữ nguyên từ `Normalization` vì cách dịch sang tiếng Trung (归一化, "quy về một") còn chưa thống nhất. Bản dịch này dùng "chuẩn hoá" cho `normalization` nói chung, và gọi riêng tên từng cách (z-score, min-max).

### 1.1 Định nghĩa Normalization

`Normalization` là phép co giãn các thuộc tính dữ liệu để chúng rơi vào một khoảng nhỏ hơn (biến đổi về một khoảng cố định), như $[-1,1]$ hay $[0,1]$; hiểu đơn giản là **co giãn đặc trưng** (feature scaling). Nhiều thuật toán học máy được lợi từ `Normalization`, chẳng hạn:

- thường có ích cho các thuật toán phân loại;
- có ích cho các thuật toán tối ưu nằm ở lõi của các thuật toán học máy như gradient descent;
- có ích cho các thuật toán lấy tổng có trọng số của đầu vào (như hồi quy, mạng neuron) và các thuật toán dùng khoảng cách (như K láng giềng gần nhất).

### 1.2 Khi nào cần Normalization

Khi **dữ liệu có các thang đo (khoảng giá trị) khác nhau** (`different scale`), thường cần `normalization`.

Dữ liệu có thang khác nhau khiến một thuộc tính quan trọng (có thang nhỏ) bị lấn át, vì các thuộc tính khác có khoảng giá trị lớn hơn; hiểu đơn giản là **thuộc tính có thang lớn được mô hình ưu tiên hơn**. Ví dụ ở hình dưới.

![Hai thuộc tính có thang khác nhau](../images/data_normalization/diffenent_scale.png)

Hình 1.1: Lương (`Salary`) và số năm kinh nghiệm (`Year_of_experience`) có thang rất khác nhau, nên trong mô hình thuộc tính lương có thể được ưu tiên hơn số năm kinh nghiệm.

Tóm lại, khi dữ liệu có nhiều thuộc tính với thang khác nhau, mô hình có thể làm việc kém khi khai phá dữ liệu, nên cần `normalization` để đưa mọi thuộc tính về cùng một thang.

### 1.3 Vì sao cần Normalization

1. **Giảm độ nhạy của quá trình huấn luyện với thang của đặc trưng.** Khi khoảng giá trị của các đặc trưng đầu vào gần nhau (và gần với khoảng của nhãn đầu ra), các đường đồng mức của hàm mất mát gần tròn; không chuẩn hoá thì các đường đồng mức là elip dẹt và gradient descent đi đường zíc zắc.

<div align=center><img src="../img/code/alchemy_code/12_1.png" alt="Gradient descent trước và sau chuẩn hoá"/></div>

Hình 1.2: Hồi quy tuyến tính giá nhà theo diện tích (30–200 m²) và khoảng cách tới trung tâm (1–20 km). Trái: chỉ trừ trung bình, đường đồng mức là elip rất dẹt, gradient descent với learning rate lớn nhất còn ổn định cần 782 bước. Phải: sau z-score, đường đồng mức gần tròn, chỉ cần 4 bước (trang Code; hai trục cùng thang).

> **Biên tập:** Bản gốc mở đầu ý này bằng câu "giá trị các đặc trưng của mẫu phải phù hợp phân phối xác suất, tức nằm trong $[0,1]$"; câu này không có nghĩa rõ ràng (đặc trưng không phải xác suất) và đã bỏ. Hình đường đồng mức vẽ tay chữ Trung của bản gốc được thay bằng hình vẽ từ phép tính thật. Số bước gradient descent tỉ lệ với **số điều kiện** của ma trận Hessian ($\lambda_{max}/\lambda_{min}$): 88 trước, 1,1 sau chuẩn hoá.

2. Mạng neuron được thiết kế với giả định đầu vào (và các tín hiệu bên trong) có **trung bình 0, độ lệch chuẩn 1**, từ cách khởi tạo trọng số, chọn hàm kích hoạt tới thiết kế thuật toán tối ưu.

3. **Tránh các vấn đề số học không cần thiết.** Vùng phi tuyến của sigmoid/tanh vào khoảng $[-1{,}7;\ 1{,}7]$. Muốn hàm kích hoạt của neuron có tác dụng, đầu ra của phép tính tuyến tính nên có độ lớn cỡ 1. Nếu đầu vào lớn thì trọng số phải nhỏ; một số lớn nhân một số nhỏ dễ sinh lỗi số học.

4. **Cập nhật gradient.** Nếu đầu ra có độ lớn quá lớn, hàm mất mát cũng lớn, gradient lan truyền ngược cũng lớn, gây khó khăn số học cho việc cập nhật.

5. **Learning rate.** Khi các đặc trưng có khoảng giá trị khác nhau, learning rate phù hợp cho từng trọng số $w_1$, $w_2$ cũng khác nhau (gradient càng lớn thì learning rate càng phải nhỏ). Vì vậy learning rate (ban đầu) phải chọn theo khoảng của đầu vào; **chuẩn hoá thẳng dữ liệu thì không phải chỉnh learning rate theo khoảng dữ liệu nữa**.

### 1.4 Các cách chuẩn hoá dữ liệu thường dùng

1. **Z-score normalization**

Còn gọi là zero-mean normalization hay standardization (chuẩn hoá theo trung bình và độ lệch chuẩn): co giãn đặc trưng để có trung bình 0, phương sai 1. Công thức:

$$
{x}' = \frac{x-\mu}{\sigma}
$$

trong đó:

- $x$: giá trị gốc;
- $\mu$: trung bình của biến $x$, $\mu =\frac{1}{N}\sum_{i=1}^{N} x_i$;
- $\sigma$: độ lệch chuẩn của biến (độ lệch chuẩn tổng thể $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N}(x_i - \mu)^2}$);
- ${x}'$: giá trị mới sau khi co giãn.

Nếu dữ liệu vốn có phân phối chuẩn, sau z-score đường cong phân phối biến đổi như hình dưới.

![Z-score biến N(μ, σ²) thành N(0, 1)](../images/data_normalization/z-score-1.png)

Hình 1.3: Với dữ liệu phân phối chuẩn $\mathcal N(\mu,\sigma^2)$, z-score cho phân phối chuẩn tắc $\mathcal N(0,1)$.

Vị trí của trung bình và các mức độ lệch chuẩn trên đường cong phân phối chuẩn xem hình dưới.

![Trung bình và các mức độ lệch chuẩn của phân phối chuẩn](../images/data_normalization/Normal_Distribution.png)

Hình 1.4: Trung bình (Mean) và các mức 1, 2, 3 độ lệch chuẩn (std. dev) trên đường cong phân phối chuẩn.

2. **Min-max normalization**

Phép biến đổi tuyến tính đưa khoảng giá trị của dữ liệu về $[0,1]$:

$$
{x}' = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

trong đó $\max(x)$ là giá trị lớn nhất, $\min(x)$ là giá trị nhỏ nhất của biến.

> **Biên tập:** Bản gốc mô tả z-score là biến dữ liệu "thành phân phối chuẩn" ("最后为正态分布", và ở mục 1.5 "phân phối gốc biến thành phân phối chuẩn tắc"). Z-score chỉ là phép co giãn và tịnh tiến tuyến tính: nó đưa trung bình về 0 và độ lệch chuẩn về 1 nhưng **không đổi hình dạng** phân phối. Dữ liệu phân phối đều vẫn đều, dữ liệu lệch vẫn lệch. Muốn đưa dữ liệu lệch về gần chuẩn phải dùng biến đổi phi tuyến như log, Box-Cox, Yeo-Johnson hay biến đổi theo phân vị (xem [mục D.7 của bài Tiền xử lý dữ liệu](ml/prep_vi)).

### 1.5 Cài đặt bằng code

1. `Z-score normalization` có thể gọi API của thư viện Python hoặc tự cài. Dưới đây dùng lớp của sklearn và tự cài bằng numpy, rồi vẽ histogram so sánh.

```python
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
# 100 mẫu, 15 đặc trưng, phân phối đều trên [0, 100)
X_train = rng.uniform(low=0.0, high=100.0, size=(100, 15))

# 1. z-score bằng sklearn
X_scaled = preprocessing.StandardScaler().fit_transform(X_train)

# 2. z-score tự cài bằng numpy
def z_score_normalization(data):
    data = np.asarray(data)
    return (data - data.mean(axis=0)) / data.std(axis=0)

data_scaled = z_score_normalization(X_train)

fig, axes = plt.subplots(1, 3, figsize=(13, 2.8))
for ax, (title, d, c) in zip(axes, [("dữ liệu gốc", X_train, 'y'),
                                     ("z-score (sklearn)", X_scaled, 'r'),
                                     ("z-score (numpy)", data_scaled, 'g')]):
    ax.hist(d.ravel(), bins=20, density=True, color=c, alpha=0.6)
    ax.set_title(title)
```

> **Biên tập:** Bản gốc dùng các lệnh `%matplotlib`/`%config` của Jupyter và `sns.distplot`, hàm đã lỗi thời từ seaborn 0.11 (2020); đã viết lại bằng matplotlib. Hàm z-score tự cài và sklearn cho kết quả như nhau vì `StandardScaler` cũng dùng độ lệch chuẩn tổng thể (chia cho $N$).

Kết quả như hình dưới. Sau z-score, dữ liệu có trung bình $\mu=0$ và độ lệch chuẩn $\sigma = 1$, nhưng **vẫn là phân phối đều**: đường cong $\mathcal N(0,1)$ vẽ thêm ở ô giữa không khớp với histogram.

<div align=center><img src="../img/code/alchemy_code/10_1.png" alt="Histogram trước và sau z-score"/></div>

Hình 1.5: Histogram của dữ liệu đều trên $[0, 100)$ trước và sau z-score (trang Code). Độ nhọn dư (excess kurtosis) là $-1{,}22$ cả trước lẫn sau (phân phối đều: $-1{,}2$; phân phối chuẩn: 0), và kiểm định Shapiro–Wilk bác bỏ giả thuyết phân phối chuẩn với $p \approx 10^{-21}$.

2. `Min-max normalization` cài đặt đơn giản; dưới đây là bản tự cài bằng numpy:

```python
import numpy as np

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

def min_max_normalization(X):
    # giá trị nhỏ nhất, lớn nhất của toàn mảng
    Xmin = X.min()
    Xmax = X.max()
    return (X - Xmin) / (Xmax - Xmin)

print(min_max_normalization(X))
```

Kết quả: mọi giá trị của mảng được co về khoảng $[0, 1]$.

```output
[[0.    0.125 0.25 ]
 [0.375 0.5   0.625]
 [0.75  0.875 1.   ]]
```

> **Biên tập:** Hàm trên lấy min, max của **toàn mảng**, trong khi min-max (cũng như z-score ở trên) phải tính **theo từng đặc trưng (cột)**: `X.min(axis=0)`, `X.max(axis=0)`, như `sklearn.preprocessing.MinMaxScaler`. Với mảng ví dụ, hai cách tình cờ cho kết quả gần giống nhau vì các cột có cùng thang. Với bảng lương và số năm kinh nghiệm ở hình 1.1, cách tính trên toàn mảng ép cột số năm về khoảng $0{,}0000$–$0{,}0001$, tức đúng vấn đề mà chuẩn hoá cần giải quyết; tính theo cột cho $0$–$1$ (trang Code). Ngoài ra, các thống kê (min, max, trung bình, độ lệch chuẩn) chỉ được tính trên **tập huấn luyện** rồi áp nguyên cho tập kiểm định và kiểm tra, nếu không sẽ rò rỉ dữ liệu; xem [mục D.10 của bài Tiền xử lý dữ liệu](ml/prep_vi).

## 2 Chuẩn hoá ảnh

### 2.1 Định nghĩa chuẩn hoá ảnh

Khi dùng mạng neuron tích chập cho bài toán thị giác máy tính, thường cần chuẩn hoá ảnh đầu vào trong bước tiền xử lý. Hai cách chuẩn hoá ảnh thường gặp là `min-max normalization` và `zero-mean normalization` (z-score).

Lấy zero-mean normalization làm ví dụ: nó đưa trung bình và độ lệch chuẩn của ảnh về `0.0` và `1.0`. Vì là dữ liệu nhiều chiều, khác dữ liệu bảng thuần, **ta trừ trung bình của từng kênh đầu vào, rồi chia kết quả cho độ lệch chuẩn của kênh đó**. Hai dạng chuẩn hoá được định nghĩa như sau:

```shell
# min-max normalization
output[channel] = (input[channel] - min[channel]) / (max[channel] - min[channel])
# zero-mean normalization
output[channel] = (input[channel] - mean[channel]) / std[channel]
```

> **Biên tập:** Bản gốc nói về chuẩn hoá "một ảnh". Trong thực tế `mean[channel]`, `std[channel]` thường là thống kê của **cả tập huấn luyện** (với ImageNet: mean = (0,485; 0,456; 0,406), std = (0,229; 0,224; 0,225) trên thang $[0,1]$), cố định cho mọi ảnh. Chuẩn hoá theo thống kê của từng ảnh (per-image standardization) cũng có, nhưng nó xoá thông tin độ sáng, độ tương phản tổng thể của ảnh. Khi dùng mô hình tiền huấn luyện, phải dùng đúng thống kê mà mô hình đó đã dùng lúc huấn luyện (`weights.transforms()` trong torchvision).

### 2.2 Lợi ích của chuẩn hoá ảnh

Chuẩn hoá ảnh đưa dữ liệu về một khoảng nhất định, giúp **mô hình học nhanh hơn và tốt hơn** (xem mục 1.3).

> **Biên tập:** Bản gốc viết chuẩn hoá ảnh "giảm độ lệch (skewness)" và "giải quyết được vấn đề gradient biến mất và bùng nổ". Chuẩn hoá tuyến tính không đổi độ lệch của phân phối (xem mục 1.4), và chỉ chuẩn hoá đầu vào thì không giải quyết được gradient biến mất hay bùng nổ trong mạng sâu; việc đó cần khởi tạo phù hợp, các tầng chuẩn hoá bên trong mạng (BN, LayerNorm) và kết nối tắt.

### 2.3 Thực hành chuẩn hoá ảnh với PyTorch

Trong `PyTorch`, biến đổi ảnh (image transformation) là **quá trình đổi giá trị điểm ảnh gốc thành giá trị mới**. Phép biến đổi thường gặp là `torchvision.transforms.ToTensor()`, biến ảnh `PIL` có điểm ảnh trong $[0, 255]$ thành `FloatTensor` hình dạng (C, H, W) có giá trị trong $[0.0, 1.0]$. Còn `torchvision.transforms.Normalize()` chuẩn hoá ảnh theo từng kênh (trừ trung bình, chia độ lệch chuẩn). Tóm lại:

- `ToTensor()`: chia giá trị điểm ảnh cho 255;
- `Normalize(mean, std)`: zero-mean normalization theo từng kênh với trung bình và độ lệch chuẩn cho trước.

> **Biên tập:** Bản gốc gọi `ToTensor()` là "min-max normalization". Nó chỉ chia cho 255, hằng số cố định của kiểu `uint8`, không dùng min và max của ảnh. Hai cách chỉ trùng nhau khi ảnh có đúng điểm ảnh 0 và 255 ở mọi kênh. Ảnh mẫu trên trang Code có điểm ảnh từ 0–4 đến 189–231 tuỳ kênh, nên sau `ToTensor` khoảng giá trị là $[0{,}016;\ 0{,}741]$ ở kênh lục, chưa phải $[0, 1]$. Trong `torchvision.transforms.v2`, `ToTensor` được khuyên thay bằng `v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])`, cho kết quả giống hệt.

Cú pháp của `ToTensor()`:

```shell
"""
Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
Converts a PIL Image or numpy.ndarray (H x W x C) in the range
[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

Args:
    pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
Returns:
    Tensor: Converted image.
"""
```

Cú pháp của `Normalize()`:

```shell
Syntax: torchvision.transforms.Normalize()

Parameter:
    * mean: Sequence of means for each channel.
    * std: Sequence of standard deviations for each channel.
    * inplace: Bool to make this operation in-place.
Returns: Normalized Tensor image.
```

Cú pháp của `torch.mean()` (`Tensor.mean()` và `torch.mean()` làm cùng một việc):

```shell
Syntax: torch.mean(input, dim, keepdim=False, *, dtype=None, out=None)

Parameters:
    * input (Tensor): the input tensor.
    * dim (int or tuple of ints): the dimension or dimensions to reduce.
    * keepdim (bool): whether the output tensor has dim retained or not.

Returns: the mean value of each row of the input tensor in the given dimension dim.
If dim is a list of dimensions, reduce over all of them.
```

`torch.mean()` trả về trung bình của tensor theo chiều `dim`. Có thể hiểu `dim` là các chiều sẽ bị rút gọn (biến mất) trong tensor kết quả, dù `dim` là số nguyên hay danh sách. Nếu đầu vào là tensor đặc trưng 3 chiều (C, H, W) và `dim = (1, 2)` thì ta được trung bình của từng kênh. Ví dụ:

```python
import torch
x = torch.rand(3, 4, 8)
y1 = x.mean((1, 2))  # tương đương torch.mean(x, (1, 2))
y2 = x.mean(2)
print(y1.shape)
print(y2.shape)
```

```output
torch.Size([3])
torch.Size([3, 4])
```

Các bước zero-mean normalization một ảnh trong PyTorch:

1. nạp ảnh gốc;
2. dùng `ToTensor()` đổi ảnh thành tensor;
3. tính trung bình và độ lệch chuẩn của tensor theo từng kênh;
4. dùng `Normalize()` để chuẩn hoá.

Code đầy đủ:

```python
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def normalize_image(img_path):
    img = Image.open(img_path)              # nạp ảnh
    # 1. ToTensor
    img_tensor = transforms.ToTensor()(img)  # ảnh PIL -> tensor
    # 2. trung bình và độ lệch chuẩn theo kênh
    mean, std = img_tensor.mean([1, 2]), img_tensor.std([1, 2])
    # 3. Normalize
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img_normalized = transform_norm(img)

    img_np = np.array(img)                  # ảnh PIL -> mảng numpy
    print(img_np.shape)                     # (cao, rộng, kênh)
    print("Mean of the image:", mean)
    print("Std of the image:", std)
    return img_np, img_tensor, img_normalized
```

> **Biên tập:** Hàm của bản gốc nhận tham số `image_path` nhưng bên trong dùng biến toàn cục `img_path` (chỉ chạy được nhờ khối `__main__` đặt đúng tên biến đó); đã sửa tên tham số. Hàm vẽ `show_images` (lấy từ sách *Dive into Deep Learning*) được lược; trang Code có bản vẽ đầy đủ.

Kết quả: ảnh và histogram giá trị điểm ảnh trước và sau hai phép biến đổi.

<div align=center><img src="../img/code/alchemy_code/13_1.png" alt="Ảnh và histogram điểm ảnh trước và sau chuẩn hoá"/></div>

Hình 2.1: Hàng trên: ảnh gốc, sau `ToTensor` và sau `Normalize` bằng thống kê của chính ảnh. Hàng dưới: histogram giá trị điểm ảnh tương ứng (trang Code; ảnh mèo `chelsea` của scikit-image thay cho ảnh gấu koala của bản gốc).

`ToTensor` chỉ đổi thang của trục hoành ($[0, 255] \to [0, 1]$), hình dạng histogram giữ nguyên. Sau `Normalize`, histogram có tâm ở 0 và độ lệch chuẩn 1 nhưng vẫn giữ hình dạng cũ. Ảnh sau `Normalize` trông tối và gắt vì khi hiển thị, `imshow` cắt các giá trị âm về 0 và các giá trị lớn hơn 1 về 1: đó là cách hiển thị, mạng nhận đầy đủ các giá trị.

Đoạn code vẽ phân phối giá trị điểm ảnh:

```python
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("giá trị điểm ảnh")
plt.ylabel("tần suất tương đối")
plt.title("phân phối giá trị điểm ảnh")
```

## 3 Bổ sung: chuẩn hoá trong thực hành hiện nay

Mục này do người biên soạn viết thêm.

- **Thống kê của tập dữ liệu, không phải của từng ảnh.** Với mô hình tiền huấn luyện, `weights.transforms()` của torchvision gói sẵn resize, crop và `Normalize` đúng với lúc huấn luyện mô hình; dùng sai thống kê (hoặc quên chuẩn hoá) là lỗi phổ biến khiến độ chính xác giảm mạnh mà không có thông báo lỗi nào.
- **Chuẩn hoá trên GPU.** Với dữ liệu lớn, có thể giữ ảnh ở `uint8` khi nạp dữ liệu (bộ nhớ nhỏ hơn 4 lần, tải nhanh hơn) rồi chia 255, trừ trung bình, chia độ lệch chuẩn ngay trên GPU, hoặc gộp phép chuẩn hoá vào tầng tích chập đầu tiên khi triển khai (phép affine theo kênh gộp được vào trọng số, giống cách hấp thụ BN).
- **Chuẩn hoá bên trong mạng.** Chuẩn hoá đầu vào chỉ giúp tầng đầu; các tầng sau dựa vào BatchNorm (CNN), LayerNorm hoặc RMSNorm (Transformer) để giữ thang tín hiệu ổn định; xem [bài Batch Normalization](dlnote/dl_bn_vi).
- **Dữ liệu bảng.** Với mạng neuron trên dữ liệu bảng, z-score thường đủ; cột có phân phối lệch nặng (thu nhập, số lượt xem) nên log trước rồi z-score, hoặc dùng `QuantileTransformer`. Mô hình cây (gradient boosting) không cần chuẩn hoá.

## Tài liệu tham khảo

- [A Simple Guide to Data Preprocessing in Machine Learning](https://www.v7labs.com/blog/data-preprocessing-guide)
- [Data Normalization in Data Mining](https://www.geeksforgeeks.org/data-normalization-in-data-mining/)
- [scikit-learn 6.3. Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)
- [numpy.ravel](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html)
- [AI-EDU 05.3 Chuẩn hoá đặc trưng của mẫu](https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC2%E6%AD%A5%20-%20%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92/05.3-%E6%A0%B7%E6%9C%AC%E7%89%B9%E5%BE%81%E6%95%B0%E6%8D%AE%E6%A0%87%E5%87%86%E5%8C%96.html)
- [How To Compute Z-scores in Python](https://cmdlinetips.com/2020/12/compute-standardized-values-z-score-python/)
- [torchvision: Transforming and augmenting images](https://pytorch.org/vision/stable/transforms.html)
