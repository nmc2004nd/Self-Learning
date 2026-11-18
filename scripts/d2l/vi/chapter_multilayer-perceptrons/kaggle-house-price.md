# Dự đoán Giá nhà trên Kaggle
:label:`sec_kaggle_house`

Giờ khi đã giới thiệu một số công cụ cơ bản để xây dựng và huấn luyện mạng sâu và điều chuẩn chúng bằng các kỹ thuật gồm suy giảm trọng số và dropout, ta đã sẵn sàng đưa toàn bộ kiến thức này vào thực hành bằng cách tham gia một cuộc thi Kaggle. Cuộc thi dự đoán giá nhà là một nơi tuyệt vời để bắt đầu. Dữ liệu khá thông thường và không có cấu trúc lạ có thể đòi hỏi các mô hình chuyên biệt (như âm thanh hay video). Tập dữ liệu này, do :citet:`De-Cock.2011` thu thập, bao gồm giá nhà ở Ames, Iowa trong giai đoạn 2006–2010. Nó lớn hơn đáng kể so với [tập dữ liệu nhà ở Boston](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) nổi tiếng của Harrison và Rubinfeld (1978), với cả nhiều mẫu hơn lẫn nhiều đặc trưng hơn.

Trong mục này, chúng tôi sẽ dẫn bạn qua các chi tiết về tiền xử lý dữ liệu, thiết kế mô hình và lựa chọn siêu tham số. Chúng tôi hy vọng rằng thông qua cách tiếp cận thực hành, bạn sẽ có được một số trực giác dẫn dắt bạn trong sự nghiệp nhà khoa học dữ liệu.

```{.python .input}
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
```

## Tải Dữ liệu

Xuyên suốt cuốn sách, ta sẽ huấn luyện và kiểm tra mô hình trên nhiều tập dữ liệu được tải về. Ở đây, ta (**cài đặt hai hàm tiện ích**) để tải về và giải nén các tệp zip hoặc tar. Một lần nữa, ta bỏ qua chi tiết cài đặt của những hàm tiện ích như vậy.

```{.python .input  n=2}
def download(url, folder, sha1_hash=None):
    """Tải một tệp về thư mục và trả về đường dẫn cục bộ của tệp."""

def extract(filename, folder):
    """Giải nén một tệp zip/tar vào thư mục."""
```

## Kaggle

[Kaggle](https://www.kaggle.com) là một nền tảng phổ biến tổ chức các cuộc thi học máy. Mỗi cuộc thi xoay quanh một tập dữ liệu và nhiều cuộc thi được tài trợ bởi các bên liên quan, những người trao giải cho các giải pháp chiến thắng. Nền tảng giúp người dùng tương tác qua các diễn đàn và code được chia sẻ, thúc đẩy cả hợp tác lẫn cạnh tranh. Dù việc đuổi theo bảng xếp hạng thường vượt khỏi tầm kiểm soát, với các nhà nghiên cứu tập trung một cách thiển cận vào các bước tiền xử lý thay vì đặt ra những câu hỏi cơ bản, tính khách quan của một nền tảng tạo điều kiện so sánh định lượng trực tiếp giữa các cách tiếp cận cạnh tranh, cùng với việc chia sẻ code để mọi người có thể học được điều gì hiệu quả và điều gì không, cũng mang lại giá trị to lớn. Nếu muốn tham gia một cuộc thi Kaggle, trước tiên bạn cần đăng ký một tài khoản (xem :numref:`fig_kaggle`).

![Trang web Kaggle.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

Trên trang cuộc thi dự đoán giá nhà, như minh họa trong :numref:`fig_house_pricing`, bạn có thể tìm thấy tập dữ liệu (ở thẻ "Data"), nộp dự đoán và xem thứ hạng của mình. Địa chỉ ở ngay đây:

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![Trang cuộc thi dự đoán giá nhà.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## Truy cập và Đọc Tập dữ liệu

Lưu ý rằng dữ liệu cuộc thi được chia thành tập huấn luyện và tập kiểm tra. Mỗi bản ghi gồm giá trị bất động sản của ngôi nhà và các thuộc tính như loại đường, năm xây dựng, loại mái, tình trạng tầng hầm, v.v. Các đặc trưng gồm nhiều kiểu dữ liệu khác nhau. Chẳng hạn, năm xây dựng được biểu diễn bằng một số nguyên, loại mái bằng các giá trị phân loại rời rạc, và các đặc trưng khác bằng số thực dấu phẩy động. Và đây là lúc thực tế làm mọi thứ phức tạp: với một số mẫu, một số dữ liệu bị thiếu hoàn toàn, với giá trị thiếu chỉ được đánh dấu đơn giản là "na". Giá của mỗi ngôi nhà chỉ được cung cấp cho tập huấn luyện (suy cho cùng, đây là một cuộc thi). Ta sẽ muốn chia tập huấn luyện để tạo một tập xác thực, nhưng ta chỉ được đánh giá mô hình trên tập kiểm tra chính thức sau khi tải dự đoán lên Kaggle. Thẻ "Data" trên trang cuộc thi trong :numref:`fig_house_pricing` có các liên kết để tải dữ liệu.

Để bắt đầu, ta sẽ [**đọc và xử lý dữ liệu bằng `pandas`**], thư viện ta đã giới thiệu trong :numref:`sec_pandas`. Để tiện lợi, ta có thể tải về và lưu đệm tập dữ liệu nhà ở của Kaggle. Nếu một tệp tương ứng với tập dữ liệu này đã tồn tại trong thư mục đệm và SHA-1 của nó khớp với `sha1_hash`, code của ta sẽ dùng tệp đã lưu đệm để tránh làm nghẽn Internet của bạn với những lần tải thừa.

```{.python .input  n=30}
class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
```

Tập dữ liệu huấn luyện gồm 1460 mẫu, 80 đặc trưng và một nhãn, trong khi dữ liệu xác thực chứa 1459 mẫu và 80 đặc trưng.

```{.python .input  n=31}
data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)
```

## Tiền xử lý Dữ liệu

Hãy [**xem bốn đặc trưng đầu tiên và hai đặc trưng cuối cùng cùng với nhãn (SalePrice)**] của bốn mẫu đầu tiên.

```{.python .input  n=10}
print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
```

Ta có thể thấy rằng trong mỗi mẫu, đặc trưng đầu tiên là định danh (ID). Điều này giúp mô hình xác định từng mẫu huấn luyện. Dù tiện lợi, nó không mang thông tin nào cho mục đích dự đoán. Do đó, ta sẽ loại nó khỏi tập dữ liệu trước khi đưa dữ liệu vào mô hình. Hơn nữa, với nhiều kiểu dữ liệu khác nhau, ta sẽ cần tiền xử lý dữ liệu trước khi có thể bắt đầu mô hình hóa.

Hãy bắt đầu với các đặc trưng số. Trước tiên, ta áp dụng một phương pháp kinh nghiệm, [**thay mọi giá trị bị thiếu bằng giá trị trung bình của đặc trưng tương ứng.**] Sau đó, để đưa mọi đặc trưng về cùng một thang đo, ta (***chuẩn hóa* dữ liệu bằng cách co giãn các đặc trưng về trung bình không và phương sai đơn vị**):

$$x \leftarrow \frac{x - \mu}{\sigma},$$

trong đó $\mu$ và $\sigma$ lần lượt là trung bình và độ lệch chuẩn. Để kiểm chứng rằng phép này thực sự biến đổi đặc trưng (biến) của ta để có trung bình không và phương sai đơn vị, lưu ý rằng $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$ và $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$. Một cách trực quan, ta chuẩn hóa dữ liệu vì hai lý do. Thứ nhất, nó tỏ ra tiện lợi cho việc tối ưu. Thứ hai, vì ta không biết *trước* những đặc trưng nào sẽ có liên quan, ta không muốn phạt hệ số gán cho một đặc trưng nhiều hơn bất kỳ đặc trưng nào khác.

[**Tiếp theo ta xử lý các giá trị rời rạc.**] Những giá trị này gồm các đặc trưng như "MSZoning". (**Ta thay chúng bằng mã hóa one-hot**) theo cùng cách mà trước đây ta đã biến đổi nhãn đa lớp thành vector (xem :numref:`subsec_classification-problem`). Chẳng hạn, "MSZoning" nhận các giá trị "RL" và "RM". Bỏ đặc trưng "MSZoning", hai đặc trưng chỉ thị mới "MSZoning_RL" và "MSZoning_RM" được tạo ra với giá trị là 0 hoặc 1. Theo mã hóa one-hot, nếu giá trị ban đầu của "MSZoning" là "RL", thì "MSZoning_RL" bằng 1 và "MSZoning_RM" bằng 0. Gói `pandas` tự động làm điều này cho ta.

```{.python .input  n=32}
@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Bỏ cột ID và cột nhãn
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # Chuẩn hóa các cột số
    # Người biên soạn sửa: bản gốc lọc features.dtypes!='object', sai với pandas >= 3
    numeric_features = features.select_dtypes(include='number').columns
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # Thay các đặc trưng số bị thiếu (NAN) bằng 0
    features[numeric_features] = features[numeric_features].fillna(0)
    # Thay các đặc trưng rời rạc bằng mã hóa one-hot
    features = pd.get_dummies(features, dummy_na=True)
    # Lưu các đặc trưng đã tiền xử lý
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()
```

Bạn có thể thấy phép chuyển đổi này tăng số đặc trưng từ 79 lên 331 (không tính cột ID và cột nhãn).

```{.python .input  n=33}
data.preprocess()
data.train.shape
```

## Thước đo Lỗi

Để bắt đầu, ta sẽ huấn luyện một mô hình tuyến tính với mất mát bình phương. Không có gì ngạc nhiên, mô hình tuyến tính sẽ không dẫn đến một bài nộp chiến thắng cuộc thi nhưng nó cung cấp một phép kiểm tra nhanh để xem liệu dữ liệu có thông tin ý nghĩa hay không. Nếu ở đây ta không làm tốt hơn đoán bừa, rất có thể ta có lỗi trong khâu xử lý dữ liệu. Và nếu mọi thứ hoạt động, mô hình tuyến tính sẽ đóng vai trò mốc so sánh, cho ta trực giác về mô hình đơn giản tiến gần đến các mô hình tốt nhất được báo cáo đến đâu, cho ta cảm nhận về mức cải thiện nên mong đợi từ các mô hình phức tạp hơn.

Với giá nhà, cũng như giá cổ phiếu, ta quan tâm đến các đại lượng tương đối hơn là đại lượng tuyệt đối. Do đó [**ta thường quan tâm đến sai số tương đối $\frac{y - \hat{y}}{y}$**] hơn là sai số tuyệt đối $y - \hat{y}$. Chẳng hạn, nếu dự đoán của ta lệch \$100 000 khi ước lượng giá một ngôi nhà ở vùng nông thôn Ohio, nơi giá trị một ngôi nhà điển hình là \$125 000, thì có lẽ ta đang làm rất tệ. Mặt khác, nếu ta sai số đúng chừng ấy ở Los Altos Hills, California, điều này có thể là một dự đoán chính xác đáng kinh ngạc (ở đó, giá nhà trung vị vượt quá \$4 triệu).

(**Một cách để giải quyết vấn đề này là đo độ chênh lệch theo logarit của giá ước lượng.**) Thực ra, đây cũng là thước đo lỗi chính thức mà cuộc thi dùng để đánh giá chất lượng các bài nộp. Suy cho cùng, một giá trị nhỏ $\delta$ cho $|\log y - \log \hat{y}| \leq \delta$ chuyển thành $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$. Điều này dẫn đến căn bậc hai của trung bình bình phương sai số giữa logarit của giá dự đoán và logarit của giá nhãn:

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input  n=60}
@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data: return
    get_tensor = lambda x: d2l.tensor(x.values.astype(float),
                                      dtype=d2l.float32)
    # Logarit của giá
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               d2l.reshape(d2l.log(get_tensor(data[label])), (-1, 1)))  # Y
    return self.get_tensorloader(tensors, train)
```

## Kiểm định Chéo $K$ phần

Có thể bạn còn nhớ rằng ta đã giới thiệu [**kiểm định chéo**] trong :numref:`subsec_generalization-model-selection`, nơi ta thảo luận cách xử lý việc lựa chọn mô hình. Ta sẽ tận dụng nó để chọn thiết kế mô hình và điều chỉnh siêu tham số. Trước tiên ta cần một hàm trả về phần thứ $i$ của dữ liệu trong quy trình kiểm định chéo $K$ phần. Nó cắt ra đoạn thứ $i$ làm dữ liệu xác thực và trả về phần còn lại làm dữ liệu huấn luyện. Lưu ý rằng đây không phải cách hiệu quả nhất để xử lý dữ liệu và chắc chắn ta sẽ làm điều gì đó thông minh hơn nhiều nếu tập dữ liệu lớn hơn đáng kể. Nhưng sự phức tạp thêm vào này có thể làm rối code một cách không cần thiết nên ta có thể yên tâm bỏ qua ở đây nhờ sự đơn giản của bài toán.

```{.python .input}
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),  
                                data.train.loc[idx]))    
    return rets
```

[**Lỗi xác thực trung bình được trả về**] khi ta huấn luyện $K$ lần trong kiểm định chéo $K$ phần.

```{.python .input}
def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models
```

## [**Lựa chọn Mô hình**]

Trong ví dụ này, ta chọn một bộ siêu tham số chưa được tinh chỉnh và để người đọc tự cải thiện mô hình. Tìm được một lựa chọn tốt có thể mất thời gian, tùy thuộc vào số biến cần tối ưu. Với một tập dữ liệu đủ lớn và các loại siêu tham số thông thường, kiểm định chéo $K$ phần có xu hướng khá bền vững trước việc kiểm định nhiều lần. Tuy nhiên, nếu ta thử một số lượng lựa chọn lớn bất hợp lý, ta có thể thấy rằng hiệu năng xác thực không còn đại diện cho lỗi thực nữa.

```{.python .input}
trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
```

Lưu ý rằng đôi khi số lỗi huấn luyện của một bộ siêu tham số có thể rất thấp, trong khi số lỗi trong kiểm định chéo $K$ phần lại cao hơn đáng kể. Điều này cho thấy ta đang quá khớp. Trong suốt quá trình huấn luyện, bạn sẽ muốn theo dõi cả hai con số. Ít quá khớp hơn có thể cho thấy dữ liệu của ta có thể hỗ trợ một mô hình mạnh hơn. Quá khớp nghiêm trọng có thể gợi ý rằng ta có thể được lợi từ việc đưa vào các kỹ thuật điều chuẩn.

##  [**Nộp Dự đoán lên Kaggle**]

Giờ khi đã biết một lựa chọn siêu tham số tốt nên là gì, ta có thể tính trung bình dự đoán trên tập kiểm tra của cả $K$ mô hình. Lưu dự đoán vào một tệp csv sẽ giúp đơn giản hóa việc tải kết quả lên Kaggle. Đoạn code sau sẽ tạo ra một tệp tên `submission.csv`.

```{.python .input}
preds = [model(d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
         for model in models]
# Lấy hàm mũ của các dự đoán trên thang logarit
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

Tiếp theo, như minh họa trong :numref:`fig_kaggle_submit2`, ta có thể nộp dự đoán lên Kaggle và xem chúng so với giá nhà thực tế (nhãn) trên tập kiểm tra như thế nào. Các bước khá đơn giản:

* Đăng nhập trang web Kaggle và vào trang cuộc thi dự đoán giá nhà.
* Bấm nút "Submit Predictions" hoặc "Late Submission".
* Bấm nút "Upload Submission File" trong ô viền nét đứt ở cuối trang và chọn tệp dự đoán bạn muốn tải lên.
* Bấm nút "Make Submission" ở cuối trang để xem kết quả.

![Nộp dữ liệu lên Kaggle.](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## Tóm tắt và Thảo luận

Dữ liệu thực thường chứa nhiều kiểu dữ liệu khác nhau trộn lẫn và cần được tiền xử lý. Co giãn dữ liệu giá trị thực về trung bình không và phương sai đơn vị là một mặc định tốt. Thay giá trị bị thiếu bằng giá trị trung bình cũng vậy. Hơn nữa, biến đổi các đặc trưng phân loại thành các đặc trưng chỉ thị cho phép ta xử lý chúng như các vector one-hot. Khi ta quan tâm đến sai số tương đối hơn là sai số tuyệt đối, ta có thể đo độ chênh lệch theo logarit của dự đoán. Để lựa chọn mô hình và điều chỉnh siêu tham số, ta có thể dùng kiểm định chéo $K$ phần.

## Bài tập

1. Nộp dự đoán của mục này lên Kaggle. Chúng tốt đến đâu?
1. Thay giá trị bị thiếu bằng giá trị trung bình có luôn là một ý hay không? Gợi ý: bạn có thể dựng một tình huống mà các giá trị không bị thiếu một cách ngẫu nhiên không?
1. Cải thiện điểm số bằng cách tinh chỉnh siêu tham số thông qua kiểm định chéo $K$ phần.
1. Cải thiện điểm số bằng cách cải thiện mô hình (chẳng hạn tầng, suy giảm trọng số và dropout).
1. Điều gì xảy ra nếu ta không chuẩn hóa các đặc trưng số liên tục như đã làm trong mục này?

[Discussions](https://discuss.d2l.ai/t/107)
