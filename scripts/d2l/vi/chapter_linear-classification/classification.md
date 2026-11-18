# Mô hình Phân loại Cơ sở
:label:`sec_classification`

Có thể bạn đã để ý rằng trong trường hợp hồi quy, cách cài đặt từ đầu và cách cài đặt súc tích dùng chức năng của framework khá giống nhau. Điều tương tự cũng đúng với phân loại. Vì nhiều mô hình trong cuốn sách này xử lý bài toán phân loại, đáng để thêm các chức năng hỗ trợ riêng cho bối cảnh này. Mục này cung cấp một lớp cơ sở cho các mô hình phân loại để đơn giản hóa code về sau.

```{.python .input}
from d2l import torch as d2l
import torch
```

## Lớp `Classifier`

Ta định nghĩa lớp `Classifier` dưới đây. Trong `validation_step`, ta báo cáo cả giá trị mất mát lẫn độ chính xác phân loại trên một batch xác thực. Ta vẽ một lần cập nhật sau mỗi `num_val_batches` batch. Cách này có lợi là tạo ra mất mát và độ chính xác trung bình trên toàn bộ dữ liệu xác thực. Các con số trung bình này không hoàn toàn chính xác nếu batch cuối cùng chứa ít mẫu hơn, nhưng ta bỏ qua khác biệt nhỏ này để giữ code đơn giản.

```{.python .input}
class Classifier(d2l.Module):  #@save
    """Lớp cơ sở của các mô hình phân loại."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

Theo mặc định, ta dùng bộ tối ưu hạ gradient ngẫu nhiên, hoạt động trên các minibatch, giống như ta đã làm trong bối cảnh hồi quy tuyến tính.

```{.python .input}
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

## Độ chính xác

Cho phân phối xác suất dự đoán `y_hat`, ta thường chọn lớp có xác suất dự đoán cao nhất bất cứ khi nào phải đưa ra một dự đoán dứt khoát. Thật vậy, nhiều ứng dụng đòi hỏi ta phải đưa ra lựa chọn. Chẳng hạn, Gmail phải xếp một email vào "Chính", "Mạng xã hội", "Cập nhật", "Diễn đàn" hoặc "Thư rác". Bên trong nó có thể ước lượng các xác suất, nhưng sau cùng nó phải chọn một trong các lớp.

Khi dự đoán khớp với lớp nhãn `y`, chúng là đúng. Độ chính xác phân loại (accuracy) là tỉ lệ các dự đoán đúng trên tổng số dự đoán. Dù có thể khó tối ưu trực tiếp độ chính xác (nó không khả vi), đây thường là thước đo hiệu năng mà ta quan tâm nhất. Nó thường là *đại lượng* được quan tâm trong các bộ đánh giá chuẩn. Vì vậy, ta sẽ hầu như luôn báo cáo nó khi huấn luyện bộ phân loại.

Độ chính xác được tính như sau. Trước tiên, nếu `y_hat` là một ma trận, ta giả định rằng chiều thứ hai lưu điểm dự đoán cho mỗi lớp. Ta dùng `argmax` để lấy lớp được dự đoán qua chỉ số của phần tử lớn nhất trong mỗi hàng. Sau đó ta [**so sánh lớp được dự đoán với giá trị thực `y` theo từng phần tử.**] Vì toán tử bằng `==` nhạy cảm với kiểu dữ liệu, ta chuyển kiểu dữ liệu của `y_hat` cho khớp với `y`. Kết quả là một tensor chứa các phần tử 0 (sai) và 1 (đúng). Lấy tổng sẽ cho số dự đoán đúng.

```{.python .input  n=9}
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Tính số dự đoán đúng."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

## Tóm tắt

Phân loại là một bài toán đủ phổ biến để xứng đáng có các hàm tiện lợi riêng. Điều tối quan trọng trong phân loại là *độ chính xác* của bộ phân loại. Lưu ý rằng dù ta thường quan tâm chủ yếu đến độ chính xác, ta huấn luyện bộ phân loại để tối ưu nhiều mục tiêu khác vì những lý do thống kê và tính toán. Tuy nhiên, bất kể hàm mất mát nào được cực tiểu hóa trong quá trình huấn luyện, có một phương thức tiện lợi để đánh giá độ chính xác của bộ phân loại theo thực nghiệm là điều hữu ích.

## Bài tập

1. Ký hiệu $L_\textrm{v}$ là mất mát xác thực, và gọi $L_\textrm{v}^\textrm{q}$ là ước lượng nhanh và thô của nó được tính bằng cách lấy trung bình hàm mất mát như trong mục này. Cuối cùng, ký hiệu $l_\textrm{v}^\textrm{b}$ là mất mát trên minibatch cuối cùng. Hãy biểu diễn $L_\textrm{v}$ theo $L_\textrm{v}^\textrm{q}$, $l_\textrm{v}^\textrm{b}$, và kích thước mẫu cùng kích thước minibatch.
1. Chứng minh rằng ước lượng nhanh và thô $L_\textrm{v}^\textrm{q}$ là không chệch. Tức là, chứng minh rằng $E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$. Vì sao bạn vẫn muốn dùng $L_\textrm{v}$ thay vì nó?
1. Cho một hàm mất mát phân loại đa lớp, ký hiệu $l(y,y')$ là mức phạt khi ước lượng $y'$ trong khi thấy $y$, và cho một xác suất $p(y \mid x)$, hãy phát biểu quy tắc chọn $y'$ tối ưu. Gợi ý: biểu diễn mất mát kỳ vọng bằng $l$ và $p(y \mid x)$.

[Discussions](https://discuss.d2l.ai/t/6809)
