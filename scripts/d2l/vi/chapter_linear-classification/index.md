# Mạng Neuron Tuyến tính cho Phân loại
:label:`chap_classification`

Giờ khi đã đi qua toàn bộ cơ chế, bạn đã sẵn sàng áp dụng những kỹ năng đã học vào những loại tác vụ rộng hơn. Ngay cả khi ta chuyển hướng sang phân loại, phần lớn hệ thống ống nước vẫn giữ nguyên: nạp dữ liệu, đưa nó qua mô hình, tạo đầu ra, tính mất mát, lấy gradient theo trọng số và cập nhật mô hình. Tuy nhiên, dạng chính xác của mục tiêu, cách tham số hóa tầng đầu ra và lựa chọn hàm mất mát sẽ thay đổi để phù hợp với bối cảnh *phân loại*.

> **Ghi chú của người biên soạn:** hồi quy softmax đã được trình bày kỹ ở [chương 15 khoá Học máy](ml/ch15_vi), kèm code từ đầu, code dùng thư viện và ví dụ trên chính Fashion-MNIST. Vì vậy khoá này chỉ dịch các mục mới:
>
> | Mục của sách | Trên site |
> | --- | --- |
> | 4.1 Hồi quy softmax | [ML chương 15](ml/ch15_vi) |
> | 4.2 Tập dữ liệu phân loại ảnh | **Dịch ở khoá này**: lớp `FashionMNIST` dùng xuyên suốt sách |
> | 4.3 Mô hình phân loại cơ sở | **Dịch ở khoá này**: lớp `Classifier` mà LeNet (mục 7.6) kế thừa |
> | 4.4–4.5 Cài đặt từ đầu, cài đặt súc tích | [ML chương 15 · Code từ đầu](ml/ch15_code_vi), [Dùng thư viện](ml/ch15_lib_vi), [Ví dụ áp dụng](ml/ch15_app_vi) |
> | 4.6 Tổng quát hóa trong phân loại | **Dịch ở khoá này** (đánh giá mô hình cũng có ở [Model evaluation](ml/eval_vi)) |
> | 4.7 Môi trường và dịch chuyển phân phối | **Dịch ở khoá này** |

```toc
:maxdepth: 2

softmax-regression
image-classification-dataset
classification
softmax-regression-scratch
softmax-regression-concise
generalization-classification
environment-and-distribution-shift
```
