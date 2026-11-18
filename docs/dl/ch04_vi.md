# Chương 4 Mạng Neuron Tuyến tính cho Phân loại

> Ghi chú của người biên soạn: trang này dịch nguyên văn từ sách *Dive into Deep Learning* của **Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola** ([bản gốc](https://d2l.ai/chapter_linear-classification/index.html), [mã nguồn](https://github.com/d2l-ai/d2l-en), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); code theo giấy phép MIT-0). Chỉ giữ code PyTorch. Số mục, hình và công thức giữ như bản HTML d2l.ai; thuật ngữ đối chiếu với bản dịch [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn).

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

## Các mục trong chương

- [4.1 Softmax Regression](https://d2l.ai/chapter_linear-classification/softmax-regression.html)
- [4.2 Tập dữ liệu Phân loại Ảnh](dl/ch04_s2_vi)
- [4.3 Mô hình Phân loại Cơ sở](dl/ch04_s3_vi)
- [4.4 Softmax Regression Implementation from Scratch](https://d2l.ai/chapter_linear-classification/softmax-regression-scratch.html)
- [4.5 Concise Implementation of Softmax Regression](https://d2l.ai/chapter_linear-classification/softmax-regression-concise.html)
- [4.6 Tổng quát hóa trong Phân loại](dl/ch04_s6_vi)
- [4.7 Môi trường và Dịch chuyển Phân phối](dl/ch04_s7_vi)
