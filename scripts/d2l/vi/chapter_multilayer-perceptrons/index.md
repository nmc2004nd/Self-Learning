# Perceptron Đa tầng
:label:`chap_perceptrons`

Trong chương này, chúng tôi sẽ giới thiệu mạng thực sự *sâu* đầu tiên của bạn. Các mạng sâu đơn giản nhất được gọi là *perceptron đa tầng* (multilayer perceptron), chúng gồm nhiều tầng neuron, mỗi tầng kết nối đầy đủ với tầng bên dưới (từ đó chúng nhận đầu vào) và tầng bên trên (mà đến lượt chúng tác động lên). Dù tính vi phân tự động đơn giản hóa đáng kể việc cài đặt các thuật toán học sâu, ta sẽ đi sâu vào cách các gradient này được tính trong mạng sâu. Sau đó ta sẽ sẵn sàng thảo luận các vấn đề liên quan đến ổn định số và khởi tạo tham số, vốn then chốt để huấn luyện thành công mạng sâu. Khi huấn luyện những mô hình có năng lực cao như vậy, ta có nguy cơ quá khớp. Vì vậy, ta sẽ xem lại điều chuẩn và tổng quát hóa cho mạng sâu. Xuyên suốt, chúng tôi muốn giúp bạn nắm vững không chỉ khái niệm mà cả cách thực hành với mạng sâu. Ở cuối chương, ta áp dụng những gì đã giới thiệu vào một trường hợp thực tế: dự đoán giá nhà. Các vấn đề liên quan đến hiệu năng tính toán, khả năng mở rộng và hiệu quả của mô hình được để lại cho các chương sau.

> **Ghi chú của người biên soạn:** MLP và lan truyền ngược đã được trình bày kỹ ở [chương 16 khoá Học máy](ml/ch16_vi), kèm code NumPy tự viết cả lan truyền ngược. Vì vậy khoá này chỉ dịch các mục mới:
>
> | Mục của sách | Trên site |
> | --- | --- |
> | 5.1 Perceptron đa tầng | [ML chương 16](ml/ch16_vi), [Mô phỏng](ml/ch16_viz_vi) |
> | 5.2 Cài đặt MLP | [ML chương 16 · Code từ đầu](ml/ch16_code_vi), [Dùng thư viện](ml/ch16_lib_vi), [Ví dụ áp dụng](ml/ch16_app_vi) |
> | 5.3 Lan truyền xuôi, lan truyền ngược và đồ thị tính toán | [ML chương 16](ml/ch16_vi) mục 16.4; [Chương 2 · Mô phỏng](dl/ch02_viz_vi) (đồ thị tính toán) |
> | 5.4 Ổn định số và khởi tạo | **Dịch ở khoá này** |
> | 5.5 Tổng quát hóa trong học sâu | **Dịch ở khoá này** |
> | 5.6 Dropout | **Dịch ở khoá này** |
> | 5.7 Dự đoán giá nhà trên Kaggle | **Dịch ở khoá này** |

```toc
:maxdepth: 2

mlp
mlp-implementation
backprop
numerical-stability-and-init
generalization-deep
dropout
kaggle-house-price
```
