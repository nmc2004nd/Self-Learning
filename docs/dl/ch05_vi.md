# Chương 5 Perceptron Đa tầng

> Ghi chú của người biên soạn: trang này dịch nguyên văn từ sách *Dive into Deep Learning* của **Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola** ([bản gốc](https://d2l.ai/chapter_multilayer-perceptrons/index.html), [mã nguồn](https://github.com/d2l-ai/d2l-en), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); code theo giấy phép MIT-0). Chỉ giữ code PyTorch. Số mục, hình và công thức giữ như bản HTML d2l.ai; thuật ngữ đối chiếu với bản dịch [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn).

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

## Các mục trong chương

- [5.1 Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)
- [5.2 Implementation of Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp-implementation.html)
- [5.3 Forward Propagation, Backward Propagation, and Computational Graphs](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html)
- [5.4 Ổn định Số và Khởi tạo](dl/ch05_s4_vi)
- [5.5 Tổng quát hóa trong Học sâu](dl/ch05_s5_vi)
- [5.6 Dropout](dl/ch05_s6_vi)
- [5.7 Dự đoán Giá nhà trên Kaggle](dl/ch05_s7_vi)
