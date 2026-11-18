# Khoá Học sâu

Học sâu (deep learning) dùng các mạng neuron nhiều tầng để tự học biểu diễn từ dữ liệu thô: ảnh, âm thanh, văn bản. Khoá này đi qua các kiến trúc cốt lõi (mạng tích chập, mạng hồi tiếp, cơ chế tập trung và Transformer) cùng các kỹ thuật huấn luyện, với code PyTorch chạy được.

## Nguồn và cách biên soạn

Bài đọc dịch nguyên văn từ sách [*Dive into Deep Learning*](https://d2l.ai) (d2l.ai, bản 1.0.3) của Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola, giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Phần dịch trên trang này cũng theo giấy phép CC BY-SA 4.0. Code của sách theo giấy phép MIT-0.

- Chỉ giữ code **PyTorch** (sách gốc có thêm MXNet, JAX và TensorFlow).
- Số chương, mục, hình và công thức giữ như bản HTML trên d2l.ai, nên bạn dễ đối chiếu với bản gốc.
- Thuật ngữ đối chiếu với bản dịch cộng đồng [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn) (dịch từ một phiên bản cũ hơn của sách), riêng *neuron* viết như khoá Học máy. Thuật ngữ tiếng Anh được ghi kèm ở lần xuất hiện đầu tiên.
- Lỗi nhỏ của bản gốc được ghi ở mục **Đính chính** đầu trang, không sửa vào bản dịch.
- Liên kết tới mục chưa dịch dẫn sang bản gốc tiếng Anh trên d2l.ai.

Code trong bài đọc dùng thư viện `d2l` của sách (`from d2l import torch as d2l`). Bản sao của thư viện này nằm ở [`docs/code/dl/d2l`](https://github.com/nmc2004nd/Self-Learning/tree/main/docs/code/dl/d2l), chỉ sửa vài chỗ để chạy ngoài Jupyter. Các trang *Code từ đầu*, *Dùng thư viện* và *Ví dụ áp dụng* do người biên soạn viết thêm, không phụ thuộc `d2l`.

## Các chương

Chương 3–5 của sách (hồi quy tuyến tính, hồi quy softmax, MLP và lan truyền ngược) trùng nhiều với [khoá Học máy](ml/README_vi) chương 7, 12–16, nên chỉ dịch các mục mới; trang mở đầu mỗi chương có bảng dẫn sang phần đã có. Chương 17 (học tăng cường) và 21 (hệ thống gợi ý) không dịch vì đã có ở [khoá Học tăng cường](rl_vi) và khoá Học máy.

| Chương | Bài đọc | Code từ đầu | Dùng thư viện | Ví dụ áp dụng | Mô phỏng | Bổ sung |
| --- | --- | --- | --- | --- | --- | --- |
| – | [Ký hiệu](dl/notation_vi) | | | | | |
| 1 | [Giới thiệu](dl/ch01_vi) | | | | | |
| 2 | [Kiến thức chuẩn bị](dl/ch02_vi) (2.1, 2.5, 2.7) | [Autograd tự viết](dl/ch02_code_vi) | | | [Đồ thị tính toán](dl/ch02_viz_vi) | |
| 3 | [Mạng tuyến tính cho hồi quy](dl/ch03_vi) (3.2, 3.6, 3.7) | | | | | |
| 4 | [Mạng tuyến tính cho phân loại](dl/ch04_vi) (4.2, 4.3, 4.6, 4.7) | | | | | |
| 5 | [Perceptron đa tầng](dl/ch05_vi) (5.4–5.7) | | | | | |
| 6 | [Hướng dẫn xây dựng mô hình](dl/ch06_vi) (6.1–6.7) | [`Module` tự viết](dl/ch06_code_vi) | | [Khởi tạo, checkpoint](dl/ch06_app_vi) | | |
| 7 | [Mạng neuron tích chập](dl/ch07_vi) (7.1–7.6) | [Conv, pooling, LeNet](dl/ch07_code_vi) | [`nn.Conv2d`, huấn luyện](dl/ch07_lib_vi) | [Phân loại ảnh CIFAR-10](dl/ch07_app_vi) | [Cửa sổ trượt](dl/ch07_viz_vi) | [Lan truyền ngược qua tích chập](dl/ch07_bwd_vi) |
| 12 | [Thuật toán tối ưu](dl/ch12_vi) (12.1–12.11) | [Bảy bộ tối ưu tự viết](dl/ch12_code_vi) | [`torch.optim`, lịch tốc độ học](dl/ch12_lib_vi) | [Chọn tốc độ học cho CNN](dl/ch12_app_vi) | [Đua các bộ tối ưu](dl/ch12_viz_vi) | |

Chương 2 và 6 là phần thực hành PyTorch nên chỉ có các trang phụ phù hợp. Các chương tiếp theo, theo thứ tự sẽ soạn: chương 8 (CNN hiện đại); chương 9–10 (mạng hồi tiếp); chương 11 (cơ chế tập trung và Transformer); chương 14, 15–16, 20 (thị giác máy tính, xử lý ngôn ngữ tự nhiên, GAN).

## Cách học một chương

| Trang | Bạn sẽ làm gì |
| --- | --- |
| **Bài đọc** | Đọc lý thuyết và chạy các đoạn code ngắn xen giữa bài, như trong sách |
| **Mô phỏng** | Thao tác trực tiếp trong trình duyệt để có trực giác trước khi viết code |
| **Code từ đầu** | Tự viết lan truyền xuôi của từng phép toán bằng tensor; gradient do autograd tính |
| **Dùng thư viện** | Giải lại bằng các lớp có sẵn của `torch.nn` và một vòng lặp huấn luyện tự viết |
| **Ví dụ áp dụng** | Đi trọn một bài toán trên dữ liệu thật |
| **Bổ sung** | Viết cả lan truyền ngược bằng tay ở những chỗ đáng làm một lần, rồi so với autograd |

## Về kết quả huấn luyện

Các đoạn code ngắn (tính tích chập, kiểm tra kích thước, …) đã được chạy sẵn và in kết quả ngay dưới code. Các khối **huấn luyện mô hình** cần GPU nên tạm hiện dòng *"Chưa có kết quả"*. Kết quả sẽ được bổ sung khi chạy trên máy có GPU. Bạn có thể tự chạy các khối này trên [Google Colab](https://colab.research.google.com) (chọn GPU trong *Runtime → Change runtime type*) hoặc trên máy của mình.
