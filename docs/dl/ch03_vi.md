# Chương 3 Mạng Neuron Tuyến tính cho Hồi quy

> Ghi chú của người biên soạn: trang này dịch nguyên văn từ sách *Dive into Deep Learning* của **Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola** ([bản gốc](https://d2l.ai/chapter_linear-regression/index.html), [mã nguồn](https://github.com/d2l-ai/d2l-en), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); code theo giấy phép MIT-0). Chỉ giữ code PyTorch. Số mục, hình và công thức giữ như bản HTML d2l.ai; thuật ngữ đối chiếu với bản dịch [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn).

Trước khi bận tâm đến việc làm cho mạng neuron trở nên sâu, sẽ có ích nếu ta cài đặt một số mạng nông, trong đó đầu vào kết nối trực tiếp với đầu ra. Điều này sẽ quan trọng vì một vài lý do. Thứ nhất, thay vì bị phân tâm bởi những kiến trúc phức tạp, ta có thể tập trung vào những điều cơ bản của việc huấn luyện mạng neuron, gồm tham số hóa tầng đầu ra, xử lý dữ liệu, đặc tả hàm mất mát và huấn luyện mô hình. Thứ hai, lớp mạng nông này tình cờ bao gồm tập các mô hình tuyến tính, vốn bao hàm nhiều phương pháp dự đoán thống kê cổ điển, gồm hồi quy tuyến tính và hồi quy softmax. Hiểu những công cụ cổ điển này là then chốt vì chúng được dùng rộng rãi trong nhiều bối cảnh và ta sẽ thường cần dùng chúng làm mốc so sánh khi biện minh cho việc dùng những kiến trúc phức tạp hơn. Chương này sẽ tập trung hẹp vào hồi quy tuyến tính và chương tiếp theo sẽ mở rộng vốn mô hình của ta bằng cách phát triển các mạng neuron tuyến tính cho bài toán phân loại.

> **Ghi chú của người biên soạn:** hồi quy tuyến tính đã được trình bày kỹ ở [chương 7 khoá Học máy](ml/ch07_vi), kèm code từ đầu bằng NumPy và code dùng thư viện. Vì vậy khoá này chỉ dịch các mục mới so với khoá Học máy:
>
> | Mục của sách | Trên site |
> | --- | --- |
> | 3.1 Hồi quy tuyến tính | [ML chương 7](ml/ch07_vi), [ML chương 12](ml/ch12_vi) (hạ gradient theo minibatch) |
> | 3.2 Thiết kế hướng đối tượng cho việc cài đặt | **Dịch ở khoá này**: các lớp `Module`, `DataModule`, `Trainer` mà mọi đoạn code của sách về sau đều dùng |
> | 3.3–3.5 Dữ liệu tổng hợp, cài đặt từ đầu, cài đặt súc tích | [ML chương 7 · Code từ đầu](ml/ch07_code_vi), [Dùng thư viện](ml/ch07_lib_vi) |
> | 3.6 Tổng quát hóa | **Dịch ở khoá này** |
> | 3.7 Suy giảm trọng số | **Dịch ở khoá này** (khái niệm có ở [ML chương 8](ml/ch08_vi)) |

## Các mục trong chương

- [3.1 Linear Regression](https://d2l.ai/chapter_linear-regression/linear-regression.html)
- [3.2 Thiết kế Hướng đối tượng cho việc Cài đặt](dl/ch03_s2_vi)
- [3.3 Synthetic Regression Data](https://d2l.ai/chapter_linear-regression/synthetic-regression-data.html)
- [3.4 Linear Regression Implementation from Scratch](https://d2l.ai/chapter_linear-regression/linear-regression-scratch.html)
- [3.5 Concise Implementation of Linear Regression](https://d2l.ai/chapter_linear-regression/linear-regression-concise.html)
- [3.6 Tổng quát hóa](dl/ch03_s6_vi)
- [3.7 Suy giảm Trọng số](dl/ch03_s7_vi)
