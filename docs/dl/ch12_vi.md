# Chương 12 Thuật toán Tối ưu

> Ghi chú của người biên soạn: trang này dịch nguyên văn từ sách *Dive into Deep Learning* của **Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola** ([bản gốc](https://d2l.ai/chapter_optimization/index.html), [mã nguồn](https://github.com/d2l-ai/d2l-en), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); code theo giấy phép MIT-0). Chỉ giữ code PyTorch. Số mục, hình và công thức giữ như bản HTML d2l.ai; thuật ngữ đối chiếu với bản dịch [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn).

Nếu bạn đọc cuốn sách theo thứ tự đến đây, bạn đã dùng một số thuật toán tối ưu để huấn luyện các mô hình học sâu. Chúng là những công cụ cho phép ta liên tục cập nhật tham số mô hình và cực tiểu hóa giá trị của hàm mất mát khi đánh giá trên tập huấn luyện. Thật vậy, bất kỳ ai bằng lòng coi tối ưu như một chiếc hộp đen để cực tiểu hóa hàm mục tiêu trong một bối cảnh đơn giản có thể hài lòng với việc biết rằng tồn tại cả một loạt "câu thần chú" cho một thủ tục như vậy (với những cái tên như "SGD" và "Adam").

Tuy nhiên, để làm tốt, cần có kiến thức sâu hơn. Các thuật toán tối ưu rất quan trọng đối với học sâu. Một mặt, huấn luyện một mô hình học sâu phức tạp có thể mất hàng giờ, hàng ngày, thậm chí hàng tuần. Hiệu năng của thuật toán tối ưu ảnh hưởng trực tiếp đến hiệu quả huấn luyện của mô hình. Mặt khác, hiểu các nguyên lý của những thuật toán tối ưu khác nhau và vai trò của các siêu tham số của chúng sẽ cho phép ta tinh chỉnh siêu tham số một cách có định hướng để cải thiện hiệu năng của các mô hình học sâu.

Trong chương này, ta tìm hiểu sâu các thuật toán tối ưu thông dụng trong học sâu. Hầu như mọi bài toán tối ưu nảy sinh trong học sâu đều *không lồi*. Dù vậy, việc thiết kế và phân tích thuật toán trong bối cảnh các bài toán *lồi* đã chứng tỏ rất bổ ích. Chính vì lý do đó mà chương này có phần nhập môn về tối ưu lồi và chứng minh cho một thuật toán hạ gradient ngẫu nhiên rất đơn giản trên một hàm mục tiêu lồi.

> **Ghi chú của người biên soạn:** chương này dịch đầy đủ. Một số nội dung đã có ở các khoá khác trên site và có thể đọc kèm: hạ gradient, momentum và NAG ở [chương 12 khoá Học máy](ml/ch12_vi) (có mô phỏng); tập lồi, hàm lồi và tối ưu lồi ở [chương 23–25 khoá Toán](math/ch23_vi).

## Các mục trong chương

- [12.1 Tối ưu và Học sâu](dl/ch12_s1_vi)
- [12.2 Tính Lồi](dl/ch12_s2_vi)
- [12.3 Hạ Gradient](dl/ch12_s3_vi)
- [12.4 Hạ Gradient Ngẫu nhiên](dl/ch12_s4_vi)
- [12.5 Hạ Gradient Ngẫu nhiên theo Minibatch](dl/ch12_s5_vi)
- [12.6 Động lượng](dl/ch12_s6_vi)
- [12.7 Adagrad](dl/ch12_s7_vi)
- [12.8 RMSProp](dl/ch12_s8_vi)
- [12.9 Adadelta](dl/ch12_s9_vi)
- [12.10 Adam](dl/ch12_s10_vi)
- [12.11 Lập lịch Tốc độ học](dl/ch12_s11_vi)
