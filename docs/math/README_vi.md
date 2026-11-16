# Khoá Toán nền tảng

Hầu hết thuật toán học máy đều viết bằng ngôn ngữ của đại số tuyến tính, giải tích và xác suất. Khoá này ôn lại đúng những phần toán mà các khoá [Học máy](ml/README_vi) và [Học tăng cường](rl_vi) cần dùng, rồi đi tiếp tới tối ưu lồi, nền móng của máy vector hỗ trợ và nhiều thuật toán khác.

## Các chương

Nội dung lấy nguyên văn từ sách [*Machine Learning cơ bản*](https://github.com/tiepvupsu/ebookMLCB) của Vũ Hữu Tiệp (giấy phép CC BY-SA 4.0): phần *Kiến thức toán cơ bản* (chương 1–4), phần *Tối ưu lồi* (chương 23–25) và phụ lục A. Số chương, công thức và hình giữ như trong sách để tiện tra cứu.

| Chương | Bài đọc | Code từ đầu | Dùng thư viện | Ví dụ áp dụng | Mô phỏng |
| --- | --- | --- | --- | --- | --- |
| 1 | [Ôn tập Đại số tuyến tính](math/ch01_vi) | [NumPy](math/ch01_code_vi) | [NumPy, SciPy](math/ch01_lib_vi) | [Xếp hạng nhân vật bằng vector riêng](math/ch01_app_vi) | [Ma trận là một phép biến đổi](math/ch01_viz_vi) |
| 2 | [Giải tích ma trận](math/ch02_vi) | [NumPy](math/ch02_code_vi) | [PyTorch, SymPy](math/ch02_lib_vi) | [Phương pháp Newton cho hồi quy logistic](math/ch02_app_vi) | [Gradient và đạo hàm số](math/ch02_viz_vi) |
| 3 | [Ôn tập Xác suất](math/ch03_vi) | [NumPy](math/ch03_code_vi) | [scipy.stats](math/ch03_lib_vi) | [Đọc dữ liệu giá nhà bằng xác suất](math/ch03_app_vi) | [Phân phối chuẩn hai chiều](math/ch03_viz_vi) |
| 4 | [Ước lượng tham số mô hình](math/ch04_vi) | [NumPy](math/ch04_code_vi) | [SciPy, scikit-learn](math/ch04_lib_vi) | [MLE và MAP trên chữ số MNIST](math/ch04_app_vi) | [Tung đồng xu: MLE và MAP](math/ch04_viz_vi) |
| 23 | [Tập lồi và hàm lồi](math/ch23_vi) | [NumPy](math/ch23_code_vi) | [SciPy, SymPy](math/ch23_lib_vi) | [Hoa Iris có tách được tuyến tính?](math/ch23_app_vi) | [Tập lồi và hàm lồi](math/ch23_viz_vi) |
| 24 | [Bài toán tối ưu lồi](math/ch24_vi) | [Phương pháp hàm chắn](math/ch24_code_vi) | [scipy.optimize](math/ch24_lib_vi) | [Hồi quy bằng quy hoạch tuyến tính](math/ch24_app_vi) | [LP và QP](math/ch24_viz_vi) |
| 25 | [Đối ngẫu](math/ch25_vi) | [NumPy](math/ch25_code_vi) | [scipy.optimize](math/ch25_lib_vi) | [Đối ngẫu trong SVM](math/ch25_app_vi) | [Hàm đối ngẫu Lagrange](math/ch25_viz_vi) |
| A | [Phương pháp nhân tử Lagrange](math/appendix_a_vi) | [SymPy, SciPy, NumPy](math/appendix_a_code_vi) | | | [Nhân tử Lagrange](math/appendix_a_viz_vi) |

## Mỗi chương gồm bốn lớp

| Lớp | Bạn sẽ làm gì |
| --- | --- |
| **1. Bài đọc** | Định nghĩa, tính chất, chứng minh, như trong sách |
| **2. Code từ đầu** | Biến mỗi định nghĩa và định lý thành một phép kiểm tra bằng số, chỉ với NumPy |
| **3. Dùng thư viện** | Các hàm có sẵn (NumPy, SciPy, SymPy, PyTorch) và những cái bẫy số học khi dùng chúng |
| **4. Ví dụ áp dụng** | Thấy phần toán đó xuất hiện ở đâu trong một bài toán thật |

## Học theo thứ tự nào

- **Trước khoá Học máy:** chương 1–4. Chương 2 (gradient) dùng ngay ở hồi quy tuyến tính và gradient descent; chương 3–4 dùng ở hồi quy logistic, softmax và naive Bayes.
- **Trước phần máy vector hỗ trợ ([chương 26–29](ml/ch26_vi) của khoá Học máy):** chương 23–25 và phụ lục A.
- Không cần học hết trước khi bắt đầu khoá Học máy. Các bài của khoá Học máy có link ngược về đúng mục toán cần dùng.

## Cần chuẩn bị gì

Toán phổ thông (đạo hàm, tích phân một biến) và Python cơ bản.
