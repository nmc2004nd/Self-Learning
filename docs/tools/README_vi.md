# Công cụ lập trình

Khoá này gom các ghi chú về **công cụ** mà người làm học sâu dùng hằng ngày: đăng nhập máy chủ GPU từ xa, Python, C++ và thư viện mô hình. Phần lý thuyết học sâu nằm ở các khoá [Học sâu](dl/README_vi) và [Ghi chú học sâu](dlnote/README_vi); ở đây chỉ có công cụ.

## Nguồn và cách biên soạn

Nội dung lấy từ phần ghi chú công cụ trong kho [harleyszhang/dl_note](https://github.com/harleyszhang/dl_note) của **Harley Zhang**, giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0); phần dịch và phần viết thêm trên site cũng theo Apache-2.0. Cách biên soạn giống khoá Ghi chú học sâu:

- **Dịch có biên tập.** Chỗ bản gốc sai hoặc lỗi thời được sửa trong bài, mỗi chỗ có ô **Biên tập** ghi bản gốc viết gì và vì sao sửa. Ở phần này, nhiều chỗ sửa là lỗi thật trong code: chương trình C++ không biên dịch được, bản đa tiến trình trả về ma trận toàn số 0, hướng dẫn SSH bảo gửi khoá riêng lên máy chủ.
- **Code chạy thật.** Mọi đoạn code trên các trang Code demo và trang HuggingFace đều được chạy, kết quả in ngay dưới code.
- **Cập nhật.** Bài HuggingFace viết lại theo transformers 5; bài SSH theo OpenSSH 9.x.

## Nội dung

| Phần | Trang | Ghi chú |
| --- | --- | --- |
| Dòng lệnh và máy chủ | [SSH: đăng nhập máy chủ từ xa](tools/ssh_vi) | Khoá Ed25519, `ssh-copy-id`, quyền thư mục, `~/.ssh/config` với `ProxyJump`, VS Code Remote-SSH, tmux và chuyển tiếp cổng |
| Lập trình Python | Sơ đồ tư duy: [kiểu dữ liệu và tính năng nâng cao](tools/mm_py_data_vi), [lập trình hàm](tools/mm_py_func_vi), [hướng đối tượng cơ bản](tools/mm_py_oop_vi), [hướng đối tượng nâng cao](tools/mm_py_oop2_vi) | Chuyển từ bốn tệp XMind |
| | [Code demo Python](tools/py_demo_vi) | Generator, decorator, OOP, đa luồng và đa tiến trình, tích chập NumPy so với PyTorch, Gradio, API OpenAI |
| Lập trình C++ | [Sơ đồ tư duy: đa luồng](tools/mm_cpp_thread_vi), [Code demo C++](tools/cpp_demo_vi) | Sáu chương trình đã sửa, biên dịch bằng GCC 13 |
| PyTorch cơ bản | [Cấu trúc dữ liệu tensor](tools/pt_tensor_vi), [thao tác cấu trúc](tools/pt_ops_vi), [phép toán](tools/pt_math_vi) | Cập nhật từ PyTorch 1.7 lên 2.14, mọi ví dụ chạy thật; thêm tích chập chuyển vị, broadcast, einsum |
| PyTorch nâng cao | [Tổng quan kiến trúc](tools/pt_arch_vi), [cấu trúc mã nguồn](tools/pt_codebase_vi), [TensorImpl](tools/pt_tensorimpl_vi), [c10: Stream và Event](tools/pt_c10_vi), [quá trình build](tools/pt_build_vi) | Đọc mã nguồn PyTorch; ảnh chụp màn hình được thay bằng code soi trực tiếp bản đã cài (dispatcher, storage, autograd). Một khối về CUDA stream chờ chạy trên máy có GPU |
| Thư viện học sâu | [HuggingFace Transformers](tools/hf_vi) | Tokenizer, attention mask, pipeline, AutoClass, sinh văn bản với mô hình hội thoại |

