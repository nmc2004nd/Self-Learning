# PyTorch nâng cao: cấu trúc mã nguồn

> Ghi chú của người biên soạn: trang này dịch từ ghi chú *Pytorch 代码库结构拆解* trong *dl_note* của **Harley Zhang** ([bản gốc](https://github.com/harleyszhang/dl_note/blob/main/2-deep_learning_basic/pytorch_code/pytorch代码库结构拆解.md), giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)), viết tháng 3/2025. Mục 2 của bài [Tổng quan kiến trúc](tools/pt_arch_vi) chép lại gần nguyên văn mục 1–2 ở đây, nên trên site chỉ giữ một bản tại trang này. Hai ảnh tiếng Trung (sơ đồ số dòng mã theo thư mục, bảng thư mục `native/`) được gõ lại; các khối code soi bản PyTorch 2.14 đã cài.

## 1 Cấu trúc kho mã PyTorch

Mã nguồn PyTorch 2.x chia thành nhiều thư mục cấp cao, mỗi thư mục một chức năng. Lệnh `tree -L 1 -d` liệt kê các thư mục con cấp 1:

```text
├── android        # biên dịch, triển khai PyTorch trên Android
├── aten           # ATen ("A Tensor"): thư viện tensor và toán tử lõi, cài đặt kernel
├── benchmarks     # script và công cụ đo hiệu năng
├── binaries       # các chương trình thực thi nhỏ (công cụ, benchmark C++)
├── build          # thư mục đầu ra và tệp trung gian khi biên dịch
├── c10            # thư viện lõi: TensorImpl, Storage, Dispatcher, thiết bị...
├── caffe2         # phần còn lại của Caffe2 (chủ yếu là CMake và vài tiện ích)
├── cmake          # script và mô-đun CMake
├── docs           # tài liệu
├── functorch      # biến đổi hàm (vmap, grad...), nay thành torch.func
├── mypy_plugins   # plugin cho mypy (kiểm tra kiểu tĩnh)
├── scripts        # script phụ trợ
├── test           # kiểm thử đơn vị và tích hợp
├── third_party    # mã nguồn thư viện bên thứ ba (submodule)
├── tools          # công cụ build, sinh mã (autograd, jit, onnx), coverage, linter
├── torch          # mã nguồn gói Python torch (và torch/csrc: phần C++ gắn với Python)
├── torch.egg-info
└── torchgen       # sinh mã toán tử từ native_functions.yaml: đăng ký, hàm kích thước...
```

Có nhiều thư mục, nhưng với người phát triển, quan trọng nhất là vài thư mục sau:

1. `c10/`: **thư viện lõi** của PyTorch.
   - Cung cấp các **thành phần dùng chung trên mọi nền tảng**: siêu dữ liệu và lưu trữ của tensor, cơ chế điều phối (dispatcher), stream, event.
   - Là tầng lõi thống nhất được tách ra khi PyTorch và Caffe2 hợp nhất; tên "c10" đọc là "C-ten", chơi chữ giữa "Caffe" và "ATen".
   - Bản thân c10 **không chứa cài đặt toán tử** nào; nó cung cấp các cấu trúc dữ liệu và cơ chế nền (tensor, thiết bị, dispatcher) mà ATen và autograd dựa vào. Chi tiết ở bài [c10](tools/pt_c10_vi).
2. `aten/`: thư viện ATen ("A Tensor"), **thư viện phép toán tensor lõi** (C++): định nghĩa và cài đặt tensor cùng các phép toán trên nó. ATen không chứa logic vi phân tự động; nó tập trung vào **kernel**: tạo, đánh chỉ số, toán học, các phép trên tensor. Các thư mục con chính của `aten/src/ATen`:
   - `ATen/core`: chức năng lõi của ATen (một phần đang dần chuyển lên `c10`);
   - `ATen/native`: cài đặt `native` của từng toán tử; thêm toán tử mới thường đặt ở đây. Chia theo thiết bị:
     - `native/cpu`: các kernel được biên dịch nhiều lần với tập lệnh khác nhau (AVX2, AVX-512...) để chọn lúc chạy theo CPU; cài đặt CPU chung nằm ngay ở `native/`;
     - `native/cuda`: cài đặt CUDA;
     - `native/sparse`: tensor thưa định dạng COO trên CPU và CUDA;
     - `native/quantized`: toán tử của tensor lượng tử hoá (QTensor).
3. `torch/`: thư viện PyTorch thật sự; trừ `csrc`, phần còn lại là mô-đun Python, theo cấu trúc frontend Python.
   - `csrc`: các tệp C++ tạo nên thư viện PyTorch, trộn lẫn **mã ràng buộc Python** và nhiều cài đặt C++ tầng thấp; tệp ràng buộc thường có tiền tố `python_`.
     - `jit`: trình biên dịch và frontend TorchScript JIT, dùng để tạo mô hình tuần tự hoá, tối ưu được từ code PyTorch (đã lỗi thời, thay bằng `torch.compile` và `torch.export`).
     - `autograd`: **cài đặt vi phân tự động**. Nguyên tắc thiết kế là mỗi kiểu dữ liệu quan trọng có hai bản: kiểu C++ và kiểu đối tượng Python. Ví dụ biến (Variable) có kiểu C++ trong `variable.h` và kiểu Python `THPVariable` trong `python_variable.h`.
     - `api`: frontend C++ của PyTorch (LibTorch).
     - `distributed`: hỗ trợ huấn luyện phân tán.
4. `tools/`: các script sinh mã mà thư viện PyTorch dùng (ví dụ sinh công thức đạo hàm từ `derivatives.yaml`).

> **Biên tập:** Bản gốc viết c10 "chủ yếu cung cấp các mô-đun, lớp trừu tượng hỗ trợ cơ chế vi phân tự động của tensor". Vi phân tự động nằm ở `torch/csrc/autograd`; c10 không biết gì về autograd ngoài một con trỏ `autograd_meta_` trong `TensorImpl` (bài [TensorImpl](tools/pt_tensorimpl_vi)). Bản gốc mô tả `native/cpu` là "không thật sự là cài đặt CPU"; ý đúng là các tệp trong đó được biên dịch nhiều lần với các cờ tập lệnh khác nhau (CPU dispatch).

Sơ đồ quy mô mã nguồn trong bản gốc (theo số dòng, khoảng 2,13 triệu dòng cho cả kho) cho thấy ba khối lớn nhất là `aten/src/ATen` (khoảng 470 nghìn dòng, phần lớn trong `native/`), `torch/` (khoảng 620 nghìn dòng: `csrc` C++ và các mô-đun Python như `_inductor`, `distributed`, `nn`, `onnx`), và `test/` (khoảng 360 nghìn dòng); `c10/` chỉ khoảng 50 nghìn dòng. Kích thước gói đã cài cũng phản ánh điều này:

<!-- run -->
```python
import os, torch
root = os.path.dirname(torch.__file__)
def size_of(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 2**20
subs = sorted(((size_of(os.path.join(root, d)), d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))), reverse=True)
for mb, d in subs[:10]:
    print(f"torch/{d:<14} {mb:8.1f} MB")
heads = os.path.join(root, "include")
for sub in ["ATen", "c10", "torch"]:
    n = sum(f.endswith(".h") for _, _, fs in os.walk(os.path.join(heads, sub)) for f in fs)
    print(f"include/{sub:<6}: {n} tệp header")
```

<!-- output -->
```output
torch/lib               454.2 MB
torch/test               84.7 MB
torch/bin                51.0 MB
torch/include            37.7 MB
torch/_inductor          23.7 MB
torch/distributed        11.5 MB
torch/_dynamo            10.7 MB
torch/testing            10.5 MB
torch/utils               4.7 MB
torch/fx                  4.3 MB
include/ATen  : 7742 tệp header
include/c10   : 232 tệp header
include/torch : 1057 tệp header
```
<!-- /output -->

`torch/lib` (các thư viện động C++) chiếm phần lớn dung lượng; `torch/include` chứa header của ATen, c10 và torch, để biên dịch phần mở rộng C++ (`torch.utils.cpp_extension`) mà không cần mã nguồn PyTorch.

## 2 Thư viện lõi c10

`c10` là **thư viện lõi** của PyTorch, gồm các mô-đun con:

- `c10/core/`: thành phần lõi, định nghĩa **cấu trúc dữ liệu và cơ chế lõi**: `TensorImpl` (cài đặt tầng thấp của tensor), `Storage` (lưu trữ), `DispatchKey` (khoá điều phối; bộ điều phối `Dispatcher` thì nằm ở `ATen/core/dispatch`), kiểu thiết bị `Device`, siêu thông tin kiểu `TypeMeta`...
- `c10/util/`: tiện ích C++ dùng chung: con trỏ thông minh `intrusive_ptr`, bộ bọc con trỏ `UniqueVoidPtr`, xử lý ngoại lệ `Exception`, log và công cụ lập trình meta.
- `c10/macros/`: macro cấu hình biên dịch, ví dụ `cmake_macros.h` sinh theo hệ điều hành và tuỳ chọn biên dịch, các macro điều khiển xuất ký hiệu `C10_API`, `TORCH_API`.
- `c10/cuda/`, `c10/hip/`, `c10/metal/`, `c10/xpu/`...: mã hỗ trợ từng nền tảng, giúp c10 thích ứng với phần cứng khác nhau:
  - `c10/cuda`: khởi tạo backend CUDA, quản lý stream, bộ cấp phát bộ nhớ;
  - `c10/hip`: tương tự cho HIP của AMD;
  - `c10/metal`: backend Metal của Apple;
  - `c10/xpu`: GPU Intel (XPU).
- `c10/mobile/`: hỗ trợ cắt gọn, tối ưu PyTorch cho thiết bị di động, nhúng.
- `c10/test/`: kiểm thử đơn vị của c10.

> **Biên tập:** Bản gốc xếp `Dispatcher` vào `c10/core`; trong mã nguồn, `DispatchKey`, `DispatchKeySet` nằm ở `c10/core`, còn lớp `Dispatcher` nằm ở `aten/src/ATen/core/dispatch/Dispatcher.h`.

## 3 Mô-đun ATen

`aten/` (A Tensor Library) là thành phần lõi, cài đặt phép toán tensor, điều phối toán tử tới các backend, và cài đặt cụ thể của toán tử trên từng thiết bị (`aten/src/ATen/native/`).

Các thư mục con chính của `aten/src/ATen/`:

- `core/`: thư viện hàm lõi (đang dần chuyển sang c10), định nghĩa cấu trúc dữ liệu, hệ thống kiểu và API lõi của tensor; cả bộ điều phối (`core/dispatch`).
- `native/`: thư viện toán tử gốc: cài đặt thật của toán tử (tích chập, nhân ma trận...) cho từng backend.
- `cuda/`, `mps/`, `vulkan/`, `xpu/`...: phần hỗ trợ riêng cho GPU NVIDIA, Apple, Vulkan (di động), Intel.
- `quantized/`: phần hỗ trợ tensor lượng tử hoá.

> **Biên tập:** Bản gốc viết ATen "chịu trách nhiệm vi phân tự động (Autograd)" và liệt kê `autograd/` (Variable, Function) như một thư mục con của `aten/src/ATen`. Engine autograd và Variable nằm ở `torch/csrc/autograd`; phía ATen chỉ có các kernel và (do `torchgen` sinh ra) phần đăng ký toán tử cho khoá điều phối Autograd.

Mô-đun con `aten/src/ATen/core`:

- vai trò: cấu trúc dữ liệu lõi, hệ thống kiểu và giao diện nền của tensor;
- tệp chính:
  - `Tensor.h` (đúng hơn là `TensorBody.h`, sinh tự động): định nghĩa lớp `at::Tensor`;
  - `TensorBase.h`: lớp cơ sở của Tensor, giao diện nhẹ;
  - `ScalarType.h` (nay ở `c10/core`): enum các kiểu dữ liệu (`dtype`).

Mô-đun con `aten/src/ATen/native`, các thư mục và tệp chính (bản gốc gõ dạng ảnh bảng tiếng Trung):

| Tệp / thư mục | Vai trò |
| --- | --- |
| `native/` | toán tử CPU và cài đặt chung cho mọi thiết bị (ví dụ `Convolution.cpp`, `LinearAlgebra.cpp`, `Loss.cpp`) |
| `native/native_functions.yaml` | danh sách mọi toán tử: chữ ký và kernel cho từng khoá điều phối; `torchgen` đọc tệp này để sinh mã |
| `native/cpu/` | kernel CPU vector hoá, biên dịch cho nhiều tập lệnh |
| `native/cuda/` | kernel CUDA |
| `native/mps/` | Apple Metal (backend MPS) |
| `native/mkldnn/` | toán tử dùng oneDNN (MKL-DNN) |
| `native/sparse/` | tensor thưa |
| `native/quantized/` | tensor lượng tử hoá |
| `native/vulkan/` | backend tính toán Vulkan |
| `native/transformers/` | attention (`scaled_dot_product_attention`, FlashAttention...) |

> **Biên tập:** Bản gốc liệt kê `native/Composite/` ("toán tử tổ hợp") và `native/meta/` ("toán tử meta để suy kích thước") như thư mục con. Đó không phải thư mục: `CompositeImplicitAutograd`, `CompositeExplicitAutograd` và `Meta` là **khoá điều phối** ghi trong `native_functions.yaml`, quyết định một toán tử được cài bằng cách ghép các toán tử khác hay có kernel riêng, và có hàm suy kích thước (chạy được trên tensor `device="meta"` không có dữ liệu).

Có thể xem ngay từ bản đã cài: số toán tử đã đăng ký, chữ ký của một toán tử, và các khoá điều phối có kernel cho nó:

<!-- run -->
```python
ops = torch._C._dispatch_get_all_op_names()
aten = [o for o in ops if o.startswith("aten::")]
print("số toán tử đã đăng ký:", len(ops), "| trong đó aten::", len(aten))
print(torch.ops.aten.add.Tensor._schema)
dump = torch._C._dispatch_dump("aten::add.Tensor")
keys = [l.split(":")[0] for l in dump.splitlines() if ": registered at" in l and not l.startswith(("name", "schema", "debug"))]
print("khoá có kernel cho aten::add.Tensor:", keys)
x = torch.randn(2, 3, device="meta")            # tensor không có dữ liệu: chỉ chạy hàm suy kích thước
print("trên device meta:", (x @ torch.randn(3, 5, device="meta")).shape)
```

<!-- output -->
```output
số toán tử đã đăng ký: 3607 | trong đó aten:: 3113
aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
khoá có kernel cho aten::add.Tensor: ['MkldnnCPU', 'ZeroTensor', 'Tracer', 'FuncTorchBatched', 'Batched', 'CPU', 'Meta', 'Meta (inactive)', 'SparseCPU', 'SparseMeta', 'SparseCsrCPU', 'SparseCsrMeta', 'NestedTensorCPU', 'NestedTensorHPU', 'Autograd[alias]', 'CompositeExplicitAutogradNonFunctional[alias]']
trên device meta: torch.Size([2, 5])
```
<!-- /output -->

## Tài liệu tham khảo

- [万字综述，核心开发者全面解读PyTorch内部张量机制](https://mp.weixin.qq.com/s/8J-vsOukt7xwWQFtwnSnWw) (bản dịch tiếng Trung của [*PyTorch internals*](http://blog.ezyang.com/2019/05/pytorch-internals/), Edward Yang)
- [PyTorch codebase structure (CONTRIBUTING.md)](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#codebase-structure)
- [native_functions.yaml README](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md)
