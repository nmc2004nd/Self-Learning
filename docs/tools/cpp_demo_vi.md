# Code demo C++

> Ghi chú của người biên soạn: trang này chạy lại sáu chương trình trong thư mục [`1-math_ml_basic/src/cpp`](https://github.com/harleyszhang/dl_note/tree/main/1-math_ml_basic/src/cpp) của *dl_note* (**Harley Zhang**, giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)). Cả sáu đều có vấn đề ở bản gốc: một tệp không biên dịch được, một tệp tìm sai, một tệp có hàm huỷ ném ngoại lệ, và vài tệp không minh hoạ được điều tên tệp hứa hẹn. Mỗi mục ghi bản gốc sai ở đâu (ô **Biên tập**), rồi đưa bản sửa, kèm kết quả biên dịch và chạy thật. Mã nguồn bản sửa nằm ở `docs/code/tools/cpp/`.

## Cách dùng trang này

Mỗi chương trình được biên dịch bằng `g++ -std=c++20 -O2 -Wall -Wextra -pthread` (GCC 13.3) rồi chạy; khối Python ngắn bên dưới mỗi chương trình chỉ làm việc đó. Trên máy bạn, chạy thẳng:

```bash
g++ -std=c++20 -O2 -Wall -Wextra -pthread binary_search.cpp -o binary_search && ./binary_search
```

<!-- run -->
```python
import os, subprocess, tempfile
from pathlib import Path

CPP_DIR = Path(os.environ["CODE_FIG_DIR"]).parents[2] / "code" / "tools" / "cpp"   # docs/code/tools/cpp
BUILD = Path(tempfile.mkdtemp(prefix="cpp_demo_"))

def run_cpp(name):
    """Biên dịch docs/code/tools/cpp/<name>.cpp, in cảnh báo (nếu có) rồi chạy."""
    exe = BUILD / name
    cmd = ["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-pthread", str(CPP_DIR / f"{name}.cpp"), "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"$ g++ ... {name}.cpp:", "biên dịch OK, không cảnh báo" if r.returncode == 0 and not r.stderr else r.stderr)
    if r.returncode == 0:
        out = subprocess.run([str(exe)], capture_output=True, text=True, cwd=BUILD)
        print(out.stdout + out.stderr, end="")

print(subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0])
```

<!-- output -->
```output
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
```
<!-- /output -->

## 1. Tìm kiếm nhị phân

> **Biên tập:** Hàm `binary_search` của bản gốc đổi chiều hai nhánh: khi `arr[mid] < target` nó tìm sang **trái** (`right = mid - 1`), còn nhánh kia chỉ tăng `left` thêm 1 thay vì đặt `right = mid - 1`. Hàm `main` của bản gốc chỉ thử `target = 5`, vừa đúng phần tử giữa mảng nên lần so sánh đầu tiên đã trúng và lỗi không lộ ra. Bản dưới giữ hàm gốc để so sánh trên nhiều giá trị. Ngoài ra vector nên truyền bằng tham chiếu `const` (bản gốc chép cả vector mỗi lần gọi), và `mid` tính bằng `left + (right - left) / 2` để không tràn số với mảng rất lớn.

<!-- cpp: binary_search -->
```cpp
#include <algorithm>
#include <iostream>
#include <vector>

// Bản gốc: đổi chiều hai nhánh và chỉ tăng left thêm 1.
int binary_search_original(std::vector<int> arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left <= right) {
        auto mid = (left + right) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) right = mid - 1;   // sai: phải tìm bên phải
        else left = left + 1;                          // sai: phải là right = mid - 1
    }
    return -1;
}

// Bản sửa: truyền tham chiếu const (không chép vector), tính mid không tràn số.
int binary_search(const std::vector<int>& arr, int target) {
    int left = 0, right = static_cast<int>(arr.size()) - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

int main() {
    std::vector<int> arr{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::cout << "target  gốc  sửa  std::lower_bound\n";
    for (int target : {1, 2, 5, 7, 10, 11}) {
        auto it = std::lower_bound(arr.begin(), arr.end(), target);
        int std_idx = (it != arr.end() && *it == target) ? int(it - arr.begin()) : -1;
        std::cout << "  " << target << "\t" << binary_search_original(arr, target) << "\t"
                  << binary_search(arr, target) << "\t" << std_idx << "\n";
    }
}
```

<!-- run -->
```python
run_cpp("binary_search")
```

<!-- output -->
```output
$ g++ ... binary_search.cpp: biên dịch OK, không cảnh báo
target  gốc  sửa  std::lower_bound
  1	-1	0	0
  2	-1	1	1
  5	4	4	4
  7	-1	6	6
  10	-1	9	9
  11	-1	-1	-1
```
<!-- /output -->

Hàm gốc chỉ tìm thấy đúng số 5; mọi giá trị khác trả về -1. Trong code thật, dùng `std::lower_bound` hoặc `std::binary_search` của thư viện chuẩn.

## 2. Nạp chồng toán tử

> **Biên tập:** Tệp gốc tên `class_copy_move.cpp` nhưng nội dung là nạp chồng toán tử cho số phức, không có hàm tạo sao chép hay di chuyển nào; ở đây đổi tên thành `complex_operators.cpp`. Bản gốc tính `c4 = c1 - c2` mà không in ra (GCC cảnh báo biến không dùng); nếu in thì toán tử `<<` của bản gốc cho `(-2+-2i)` vì luôn chèn dấu `+`.

<!-- cpp: complex_operators -->
```cpp
#include <iostream>

class Complex {
public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}

    // Nạp chồng toán tử +, -: hàm thành viên const, không sửa đối tượng hiện tại
    Complex operator+(const Complex& other) const { return {real + other.real, imag + other.imag}; }
    Complex operator-(const Complex& other) const { return {real - other.real, imag - other.imag}; }

    // Hàm bạn (friend) nạp chồng <<, để in bằng std::cout
    friend std::ostream& operator<<(std::ostream& os, const Complex& c) {
        return os << "(" << c.real << (c.imag < 0 ? "" : "+") << c.imag << "i)";
    }

private:
    double real, imag;
};

int main() {
    Complex c1(1, 2), c2(3, 4);
    std::cout << "c1 = " << c1 << ", c2 = " << c2 << "\n";
    std::cout << "c1 + c2 = " << c1 + c2 << "\n";
    std::cout << "c1 - c2 = " << c1 - c2 << "\n";
    std::cout << "c1 + 5  = " << c1 + 5 << "   (5 tự đổi thành Complex(5, 0) qua hàm tạo)\n";
}
```

<!-- run -->
```python
run_cpp("complex_operators")
```

<!-- output -->
```output
$ g++ ... complex_operators.cpp: biên dịch OK, không cảnh báo
c1 = (1+2i), c2 = (3+4i)
c1 + c2 = (4+6i)
c1 - c2 = (-2-2i)
c1 + 5  = (6+2i)   (5 tự đổi thành Complex(5, 0) qua hàm tạo)
```
<!-- /output -->

## 3. Sao chép và di chuyển

> **Biên tập:** Tệp gốc `cpp_basic.cpp` dùng một lớp in ra khi hàm tạo mặc định, sao chép, di chuyển được gọi, rồi `push_back` hai lần và di chuyển cả vector. Chương trình chạy được, nhưng bỏ qua chi tiết thú vị nhất: lần `push_back` thứ hai làm vector cấp phát lại bộ nhớ, và phần tử cũ bị **sao chép** chứ không di chuyển, vì hàm tạo di chuyển không có `noexcept`. Bản dưới so sánh hai lớp giống hệt nhau, chỉ khác `noexcept`.

<!-- cpp: copy_move -->
```cpp
#include <iostream>
#include <vector>

// Hàm tạo di chuyển KHÔNG có noexcept (như bản gốc)
struct Widget {
    Widget() { std::cout << "  default\n"; }
    Widget(const Widget&) { std::cout << "  copy\n"; }
    Widget(Widget&&) { std::cout << "  move\n"; }
};

// Giống hệt, chỉ thêm noexcept cho hàm tạo di chuyển
struct WidgetNoexcept {
    WidgetNoexcept() { std::cout << "  default\n"; }
    WidgetNoexcept(const WidgetNoexcept&) { std::cout << "  copy\n"; }
    WidgetNoexcept(WidgetNoexcept&&) noexcept { std::cout << "  move\n"; }
};

template <typename T>
void demo(const char* name) {
    std::cout << name << ":\n";
    std::vector<T> v1;
    v1.push_back(T());          // tạo đối tượng tạm rồi di chuyển vào vector
    std::cout << " -- push_back thứ hai (vector phải cấp phát lại và chuyển phần tử cũ):\n";
    v1.push_back(T());
    std::cout << " -- std::vector<T> v2(std::move(v1)):\n";
    std::vector<T> v2(std::move(v1));   // chỉ chuyển con trỏ bộ nhớ, không chạm vào từng phần tử
    std::cout << "  v1.size() = " << v1.size() << ", v2.size() = " << v2.size() << "\n\n";
}

int main() {
    demo<Widget>("Widget (move không noexcept)");
    demo<WidgetNoexcept>("WidgetNoexcept");
}
```

<!-- run -->
```python
run_cpp("copy_move")
```

<!-- output -->
```output
$ g++ ... copy_move.cpp: biên dịch OK, không cảnh báo
Widget (move không noexcept):
  default
  move
 -- push_back thứ hai (vector phải cấp phát lại và chuyển phần tử cũ):
  default
  move
  copy
 -- std::vector<T> v2(std::move(v1)):
  v1.size() = 0, v2.size() = 2

WidgetNoexcept:
  default
  move
 -- push_back thứ hai (vector phải cấp phát lại và chuyển phần tử cũ):
  default
  move
  move
 -- std::vector<T> v2(std::move(v1)):
  v1.size() = 0, v2.size() = 2
```
<!-- /output -->

Với `Widget`, dòng `copy` xuất hiện khi vector chuyển phần tử cũ sang vùng nhớ mới; với `WidgetNoexcept` dòng đó thành `move`. Lý do: `std::vector` đảm bảo nếu cấp phát lại bị ngoại lệ thì vector cũ còn nguyên (strong exception guarantee). Nếu hàm tạo di chuyển có thể ném ngoại lệ, di chuyển dở dang sẽ làm hỏng vector cũ, nên vector chọn sao chép cho an toàn (`std::move_if_noexcept`). Với kiểu dữ liệu lớn (như một tensor), khác biệt này là chép toàn bộ dữ liệu so với chép một con trỏ. Còn `std::vector<T> v2(std::move(v1))` không gọi hàm tạo nào của phần tử: nó chỉ lấy con trỏ vùng nhớ của `v1`.

## 4. RAII và chuyển tiếp hoàn hảo

> **Biên tập:** Hàm huỷ của `fileWrapper` trong bản gốc **ném ngoại lệ** khi tệp còn mở (`throw std::runtime_error("File is need to close")`), và dòng `m_file.close()` nằm sau `throw` nên không bao giờ chạy. Từ C++11 hàm huỷ mặc định là `noexcept`, nên ném ngoại lệ trong đó gọi `std::terminate`: chương trình gốc biên dịch được (GCC cảnh báo "`throw` will always call `terminate`") nhưng chạy tới cuối khối `try` thì dừng với `terminate called after throwing an instance of 'std::runtime_error'`. Đây là điều ngược với RAII: hàm huỷ phải dọn dẹp lặng lẽ. Phần chuyển tiếp hoàn hảo của bản gốc chỉ in giá trị, nên không thấy được tác dụng của `std::forward`; bản dưới gọi hai hàm nạp chồng cho lvalue và rvalue.

<!-- cpp: file_wrapper -->
```cpp
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

// RAII: hàm tạo mở tệp, hàm huỷ tự đóng tệp, kể cả khi có ngoại lệ.
class FileWrapper {
public:
    explicit FileWrapper(const std::string& filename) : m_file(filename, std::ios::out | std::ios::app) {
        if (!m_file.is_open()) throw std::runtime_error("không mở được tệp: " + filename);
        std::cout << "mở " << filename << "\n";
    }
    ~FileWrapper() {            // hàm huỷ không được ném ngoại lệ
        if (m_file.is_open()) m_file.close();
        std::cout << "hàm huỷ: đã đóng tệp\n";
    }
    FileWrapper(const FileWrapper&) = delete;             // một tệp, một chủ sở hữu
    FileWrapper& operator=(const FileWrapper&) = delete;

    void writeLine(const std::string& line) { m_file << line << '\n'; }

private:
    std::fstream m_file;
};

// Chuyển tiếp hoàn hảo: giữ nguyên lvalue/rvalue khi truyền tiếp
void sink(const std::string& s) { std::cout << "  sink(const&)  nhận lvalue: " << s << "\n"; }
void sink(std::string&& s)      { std::cout << "  sink(&&)      nhận rvalue: " << s << "\n"; }

template <typename T>
void wrapper(T&& val) { sink(std::forward<T>(val)); }

int main() {
    std::string name = "lvalue";
    wrapper(name);                     // T = std::string&  → gọi bản const&
    wrapper(std::string("rvalue"));    // T = std::string   → gọi bản &&

    try {
        FileWrapper fw("test.txt");
        fw.writeLine("Hello World");
        throw std::runtime_error("lỗi giữa chừng");
    } catch (const std::exception& e) {
        std::cout << "bắt được: " << e.what() << " (tệp đã được đóng trước khi vào catch)\n";
    }
}
```

<!-- run -->
```python
run_cpp("file_wrapper")
```

<!-- output -->
```output
$ g++ ... file_wrapper.cpp: biên dịch OK, không cảnh báo
  sink(const&)  nhận lvalue: lvalue
  sink(&&)      nhận rvalue: rvalue
mở test.txt
hàm huỷ: đã đóng tệp
bắt được: lỗi giữa chừng (tệp đã được đóng trước khi vào catch)
```
<!-- /output -->

## 5. Đa luồng

> **Biên tập:** Chương trình gốc `multi_thread_demo.cpp` tạo một luồng in lời chào và một luồng gọi `std::call_once`. Vì chỉ một luồng gọi nên không thấy được tác dụng "chỉ chạy một lần". Bản dưới cho bốn luồng cùng gọi `call_once`, và thêm ví dụ tranh chấp dữ liệu (data race) với ba cách tăng biến đếm: biến thường, `std::atomic`, và khoá `std::mutex`. Xem thêm [sơ đồ C++: đa luồng](tools/mm_cpp_thread_vi).

<!-- cpp: threads -->
```cpp
#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

std::once_flag init_flag;
void init_resource() { std::cout << "khởi tạo tài nguyên (chỉ một lần)\n"; }

void worker(int id, std::mutex& io) {
    std::call_once(init_flag, init_resource);       // 4 luồng cùng gọi, chỉ một luồng chạy hàm
    std::lock_guard<std::mutex> lock(io);           // tránh các dòng in chen vào nhau
    std::cout << "luồng " << id << " chạy xong\n";
}

int main() {
    std::mutex io;
    std::vector<std::thread> ts;
    for (int i = 0; i < 4; ++i) ts.emplace_back(worker, i, std::ref(io));
    for (auto& t : ts) if (t.joinable()) t.join();

    // Tranh chấp dữ liệu: 4 luồng cùng tăng một biến 1 triệu lần
    const int N = 1'000'000;
    volatile long plain = 0;   // volatile: buộc mỗi lần tăng là đọc-cộng-ghi thật, để thấy tranh chấp (với -O2, trình biên dịch gộp vòng lặp lại)
    std::atomic<long> atomic_counter{0};
    long locked = 0;
    std::mutex m;
    auto run = [&](auto body) {
        std::vector<std::thread> v;
        for (int i = 0; i < 4; ++i) v.emplace_back([&] { for (int k = 0; k < N; ++k) body(); });
        for (auto& t : v) t.join();
    };
    run([&] { plain = plain + 1; });                               // data race: kết quả sai, hành vi không xác định
    run([&] { ++atomic_counter; });                                // phép tăng nguyên tử
    run([&] { std::lock_guard<std::mutex> g(m); ++locked; });      // khoá mutex
    std::cout << "mong đợi " << 4L * N << " | biến thường " << plain
              << " | atomic " << atomic_counter << " | mutex " << locked << "\n";
}
```

<!-- run -->
```python
run_cpp("threads")
```

<!-- output -->
```output
$ g++ ... threads.cpp: biên dịch OK, không cảnh báo
khởi tạo tài nguyên (chỉ một lần)
luồng 0 chạy xong
luồng 3 chạy xong
luồng 1 chạy xong
luồng 2 chạy xong
mong đợi 4000000 | biến thường 1262806 | atomic 4000000 | mutex 4000000
```
<!-- /output -->

Biến thường mất hơn một nửa số lần tăng: hai luồng cùng đọc một giá trị, cùng cộng 1, cùng ghi lại, nên một lần tăng biến mất. Kết quả còn đổi giữa các lần chạy. `std::atomic` và `std::mutex` đều cho đúng 4 000 000; atomic nhanh hơn với phép đơn giản như tăng biến đếm, mutex cần khi phải bảo vệ nhiều thao tác cùng lúc. Thứ tự các dòng "luồng i chạy xong" cũng có thể khác giữa các lần chạy.

## 6. Template: đặc tả một phần và toàn phần

> **Biên tập:** Tệp gốc `test_template.cpp` **không biên dịch được**: lớp có cả biến thành viên lẫn hàm thành viên cùng tên `size` và `shape`; `assert size == input_t.size();` sai cú pháp (thiếu ngoặc); dữ liệu khai báo là `std::vector<size_t>` thay vì `std::vector<T>`; `main` gọi hàm `add` không tồn tại; và đặc tả toàn phần của một mẫu hàm có kiểu trả về `auto` không khớp được với mẫu. Bản dưới giữ ý tưởng (một lớp `Tensor` theo thiết bị và kiểu dữ liệu, đặc tả một phần cho CPU, giống cách các framework học sâu chia backend) và sửa các lỗi trên.

<!-- cpp: tensor_template -->
```cpp
#include <cassert>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

enum class Device { CPU, GPU, NPU };

// 1. Khai báo mẫu tổng quát Tensor<Device, T>, chưa định nghĩa
template <Device Dev, typename T>
class Tensor;

// 2. Đặc tả một phần cho Dev = CPU: tham số nào đã cố định thì không còn trong template<...>
template <typename T>
class Tensor<Device::CPU, T> {
public:
    explicit Tensor(const std::vector<size_t>& shape, T fill = T{}) : shape_(shape) {
        size_t n = 1;
        for (auto d : shape) n *= d;
        data_.assign(n, fill);
    }
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    size_t size() const noexcept { return data_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }

    void add(const Tensor& other) {
        assert(size() == other.size());
        for (size_t i = 0; i < size(); ++i) data_[i] += other.data_[i];
    }
    void printDebug(const std::string& name) const {
        std::cout << "[CPU Tensor<" << typeid(T).name() << ">] " << name << " shape: [ ";
        for (auto s : shape_) std::cout << s << " ";
        std::cout << "], data[0] = " << data_[0] << ", size = " << size() << "\n";
    }

private:
    std::vector<size_t> shape_;
    std::vector<T> data_;
};

// Mẫu hàm và đặc tả toàn phần
template <typename T1, typename T2>
auto func(const T1& x, const T2& y) -> decltype(x + y) { return x + y; }

template <>
double func<int, double>(const int& x, const double& y) { return x - y; }   // riêng (int, double): trừ

int main() {
    Tensor<Device::CPU, float> a({2, 3}, 1.5f), b({2, 3}, 2.0f);
    a.add(b);
    a.printDebug("a");
    Tensor<Device::CPU, int> c({4}, 7);
    c.printDebug("c");
    // Tensor<Device::GPU, float> g({2});   // lỗi biên dịch: chưa có đặc tả cho GPU

    std::cout << "func(3, 5)     = " << func(3, 5) << "   (mẫu tổng quát: cộng)\n";
    std::cout << "func(8.7, 9.0) = " << func(8.7, 9.0) << "\n";
    std::cout << "func(3, 5.0)   = " << func(3, 5.0) << "   (đặc tả toàn phần int, double: trừ)\n";
}
```

<!-- run -->
```python
run_cpp("tensor_template")
```

<!-- output -->
```output
$ g++ ... tensor_template.cpp: biên dịch OK, không cảnh báo
[CPU Tensor<f>] a shape: [ 2 3 ], data[0] = 3.5, size = 6
[CPU Tensor<i>] c shape: [ 4 ], data[0] = 7, size = 4
func(3, 5)     = 8   (mẫu tổng quát: cộng)
func(8.7, 9.0) = 17.7
func(3, 5.0)   = -2   (đặc tả toàn phần int, double: trừ)
```
<!-- /output -->

`typeid(T).name()` in tên kiểu đã bị mã hoá theo trình biên dịch (`f` là `float`, `i` là `int` với GCC). Đặc tả một phần cho phép mỗi thiết bị có cài đặt riêng (CPU dùng `std::vector`, GPU sẽ dùng bộ nhớ CUDA); thiết bị chưa có đặc tả sẽ báo lỗi ngay khi biên dịch chứ không phải lúc chạy.

## Tóm tắt

- `binary_search` gốc chỉ đúng với phần tử giữa mảng; thử nhiều đầu vào, kể cả đầu, cuối và không có trong mảng, là cách rẻ nhất để bắt lỗi kiểu này.
- Hàm tạo di chuyển nên có `noexcept`; thiếu nó, `std::vector` sao chép phần tử khi cấp phát lại.
- Hàm huỷ không được ném ngoại lệ; RAII dựa vào việc hàm huỷ luôn dọn dẹp được.
- Biến dùng chung giữa các luồng phải là `std::atomic` hoặc được bảo vệ bằng mutex; với biến thường, kết quả sai và khác nhau giữa các lần chạy.
- `test_template.cpp` của bản gốc không biên dịch được; bản sửa minh hoạ đặc tả một phần cho lớp và đặc tả toàn phần cho hàm.
