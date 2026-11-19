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
