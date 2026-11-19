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
