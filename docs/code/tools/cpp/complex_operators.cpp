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
