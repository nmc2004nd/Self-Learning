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
