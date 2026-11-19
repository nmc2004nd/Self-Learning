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
