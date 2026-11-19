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
