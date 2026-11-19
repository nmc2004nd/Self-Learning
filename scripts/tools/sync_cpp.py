"""Chép mã nguồn C++ từ docs/code/tools/cpp/<tên>.cpp vào trang docs/tools/cpp_demo_vi.md.

Trên trang, mỗi chương trình đặt sau dòng <!-- cpp: <tên> --> và một khối ```cpp; script thay nội
dung khối đó bằng tệp .cpp, để mã hiển thị luôn trùng với mã được biên dịch (khối <!-- run --> ngay
sau gọi run_cpp("<tên>") để biên dịch và chạy, qua build_code_pages.py).

Dùng:  python3 -B scripts/tools/sync_cpp.py && python3 scripts/build_code_pages.py docs/tools/cpp_demo_vi.md
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "tools" / "cpp_demo_vi.md"
SRC = ROOT / "docs" / "code" / "tools" / "cpp"
BLOCK = re.compile(r"(<!-- cpp: ([\w-]+) -->\n```cpp\n)(.*?\n)?(```\n)", re.S)   # khối có thể đang rỗng


def main():
    text = PAGE.read_text(encoding="utf-8")
    names = []

    def sub(m):
        names.append(m.group(2))
        code = (SRC / f"{m.group(2)}.cpp").read_text(encoding="utf-8").rstrip("\n")
        return m.group(1) + code + "\n" + m.group(4)

    PAGE.write_text(BLOCK.sub(sub, text), encoding="utf-8")
    print("đã chép:", ", ".join(names))


if __name__ == "__main__":
    main()
