"""Chuyển 5 sơ đồ XMind của dl_note thành JSON cho trình xem docs/mindmap/ml.html?map=<khoá>.

Tệp .xmind là tệp zip, nội dung ở content.json (rootTopic → children.attached). Bản dịch từng nút
nằm ở xmind_vi.py, theo thứ tự duyệt cây; script kiểm tra từng nút khớp với đầu câu gốc rồi ghi
docs/mindmap/<khoá>-mindmap.json dạng {t, c, l}. Nút có ảnh chụp code được thay bằng liên kết sang
trang Code demo, nơi đoạn code đó chạy được.

Dùng:  python3 -B scripts/tools/convert_xmind.py [--src <clone dl_note>]
"""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xmind_vi import MAPS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = Path("/tmp/claude-1000/-home-nmc-Downloads-easy-rl-vi/963822cf-d45a-4311-aa91-22a954d0ef64/scratchpad/dl_note_src/dl_note")
FILES = {
    "cpp-thread": "1-math_ml_basic/cpp_learn_xmind/C++多线程-并发编程.xmind",
    "py-data": "1-math_ml_basic/python_learn_xmind/Python数据结构和高级特性.xmind",
    "py-func": "1-math_ml_basic/python_learn_xmind/Python编程笔记- 函数式编程.xmind",
    "py-oop": "1-math_ml_basic/python_learn_xmind/Python编程笔记-面向对象基础编程.xmind",
    "py-oop2": "1-math_ml_basic/python_learn_xmind/Python 编程笔记-面向对象高级编程.xmind",
}


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def convert(key, src):
    with zipfile.ZipFile(src / FILES[key]) as z:
        root = json.loads(z.read("content.json"))[0]["rootTopic"]
    table = list(MAPS[key])
    pos = [0]

    def walk(n):
        i = pos[0]
        pos[0] += 1
        if i >= len(table):
            sys.exit(f"{key}: bảng dịch thiếu nút thứ {i}: {norm(n.get('title', ''))!r}")
        entry = table[i]
        title = norm(n.get("title", ""))
        if not title.startswith(entry[0]):
            sys.exit(f"{key}: nút {i} là {title[:30]!r}, bảng dịch ghi {entry[0]!r}")
        kids = n.get("children", {}).get("attached", [])
        if entry[1] is None:          # bỏ cả nhánh: vẫn phải đi qua để giữ thứ tự
            for k in kids:
                walk(k)
            return None
        out = {"t": entry[1]}
        extra = entry[2] if len(entry) > 2 else {}
        if "l" in extra:
            out["l"] = extra["l"]
        children = [c for c in (walk(k) for k in kids) if c]
        children += [{"t": e} for e in extra.get("edit", [])]
        if children:
            out["c"] = children
        return out

    tree = walk(root)
    if pos[0] != len(table):
        sys.exit(f"{key}: bảng dịch thừa {len(table) - pos[0]} dòng")
    dst = ROOT / "docs" / "mindmap" / f"{key}-mindmap.json"
    dst.write_text(json.dumps(tree, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    def count(t):
        return 1 + sum(count(c) for c in t.get("c", []))
    print(f"{key}: {count(tree)} nút → {dst.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    a = ap.parse_args()
    for key in FILES:
        convert(key, a.src)


if __name__ == "__main__":
    main()
