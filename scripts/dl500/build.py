"""Dựng khu "Ôn tập phỏng vấn" (docs/qa) từ bản dịch của *Deep Learning 500 câu hỏi*.

Nguồn: github.com/scutan90/DeepLearning-500-questions (深度学习500问, Tan Jiyong chủ biên, GPL-3.0).
Bản dịch tiếng Việt (đã biên tập, sửa lỗi) nằm ở scripts/dl500/vi/chNN.md; script này:

- thêm ghi chú nguồn dưới tiêu đề chương;
- thay mỗi dòng `<!-- theory: math/ch01_vi#1.14 ; ml/ch20_vi -->` bằng hộp "Lý thuyết" chứa
  link tới trang (và mục, theo số mục) của các khoá khác. Mục không tồn tại → báo lỗi;
- đổi đường dẫn ảnh `img/chK/x.png` thành `../img/qa/chNN/x.png` và chép ảnh từ bản clone;
- ghi docs/qa/chNN_vi.md.

Dùng:
  python3 -B scripts/dl500/build.py --src <clone> 1 3 4      # các chương
  python3 -B scripts/dl500/build.py --src <clone> --all      # mọi chương đã dịch
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
VI = Path(__file__).resolve().parent / "vi"
DEFAULT_SRC = Path("/tmp/claude-1000/-home-nmc-Downloads-easy-rl-vi/20d21f98-0dc6-4e91-80ef-3946c9c08948/scratchpad/dl500/repo")
REPO = "https://github.com/scutan90/DeepLearning-500-questions"

# Thư mục chương trong bản gốc (để chép ảnh và dẫn link).
CHAPTER_DIRS = {
    1: "ch01_数学基础", 2: "ch02_机器学习基础", 3: "ch03_深度学习基础", 4: "ch04_经典网络",
    5: "ch05_卷积神经网络(CNN)", 6: "ch06_循环神经网络(RNN)", 7: "ch07_生成对抗网络(GAN)",
    8: "ch08_目标检测", 9: "ch09_图像分割", 10: "ch10_强化学习", 11: "ch11_迁移学习",
    12: "ch12_网络搭建及训练", 13: "ch13_优化算法", 14: "ch14_超参数调整",
    15: "ch15_GPU和框架选型", 16: "ch16_自然语言处理(NLP)", 17: "ch17_模型压缩、加速及移动端部署",
    18: "ch18_后端架构选型及应用场景", 19: "ch19_软件专利申请及权利保护",
}

# Tên khoá hiển thị trước link lý thuyết, theo thư mục trang đích.
COURSE_LABEL = {"math": "Toán", "ml": "Học máy", "dl": "Học sâu", "mathrl": "Toán RL", "qa": "Ôn tập"}

THEORY_RE = re.compile(r"^<!--\s*theory:\s*(.+?)\s*-->\s*$")
IMG_RE = re.compile(r"!\[([^\]]*)\]\((?:\./)?img/ch\d+/([^)]+?)\s*\)")  # tên ảnh có thể chứa dấu cách
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def slugify(text):
    """Giống docsify 4 (src/core/render/slugify.js), để link ?id=… trỏ đúng tiêu đề."""
    s = text.strip()
    s = re.sub(r"[A-Z]+", lambda m: m.group(0).lower(), s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[ -⁯⸀-⹿\\'!\"#$%&()*+,./:;<=>?@\[\]^`{|}~]", "", s)
    s = re.sub(r"\s", "-", s)
    s = re.sub(r"-+", "-", s)
    s = re.sub(r"^(\d)", r"_\1", s)
    return s


_TEX_SYM = {r"\times": "×", r"\cdot": "⋅", r"\le": "≤", r"\ge": "≥", r"\infty": "∞"}


def _katex_text(tex):
    """Chữ mà docsify thấy trong một công thức KaTeX: MathML + annotation (mã TeX gốc) + HTML."""
    shown = tex
    for k, v in _TEX_SYM.items():
        shown = shown.replace(k, v)
    shown = re.sub(r"\\[a-zA-Z]+|[{}^_\s]", "", shown)
    return shown + tex + shown


def plain(md, slug=False):
    """Bỏ định dạng markdown trong tiêu đề để ra chữ hiển thị (docsify slug theo chữ hiển thị)."""
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)
    # Công thức trong tiêu đề: KaTeX in chữ 3 lần (MathML, annotation, HTML) và docsify lấy slug từ đó.
    # Chỉ đúng với công thức đơn giản (như $R^2$); trang đã dựng phải được kiểm tra link trên trình duyệt.
    if slug:
        md = re.sub(r"\$([^$]+)\$", lambda m: _katex_text(m.group(1)), md)
    md = re.sub(r"(\*\*|__|\*|_|`)", "", md)
    return md.strip()


_page_cache = {}


def page_headings(page):
    """Trả về (tiêu đề H1, danh sách (số mục, chữ tiêu đề)) của docs/<page>.md."""
    if page in _page_cache:
        return _page_cache[page]
    path = DOCS / (page + ".md")
    if not path.exists():
        sys.exit(f"theory: không có trang {page}")
    title, heads, in_code = None, [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = HEAD_RE.match(line)
        if not m:
            continue
        text = plain(m.group(2))
        if len(m.group(1)) == 1 and title is None:
            title = text
        heads.append((text, plain(m.group(2), slug=True)))
    _page_cache[page] = (title or page, heads)
    return _page_cache[page]


def theory_link(target):
    target = target.strip()
    if target.startswith("http"):
        # Chưa có bài trên site: dẫn ra ngoài, dạng "url|nhãn".
        url, _, label = target.partition("|")
        return f"[{label.strip() or url.strip()}]({url.strip()}) ↗"
    page, _, sec = target.partition("#")
    page = page.strip()
    title, heads = page_headings(page)
    top = page.split("/")[0]
    course = "Tăng cường" if top.startswith("chapter") else COURSE_LABEL.get(top, "")
    if not sec:
        return f"[{course} · {title}]({page})"
    sec = sec.strip()
    for h, h_slug in heads:
        if h == sec or h.startswith(sec + " "):
            return f"[{course} · {h.rstrip('.')}]({page}?id={slugify(h_slug)})"
    sys.exit(f"theory: không có mục {sec} trong {page}")


def theory_box(spec):
    links = " · ".join(theory_link(t) for t in spec.split(";") if t.strip())
    return f'<div class="qa-theory">\n\n**Lý thuyết:** {links}\n\n</div>'


def header_note(num):
    src = f"{REPO}/tree/master/{CHAPTER_DIRS[num]}"
    return (
        "> Ghi chú của người biên soạn: trang này dịch từ sách *Deep Learning 500 câu hỏi* "
        "(深度学习500问) do **Tan Jiyong (谈继勇)** chủ biên cùng cộng đồng đóng góp "
        f"([bản gốc]({src}), [mã nguồn]({REPO}), giấy phép "
        "[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html); phần dịch cũng theo GPL-3.0). "
        "Số câu hỏi giữ như bản gốc. Chỗ bản gốc sai hoặc thiếu đã được sửa ngay trong bài và ghi "
        "ở ô **Biên tập**; ô **Lý thuyết** dẫn sang bài học tương ứng trên site."
    )


MAX_W = 1600  # ảnh gốc có tấm rộng tới 5000px: thu nhỏ cho nhẹ trang


def copy_image(src, dst):
    try:
        from PIL import Image
    except ImportError:
        shutil.copyfile(src, dst)
        return
    if src.suffix.lower() == ".gif":
        shutil.copyfile(src, dst)
        return
    with Image.open(src) as im:
        # Ảnh PNG nền trong suốt vẽ chữ đen: ở giao diện tối sẽ thành chữ đen trên nền tối, nên lót nền trắng.
        if im.mode in ("RGBA", "LA", "P") and im.convert("RGBA").getchannel("A").getextrema()[0] < 255:
            bg = Image.new("RGBA", im.size, "white")
            bg.alpha_composite(im.convert("RGBA"))
            im = bg.convert("RGB")
            if im.width > MAX_W:
                im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
            im.save(dst, optimize=True)
            return
        if im.width <= MAX_W:
            shutil.copyfile(src, dst)
            return
        h = round(im.height * MAX_W / im.width)
        im = im.resize((MAX_W, h), Image.LANCZOS)
        if src.suffix.lower() in (".jpg", ".jpeg"):
            im.convert("RGB").save(dst, quality=88, optimize=True)
        else:
            im.save(dst, optimize=True)


def build(num, src):
    vi = VI / f"ch{num:02d}.md"
    if not vi.exists():
        sys.exit(f"chưa có bản dịch {vi}")
    out_img = DOCS / "img" / "qa" / f"ch{num:02d}"
    src_dir = src / CHAPTER_DIRS[num]
    lines, out, in_code, h1_done = vi.read_text(encoding="utf-8").splitlines(), [], False, False
    images = set()

    def img_sub(m):
        images.add(m.group(2))
        # Thẻ <img> như khoá DL: link ảnh markdown bị docsify gắn thêm thư mục của trang.
        alt = m.group(1).strip("* ").replace('"', "&quot;")
        return f'<img src="../img/qa/ch{num:02d}/{m.group(2).replace(" ", "%20")}" alt="{alt}"/>'


    for line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not in_code:
            m = THEORY_RE.match(line)
            if m:
                out.append(theory_box(m.group(1)))
                continue
            line = IMG_RE.sub(img_sub, line)
        out.append(line)
        if not h1_done and line.startswith("# "):
            out += ["", header_note(num)]
            h1_done = True

    for name in sorted(images):
        # ảnh trùng tên ở thư mục con (như ch18 "新建文件夹") thì ưu tiên file nông nhất
        found = sorted(src_dir.rglob(name), key=lambda p: len(p.parts)) if src_dir.exists() else []
        dst = out_img / name
        if found:
            dst.parent.mkdir(parents=True, exist_ok=True)
            copy_image(found[0], dst)
        elif not dst.exists():
            sys.exit(f"ch{num}: không thấy ảnh {name} trong {src_dir}")

    dest = DOCS / "qa" / f"ch{num:02d}_vi.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    print(f"ch{num:02d}: {dest.relative_to(ROOT)} ({len(images)} ảnh)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("chapters", nargs="*", type=int)
    a = ap.parse_args()
    nums = sorted(int(p.stem[2:]) for p in VI.glob("ch*.md")) if a.all else a.chapters
    for n in nums:
        build(n, a.src)


if __name__ == "__main__":
    main()
