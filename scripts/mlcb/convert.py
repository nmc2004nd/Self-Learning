"""Chuyển các chương của sách "Machine Learning cơ bản" (Vũ Hữu Tiệp) từ LaTeX sang Markdown cho docsify.

Nguồn: https://github.com/tiepvupsu/ebookMLCB (bản clone cục bộ, truyền qua --src).
Văn bản giữ nguyên văn; script chỉ đổi định dạng:
  - đánh số chương/mục/phương trình/hình/bảng/thuật toán như bản PDF (đọc thứ tự chương trong book_ML.tex),
    thay \\ref, \\eqref bằng số tương ứng; tham chiếu sang chương khác kèm tên chương và link nếu trang đã có;
  - khai triển macro của sách (\\bx, \\bw, \\bmt, \\L, \\argmin...) để KaTeX vẽ được;
  - khối lstlisting -> ```python / ```text, \\pythoninline -> `code`, \\newnote, myalg, mydeff -> khối trích dẫn;
  - hình: chép PNG, đổi PDF sang SVG (pdftocairo) vào docs/img/<khoá>/chNN/;
  - chú thích cuối trang -> mục "Chú thích" ở cuối bài;
  - ghi chú nguồn, giấy phép và đính chính (scripts/mlcb/notes/chNN.md) ngay dưới tiêu đề.

Chương 1–4, 23–25 và phụ lục A thuộc khoá Toán nền tảng (docs/math/), còn lại thuộc khoá Học máy (docs/ml/).

Dùng:
  python3 -B scripts/mlcb/convert.py --src <ebookMLCB> 7 8 9      # theo số chương
  python3 -B scripts/mlcb/convert.py --src <ebookMLCB> --list     # in danh sách chương
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
MATH_CHAPTERS = {1, 2, 3, 4, 23, 24, 25, "A"}

# ---------------------------------------------------------------- tiện ích đọc LaTeX


def strip_comments(tex):
    out = []
    for line in tex.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        if m:
            line = line[: m.start()]
            if not line.strip():
                continue          # dòng chỉ có chú thích: bỏ hẳn, không tạo dòng trống
        out.append(line)
    return "\n".join(out)


def balanced(s, i, open_="{", close="}"):
    """s[i] == open_; trả về vị trí ngay sau dấu đóng tương ứng."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    raise ValueError("ngoặc không cân bằng tại %d" % i)


def args_of(s, i, n, optional=False):
    """Đọc n đối số {...} bắt đầu từ s[i] (bỏ qua khoảng trắng và một [..] tuỳ chọn)."""
    out = []
    opt = None
    while i < len(s) and s[i] in " \t\n":
        i += 1
    if optional and i < len(s) and s[i] == "[":
        j = balanced(s, i, "[", "]")
        opt = s[i + 1: j - 1]
        i = j
    for _ in range(n):
        while i < len(s) and s[i] in " \t\n":
            i += 1
        j = balanced(s, i)
        out.append(s[i + 1: j - 1])
        i = j
    return out, opt, i


def replace_cmd(tex, name, fn, nargs=1, optional=False):
    """Thay mọi \\name{..}...{..} bằng fn(args, opt)."""
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])")
    out, pos = [], 0
    while True:
        m = pat.search(tex, pos)
        if not m:
            break
        args, opt, end = args_of(tex, m.end(), nargs, optional)
        out.append(tex[pos: m.start()])
        out.append(fn(args, opt))
        pos = end
    out.append(tex[pos:])
    return "".join(out)


def env_end(tex, start, env):
    """Vị trí của \\end{env} khớp với \\begin{env} vừa đọc (có tính lồng nhau)."""
    pat = re.compile(r"\\(begin|end)\{" + re.escape(env) + r"\}")
    depth = 1
    for m in pat.finditer(tex, start):
        depth += 1 if m.group(1) == "begin" else -1
        if depth == 0:
            return m.start(), m.end()
    raise ValueError(f"thiếu \\end{{{env}}}")


def split_top(s, sep):
    """Tách s theo sep ('\\\\\\\\' hoặc '&') ở mức ngoài cùng: không trong {...} hay \\begin..\\end lồng."""
    parts, depth, env, i, cur = [], 0, 0, 0, []
    while i < len(s):
        if s.startswith("\\begin{", i):
            env += 1
        elif s.startswith("\\end{", i):
            env -= 1
        c = s[i]
        if c == "\\" and sep == "\\\\" and s.startswith("\\\\", i) and depth == 0 and env == 0:
            parts.append("".join(cur))
            cur = []
            i += 2
            continue
        if c == "\\":
            cur.append(s[i: i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == sep and depth == 0 and env == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    return parts


def read_macros(src):
    """\\def\\name{body} không đối số trong packages_book.tex (bản sách thật sự dùng)."""
    tex = strip_comments((src / "packages_book.tex").read_text(encoding="utf-8"))
    macros = {}
    for m in re.finditer(r"\\def\\([A-Za-z]+)\{", tex):
        end = balanced(tex, m.end() - 1)
        macros[m.group(1)] = tex[m.end(): end - 1]
    macros["argmin"] = r"\operatorname*{argmin}"
    macros["argmax"] = r"\operatorname*{argmax}"
    macros["lbar"] = r"\overline"          # \newcommand*\lbar[2][.75]: gạch trên
    # Macro dùng cho chữ hoặc bố cục, không phải công thức
    for k in ("lg", "vd", "kq", "myenum", "dom"):
        macros.pop(k, None)
    return macros


def tidy_math(math):
    """Làm sạch công thức cho KaTeX và Markdown:
    - '\\ ' đơn lẻ (khoảng trắng điều khiển, hay gặp dạng '\\\\\\' cuối dòng) -> khoảng trắng;
    - bỏ dòng trống (Markdown sẽ cắt khối $$ làm đôi);
    - '_' trong \\texttt{...} phải thoát;
    - công thức nhiều dòng (có '\\\\' ở mức ngoài) không nằm trong môi trường nào -> bọc trong gathered."""
    math = re.sub(r"\\\\|\\(?=\s)", lambda m: m.group(0) if m.group(0) == "\\\\" else " ", math)
    math = re.sub(r"\n[ \t]*(?=\n)", "", math)
    # dòng mở đầu bằng "* ", "+ ", "- " hay "1. " sẽ bị Markdown hiểu là danh sách: nối vào dòng trước
    math = re.sub(r"\n[ \t]*(?=(?:[*+-]|\d+\.)[ \t])", " ", math)
    # '$' trong \text{...} của công thức hiển thị làm docsify cắt nhầm khối $$: tách thành \text{..} x \text{..}
    def text_dollar(a, o):
        parts = a[0].split("$")
        return "".join(("\\text{" + t + "}" if t else "") if k % 2 == 0 else t for k, t in enumerate(parts))
    for cmd in ("text", "mbox", "textrm"):
        math = replace_cmd(math, cmd, lambda a, o: text_dollar(a, o) if "$" in a[0] else "\\" + cmd + "{" + a[0] + "}")
    math = replace_cmd(math, "texttt", lambda a, o: "\\texttt{" + re.sub(r"(?<!\\)_", r"\\_", a[0]) + "}")
    if len(split_top(math, "\\\\")) > 1 and not re.match(r"\s*\\begin\{", math):
        tag = re.search(r"\\tag\{[^}]*\}", math)
        body = re.sub(r"\\tag\{[^}]*\}", "", math).strip()
        math = "\\begin{gathered}\n" + body + "\n\\end{gathered}" + (" " + tag.group(0) if tag else "")
    return math.strip()


def expand_macros(math, macros):
    for _ in range(5):
        new = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])",
                     lambda m: macros[m.group(1)] if m.group(1) in macros else m.group(0), math)
        if new == math:
            break
        math = new
    return math


# ---------------------------------------------------------------- thứ tự chương, phần và trang đích


def read_chapter(path, src):
    """Đọc một tệp chương và chèn nội dung các \\input{...} (không nằm trong dòng chú thích) vào đúng chỗ."""
    raw = path.read_text(encoding="utf-8")

    def inc(m):
        line_start = raw.rfind("\n", 0, m.start()) + 1
        if "%" in raw[line_start: m.start()]:
            return m.group(0)
        f = src / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        if not f.exists():
            return m.group(0)
        body = read_chapter(f, src)
        # Tệp được chèn có thể để hở ngoặc nhọn (LaTeX bỏ qua được, pandoc thì không): đóng lại ở cuối tệp
        plain = re.sub(r"\\[{}]", "", strip_comments(body))
        return body + "}" * max(0, plain.count("{") - plain.count("}"))
    return re.sub(r"\\input\{([^}]+)\}", inc, raw)


def book_layout(src):
    """(danh sách (đường dẫn .tex, số chương), {nhãn phần: (số phần, tên phần)}) theo book_ML.tex.

    Theo đúng bộ đếm của LaTeX: \\setcounter{chapter}{n} ở book_ML.tex hoặc ngay trong tệp chương
    (Lời nói đầu đặt -1 để thành chương 0), và sau \\appendix thì phụ lục được đánh A, B, ...
    """
    book = strip_comments((src / "book_ML.tex").read_text(encoding="utf-8"))
    chap = part = 0
    appendix_at = book.find("\\appendix")
    chapters, parts = [], {}
    for m in re.finditer(r"\\setcounter\{(chapter|part)\}\{(-?\d+)\}|\\(?:include|input)\{(Chapters/[^}]+)\}", book):
        if m.group(1) == "chapter":
            chap = int(m.group(2))
            continue
        if m.group(1) == "part":
            part = int(m.group(2))
            continue
        path = src / (m.group(3) + ".tex")
        if not path.exists():
            continue
        tex = strip_comments(path.read_text(encoding="utf-8"))
        pm = re.search(r"\\part\{([^}]*)\}\s*\\label\{([^}]+)\}", tex)
        if pm:
            part += 1
            parts[pm.group(2)] = (part, pm.group(1))
        ch = re.search(r"\\chapter\{", tex)
        if not ch:
            continue
        inner = re.findall(r"\\setcounter\{chapter\}\{(-?\d+)\}", tex[: ch.start()])
        if inner:
            chap = int(inner[-1])
        chap += 1
        if 0 <= appendix_at < m.start():
            n_app = sum(1 for _, c in chapters if isinstance(c, str))
            chapters.append((path, "ABCDEFGH"[n_app]))
        else:
            chapters.append((path, chap))
    return chapters, parts


# Tên chương hiển thị bằng tiếng Anh (thuật ngữ quen dùng hơn bản dịch); nội dung chương vẫn giữ nguyên văn.
EN_TITLES = {
    6: "Feature engineering", 7: "Linear regression", 8: "Overfitting",
    9: "$K$-nearest neighbors (KNN)", 10: "$K$-means clustering", 11: "Naive Bayes classifier",
    13: "Perceptron learning algorithm (PLA)", 14: "Logistic regression", 15: "Softmax regression",
    16: "Multilayer perceptron and backpropagation", 17: "Content-based recommendation systems",
    18: "Neighborhood-based collaborative filtering", 19: "Matrix factorization collaborative filtering",
    20: "Singular value decomposition (SVD)", 21: "Principal component analysis (PCA)",
    22: "Linear discriminant analysis (LDA)", 26: "Support vector machine (SVM)", 27: "Soft-margin SVM",
    28: "Kernel SVM", 29: "Multi-class SVM",
}


def page_of(n):
    """Trang đích (đường dẫn docsify, không đuôi) và thư mục hình của một chương."""
    course = "math" if n in MATH_CHAPTERS else "ml"
    name = f"ch{n:02d}" if isinstance(n, int) else f"appendix_{n.lower()}"
    return f"{course}/{name}_vi", f"{course}/{name}"


# ---------------------------------------------------------------- duyệt và đánh số một chương

NUM_ENVS = ("equation", "alignat", "align", "eqnarray", "gather", "figure", "table", "myalg", "mydef", "mytheo")
# khung đánh số kiểu tcolorbox theorem: môi trường -> (tên hiển thị, tiền tố nhãn)
TCB_ENVS = {"myalg": ("Thuật toán", "alg"), "mydef": ("Định nghĩa", "def"), "mytheo": ("Định lý", "th")}



def trim_columns(tab):
    """Bỏ các cột khai báo thừa trong \\begin{tabular}{...} (vd. {|l|l|l|l|} nhưng mỗi hàng chỉ có 3 ô)."""
    m = re.match(r"\\begin\{tabular\}\{([^{}]*)\}", tab)
    if not m:
        return tab
    spec = m.group(1)
    cols = re.findall(r"[lcr]|p\{[^}]*\}", spec)
    body = tab[m.end():]
    rows = [r for r in re.split(r"\\\\", body) if re.sub(r"\\hline|\\end\{tabular\}|\s", "", r)]
    ncell = max((len(re.split(r"(?<!\\)&", r)) for r in rows), default=len(cols))
    if ncell >= len(cols):
        return tab
    keep = "|" + "|".join(cols[:ncell]) + "|" if spec.strip().startswith("|") else "".join(cols[:ncell])
    return tab[:m.start(1)] + keep + tab[m.end(1):]

class Walker:
    """Đánh số chương/mục/công thức/hình/bảng/thuật toán theo thứ tự xuất hiện.

    Dùng chung cho việc quét nhãn toàn sách (render=None) và cho việc chuyển đổi (render=Converter),
    để hai việc luôn ra cùng một số.
    """

    def __init__(self, chap, render=None, macros=None):
        self.chap = chap
        self.render = render
        # macro (vd. \\bmt ... \\emt) phải mở trước khi tách dòng, để '\\\\' trong ma trận không bị coi là xuống dòng
        self.macros = macros if macros is not None else (render.macros if render else {})
        self.labels = {}
        self.title = ""
        self.sec = self.sub = self.eq = self.fig = self.tab = 0
        self.tcb = dict.fromkeys(TCB_ENVS, 0)
        self.ctx = ("chapter", str(chap))

    def label_all(self, body, kind, num):
        for lab in re.findall(r"\\label\{([^}]+)\}", body):
            self.labels[lab] = (kind, num)

    def math_env(self, env, star, body):
        """Công thức hiển thị: trả về mã KaTeX đã gắn \\tag cho các dòng được đánh số."""
        numbered = not star
        body = expand_macros(body, self.macros)
        if env == "equation":
            has_no = re.search(r"\\nonumber|\\notag", body)
            body2 = re.sub(r"\\label\{[^}]+\}|\\nonumber|\\notag", "", body).strip()
            if not (numbered and not has_no):
                # như LaTeX: môi trường tăng bộ đếm trước rồi mới bỏ số, nên nhãn trỏ tới số kế tiếp
                self.label_all(body, "equation", f"{self.chap}.{self.eq + 1}")
            if numbered and not has_no:
                self.eq += 1
                num = f"{self.chap}.{self.eq}"
                self.label_all(body, "equation", num)
                body2 += f" \\tag{{{num}}}"
            return body2
        rows = split_top(body, "\\\\")
        while rows and not rows[-1].strip():
            rows.pop()
        out = []
        for row in rows:
            sc = re.search(r"\\setcounter\{equation\}\{(\d+)\}", row)
            if sc:                               # trong eqnarray: dòng chứa \setcounter{equation}{k} mang số k
                self.eq = int(sc.group(1)) - 1
                row = row.replace(sc.group(0), "")
            has_no = re.search(r"\\nonumber|\\notag", row)
            labs = re.findall(r"\\label\{([^}]+)\}", row)
            r = re.sub(r"\\label\{[^}]+\}|\\nonumber|\\notag", "", row).strip()
            if env == "eqnarray":
                cells = split_top(r, "&")
                if len(cells) == 3:
                    r = cells[0] + "&" + cells[1] + cells[2]
            if not (numbered and not has_no and r):
                for lab in labs:
                    self.labels[lab] = ("equation", f"{self.chap}.{self.eq + 1}")
            if numbered and not has_no and r:
                self.eq += 1
                num = f"{self.chap}.{self.eq}"
                for lab in labs:
                    self.labels[lab] = ("equation", num)
                r += f" \\tag{{{num}}}"
            out.append(r)
        inner = " \\\\\n".join(out)
        if env == "gather":
            return "\\begin{gather*}\n" + inner + "\n\\end{gather*}"
        return "\\begin{align*}\n" + inner + "\n\\end{align*}"

    def walk(self, tex):
        tex = replace_cmd(tex, "myeqnbox", lambda a, o: a[0])   # khung viền quanh công thức: chỉ giữ công thức
        chap = self.chap
        R = self.render
        token = re.compile(r"\\(chapter|section|subsection|subsubsection)\*?\{"
                           r"|\\begin\{(" + "|".join(NUM_ENVS) + r")(\*?)\}|\\label\{([^}]+)\}|\\setcounter\{equation\}\{(\d+)\}")
        out, pos = [], 0
        while True:
            m = token.search(tex, pos)
            if not m:
                break
            out.append(tex[pos: m.start()])
            if m.group(5):                       # đặt lại bộ đếm công thức (ch27)
                self.eq = int(m.group(5))
                pos = m.end()
                continue
            if m.group(4):                       # nhãn rời: gắn với mục/chương hiện tại
                self.labels[m.group(4)] = self.ctx
                pos = m.end()
                continue
            if m.group(1):
                end = balanced(tex, m.end() - 1)
                title = tex[m.end(): end - 1]
                lvl = m.group(1)
                if lvl == "chapter":
                    self.title = title
                    self.ctx = ("chapter", str(chap))
                    out.append(f"\\chapter{{{'Chương ' if isinstance(chap, int) else 'Phụ lục '}{chap} {EN_TITLES.get(chap, title)}}}")
                elif lvl == "section":
                    self.sec += 1
                    self.sub = 0
                    self.ctx = ("section", f"{chap}.{self.sec}")
                    out.append(f"\\section{{{chap}.{self.sec} {title}}}")
                elif lvl == "subsection":
                    self.sub += 1
                    self.ctx = ("subsection", f"{chap}.{self.sec}.{self.sub}")
                    out.append(f"\\subsection{{{chap}.{self.sec}.{self.sub} {title}}}")
                else:
                    out.append(f"\\subsubsection{{{title}}}")
                pos = end
                continue
            env, star = m.group(2), m.group(3)
            b_end, e_end = env_end(tex, m.end(), env + star)
            body = tex[m.end(): b_end]
            pos = e_end
            if env in ("equation", "align", "alignat", "eqnarray", "gather"):
                ncols = None
                if env == "alignat":
                    (ncols,), _, after = args_of(body, 0, 1)
                    body = body[after:]
                math = self.math_env(env, star, body)
                if ncols:
                    math = math.replace("\\begin{align*}", "\\begin{alignat*}{%s}" % ncols).replace("\\end{align*}", "\\end{alignat*}")
                if R:
                    out.append(R.slot("$$\n" + tidy_math(expand_macros(math, R.macros)) + "\n$$"))
            elif env == "figure":
                self.fig += 1
                num = f"{chap}.{self.fig}"
                k = 0
                for sm in re.finditer(r"\\begin\{subfigure\}(.*?)\\end\{subfigure\}", body, re.S):
                    k += 1
                    for lab in re.findall(r"\\label\{([^}]+)\}", sm.group(1)):
                        self.labels[lab] = ("subfigure", num + "abcdefgh"[k - 1])
                rest = re.sub(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", "", body, flags=re.S)
                self.label_all(rest, "figure", num)
                if R:
                    out.append(R.slot(R.figure(body, num)))
            elif env == "table":
                num = None
                if "\\caption" in body:                # bảng không có \caption thì LaTeX không đánh số
                    self.tab += 1
                    num = f"{chap}.{self.tab}"
                    self.label_all(body, "table", num)
                if R:
                    out.append(R.table(body, num))
            elif env in TCB_ENVS:
                word, prefix = TCB_ENVS[env]
                self.tcb[env] += 1
                num = f"{chap}.{self.tcb[env]}"
                (title,), _, after = args_of(body, 0, 1)
                lab = ""
                if body[after:].lstrip(" \t").startswith("{"):     # đối số nhãn (sách có chỗ bỏ quên)
                    (lab,), _, after = args_of(body, after, 1)
                kind = {"myalg": "algorithm", "mydef": "definition", "mytheo": "theorem"}[env]
                self.labels[prefix + ":" + lab] = (kind, num)
                inner = self.walk(body[after:])          # công thức trong khung cũng được đánh số
                if R:
                    head = f"{word} {num}" + (f": {R.inline_md(title)}" if title.strip() else "")
                    out.append(R.box(head, inner))
        out.append(tex[pos:])
        return "".join(out)


def global_labels(src):
    labels, titles = {}, {}
    chapters, parts = book_layout(src)
    macros = read_macros(src)
    for path, chap in chapters:
        tex = strip_comments(read_chapter(path, src))
        w = Walker(chap, macros=macros)
        w.walk(tex)
        for k, v in w.labels.items():
            labels.setdefault(k, (v[0], v[1], chap))
        titles[chap] = w.title
    for lab, (num, title) in parts.items():
        labels[lab] = ("part", str(num), None)
        titles[lab] = title
    return labels, titles


# ---------------------------------------------------------------- chuyển một chương


class Converter:
    def __init__(self, src, chap):
        self.src = src
        self.chap = chap
        chapters, _ = book_layout(src)
        self.path = dict((c, p) for p, c in chapters)[chap]
        self.page, img = page_of(chap)
        self.out_md = DOCS / (self.page + ".md")
        self.img_rel = img
        self.img_dir = DOCS / "img" / img
        self.note_name = Path(img).name
        self.macros = read_macros(src)
        self.labels, self.titles = global_labels(src)
        self.slots = {}           # khối đã dựng sẵn (code, hình, công thức) thay vào sau pandoc
        self.warnings = []
        self.svg_count = 0

    def slot(self, md):
        key = f"MLCBSLOT{len(self.slots):03d}X"
        self.slots[key] = md
        return f"\n\n{key}\n\n"

    # --- tham chiếu
    def lookup(self, label):
        if label not in self.labels:
            self.warnings.append(f"không rõ nhãn {label}")
            return None
        kind, num, chap = self.labels[label]
        if kind == "subfigure":
            num = num[:-1] + "(" + num[-1] + ")"
        return kind, num, chap

    def ref(self, label, paren=False):
        r = self.lookup(label)
        if not r:
            return "??"
        return f"({r[1]})" if paren else r[1]

    def chapter_link(self, chap, text):
        page, _ = page_of(chap)
        if (DOCS / (page + ".md")).exists() and chap != self.chap:
            return f"\\href{{{page}}}{{{text}}}"
        return text

    def ref_text(self, label):
        """\\ref trong câu chữ: số, kèm tên (và link) nếu trỏ sang chương hoặc phần khác."""
        r = self.lookup(label)
        if not r:
            return "??"
        kind, num, chap = r
        if kind == "part":
            return f"{num} (\\textit{{{self.titles.get(label, '')}}})"
        if kind == "chapter":
            return self.chapter_link(chap, num) + f" (\\textit{{{self.titles.get(chap, '')}}})"
        if chap != self.chap:
            word = "chương" if isinstance(chap, int) else "phụ lục"
            return f"{num} ({word} {self.chapter_link(chap, str(chap))}, \\textit{{{self.titles.get(chap, '')}}})"
        return num

    # --- hình, bảng, khung
    def figure_file(self, rel):
        p = self.src / rel
        if not p.suffix:
            for ext in (".pdf", ".png", ".jpg"):
                if p.with_suffix(ext).exists():
                    p = p.with_suffix(ext)
                    break
        if not p.exists():
            self.warnings.append(f"thiếu hình {rel}")
            return ""
        self.img_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", p.stem)
        if p.suffix.lower() == ".pdf":
            out = self.img_dir / (stem + ".svg")
            subprocess.run(["pdftocairo", "-svg", str(p), str(out)], check=True)
            self.svg_count += 1
        else:
            out = self.img_dir / (stem + p.suffix.lower())
            shutil.copyfile(p, out)
        return f"../img/{self.img_rel}/{out.name}"

    def inline_md(self, tex):
        """Chuyển một đoạn LaTeX ngắn (chú thích hình, tiêu đề) sang Markdown một dòng."""
        md = self.math_fix(pandoc(self.pre_text(tex))).strip()
        md = md.replace("\\(", "(").replace("\\)", ")")
        return re.sub(r"\s*\n\s*", " ", md)

    def caption_of(self, body):
        c = re.search(r"\\caption(?:\[[^\]]*\])?\{", body)
        return body[c.end(): balanced(body, c.end() - 1) - 1] if c else ""

    def figure(self, body, number):
        imgs = []
        for m in re.finditer(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}", body):
            opt = m.group(1) or ""
            w = re.search(r"width\s*=\s*([\d.]*)\\(?:text|line)width", opt)
            frac = float(w.group(1) or 1) if w else 0.6
            imgs.append((self.figure_file(m.group(2)), frac))
        subcaps = [self.caption_of(sm.group(1))
                   for sm in re.finditer(r"\\begin\{subfigure\}(.*?)\\end\{subfigure\}", body, re.S)]
        outer = re.sub(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", "", body, flags=re.S)
        cap = self.caption_of(outer)
        if len(imgs) > 1:
            cells = []
            for k, (src, _) in enumerate(imgs):
                lab = ""
                if subcaps:
                    sc = self.inline_md(subcaps[k]) if k < len(subcaps) and subcaps[k].strip() else ""
                    lab = f"<br><small>({'abcdefgh'[k]}) {sc}</small>".replace(" </small>", "</small>")
                cells.append(f'<span class="ml-subfig"><img src="{src}"/>{lab}</span>')
            html = '<div align=center class="ml-fig">\n' + "".join(cells) + "\n</div>"
        elif imgs:
            width = int(max(0.35, min(imgs[0][1], 1)) * 900)
            html = f'<div align=center>\n<img width="{width}" src="{imgs[0][0]}"/>\n</div>'
        else:
            self.warnings.append(f"hình {number} không có ảnh")
            html = ""
        html += f"\n<div align=center>Hình {number}: {self.inline_md(cap)}</div>"
        return html

    def table(self, body, number):
        cap_md = self.inline_md(self.caption_of(body))
        tabular = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", body, re.S)
        tab = tabular.group(0) if tabular else ""
        tab = trim_columns(tab)
        if number is None:
            return tab
        return self.slot(f"<div align=center>Bảng {number}: {cap_md}</div>") + tab

    def box(self, title, body):
        """Khung (thuật toán, chú ý, ghi chú) -> khối trích dẫn có tiêu đề in đậm."""
        md = self.math_fix(pandoc(self.pre_text(body))).strip()
        lines = ([f"> **{title}**", ">"] if title else []) + ["> " + ln if ln else ">" for ln in md.split("\n")]
        return self.slot("\n".join(lines))

    # --- tiền xử lý văn bản
    def pre_text(self, tex):
        # macro chữ của sách (packages_book.tex): \kq, \vd, \lg
        tex = re.sub(r"\\kq(?![A-Za-z])[ \t]*", lambda m: "\\textit{Kết quả:} ", tex)
        tex = re.sub(r"\\vd(?![A-Za-z])[ \t]*", lambda m: "\\textbf{Ví dụ}: ", tex)
        tex = re.sub(r"\\lg(?![A-Za-z])[ \t]*", lambda m: "\\textbf{\\textit{Lời giải}}: ", tex)
        tex = replace_cmd(tex, "index", lambda a, o: "")
        tex = re.sub(r"\\eqref\{([^}]+)\}", lambda m: self.ref(m.group(1), paren=True), tex)
        tex = re.sub(r"\\(?:ref|autoref|cref)\{([^}]+)\}", lambda m: self.ref_text(m.group(1)), tex)
        tex = re.sub(r"(?:trang|page)~?\\pageref\{[^}]+\}", "phần tương ứng của bản PDF", tex)
        tex = replace_cmd(tex, "pythoninline", lambda a, o: "\\texttt{" + a[0].replace("_", "\\_") + "}")
        tex = replace_cmd(tex, "bi", lambda a, o: "\\textbf{\\textit{" + a[0] + "}}")
        return tex

    def convert(self):
        raw = read_chapter(self.path, self.src)

        # code: lstlisting / python env. Lấy ra TRƯỚC khi bỏ chú thích LaTeX, vì '%' trong code Python là toán tử.
        # Bỏ qua khối nằm trong vùng đã bị chú thích (dòng \begin bắt đầu bằng %).
        def code_env(m):
            line_start = raw.rfind("\n", 0, m.start()) + 1
            if raw[line_start: m.start()].lstrip().startswith("%"):
                return m.group(0)
            opt = m.group(1) or ""
            body = m.group(2).strip("\n")
            lang = "python" if ("Python" in opt or m.group(0).startswith("\\begin{python}")) else "text"
            # khối kết quả in ra trong sách cũng hay gắn language=Python: nhận diện qua câu "Kết quả" ngay trước
            before = strip_comments(raw[max(0, m.start() - 60): m.start()])
            if re.search(r"(?:Kết quả:?|\\kq)\s*$", before):
                lang = "text"
            return self.slot(f"```{lang}\n{body}\n```")
        raw = re.sub(r"\\begin\{lstlisting\}(?:\[([^\]]*)\])?\n?(.*?)\\end\{lstlisting\}", code_env, raw, flags=re.S)
        raw = re.sub(r"\\begin\{python\}(?:\[([^\]]*)\])?\n?(.*?)\\end\{python\}", code_env, raw, flags=re.S)
        tex = strip_comments(raw)

        tex = Walker(self.chap, self).walk(tex)

        # Các khung không đánh số
        tex = replace_cmd(tex, "newnote", lambda a, o: self.box(self.inline_md(a[0]), a[1]), nargs=2)
        tex = re.sub(r"\\begin\{mydeff\}(.*?)\\end\{mydeff\}", lambda m: self.box("Chú ý", m.group(1)), tex, flags=re.S)
        tex = re.sub(r"\\begin\{mynote\}(.*?)\\end\{mynote\}", lambda m: self.box("", m.group(1)), tex, flags=re.S)

        tex = self.pre_text(tex)
        tex = re.sub(r"\\label\{[^}]+\}", "", tex)
        md = pandoc(tex)
        md = self.math_fix(md)
        md = self.footnotes(md)
        for _ in range(3):        # khối lồng nhau (công thức trong khung thuật toán)
            md = self.fill_slots(md)
        md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
        leftovers = sorted(set(re.findall(r"MLCBSLOT\d+X", md)))
        if leftovers:
            self.warnings.append(f"còn {len(leftovers)} khối chưa thay")
        return self.header(md)

    def header(self, md):
        """Ghi chú nguồn, giấy phép và đính chính (nếu có scripts/mlcb/notes/chNN.md) ngay dưới tiêu đề."""
        lines = [
            "> Ghi chú của người biên soạn: chương này lấy nguyên văn từ sách *Machine Learning cơ bản* của "
            "**Vũ Hữu Tiệp** ([mã nguồn LaTeX](https://github.com/tiepvupsu/ebookMLCB), giấy phép "
            "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); trang gốc "
            "[machinelearningcoban.com](https://machinelearningcoban.com)). Người biên soạn chỉ đổi định dạng sang "
            "trang web: số chương, mục, công thức và hình giữ như bản PDF"
            + (", hình vector được vẽ lại dưới dạng SVG." if self.svg_count else "."),
        ]
        notes = Path(__file__).parent / "notes" / f"{self.note_name}.md"
        if notes.exists():
            lines += [">", "> **Đính chính** *(người biên soạn)*:", ">"]
            lines += ["> " + ln if ln.strip() else ">" for ln in notes.read_text(encoding="utf-8").strip().split("\n")]
        head, _, rest = md.partition("\n")
        return head + "\n\n" + "\n".join(lines) + "\n" + rest

    def fill_slots(self, md):
        """Thay khối dựng sẵn vào chỗ giữ. Nếu chỗ giữ nằm trong danh sách (thụt lề) hay khung trích dẫn ("> "),
        mọi dòng của khối đều phải mang cùng tiền tố đó, nếu không Markdown sẽ hiểu sai."""
        def sub(m):
            prefix, key = m.group(1), m.group(2)
            block = self.slots[key]
            return "\n".join(prefix + ln if ln else prefix.rstrip() for ln in block.split("\n"))
        return re.sub(r"^([ >]*)(MLCBSLOT\d{3}X)[ \t]*$", sub, md, flags=re.M)

    @staticmethod
    def indent_display(md):
        """$$ ... $$ của pandoc nằm giữa đoạn văn: tách ra thành khối riêng, giữ tiền tố của dòng."""
        out, pos = [], 0
        for m in re.finditer(r"\$\$(.+?)\$\$", md, flags=re.S):
            line_start = md.rfind("\n", 0, m.start()) + 1
            lead = re.match(r"[ >]*", md[line_start:]).group(0)
            body = [ln.strip() for ln in m.group(1).strip().split("\n")]
            block = ["", lead + "$$"] + [lead + ln for ln in body] + [lead + "$$", lead.rstrip()]
            out.append(md[pos: m.start()].rstrip(" "))
            out.append("\n".join(block))
            pos = m.end()
            # phần chữ còn lại trên cùng dòng sau $$: đưa xuống dòng mới với cùng tiền tố
            rest = re.match(r"[ \t]*", md[pos:]).end()
            pos += rest
            if pos < len(md) and md[pos] != "\n":
                out.append("\n" + lead)
        out.append(md[pos:])
        return "".join(out)

    def math_fix(self, md):
        # công thức nội dòng: khai triển macro, nối dòng
        md = re.sub(r"(?<!\$)\$([^$]+?)\$(?!\$)",
                    lambda m: "$" + re.sub(r"\s*\n\s*", " ", tidy_math(expand_macros(m.group(1), self.macros))) + "$", md)
        md = re.sub(r"\$\$(.+?)\$\$", lambda m: "$$" + tidy_math(expand_macros(m.group(1), self.macros)) + "$$", md, flags=re.S)
        return self.indent_display(md)

    def footnotes(self, md):
        defs = {}

        def grab(m):
            defs[m.group(1)] = m.group(2).strip()
            return ""
        md = re.sub(r"^\[\^(\d+)\]:\s*(.*?)(?=\n\[\^\d+\]:|\n\n(?!    )|\Z)", grab, md, flags=re.S | re.M)
        if not defs:
            return md
        md = re.sub(r"\[\^(\d+)\]", r"<sup>\1</sup>", md)
        notes = "\n".join(k + ". " + re.sub(r"\s+", " ", v) for k, v in sorted(defs.items(), key=lambda kv: int(kv[0])))
        return md.rstrip() + "\n\n---\n\n**Chú thích**\n\n" + notes + "\n"


def pandoc(tex):
    r = subprocess.run(["pandoc", "-f", "latex", "-t", "gfm+tex_math_dollars-raw_html", "--wrap=none"],
                       input=tex, capture_output=True, text=True)
    if r.returncode:
        sys.exit("pandoc lỗi:\n" + r.stderr)
    return r.stdout


def parse_chap(s):
    return s.upper() if s.isalpha() else int(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="thư mục clone ebookMLCB")
    ap.add_argument("--list", action="store_true", help="in danh sách chương rồi thoát")
    ap.add_argument("chapters", nargs="*", type=parse_chap, help="số chương (0–29) hoặc A, B")
    a = ap.parse_args()
    src = a.src.resolve()
    if a.list:
        labels, titles = global_labels(src)
        for path, n in book_layout(src)[0]:
            page, _ = page_of(n)
            print(f"{n!s:>2}  {titles[n]:45s} {path.relative_to(src)}  ->  docs/{page}.md")
        return
    for n in a.chapters:
        c = Converter(src, n)
        shutil.rmtree(c.img_dir, ignore_errors=True)
        md = c.convert()
        c.out_md.parent.mkdir(parents=True, exist_ok=True)
        c.out_md.write_text(md, encoding="utf-8")
        n_img = len(list(c.img_dir.glob("*"))) if c.img_dir.exists() else 0
        print(f"{c.out_md.relative_to(ROOT)}: {len(md.splitlines())} dòng, {n_img} hình")
        for w in sorted(set(c.warnings)):
            print("  cảnh báo:", w)


if __name__ == "__main__":
    main()
