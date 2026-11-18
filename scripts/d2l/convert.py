"""Chuyển sách Dive into Deep Learning (d2l-en, CC BY-SA 4.0) thành trang docsify của khoá Học sâu.

Hai bước:

1. extract: lấy các mục của một chương từ bản clone d2l-en, chỉ giữ tab PyTorch, ghi ra
   scripts/d2l/vi/<thư mục chương>/<mục>.md (vẫn là cú pháp d2lbook). Không ghi đè tệp đã có,
   vì các tệp này là nơi người biên soạn dịch nguyên văn sang tiếng Việt.
2. build: đọc các tệp tiếng Việt, đánh số mục, hình, công thức (giống bản HTML d2l.ai),
   thay :numref:/:eqref:/:cite:, chép hình SVG, ghi docs/dl/chNN_vi.md (mở đầu chương)
   và docs/dl/chNN_sK_vi.md (mục K).

Dùng:
  python3 -B scripts/d2l/convert.py --src <clone d2l-en> extract 7
  python3 -B scripts/d2l/convert.py --src <clone d2l-en> build 7
  python3 -B scripts/d2l/convert.py --src <clone d2l-en> --list

Khối code chạy được: ```{.python .input} → <!-- run --> (build_code_pages.py chạy và chèn kết
quả). Thêm .gpu vào thuộc tính khối (```{.python .input .gpu}) cho khối huấn luyện nặng:
khối đó thành <!-- run gpu -->, chỉ chạy trên máy có GPU. .norun: chỉ hiển thị, không bao giờ chạy
(chẳng hạn code cần hai GPU).
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
VI = Path(__file__).resolve().parent / "vi"
NOTES = Path(__file__).resolve().parent / "notes"
DEFAULT_SRC = Path("/tmp/claude-1000/-home-nmc-Downloads-easy-rl-vi/ea390a1f-2f7f-4083-931e-315aa1fb73a8/scratchpad/src/d2l-en")
TAB = "pytorch"
D2L_URL = "https://d2l.ai"


# ---------------------------------------------------------------- cấu trúc sách

def toc_entries(index_md):
    """Danh sách mục trong khối ```toc của một index.md."""
    m = re.search(r"```toc\n(.*?)```", index_md, re.S)
    if not m:
        return []
    return [ln.strip() for ln in m.group(1).splitlines()
            if ln.strip() and not ln.strip().startswith(":")]


def book_layout(src):
    """[(số chương, thư mục chương, [tên mục...])] theo thứ tự của sách, kể cả phụ lục (A, B)."""
    text = (src / "index.md").read_text()
    blocks = re.findall(r"```toc\n(.*?)```", text, re.S)
    numbered = next(b for b in blocks if ":numbered:" in b)
    chaps = [ln.strip().split("/")[0] for ln in numbered.splitlines()
             if ln.strip() and not ln.strip().startswith(":")]
    out, n = [], 0
    for d in chaps:
        if "appendix" in d:
            num = "AB"[sum(1 for c in out if isinstance(c[0], str))]
        else:
            n += 1
            num = n
        secs = [s.removesuffix(".md") for s in toc_entries((src / d / "index.md").read_text())]
        out.append((num, d, secs))
    return out


def page_name(num, k=None):
    if num == "N":                      # phần Ký hiệu, không đánh số
        return "notation_vi"
    base = f"ch{num:02d}" if isinstance(num, int) else f"appendix_{num.lower()}"
    return f"{base}_vi" if k is None else f"{base}_s{k}_vi"


# ---------------------------------------------------------------- chọn tab PyTorch

CELL = re.compile(r"^```\{\.python \.input([^}]*)\}\n(.*?)^```\n?", re.S | re.M)


def tabs_of(line):
    return [t.strip() for t in line[len("%%tab"):].split(",")]


def resolve_selected(code):
    """Giữ nhánh `if tab.selected(...)` của PyTorch trong khối %%tab all."""
    lines = code.split("\n")
    out, i = [], 0
    head = re.compile(r"^(\s*)(if|elif) (not )?tab\.selected\((.*)\):\s*$")
    while i < len(lines):
        m = head.match(lines[i])
        if not m or m.group(2) != "if":
            out.append(lines[i]); i += 1
            continue
        ind = m.group(1)
        branches = []  # (điều kiện đúng với PyTorch?, thân)
        while i < len(lines):
            m2 = head.match(lines[i])
            if m2 and m2.group(1) == ind and (m2.group(2) == "if" and not branches or m2.group(2) == "elif"):
                names = re.findall(r"'([a-z]+)'", m2.group(4))
                ok = (TAB in names) != bool(m2.group(3))
            elif lines[i] == ind + "else:" and branches:
                ok = True
            else:
                break
            i += 1
            body = []
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > len(ind)):
                body.append(lines[i]); i += 1
            while body and not body[-1].strip():
                body.pop(); i -= 1
            branches.append((ok, body))
        chosen = next((b for ok, b in branches if ok), [])
        if chosen:
            cut = min(len(b) - len(b.lstrip()) for b in chosen if b.strip()) - len(ind)
            out.extend(ind + b[len(ind) + cut:] if b.strip() else "" for b in chosen)
        # khối thân rỗng (vd. chỉ có nhánh của framework khác): bỏ hẳn
    return "\n".join(out)


def pytorch_only(md):
    """Bỏ khối code và đoạn văn của MXNet/JAX/TensorFlow.

    Hai cú pháp của d2lbook: mới (dòng đầu khối "%%tab ...", khối không nhãn dùng cho mọi
    framework) và cũ (dòng đầu "#@tab ...", khối không nhãn chỉ dành cho MXNet)."""
    old_style = re.search(r"^#@tab ", md, re.M) is not None
    def cell(m):
        attrs, code = m.group(1), m.group(2)
        first = code.split("\n", 1)[0]
        if "%load_ext d2lbook.tab" in code:
            return ""
        if first.startswith("%%tab") or first.startswith("#@tab"):
            tabs = tabs_of(first.replace("#@tab", "%%tab", 1))
            if TAB not in tabs and "all" not in tabs:
                return ""
            code = code.split("\n", 1)[1] if "\n" in code else ""
        elif old_style:
            return ""                         # khối không nhãn ở cú pháp cũ: MXNet
        code = resolve_selected(code.rstrip("\n"))
        if not code.strip():
            return ""
        return "```{.python .input" + attrs + "}\n" + code + "\n```\n"
    md = CELL.sub(cell, md)

    def tab_block(m):
        names = [t.strip() for t in m.group(1).split(",")]
        return m.group(2) if TAB in names or "all" in names else ""
    md = re.sub(r"^:begin_tab:`([^`]*)`\n(.*?)^:end_tab:\n?", tab_block, md, flags=re.S | re.M)
    return re.sub(r"\n{3,}", "\n\n", md)


# ---------------------------------------------------------------- trích dẫn

def read_bib(src):
    bib = {}
    for chunk in re.split(r"\n(?=@\w+\{)", "\n" + (src / "d2l.bib").read_text()):
        m = re.match(r"@\w+\{\s*([^,\s]+),(.*)", chunk.strip(), re.S)
        if not m:
            continue
        key, body = m.group(1), m.group(2)
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body):
            fields[f.group(1).lower()] = re.sub(r"[{}]", "", " ".join(f.group(2).split()))
        bib[key] = fields
    return bib


def authors_of(fields):
    names = [a.strip() for a in fields.get("author", "").split(" and ") if a.strip()]
    last = [n.split(",")[0].strip() if "," in n else n.split()[-1] for n in names]
    return last


def cite_short(fields, key):
    last = authors_of(fields) or [key.split(".")[0]]
    who = last[0] if len(last) == 1 else (f"{last[0]} và {last[1]}" if len(last) == 2 else f"{last[0]} *et al.*")
    return who, fields.get("year", "")


def cite_full(fields):
    last = authors_of(fields)
    names = [a.strip() for a in fields.get("author", "").split(" and ") if a.strip()]
    who = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
    venue = fields.get("journal") or fields.get("booktitle") or fields.get("publisher") or ""
    s = f"{who} ({fields.get('year', '')}). *{fields.get('title', '')}*."
    if venue:
        s += f" {venue}."
    return s if last else s


# ---------------------------------------------------------------- nhãn toàn sách

class Book:
    def __init__(self, src):
        self.src = src
        self.layout = [("N", "chapter_notation", [])] + book_layout(src)
        self.bib = read_bib(src)
        self.labels = {}   # nhãn → (loại, số, trang, chương, mục)
        for num, d, secs in self.layout:
            self.scan(num, d, None, (src / d / "index.md").read_text())
            for k, s in enumerate(secs, 1):
                f = src / d / f"{s}.md"
                if f.exists():
                    self.scan(num, d, (k, s), f.read_text())

    def scan(self, num, d, sec, md):
        """Đánh số giống d2lbook: mục/tiểu mục theo cấp tiêu đề, hình và công thức theo mục."""
        md = CELL.sub("", md)
        md = re.sub(r"^```.*?^```", "", md, flags=re.S | re.M)
        prefix = f"{num}" if sec is None else f"{num}.{sec[0]}"
        page = page_name(num, sec[0] if sec else None)
        heads, fig, eq, tab = [0, 0, 0], 0, 0, 0
        last = None
        fprefix = prefix   # hình, công thức: đánh số theo mục; trang chương không tách tệp (chương 1) thì theo tiêu đề ##
        for ln in md.split("\n"):
            h = re.match(r"^(#{1,4}) (.*)", ln)
            if h:
                lvl = len(h.group(1))
                if lvl == 1:
                    last = ("sec" if sec else "chap", prefix)
                else:
                    heads[lvl - 2] += 1
                    for j in range(lvl - 1, 3):
                        heads[j] = 0
                    last = ("sec", prefix + "." + ".".join(str(x) for x in heads[:lvl - 1]))
                    if sec is None and lvl == 2:
                        fprefix, fig, eq, tab = f"{prefix}.{heads[0]}", 0, 0, 0
                continue
            if re.match(r"^!\[.*\]\(.*\)", ln):
                fig += 1
                last = ("fig", f"{fprefix}.{fig}")
                continue
            lab = re.match(r"^:label:`([^`]+)`", ln)
            if lab and last:
                kind, n = last
                if lab.group(1).startswith("tab_"):
                    tab += 1
                    kind, n = "tab", f"{fprefix}.{tab}"
                self.labels[lab.group(1)] = (kind, n, page, num, d, sec[1] if sec else "index")
                continue
            eql = re.match(r"^:eqlabel:`([^`]+)`", ln)
            if eql:
                eq += 1
                self.labels[eql.group(1)] = ("eq", f"{fprefix}.{eq}", page, num, d, sec[1] if sec else "index")

    def built(self, page):
        return (DOCS / "dl" / f"{page}.md").exists() or page in self.building

    def link(self, label, text):
        kind, n, page, num, d, s = self.labels[label]
        if self.built(page):
            anchor = f"?id={label.replace('_', '-').lower()}" if kind in ("fig", "tab") else ""
            return f"[{text}](dl/{page}{anchor})"
        return f"[{text}]({D2L_URL}/{d}/{s}.html)"

    def ref(self, label):
        if label not in self.labels:
            print(f"  ! thiếu nhãn {label}", file=sys.stderr)
            return label
        kind, n = self.labels[label][:2]
        word = {"chap": "Chương", "sec": "Mục", "fig": "Hình", "tab": "Bảng", "eq": ""}[kind]
        if kind == "chap" and isinstance(self.labels[label][3], str):
            word = "Phụ lục"
        text = f"({n})" if kind == "eq" else f"{word} {n}"
        if self.labels[label][3] == "N":
            text = "phần Ký hiệu"
        return self.link(label, text)


# ---------------------------------------------------------------- dựng trang

def strip_marks(md):
    md = re.sub(r"\(\*\*(.+?)\*\*\)", r"\1", md, flags=re.S)
    md = re.sub(r"[ \t]*[\(\[]~~.+?~~[\)\]]", "", md, flags=re.S)   # (~~…~~), [~~…~~]: chỉ có trong slide, không có trong sách
    md = re.sub(r"\[\*\*(.+?)\*\*\]", r"\1", md, flags=re.S)
    return md


class Page:
    def __init__(self, book, num, d, sec, md, title_prefix):
        self.book, self.num, self.d, self.sec, self.md = book, num, d, sec, md
        self.page = page_name(num, sec)
        self.prefix = title_prefix
        self.cites = []

    def cite(self, keys, textual=False):
        parts = []
        for key in [k.strip() for k in keys.split(",")]:
            f = self.book.bib.get(key)
            if not f:
                print(f"  ! thiếu trích dẫn {key}", file=sys.stderr)
                parts.append(key)
                continue
            if key not in self.cites:
                self.cites.append(key)
            who, year = cite_short(f, key)
            anchor = f"dl/{self.page}?id=ref-{key.lower().replace('.', '-')}"
            parts.append(f"[{who} ({year})]({anchor})" if textual else f"[{who}, {year}]({anchor})")
        if textual:
            return ", ".join(parts)
        return "(" + "; ".join(parts) + ")"

    def render(self):
        md = strip_marks(self.md)
        # bảo vệ code trước khi xử lý chữ
        codes = []

        def keep_cell(m):
            attrs, code = m.group(1), m.group(2).rstrip("\n")
            code = "\n".join(ln for ln in code.split("\n") if not ln.startswith("%"))   # lệnh magic của Jupyter
            marker = "<!-- run gpu -->\n" if ".gpu" in attrs else "" if ".norun" in attrs else "<!-- run -->\n"
            codes.append(f"{marker}```python\n{code}\n```\n")
            return f"\x00CODE{len(codes) - 1}\x00\n"
        md = CELL.sub(keep_cell, md)

        def keep_fence(m):
            codes.append(m.group(0))
            return f"\x00CODE{len(codes) - 1}\x00"
        md = re.sub(r"^```.*?^```\n?", keep_fence, md, flags=re.S | re.M)

        md = re.sub(r":cite:`([^`]+)`", lambda m: self.cite(m.group(1)), md)
        md = re.sub(r":citet:`([^`]+)`", lambda m: self.cite(m.group(1), True), md)
        md = re.sub(r":numref:`([^`]+)`", lambda m: self.book.ref(m.group(1)), md)
        md = re.sub(r":eqref:`([^`]+)`", lambda m: self.book.ref(m.group(1)), md)

        out, heads, fig = [], [0, 0, 0], 0
        fbase = f"{self.num}" if self.sec is None else f"{self.num}.{self.sec}"
        lines = md.split("\n")
        i = 0
        while i < len(lines):
            ln = lines[i]
            h = re.match(r"^(#{1,4}) (.*)", ln)
            if h:
                lvl, title = len(h.group(1)), h.group(2).strip()
                if lvl == 1:
                    out.append(f"# {self.prefix} {title}".replace("#  ", "# "))
                elif self.num == "N":
                    out.append(f"{h.group(1)} {title}")
                else:
                    heads[lvl - 2] += 1
                    for j in range(lvl - 1, 3):
                        heads[j] = 0
                    num = ".".join(str(x) for x in heads[:lvl - 1])
                    base = f"{self.num}" if self.sec is None else f"{self.num}.{self.sec}"
                    out.append(f"{h.group(1)} {base}.{num} {title}")
                    if self.sec is None and lvl == 2:
                        fbase, fig = f"{self.num}.{heads[0]}", 0
                i += 1
                continue
            fm = re.match(r"^!\[(.*)\]\((.*?)\)\s*$", ln)
            if fm:
                fig += 1
                cap, path = fm.group(1), fm.group(2)
                width, fid = None, ""
                j = i + 1
                while j < len(lines) and re.match(r"^:(label|width):`", lines[j]):
                    w = re.match(r"^:width:`([^`]+)`", lines[j])
                    if w:
                        width = w.group(1).replace("px", "")
                    lb = re.match(r"^:label:`([^`]+)`", lines[j])
                    if lb:
                        fid = f' id="{lb.group(1).replace("_", "-").lower()}"'
                    j += 1
                src_img = path.split("/")[-1]
                self.copy_img(path)
                base = fbase
                wattr = f' width="{width}"' if width else ""
                out += [f"<div align=center{fid}>",
                        f'<img{wattr} src="../img/dl/{src_img}" alt="Hình {base}.{fig}"/>',
                        "</div>",
                        f"<div align=center>Hình {base}.{fig}: {cap}</div>"]
                i = j
                continue
            # chú thích bảng: dòng ":Chú thích" rồi ":label:`tab_...`" ngay trước bảng
            if re.match(r"^:[^:`]", ln) and i + 1 < len(lines) and re.match(r"^:label:`tab_", lines[i + 1]):
                lab = re.match(r"^:label:`([^`]+)`", lines[i + 1]).group(1)
                n = self.book.labels[lab][1]
                out.append(f'<div align=center id="{lab.replace("_", "-").lower()}">Bảng {n}: {ln[1:].strip()}</div>')
                out.append("")
                i += 2
                continue
            if re.match(r"^:(label|width):`", ln):
                i += 1
                continue
            eql = re.match(r"^:eqlabel:`([^`]+)`", ln)
            if eql:
                n = self.book.labels[eql.group(1)][1]
                # gắn \tag vào $$ ngay phía trên
                k = len(out) - 1
                while k >= 0 and not out[k].strip():
                    k -= 1
                if out[k].rstrip().endswith("$$"):
                    out[k] = out[k].rstrip()[:-2].rstrip() + f" \\tag{{{n}}}$$"
                i += 1
                continue
            out.append(ln)
            i += 1
        md = "\n".join(out)
        md = re.sub(r"\[Discussions\]\((https://discuss\.d2l\.ai/[^)]+)\)", r"[Thảo luận (tiếng Anh)](\1)", md)
        md = self.split_display(md)
        md = re.sub(r"\x00CODE(\d+)\x00", lambda m: codes[int(m.group(1))], md)
        if self.cites:
            refs = ["", "## Tài liệu tham khảo", ""]
            for key in self.cites:
                anchor = f"ref-{key.lower().replace('.', '-')}"
                refs.append(f'- <a id="{anchor}"></a>{cite_full(self.book.bib[key])}')
            md = md.rstrip() + "\n" + "\n".join(refs) + "\n"
        return re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"

    @staticmethod
    def split_display(md):
        """$$...$$ dính liền chữ: tách thành khối riêng cho KaTeX của docsify."""
        def sub(m):
            body = m.group(1).strip()
            if "\\\\" in body and "\\begin" not in body:
                # nhiều dòng ngoài môi trường: KaTeX cảnh báo, gói trong gathered
                body = "\\begin{gathered}\n" + body.rstrip("\\ \n") + "\n\\end{gathered}"
            return "\n\n$$\n" + body + "\n$$\n\n"
        return re.sub(r"\n?\$\$(.+?)\$\$\n?", sub, md, flags=re.S)

    def copy_img(self, path):
        src = (self.book.src / self.d / path).resolve()
        dst = DOCS / "img" / "dl" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy(src, dst)
        else:
            print(f"  ! thiếu hình {src}", file=sys.stderr)


def header(book, num, d, sec):
    url = f"{D2L_URL}/{d}/{sec or 'index'}.html"
    note = NOTES / f"ch{num:02d}{'_' + sec if sec else ''}.md" if isinstance(num, int) else None
    lines = [f"> Ghi chú của người biên soạn: trang này dịch nguyên văn từ sách *Dive into Deep Learning* "
             f"của **Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola** "
             f"([bản gốc]({url}), [mã nguồn](https://github.com/d2l-ai/d2l-en), giấy phép "
             f"[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); code theo giấy phép MIT-0). "
             f"Chỉ giữ code PyTorch. Số mục, hình và công thức giữ như bản HTML d2l.ai; "
             f"thuật ngữ đối chiếu với bản dịch [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn)."]
    if note and note.exists():
        lines += [">", "> **Đính chính** *(người biên soạn)*:", ">"]
        lines += ["> " + ln if ln.strip() else ">" for ln in note.read_text().strip().split("\n")]
    return "\n".join(lines) + "\n"


def build(book, num):
    entry = next(c for c in book.layout if c[0] == num)
    _, d, secs = entry
    vdir = VI / d
    book.building = {page_name(num)} | {page_name(num, k) for k, s in enumerate(secs, 1) if (vdir / f"{s}.md").exists()}
    (DOCS / "dl").mkdir(exist_ok=True)
    jobs = [(None, "index")] + [(k, s) for k, s in enumerate(secs, 1)]
    for k, s in jobs:
        f = vdir / f"{s}.md"
        if not f.exists():
            continue
        prefix = "" if num == "N" else f"Chương {num}" if k is None else f"{num}.{k}"
        page = Page(book, num, d, k, f.read_text(), prefix)
        body = page.render()
        title, rest = body.split("\n", 1)
        if k is None:
            rest = re.sub(r"```toc.*?```\n?", "", rest, flags=re.S)
            toc = ["", "## Các mục trong chương", ""]
            for kk, ss in enumerate(secs, 1):
                t = re.search(r"^# (.*)", (vdir / f"{ss}.md").read_text() if (vdir / f"{ss}.md").exists()
                              else (book.src / d / f"{ss}.md").read_text(), re.M).group(1)
                pg = page_name(num, kk)
                link = f"dl/{pg}" if book.built(pg) else f"{D2L_URL}/{d}/{ss}.html"
                toc.append(f"- [{num}.{kk} {t.strip()}]({link})")
            head, sep, refs = rest.partition("\n## Tài liệu tham khảo")
            rest = head.rstrip() + "\n" + "\n".join(toc) + "\n" + (sep + refs if sep else "")
        out = title + "\n\n" + header(book, num, d, None if k is None else s) + "\n" + rest.lstrip("\n")
        dst = DOCS / "dl" / f"{page_name(num, k)}.md"
        old = dst.read_text() if dst.exists() else ""
        dst.write_text(keep_outputs(old, out))
        print(f"  {dst.relative_to(ROOT)}")


def keep_outputs(old, new):
    """Giữ kết quả đã chạy (<!-- output -->...) của khối code không đổi, để build lại không mất kết quả."""
    blk = re.compile(r"(<!-- run(?: gpu)? -->\n```python\n.*?\n```\n)(\n<!-- output -->\n.*?<!-- /output -->\n)?", re.S)
    outputs = {m.group(1): m.group(2) for m in blk.finditer(old) if m.group(2)}
    return blk.sub(lambda m: m.group(1) + outputs.get(m.group(1), ""), new)


def extract(book, num):
    _, d, secs = next(c for c in book.layout if c[0] == num)
    vdir = VI / d
    vdir.mkdir(parents=True, exist_ok=True)
    for s in ["index"] + secs:
        dst = vdir / f"{s}.md"
        if dst.exists():
            print(f"  giữ {dst.relative_to(ROOT)} (đã có)")
            continue
        dst.write_text(pytorch_only((book.src / d / f"{s}.md").read_text()))
        print(f"  {dst.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("cmd", nargs="?", choices=["extract", "build"])
    ap.add_argument("chapters", nargs="*")
    a = ap.parse_args()
    book = Book(a.src)
    book.building = set()
    if a.list:
        for num, d, secs in book.layout:
            print(num, d, len(secs), "mục")
        return
    for c in a.chapters:
        num = int(c) if c.isdigit() else "N" if c.lower() in ("n", "notation") else c.upper()
        print(f"Chương {num}:")
        (extract if a.cmd == "extract" else build)(book, num)


if __name__ == "__main__":
    main()
