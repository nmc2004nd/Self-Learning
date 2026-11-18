"""Chạy code trên các trang Code rồi ghi sẵn kết quả (chữ, hình PNG, ảnh GIF) vào trang.

Trong Markdown, khối code cần chạy được đặt ngay sau dòng <!-- run --> (dòng này không hiện ra
trên trang). Các khối của một trang chạy lần lượt trong cùng một không gian biến, như notebook.
Ngay sau mỗi khối, script chèn (hoặc thay) một vùng giữa <!-- output --> và <!-- /output -->:
khối ```output chứa những gì in ra, rồi tới các hình matplotlib và GIF (viz.save_gif).
Hình lưu ở docs/img/code/<tên trang>/.

Khối huấn luyện nặng (khoá Học sâu) đặt sau <!-- run gpu -->: mặc định KHÔNG chạy, trang hiện
dòng "chưa có kết quả" (kết quả cũ, nếu có, được giữ). Chạy chúng bằng --gpu trên máy có GPU.
--smoke chạy mọi khối, kể cả khối GPU, ở chế độ thu nhỏ (D2L_SMOKE=1: 1 epoch, vài batch) chỉ để
kiểm tra code không lỗi; không ghi gì vào trang.

Dùng:  python3 scripts/build_code_pages.py [--gpu | --smoke] docs/chapter3/chapter3_code_vi.md [...]
"""

import ast
import contextlib
import io
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True    # không để lại __pycache__ trong docs/ và notebooks/
ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
sys.path.insert(0, str(DOCS / "code" / "py"))
sys.path.insert(0, str(DOCS / "code" / "dl"))
MODE = "gpu" if "--gpu" in sys.argv else "smoke" if "--smoke" in sys.argv else "cpu"
if MODE == "smoke":
    os.environ["D2L_SMOKE"] = "1"
    try:  # vòng lặp tự viết trên các trang: mỗi DataLoader chỉ cho 2 batch
        import itertools
        import torch.utils.data as _tud
        _iter = _tud.DataLoader.__iter__
        _tud.DataLoader.__iter__ = lambda self: itertools.islice(_iter(self), 2)
        _tud.DataLoader.__len__ = lambda self: 2
    except ImportError:
        pass
PENDING = ('<div class="gpu-pending">Chưa có kết quả: khối này huấn luyện mô hình nên sẽ được chạy '
           'trên máy có GPU. Bạn có thể tự chạy code trên Colab hoặc máy của mình.</div>')

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 100, "font.size": 9, "savefig.bbox": "tight"})

import viz  # noqa: E402

BLOCK = re.compile(r"<!-- run( gpu)? -->\n```python\n(.*?)\n```\n(?:\n<!-- output -->\n(.*?)<!-- /output -->\n)?", re.S)


def run_block(code, name, scope, notebook=False):
    """Chạy một khối. notebook=True (code d2l): như Jupyter, in giá trị biểu thức cuối khối nếu khác None."""
    tree = ast.parse(code, name)
    last = tree.body[-1] if notebook and tree.body and isinstance(tree.body[-1], ast.Expr) else None
    if last is not None:
        tree.body.pop()
    exec(compile(tree, name, "exec"), scope)
    if last is not None:
        val = eval(compile(ast.Expression(last.value), name, "eval"), scope)
        if val is not None and not hasattr(val, "savefig") and type(val).__module__.split(".")[0] != "matplotlib":
            print(repr(val))


def build(md_path):
    md_path = Path(md_path).resolve()
    text = md_path.read_text()
    name = md_path.stem.removesuffix("_vi")
    # Khoá Toán/Học máy dùng tên chương trùng với mathrl (ch07_code...): thêm tiền tố để không đè thư mục hình của nhau.
    prefix = {"ml": "ml_", "math": "math_", "dl": "dl_"}.get(md_path.parent.name, "")
    name = prefix + name
    fig_dir = DOCS / "img" / "code" / name
    if MODE == "smoke":
        fig_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "smoke_figs" / name
    keep = set()   # hình của khối GPU bị bỏ qua: giữ lại
    for m in BLOCK.finditer(text):
        if m.group(1) and MODE == "cpu" and m.group(3):
            keep |= set(re.findall(r'src="[^"]*/([^"/]+)"', m.group(3)))
    if fig_dir.exists():
        for f in fig_dir.iterdir():
            if f.name not in keep:
                f.unlink()
    fig_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CODE_FIG_DIR"] = str(fig_dir)
    rel = os.path.relpath(fig_dir, md_path.parent)

    # Chạy từ docs/code/py (khoá Học sâu: docs/code/dl), giống hướng dẫn "Chạy trên máy" trên trang.
    if md_path.parent.name == "dl":
        # code của sách ghi tệp ra thư mục hiện hành (mục 6.6): chạy trong thư mục tạm, không làm bẩn kho mã
        import tempfile
        os.chdir(tempfile.mkdtemp(prefix="dl_run_"))
    else:
        os.chdir(DOCS / "code" / "py")
    scope = {"__name__": "__main__"}
    out_parts, pos, n = [], 0, 0
    for m in BLOCK.finditer(text):
        n += 1
        gpu, code, old = m.group(1), m.group(2), m.group(3)
        head = "<!-- run gpu -->" if gpu else "<!-- run -->"
        if gpu and MODE == "cpu":
            block = head + "\n```python\n" + code + "\n```\n"
            block += "\n<!-- output -->\n" + (old if old and PENDING not in old else PENDING + "\n") + "<!-- /output -->\n"
            out_parts.append(text[pos:m.start()] + block)
            pos = m.end()
            print(f"  khối {n}: bỏ qua (GPU)")
            continue
        buf = io.StringIO()
        viz.SAVED.clear()
        plt.close("all")
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            run_block(code, f"{md_path.name}:khối {n}", scope, notebook=md_path.parent.name == "dl")
        dt = time.perf_counter() - t0
        pieces = []
        printed = buf.getvalue().rstrip("\n")
        if printed:
            pieces.append("```output\n" + printed + "\n```")
        for k, num in enumerate(plt.get_fignums(), 1):
            fn = f"{n:02d}_{k}.png"
            plt.figure(num).savefig(fig_dir / fn)
            pieces.append(f'<div align=center><img src="{rel}/{fn}" alt="Hình do khối code vẽ"/></div>')
        plt.close("all")
        for fn in viz.SAVED:
            pieces.append(f'<div align=center><img src="{rel}/{fn}" alt="Ảnh động do khối code tạo"/></div>')
        block = head + "\n```python\n" + code + "\n```\n"
        if pieces:
            block += "\n<!-- output -->\n" + "\n\n".join(pieces) + "\n<!-- /output -->\n"
        out_parts.append(text[pos:m.start()] + block)
        pos = m.end()
        print(f"  khối {n}: {dt:.1f} s, {len(printed.splitlines())} dòng, "
              f"{len(pieces) - bool(printed)} hình")
    out_parts.append(text[pos:])
    if MODE == "smoke":
        print(f"{md_path.relative_to(ROOT)}: chạy thử {n} khối, không lỗi")
        return
    md_path.write_text("".join(out_parts))
    size = sum(f.stat().st_size for f in fig_dir.iterdir()) / 1024
    print(f"{md_path.relative_to(ROOT)}: {n} khối, hình {size:.0f} KB")


if __name__ == "__main__":
    for p in [a for a in sys.argv[1:] if not a.startswith("--")]:
        build(p)
