"""Dựng khoá "Ghi chú học sâu" (docs/dlnote) từ bản dịch của dl_note.

Nguồn: github.com/harleyszhang/dl_note (Harley Zhang, Apache-2.0), ghi chú tiếng Trung.
Bản dịch tiếng Việt (có biên tập, sửa lỗi, thêm mục cập nhật) nằm ở scripts/dlnote/vi/<trang>.md,
viết theo đường dẫn ảnh của bản gốc (`../images/<thư mục>/<tệp>`). Script này:

- thêm ghi chú nguồn (link tới đúng tệp gốc) dưới tiêu đề H1;
- đổi ảnh `<img src="../images/...">` / `![..](../images/...)` thành `<img src="../img/dlnote/...">`
  và chép ảnh từ bản clone (ảnh PNG nền trong suốt được lót nền trắng, ảnh quá rộng được thu nhỏ);
- thay dòng `<!-- theory: dl/ch12_vi#12.10 ; ... -->` bằng ô "Lý thuyết" như khoá Ôn tập;
- ghi docs/dlnote/<trang>_vi.md.

Dùng (trong env conda `ai`, cần Pillow):
  python3 -B scripts/dlnote/build.py --src <clone dl_note> compress_quant
  python3 -B scripts/dlnote/build.py --src <clone dl_note> --all
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
VI = Path(__file__).resolve().parent / "vi"
DEFAULT_SRC = Path("/tmp/claude-1000/-home-nmc-Downloads-easy-rl-vi/963822cf-d45a-4311-aa91-22a954d0ef64/scratchpad/dl_note_src/dl_note")
REPO = "https://github.com/harleyszhang/dl_note"

# Dùng lại phần chép ảnh và ô "Lý thuyết" của khoá Ôn tập.
_spec = importlib.util.spec_from_file_location("dl500_build", ROOT / "scripts" / "dl500" / "build.py")
dl500 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl500)
dl500.COURSE_LABEL.setdefault("dlnote", "Ghi chú DL")

# Trang → tệp gốc trong repo dl_note.
PAGES = {
    "math_prob": "1-math_ml_basic/深度学习数学基础-概率与信息论.md",
    "math_ml": "1-math_ml_basic/深度学习基础-机器学习基本原理.md",
    "math_grad": "1-math_ml_basic/随机梯度下降法的数学基础.md",
    "dl_conv": "2-deep_learning_basic/cnn基础部件-卷积层详解.md",
    "dl_bn": "2-deep_learning_basic/cnn基础部件-BN层详解.md",
    "dl_act": "2-deep_learning_basic/cnn基础部件-激活函数详解.md",
    "dl_loss": "2-deep_learning_basic/深度学习基础-损失函数详解.md",
    "dl_init": "2-deep_learning_basic/深度学习基础-参数初始化详解.md",
    "dl_bp": "2-deep_learning_basic/反向传播与梯度下降详解.md",
    "dl_optim": "2-deep_learning_basic/深度学习基础-优化算法详解.md",
    "dl_summary": "2-deep_learning_basic/深度学习基础总结.md",
    "bb_summary": "3-classic_backbone/经典backbone总结.md",
    "bb_resnet": "3-classic_backbone/ResNet网络详解.md",
    "bb_resnetv2": "3-classic_backbone/ResNetv2论文解读.md",
    "bb_densenet": "3-classic_backbone/DenseNet论文解读.md",
    "bb_mobilenet": "3-classic_backbone/efficient_cnn/MobileNetv1论文详解.md",
    "bb_shufflenet": "3-classic_backbone/efficient_cnn/ShuffleNetv2论文详解.md",
    "bb_repvgg": "3-classic_backbone/efficient_cnn/RepVGG论文详解.md",
    "bb_cspnet": "3-classic_backbone/efficient_cnn/CSPNet论文详解.md",
    "bb_vovnet": "3-classic_backbone/efficient_cnn/VoVNet论文解读.md",
    "alch_hyper": "4-deep_learning_alchemy/深度学习炼丹-超参数调整.md",
    "alch_norm": "4-deep_learning_alchemy/深度学习炼丹-数据标准化.md",
    "alch_imbalance": "4-deep_learning_alchemy/深度学习炼丹-不平衡样本的处理.md",
    "alch_augment": "4-deep_learning_alchemy/深度学习炼丹-数据增强.md",
    "alch_reg": "4-deep_learning_alchemy/深度学习炼丹-正则化策略.md",
    "alch_vis": "4-deep_learning_alchemy/深度学习炼丹-模型可视化.md",
    "compress_overview": "5-model_compression/深度学习模型压缩方法概述.md",
    "compress_light": "5-model_compression/模型压缩-轻量化网络总结.md",
    "compress_prune": "5-model_compression/模型压缩-剪枝算法详解.md",
    "compress_prune_torch": "5-model_compression/基于pytorch实现模型剪枝.md",
    "compress_quant": "5-model_compression/模型压缩-神经网络量化基础.md",
    "compress_kd": None,   # tệp gốc rỗng: bài do người biên soạn viết
    "deploy_overview": "6-model_deploy/模型压缩部署概述.md",
    "deploy_flops": "6-model_deploy/卷积神经网络复杂度分析.md",
    "deploy_fuse": "6-model_deploy/模型推理加速技巧-融合卷积和BN层.md",
    "deploy_onnx": "6-model_deploy/ONNX模型分析与使用.md",
    "deploy_trt": "6-model_deploy/TensorRT基础笔记.md",
    "deploy_ncnn_sample": "6-model_deploy/ncnn源码解析-sample运行.md",
    "deploy_ncnn_net": "6-model_deploy/ncnn源码解析-Net类.md",
    "deploy_chips": "6-model_deploy/AI芯片速览.md",
    # Khoá "Công cụ lập trình" (docs/tools): trang tools_<tên> → docs/tools/<tên>_vi.md
    "tools_ssh": "1-math_ml_basic/ssh远程登录服务.md",
}
IMG_VI = Path(__file__).resolve().parent / "img_vi"   # ảnh đã Việt hoá (fix_images.py)

IMG_HTML = re.compile(r'<img\s+src="\.\./images/([^"]+)"([^>]*)>')
IMG_MD = re.compile(r"!\[([^\]]*)\]\(\.\./images/([^)\s]+)\s*\)")


def header_note(page):
    if PAGES[page] is None:
        return (
            "> Ghi chú của người biên soạn: tệp tương ứng trong *dl_note* của **Harley Zhang** "
            f"([kho mã]({REPO})) để trống, nên toàn bộ bài này do người biên soạn viết, dựa trên mục "
            "chưng cất tri thức của bài tổng quan và các bài báo gốc ghi ở cuối trang."
        )
    src = f"{REPO}/blob/main/{PAGES[page]}"
    return (
        "> Ghi chú của người biên soạn: trang này dịch từ ghi chú *dl_note* của **Harley Zhang** "
        f"([bản gốc]({src}), [kho mã]({REPO}), giấy phép "
        "[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0); phần dịch có chỉnh sửa). "
        "Số mục giữ như bản gốc. Chỗ bản gốc sai, thiếu hoặc đã lỗi thời được sửa ngay trong bài và "
        "ghi ở ô **Biên tập**. Các mục có chữ **Bổ sung** do người biên soạn viết thêm."
    )


def build(page, src):
    vi = VI / f"{page}.md"
    if not vi.exists():
        sys.exit(f"chưa có bản dịch {vi}")
    images = set()

    def html_sub(m):
        images.add(m.group(1))
        return f'<img src="../img/dlnote/{m.group(1)}"{m.group(2)}>'

    def md_sub(m):
        images.add(m.group(2))
        alt = m.group(1).strip("* ").replace('"', "&quot;")
        return f'<img src="../img/dlnote/{m.group(2)}" alt="{alt}"/>'

    out, in_code, h1_done = [], False, False
    for line in vi.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not in_code:
            m = dl500.THEORY_RE.match(line)
            if m:
                out.append(dl500.theory_box(m.group(1)))
                continue
            line = IMG_MD.sub(md_sub, IMG_HTML.sub(html_sub, line))
        out.append(line)
        if not h1_done and line.startswith("# "):
            out += ["", header_note(page)]
            h1_done = True

    for rel in sorted(images):
        s, d = src / "images" / rel, DOCS / "img" / "dlnote" / rel
        if (IMG_VI / rel).exists():
            s = IMG_VI / rel
        if s.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            if s.suffix.lower() == ".svg":
                d.write_bytes(s.read_bytes())
            else:
                dl500.copy_image(s, d)
        elif not d.exists():
            sys.exit(f"{page}: không thấy ảnh {s}")

    dest = (DOCS / "tools" / f"{page[6:]}_vi.md") if page.startswith("tools_") else (DOCS / "dlnote" / f"{page}_vi.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    print(f"{page}: {dest.relative_to(ROOT)} ({len(images)} ảnh)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("pages", nargs="*")
    a = ap.parse_args()
    pages = sorted(p.stem for p in VI.glob("*.md")) if a.all else a.pages
    for p in pages:
        if p not in PAGES:
            sys.exit(f"trang {p} chưa có trong PAGES")
        build(p, a.src)


if __name__ == "__main__":
    main()
