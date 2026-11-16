"""Vẽ hình minh hoạ cho bài bổ sung "Đánh giá mô hình" (docs/ml/eval_vi.md).

Dùng:  python3 -B scripts/ml_extra/eval_figs.py
Hình SVG lưu ở docs/img/ml/eval/. Các con số in ra màn hình là số được dẫn trong bài.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Ellipse, Circle

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parents[2] / "docs" / "img" / "ml" / "eval"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "path", "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

POS, NEG = "#e8487f", "#2f80f5"           # lớp dương, lớp âm
TP_C, TN_C, FP_C, FN_C = "#16a34a", "#0ea5a4", "#f08a24", "#dc2626"
INK, MUTED = "#1f2340", "#8a90b4"


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print("  ->", name + ".svg")


# ---------------------------------------------------------------- 1. chia dữ liệu
def fig_split():
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.0), gridspec_kw={"height_ratios": [1, 2.3]})
    ax = axes[0]
    parts = [(0, 0.6, "#2f80f5", "Huấn luyện (60%)\nhọc tham số"),
             (0.6, 0.2, "#f08a24", "Xác thực (20%)\nchọn mô hình"),
             (0.8, 0.2, "#e8487f", "Kiểm tra (20%)\ndùng một lần, ở cuối")]
    for x, w, c, t in parts:
        ax.add_patch(FancyBboxPatch((x + 0.004, 0.1), w - 0.008, 0.8, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=c, ec="none", alpha=0.85))
        ax.text(x + w / 2, 0.5, t, ha="center", va="center", color="white", fontsize=9.5, weight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(a) Chia một lần: huấn luyện / xác thực / kiểm tra", loc="left")

    ax = axes[1]
    K = 5
    for i in range(K):
        y = K - 1 - i
        for j in range(K):
            val = (i == j)
            ax.add_patch(FancyBboxPatch((0.16 + j * 0.13 + 0.004, y + 0.15), 0.122, 0.7,
                                        boxstyle="round,pad=0,rounding_size=0.03",
                                        fc="#f08a24" if val else "#2f80f5", ec="none", alpha=0.85))
        ax.text(0.15, y + 0.5, f"Lần {i + 1}", ha="right", va="center", fontsize=9.5)
        ax.text(0.84, y + 0.5, f"→ điểm số $s_{i + 1}$", ha="left", va="center", fontsize=9.5)
    ax.text(0.5, -0.45, r"Điểm xác thực chéo $=\frac{1}{5}(s_1+s_2+\dots+s_5)$; mỗi điểm dữ liệu làm xác thực đúng một lần",
            ha="center", fontsize=9.5)
    ax.set_xlim(0, 1.02); ax.set_ylim(-0.8, K); ax.axis("off")
    ax.set_title("(b) Xác thực chéo 5-fold trên phần huấn luyện (cam: phần xác thực của mỗi lần)", loc="left")
    fig.tight_layout()
    save(fig, "split")


# ---------------------------------------------------------------- 2. ma trận nhầm lẫn
def fig_confusion():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cells = [  # (cột, hàng-từ-trên, màu, tên, giải thích)
        (0, 0, TP_C, "TP", "True Positive\nDương thật\n(bắt đúng)"),
        (1, 0, FN_C, "FN", "False Negative\nÂm giả\n(bỏ sót)"),
        (0, 1, FP_C, "FP", "False Positive\nDương giả\n(báo nhầm)"),
        (1, 1, TN_C, "TN", "True Negative\nÂm thật\n(loại đúng)"),
    ]
    for c, r, col, name, txt in cells:
        x, y = 1 + c * 2, 2 - r * 2
        ax.add_patch(FancyBboxPatch((x + 0.05, y + 0.05), 1.9, 1.9, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=col, ec="none", alpha=0.18))
        ax.add_patch(FancyBboxPatch((x + 0.05, y + 0.05), 1.9, 1.9, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="none", ec=col, lw=2))
        ax.text(x + 1, y + 1.45, name, ha="center", va="center", fontsize=20, weight="bold", color=col)
        ax.text(x + 1, y + 0.72, txt, ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.35)
    ax.text(2, 4.35, "Dự đoán: dương", ha="center", fontsize=10.5, weight="bold")
    ax.text(4, 4.35, "Dự đoán: âm", ha="center", fontsize=10.5, weight="bold")
    ax.text(3, 4.85, "MÔ HÌNH DỰ ĐOÁN", ha="center", fontsize=9, color=MUTED)
    ax.text(0.85, 3, "Thực tế:\ndương", ha="right", va="center", fontsize=10.5, weight="bold")
    ax.text(0.85, 1, "Thực tế:\nâm", ha="right", va="center", fontsize=10.5, weight="bold")
    ax.text(-0.75, 2, "NHÃN THẬT", ha="center", va="center", rotation=90, fontsize=9, color=MUTED)
    ax.annotate("", xy=(5.25, 2.1), xytext=(5.25, 3.9), arrowprops=dict(arrowstyle="-", color=POS, lw=2))
    ax.text(5.4, 3, "$P$ = TP + FN\n(tổng số\nca dương)", va="center", fontsize=9, color=POS)
    ax.annotate("", xy=(5.25, 0.1), xytext=(5.25, 1.9), arrowprops=dict(arrowstyle="-", color=NEG, lw=2))
    ax.text(5.4, 1, "$N$ = FP + TN\n(tổng số\nca âm)", va="center", fontsize=9, color=NEG)
    ax.set_xlim(-1.1, 6.6); ax.set_ylim(-0.1, 5.1); ax.set_aspect("equal"); ax.axis("off")
    save(fig, "confusion")


# ---------------------------------------------------------------- 3. precision và recall bằng chấm
def dots_layout(rng):
    # 30 dương bên trái, 30 âm bên phải trong hình chữ nhật [0,2]x[0,1.2]
    pos = np.c_[rng.uniform(0.08, 0.92, 30), rng.uniform(0.08, 1.12, 30)]
    neg = np.c_[rng.uniform(1.08, 1.92, 30), rng.uniform(0.08, 1.12, 30)]
    return pos, neg


def fig_precision_recall():
    rng = np.random.default_rng(4)
    # lấy mẫu kiểu poisson-disk thô để các chấm không chồng nhau
    def spread(xlo, xhi, n):
        pts = []
        while len(pts) < n:
            p = np.array([rng.uniform(xlo, xhi), rng.uniform(0.08, 1.12)])
            if all(np.hypot(*(p - q)) > 0.105 for q in pts):
                pts.append(p)
        return np.array(pts)
    pos, neg = spread(0.07, 0.93, 26), spread(1.07, 1.93, 26)
    cx, cy, rx, ry = 0.95, 0.6, 0.55, 0.47
    inside = lambda P: ((P[:, 0] - cx) / rx) ** 2 + ((P[:, 1] - cy) / ry) ** 2 <= 1
    ip, ineg = inside(pos), inside(neg)
    TP, FN, FP, TN = ip.sum(), (~ip).sum(), ineg.sum(), (~ineg).sum()
    print(f"  chấm: TP={TP} FN={FN} FP={FP} TN={TN}  precision={TP/(TP+FP):.2f} recall={TP/(TP+FN):.2f}")

    fig = plt.figure(figsize=(9.6, 4.0))
    ax = fig.add_axes([0.0, 0.0, 0.56, 1.0])
    ax.add_patch(Rectangle((0, 0), 1, 1.2, fc=POS, alpha=0.07, ec="none"))
    ax.add_patch(Rectangle((1, 0), 1, 1.2, fc=NEG, alpha=0.07, ec="none"))
    ax.add_patch(Rectangle((0, 0), 2, 1.2, fc="none", ec=INK, lw=1))
    ax.plot([1, 1], [0, 1.2], color=INK, lw=1)
    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, fc="#fde68a", alpha=0.45, ec="#b45309", lw=1.6, zorder=1))
    ax.scatter(pos[ip, 0], pos[ip, 1], s=46, c=TP_C, zorder=3, edgecolors="white", linewidths=0.6)
    ax.scatter(pos[~ip, 0], pos[~ip, 1], s=46, c=FN_C, zorder=3, edgecolors="white", linewidths=0.6)
    ax.scatter(neg[ineg, 0], neg[ineg, 1], s=46, c=FP_C, marker="s", zorder=3, edgecolors="white", linewidths=0.6)
    ax.scatter(neg[~ineg, 0], neg[~ineg, 1], s=46, c=TN_C, marker="s", zorder=3, edgecolors="white", linewidths=0.6)
    ax.text(0.5, 1.27, "Thực tế dương (●)", ha="center", color=POS, weight="bold")
    ax.text(1.5, 1.27, "Thực tế âm (■)", ha="center", color=NEG, weight="bold")
    ax.text(cx, cy - ry - 0.09, "vùng mô hình đoán dương", ha="center", color="#b45309", fontsize=9)
    leg = [(TP_C, "o", f"TP = {TP}"), (FN_C, "o", f"FN = {FN}"), (FP_C, "s", f"FP = {FP}"), (TN_C, "s", f"TN = {TN}")]
    for i, (c, m, t) in enumerate(leg):
        ax.scatter([0.12 + i * 0.5], [-0.13], s=40, c=c, marker=m)
        ax.text(0.17 + i * 0.5, -0.13, t, va="center", fontsize=9.5)
    ax.set_xlim(-0.05, 2.05); ax.set_ylim(-0.24, 1.36); ax.set_aspect("equal"); ax.axis("off")

    # hai "phân số bằng hình" bên phải
    def frac(axr, title, top, bottom, value):
        axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
        axr.text(0.0, 0.93, title, fontsize=11, weight="bold", va="top")
        def row(y, items):
            tot = sum(n for _, _, n in items); dx = min(0.075, 0.92 / tot)
            x = 0.5 - dx * (tot - 1) / 2
            for c, m, n in items:
                for _ in range(n):
                    axr.scatter([x], [y], s=min(44, 3600 * dx * dx), c=c, marker=m, clip_on=False); x += dx
        row(0.62, top)
        axr.plot([0.04, 0.96], [0.5, 0.5], color=INK, lw=1.2)
        row(0.36, bottom)
        axr.text(0.5, 0.1, value, ha="center", fontsize=10.5)
    frac(fig.add_axes([0.6, 0.52, 0.38, 0.46]), "Precision: trong các ca đoán dương,\nbao nhiêu ca đúng?",
         [(TP_C, "o", TP)], [(TP_C, "o", TP), (FP_C, "s", FP)],
         f"= TP / (TP + FP) = {TP}/{TP + FP} = {TP / (TP + FP):.2f}")
    frac(fig.add_axes([0.6, 0.02, 0.38, 0.46]), "Recall: trong các ca dương thật,\nmô hình tìm ra bao nhiêu?",
         [(TP_C, "o", TP)], [(TP_C, "o", TP), (FN_C, "o", FN)],
         f"= TP / (TP + FN) = {TP}/{TP + FN} = {TP / (TP + FN):.2f}")
    save(fig, "precision_recall")


# ---------------------------------------------------------------- 4. dữ liệu mất cân bằng
def fig_imbalance():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    rng = np.random.default_rng(2)
    idx = rng.choice(500, 5, replace=False)
    yy, xx = np.divmod(np.arange(500), 25)
    for ax, title, found in [(axes[0], "Mô hình A: luôn đoán \"khoẻ\"", []),
                             (axes[1], "Mô hình B: bắt được 4/5 ca bệnh,\nbáo nhầm 15 người khoẻ", idx[:4])]:
        col = np.full(500, "#cfd6ea", dtype=object)
        col[idx] = FN_C
        if len(found):
            col[found] = TP_C
            fp = rng.choice(np.setdiff1d(np.arange(500), idx), 15, replace=False)
            col[fp] = FP_C
        ax.scatter(xx, -yy, s=26, c=list(col), marker="o")
        ax.set_title(title, fontsize=10.5)
        ax.set_aspect("equal"); ax.axis("off")
    axes[0].text(12, -21.5, "accuracy = 495/500 = 99.0%\nrecall = 0/5 = 0%", ha="center", va="top", fontsize=10.5)
    axes[1].text(12, -21.5, "accuracy = 484/500 = 96.8%\nrecall = 4/5 = 80%,  precision = 4/19 = 21%",
                 ha="center", va="top", fontsize=10.5)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="", marker="o", ms=7, color=c) for c in ["#cfd6ea", FN_C, TP_C, FP_C]]
    fig.legend(hs, ["khoẻ, đoán khoẻ", "bệnh nhưng bị bỏ sót (FN)", "bệnh, phát hiện (TP)", "khoẻ, báo nhầm (FP)"],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12), fontsize=9,
               title="500 người, trong đó 5 người bệnh (1%)", title_fontsize=9)
    save(fig, "imbalance")


# ---------------------------------------------------------------- 5. ngưỡng quyết định
def scores(n_pos, n_neg, sep, rng):
    """Điểm số kiểu logistic: logit ~ N(±sep/2, 1)."""
    s_pos = 1 / (1 + np.exp(-rng.normal(sep / 2, 1, n_pos)))
    s_neg = 1 / (1 + np.exp(-rng.normal(-sep / 2, 1, n_neg)))
    return s_pos, s_neg


def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def fig_threshold():
    x = np.linspace(-4.5, 4.5, 600)
    mu, sd = 1.0, 1.0
    fp_ = gauss(x, mu, sd); fn_ = gauss(x, -mu, sd)
    from math import erf, sqrt
    Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, t in zip(axes, [-1.0, 0.0, 1.0]):
        ax.fill_between(x, fn_, where=x >= t, color=FP_C, alpha=0.55, lw=0)
        ax.fill_between(x, fn_, where=x < t, color=TN_C, alpha=0.30, lw=0)
        ax.fill_between(x, fp_, where=x >= t, color=TP_C, alpha=0.45, lw=0)
        ax.fill_between(x, fp_, where=x < t, color=FN_C, alpha=0.55, lw=0)
        ax.plot(x, fn_, color=NEG, lw=1.8); ax.plot(x, fp_, color=POS, lw=1.8)
        ax.axvline(t, color=INK, lw=1.6, ls="--")
        rec = 1 - Phi((t - mu) / sd); fpr = 1 - Phi((t + mu) / sd)
        prec = rec / (rec + fpr)
        ax.set_title(f"ngưỡng $t={t:+.0f}$:  recall {rec:.2f}, precision {prec:.2f}", fontsize=10)
        ax.set_xlabel("điểm số của mô hình (logit)")
        ax.set_yticks([])
        print(f"  ngưỡng {t:+.0f}: recall={rec:.3f} FPR={fpr:.3f} precision={prec:.3f}")
    axes[0].text(-1.9, 0.43, "lớp âm", color=NEG, ha="center", weight="bold")
    axes[0].text(1.9, 0.43, "lớp dương", color=POS, ha="center", weight="bold")
    axes[0].set_ylim(0, 0.48)
    handles = [Rectangle((0, 0), 1, 1, fc=c, alpha=a) for c, a in [(TP_C, .45), (FN_C, .55), (FP_C, .55), (TN_C, .3)]]
    fig.legend(handles, ["TP", "FN (bỏ sót)", "FP (báo nhầm)", "TN"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    save(fig, "threshold")


def curves(s_pos, s_neg):
    """Trả về FPR, TPR, precision theo các ngưỡng giảm dần (tự viết, giống trang Code)."""
    s = np.r_[s_pos, s_neg]; y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
    o = np.argsort(-s, kind="mergesort"); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = np.r_[0, tp / len(s_pos)]; fpr = np.r_[0, fp / len(s_neg)]
    prec = np.r_[1, tp / (tp + fp)]
    return fpr, tpr, prec, s[o]


def fig_tradeoff():
    rng = np.random.default_rng(0)
    s_pos, s_neg = scores(300, 700, 2.0, rng)
    ts = np.linspace(0.02, 0.98, 97)
    P = np.array([((s_pos >= t).sum()) / max(1, (s_pos >= t).sum() + (s_neg >= t).sum()) for t in ts])
    R = np.array([(s_pos >= t).mean() for t in ts])
    F = 2 * P * R / np.maximum(P + R, 1e-12)
    k = int(np.argmax(F))
    print(f"  tradeoff: F1 max={F[k]:.3f} tại t={ts[k]:.2f} (P={P[k]:.3f}, R={R[k]:.3f})")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ts, P, color="#7c3aed", lw=2.2, label="precision")
    ax.plot(ts, R, color=TP_C, lw=2.2, label="recall")
    ax.plot(ts, F, color=INK, lw=1.6, ls="--", label="F1")
    ax.axvline(ts[k], color=MUTED, lw=1, ls=":")
    ax.scatter([ts[k]], [F[k]], c=INK, zorder=3)
    ax.annotate(f"F1 lớn nhất = {F[k]:.2f}\ntại ngưỡng {ts[k]:.2f}", (ts[k], F[k]), (0.3, 0.12),
                arrowprops=dict(arrowstyle="->", color=INK), fontsize=9)
    ax.set_xlabel("ngưỡng t trên xác suất dự đoán"); ax.set_ylim(0, 1.03); ax.set_xlim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False); ax.grid(alpha=0.25)
    ax.text(0.02, 1.06, "ngưỡng thấp: bắt nhiều, nhầm nhiều", fontsize=8.5, color=MUTED, transform=ax.transAxes)
    ax.text(0.98, 1.06, "ngưỡng cao: chắc chắn, nhưng bỏ sót", fontsize=8.5, color=MUTED, ha="right",
            transform=ax.transAxes)
    save(fig, "tradeoff")


# ---------------------------------------------------------------- 6. ROC và AUC
def auc(x, y):
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def fig_roc():
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    ax = axes[0]
    for sep, c in [(0.6, "#9aa0c8"), (1.5, NEG), (3.0, POS)]:
        sp, sn = scores(500, 500, sep, rng)
        fpr, tpr, _, _ = curves(sp, sn)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=c, lw=2.2, label=f"tách {'yếu' if sep < 1 else 'vừa' if sep < 2 else 'tốt'}: AUC = {a:.2f}")
        print(f"  ROC sep={sep}: AUC={a:.3f}")
    ax.plot([0, 1], [0, 1], color=INK, ls="--", lw=1, label="đoán ngẫu nhiên: AUC = 0.50")
    ax.plot([0, 0, 1], [0, 1, 1], color=TP_C, ls=":", lw=2, label="hoàn hảo: AUC = 1")
    ax.set_xlabel("FPR = FP / N  (tỉ lệ báo nhầm)"); ax.set_ylabel("TPR = recall = TP / P")
    ax.set_title("(a) Đường ROC của ba mô hình"); ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_aspect("equal"); ax.grid(alpha=0.25); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)

    ax = axes[1]
    sp, sn = scores(500, 500, 1.5, np.random.default_rng(5))
    fpr, tpr, _, sc = curves(sp, sn)
    a = auc(fpr, tpr)
    ax.fill_between(fpr, tpr, color=NEG, alpha=0.15, step=None)
    ax.plot(fpr, tpr, color=NEG, lw=2.2)
    ax.plot([0, 1], [0, 1], color=INK, ls="--", lw=1)
    for t, off in [(0.8, (0.1, -0.12)), (0.5, (0.1, -0.1)), (0.2, (0.08, -0.13))]:
        i = np.searchsorted(-sc, -t)   # số điểm có điểm số >= t
        ax.scatter([fpr[i]], [tpr[i]], c=INK, zorder=3, s=26)
        ax.annotate(f"ngưỡng {t}", (fpr[i], tpr[i]), (fpr[i] + off[0], tpr[i] + off[1]), fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.text(0.7, 0.25, f"diện tích = AUC\n= {a:.2f}", ha="center", fontsize=11, color=NEG)
    ax.annotate("hạ ngưỡng: đi lên\nvà sang phải", (0.33, 0.8), (0.03, 0.9), fontsize=8.5, color=MUTED)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("(b) Mỗi ngưỡng là một điểm trên đường ROC"); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    save(fig, "roc")


def fig_roc_vs_pr():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for n_neg, c, lab in [(1000, NEG, "dương 50%"), (19000, POS, "dương 5%")]:
        sp, sn = scores(1000, n_neg, 2.0, rng)
        fpr, tpr, prec, _ = curves(sp, sn)
        a = auc(fpr, tpr)
        ap = float(np.sum(np.diff(tpr) * prec[1:]))
        axes[0].plot(fpr, tpr, color=c, lw=2.2, label=f"{lab}: ROC-AUC = {a:.2f}")
        axes[1].plot(tpr, prec, color=c, lw=2.2, label=f"{lab}: AP = {ap:.2f}")
        axes[1].axhline(1000 / (1000 + n_neg), color=c, lw=1, ls=":")
        print(f"  {lab}: ROC-AUC={a:.3f} AP={ap:.3f}")
    axes[0].plot([0, 1], [0, 1], color=INK, ls="--", lw=1)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR (recall)"); axes[0].set_title("(a) ROC gần như không đổi")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("(b) Đường precision–recall tụt mạnh")
    axes[1].set_xlabel("recall\n(đường chấm: precision của đoán ngẫu nhiên = tỉ lệ dương)")
    for ax in axes:
        ax.legend(loc="lower left" if ax is axes[1] else "lower right", fontsize=9, frameon=False)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02); ax.grid(alpha=0.25); ax.set_aspect("equal")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0.0, 0.12), fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, "roc_vs_pr")


# ---------------------------------------------------------------- 7. đa lớp
def fig_multiclass():
    C = np.array([[45, 3, 2], [6, 30, 14], [1, 9, 40]])
    names = ["mèo", "chó", "cáo"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.imshow(C / C.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            v = C[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v / C[i].sum() > 0.5 else INK, weight="bold" if i == j else None)
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    ax.set_xlabel("dự đoán"); ax.set_ylabel("nhãn thật")
    ax.set_title("(a) Ma trận nhầm lẫn 3 lớp\n(đường chéo: đoán đúng)")
    for s in ax.spines.values(): s.set_visible(False)
    ax.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False, ec=FP_C, lw=0))

    P = np.diag(C) / C.sum(0); R = np.diag(C) / C.sum(1); F = 2 * P * R / (P + R)
    acc = np.trace(C) / C.sum()
    w = C.sum(1) / C.sum()
    print("  đa lớp: P", P.round(3), "R", R.round(3), "F1", F.round(3), "acc", round(acc, 3),
          "macroF1", round(F.mean(), 3), "weightedF1", round((w * F).sum(), 3))
    ax = axes[1]; ax.axis("off")
    rows = [["lớp", "precision", "recall", "F1", "số mẫu"]]
    for k in range(3):
        rows.append([names[k], f"{P[k]:.2f}", f"{R[k]:.2f}", f"{F[k]:.2f}", str(C[k].sum())])
    rows.append(["macro", f"{P.mean():.2f}", f"{R.mean():.2f}", f"{F.mean():.2f}", "TB đều các lớp"])
    rows.append(["micro", f"{acc:.2f}", f"{acc:.2f}", f"{acc:.2f}", "= accuracy"])
    tb = ax.table(cellText=rows, loc="center", cellLoc="center", colWidths=[0.17, 0.2, 0.17, 0.13, 0.3])
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.55)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#d6dae8")
        if r == 0: cell.set_facecolor("#eef1f8"); cell.set_text_props(weight="bold")
        if r >= 4: cell.set_facecolor("#fff7e6")
    ax.set_title("(b) Chỉ số theo từng lớp và hai cách lấy trung bình")
    fig.tight_layout()
    save(fig, "multiclass")


# ---------------------------------------------------------------- 8. hồi quy
def fig_regression():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 10, 12)); y = 1.2 + 0.6 * x + rng.normal(0, 1.0, 12)
    w = np.polyfit(x, y, 1); yh = np.polyval(w, x); ybar = y.mean()
    r = y - yh
    mae, mse = np.abs(r).mean(), (r ** 2).mean()
    sst, ssr = ((y - ybar) ** 2).sum(), (r ** 2).sum()
    print(f"  hồi quy: MAE={mae:.3f} MSE={mse:.3f} RMSE={np.sqrt(mse):.3f} SS_res={ssr:.2f} SS_tot={sst:.2f} R2={1 - ssr / sst:.3f}")
    xs = np.linspace(-0.3, 10.3, 50)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    ax = axes[0]
    for xi, yi, hi in zip(x, y, yh):
        ax.plot([xi, xi], [hi, yi], color=FN_C, lw=2)
    ax.plot(xs, np.polyval(w, xs), color=INK, lw=1.8)
    ax.scatter(x, y, c=NEG, zorder=3, s=30)
    ax.set_title(f"(a) MAE: trung bình độ dài đoạn đỏ\nMAE = {mae:.2f}")
    ax = axes[1]
    for xi, yi, hi in zip(x, y, yh):
        d = yi - hi
        ax.add_patch(Rectangle((xi, min(yi, hi)), abs(d) * 0.75, abs(d), fc=FP_C, alpha=0.35, ec=FP_C))
    ax.plot(xs, np.polyval(w, xs), color=INK, lw=1.8)
    ax.scatter(x, y, c=NEG, zorder=3, s=30)
    ax.set_title(f"(b) MSE: trung bình diện tích ô vuông\nMSE = {mse:.2f},  RMSE = {np.sqrt(mse):.2f}")
    ax = axes[2]
    for xi, yi, hi in zip(x, y, yh):
        ax.plot([xi - 0.14, xi - 0.14], [ybar, yi], color=MUTED, lw=2)
        ax.plot([xi + 0.14, xi + 0.14], [hi, yi], color=FN_C, lw=2)
    ax.axhline(ybar, color=MUTED, ls="--", lw=1.5)
    ax.plot(xs, np.polyval(w, xs), color=INK, lw=1.8)
    ax.scatter(x, y, c=NEG, zorder=3, s=30)
    ax.text(10.2, ybar - 0.7, r"$\bar{y}$ (mô hình hằng)", ha="right", color=MUTED, fontsize=9)
    ax.set_title(f"(c) $R^2 = 1 - SS_{{res}}/SS_{{tot}}$\n= 1 − {ssr:.1f}/{sst:.1f} = {1 - ssr / sst:.2f}")
    for ax in axes:
        ax.set_xlabel("x"); ax.set_aspect("equal"); ax.set_xlim(-0.5, 10.6); ax.grid(alpha=0.2)
    axes[0].set_ylabel("y")
    fig.text(0.86, -0.04, "xám: lệch so với trung bình ($SS_{tot}$),  đỏ: lệch so với mô hình ($SS_{res}$)",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, "regression")


# ---------------------------------------------------------------- 9. bias–variance
def fig_dartboard():
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.6))
    cfg = [((0, 0), 0.12, "bias thấp, variance thấp"), ((0, 0), 0.42, "bias thấp, variance cao"),
           ((0.55, 0.35), 0.12, "bias cao, variance thấp"), ((0.55, 0.35), 0.42, "bias cao, variance cao")]
    for ax, (c, s, t) in zip(axes.ravel(), cfg):
        for rr, col in zip([1.0, 0.75, 0.5, 0.25], ["#eef1f8", "#dfe5f5", "#c9d3ef", "#fbd2df"]):
            ax.add_patch(Circle((0, 0), rr, fc=col, ec="#b8c1dc", lw=0.8))
        p = rng.normal(c, s, (12, 2))
        ax.scatter(p[:, 0], p[:, 1], c=NEG, s=26, edgecolors="white", linewidths=0.6, zorder=3)
        ax.scatter([p[:, 0].mean()], [p[:, 1].mean()], marker="x", c=POS, s=70, lw=2.5, zorder=4)
        ax.set_title(t, fontsize=10)
        ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08); ax.set_aspect("equal"); ax.axis("off")
    fig.text(0.5, 0.0, "Tâm bia: hàm thật.  Mỗi chấm: mô hình học từ một tập huấn luyện khác nhau.\n"
             "Dấu ×: trung bình các mô hình. Bias = × lệch tâm bao xa; variance = các chấm toả rộng bao nhiêu.",
             ha="center", va="top", fontsize=8.8)
    fig.tight_layout()
    save(fig, "dartboard")


def poly_feats(x, k):
    return np.vander(2 * x - 1, k + 1, increasing=True)


def fig_bias_variance():
    """Mô phỏng thật: 400 tập huấn luyện 20 điểm từ y = sin(2πx) + N(0, 0.3²), hồi quy đa thức bậc k
    (thêm λ = 1e-4 rất nhỏ chỉ để hệ phương trình không suy biến)."""
    rng = np.random.default_rng(0)
    f = lambda x: np.sin(2 * np.pi * x)
    sigma, n, R, lam = 0.3, 20, 400, 1e-4
    xt = np.linspace(0.02, 0.98, 200)
    degs = range(0, 11)
    b2, var = [], []
    preds_keep = {}
    for k in degs:
        P = []
        for _ in range(R):
            x = rng.uniform(0, 1, n); y = f(x) + rng.normal(0, sigma, n)
            A = poly_feats(x, k); I = lam * np.eye(k + 1); I[0, 0] = 0
            w = np.linalg.solve(A.T @ A + I, A.T @ y)
            P.append(poly_feats(xt, k) @ w)
        P = np.array(P)
        b2.append(np.mean((P.mean(0) - f(xt)) ** 2)); var.append(np.mean(P.var(0)))
        if k in (1, 3, 10):
            preds_keep[k] = P
    b2, var = np.array(b2), np.array(var)
    tot = b2 + var + sigma ** 2
    kb = int(np.argmin(tot))
    print("  bias²:", b2.round(3)); print("  var  :", var.round(3)); print(f"  tổng nhỏ nhất ở bậc {kb}: {tot[kb]:.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), gridspec_kw={"width_ratios": [1, 1, 1, 1.35]})
    for ax, k, t in zip(axes, (1, 3, 10), ("(a) bậc 1: bias cao", "(b) bậc 3: cân bằng", "(c) bậc 10: variance cao")):
        for p in preds_keep[k][:25]:
            ax.plot(xt, p, color=NEG, alpha=0.18, lw=1)
        ax.plot(xt, preds_keep[k].mean(0), color=POS, lw=2.2)
        ax.plot(xt, f(xt), color=INK, ls="--", lw=1.4)
        ax.set_ylim(-1.8, 1.8); ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    axes[0].text(0.03, -1.65, "xanh: 25 mô hình, hồng: trung bình,\nnét đứt: hàm thật", fontsize=7.6, color=INK)
    ax = axes[3]
    ks = np.array(list(degs))
    ax.plot(ks, b2, "o-", color="#7c3aed", lw=2, ms=4, label="bias²")
    ax.plot(ks, var, "s-", color=FP_C, lw=2, ms=4, label="variance")
    ax.plot(ks, tot, "-", color=INK, lw=2.4, label="sai số kỳ vọng")
    ax.axhline(sigma ** 2, color=MUTED, ls=":", lw=1.2)
    ax.text(0, sigma ** 2 * 0.72, "nhiễu σ² (không giảm được)", ha="left", fontsize=8, color=MUTED)
    ax.axvline(kb, color=MUTED, lw=1, ls="--")
    ax.set_yscale("log"); ax.set_ylim(0.002, 12)
    ax.set_xlabel("bậc đa thức (độ phức tạp)"); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3, handlelength=1.2, columnspacing=0.8)
    ax.set_title("(d) Phân tích sai số theo bậc"); ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "bias_variance")


def fig_learning_curve():
    rng = np.random.default_rng(2)
    f = lambda x: np.sin(2 * np.pi * x)
    sigma = 0.3
    ns = np.unique(np.round(np.logspace(np.log10(15), np.log10(400), 14)).astype(int))
    xv = rng.uniform(0, 1, 2000); yv = f(xv) + rng.normal(0, sigma, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, k, t in [(axes[0], 1, "(a) Bậc 1 (bias cao): thêm dữ liệu không giúp"),
                     (axes[1], 9, "(b) Bậc 9 (variance cao): thêm dữ liệu giúp nhiều")]:
        tr, va = [], []
        for n in ns:
            a, b = [], []
            for _ in range(40):
                x = rng.uniform(0, 1, n); y = f(x) + rng.normal(0, sigma, n)
                w = np.linalg.lstsq(poly_feats(x, k), y, rcond=None)[0]
                a.append(np.sqrt(np.mean((poly_feats(x, k) @ w - y) ** 2)))
                b.append(np.sqrt(np.mean((poly_feats(xv, k) @ w - yv) ** 2)))
            tr.append(np.median(a)); va.append(np.median(b))
        ax.plot(ns, tr, "o-", color=NEG, ms=4, lw=2, label="sai số huấn luyện")
        ax.plot(ns, va, "s-", color=POS, ms=4, lw=2, label="sai số xác thực")
        ax.axhline(sigma, color=MUTED, ls=":", lw=1.2)
        ax.set_xscale("log"); ax.set_ylim(0, 1.0); ax.set_title(t, fontsize=10)
        ax.set_xticks([15, 30, 60, 100, 200, 400], ["15", "30", "60", "100", "200", "400"]); ax.minorticks_off()
        ax.set_xlabel("số điểm huấn luyện"); ax.grid(alpha=0.25)
        print(f"  learning curve bậc {k}: n={ns[0]} tr={tr[0]:.3f} va={va[0]:.3f};  n={ns[-1]} tr={tr[-1]:.3f} va={va[-1]:.3f}")
    axes[0].set_ylabel("RMSE (trung vị 40 lần)"); axes[0].legend(frameon=False, loc="upper right")
    axes[1].text(400, sigma + 0.02, "mức nhiễu", ha="right", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "learning_curve")


if __name__ == "__main__":
    for fn in [fig_split, fig_confusion, fig_precision_recall, fig_imbalance, fig_threshold, fig_tradeoff,
               fig_roc, fig_roc_vs_pr, fig_multiclass, fig_regression, fig_dartboard, fig_bias_variance,
               fig_learning_curve]:
        print(fn.__name__)
        fn()
