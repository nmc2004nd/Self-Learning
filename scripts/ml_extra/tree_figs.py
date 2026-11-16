"""Vẽ hình minh hoạ cho hai bài bổ sung "Cây quyết định" (docs/ml/tree_vi.md)
và "Học tổ hợp" (docs/ml/ensemble_vi.md).

Dùng:  python3 -B scripts/ml_extra/tree_figs.py
Hình SVG lưu ở docs/img/ml/tree/ và docs/img/ml/ensemble/. Các con số in ra là số được dẫn trong bài.
"""

import sys
from math import comb
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingRegressor)
from sklearn.model_selection import train_test_split

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2] / "docs" / "img" / "ml"
OUT_T, OUT_E = ROOT / "tree", ROOT / "ensemble"
OUT_T.mkdir(parents=True, exist_ok=True); OUT_E.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "path", "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})
C0, C1 = "#2f80f5", "#e8487f"           # hai lớp
INK, MUTED, ACC = "#1f2340", "#8a90b4", "#f08a24"
GREEN, PURPLE = "#16a34a", "#7c3aed"


def save(fig, out, name):
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)
    print("  ->", out.name + "/" + name + ".svg")


def region(ax, model, X, y, lim=(-1.6, 2.6, -1.1, 1.6), n=300, proba=True, pts=True):
    xx, yy = np.meshgrid(np.linspace(lim[0], lim[1], n), np.linspace(lim[2], lim[3], n))
    G = np.c_[xx.ravel(), yy.ravel()]
    z = model.predict_proba(G)[:, 1] if proba else model.predict(G)
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("b", ["#cfe0fd", "#ffffff", "#fbd0de"])
    ax.contourf(xx, yy, z.reshape(xx.shape), levels=np.linspace(0, 1, 21), cmap=cm)
    ax.contour(xx, yy, z.reshape(xx.shape), levels=[0.5], colors=INK, linewidths=1.2)
    if pts:
        ax.scatter(*X[y == 0].T, s=12, c=C0, edgecolors="white", linewidths=0.4)
        ax.scatter(*X[y == 1].T, s=12, c=C1, edgecolors="white", linewidths=0.4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(lim[:2]); ax.set_ylim(lim[2:])


# ======================================================================= CÂY QUYẾT ĐỊNH
def node_box(ax, x, y, text, fc, ec, w=1.9, h=0.62, fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)


def edge(ax, x0, y0, x1, y1, label):
    ax.plot([x0, x1], [y0 - 0.31, y1 + 0.31], color=MUTED, lw=1.3, zorder=0)
    ax.text((x0 + x1) / 2 + (0.12 if x1 > x0 else -0.12), (y0 + y1) / 2, label, fontsize=8.5,
            ha="left" if x1 > x0 else "right", va="center", color=MUTED)


def fig_intro():
    """Một cây nhỏ trên dữ liệu 2D và cách nó chia mặt phẳng thành các hình chữ nhật."""
    rng = np.random.default_rng(3)
    n = 120
    X = rng.uniform(0, 10, (n, 2))
    y = (((X[:, 0] > 4) & (X[:, 1] > 3)) | ((X[:, 0] > 7) & (X[:, 1] <= 3))).astype(int)
    flip = rng.uniform(size=n) < 0.06; y[flip] = 1 - y[flip]
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, random_state=0).fit(X, y)
    T = t.tree_
    print("  cây giới thiệu: số lá", t.get_n_leaves(), " accuracy huấn luyện", round(t.score(X, y), 3))

    fig = plt.figure(figsize=(11, 4.3))
    ax = fig.add_axes([0.0, 0.0, 0.56, 1.0]); ax.axis("off")
    names = ["x₁", "x₂"]
    pos = {}

    def place(i, x, y, dx):
        pos[i] = (x, y)
        if T.children_left[i] != -1:
            place(T.children_left[i], x - dx, y - 1.25, dx / 2)
            place(T.children_right[i], x + dx, y - 1.25, dx / 2)
    place(0, 0, 0, 3.0)
    for i, (x, y_) in pos.items():
        v = T.value[i][0]; cnt = (v / v.sum() * T.n_node_samples[i]).round().astype(int)
        if T.children_left[i] == -1:
            cls = int(np.argmax(v))
            node_box(ax, x, y_, f"lá: {'hồng' if cls else 'xanh'}\n{cnt[0]} xanh\n{cnt[1]} hồng",
                     "#fde7ef" if cls else "#e3edfe", C1 if cls else C0, w=1.3, h=0.85, fs=8)
        else:
            node_box(ax, x, y_, f"{names[T.feature[i]]} ≤ {T.threshold[i]:.1f} ?\n{cnt[0]} xanh / {cnt[1]} hồng",
                     "#fff7e6", ACC, w=1.75, fs=8.5)
            edge(ax, x, y_, *pos[T.children_left[i]], "đúng")
            edge(ax, x, y_, *pos[T.children_right[i]], "sai")
    ax.set_xlim(-5.4, 5.4); ax.set_ylim(-4.4, 0.5)
    ax.set_title("(a) Cây: mỗi nút trong là một câu hỏi, mỗi lá là một dự đoán", loc="left", fontsize=10.5)

    ax = fig.add_axes([0.61, 0.08, 0.38, 0.84])
    xx, yy = np.meshgrid(np.linspace(0, 10, 400), np.linspace(0, 10, 400))
    z = t.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, z, levels=[-0.5, 0.5, 1.5], colors=["#e3edfe", "#fde7ef"])

    def lines(i, x0, x1, y0, y1):
        if T.children_left[i] == -1:
            return
        f, th = T.feature[i], T.threshold[i]
        if f == 0:
            ax.plot([th, th], [y0, y1], color=ACC, lw=1.8)
            lines(T.children_left[i], x0, th, y0, y1); lines(T.children_right[i], th, x1, y0, y1)
        else:
            ax.plot([x0, x1], [th, th], color=ACC, lw=1.8)
            lines(T.children_left[i], x0, x1, y0, th); lines(T.children_right[i], x0, x1, th, y1)
    lines(0, 0, 10, 0, 10)
    ax.scatter(*X[y == 0].T, s=16, c=C0, edgecolors="white", linewidths=0.5)
    ax.scatter(*X[y == 1].T, s=16, c=C1, edgecolors="white", linewidths=0.5)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.set_title("(b) Mỗi câu hỏi là một nhát cắt song song trục", fontsize=10.5)
    save(fig, OUT_T, "intro")


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def fig_impurity():
    p = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(p, entropy(p), color=PURPLE, lw=2.4, label="entropy  $-\\sum_k p_k\\log_2 p_k$")
    ax.plot(p, 2 * p * (1 - p), color=GREEN, lw=2.4, label="Gini  $1-\\sum_k p_k^2$")
    ax.plot(p, np.minimum(p, 1 - p), color=ACC, lw=2.0, ls="--", label="tỉ lệ lỗi  $1-\\max_k p_k$")
    ax.set_xlabel("$p$ = tỉ lệ lớp hồng trong nút"); ax.set_ylabel("độ vẩn đục")
    ax.legend(frameon=False, loc="upper right", fontsize=9); ax.grid(alpha=0.25); ax.set_ylim(0, 1.25)
    ax.text(0.02, 0.06, "nút thuần khiết\n(chỉ một lớp)", fontsize=8.5, color=MUTED)
    ax.text(0.5, 1.03, "trộn đều nhất", ha="center", fontsize=8.5, color=MUTED)
    save(fig, OUT_T, "impurity")


def fig_split_search():
    """Quét mọi ngưỡng trên một thuộc tính liên tục, đo độ giảm Gini."""
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 10, 24))
    y = (x + rng.normal(0, 1.6, 24) > 5.5).astype(int)
    def gini(v):
        if len(v) == 0: return 0.0
        q = v.mean(); return 2 * q * (1 - q)
    G0 = gini(y)
    cand = (x[1:] + x[:-1]) / 2
    gain = np.array([G0 - (np.sum(x <= c) * gini(y[x <= c]) + np.sum(x > c) * gini(y[x > c])) / len(x) for c in cand])
    k = int(np.argmax(gain))
    print(f"  quét ngưỡng: Gini gốc {G0:.3f}, ngưỡng tốt nhất {cand[k]:.2f}, độ giảm {gain[k]:.3f}, {len(cand)} ứng viên")
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.0), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    ax = axes[0]
    ax.scatter(x[y == 0], np.zeros((y == 0).sum()), s=55, c=C0, edgecolors="white", zorder=3)
    ax.scatter(x[y == 1], np.zeros((y == 1).sum()), s=55, c=C1, edgecolors="white", zorder=3)
    for c in cand: ax.axvline(c, color=MUTED, lw=0.6, alpha=0.6)
    ax.axvline(cand[k], color=ACC, lw=2.2)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_title("Ứng viên ngưỡng: điểm giữa mỗi hai giá trị liên tiếp (vạch xám)", fontsize=10)
    ax = axes[1]
    ax.plot(cand, gain, "o-", color=GREEN, ms=4)
    ax.scatter([cand[k]], [gain[k]], s=70, c=ACC, zorder=3)
    ax.annotate(f"ngưỡng tốt nhất x ≤ {cand[k]:.2f}\nGini giảm {gain[k]:.3f}", (cand[k], gain[k]),
                (1.0, 0.2), arrowprops=dict(arrowstyle="->", color=INK), fontsize=9)
    ax.set_xlabel("ngưỡng t trên thuộc tính x"); ax.set_ylabel("độ giảm Gini"); ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, OUT_T, "split_search")


def fig_regression_tree():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 80)); y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, 80)
    xs = np.linspace(0, 1, 800)[:, None]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
    for ax, d in zip(axes, [1, 3, 10]):
        m = DecisionTreeRegressor(max_depth=d, random_state=0).fit(x[:, None], y)
        ax.scatter(x, y, s=12, c=C0, alpha=0.8)
        ax.plot(xs, np.sin(2 * np.pi * xs), "--", color=MUTED, lw=1)
        ax.step(xs[:, 0], m.predict(xs), where="mid", color=C1, lw=2)
        ax.set_title(f"max_depth = {d}: {m.get_n_leaves()} lá", fontsize=10)
        ax.set_xticks([]); ax.set_ylim(-1.9, 1.9)
    axes[0].set_ylabel("y")
    fig.tight_layout()
    save(fig, OUT_T, "regression_tree")


def moons(n=300, noise=0.3, seed=0):
    return make_moons(n_samples=n, noise=noise, random_state=seed)


def fig_depth():
    X, y = moons(600, 0.32, 1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
    depths = list(range(1, 16))
    tr, te = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        tr.append(m.score(Xtr, ytr)); te.append(m.score(Xte, yte))
    kb = depths[int(np.argmax(te))]
    print("  độ sâu: train", np.round(tr, 3), "\n          test ", np.round(te, 3), " tốt nhất d =", kb)
    fig = plt.figure(figsize=(12, 3.4))
    for i, d in enumerate([1, kb, None]):
        ax = fig.add_axes([0.0 + i * 0.235, 0.0, 0.22, 0.86])
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        region(ax, m, Xtr, ytr)
        ax.set_title(f"max_depth = {d if d else 'không giới hạn'}\n{m.get_n_leaves()} lá, test {m.score(Xte, yte):.2f}", fontsize=9.5)
    ax = fig.add_axes([0.74, 0.12, 0.26, 0.74])
    ax.plot(depths, tr, "o-", color=C0, ms=4, label="huấn luyện")
    ax.plot(depths, te, "s-", color=C1, ms=4, label="kiểm tra")
    ax.axvline(kb, color=MUTED, ls=":")
    ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy"); ax.legend(frameon=False); ax.grid(alpha=0.25)
    save(fig, OUT_T, "depth")


def fig_pruning():
    X, y = moons(600, 0.32, 1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
    path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(Xtr, ytr)
    alphas = path.ccp_alphas[:-1]
    leaves, tr, te = [], [], []
    for a in alphas:
        m = DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(Xtr, ytr)
        leaves.append(m.get_n_leaves()); tr.append(m.score(Xtr, ytr)); te.append(m.score(Xte, yte))
    k = int(np.argmax(te))
    print(f"  cắt tỉa: {len(alphas)} giá trị α, cây đầy đủ {leaves[0]} lá test {te[0]:.3f}; "
          f"tốt nhất α={alphas[k]:.4f}, {leaves[k]} lá, test {te[k]:.3f}")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(leaves, tr, "o-", color=C0, ms=3.5, label="huấn luyện", drawstyle="steps-post")
    ax.plot(leaves, te, "s-", color=C1, ms=3.5, label="kiểm tra", drawstyle="steps-post")
    ax.scatter([leaves[k]], [te[k]], s=80, facecolors="none", edgecolors=INK, lw=1.5, zorder=4)
    ax.annotate(f"α = {alphas[k]:.4f}: {leaves[k]} lá,\nkiểm tra {te[k]:.3f}", (leaves[k], te[k]),
                (30, 0.79), arrowprops=dict(arrowstyle="->", color=INK), fontsize=9)
    ax.invert_xaxis()
    ax.set_xlabel("số lá (α tăng dần → cây nhỏ dần)"); ax.set_ylabel("accuracy")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    save(fig, OUT_T, "pruning")


def fig_instability():
    X, y = moons(200, 0.3, 4)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0))
    for i, ax in enumerate(axes):
        idx = rng.integers(0, len(X), len(X))
        m = DecisionTreeClassifier(random_state=0).fit(X[idx], y[idx])
        region(ax, m, X[idx], y[idx])
        ax.set_title(f"mẫu bootstrap {i + 1}", fontsize=10)
    fig.tight_layout()
    save(fig, OUT_T, "instability")


# ======================================================================= HỌC TỔ HỢP
def fig_vote():
    """Biểu quyết đa số của n bộ phân loại độc lập, mỗi bộ đúng với xác suất p."""
    ns = np.arange(1, 102, 2)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for p, c in [(0.55, MUTED), (0.6, C0), (0.7, C1)]:
        acc = [sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1)) for n in ns]
        ax.plot(ns, acc, color=c, lw=2.2, label=f"mỗi mô hình đúng {int(p * 100)}%")
        print(f"  biểu quyết p={p}: n=1 {acc[0]:.3f}, n=11 {acc[5]:.3f}, n=101 {acc[-1]:.3f}")
    ax.set_xlabel("số mô hình độc lập n (lẻ)"); ax.set_ylabel("accuracy của biểu quyết đa số")
    ax.legend(frameon=False, loc="lower right"); ax.grid(alpha=0.25); ax.set_ylim(0.5, 1.01)
    save(fig, OUT_E, "vote")


def fig_correlation():
    n = np.arange(1, 201)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for rho, c in [(0.0, GREEN), (0.1, C0), (0.3, PURPLE), (0.6, C1)]:
        ax.plot(n, rho + (1 - rho) / n, color=c, lw=2.2, label=f"ρ = {rho}")
        ax.axhline(rho, color=c, ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("số mô hình n"); ax.set_ylabel("phương sai trung bình / σ²")
    ax.legend(frameon=False, title="tương quan giữa hai mô hình", fontsize=9, title_fontsize=9)
    ax.grid(alpha=0.25)
    save(fig, OUT_E, "correlation")


def fig_bootstrap():
    rng = np.random.default_rng(2)
    n = 12
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for r in range(3):
        idx = rng.integers(0, n, n)
        cnt = np.bincount(idx, minlength=n)
        for i in range(n):
            col = C0 if cnt[i] else "#e5e7ef"
            ax.add_patch(FancyBboxPatch((i * 0.62, -r * 0.8), 0.5, 0.5, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=col, ec="none", alpha=0.25 + 0.25 * min(cnt[i], 3)))
            ax.text(i * 0.62 + 0.25, -r * 0.8 + 0.25, str(cnt[i]) if cnt[i] else "OOB",
                    ha="center", va="center", fontsize=8 if cnt[i] == 0 else 9.5,
                    color=INK if cnt[i] else MUTED, weight="bold" if cnt[i] else None)
        ax.text(-0.2, -r * 0.8 + 0.25, f"mẫu {r + 1}", ha="right", va="center", fontsize=9.5)
        ax.text(n * 0.62 + 0.1, -r * 0.8 + 0.25, f"{(cnt == 0).sum()} điểm không được chọn", va="center", fontsize=9, color=MUTED)
    for i in range(n):
        ax.text(i * 0.62 + 0.25, 0.75, f"#{i + 1}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(-1.3, n * 0.62 + 2.7); ax.set_ylim(-2.0, 1.0); ax.axis("off")
    fr = np.mean([np.unique(np.random.default_rng(s).integers(0, 1000, 1000)).size / 1000 for s in range(200)])
    print(f"  bootstrap: tỉ lệ điểm khác nhau trung bình (n=1000) = {fr:.4f}; 1-1/e = {1 - np.exp(-1):.4f}")
    save(fig, OUT_E, "bootstrap")


def fig_bagging():
    X, y = moons(300, 0.32, 1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
    models = [("1 cây quyết định", DecisionTreeClassifier(random_state=0)),
              ("bagging 200 cây", BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=0)),
              ("random forest 200 cây", RandomForestClassifier(n_estimators=200, random_state=0))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, (name, m) in zip(axes, models):
        m.fit(Xtr, ytr)
        region(ax, m, Xtr, ytr)
        ax.set_title(f"{name}\naccuracy kiểm tra {m.score(Xte, yte):.3f}", fontsize=10)
        print(f"  {name}: test {m.score(Xte, yte):.3f}")
    fig.tight_layout()
    save(fig, OUT_E, "bagging")


def fig_rf_trees():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=6, n_redundant=6,
                               flip_y=0.05, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
    ns = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 400]
    single = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).score(Xte, yte)
    res = {}
    for name, mf in [("bagging (xét mọi đặc trưng)", None), ("random forest (√d đặc trưng mỗi lần chia)", "sqrt")]:
        rf = RandomForestClassifier(n_estimators=max(ns), max_features=mf, oob_score=False, random_state=0).fit(Xtr, ytr)
        P = np.cumsum([t.predict_proba(Xte)[:, 1] for t in rf.estimators_], axis=0)
        res[name] = [np.mean((P[n - 1] / n > 0.5) == yte) for n in ns]
    print("  số cây:", ns)
    for k, v in res.items(): print("   ", k, np.round(v, 3))
    print(f"    1 cây: {single:.3f}")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ns, res["bagging (xét mọi đặc trưng)"], "o-", color=C0, ms=4, label="bagging (xét mọi đặc trưng)")
    ax.plot(ns, res["random forest (√d đặc trưng mỗi lần chia)"], "s-", color=C1, ms=4,
            label="random forest (√d đặc trưng mỗi lần chia)")
    ax.axhline(single, color=MUTED, ls=":", lw=1.2)
    ax.text(400, single + 0.004, "một cây", ha="right", fontsize=8.5, color=MUTED)
    ax.set_xscale("log"); ax.set_xlabel("số cây"); ax.set_ylabel("accuracy kiểm tra")
    ax.legend(frameon=False, loc="lower right", fontsize=9); ax.grid(alpha=0.25)
    save(fig, OUT_E, "rf_trees")


def fig_adaboost():
    rng = np.random.default_rng(5)
    n = 40
    X = rng.uniform(-1, 1, (n, 2))
    y = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 < 0.45, 1, -1)
    w = np.ones(n) / n
    F = np.zeros(n)
    stumps = []
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.4))
    rounds_to_show = {1: 0, 2: 1, 3: 2}
    for t in range(1, 31):
        s = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
        h = s.predict(X); err = w[h != y].sum()
        alpha = 0.5 * np.log((1 - err) / err)
        stumps.append((alpha, s))
        if t in rounds_to_show:
            ax = axes[rounds_to_show[t]]
            f, th = s.tree_.feature[0], s.tree_.threshold[0]
            sz = 1500 * w
            ax.scatter(*X[y < 0].T, s=sz[y < 0], c=C0, alpha=0.75, edgecolors="white")
            ax.scatter(*X[y > 0].T, s=sz[y > 0], c=C1, alpha=0.75, edgecolors="white")
            wrong = h != y
            ax.scatter(*X[wrong].T, s=sz[wrong] * 1.4, facecolors="none", edgecolors=INK, lw=1)
            if f == 0: ax.axvline(th, color=ACC, lw=2.2)
            else: ax.axhline(th, color=ACC, lw=2.2)
            ax.set_title(f"vòng {t}: lỗi có trọng số ε = {err:.2f}\nα = ½ln((1−ε)/ε) = {alpha:.2f}", fontsize=9.5)
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            print(f"  AdaBoost vòng {t}: eps={err:.3f} alpha={alpha:.3f} sai {wrong.sum()} điểm")
        w = w * np.exp(-alpha * y * h); w /= w.sum()
        F += alpha * h
    ax = axes[3]
    xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 300), np.linspace(-1.1, 1.1, 300))
    G = np.c_[xx.ravel(), yy.ravel()]
    Fg = sum(a * s.predict(G) for a, s in stumps).reshape(xx.shape)
    ax.contourf(xx, yy, Fg > 0, levels=[-0.5, 0.5, 1.5], colors=["#e3edfe", "#fde7ef"])
    ax.scatter(*X[y < 0].T, s=22, c=C0, edgecolors="white"); ax.scatter(*X[y > 0].T, s=22, c=C1, edgecolors="white")
    acc = np.mean(np.sign(F) == y)
    ax.set_title(f"sau 30 vòng: Σ αₜhₜ(x)\naccuracy huấn luyện {acc:.2f}", fontsize=9.5)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    print(f"  AdaBoost 30 vòng: accuracy huấn luyện {acc:.3f}")
    fig.tight_layout()
    save(fig, OUT_E, "adaboost")


def fig_gradient_boosting():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 100)); y = np.sin(2 * np.pi * x) + 0.3 * x + rng.normal(0, 0.2, 100)
    xs = np.linspace(0, 1, 600)
    nu, M = 0.3, 50
    F = np.full_like(y, y.mean()); Fs = np.full_like(xs, y.mean())
    snaps = {}
    for m in range(1, M + 1):
        r = y - F
        t = DecisionTreeRegressor(max_depth=2).fit(x[:, None], r)
        if m in (1, 2):
            snaps[m] = (r.copy(), t.predict(xs[:, None]), Fs.copy())
        F += nu * t.predict(x[:, None]); Fs += nu * t.predict(xs[:, None])
        if m in (3, 10, 50):
            snaps[m] = Fs.copy()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.6))
    for j, m in enumerate((1, 2)):
        r, tp, Fprev = snaps[m]
        ax = axes[0, j]
        ax.scatter(x, r, s=10, c=PURPLE, alpha=0.8)
        ax.plot(xs, tp, color=ACC, lw=2.2)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"bước {m}: phần dư rᵢ = yᵢ − F{m - 1}(xᵢ)\nvà cây độ sâu 2 khớp phần dư (cam)", fontsize=9.5)
    ax = axes[0, 2]
    ax.axis("off")
    ax.text(0.02, 0.5, "Mỗi bước:\n\n1. tính phần dư r = y − F(x)\n2. học một cây nhỏ h khớp r\n3. F(x) ← F(x) + ν·h(x)\n\nν = 0.3 là tốc độ học\n(shrinkage)",
            fontsize=10.5, va="center", linespacing=1.5)
    for j, m in enumerate((3, 10, 50)):
        ax = axes[1, j]
        ax.scatter(x, y, s=10, c=C0, alpha=0.7)
        ax.plot(xs, snaps[m], color=C1, lw=2.2)
        mse = np.mean((y - (F if m == 50 else 0)) ** 2) if m == 50 else None
        ax.set_title(f"F sau {m} bước", fontsize=10)
    for ax in axes.ravel(): ax.set_xticks([])
    print(f"  gradient boosting: MSE huấn luyện sau 50 bước {np.mean((y - F) ** 2):.4f}, phương sai y {y.var():.4f}")
    fig.tight_layout()
    save(fig, OUT_E, "gradient_boosting")


def fig_learning_rate():
    rng = np.random.default_rng(1)
    def data(n):
        x = rng.uniform(0, 1, (n, 1)); return x, np.sin(2 * np.pi * x[:, 0]) + 0.3 * x[:, 0] + rng.normal(0, 0.3, n)
    Xtr, ytr = data(200); Xte, yte = data(2000)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for lr, c in [(1.0, C1), (0.3, ACC), (0.05, C0)]:
        g = GradientBoostingRegressor(n_estimators=500, learning_rate=lr, max_depth=3, random_state=0).fit(Xtr, ytr)
        tr = [np.mean((ytr - p) ** 2) for p in g.staged_predict(Xtr)]
        te = [np.mean((yte - p) ** 2) for p in g.staged_predict(Xte)]
        ax.plot(range(1, 501), te, color=c, lw=2, label=f"ν = {lr}: kiểm tra")
        ax.plot(range(1, 501), tr, color=c, lw=1, ls="--")
        print(f"  GB ν={lr}: test MSE nhỏ nhất {min(te):.4f} ở bước {int(np.argmin(te)) + 1}, sau 500 bước {te[-1]:.4f}, train {tr[-1]:.4f}")
    ax.axhline(0.09, color=MUTED, ls=":", lw=1); ax.text(500, 0.095, "nhiễu σ² = 0.09", ha="right", fontsize=8, color=MUTED)
    ax.set_xscale("log"); ax.set_ylim(0, 0.35)
    ax.set_xlabel("số cây"); ax.set_ylabel("MSE"); ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, title="nét liền: kiểm tra, nét đứt: huấn luyện", title_fontsize=8.5)
    save(fig, OUT_E, "learning_rate")


def fig_stacking():
    fig, ax = plt.subplots(figsize=(8, 3.4)); ax.axis("off")
    node_box(ax, 0, 0, "dữ liệu\nhuấn luyện", "#eef1f8", MUTED, w=1.6, h=0.9, fs=9.5)
    bases = ["hồi quy logistic", "KNN", "random forest", "gradient boosting"]
    for i, b in enumerate(bases):
        yb = 1.5 - i
        node_box(ax, 3.2, yb, b, "#e3edfe", C0, w=2.2, h=0.6, fs=9.5)
        ax.annotate("", xy=(2.1, yb), xytext=(0.8, 0), arrowprops=dict(arrowstyle="->", color=MUTED))
        ax.annotate("", xy=(5.9, 0), xytext=(4.3, yb), arrowprops=dict(arrowstyle="->", color=MUTED))
        ax.text(4.75, yb * 0.55 + 0.1, "ŷ" + "₁₂₃₄"[i], fontsize=9, color=INK)
    node_box(ax, 7.0, 0, "mô hình tầng trên\n(meta-learner)\nhọc cách kết hợp\nŷ₁ … ŷ₄", "#fde7ef", C1, w=2.2, h=1.25, fs=9.5)
    ax.annotate("", xy=(9.4, 0), xytext=(8.1, 0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(9.5, 0, "dự đoán\ncuối", va="center", fontsize=9.5)
    ax.text(4.6, -2.25, "ŷ dùng để huấn luyện tầng trên phải là dự đoán out-of-fold (xác thực chéo), "
            "không phải dự đoán trên chính dữ liệu mô hình đã học", ha="center", fontsize=8.8, color=MUTED)
    ax.set_xlim(-1, 10.4); ax.set_ylim(-2.5, 2.1)
    save(fig, OUT_E, "stacking")


if __name__ == "__main__":
    for fn in [fig_intro, fig_impurity, fig_split_search, fig_regression_tree, fig_depth, fig_pruning,
               fig_instability, fig_vote, fig_correlation, fig_bootstrap, fig_bagging, fig_rf_trees,
               fig_adaboost, fig_gradient_boosting, fig_learning_rate, fig_stacking]:
        print(fn.__name__)
        fn()
