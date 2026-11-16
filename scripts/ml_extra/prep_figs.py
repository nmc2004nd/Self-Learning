"""Vẽ hình minh hoạ cho bài bổ sung "Tiền xử lý dữ liệu" (docs/ml/prep_vi.md).

Dùng:  python3 -B scripts/ml_extra/prep_figs.py
Hình SVG lưu ở docs/img/ml/prep/. Các con số in ra là số được dẫn trong bài.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from sklearn.datasets import fetch_california_housing, make_friedman1
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer,
                                   PowerTransformer, OneHotEncoder, KBinsDiscretizer, SplineTransformer)
from sklearn.linear_model import LinearRegression, HuberRegressor, LogisticRegression, lasso_path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, f_classif
from sklearn.pipeline import make_pipeline

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parents[2] / "docs" / "img" / "ml" / "prep"
OUT.mkdir(parents=True, exist_ok=True)
PREVIEW = __import__("os").environ.get("PREVIEW")          # thư mục lưu thêm PNG để xem nhanh

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "path", "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})
PAL = np.array(["#2f80f5", "#e8487f", "#16a34a", "#f08a24", "#7c3aed", "#0ea5a4", "#a16207", "#64748b"])
INK, MUTED, LIGHT = "#1f2340", "#8a90b4", "#b8bdd3"


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg")
    if PREVIEW: fig.savefig(Path(PREVIEW) / f"{name}.png", dpi=80)
    plt.close(fig)
    print("  ->", name + ".svg")


# ------------------------------------------------------------------ 1. dữ liệu thiếu
def fig_missing():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(0, 1, n)
    y = 0.8 * x + rng.normal(0, 0.6, n)
    p_miss = 1 / (1 + np.exp(-(1.6 * x - 0.6)))            # MAR: x càng lớn, y càng hay bị thiếu
    miss = rng.random(n) < p_miss
    X = np.c_[x, np.where(miss, np.nan, y)]
    print(f"  dữ liệu thiếu: {miss.sum()} / {n} giá trị y bị thiếu ({miss.mean():.0%})")
    r = lambda a, b: np.corrcoef(a, b)[0, 1]
    print(f"  thật:            mean y = {y.mean():+.3f}  std y = {y.std():.3f}  corr = {r(x, y):.3f}")
    print(f"  bỏ dòng thiếu:   mean y = {y[~miss].mean():+.3f}  std y = {y[~miss].std():.3f}  corr = {r(x[~miss], y[~miss]):.3f}")
    methods = [("điền trung bình", SimpleImputer(strategy="mean")),
               ("điền KNN (k = 5)", KNNImputer(n_neighbors=5)),
               ("điền bằng hồi quy (lặp)", IterativeImputer(random_state=0))]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), sharex=True, sharey=True)
    ax = axes[0]
    ax.scatter(x[~miss], y[~miss], s=9, c=PAL[0], alpha=0.75, label="quan sát được")
    ax.scatter(x[miss], y[miss], s=9, facecolors="none", edgecolors=MUTED, lw=0.8, label="bị thiếu (giá trị thật)")
    ax.set_title(f"(a) {miss.mean():.0%} giá trị y bị thiếu,\nhay thiếu khi x lớn (MAR)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for j, (name, imp) in enumerate(methods, start=1):
        Z = imp.fit_transform(X)
        yi = Z[:, 1]
        err = np.sqrt(np.mean((yi[miss] - y[miss]) ** 2))
        ax = axes[j]
        ax.scatter(x[~miss], y[~miss], s=9, c=PAL[0], alpha=0.35)
        ax.scatter(x[miss], yi[miss], s=11, c=PAL[1], alpha=0.85, label="giá trị điền vào")
        ax.set_title(f"({'bcd'[j - 1]}) {name}\nstd y = {yi.std():.2f}, corr = {r(x, yi):.2f}")
        print(f"  {name:24s} mean y = {yi.mean():+.3f}  std y = {yi.std():.3f}  corr = {r(x, yi):.3f}  RMSE điền = {err:.3f}")
        if j == 1: ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for a in axes: a.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout()
    save(fig, "missing")


# ------------------------------------------------------------------ 2. ngoại lai
def fig_outliers():
    rng = np.random.default_rng(4)
    v = np.r_[rng.normal(50, 8, 200), [95, 100, 105, 110, 200, 250, 300]]
    z = (v - v.mean()) / v.std()
    q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    nz, niq = int(np.sum(np.abs(z) > 3)), int(np.sum((v < lo) | (v > hi)))
    print(f"  ngoại lai: mean {v.mean():.1f}, std {v.std():.1f} (không có ngoại lai: std {v[:200].std():.1f})")
    print(f"  ngưỡng z-score |z|>3: [{v.mean() - 3 * v.std():.1f}, {v.mean() + 3 * v.std():.1f}] -> {nz} điểm")
    print(f"  ngưỡng IQR: Q1 {q1:.1f}, Q3 {q3:.1f}, [{lo:.1f}, {hi:.1f}] -> {niq} điểm")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.hist(v, bins=60, color=PAL[0], alpha=0.75)
    ax.axvspan(lo, hi, color=PAL[2], alpha=0.10)
    for b in (lo, hi): ax.axvline(b, color=PAL[2], lw=1.6)
    zl, zh = v.mean() - 3 * v.std(), v.mean() + 3 * v.std()
    for b in (zl, zh): ax.axvline(b, color=PAL[1], lw=1.6, ls="--")
    ax.text(hi + 2, ax.get_ylim()[1] * 0.85, f"IQR: Q3 + 1.5·IQR\n= {hi:.0f} → {niq} điểm", color=PAL[2], fontsize=8.5)
    ax.text(zh + 2, ax.get_ylim()[1] * 0.55, f"z-score: x̄ + 3s\n= {zh:.0f} → {nz} điểm", color=PAL[1], fontsize=8.5)
    ax.set_xlabel("giá trị"); ax.set_yticks([])
    ax.set_title("(a) 200 điểm bình thường + 7 ngoại lai: ngoại lai kéo s lên,\nnên ngưỡng z-score nới ra và bỏ sót (hiệu ứng che khuất)")
    ax = axes[1]
    x = rng.uniform(0, 10, 40); y = 2 + 0.5 * x + rng.normal(0, 0.5, 40)
    xo = np.r_[x, [9.0, 9.5, 9.8]]; yo = np.r_[y, [-2.0, -1.0, -2.5]]
    lr0 = LinearRegression().fit(x[:, None], y)
    lr = LinearRegression().fit(xo[:, None], yo)
    hb = HuberRegressor(epsilon=1.35).fit(xo[:, None], yo)
    g = np.linspace(0, 10, 50)
    ax.scatter(x, y, s=14, c=PAL[0]); ax.scatter(xo[-3:], yo[-3:], s=40, c=PAL[1], marker="X", label="3 ngoại lai")
    ax.plot(g, lr0.predict(g[:, None]), color=MUTED, lw=1.4, ls=":", label=f"OLS, không ngoại lai: hệ số {lr0.coef_[0]:.2f}")
    ax.plot(g, lr.predict(g[:, None]), color=PAL[1], lw=2, label=f"OLS, có ngoại lai: hệ số {lr.coef_[0]:.2f}")
    ax.plot(g, hb.predict(g[:, None]), color=PAL[2], lw=2, ls="--", label=f"Huber, có ngoại lai: hệ số {hb.coef_[0]:.2f}")
    print(f"  hồi quy: hệ số OLS sạch {lr0.coef_[0]:.3f}, OLS có ngoại lai {lr.coef_[0]:.3f}, Huber {hb.coef_[0]:.3f}")
    ax.legend(fontsize=7.8, frameon=False, loc="upper left"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(b) Ba ngoại lai bẻ đường hồi quy bình phương tối thiểu")
    fig.tight_layout()
    save(fig, "outliers")


# ------------------------------------------------------------------ 3. mã hoá biến phân loại
def fig_encoding():
    rng = np.random.default_rng(2)
    cities = ["Cần Thơ", "Đà Nẵng", "Hà Nội", "Huế", "Hải Phòng", "TP.HCM"]
    means = np.array([2.2, 3.4, 5.6, 1.8, 2.6, 6.0])
    c = rng.integers(0, 6, 600); y = means[c] + rng.normal(0, 0.6, 600)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    ax.scatter(c + rng.uniform(-0.18, 0.18, 600), y, s=6, c=LIGHT)
    lr = LinearRegression().fit(c[:, None], y)
    oh = OneHotEncoder(sparse_output=False).fit_transform(c[:, None])
    lr2 = LinearRegression().fit(oh, y)
    r2o, r2h = lr.score(c[:, None], y), lr2.score(oh, y)
    print(f"  mã hoá: R² mã thứ tự {r2o:.3f}, one-hot {r2h:.3f}")
    g = np.linspace(-0.4, 5.4, 10)
    ax.plot(g, lr.predict(g[:, None]), color=PAL[1], lw=2, label=f"mã thứ tự 0…5, hồi quy tuyến tính: R² = {r2o:.2f}")
    ax.scatter(range(6), lr2.predict(np.eye(6)), s=110, marker="_", lw=3, c=PAL[2], label=f"one-hot (6 cột): R² = {r2h:.2f}", zorder=3)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{k}\n{n}" for k, n in enumerate(cities)], fontsize=8.5)
    ax.set_ylabel("giá (tỉ đồng)"); ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("(a) Đánh số các thành phố theo bảng chữ cái tạo ra một thứ tự giả")
    ax = axes[1]
    gm, m = 3.0, 10
    counts = np.unique(np.round(np.geomspace(1, 300, 40)).astype(int))
    raw, sm = [], []
    for n_ in counts:
        yy = rng.normal(gm, 1.5, n_)
        raw.append(yy.mean()); sm.append((n_ * yy.mean() + m * gm) / (n_ + m))
    raw, sm = np.array(raw), np.array(sm)
    ax.axhline(gm, color=MUTED, lw=1, ls=":")
    ax.text(1.0, 3.62, "đường chấm: trung bình toàn cục 3.0", fontsize=8, color=MUTED)
    ax.scatter(counts, raw, s=22, c=PAL[1], label="trung bình nhóm (thô)")
    ax.scatter(counts, sm, s=22, c=PAL[0], marker="s", label=f"đã làm trơn, m = {m}")
    for a_, b_, n_ in zip(raw, sm, counts): ax.plot([n_, n_], [a_, b_], color=LIGHT, lw=0.8, zorder=0)
    ax.set_xscale("log"); ax.set_xlabel("số mẫu trong nhóm $n_c$"); ax.set_ylabel("giá trị mã hoá")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("(b) Mã hoá theo mục tiêu: mọi nhóm có cùng trung bình thật 3.0;\nnhóm nhỏ bị kéo về trung bình toàn cục")
    print(f"  target encoding: trung bình thô lệch xa nhất {np.abs(raw - gm).max():.2f} (n={counts[np.argmax(np.abs(raw - gm))]}), "
          f"sau làm trơn {np.abs(sm - gm).max():.2f}")
    fig.tight_layout()
    save(fig, "encoding")


# ------------------------------------------------------------------ 4. các bộ co giãn
def scale_data():
    rng = np.random.default_rng(0)
    age = rng.uniform(22, 65, 300)
    inc = np.exp(rng.normal(2.6, 0.45, 300)) + 0.25 * (age - 22)     # thu nhập, triệu đồng/tháng
    inc[:4] = [180, 220, 260, 300]                                  # vài người thu nhập rất cao
    return np.c_[inc, age]


def fig_scalers():
    X = scale_data()
    tfs = [("(a) dữ liệu gốc", None), ("(b) MinMaxScaler", MinMaxScaler()), ("(c) StandardScaler", StandardScaler()),
           ("(d) RobustScaler", RobustScaler()), ("(e) QuantileTransformer\n(phân phối chuẩn)", QuantileTransformer(n_quantiles=300, output_distribution="normal"))]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
    col = np.where(np.arange(len(X)) < 4, PAL[1], PAL[0])
    for ax, (name, tf) in zip(axes, tfs):
        Z = X if tf is None else tf.fit_transform(X)
        ax.scatter(Z[:, 0], Z[:, 1], s=8, c=col, alpha=0.8)
        main = Z[4:, 0]
        ax.set_title(name, fontsize=9.5)
        ax.set_xlabel("thu nhập" + (" (triệu/tháng)" if tf is None else "")); ax.set_ylabel("tuổi" if tf is None else "")
        print(f"  {name.splitlines()[0]:24s} thu nhập: 296 điểm thường nằm trong [{main.min():.2f}, {main.max():.2f}] "
              f"(rộng {main.max() - main.min():.2f}); cả cột [{Z[:, 0].min():.2f}, {Z[:, 0].max():.2f}]; tuổi [{Z[:, 1].min():.2f}, {Z[:, 1].max():.2f}]")
    fig.tight_layout()
    save(fig, "scalers")


# ------------------------------------------------------------------ 5. vì sao phải co giãn
def fig_scale_effect():
    rng = np.random.default_rng(5)
    n = 400
    a = rng.uniform(0, 1, n); b = rng.uniform(0, 1, n)
    y = ((a - 0.5) ** 2 + (b - 0.5) ** 2 < 0.12).astype(int)
    flip = rng.random(n) < 0.05; y[flip] = 1 - y[flip]
    X = np.c_[a * 1000, b]                                      # cột 1 đo bằng đơn vị lớn gấp 1000
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    raw = KNeighborsClassifier(7).fit(Xtr, ytr)
    sc = make_pipeline(StandardScaler(), KNeighborsClassifier(7)).fit(Xtr, ytr)
    acc_raw, acc_sc = raw.score(Xte, yte), sc.score(Xte, yte)
    print(f"  kNN (k=7): accuracy không co giãn {acc_raw:.3f}, có StandardScaler {acc_sc:.3f}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    gx, gy = np.meshgrid(np.linspace(0, 1000, 220), np.linspace(0, 1, 220))
    G = np.c_[gx.ravel(), gy.ravel()]
    for ax, mdl, name, acc in [(axes[0], raw, "(a) kNN trên dữ liệu gốc", acc_raw), (axes[1], sc, "(b) kNN sau StandardScaler", acc_sc)]:
        Z = mdl.predict(G).reshape(gx.shape)
        ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5], colors=["#dbe8fd", "#fcdbe6"])
        ax.scatter(Xte[:, 0], Xte[:, 1], s=9, c=PAL[yte])
        ax.set_xlabel("đặc trưng 1 (đơn vị: 0…1000)"); ax.set_ylabel("đặc trưng 2 (đơn vị: 0…1)")
        ax.set_title(f"{name}\naccuracy trên tập kiểm tra = {acc:.2f}")
    ax = axes[2]
    rng = np.random.default_rng(6)
    m = 200
    f1 = rng.normal(0, 1, m); f2 = rng.normal(0, 30, m)
    yy = 1.5 * f1 + 0.05 * f2 + rng.normal(0, 0.3, m)
    def gd(A, t, lr, iters=500):
        A = np.c_[np.ones(len(A)), A]; w = np.zeros(A.shape[1]); out = []
        for _ in range(iters):
            g = A.T @ (A @ w - t) / len(t); w -= lr * g
            out.append(np.mean((A @ w - t) ** 2) / 2)
        return np.array(out)
    A_raw = np.c_[f1, f2]; A_sc = StandardScaler().fit_transform(A_raw)
    def lmax(A):
        A1 = np.c_[np.ones(len(A)), A]; return np.linalg.eigvalsh(A1.T @ A1 / len(A1)).max()
    def cond(A):
        A1 = np.c_[np.ones(len(A)), A]; e = np.linalg.eigvalsh(A1.T @ A1 / len(A1)); return e.max() / e.min()
    lr_raw, lr_sc = 1.9 / lmax(A_raw), 1.9 / lmax(A_sc)
    L_raw, L_sc = gd(A_raw, yy, lr_raw), gd(A_sc, yy, lr_sc)
    best = np.min([L_raw.min(), L_sc.min()])
    w_opt = np.linalg.lstsq(np.c_[np.ones(m), A_raw], yy, rcond=None)[0]
    Lopt = np.mean((np.c_[np.ones(m), A_raw] @ w_opt - yy) ** 2) / 2
    it = lambda L: int(np.argmax(L - Lopt < 1e-3 * Lopt)) + 1 if np.any(L - Lopt < 1e-3 * Lopt) else None
    print(f"  GD: số điều kiện gốc {cond(A_raw):.0f}, sau co giãn {cond(A_sc):.2f}; "
          f"số vòng tới sai số 0.1% của tối ưu: gốc {it(L_raw)}, co giãn {it(L_sc)}")
    ax.semilogy(np.arange(1, 501), L_raw - Lopt + 1e-12, color=PAL[1], lw=2, label=f"đặc trưng gốc (số điều kiện {cond(A_raw):.0f})")
    ax.semilogy(np.arange(1, 501), L_sc - Lopt + 1e-12, color=PAL[0], lw=2, label=f"sau StandardScaler (số điều kiện {cond(A_sc):.1f})")
    ax.set_ylim(1e-8, 10)
    ax.set_xlabel("vòng lặp gradient descent"); ax.set_ylabel("hàm mất mát − tối ưu")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("(c) Hồi quy tuyến tính, hai đặc trưng lệch thang đo 30 lần,\nmỗi bên dùng tốc độ học lớn nhất còn ổn định")
    fig.tight_layout()
    save(fig, "scale_effect")


# ------------------------------------------------------------------ 6. biến đổi phân phối lệch
def fig_power():
    cal = fetch_california_housing()
    v = cal.data[:, cal.feature_names.index("Population")]
    tfs = [("(a) Population (gốc)", v),
           ("(b) log(1 + x)", np.log1p(v)),
           ("(c) Yeo–Johnson", PowerTransformer(method="yeo-johnson").fit_transform(v[:, None])[:, 0]),
           ("(d) Quantile → chuẩn", QuantileTransformer(output_distribution="normal", random_state=0).fit_transform(v[:, None])[:, 0])]
    pt = PowerTransformer(method="yeo-johnson").fit(v[:, None])
    print(f"  Population: n={len(v)}, min {v.min():.0f}, trung vị {np.median(v):.0f}, max {v.max():.0f}; λ Yeo–Johnson = {pt.lambdas_[0]:.3f}")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.0))
    for ax, (name, w) in zip(axes, tfs):
        sk = stats.skew(w)
        ax.hist(w, bins=60, color=PAL[0], alpha=0.8)
        ax.set_yticks([]); ax.set_title(f"{name}\nđộ lệch (skewness) = {sk:.2f}", fontsize=9.5)
        print(f"  {name:22s} skew = {sk:.3f}")
    fig.tight_layout()
    save(fig, "power")


# ------------------------------------------------------------------ 7. tạo đặc trưng phi tuyến
def fig_binning():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 10, 200)); y = np.sin(x) + 0.1 * x + rng.normal(0, 0.25, 200)
    xt = rng.uniform(0, 10, 2000); yt = np.sin(xt) + 0.1 * xt + rng.normal(0, 0.25, 2000)
    g = np.linspace(0, 10, 400)[:, None]
    models = [("x thô", LinearRegression()),
              ("chia 10 khoảng + one-hot", make_pipeline(KBinsDiscretizer(10, encode="onehot-dense", strategy="uniform", quantile_method="averaged_inverted_cdf"), LinearRegression())),
              ("spline bậc 3, 8 nút", make_pipeline(SplineTransformer(n_knots=8, degree=3), LinearRegression()))]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.scatter(x, y, s=8, c=LIGHT)
    for k, (name, mdl) in enumerate(models):
        mdl.fit(x[:, None], y)
        r2 = mdl.score(xt[:, None], yt)
        ax.plot(g[:, 0], mdl.predict(g), color=PAL[[1, 3, 0][k]], lw=2, label=f"{name}: R² kiểm tra = {r2:.2f}")
        print(f"  {name:28s} R² test = {r2:.3f}")
    ax.legend(fontsize=8.2, frameon=False, loc="lower left"); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    save(fig, "binning")


# ------------------------------------------------------------------ 8. chọn đặc trưng
def fig_selection():
    X, y = make_friedman1(n_samples=800, n_features=10, noise=1.0, random_state=0)
    F, _ = f_regression(X, y)
    MI = mutual_info_regression(X, y, random_state=0)
    print("  Friedman #1, F-score:", np.round(F, 1).tolist())
    print("  Friedman #1, MI     :", np.round(MI, 3).tolist())
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.7), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    idx = np.arange(10)
    ax.bar(idx - 0.2, F / F.max(), 0.4, color=PAL[1], label="F-score (tương quan tuyến tính)")
    ax.bar(idx + 0.2, MI / MI.max(), 0.4, color=PAL[0], label="thông tin tương hỗ")
    ax.set_xticks(idx); ax.set_xticklabels([f"$x_{{{i + 1}}}$" for i in idx])
    ax.axvspan(4.5, 9.6, color=LIGHT, alpha=0.25); ax.text(7, 1.02, "5 đặc trưng nhiễu", ha="center", fontsize=8.5, color=MUTED)
    ax.set_ylabel("điểm (chia cho giá trị lớn nhất)"); ax.legend(fontsize=7.8, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.93))
    ax.set_title("(a) Lọc: $x_3$ tác động theo $(x_3-0.5)^2$,\nF-score không thấy, thông tin tương hỗ thấy")
    ax = axes[1]
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import RFECV
    rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    rfe = RFECV(rf, step=1, cv=5, scoring="r2", n_jobs=-1).fit(X, y)
    sc = rfe.cv_results_["mean_test_score"]; sd = rfe.cv_results_["std_test_score"]
    ks = np.arange(1, 11)
    ax.errorbar(ks, sc, yerr=sd, color=PAL[0], marker="o", ms=4, capsize=3)
    ax.axvline(rfe.n_features_, color=PAL[1], ls="--", lw=1.2)
    ax.set_xlabel("số đặc trưng giữ lại"); ax.set_ylabel("R² kiểm định chéo"); ax.set_xticks(ks)
    ax.set_title(f"(b) Bọc: RFE với rừng ngẫu nhiên chọn {rfe.n_features_} đặc trưng\n" +
                 ", ".join(f"$x_{{{i + 1}}}$" for i in np.where(rfe.support_)[0]))
    print("  RFECV R²:", np.round(sc, 3).tolist(), "chọn", (np.where(rfe.support_)[0] + 1).tolist())
    ax = axes[2]
    Xs = StandardScaler().fit_transform(X)
    alphas, coefs, _ = lasso_path(Xs, y - y.mean(), n_alphas=100)
    for i in range(10):
        ax.plot(alphas, coefs[i], color=PAL[i] if i < 5 else LIGHT, lw=2 if i < 5 else 1.2,
                label=f"$x_{{{i + 1}}}$" if i < 5 else ("$x_6…x_{10}$" if i == 5 else None))
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("α (giảm dần →)"); ax.set_ylabel("hệ số")
    ax.legend(fontsize=7.8, frameon=False, ncol=2, loc="upper left")
    order = [int(np.argmax(np.abs(coefs[i]) > 1e-9)) for i in range(10)]
    print("  Lasso: thứ tự đi vào mô hình:", (np.argsort(order) + 1).tolist(), "α vào:", [round(float(alphas[o]), 3) for o in order])
    ax.set_title("(c) Nhúng: đường đi Lasso, đặc trưng nào\nđi vào mô hình trước là quan trọng hơn")
    fig.tight_layout()
    save(fig, "selection")


# ------------------------------------------------------------------ 9. rò rỉ dữ liệu
def fig_leakage(reps=30):
    leak, good = [], []
    for s in range(reps):
        rng = np.random.default_rng(s)
        X = rng.normal(size=(60, 5000)); y = np.repeat([0, 1], 30); rng.shuffle(y)   # nhãn ngẫu nhiên: không học được gì
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        Xsel = SelectKBest(f_classif, k=20).fit_transform(X, y)                       # SAI: chọn trên toàn bộ dữ liệu
        leak.append(cross_val_score(LogisticRegression(max_iter=1000), Xsel, y, cv=cv).mean())
        pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
        good.append(cross_val_score(pipe, X, y, cv=cv).mean())
    leak, good = np.array(leak), np.array(good)
    print(f"  rò rỉ: chọn trước rồi CV: {leak.mean():.3f} ± {leak.std():.3f} (min {leak.min():.2f}); "
          f"chọn trong từng fold: {good.mean():.3f} ± {good.std():.3f}")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    rng = np.random.default_rng(0)
    for k, (v, name, c) in enumerate([(leak, "chọn 20 đặc trưng trên\ntoàn bộ dữ liệu, rồi CV", PAL[1]),
                                      (good, "chọn đặc trưng bên trong\ntừng fold (Pipeline)", PAL[0])]):
        ax.scatter(v, k + rng.uniform(-0.12, 0.12, len(v)), s=18, c=c, alpha=0.75)
        ax.plot([v.mean()] * 2, [k - 0.25, k + 0.25], color=INK, lw=2.5)
        ax.text(v.mean(), k + 0.32, f"trung bình {v.mean():.2f}", ha="center", fontsize=8.5, color=INK)
    ax.axvline(0.5, color=MUTED, ls=":", lw=1.2); ax.text(0.505, -0.55, "đoán mò = 0.5", fontsize=8, color=MUTED)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["chọn trên toàn bộ\ndữ liệu, rồi CV", "chọn bên trong\ntừng fold (Pipeline)"])
    ax.set_ylim(-0.7, 1.6); ax.set_xlim(0.2, 1.02); ax.set_xlabel("accuracy kiểm định chéo 5-fold")
    fig.tight_layout()
    save(fig, "leakage")


# ------------------------------------------------------------------ 10. sơ đồ fit / transform
def box(ax, x, y, w, h, text, fc, ec=INK, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=INK, ls="-"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, ls=ls, shrinkA=0, shrinkB=0))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 4.1))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 4.3); ax.axis("off")
    steps = ["Điền thiếu", "Mã hoá", "Co giãn", "Chọn đặc trưng", "Mô hình"]
    learned = ["trung vị\nmỗi cột", "danh sách\nnhóm", "μ, σ mỗi cột", "chỉ số cột\nđược chọn", "trọng số w"]
    xs = [1.9 + 1.9 * i for i in range(5)]
    box(ax, 0.1, 2.85, 1.35, 0.8, "Tập huấn luyện\nX_train, y", "#dbe8fd", fs=9)
    box(ax, 0.1, 0.35, 1.35, 0.8, "Tập kiểm tra\nX_test", "#fcdbe6", fs=9)
    for i, (x, s, l) in enumerate(zip(xs, steps, learned)):
        box(ax, x, 2.85, 1.45, 0.8, s + "\n.fit_transform" if i < 4 else s + "\n.fit", "#eef0fa", fs=8.5)
        box(ax, x + 0.12, 1.75, 1.21, 0.6, l, "#fff6dc", ec=PAL[3], fs=8)
        box(ax, x, 0.35, 1.45, 0.8, s + "\n.transform" if i < 4 else s + "\n.predict", "#eef0fa", fs=8.5)
        arrow(ax, x + 0.725, 2.85, x + 0.725, 2.35, PAL[3]); arrow(ax, x + 0.725, 1.75, x + 0.725, 1.15, PAL[3])
        prev = 1.45 if i == 0 else xs[i - 1] + 1.45
        arrow(ax, prev + 0.02, 3.25, x - 0.02, 3.25); arrow(ax, prev + 0.02, 0.75, x - 0.02, 0.75)
    ax.text(xs[-1] + 1.55, 0.75, "ŷ_test", va="center", fontsize=10, color=INK)
    ax.text(5.75, 4.05, "Học tham số CHỈ từ tập huấn luyện", ha="center", fontsize=10.5, color=PAL[0], weight="bold")
    ax.text(5.75, 0.02, "Tập kiểm tra chỉ đi qua các tham số đã học, không bao giờ được dùng để học chúng", ha="center", fontsize=10, color=PAL[1], weight="bold")
    save(fig, "pipeline")


if __name__ == "__main__":
    names = sys.argv[1:] or ["missing", "outliers", "encoding", "scalers", "scale_effect", "power", "binning", "selection", "leakage", "pipeline"]
    for n in names:
        print(n); globals()["fig_" + n]()
