"""Vẽ hình minh hoạ cho bài bổ sung "Phân cụm nâng cao" (docs/ml/cluster_vi.md).

Dùng:  python3 -B scripts/ml_extra/cluster_figs.py
Hình SVG lưu ở docs/img/ml/cluster/. Các con số in ra là số được dẫn trong bài.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parents[2] / "docs" / "img" / "ml" / "cluster"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "path", "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})
PAL = np.array(["#2f80f5", "#e8487f", "#16a34a", "#f08a24", "#7c3aed", "#0ea5a4", "#a16207", "#64748b"])
NOISE = "#b8bdd3"
INK, MUTED = "#1f2340", "#8a90b4"


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print("  ->", name + ".svg")


def colors(lab):
    lab = np.asarray(lab)
    return np.where(lab < 0, NOISE, PAL[np.clip(lab, 0, len(PAL) - 1) % len(PAL)])


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def ellipse(ax, mean, cov, color, nsig=(1, 2), lw=1.8):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for k in nsig:
        ax.add_patch(Ellipse(mean, 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0]), angle=ang,
                             fill=False, ec=color, lw=lw if k == 1 else lw * 0.6, ls="-" if k == 1 else "--"))


# ------------------------------------------------------------------ dữ liệu dùng chung
def data_aniso(seed=170):
    X, y = make_blobs(n_samples=600, centers=3, random_state=seed)
    return X @ np.array([[0.6, -0.64], [-0.41, 0.85]]), y


def data_varied(seed=170):
    return make_blobs(n_samples=600, cluster_std=[1.0, 2.5, 0.5], random_state=seed)


def data_unequal(seed=3):
    rng = np.random.default_rng(seed)
    A = rng.normal([0, 0], 1.6, (500, 2)); B = rng.normal([4.2, 0.6], 0.45, (60, 2))
    return np.r_[A, B], np.r_[np.zeros(500, int), np.ones(60, int)]


def data_moons():
    return make_moons(n_samples=500, noise=0.06, random_state=0)


def data_circles():
    return make_circles(n_samples=500, factor=0.45, noise=0.05, random_state=0)


# ------------------------------------------------------------------ 1. K-means thất bại ở đâu
def fig_kmeans_fail():
    sets = [("cụm bị kéo dẹt", data_aniso(), 3), ("cụm khác độ rộng", data_varied(), 3),
            ("cụm khác kích thước", data_unequal(), 2), ("hình dạng không lồi", data_moons(), 2)]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))
    for j, (name, (X, y), k) in enumerate(sets):
        km = KMeans(k, n_init=10, random_state=0).fit(X)
        axes[0, j].scatter(*X.T, s=5, c=colors(y)); axes[0, j].set_title(f"{name}\nnhãn thật", fontsize=9.5)
        axes[1, j].scatter(*X.T, s=5, c=colors(km.labels_))
        axes[1, j].scatter(*km.cluster_centers_.T, s=90, marker="X", c=INK, edgecolors="white")
        ari = adjusted_rand_score(y, km.labels_)
        axes[1, j].set_title(f"K-means: ARI = {ari:.2f}", fontsize=9.5)
        print(f"  K-means trên '{name}': ARI = {ari:.3f}")
        for a in axes[:, j]: clean(a); a.set_aspect("equal", "datalim")
    fig.tight_layout()
    save(fig, "kmeans_fail")


# ------------------------------------------------------------------ 2. hỗn hợp Gauss một chiều
def gpdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def fig_mixture_1d():
    pis, mus, sds = [0.5, 0.3, 0.2], [-2.0, 1.0, 3.5], [1.0, 0.6, 0.8]
    x = np.linspace(-6, 6.5, 800)
    rng = np.random.default_rng(0)
    z = rng.choice(3, 400, p=pis); s = rng.normal(np.array(mus)[z], np.array(sds)[z])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    tot = np.zeros_like(x)
    for k in range(3):
        c = pis[k] * gpdf(x, mus[k], sds[k]); tot += c
        ax.fill_between(x, c, color=PAL[k], alpha=0.25); ax.plot(x, c, color=PAL[k], lw=1.6,
                                                                   label=f"$\\pi_{k + 1}\\,\\mathcal{{N}}(x\\mid{mus[k]:g},\\,{sds[k]:g}^2)$, $\\pi_{k + 1}={pis[k]:g}$")
    ax.plot(x, tot, color=INK, lw=2.4, label="$p(x)=\\sum_k \\pi_k\\,\\mathcal{N}(x\\mid\\mu_k,\\sigma_k^2)$")
    ax.hist(s, bins=40, density=True, color=MUTED, alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper left"); ax.set_yticks([]); ax.set_xlabel("x")
    ax.set_ylim(0, 0.42)
    ax.set_title("(a) Mật độ hỗn hợp = tổng có trọng số của các Gauss")
    ax = axes[1]
    R = np.array([pis[k] * gpdf(x, mus[k], sds[k]) for k in range(3)]); R /= R.sum(0)
    ax.stackplot(x, R, colors=PAL[:3], alpha=0.75)
    ax.set_xlim(-6, 6.5); ax.set_ylim(0, 1); ax.set_xlabel("x"); ax.set_ylabel("$\\gamma_k(x)=P(z=k\\mid x)$")
    for xv in [-0.4, 2.3]:
        ax.axvline(xv, color=INK, lw=1, ls=":")
        g = R[:, np.argmin(np.abs(x - xv))]
        ax.text(xv + 0.1, 0.92, " / ".join(f"{v:.2f}" for v in g), fontsize=8, color=INK)
    ax.set_title("(b) Trách nhiệm: xác suất mỗi điểm thuộc từng cụm")
    fig.tight_layout()
    save(fig, "mixture_1d")


# ------------------------------------------------------------------ 3. EM từng bước
def em_steps(X, K, iters, seed=2):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    mu = X[rng.choice(n, K, replace=False)]
    cov = np.array([np.cov(X.T) for _ in range(K)]); pi = np.ones(K) / K
    hist = [(mu.copy(), cov.copy(), pi.copy(), None)]
    ll = []
    for t in range(iters):
        logp = np.stack([np.log(pi[k]) + logN(X, mu[k], cov[k]) for k in range(K)], 1)
        m = logp.max(1, keepdims=True); lse = m[:, 0] + np.log(np.exp(logp - m).sum(1))
        ll.append(lse.mean())
        R = np.exp(logp - lse[:, None])
        Nk = R.sum(0); pi = Nk / n; mu = (R.T @ X) / Nk[:, None]
        cov = np.array([((R[:, k:k + 1] * (X - mu[k])).T @ (X - mu[k])) / Nk[k] + 1e-6 * np.eye(d) for k in range(K)])
        hist.append((mu.copy(), cov.copy(), pi.copy(), R))
    return hist, ll


def logN(X, m, C):
    d = X.shape[1]; L = np.linalg.cholesky(C); z = np.linalg.solve(L, (X - m).T)
    return -0.5 * (z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)


def fig_em():
    X, y = data_aniso()
    hist, ll = em_steps(X, 3, 60)
    fig = plt.figure(figsize=(13, 3.6))
    for i, it in enumerate([0, 1, 5, 40]):
        ax = fig.add_axes([0.0 + i * 0.205, 0.0, 0.19, 0.84])
        mu, cov, pi, R = hist[it]
        c = (R @ np.array([matplotlib.colors.to_rgb(h) for h in PAL[:3]])) if R is not None else np.full((len(X), 3), 0.6)
        ax.scatter(*X.T, s=5, c=np.clip(c, 0, 1))
        for k in range(3): ellipse(ax, mu[k], cov[k], PAL[k])
        ax.scatter(*mu.T, s=60, marker="X", c=INK, edgecolors="white", zorder=3)
        ax.set_title("khởi tạo" if it == 0 else f"sau {it} vòng EM", fontsize=10)
        clean(ax); ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1); ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    ax = fig.add_axes([0.85, 0.14, 0.15, 0.7])
    ax.plot(range(1, len(ll) + 1), ll, color=INK, lw=2)
    ax.set_xscale("log"); ax.set_xlabel("vòng EM"); ax.set_ylabel("log-likelihood / điểm")
    ax.grid(alpha=0.25)
    print(f"  EM: log-lik/điểm vòng 1 {ll[0]:.3f}, vòng 5 {ll[4]:.3f}, vòng 60 {ll[-1]:.3f};"
          f" luôn tăng: {bool(np.all(np.diff(ll) > -1e-9))}")
    gm_lab = np.argmax(hist[-1][3], 1)
    print(f"  EM ARI so với nhãn thật: {adjusted_rand_score(y, gm_lab):.3f}")
    save(fig, "em")


# ------------------------------------------------------------------ 4. kiểu ma trận hiệp phương sai và BIC
def fig_cov_types():
    X, _ = data_aniso()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, ct, name in zip(axes, ["spherical", "diag", "tied", "full"],
                            ["spherical: hình tròn", "diag: elip song song trục", "tied: cùng một elip", "full: elip tự do"]):
        g = GaussianMixture(3, covariance_type=ct, random_state=0, n_init=3).fit(X)
        lab = g.predict(X)
        ax.scatter(*X.T, s=4, c=colors(lab))
        for k in range(3):
            C = {"spherical": np.eye(2) * g.covariances_[k] if ct == "spherical" else None,
                 "diag": np.diag(g.covariances_[k]) if ct == "diag" else None,
                 "tied": g.covariances_ if ct == "tied" else None,
                 "full": g.covariances_[k] if ct == "full" else None}[ct]
            ellipse(ax, g.means_[k], C, INK, nsig=(1.5,), lw=1.5)
        npar = g._n_parameters()
        ax.set_title(f"{name}\n{npar} tham số, BIC = {g.bic(X):.0f}", fontsize=9.5)
        clean(ax); ax.set_aspect("equal", "datalim")
        print(f"  {ct}: số tham số {npar}, BIC {g.bic(X):.1f}")
    fig.tight_layout()
    save(fig, "cov_types")


def fig_bic():
    X, _ = make_blobs(n_samples=800, centers=[[0, 0], [4, 0], [2, 3.5], [6, 4]], cluster_std=[0.8, 0.8, 0.8, 1.0], random_state=4)
    Ks = range(1, 10)
    bic, aic, ll = [], [], []
    for k in Ks:
        g = GaussianMixture(k, random_state=0, n_init=3).fit(X)
        bic.append(g.bic(X)); aic.append(g.aic(X)); ll.append(-2 * g.score(X) * len(X))
    kb = list(Ks)[int(np.argmin(bic))]
    print(f"  BIC nhỏ nhất ở K = {kb}; AIC nhỏ nhất ở K = {list(Ks)[int(np.argmin(aic))]}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1, 1.4]})
    g = GaussianMixture(kb, random_state=0, n_init=3).fit(X)
    axes[0].scatter(*X.T, s=4, c=colors(g.predict(X)))
    for k in range(kb): ellipse(axes[0], g.means_[k], g.covariances_[k], INK, nsig=(2,), lw=1.4)
    clean(axes[0]); axes[0].set_aspect("equal", "datalim"); axes[0].set_title(f"GMM với K = {kb}")
    ax = axes[1]
    ax.plot(Ks, ll, "o-", color=MUTED, label="$-2\\log L$ (chỉ khớp dữ liệu)")
    ax.plot(Ks, bic, "s-", color=PAL[1], lw=2, label="BIC $= -2\\log L + p\\ln N$")
    ax.plot(Ks, aic, "^-", color=PAL[0], label="AIC $= -2\\log L + 2p$")
    ax.axvline(kb, color=INK, ls=":")
    ax.set_xlabel("số cụm K"); ax.legend(frameon=False, fontsize=8.5); ax.grid(alpha=0.25)
    ax.set_title("Chọn K: phạt số tham số")
    save(fig, "bic")


# ------------------------------------------------------------------ 5. DBSCAN
def fig_dbscan_concept():
    rng = np.random.default_rng(7)
    A = rng.normal([0, 0], 0.35, (22, 2)); B = rng.normal([2.4, 0.6], 0.3, (16, 2))
    N = np.array([[1.2, 1.6], [-1.3, 1.3], [3.6, -0.9], [1.3, -1.2]])
    X = np.r_[A, B, N]
    eps, mp = 0.45, 6
    db = DBSCAN(eps=eps, min_samples=mp).fit(X)
    core = np.zeros(len(X), bool); core[db.core_sample_indices_] = True
    lab = db.labels_
    border = (lab >= 0) & ~core
    print(f"  DBSCAN minh hoạ: {core.sum()} điểm lõi, {border.sum()} điểm biên, {(lab < 0).sum()} nhiễu, {lab.max() + 1} cụm")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i in np.where(core)[0]:
        ax.add_patch(Circle(X[i], eps, fc=PAL[lab[i]], alpha=0.06, ec="none"))
    i0 = np.where(core)[0][np.argmin(np.linalg.norm(X[core] - [0.05, 0.05], axis=1))]
    ax.add_patch(Circle(X[i0], eps, fill=False, ec=INK, lw=1.4, ls="--"))
    ax.annotate(f"ε = {eps}", X[i0] + [eps * 0.7, eps * 0.72], X[i0] + [0.75, 1.05], fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK))
    ax.scatter(*X[core].T, s=55, c=colors(lab[core]), edgecolors=INK, linewidths=0.8, label=f"điểm lõi (≥ {mp} điểm trong bán kính ε, kể cả chính nó)")
    ax.scatter(*X[border].T, s=55, facecolors="white", edgecolors=colors(lab[border]), linewidths=2, label="điểm biên (gần một điểm lõi)")
    ax.scatter(*X[lab < 0].T, s=60, marker="x", c=INK, label="nhiễu")
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.02))
    clean(ax); ax.set_aspect("equal"); ax.set_xlim(-2, 4.3); ax.set_ylim(-1.7, 2.3)
    save(fig, "dbscan_concept")


def fig_dbscan_eps():
    X, y = data_moons()
    rng = np.random.default_rng(1)
    X = np.r_[X, rng.uniform([-1.3, -0.8], [2.3, 1.3], (25, 2))]
    k = 5
    d = np.sort(NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)[0][:, -1])
    fig = plt.figure(figsize=(12.5, 3.3))
    ax = fig.add_axes([0.0, 0.12, 0.22, 0.78])
    ax.plot(d, color=INK, lw=2); ax.set_xlabel("điểm, sắp theo khoảng cách"); ax.set_ylabel(f"khoảng cách tới láng giềng thứ {k - 1}")
    ax.grid(alpha=0.25); ax.set_title("đồ thị k-distance", fontsize=10)
    for i, eps in enumerate([0.05, 0.15, 0.45]):
        ax.axhline(eps, color=PAL[[3, 2, 1][i]], ls="--", lw=1.2)
        a = fig.add_axes([0.27 + i * 0.245, 0.0, 0.23, 0.9])
        lab = DBSCAN(eps=eps, min_samples=k).fit(X).labels_
        a.scatter(*X.T, s=7, c=colors(lab))
        nc, nn = lab.max() + 1, (lab < 0).sum()
        a.set_title(f"ε = {eps}: {nc} cụm, {nn} điểm nhiễu", fontsize=10, color=PAL[[3, 2, 1][i]])
        clean(a); a.set_aspect("equal", "datalim")
        print(f"  DBSCAN moons ε={eps}: {nc} cụm, {nn} nhiễu")
    save(fig, "dbscan_eps")


# ------------------------------------------------------------------ 6. phân cụm phân cấp
def fig_dendrogram():
    rng = np.random.default_rng(4)
    X = np.r_[rng.normal([0, 0], 0.5, (6, 2)), rng.normal([3, 0.5], 0.5, (5, 2)), rng.normal([1.5, 3], 0.5, (5, 2))]
    Z = linkage(X, "ward")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
    cut = (Z[-3, 2] + Z[-2, 2]) / 2
    from scipy.cluster.hierarchy import set_link_color_palette
    set_link_color_palette(list(PAL[:3]))
    dn = dendrogram(Z, no_plot=True, color_threshold=cut)
    leaf_col = dict(zip(dn["leaves"], dn["leaves_color_list"]))
    for i, p in enumerate(X):
        axes[0].scatter(*p, s=140, c=leaf_col[i], edgecolors="white", zorder=3)
        axes[0].text(*p, str(i), ha="center", va="center", fontsize=7.5, color="white", zorder=4, weight="bold")
    clean(axes[0]); axes[0].set_aspect("equal", "datalim"); axes[0].set_title("16 điểm, cắt thành 3 cụm")
    dendrogram(Z, ax=axes[1], color_threshold=cut, above_threshold_color=MUTED, leaf_font_size=8)
    axes[1].axhline(cut, color=INK, ls="--", lw=1.2)
    axes[1].text(axes[1].get_xlim()[1], cut, " cắt ở đây → 3 cụm", va="bottom", ha="right", fontsize=9)
    axes[1].set_ylabel("khoảng cách khi gộp (Ward)"); axes[1].set_title("Dendrogram: lịch sử gộp từ dưới lên")
    axes[1].spines["left"].set_visible(True)
    save(fig, "dendrogram")


def fig_linkage():
    sets = [("hai nửa trăng", data_moons()[0], 2), ("vòng tròn lồng nhau", data_circles()[0], 2),
            ("khác độ rộng", data_varied()[0], 3)]
    links = ["single", "complete", "average", "ward"]
    names = {"single": "single (gần nhất)", "complete": "complete (xa nhất)", "average": "average (trung bình)", "ward": "Ward (tăng phương sai)"}
    fig, axes = plt.subplots(3, 4, figsize=(11.5, 8))
    for i, (dn, X, k) in enumerate(sets):
        Xs = StandardScaler().fit_transform(X)
        for j, l in enumerate(links):
            lab = AgglomerativeClustering(k, linkage=l).fit_predict(Xs)
            ax = axes[i, j]; ax.scatter(*Xs.T, s=4, c=colors(lab)); clean(ax); ax.set_aspect("equal", "datalim")
            if i == 0: ax.set_title(names[l], fontsize=10)
            if j == 0: ax.set_ylabel(dn, fontsize=10)
    fig.tight_layout()
    save(fig, "linkage")


# ------------------------------------------------------------------ 7. silhouette
def fig_silhouette():
    X, _ = make_blobs(n_samples=500, centers=[[0, 0], [3.2, 0], [1.6, 3.0], [7, 2]], cluster_std=[0.7, 0.7, 0.7, 0.5], random_state=1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    scores = []
    for k in range(2, 9):
        scores.append(silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X)))
    print("  silhouette theo K:", dict(zip(range(2, 9), np.round(scores, 3))))
    for ax, k in zip(axes[:2], [2, 4]):
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
        s = silhouette_samples(X, lab)
        y0 = 0
        for c in range(k):
            v = np.sort(s[lab == c])
            ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=PAL[c], alpha=0.8)
            y0 += len(v) + 12
        ax.axvline(s.mean(), color=INK, ls="--", lw=1.2)
        ax.set_title(f"K = {k}: silhouette TB = {s.mean():.3f}", fontsize=10)
        ax.set_yticks([]); ax.set_xlim(-0.2, 1.45); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("hệ số silhouette s(i)")
        ins = ax.inset_axes([0.73, 0.3, 0.27, 0.4]); ins.scatter(*X.T, s=1.5, c=colors(lab)); clean(ins)
    ax = axes[2]
    ax.plot(range(2, 9), scores, "o-", color=INK, lw=2); ax.set_xlabel("số cụm K"); ax.set_ylabel("silhouette trung bình")
    ax.axvline(2 + int(np.argmax(scores)), color=PAL[1], ls=":"); ax.grid(alpha=0.25); ax.set_title("Chọn K theo silhouette", fontsize=10)
    fig.tight_layout()
    save(fig, "silhouette")


# ------------------------------------------------------------------ 8. so sánh tổng hợp
def fig_compare():
    rng = np.random.default_rng(0)
    sets = [("hai nửa trăng", *data_moons(), 2), ("vòng tròn", *data_circles(), 2),
            ("bị kéo dẹt", *data_aniso(), 3), ("khác độ rộng", *data_varied(), 3)]
    algs = ["K-means", "GMM (full)", "DBSCAN", "Ward", "single linkage"]
    eps = {"hai nửa trăng": 0.3, "vòng tròn": 0.3, "bị kéo dẹt": 0.17, "khác độ rộng": 0.18}
    fig, axes = plt.subplots(4, 5, figsize=(12.5, 10))
    for i, (dn, X, y, k) in enumerate(sets):
        Xs = StandardScaler().fit_transform(X)
        for j, a in enumerate(algs):
            if a == "K-means": lab = KMeans(k, n_init=10, random_state=0).fit_predict(Xs)
            elif a.startswith("GMM"): lab = GaussianMixture(k, random_state=0, n_init=5).fit(Xs).predict(Xs)
            elif a == "DBSCAN": lab = DBSCAN(eps=eps[dn], min_samples=8).fit_predict(Xs)
            elif a == "Ward": lab = AgglomerativeClustering(k, linkage="ward").fit_predict(Xs)
            else: lab = AgglomerativeClustering(k, linkage="single").fit_predict(Xs)
            ari = adjusted_rand_score(y, lab)
            ax = axes[i, j]; ax.scatter(*Xs.T, s=3, c=colors(lab)); clean(ax); ax.set_aspect("equal", "datalim")
            ax.text(0.98, 0.02, f"ARI {ari:.2f}", transform=ax.transAxes, ha="right", fontsize=9,
                    color=INK, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
            if i == 0: ax.set_title(a, fontsize=10.5)
            if j == 0: ax.set_ylabel(dn, fontsize=10)
            print(f"  {dn:<14} {a:<15} ARI {ari:.3f}" + (f" (nhiễu {np.sum(lab < 0)})" if a == "DBSCAN" else ""))
    fig.tight_layout()
    save(fig, "compare")


if __name__ == "__main__":
    for fn in [fig_kmeans_fail, fig_mixture_1d, fig_em, fig_cov_types, fig_bic, fig_dbscan_concept, fig_dbscan_eps,
               fig_dendrogram, fig_linkage, fig_silhouette, fig_compare]:
        print(fn.__name__)
        fn()
