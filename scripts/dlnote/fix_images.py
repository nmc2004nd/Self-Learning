"""Việt hoá các ảnh của dl_note có chữ tiếng Trung.

Mỗi ảnh: xoá (tô trắng) vùng chữ Trung rồi viết chữ Việt vào đó, hoặc cắt bỏ phần chữ.
Kết quả ghi vào scripts/dlnote/img_vi/<thư mục>/<tệp>; build.py ưu tiên ảnh ở đây hơn ảnh gốc.
Ảnh mà chữ Trung là nội dung chính (bảng, sơ đồ lịch sử GPU) thì không sửa ở đây mà gõ lại
thành bảng tiếng Việt trong bài.

Dùng (env conda `ai`):  python3 -B scripts/dlnote/fix_images.py --src <clone dl_note>
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "img_vi"
DEFAULT_SRC = Path("/tmp/claude-1000/-home-nmc-Downloads-easy-rl-vi/963822cf-d45a-4311-aa91-22a954d0ef64/scratchpad/dl_note_src/dl_note")
FONT = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
FONT_B = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
RED, GREEN, BLACK = (230, 50, 40), (60, 160, 70), (20, 20, 20)


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def white(d, box):
    d.rectangle(box, fill="white")


def text(d, xy, s, size, color=BLACK, bold=False, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=color, anchor=anchor)


def edit(src, rel, fn):
    im = Image.open(src / "images" / rel)
    im = im.convert("RGB")
    im = fn(im) or im
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, optimize=True)
    print("đã sửa", rel, im.size)


def pruning_pipeline(im):
    d = ImageDraw.Draw(im)
    white(d, (200, 476, 445, 522))
    white(d, (990, 468, 1350, 522))


def sparsity_grains(im):
    # Bỏ hai mũi tên chữ Trung, thay bằng hai dòng chú thích tiếng Việt dưới các lưới.
    top = im.crop((0, 0, im.width, 345))
    out = Image.new("RGB", (im.width, 345 + 150), "white")
    out.paste(top, (0, 0))
    d = ImageDraw.Draw(out)
    y1, y2 = 375, 445
    d.line((40, y1, im.width - 40, y1), fill=BLACK, width=4)
    d.polygon([(20, y1), (55, y1 - 14), (55, y1 + 14)], fill=BLACK)
    d.polygon([(im.width - 20, y1), (im.width - 55, y1 - 14), (im.width - 55, y1 + 14)], fill=BLACK)
    text(d, (40, y1 + 14), "Mịn (fine-grained), không đều", 38, BLACK, True)
    text(d, (40, y2 + 6), "chính xác cao · khó tăng tốc", 36, GREEN)
    text(d, (im.width - 40, y1 + 14), "Thô (coarse-grained), có cấu trúc", 38, BLACK, True, anchor="ra")
    text(d, (im.width - 40, y2 + 6), "kém chính xác hơn · dễ tăng tốc", 36, RED, anchor="ra")
    return out


def activation_sparse(im):
    d = ImageDraw.Draw(im)
    white(d, (50, 412, 670, 465))
    text(d, (60, 418), "Vòng đỏ: phần tử của feature map bị làm thưa", 27, RED)


def activation_pipeline(im):
    d = ImageDraw.Draw(im)
    white(d, (1650, 498, 1935, 549))
    text(d, (2020, 505), "Số neuron giảm", 34, RED, anchor="ra")


def gradient_sparse(im):
    d = ImageDraw.Draw(im)
    white(d, (1130, 518, 1372, 568))
    text(d, (1360, 522), "giảm chi phí truyền thông", 30, RED, anchor="ra")


def irregular(im):
    d = ImageDraw.Draw(im)
    white(d, (1320, 665, im.width, im.height))
    text(d, (1715, 672), "Không thân thiện với tăng tốc phần cứng, nhất là GPU", 30, RED, anchor="ma")
    d.rectangle((1355, 740, 2090, 840), fill=(56, 120, 40))
    text(d, (1722, 790), "Ma trận thưa rất không đều", 50, (255, 255, 255), anchor="mm")


def sensitivity(im):
    d = ImageDraw.Draw(im)
    white(d, (5, 92, 345, 134))
    text(d, (12, 94), "lớp tích chập đầu tiên", 30, RED)


def structured_compare(im):
    d = ImageDraw.Draw(im)
    white(d, (0, 795, im.width, im.height))
    white(d, (1488, 440, 1556, 484))
    text(d, (1550, 445), "bỏ", 30, RED, anchor="ra")
    for x, s in [(10, "a) Mô hình gốc"), (370, "b) Cắt tỉa mức filter"), (790, "c) Cắt tỉa mức stage"), (1210, "d) Cắt tỉa mức khối")]:
        text(d, (x, 800), s, 32)
    text(d, (im.width // 2, 850), "So sánh các cách cắt tỉa có cấu trúc", 32, RED, anchor="ma")


def weights_important(im):
    d = ImageDraw.Draw(im)
    white(d, (380, 0, 660, 50))
    text(d, (330, 6), "< 20% giá trị lớn nhất", 30, RED)
    white(d, (505, 200, 770, 258))
    text(d, (505, 210), "< 50% lớn nhất", 30, GREEN)
    white(d, (392, 662, im.width, 720))
    white(d, (338, 722, im.width, 778))
    white(d, (275, 778, im.width, im.height))
    text(d, (398, 668), "không quan trọng", 32)
    text(d, (345, 728), "ít quan trọng", 32)
    text(d, (282, 782), "quan trọng", 32)


def sigmoid_saturate(im):
    # Hình 3.3 sách Deep Learning: thay hai nhãn "饱和区" và bỏ dòng chú thích tiếng Trung.
    im = im.crop((0, 0, im.width, 445))
    d = ImageDraw.Draw(im)
    white(d, (150, 300, 250, 340))
    white(d, (675, 45, 775, 90))
    text(d, (110, 320), "vùng bão hoà", 22, RED, anchor="lm")
    text(d, (660, 68), "vùng bão hoà", 22, RED, anchor="lm")
    return im


def crop_caption(h):
    # Hình chụp sách bản tiếng Trung: giữ phần đồ thị, bỏ chú thích (chú thích dịch nằm trong bài).
    return lambda im: im.crop((0, 0, im.width, h))


def directional_visual(im):
    d = ImageDraw.Draw(im)
    white(d, (1300, 515, 1595, 568))
    text(d, (1300, 528), "đạo hàm theo", 30, RED, anchor="lm")
    text(d, (1300, 560), "hướng tại P", 30, RED, anchor="lm")
    return im


def fill(d, im, box, xy=None):
    """Tô vùng box bằng màu nền lấy tại điểm xy (mặc định góc trên trái của box)."""
    c = im.getpixel(xy or (box[0], box[1]))
    d.rectangle(box, fill=c)


def conv_fc_vs_conv(im):
    d = ImageDraw.Draw(im)
    white(d, (95, 198, 250, 235)); white(d, (470, 198, 580, 235)); white(d, (790, 35, 905, 135))
    text(d, (165, 217), "Lớp kết nối đầy đủ", 17, BLACK, True, "mm")
    text(d, (525, 217), "Lớp tích chập", 17, BLACK, True, "mm")
    text(d, (790, 55), "một đơn vị vào", 14, RED, anchor="lm")
    text(d, (790, 90), "các đơn vị ra", 14, RED, anchor="lm")
    text(d, (790, 108), "bị ảnh hưởng", 14, RED, anchor="lm")
    return im


def conv_weight_share(im):
    d = ImageDraw.Draw(im)
    white(d, (95, 190, 225, 228)); white(d, (470, 190, 570, 228))
    text(d, (155, 208), "Lớp kết nối đầy đủ", 16, BLACK, True, "mm")
    text(d, (515, 208), "Lớp tích chập", 16, BLACK, True, "mm")
    return im


def conv_local_slide(im):
    d = ImageDraw.Draw(im)
    white(d, (795, 40, 1338, 102))
    text(d, (800, 55), "Mỗi neuron (kernel) không cần nhìn cả ảnh:", 19, RED, anchor="lm")
    text(d, (800, 85), "chỉ nhìn một vùng, rồi trượt cửa sổ qua cả ảnh", 19, RED, anchor="lm")
    return im


def conv3d_label(im):
    d = ImageDraw.Draw(im)
    white(d, (920, 148, 1070, 174))
    text(d, (925, 161), "cộng từng phần tử", 15, (110, 80, 220), anchor="lm")
    return im


def pool_maxpool(im):
    im = im.crop((0, 0, im.width, 615))           # bỏ chú thích "汇聚层中最大汇聚过程示例"
    d = ImageDraw.Draw(im)
    white(d, (298, 315, 447, 347)); text(d, (446, 331), "nhóm đặc trưng vào", 17, BLACK, anchor="rm")
    white(d, (690, 283, 847, 312)); text(d, (846, 298), "nhóm đặc trưng ra", 17, BLACK, anchor="rm")
    white(d, (555, 428, 655, 456)); text(d, (605, 442), "gộp max", 18, BLACK, anchor="mm")
    white(d, (130, 576, 262, 606)); text(d, (261, 591), "đặc trưng vào", 17, BLACK, anchor="rm")
    white(d, (835, 513, 972, 545)); text(d, (971, 529), "đặc trưng ra", 17, BLACK, anchor="rm")
    white(d, (0, 606, 300, 615))
    return im


def cnn_structure(im):
    d = ImageDraw.Draw(im)
    im2 = im.crop((0, 0, im.width, 395))           # bỏ chú thích "常用的卷积网络整体结构"
    d = ImageDraw.Draw(im2)
    white(d, (0, 145, 75, 188)); text(d, (38, 166), "Vào", 26, BLACK, anchor="mm")
    fill(d, im2, (280, 110, 380, 160), (230, 105)); text(d, (330, 135), "Tích chập", 26, BLACK, anchor="mm")
    fill(d, im2, (950, 110, 1070, 160), (920, 100)); text(d, (1010, 135), "Gộp (pooling)", 26, BLACK, anchor="mm")
    fill(d, im2, (1270, 140, 1430, 192), (1250, 120)); text(d, (1348, 165), "Kết nối đầy đủ", 26, BLACK, anchor="mm")
    white(d, (1465, 238, 1690, 288)); text(d, (1580, 263), "phân loại nhiều lớp", 24, RED, anchor="mm")
    return im2


def normal_curve(im):
    d = ImageDraw.Draw(im)
    white(d, (60, 15, 550, 70)); text(d, (68, 42), "Hàm mật độ của phân phối chuẩn", 32, RED, anchor="lm")
    white(d, (1058, 372, 1440, 422)); text(d, (1062, 397), "độ lệch chuẩn (tham số tỉ lệ)", 27, RED, anchor="lm")
    white(d, (580, 692, 922, 740)); text(d, (585, 716), "trung bình (tham số vị trí)", 27, RED, anchor="lm")
    return im


def sigmoid_derivative(im):
    """Ảnh gốc là đồ thị có ghi chú viết tay tiếng Trung: vẽ lại sạch bằng matplotlib."""
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.linspace(-8, 8, 400)
    sg = 1 / (1 + np.exp(-x))
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=130)
    ax.plot(x, sg, color="#e24a33", label="sigmoid  σ(x)")
    ax.plot(x, sg * (1 - sg), color="#348abd", label="đạo hàm  σ'(x) = σ(x)(1 − σ(x))")
    ax.axvspan(4, 8, color="#8b5cf6", alpha=0.12)
    ax.axvspan(-8, -4, color="#8b5cf6", alpha=0.12)
    ax.annotate("cực đại 0,25 tại x = 0", xy=(0, 0.25), xytext=(0.6, 0.42), arrowprops=dict(arrowstyle="->", color="#555"), fontsize=10)
    ax.text(6, 0.55, "vùng gradient\ntiêu biến\n(σ' ≈ 0)", ha="center", fontsize=10, color="#6d28d9")
    ax.text(-6, 0.55, "vùng gradient\ntiêu biến", ha="center", fontsize=10, color="#6d28d9")
    ax.set_xlabel("x"); ax.set_title("sigmoid và đạo hàm"); ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return Image.open(buf).convert("RGB")


def neuron_arch(im):
    d = ImageDraw.Draw(im)
    white(d, (530, 408, 725, 450)); text(d, (627, 429), "hàm phi tuyến", 30, RED, anchor="mm")
    white(d, (530, 470, 725, 520)); text(d, (627, 495), "hàm kích hoạt", 34, BLACK, anchor="mm")
    white(d, (180, 582, 285, 632)); text(d, (232, 607), "trọng số", 32, BLACK, anchor="mm")
    white(d, (8, 760, 115, 815)); text(d, (12, 787), "đầu vào", 34, BLACK, anchor="lm")
    return im


def tanh_center(im):
    d = ImageDraw.Draw(im)
    fill(d, im, (498, 298, 692, 324), (700, 330))
    text(d, (503, 311), "tanh(x) có tâm tại 0", 14, RED, anchor="lm")
    return im


def act_summary(im):
    d = ImageDraw.Draw(im)
    fill(d, im, (208, 182, 358, 211), (205, 196))
    return im


def network_graph(im):
    d = ImageDraw.Draw(im)
    white(d, (0, 175, 335, 295))
    text(d, (8, 205), "Mạng neuron phi tuyến", 34, BLACK, anchor="lm")
    text(d, (8, 255), "phân loại nhiều lớp", 34, BLACK, anchor="lm")
    white(d, (0, 785, 415, 900))
    text(d, (8, 815), "Đồ thị tính xuôi", 34, BLACK, anchor="lm")
    text(d, (8, 865), "(chữ hoa: ma trận)", 30, BLACK, anchor="lm")
    return im


def _fig_to_image(fig):
    import io
    import matplotlib.pyplot as plt
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return Image.open(buf).convert("RGB")


def gd_concept(im):
    """Vẽ lại hình minh hoạ một bước gradient descent trên Y = (X-2)^2 + 1 (hình gốc chú thích tiếng Trung)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = lambda x: (x - 2) ** 2 + 1
    df = lambda x: 2 * (x - 2)
    x = np.linspace(0.6, 3.4, 300)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130)
    ax.plot(x, f(x), "k", lw=1.6)
    for x0, side in [(1.4, "trái"), (2.55, "phải")]:
        g = df(x0); x1 = x0 - 0.1 * g
        dx = 0.25 * np.sign(g)                       # đi theo hướng gradient thì f tăng: lên dốc
        ax.annotate("", xy=(x0 + dx, f(x0) + g * dx), xytext=(x0, f(x0)),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax.annotate("", xy=(x1, f(x0)), xytext=(x0, f(x0)), arrowprops=dict(arrowstyle="->", color="green", lw=2))
        ax.plot([x0, x0], [0.8, f(x0)], "r--", lw=0.8); ax.plot([x1, x1], [0.8, f(x1)], "g--", lw=0.8)
        ax.text(x0, 0.72, f"$x_0$={x0}", ha="center", fontsize=9, color="red")
        ax.text(x1, 0.60, f"$x_1$={x1:.2f}", ha="center", fontsize=9, color="green")
        tx = 0.62 if side == "trái" else 2.75
        ax.text(tx, 2.55, f"bên {side}: $x_0$={x0}\n$f'(x_0)$={g:.1f}, $\\eta$=0,1\n$x_1 = x_0-\\eta f'(x_0)$={x1:.2f}", fontsize=9, color="#1d4ed8")
    ax.plot(2, 1, "o", color="#7c3aed"); ax.text(2, 1.08, "cực tiểu", ha="center", fontsize=9, color="#7c3aed")
    ax.plot([], [], color="red", label="hướng gradient (lên dốc)")
    ax.plot([], [], color="green", label="bước đi: ngược hướng gradient")
    ax.legend(loc="upper center", fontsize=9)
    ax.set_xlim(0.55, 3.45); ax.set_ylim(0.5, 3.3)
    ax.set_xlabel("X (tham số)"); ax.set_ylabel("Y (hàm mục tiêu)"); ax.set_title("Y = (X − 2)² + 1")
    return _fig_to_image(fig)


def lr_compare(im):
    """Vẽ lại: quỹ đạo và đường mất mát của gradient descent trên f(x)=x² với các tốc độ học khác nhau."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    etas = [(0.02, "η quá nhỏ", "#2563eb"), (0.3, "η vừa phải", "#16a34a"), (0.9, "η khá lớn", "#c2410c"), (1.05, "η quá lớn", "#dc2626")]
    fig = plt.figure(figsize=(9, 4.6), dpi=120)
    xs = np.linspace(-3.2, 3.2, 200)
    for k, (eta, name, c) in enumerate(etas):
        ax = fig.add_subplot(2, 4, k + 1)
        x = [2.5]
        for _ in range(12):
            x.append(x[-1] - eta * 2 * x[-1])
        x = np.array(x)
        ax.plot(xs, xs ** 2, color="#999", lw=1)
        ax.plot(x, x ** 2, "o-", color=c, ms=3, lw=1)
        ax.set_title(name, color=c, fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-0.5, 11)
    ax = fig.add_subplot(2, 1, 2)
    for eta, name, c in etas:
        x = 2.5; L = []
        for _ in range(40):
            L.append(x * x); x -= eta * 2 * x
        ax.plot(L, color=c, label=f"{name} ({eta})")
    ax.set_ylim(0, 10); ax.set_xlabel("số vòng lặp"); ax.set_ylabel("f(x)"); ax.legend(fontsize=8, ncol=4, loc="upper center")
    fig.tight_layout()
    return _fig_to_image(fig)


def momentum_vs_sgd(im):
    d = ImageDraw.Draw(im)
    white(d, (160, 790, 815, 875))
    text(d, (168, 812), "đỏ: SGD có momentum", 28, RED, anchor="lm")
    text(d, (168, 852), "đen: SGD thuần", 28, RED, anchor="lm")
    return im


def momentum_update(im):
    d = ImageDraw.Draw(im)
    white(d, (16, 231, 100, 259))
    text(d, (18, 245), "điểm tính gradient", 12, (70, 110, 200), anchor="lm")
    return im


def receptive_field(im):
    im = im.crop((0, 0, im.width, 478))           # bỏ chú thích (a), (b) tiếng Trung
    d = ImageDraw.Draw(im)
    for box, label, xy in [((14, 178, 48, 196), "trước", (14, 187)), ((14, 280, 48, 298), "sau", (14, 289)),
                           ((384, 74, 412, 92), "L", (388, 83)), ((380, 230, 420, 248), "L+1", (382, 239)),
                           ((380, 344, 420, 362), "L+2", (382, 353)), ((380, 434, 420, 452), "L+3", (382, 443))]:
        white(d, box)
        text(d, xy, label, 12, BLACK, anchor="lm")
    return im


def relabel(items, color=BLACK, bold=False, pad=3):
    """Thay các vùng chữ Trung (hộp lấy từ OCR: x0, y0, x1, y1) bằng chữ Việt.

    Nền được tô bằng màu lấy ở ngay bên trái hộp; cỡ chữ tự co cho vừa chiều rộng hộp
    (mở rộng hộp sang phải nếu chữ Việt dài hơn nhiều). items: [(hộp, chữ)] hoặc [(hộp, chữ, màu)].
    """
    def border_colour(src, box):
        # màu nền = màu xuất hiện nhiều nhất trên viền hộp (lấy trên ảnh gốc, trước khi sửa)
        x0, y0, x1, y1 = box
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(src.width - 1, x1), min(src.height - 1, y1)
        pts = [(x, y) for x in range(x0, x1 + 1, 2) for y in (y0, y1)] + [(x, y) for y in range(y0, y1 + 1, 2) for x in (x0, x1)]
        cnt = {}
        for q in pts:
            c = src.getpixel(q); cnt[c] = cnt.get(c, 0) + 1
        return max(cnt, key=cnt.get)

    def fn(im):
        src = im.copy()
        d = ImageDraw.Draw(im)
        for it in items:
            (x0, y0, x1, y1), s = it[0], it[1]
            col = it[2] if len(it) > 2 else color
            h = y1 - y0
            size = max(10, int(h * 0.8))
            room = min((x1 - x0) * 1.15, im.width - x0 - 4)
            while size > 9 and d.textlength(s, font=font(size, bold)) > room:
                size -= 1
            tw = d.textlength(s, font=font(size, bold))
            c = it[3] if len(it) > 3 else border_colour(src, (x0 - pad - 1, y0 - pad - 1, x1 + pad + 1, y1 + pad + 1))
            d.rectangle((x0 - pad, y0 - pad, max(x1, x0 + tw) + pad, y1 + pad), fill=c)
            d.text((x0, (y0 + y1) // 2), s, font=font(size, bold), fill=col, anchor="lm")
        return im
    return fn


def vgg_resnet_table(im):
    # Tiêu đề và khung đỏ chữ Trung (khung còn ghi nhầm ResNet-152 thành "resnet18") → chữ Việt.
    d = ImageDraw.Draw(im)
    white(d, (0, 0, 935, 72)); white(d, (940, 0, im.width, 236))
    text(d, (468, 38), "Cấu hình các mạng VGG", 46, BLACK, True, "mm")
    text(d, (1615, 38), "Cấu hình các mạng ResNet", 46, BLACK, True, "mm")
    for i, s in enumerate(["Số khối ở conv2_x…conv5_x:",
                           "ResNet-18 [2,2,2,2], ResNet-34 [3,4,6,3] (khối basic);",
                           "ResNet-50 [3,4,6,3], ResNet-101 [3,4,23,3], ResNet-152 [3,8,36,3] (khối bottleneck)"]):
        text(d, (980, 98 + 46 * i), s, 30, (30, 60, 200))
    return im


def class_regroup(im):
    im = im.crop((0, 0, im.width, 388))
    d = ImageDraw.Draw(im)
    for box in [(52, 30, 178, 52), (188, 62, 270, 96), (188, 124, 270, 152), (188, 196, 270, 226),
                (30, 268, 290, 295), (282, 336, 345, 375), (340, 336, 395, 358), (398, 336, 480, 358), (548, 336, 638, 358)]:
        white(d, box)
    text(d, (95, 42), "Tên ảnh", 11, BLACK, True, "mm"); text(d, (150, 42), "Lớp", 11, BLACK, True, "mm")
    text(d, (192, 79), "lớp a: 3 ảnh", 12, (40, 90, 200), True, "lm")
    text(d, (192, 138), "lớp b: 2 ảnh", 12, (220, 80, 20), True, "lm")
    text(d, (192, 211), "lớp c: 5 ảnh", 12, (40, 130, 40), True, "lm")
    text(d, (95, 282), "1. Xếp theo lớp", 12, BLACK, True, "mm")
    text(d, (232, 282), "2. Đếm mỗi lớp", 12, BLACK, True, "mm")
    text(d, (318, 346), "3. Hoán", 11, BLACK, True, "mm"); text(d, (318, 363), "vị", 11, BLACK, True, "mm")
    text(d, (367, 346), "4. Lấy dư", 11, BLACK, True, "mm")
    text(d, (440, 346), "5. Tra ra ảnh", 11, BLACK, True, "mm")
    text(d, (636, 346), "6. Trộn ngẫu nhiên", 11, BLACK, True, "rm")
    return im


def train_vs_infer(im):
    d = ImageDraw.Draw(im)
    white(d, (210, 0, 960, 58)); white(d, (210, 292, 1000, im.height))
    text(d, (225, 8), "Cách làm thường gặp: huấn luyện bằng TensorFlow thì lúc suy luận", 19, (120, 120, 120))
    text(d, (225, 34), "cũng chạy thẳng bằng TensorFlow (test trong framework)", 19, (120, 120, 120))
    text(d, (225, 298), "Mô hình đã huấn luyện (trọng số FP32) được đưa vào TensorRT; TensorRT phân tích,", 19, (120, 120, 120))
    text(d, (225, 324), "tối ưu mạng rồi suy luận trực tuyến và trả kết quả", 19, (120, 120, 120))
    for box, lines, col in [((12, 45, 115, 135), ["Huấn luyện", "trong", "framework"], "white"),
                            ((12, 185, 115, 275), ["Huấn luyện", "trong", "framework"], "white"),
                            ((972, 45, 1080, 135), ["Suy luận", "bằng", "framework"], (225, 225, 225)),
                            ((495, 185, 598, 275), ["TensorRT", "tối ưu", "mạng"], "white"),
                            ((975, 185, 1080, 275), ["Suy luận", "bằng", "engine"], "white")]:
        fill(d, im, box, (box[0] + 4, box[1] + 4))
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        for k, ln in enumerate(lines):
            text(d, (cx, cy + (k - 1) * 24), ln, 17, col, True, "mm")
    return im


def trt_workflow(im):
    d = ImageDraw.Draw(im)
    G = (60, 60, 60)
    items = [((200, 40, 360, 105), (205, 72), "Quy trình", 44, (230, 180, 60), "lm"),
             ((100, 140, 330, 172), (108, 156), "Tối ưu mô hình đã huấn luyện", 17, G, "lm"),
             ((295, 236, 385, 266), (340, 251), "Nhập mô hình", 14, G, "mm"),
             ((105, 338, 245, 366), (175, 352), "Mô hình đã huấn luyện", 13, G, "mm"),
             ((548, 352, 600, 378), (550, 365), "Optimizer", 13, G, "lm"),
             ((690, 236, 780, 266), (735, 251), "Tuần tự hoá engine", 13, G, "mm"),
             ((895, 360, 1010, 386), (952, 373), "Đồ thị đã tối ưu", 13, G, "mm"),
             ((100, 422, 340, 452), (108, 437), "Triển khai engine đã tối ưu khi suy luận", 17, G, "lm"),
             ((295, 517, 395, 545), (345, 531), "Giải tuần tự hoá", 13, G, "mm"),
             ((690, 517, 775, 545), (732, 531), "Chạy triển khai", 13, G, "mm"),
             ((85, 643, 200, 672), (142, 657), "Đồ thị đã tối ưu", 13, G, "mm"),
             ((522, 636, 590, 662), (525, 650), "Runtime", 13, G, "lm"),
             ((965, 543, 1067, 570), (1064, 556), "Trung tâm dữ liệu", 11, G, "rm"),
             ((850, 635, 935, 672), (890, 650), "Xe tự lái", 12, G, "mm"),
             ((970, 635, 1067, 672), (1022, 650), "Hệ nhúng", 12, G, "mm")]
    for box, xy, s, size, col, anchor in items:
        white(d, box)
        text(d, xy, s, size, col, anchor=anchor)
    return im


EDITS = {
    # Phần 3: backbone kinh điển (hộp chữ lấy bằng OCR RapidOCR)
    "VoVNet/Figure2-middle-bottom.png": relabel([((260, 144, 603, 170), "So với DenseBlock, ô xanh gần hết,", RED), ((235, 176, 483, 203), "liên kết tốt hơn", RED)]),
    "backbone/3_densenet_structure_drawing_methods.png": relabel([((276, 36, 798, 65), "Cách vẽ DenseBlock trong bài báo DenseNet"), ((227, 683, 992, 716), "Cách vẽ DenseBlock (DenseNet) trong bài báo CSPNet"), ((156, 1074, 707, 1108), "Cách vẽ DenseBlock trong bài báo VoVNet")]),
    "backbone/figure_3_several_different_forms_of_csp.png": relabel([((159, 691, 486, 717), "Hình 3: các dạng CSP khác nhau")]),
    "densenet/densenet_block_structure_diagram.png": relabel([((27, 521, 592, 546), "Mỗi tầng trong khối nhận mọi feature map trước nó làm đầu vào", RED)]),
    "mobilenetv1/3x3_depth_separable_convolution_block_structure.png": relabel([((29, 221, 237, 243), "Khối tích chập tách được"), ((29, 248, 132, 272), "3×3")]),
    "mobilenetv1/dw_convolution_calculation_steps.jpg": relabel([((190, 91, 445, 108), "Feature map vào, kích thước (DF, DF, M)"), ((580, 101, 646, 118), ""), ((712, 103, 727, 115), ""), ((640, 141, 720, 156), "Feature map ra"), ((190, 202, 494, 220), "M bộ lọc, kích thước (DK, DK, 1)")]),
    "mobilenetv1/filter_size.png": relabel([((156, 336, 270, 353), "Bộ lọc DW"), ((253, 531, 366, 548), "Bộ lọc PW")]),
    "mobilenetv1/onnx_model_structure_diagram_of_depth_separable_convolution_block.png": relabel([((141, 8, 206, 30), "Tích chập thường", RED), ((2, 53, 87, 68), "Số kênh ra", RED), ((223, 47, 303, 67), "Cỡ kernel", RED), ((134, 59, 213, 77), "Số kênh vào", RED), ((140, 265, 211, 288), "Tích chập DW:", RED), ((140, 290, 282, 306), "số nhóm = số kênh vào", RED), ((140, 310, 265, 327), "= số kênh ra = 32", RED), ((151, 488, 216, 506), "Tích chập PW:", RED), ((151, 509, 265, 527), "số kênh vào = 32,", RED), ((150, 530, 263, 548), "số kênh ra = 64", RED), ((134, 567, 217, 587), "Số kênh vào", RED)]),
    "mobilenetv1/table_1.png": relabel([((16, 671, 224, 703), "Conv dw / s2 (đúng: s1)", RED)]),
    "resnet/image-20230217211129945.png": relabel([((70, 351, 92, 371), ""), ((105, 351, 456, 374), "Hình 3-4: cấu trúc khối residual [4]")]),
    "resnet/image-20230217211358863.png": relabel([((266, 739, 717, 772), "Hình 3-6: hai kiểu đường shortcut [4]")]),
    "resnet/image-20230217211429369.png": relabel([((68, 324, 342, 349), "khối residual basic"), ((528, 323, 907, 350), "(b) khối residual bottleneck"), ((289, 374, 656, 401), "Hình 3-5: hai kiểu đường residual [4]")]),
    "resnet/image-20230217212628578.png": relabel([((65, 191, 280, 221), "3×3 maxpool, /2"), ((146, 1458, 325, 1491), "Cấu trúc ResNet-18")]),
    "resnet/image-20230217212933666.png": relabel([((357, 3, 669, 30), "Bảng tham số ResNet-18"), ((468, 616, 583, 657), "kernel", RED), ((697, 630, 884, 674), "số bộ lọc", RED, "white")]),
    "resnetv2/Figure5.png": relabel([((95, 5, 172, 34), "Hậu kích hoạt", RED), ((742, 7, 823, 38), "Tiền kích hoạt", RED)]),
    "resnetv2/residual_unit_structure_diagram_of_resnet_and_resnetv2.png": relabel([((305, 437, 521, 462), "Đơn vị residual ResNet v2", RED)]),
    "shufflenetv2/bottleneck.png": relabel([((277, 132, 407, 157), "giảm chiều rồi tăng chiều", RED)]),
    "shufflenetv2/figure_3.png": relabel([((129, 571, 396, 594), "Khối ShuffleNet v1", RED), ((672, 572, 939, 597), "Khối ShuffleNet v2", RED), ((387, 806, 624, 842), "giảm kích thước 2 lần", RED, "white")]),
    "shufflenetv2/figure_4.png": relabel([((244, 513, 1001, 548), "Kiểu tái sử dụng đặc trưng của DenseNet và ShuffleNet v2", RED)]),
    "shufflenetv2/table_2.png": relabel([((207, 8, 366, 41), "Cùng FLOPs", RED)]),
    "shufflenetv2/table_3.png": relabel([((5, 26, 474, 48), "Chỉnh số kênh của 4 khối còn lại để", RED), ((5, 54, 360, 77), "FLOPs bằng khối 1 mảnh.", RED)]),
    "shufflenetv2/table_4.png": relabel([((319, 13, 968, 36), "Theo thí nghiệm, bỏ ReLU có vẻ tăng tốc nhiều hơn bỏ shortcut.", RED), ((323, 447, 1165, 481), "Mỗi biến thể của khối \"bottleneck\" được chồng 10 lần để đo tốc độ", RED, "white")]),
    "shufflenetv2/table_5.png": relabel([((78, 3, 846, 31), "Khối đầu mỗi stage: tích chập 3×3 stride 2, chỉ có 1 khối;", RED, "white"), ((82, 30, 735, 54), "các khối sau: stride 1, được lặp lại (repeat).", RED, "white"), ((163, 573, 800, 597), "Cấu trúc tổng thể ShuffleNet v2 với bốn mức độ phức tạp", RED)]),
    "deploy_summary/training_and_inference.jpg": train_vs_infer,
    "TensorRT/tensorrt_workflow.jpg": trt_workflow,
    "ncnn/cpu_architecture.png": crop_caption(625),
    "hyperparameters/vgg_resnet_parameters.png": vgg_resnet_table,
    "imbalance/class_balance1.png": class_regroup,
    "dl_basic/receptive_field.jpg": receptive_field,
    "optimizers/momentum_algorithm.png": momentum_vs_sgd,
    "optimizers/momentum_algorithm_update.png": momentum_update,
    "bp/network_structure_and_forward_computation_graph.png": network_graph,
    "bp/gd_concept.png": gd_concept,
    "bp/different_lr_loss.png": lr_compare,
    "activation_function/typical_neuron_architecture.png": neuron_arch,
    "activation_function/sigmoid_tanh_curve.png": tanh_center,
    "activation_function/relu_more.png": crop_caption(705),
    "activation_function/activation_function_summary.png": act_summary,
    "bn/normal-distribution-curve-1.png": normal_curve,
    "activation_function/sigmoid_and_gradient_curve2.png": sigmoid_derivative,
    "conv/ful_conv2.png": conv_fc_vs_conv,
    "conv/ful_conv3.png": conv_weight_share,
    "conv/ful_conv_compare.png": conv_local_slide,
    "conv/conv3d.png": conv3d_label,
    "conv/pool_cal2.png": pool_maxpool,
    "conv/cnn_structure.png": cnn_structure,
    "bp/curve_slope.png": crop_caption(285),
    "bp/Directional_Derivative_Visual.png": directional_visual,
    "dl/schematic_diagram_of_sigmoid_function.png": sigmoid_saturate,
    "ml_basc_principle/figure_5_1_an_example_of_linear_regression.png": crop_caption(620),
    "ml_basc_principle/figure_5_2_three_models_fitting_quadratic_functions.png": crop_caption(625),
    "ml_basc_principle/figure_5_3_model_capacity.png": crop_caption(785),
    "ml_basc_principle/figure_5_6.png": crop_caption(785),
    "model_compression/model_pruning_pipeline.png": pruning_pipeline,
    "model_compression/Unstructured_structured_sparsity.png": sparsity_grains,
    "pruning/Unstructured_structured_sparsity.png": sparsity_grains,
    "pruning/activation_sparse.png": activation_sparse,
    "pruning/activation_sparsification_pipeline.png": activation_pipeline,
    "pruning/deep_gradient_sparse.png": gradient_sparse,
    "pruning/irregular_matrix.png": irregular,
    "pruning/pruning_sensitivity.png": sensitivity,
    "pruning/structured_way_compare.png": structured_compare,
    "pruning/weights_important.png": weights_important,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    a = ap.parse_args()
    for rel, fn in EDITS.items():
        edit(a.src, rel, fn)


if __name__ == "__main__":
    main()
