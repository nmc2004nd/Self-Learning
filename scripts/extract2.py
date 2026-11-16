"""Cắt tài nguyên thương hiệu từ ảnh mẫu đã phóng 4x (Real-ESRGAN).
python extract2.py <up_dir> <out_dir>
Toạ độ hộp ghi theo ảnh mẫu gốc 1672x941, nhân F khi cắt.
"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage
up_dir, out = sys.argv[1], sys.argv[2]
F = 4
B = Image.open(up_dir + '/B4_6B.png').convert('RGB')
D = Image.open(up_dir + '/D4_6B.png').convert('RGB')
H = Image.open(up_dir + '/H4_6B.png').convert('RGB')

def crop(im, box):
    return im.crop(tuple(v * F for v in box))

def fit(im, w):
    """Thu về chiều rộng w (luôn là thu nhỏ từ ảnh 4x nên vẫn nét)."""
    if im.width <= w: return im
    return im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)

def cutout(im, soft=60, close=2, fill=False):
    """Tách nền sáng: dựng bóng vật (đóng khe nhỏ, lấp lỗ) -> bên trong đặc
    (bụng trắng, trong bong bóng); bên ngoài alpha mềm theo khoảng cách màu (giữ bóng, quầng sáng)."""
    a = np.asarray(im).astype(np.float32)
    border = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3), a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.abs(a - bg).max(axis=2)
    obj = dist > 16
    r = close * F
    st = np.zeros((2 * r + 1, 2 * r + 1), bool); yy, xx = np.ogrid[-r:r + 1, -r:r + 1]; st[yy * yy + xx * xx <= r * r] = True
    padded = np.pad(obj, r)
    solid = ndimage.binary_closing(padded, structure=st)[r:-r, r:-r]
    if fill: solid = ndimage.binary_fill_holes(solid)
    solid = ndimage.binary_erosion(solid, iterations=F // 2)
    # bỏ các vệt mảnh (viền khung bảng) và đốm lẻ
    lab, n = ndimage.label(dist > 4)
    junk = np.zeros_like(obj)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = (lab[sl] == i).sum()
        if (h <= 3 * F and w > 6 * h) or area < 4 * F * F:
            junk[sl] |= lab[sl] == i
    alpha = np.clip((dist - 4) / soft, 0, 1)
    alpha[solid] = 1
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
    alpha[junk & ~solid] = 0
    A = np.maximum(alpha, 1e-3)[..., None]
    rgb = np.clip((a - (1 - A) * bg) / A, 0, 255)
    rgb = np.where(alpha[..., None] > 0.02, rgb, a)
    return Image.fromarray(np.dstack([rgb, alpha * 255]).round().astype(np.uint8), 'RGBA')

def trim(im, pad=6):
    bb = im.getchannel('A').point(lambda v: 255 if v > 8 else 0).getbbox()
    if not bb: return im
    x0, y0, x1, y1 = bb
    return im.crop((max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, im.width), min(y1 + pad, im.height)))

def feather_paste(base, piece, xy, left=0):
    w, h = piece.size
    m = np.ones((h, w), np.float32)
    if left: m[:, :left] *= np.linspace(0, 1, left)[None, :] ** 1.5
    base.paste(piece, xy, Image.fromarray((m * 255).astype(np.uint8)))

def png(im, name, w):
    trim(fit(im, w)).save(f'{out}/{name}.png', optimize=True)

def jpg(im, name, w, q=86):
    fit(im, w).save(f'{out}/{name}.jpg', quality=q, optimize=True, progressive=True)

# ---- Nền trời - biển (Hero Backgrounds) ----
sky = crop(B, (526, 240, 1349, 436))
jpg(sky, 'sky', 2400)

# ---- Phần đầu trang chủ: nền sạch + cảnh cá voi Dashboard bên phải ----
W, Hh = 1672 * F, 412 * F
hero = sky.resize((W, round(W * sky.height / sky.width)), Image.LANCZOS)
canvas = Image.new('RGB', (W, Hh))
canvas.paste(hero.crop((0, hero.height - Hh, W, hero.height)) if hero.height >= Hh else hero.resize((W, Hh)), (0, 0))
reg = np.asarray(crop(D, (745, 0, 1672, 474))).copy()     # vùng cảnh, kể cả thanh menu để xoá
mask = np.zeros(reg.shape[:2], np.uint8)
for x0, y0, x1, y1 in [(1525, 24, 1650, 74), (1385, 24, 1450, 66)]:
    mask[y0 * F:y1 * F, (x0 - 745) * F:(x1 - 745) * F] = 255
reg = cv2.inpaint(reg, mask, 9 * F, cv2.INPAINT_TELEA)
scene = Image.fromarray(reg).crop((0, 62 * F, reg.shape[1], 474 * F))
feather_paste(canvas, scene, (745 * F, 0), left=150 * F)
jpg(canvas, 'hero-home', 3000, q=84)

# ---- Thẻ đầu bài học (Learning Hub) ----
card = crop(H, (622, 153, 1388, 413))
cw, ch = card.size
base = sky.resize((cw, round(cw * sky.height / sky.width)), Image.LANCZOS)
lesson = Image.new('RGB', (cw, ch))
lesson.paste(base.resize((cw, ch)) if base.height < ch else base.crop((0, base.height - ch, cw, base.height)), (0, 0))
feather_paste(lesson, card.crop((430 * F, 0, cw, ch)), (430 * F, 0), left=110 * F)
jpg(lesson, 'hero-lesson', 2000)

# ---- Thẻ Dream Big, sóng chân cột trái ----
jpg(crop(D, (1259, 588, 1634, 890)), 'dream', 1000)
jpg(crop(H, (0, 818, 265, 941)), 'side-wave', 800)

# ---- Logo + Whale Mascot ----
png(cutout(crop(B, (34, 28, 172, 148)), fill=True), 'logo', 400)
for name, box in [('whale-hero', (28, 230, 308, 484)), ('whale-sleep', (316, 238, 480, 374)),
                  ('whale-read', (316, 398, 470, 524)), ('whale-cap', (34, 498, 180, 654)),
                  ('whale-love', (196, 506, 340, 660)), ('whale-bubble', (348, 528, 502, 667))]:
    png(cutout(crop(B, box), fill=True), name, (box[2] - box[0]) * 3)

# ---- Background Assets (4 ô) ----
for i, (n, (x0, x1)) in enumerate(zip(['bg-sky', 'bg-sea', 'bg-deep', 'bg-night'], [(40, 170), (185, 305), (320, 462), (477, 607)])):
    t = crop(B, (x0, 740, x1, 897))
    jpg(t, n, 600)
    jpg(t, 'tile-%d' % (i + 1), 600)

# ---- Decoration Elements ----
for name, box in [('deco-wave', (645, 750, 834, 813)), ('deco-splash', (836, 726, 947, 817)),
                  ('deco-bubbles', (966, 730, 1066, 830)), ('deco-sparkle', (651, 826, 714, 900)),
                  ('deco-gull', (743, 828, 844, 879)), ('deco-stars', (853, 838, 934, 878)),
                  ('deco-ripple', (781, 877, 917, 905)), ('deco-planet', (936, 823, 1079, 905))]:
    png(cutout(crop(B, box)), name, (box[2] - box[0]) * 3)

# ---- Minh hoạ học tập, icon chủ đề ----
for name, box in [('illu-study', (1108, 740, 1272, 862)), ('illu-practice', (1290, 728, 1458, 862)), ('illu-build', (1474, 733, 1642, 862))]:
    png(cutout(crop(B, box)), name, (box[2] - box[0]) * 3)
for name, cx in [('ico-cpu', 159), ('ico-brain', 400), ('ico-robot', 619), ('ico-game', 818)]:
    png(cutout(crop(D, (cx - 24, 491, cx + 24, 539))), name, 192)
print('ok')
