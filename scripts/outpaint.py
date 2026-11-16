"""Mở rộng 4 ô nền (tile-1..4) thành dải ngang 4:1 bằng SD 1.5 inpainting (CPU),
rồi phóng nét bằng Real-ESRGAN. python outpaint.py <brand_dir> <out_dir> [ids] [steps]"""
import sys, os, time, numpy as np, torch, cv2
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

brand, out = sys.argv[1], sys.argv[2]
ids = [int(c) for c in (sys.argv[3] if len(sys.argv) > 3 else '1234')]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 14
STRENGTH = 0.8; WIN = 768; CTX = 128
H, W, STEP = 512, 2048, 256
STYLE = 'anime style background, soft pastel blue palette, clean digital painting, smooth gradients, highly detailed'
PROMPTS = {
    1: 'bright blue sky with big fluffy white cumulus clouds and two small white seagulls, ' + STYLE,
    2: 'blue ocean waves with white foam under a bright blue sky, small white clouds on the horizon, ' + STYLE,
    3: 'underwater ocean scene, sunlight rays from the surface, small blue fish silhouettes, coral and seaweed on the sea floor, ' + STYLE,
    4: 'night sky, deep navy blue, tiny stars, soft purple clouds near the bottom, dreamy, ' + STYLE,
}
NEG = 'black birds, crows, dark birds, text, watermark, logo, signature, people, person, frame, border, letterbox, blurry, lowres, jpeg artifacts, photo, realistic, moon, sun, planet, duplicate, trees, forest, building, city, house, tower, land, grass, mountain, symmetry, mirrored'

torch.set_num_threads(os.cpu_count())
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    'stable-diffusion-v1-5/stable-diffusion-inpainting', variant='fp16', torch_dtype=torch.float32,
    safety_checker=None, requires_safety_checker=False)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

def step(canvas, known, x0, prompt, seed):
    """Vẽ cửa sổ 512x512 bắt đầu ở x0; chỉ vùng chưa biết bị thay."""
    win = canvas.crop((x0, 0, x0 + WIN, H))
    km = known[:, x0:x0 + WIN]
    mask = (~km).astype(np.uint8) * 255
    # lấn 24px vào vùng đã biết để mép nối mềm
    m = Image.fromarray(mask).filter(ImageFilter.MaxFilter(49))
    g = torch.Generator().manual_seed(seed)
    res = pipe(prompt=prompt, negative_prompt=NEG, image=win, mask_image=m, height=H, width=WIN, strength=STRENGTH,
               num_inference_steps=STEPS, guidance_scale=7.0, generator=g).images[0]
    # ghép: chỉ lấy phần mới + dải chuyển 24px có làm mềm
    soft = np.asarray(m.filter(ImageFilter.GaussianBlur(10))).astype(np.float32) / 255
    soft = np.maximum(soft, mask / 255.0)
    a = np.asarray(win).astype(np.float32); b = np.asarray(res).astype(np.float32)
    mix = a * (1 - soft[..., None]) + b * soft[..., None]
    canvas.paste(Image.fromarray(mix.clip(0, 255).astype(np.uint8)), (x0, 0))
    known[:, x0:x0 + WIN] = True

for i in ids:
    t0 = time.time()
    tile = Image.open(f'{brand}/tile-{i}.jpg').convert('RGB')
    tile = tile.crop((8, 8, tile.width - 8, tile.height - 8))
    tw = round(tile.width * H / tile.height)
    tile = tile.resize((tw, H), Image.LANCZOS)
    full = W + 2 * CTX
    # bản nháp: lặp gương ô nền (đêm: xoá trăng ở các bản sao trước)
    src = np.asarray(tile).copy()
    if i == 4:
        lum = src.mean(2); mk = np.zeros(lum.shape, np.uint8)
        mk[:H // 2][lum[:H // 2] > 170] = 255
        mk = cv2.dilate(mk, np.ones((25, 25), np.uint8))
        src = cv2.inpaint(src, mk, 15, cv2.INPAINT_TELEA)
    cx = (full - tw) // 2
    xs = np.arange(full) - cx
    per = 2 * tw; r = np.mod(xs, per); idx = np.where(r < tw, r, per - 1 - r)
    draft = src[:, idx]
    draft = cv2.GaussianBlur(draft, (0, 0), 3)
    canvas = Image.fromarray(draft)
    canvas.paste(tile, (cx, 0))
    known = np.zeros((H, full), bool); known[:, cx:cx + tw] = True
    left, right = cx, cx + tw
    k = 0
    while left > 0 or right < full:
        if right < full:
            x0 = min(right - CTX, full - WIN)
            step(canvas, known, x0, PROMPTS[i], 100 * i + k); right = x0 + WIN; k += 1
        if left > 0:
            x0 = max(left - (WIN - CTX), 0)
            step(canvas, known, x0, PROMPTS[i], 100 * i + k); left = x0; k += 1
        print(f'tile {i}: {left}..{right}/{full}  {time.time() - t0:.0f}s', flush=True)
    # dán lại ô gốc ở giữa (VAE làm mờ nhẹ vùng đã biết)
    m = np.ones((H, tw), np.float32); fe = 40
    m[:, :fe] *= np.linspace(0, 1, fe)[None]; m[:, -fe:] *= np.linspace(1, 0, fe)[None]
    canvas.paste(tile, (cx, 0), Image.fromarray((m * 255).astype(np.uint8)))
    c0 = (full - W) // 2
    canvas.crop((c0, 0, c0 + W, H)).save(f'{out}/pano-{i}.png')
    print(f'tile {i} done {time.time() - t0:.0f}s', flush=True)
