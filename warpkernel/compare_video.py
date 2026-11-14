# SPDX-License-Identifier: GPL-3.0-or-later
"""1920x1080 results video. Per scene:
  1. full screen: keyframe-warp result, with a small timing badge
  2. full screen split: left half every-frame render | right half keyframe warp (same pixels, divider)
  3. analysis grid: every frame | keyframe warp | what was path-traced | difference x6
python3 video4.py <run tag per scene as scene=tag ...>   e.g.  interior=real outdoor=real marble=real"""
import sys, os, json, subprocess
os.environ.setdefault("KF_W", "1920")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scenes import FRAMES, ROOT

RUN = dict(a.split("=") for a in sys.argv[1:])
LOOK = {"interior": ("AgX - Medium High Contrast", -0.3), "outdoor": ("AgX - Medium High Contrast", -0.2),
        "marble": ("AgX - Punchy", 0.0)}
NAMES = {"interior": "Cosy room", "outdoor": "Village square", "marble": "Marble coaster"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F_BIG, F1, F2 = ImageFont.truetype(FONT, 40), ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 22)
FPS, LOOPS = 30, 2
OUT = f"{ROOT}/kfwarp_results_1080p"

sc = bpy.context.scene
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_depth = "8"
sc.view_settings.view_transform = "AgX"


def tonemap(which, run):
    d = f"{ROOT}/{which}/{run}"; os.makedirs(f"{d}/png", exist_ok=True)
    sc.view_settings.look, sc.view_settings.exposure = LOOK[which]
    f0, f1 = FRAMES[which]
    for f in range(f0, f1 + 1):
        p = f"{d}/png/{f:04d}.png"
        if not os.path.exists(p):
            im = bpy.data.images.load(f"{d}/f_{f:04d}.exr"); im.save_render(p, scene=sc); bpy.data.images.remove(im)


def badge(img, lines, xy=(28, 28)):
    dr = ImageDraw.Draw(img, "RGBA")
    fonts = [F_BIG] + [F2] * (len(lines) - 1)
    w = max(dr.textlength(t, font=f) for t, f in zip(lines, fonts)) + 40
    h = 20 + sum(f.size + 10 for f in fonts)
    dr.rounded_rectangle((xy[0], xy[1], xy[0] + w, xy[1] + h), 14, fill=(0, 0, 0, 165))
    y = xy[1] + 12
    for t, f in zip(lines, fonts):
        dr.text((xy[0] + 20, y), t, font=f, fill=(255, 255, 255) if f is F_BIG else (215, 215, 215)); y += f.size + 10
    return img


def label(img, txt, sub=None):
    dr = ImageDraw.Draw(img)
    dr.rectangle((0, 0, img.width, 44 if not sub else 74), fill=(0, 0, 0))
    dr.text((14, 6), txt, font=F1, fill=(255, 255, 255))
    if sub:
        dr.text((14, 44), sub, font=F2, fill=(200, 200, 200))
    return img


def enc(path):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                             "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)


parts = []
for which in ("interior", "outdoor", "marble"):
    if which not in RUN:
        continue
    run = RUN[which]
    tonemap(which, "base"); tonemap(which, run)
    S = json.load(open(f"{ROOT}/{which}/{run}/summary.json"))
    E = json.load(open(f"{ROOT}/{which}/{run}/eval.json"))["summary"]
    tb = json.load(open(f"{ROOT}/{which}/base/timing.json"))["beauty_total"]
    keys = set(S["keys"]); f0, f1 = FRAMES[which]; n = f1 - f0 + 1
    sp = tb / S["wall"]
    for seg in ("full", "split", "grid"):
        path = f"{ROOT}/_seg_{which}_{seg}.mp4"
        ff = enc(path + ".tmp.mp4")
        for rep in range(LOOPS):
            for f in range(f0, f1 + 1):
                b = Image.open(f"{ROOT}/{which}/base/png/{f:04d}.png").convert("RGB")
                k = Image.open(f"{ROOT}/{which}/{run}/png/{f:04d}.png").convert("RGB")
                if seg == "full":
                    fr = badge(k.copy(), [f"{NAMES[which]}", f"Keyframe warp · {S['wall'] / n:.1f} s/frame · {sp:.1f}x faster",
                                          f"every frame rendered: {tb / n:.1f} s/frame · 1920x1080, CPU only"])
                elif seg == "split":
                    fr = k.copy(); fr.paste(b.crop((0, 0, 960, 1080)), (0, 0))
                    dr = ImageDraw.Draw(fr); dr.rectangle((958, 0, 961, 1080), fill=(255, 255, 255))
                    badge(fr, [f"Every frame", f"{tb / n:.1f} s/frame"], (28, 28))
                    badge(fr, [f"Keyframe warp", f"{S['wall'] / n:.1f} s/frame"], (990, 28))
                else:
                    ka = np.asarray(k).astype(np.float32)
                    if f in keys:
                        m = np.ones(ka.shape[:2], bool)
                    else:
                        m = np.asarray(Image.open(f"{ROOT}/{which}/{run}/m_{f:04d}.png")) > 127
                    ov = ka * 0.45; ov[m] = ka[m] * 0.5 + np.array([255, 60, 40]) * 0.5
                    ov = Image.fromarray(ov.clip(0, 255).astype(np.uint8))
                    diff = np.abs(ka - np.asarray(b).astype(np.float32)).max(-1) * 6
                    diff = Image.fromarray(np.repeat(diff.clip(0, 255).astype(np.uint8)[..., None], 3, -1))
                    h = lambda im: im.resize((960, 540), Image.LANCZOS)
                    fr = Image.new("RGB", (1920, 1080))
                    fr.paste(label(h(b), f"Every frame rendered · {tb / n:.1f} s/frame"), (0, 0))
                    fr.paste(label(h(k), f"Keyframe warp · {S['wall'] / n:.1f} s/frame ({sp:.1f}x)"), (960, 0))
                    what = "KEYFRAME" if f in keys else f"path-traced pixels: {m.mean():.0%}"
                    fr.paste(label(h(ov), "What was path-traced (red)", f"{NAMES[which]} · frame {f} · {what}"), (0, 540))
                    fr.paste(label(h(diff), "Difference x6", f"PSNR {E['psnr_mean']:.1f} dB · SSIM {E['ssim_mean']:.3f}"), (960, 540))
                ff.stdin.write(fr.tobytes())
        ff.stdin.close(); ff.wait()
        assert ff.returncode == 0
        os.replace(path + ".tmp.mp4", path); parts.append(path)
lst = f"{ROOT}/_segs.txt"
open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", OUT + ".tmp.mp4"], check=True)
os.replace(OUT + ".tmp.mp4", OUT + ".mp4")
print("VIDEO", OUT + ".mp4")
