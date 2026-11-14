# SPDX-License-Identifier: GPL-3.0-or-later
"""Check a WarpKernel result against real renders, by numbers AND by eye.

    python3 check.py --run out/my_scene [--frames 37 52 80] [--n 3]

Renders reference frames the normal way (same settings) for a few predicted frames (default: the ones farthest from
any keyframe), then writes into <run>/check/:
    compare_FFFF.png   WarpKernel | every-frame reference | difference x6
    crops_FFFF.png     the 4 worst 64-px areas, enlarged 3x (WarpKernel left, reference right)
    report.json        PSNR / SSIM per frame + reference render time
Look at the crops: seams, ghost outlines, stale reflections and smears can score fine on PSNR."""
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--frames", nargs="*", type=int); ap.add_argument("--n", type=int, default=3)
A = ap.parse_args()
S = json.load(open(os.path.join(A.run, "summary.json")))
assert S.get("blend"), "check.py works on runs made with render.py --blend"
os.environ["KF_W"], os.environ["KF_H"] = str(S["width"]), str(S["height"])
import numpy as np
import bpy
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
from scenes import load_blend
from common import render_layer, read_layer, write_exr

sc, _ = load_blend(S["blend"])
sc.cycles.samples = S["spp"]
r = sc.render; r.resolution_x, r.resolution_y, r.resolution_percentage = S["width"], S["height"], 100
keys = set(S["keys"])
if A.frames:
    frames = A.frames
else:   # predicted frames farthest from any keyframe
    cand = [f for f in range(S["first"], S["last"] + 1) if f not in keys]
    cand.sort(key=lambda f: -min(abs(f - k) for k in keys))
    frames = sorted(cand[:A.n])
cd = os.path.join(A.run, "check"); os.makedirs(cd, exist_ok=True)
tmp = os.path.join(cd, "_r.exr")


def display(path_exr):
    """Apply the scene's colour management (view transform, look, exposure) -> 8-bit RGB array.
    The scene's output settings are restored afterwards: renders must keep writing multilayer EXR."""
    ims = sc.render.image_settings
    keep = {k: getattr(ims, k) for k in ("media_type", "file_format", "color_depth", "color_mode", "exr_codec") if hasattr(ims, k)}
    im = bpy.data.images.load(path_exr); im.colorspace_settings.name = "Linear Rec.709"
    if hasattr(ims, "media_type"):
        ims.media_type = "IMAGE"
    ims.file_format = "PNG"; ims.color_depth = "8"; ims.color_mode = "RGB"
    p = path_exr[:-4] + "_disp.png"; im.save_render(p, scene=sc); bpy.data.images.remove(im)
    for k in ("media_type", "file_format", "color_depth", "color_mode", "exr_codec"):
        if k in keep:
            try:
                setattr(ims, k, keep[k])
            except (TypeError, AttributeError):
                pass
    if os.path.exists(tmp):
        os.remove(tmp)            # never read a stale render
    return np.asarray(Image.open(p).convert("RGB"))


rep = []
for f in frames:
    sc.frame_set(f)
    t = render_layer(sc, "Beauty", tmp)
    ref_exr = os.path.join(cd, f"ref_{f:04d}.exr"); write_exr(ref_exr, read_layer(tmp, "Beauty")["rgb"])
    ref = display(ref_exr)
    ours = display(os.path.join(A.run, f"f_{f:04d}.exr")) if os.path.exists(os.path.join(A.run, f"f_{f:04d}.exr")) else None
    a, b = ours.astype(np.float32) / 255, ref.astype(np.float32) / 255
    mse = float(((a - b) ** 2).mean())
    row = dict(frame=f, psnr=round(10 * np.log10(1 / max(mse, 1e-12)), 2), ssim=round(float(ssim(a, b, channel_axis=2, data_range=1)), 4),
               ref_render_s=round(t, 1))
    diff = (np.abs(a - b).max(-1) * 6 * 255).clip(0, 255).astype(np.uint8)
    h, w = diff.shape
    strip = Image.new("RGB", (w * 3, h)); strip.paste(Image.fromarray(ours), (0, 0)); strip.paste(Image.fromarray(ref), (w, 0))
    strip.paste(Image.fromarray(np.repeat(diff[..., None], 3, -1)), (2 * w, 0))
    d = ImageDraw.Draw(strip)
    for i, s in enumerate(("WarpKernel", "reference (every frame)", "difference x6")):
        d.rectangle((i * w, 0, i * w + 330, 34), fill=(0, 0, 0)); d.text((i * w + 8, 8), s, fill=(255, 255, 0))
    strip.save(os.path.join(cd, f"compare_{f:04d}.png"))
    # worst areas
    T = 64; err = np.abs(a - b).max(-1)
    k = np.ones((T, T), np.float32) / (T * T)
    import cv2
    m = cv2.filter2D(err, -1, k)
    crops = []
    for _ in range(4):
        y, x = np.unravel_index(np.argmax(m), m.shape)
        y0, x0 = int(np.clip(y - T // 2, 0, h - T)), int(np.clip(x - T // 2, 0, w - T))
        crops.append((x0, y0)); m[max(0, y0 - T):y0 + 2 * T, max(0, x0 - T):x0 + 2 * T] = -1
    sheet = Image.new("RGB", (T * 3 * 2 + 10, (T * 3 + 10) * 4), (30, 30, 30))
    for i, (x0, y0) in enumerate(crops):
        for j, src in enumerate((ours, ref)):
            c = Image.fromarray(src[y0:y0 + T, x0:x0 + T]).resize((T * 3, T * 3), Image.NEAREST)
            sheet.paste(c, (j * (T * 3 + 10), i * (T * 3 + 10)))
        ImageDraw.Draw(sheet).text((4, i * (T * 3 + 10) + 4), f"x{x0} y{y0}", fill=(255, 255, 0))
    sheet.save(os.path.join(cd, f"crops_{f:04d}.png"))
    row["worst_areas"] = crops
    rep.append(row); print("CHECK", json.dumps(row), flush=True)
json.dump(dict(frames=rep, mean_psnr=round(float(np.mean([x["psnr"] for x in rep])), 2),
               mean_ssim=round(float(np.mean([x["ssim"] for x in rep])), 4),
               ref_s_per_frame=round(float(np.mean([x["ref_render_s"] for x in rep])), 1),
               warpkernel_s_per_frame=round(S["wall"] / S["frames"], 1)), open(os.path.join(cd, "report.json"), "w"), indent=1)
print("REPORT", os.path.join(cd, "report.json"))
