# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare a keyframe-warp run with the baseline frames (simple tonemap; same for both).
python3 evaluate.py <scene> <tag>"""
import sys, json, glob, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from skimage.metrics import structural_similarity as ssim
from common import read_rgb
from scenes import FRAMES, ROOT


def disp(x, exposure=1.0):
    x = np.maximum(x * exposure, 0)
    return np.clip((x / (1 + x)) ** (1 / 2.2), 0, 1)


def flicker(frames):
    """Mean |second temporal difference| of display luminance (x100): jitter that a smooth sequence would not have."""
    L = [0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2] for f in frames]
    return float(np.mean([np.abs(L[i - 1] - 2 * L[i] + L[i + 1]).mean() for i in range(1, len(L) - 1)]) * 100)


which, tag = sys.argv[1], sys.argv[2]
f0, f1 = FRAMES[which]
base = f"{ROOT}/{which}/base"
run = f"{ROOT}/{which}/{tag}"
S = json.load(open(f"{run}/summary.json"))
keyset = set(S["keys"])
rows, B, K = [], [], []
for f in range(f0, f1 + 1):
    b = disp(read_rgb(f"{base}/f_{f:04d}.exr")); k = disp(read_rgb(f"{run}/f_{f:04d}.exr"))
    B.append(b); K.append(k)
    mse = float(((b - k) ** 2).mean())
    rows.append(dict(frame=f, key=f in keyset, psnr=round(10 * np.log10(1 / max(mse, 1e-12)), 2),
                     ssim=round(float(ssim(b, k, channel_axis=2, data_range=1.0)), 4)))
pred = [r for r in rows if not r["key"]]
res = dict(scene=which, tag=tag, psnr_mean=round(np.mean([r["psnr"] for r in pred]), 2),
           psnr_min=round(min(r["psnr"] for r in pred), 2),
           ssim_mean=round(np.mean([r["ssim"] for r in pred]), 4), ssim_min=round(min(r["ssim"] for r in pred), 4),
           flicker_base=round(flicker(B), 3), flicker_kf=round(flicker(K), 3),
           speedup=S["speedup_render"], rendered=S["rendered_pixel_frac"], keys=S["n_keys"])
json.dump(dict(summary=res, frames=rows), open(f"{run}/eval.json", "w"), indent=1)
print("EVAL", json.dumps(res))
