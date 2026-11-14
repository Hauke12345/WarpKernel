# SPDX-License-Identifier: GPL-3.0-or-later
"""Intra-frame adaptive resolution: quarter-res lighting everywhere + full-res tiles only where needed.

need_full(tile) = object/geometry edge in the full-res geometry pass
               OR strong gradient in the quarter-res render (shadow edges, highlights, texture visible at low res)
               OR a mirror / emitter / textured material (scene knowledge)
Everything else is reconstructed from the quarter-res render, upsampled using only samples of the same object."""
import numpy as np
import cv2
from scenes import W, H


def tiles_of(mask, T):
    th, tw = (H + T - 1) // T, (W + T - 1) // T
    pm = np.zeros((th * T, tw * T), np.float32); pm[:H, :W] = mask
    return pm.reshape(th, T, tw, T).max((1, 3)) > 0


def expand(tmask, T):
    return np.repeat(np.repeat(tmask, T, 0), T, 1)[:H, :W]


def tm(x):
    x = np.maximum(x, 0)
    return (x / (1 + x)) ** (1 / 2.2)


def guided_up(lo, lo_idx, idx, s):
    """Upsample lo (H/s, W/s) to full res. Each full-res pixel only blends low-res samples of its own object
    (object id from the full-res geometry pass), so lighting never bleeds across silhouettes."""
    hl, wl = lo.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    u = (xx + 0.5) / s - 0.5; v = (yy + 0.5) / s - 0.5
    x0, y0 = np.floor(u).astype(np.int32), np.floor(v).astype(np.int32)
    up = np.zeros((H, W, 3), np.float32); ws = np.zeros((H, W), np.float32)
    for dy in (-1, 0, 1, 2):
        for dx in (-1, 0, 1, 2):
            xi, yi = np.clip(x0 + dx, 0, wl - 1), np.clip(y0 + dy, 0, hl - 1)
            w = np.maximum(0, 1.5 - np.abs(u - (x0 + dx))) * np.maximum(0, 1.5 - np.abs(v - (y0 + dy)))
            w = w * (lo_idx[yi, xi] == idx)
            up += lo[yi, xi] * w[..., None]; ws += w
    fallback = cv2.resize(lo, (W, H), interpolation=cv2.INTER_LINEAR)
    ok = ws > 1e-3
    up[ok] /= ws[ok][:, None]; up[~ok] = fallback[~ok]
    return up


def plan(lo, g, s, T, special_idx, grad_thr=0.05, use_edge=True):
    """Tile mask (True = render at full resolution)."""
    idx = g["idx"].astype(np.float32); k = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(idx, k) != cv2.erode(idx, k)
    # depth/normal-free geometry crease test: position discontinuity relative to distance
    lt = tm(lo).mean(-1)
    gx = np.abs(cv2.Sobel(lt, cv2.CV_32F, 1, 0, ksize=3)); gy = np.abs(cv2.Sobel(lt, cv2.CV_32F, 0, 1, ksize=3))
    grad = cv2.resize(np.maximum(gx, gy) / 4.0, (W, H), interpolation=cv2.INTER_NEAREST) > grad_thr
    spec = np.isin(g["idx"], special_idx) if len(special_idx) else np.zeros((H, W), bool)
    need = (edge if use_edge else np.zeros_like(edge)) | grad | spec
    return tiles_of(need, T), dict(edge=tiles_of(edge, T).mean(), grad=tiles_of(grad, T).mean(),
                                                 special=tiles_of(spec, T).mean())


def compose(up, full, have, T, idx, fade=None):
    """Seamless composite of the upsampled low-res image `up` and full-res pixels `full` (valid where `have`).
    1. level match, per object: inside each object the low-res image is shifted towards the full-res level measured
       in that object's patches (smoothed, faded out over ~`fade` px) -> no brightness step, no bleeding across objects
    2. detail fades in over half a tile inside the patches -> no sharpness step"""
    fade = fade or 3 * T
    k = int(fade) | 1
    corr = np.zeros_like(up)
    need = ~have
    near = cv2.dilate(have.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    for i in np.unique(idx[near & need]):
        obj = idx == i
        ys, xs = np.nonzero(obj & near)
        y0, y1 = max(ys.min() - k, 0), min(ys.max() + k + 1, H); x0, x1 = max(xs.min() - k, 0), min(xs.max() + k + 1, W)
        o = obj[y0:y1, x0:x1]; hv = have[y0:y1, x0:x1] & o
        if not hv.any():
            continue
        M = hv.astype(np.float32)
        D = np.where(hv[..., None], full[y0:y1, x0:x1] - up[y0:y1, x0:x1], 0).astype(np.float32)
        num = cv2.GaussianBlur(D, (k, k), fade / 3); den = cv2.GaussianBlur(M, (k, k), fade / 3)
        c = num / np.maximum(den, 1e-4)[..., None] * np.clip(den * 3, 0, 1)[..., None]
        reg = corr[y0:y1, x0:x1]; sel = o & ~hv
        reg[sel] = c[sel]
    base = np.maximum(up + corr, 0)
    dist = cv2.distanceTransform(have.astype(np.uint8), cv2.DIST_L2, 3)
    wd = np.clip(dist / max(T / 2, 1), 0, 1)[..., None]
    return np.where(have[..., None], base + wd * (full - base), base).astype(np.float32)
