# SPDX-License-Identifier: GPL-3.0-or-later
"""Keyframe warping: predict an in-between frame from two rendered keyframes using exact scene geometry.

For the target frame t we render only a cheap geometry pass (1 spp, black material override: world position +
object index, ~0.3 s). Each target pixel's surface point is moved into the keyframe's time with the owning object's
matrix delta (static objects: unchanged), projected into the keyframe camera, and its colour is fetched there,
but only from keyframe pixels that show the SAME object at the SAME point (visibility test against the keyframe's
geometry pass). That gives exact, hole-aware reprojection without motion estimation.

Where both keyframes see the point and agree, we blend them by time. Where they disagree (view-dependent shading,
moving shadows and reflections) or neither sees it (disocclusion), the pixel is "dirty" and gets rendered for real."""
import os
import numpy as np
import cv2
from numba import njit, prange
from scenes import W, H

PAD = max(2, round(8 * W / 960))                    # denoiser context around rendered patches (px)
GLOSSY_SPAN = int(os.environ.get("KF_GLOSSY_SPAN", 6))
YY, XX = np.mgrid[0:H, 0:W].astype(np.float64)


def tm(x):
    x = np.maximum(x, 0)
    return (x / (1 + x)) ** (1 / 2.2)


def project(P, X):
    c = X @ P[:, :3].T + P[:, 3]
    w = c[..., 3]
    ws = np.where(np.abs(w) < 1e-9, 1e-9, w)
    u = (c[..., 0] / ws + 1) * 0.5 * W
    v = (1 - (c[..., 1] / ws + 1) * 0.5) * H
    return u, v, w


def far_points(P):
    """World points far along each pixel's view ray (for background / sky pixels)."""
    Pi = np.linalg.inv(P)
    nx = (XX + 0.5) / W * 2 - 1
    ny = 1 - (YY + 0.5) / H * 2
    c = np.stack([nx, ny, np.full_like(nx, 0.999), np.ones_like(nx)], -1) @ Pi.T
    return c[..., :3] / c[..., 3:4]


def cam_pos(P):
    Pi = np.linalg.inv(P)
    # camera centre = null space of P (homogeneous point mapping to w=0, x=0, y=0)
    _, _, vt = np.linalg.svd(P[[0, 1, 3]])
    c = vt[-1]
    return c[:3] / c[3]


class Key:
    def __init__(self, frame, rgb, g, P, mats):
        self.f, self.rgb, self.pos, self.idx, self.P = frame, rgb.astype(np.float32), g["pos"], g["idx"], P
        self.mats = mats                       # {pass_index: 4x4 world matrix at this frame} for moving objects
        self.cpos = cam_pos(P)
        self.pix = 2.0 / (abs(P[0, 0]) * W)    # angular size of one pixel
        # "pure" pixels: whole 3x3 neighbourhood shows the same object. At silhouettes the 1-spp label may say
        # "tube" while the anti-aliased colour is mostly background -> trust those pixels much less
        fi = self.idx.astype(np.float32)
        k3 = np.ones((3, 3), np.uint8)
        self.trust = np.where(cv2.dilate(fi, k3) == cv2.erode(fi, k3), 1.0, 0.03)


def _flat(key):
    if not hasattr(key, "_fl"):
        key._fl = (np.ascontiguousarray(key.pos.reshape(-1, 3), np.float32), np.ascontiguousarray(key.idx.ravel(), np.int32),
                   np.ascontiguousarray(key.rgb.reshape(-1, 3), np.float32), np.ascontiguousarray(key.trust.ravel(), np.float32))
    return key._fl


def _tflat(gt):
    if "_fpos" not in gt:
        gt["_fpos"] = np.ascontiguousarray(gt["pos"].reshape(-1, 3), np.float32)
        gt["_fidx"] = np.ascontiguousarray(gt["idx"].ravel(), np.int32)
    return gt["_fpos"], gt["_fidx"]


def reproject(key, gt, Pt, mats_t, min_w=0.004, want_w=False, sel=None):
    """Fetch colours for target pixels from keyframe `key` (all pixels, or only the flat indices `sel`).
    Everything (object motion, sky rays, projection, visibility test, bilinear fetch) runs in one compiled kernel.
    Returns rgb, valid[, purity] as (H,W,..) images, or as compact arrays when sel is given."""
    pos, idx = _tflat(gt)
    nobj = int(max(idx.max(), max(mats_t) if mats_t else 0)) + 1
    D = np.zeros((nobj, 4, 4), np.float64); hasD = np.zeros(nobj, np.bool_)
    for i, Mt in mats_t.items():
        if i < nobj:
            D[i] = key.mats[i] @ np.linalg.inv(Mt); hasD[i] = True
    s = np.arange(W * H, dtype=np.int64) if sel is None else np.ascontiguousarray(sel, np.int64)
    rgb, valid, pur = _reproj(s, pos, idx, np.linalg.inv(Pt), key.P, D, hasD, *_flat(key),
                              key.cpos.astype(np.float64), float(key.pix), W, H, float(min_w))
    if sel is not None:
        return (rgb, valid, pur) if want_w else (rgb, valid)
    if want_w:
        return rgb.reshape(H, W, 3), valid.reshape(H, W), pur.reshape(H, W)
    return rgb.reshape(H, W, 3), valid.reshape(H, W)


@njit(parallel=True, cache=True, fastmath=True)
def _reproj(sel, pos, idx, Pti, Pk, D, hasD, kpos, kidx, krgb, ktrust, cpos, pix, W, H, min_w):
    n = sel.size
    out = np.zeros((n, 3), np.float32); valid = np.zeros(n, np.bool_); pur = np.zeros(n, np.float32)
    for q in prange(n):
        p = sel[q]
        y = p // W; x = p - y * W
        it = idx[p]
        if it == 0:      # background / sky: a far point along the target pixel's ray
            nx = (x + 0.5) / W * 2.0 - 1.0; ny = 1.0 - (y + 0.5) / H * 2.0
            c0 = Pti[0, 0] * nx + Pti[0, 1] * ny + Pti[0, 2] * 0.999 + Pti[0, 3]
            c1 = Pti[1, 0] * nx + Pti[1, 1] * ny + Pti[1, 2] * 0.999 + Pti[1, 3]
            c2 = Pti[2, 0] * nx + Pti[2, 1] * ny + Pti[2, 2] * 0.999 + Pti[2, 3]
            c3 = Pti[3, 0] * nx + Pti[3, 1] * ny + Pti[3, 2] * 0.999 + Pti[3, 3]
            X0 = c0 / c3; X1 = c1 / c3; X2 = c2 / c3
        else:
            X0 = pos[p, 0]; X1 = pos[p, 1]; X2 = pos[p, 2]
            if it < hasD.size and hasD[it]:
                a0 = D[it, 0, 0] * X0 + D[it, 0, 1] * X1 + D[it, 0, 2] * X2 + D[it, 0, 3]
                a1 = D[it, 1, 0] * X0 + D[it, 1, 1] * X1 + D[it, 1, 2] * X2 + D[it, 1, 3]
                a2 = D[it, 2, 0] * X0 + D[it, 2, 1] * X1 + D[it, 2, 2] * X2 + D[it, 2, 3]
                X0 = a0; X1 = a1; X2 = a2
        w = Pk[3, 0] * X0 + Pk[3, 1] * X1 + Pk[3, 2] * X2 + Pk[3, 3]
        if w <= 0:
            continue
        su = ((Pk[0, 0] * X0 + Pk[0, 1] * X1 + Pk[0, 2] * X2 + Pk[0, 3]) / w + 1.0) * 0.5 * W - 0.5
        sv = (1.0 - ((Pk[1, 0] * X0 + Pk[1, 1] * X1 + Pk[1, 2] * X2 + Pk[1, 3]) / w + 1.0) * 0.5) * H - 0.5
        d0 = X0 - cpos[0]; d1 = X1 - cpos[1]; d2 = X2 - cpos[2]
        tol = 4.0 * pix * np.sqrt(d0 * d0 + d1 * d1 + d2 * d2) + 2e-3
        tol2 = tol * tol
        fx0 = np.floor(su); fy0 = np.floor(sv)
        fx = su - fx0; fy = sv - fy0
        x0 = int(fx0); y0 = int(fy0)
        ws = 0.0; r = 0.0; g = 0.0; b = 0.0; bs = 0.0
        for t in range(4):
            dx = t & 1; dy = t >> 1
            xi = x0 + dx; yi = y0 + dy
            if xi < 0 or xi >= W or yi < 0 or yi >= H:
                continue
            j = yi * W + xi
            if kidx[j] != it:
                continue
            if it != 0:
                e0 = kpos[j, 0] - X0; e1 = kpos[j, 1] - X1; e2 = kpos[j, 2] - X2
                if e0 * e0 + e1 * e1 + e2 * e2 >= tol2:
                    continue
            wx = fx if dx else 1.0 - fx
            wy = fy if dy else 1.0 - fy
            wt = (wx * wy + 1e-4) * ktrust[j]
            bs += wx * wy + 1e-4
            r += krgb[j, 0] * wt; g += krgb[j, 1] * wt; b += krgb[j, 2] * wt; ws += wt
        pur[q] = ws / bs if bs > 0 else 0.0
        if ws > min_w:
            valid[q] = True
            out[q, 0] = r / ws; out[q, 1] = g / ws; out[q, 2] = b / ws
    return out, valid, pur


def predict(t, ka, kb, gt, Pt, mats_t, thr=0.05, known_dirty=None, glossy_idx=None, mirror_idx=None):
    ca, va, qa = reproject(ka, gt, Pt, mats_t, want_w=True)
    cb, vb, qb = reproject(kb, gt, Pt, mats_t, want_w=True)
    wa = (kb.f - t) / (kb.f - ka.f)
    # seen only through silhouette-contaminated samples: don't trust the colour (refilled from clean pixels of the
    # same object below), but no need to render it
    contam = (va & ~vb & (qa <= 0.3)) | (vb & ~va & (qb <= 0.3))
    both = va & vb
    d = np.abs(tm(ca) - tm(cb)).max(-1)
    # blend by time AND by how clean each keyframe's samples were (silhouette-contaminated samples have ~3% trust)
    ea, eb = wa * np.minimum(qa, 1.0), (1 - wa) * np.minimum(qb, 1.0)
    al = (ea / np.maximum(ea + eb, 1e-9))[..., None]
    out = np.where(both[..., None], al * ca + (1 - al) * cb, np.where(va[..., None], ca, cb))
    # a contaminated side must not make the clean side look "disagreeing"
    both_clean = both & (qa > 0.3) & (qb > 0.3)
    hole = ~(va | vb)
    # silhouette pixels differ between keyframes by anti-aliasing phase alone -> not a shading change
    It = gt["idx"].astype(np.float32)
    edge = (cv2.dilate(It, np.ones((3, 3), np.uint8)) != cv2.erode(It, np.ones((3, 3), np.uint8)))
    disagree = both_clean & (d > thr) & ~edge
    # real changes (shadows, highlights, reflections) are blobs; thin lines are resampling noise
    disagree = cv2.morphologyEx(disagree.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    dirty = hole | disagree
    gforce = np.zeros_like(dirty)
    if mirror_idx is not None and len(mirror_idx) and (kb.f - ka.f) > GLOSSY_SPAN:
        # mirror-like surfaces: two far keyframes agreeing proves nothing (each may reflect something different
        # that happens to match) -> always route them through the patch cache / renderer
        gforce = np.isin(gt["idx"], mirror_idx) & ~hole & ~dirty
        dirty |= gforce
    if glossy_idx is not None and len(glossy_idx):
        # seen by one keyframe only: nothing to compare against. Fine for diffuse surfaces, but reflections and
        # glass change with the viewpoint -> the scene tells us which objects are view-dependent
        one = (va ^ vb) & np.isin(gt["idx"], glossy_idx)
        dirty |= cv2.morphologyEx(one.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    if known_dirty is not None:
        dirty |= known_dirty
    if hole.any() or contam.any():   # small holes: fill from neighbours (big ones get rendered anyway)
        out = fill_holes(out, hole | contam, gt["idx"])
    return out.astype(np.float32), dirty, dict(_hole=hole, _gforce=gforce, _contam=contam, _d=d, _both=both_clean, hole=float(hole.mean()), disagree=float(disagree.mean()),
                                              single=float((va ^ vb).mean()))


def dirty_rects(dirty, tile=int(os.environ.get("KF_TILE", max(4, round(16 * W / 960)))), pad=0, max_rects=int(os.environ.get("KF_MAXR", 32)),
                min_px=int(os.environ.get("KF_MINPX", max(1, round(8 * (W / 960) ** 2)))), pad_px=None):
    """Dirty pixel mask -> tile mask -> a few rectangles (x0, y0, x1, y1) covering it.
    A tile only counts as dirty if it holds >= min_px dirty pixels; isolated pixels are filled from neighbours."""
    m = dirty.astype(np.uint8)
    th, tw = (H + tile - 1) // tile, (W + tile - 1) // tile
    pm = np.zeros((th * tile, tw * tile), np.int32); pm[:H, :W] = m
    tg = pm.reshape(th, tile, tw, tile).sum((1, 3)) >= min_px
    if pad:
        tg = cv2.dilate(tg.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0
    # row runs
    rects = []
    for r in range(th):
        c = 0
        while c < tw:
            if tg[r, c]:
                s = c
                while c < tw and tg[r, c]:
                    c += 1
                rects.append([s, r, c, r + 1])
            else:
                c += 1
    # vertical merge of identical runs
    merged = []
    for rc in rects:
        for m_ in merged:
            if m_[0] == rc[0] and m_[2] == rc[2] and m_[3] == rc[1]:
                m_[3] = rc[3]; break
        else:
            merged.append(rc)
    pt = (PAD if pad_px is None else pad_px) / tile                                                    # denoiser padding per side, in tiles
    area = lambda r: (r[2] - r[0] + 2 * pt) * (r[3] - r[1] + 2 * pt)      # what actually gets path-traced
    union = lambda a, b: [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
    # greedy pairwise merge while it is cheap (render overhead per rect ~ a few tiles) or too many rects
    OVERHEAD = float(os.environ.get("KF_OVERHEAD_TILES", 12))   # cost of one extra render call, in default tiles
    OVERHEAD *= (max(4, round(16 * W / 960)) / tile) ** 2                         # ... converted to this call's tile size
    while len(merged) > 1:
        best = None
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                u = union(merged[i], merged[j])
                extra = area(u) - area(merged[i]) - area(merged[j])
                if best is None or extra < best[0]:
                    best = (extra, i, j, u)
        if best[0] <= OVERHEAD or len(merged) > max_rects:
            _, i, j, u = best
            merged = [m_ for k, m_ in enumerate(merged) if k not in (i, j)] + [u]
        else:
            break
    out = [(r[0] * tile, r[1] * tile, min(r[2] * tile, W), min(r[3] * tile, H)) for r in merged]
    mask = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in out:
        mask[y0:y1, x0:x1] = True
    return out, mask


def fill_holes(img, hole, idx=None):
    """Fill hole pixels from the nearest non-hole pixels of the SAME object (never borrow colour across silhouettes)."""
    ids = np.flatnonzero(hole)
    if ids.size == 0:
        return img
    out = np.ascontiguousarray(img, np.float32).copy()
    I = np.zeros(hole.size, np.int32) if idx is None else np.ascontiguousarray(idx.ravel(), np.int32)
    _fill(out.reshape(-1, 3), ids, hole.ravel(), I, W, H, 40)
    return out


@njit(parallel=True, cache=True)
def _fill(img, ids, hole, idx, W, H, maxr):
    vals = np.zeros((ids.size, 3), np.float32); ok = np.zeros(ids.size, np.bool_)
    for q in prange(ids.size):
        p = ids[q]; y = p // W; x = p - y * W; it = idx[p]
        for r in range(1, maxr + 1):
            s0 = 0.0; s1 = 0.0; s2 = 0.0; c = 0
            for yy in range(max(y - r, 0), min(y + r, H - 1) + 1):
                edge_row = (yy == y - r) or (yy == y + r)
                step = 1 if edge_row else 2 * r
                xx = x - r
                while xx <= x + r:
                    if 0 <= xx < W:
                        j = yy * W + xx
                        if (not hole[j]) and idx[j] == it:
                            s0 += img[j, 0]; s1 += img[j, 1]; s2 += img[j, 2]; c += 1
                    xx += step
            if c > 0:
                vals[q, 0] = s0 / c; vals[q, 1] = s1 / c; vals[q, 2] = s2 / c; ok[q] = True
                break
    for q in range(ids.size):
        if ok[q]:
            img[ids[q], 0] = vals[q, 0]; img[ids[q], 1] = vals[q, 1]; img[ids[q], 2] = vals[q, 2]


class RateModel:
    """Measured change rate (display units per frame) per 32-px screen tile, learned for free: whenever pixels get
    path-traced anyway, compare them with what the patch cache would have predicted."""
    T = 32

    def __init__(self):
        self.r = np.full(((H + 31) // 32, (W + 31) // 32), np.nan, np.float32)

    def update(self, ys, xs, err, age):
        ty, tx = ys // self.T, xs // self.T
        key = ty * self.r.shape[1] + tx
        for k in np.unique(key):
            e = err[key == k]
            if e.size >= 8:
                v = float(np.percentile(e, 95)) / age
                i, j = divmod(int(k), self.r.shape[1])
                old = self.r[i, j]
                self.r[i, j] = v if np.isnan(old) else max(v, 0.85 * old)     # fast attack, slow release

    def lookup(self, shape):
        return np.repeat(np.repeat(self.r, self.T, 0), self.T, 1)[:shape[0], :shape[1]]


def reuse(t, cache, gt, Pt, mats_t, pred, dirty, st, span, thr, known=None, glossy_idx=None, rates=None,
          max_age_other=30, max_age_glossy=6):
    """Patch cache (codec P-frame idea): pixels that were really path-traced a few frames ago are warped from that
    frame and reused instead of rendering them again, as long as the estimated drift stays small.
    Rate per pixel = max(keyframe disagreement / keyframe distance, measured rate of its object).
    Glossy objects with no measured rate yet are not reused. Animated-material regions are never reused.
    Returns pred, dirty, n_reused, probe = (cache prediction, age) for pixels that will be rendered anyway."""
    cand = dirty.copy()
    if known is not None:
        cand &= ~known
    probe_c = np.zeros_like(pred); probe_age = np.zeros(dirty.shape, np.int32)
    if not cand.any() or not cache:
        return pred, dirty, 0, (probe_c, probe_age)
    idx = gt["idx"]
    rate_ab = np.where(st["_both"], st["_d"] / span, np.nan)
    rate_obj = rates.lookup(dirty.shape) if rates is not None else np.full(dirty.shape, np.nan, np.float32)
    glossy = np.isin(idx, glossy_idx) if glossy_idx is not None and len(glossy_idx) else np.zeros_like(dirty)   # mirrors
    rate = np.where(glossy, rate_obj, np.fmax(rate_ab, rate_obj))           # glossy: only the measured rate counts
    rate = np.where(glossy & np.isnan(rate_obj), np.inf, rate)             # mirrors: only once measured
    out = pred.copy(); left = cand.copy(); n = 0
    of, lf, rf, gf = out.reshape(-1, 3), left.ravel(), rate.ravel(), glossy.ravel()
    pcf, paf = probe_c.reshape(-1, 3), probe_age.ravel()
    for ck in sorted(cache, key=lambda k: -k.f):                          # newest first
        age = t - ck.f
        sel = np.flatnonzero(lf & (paf == 0))                              # only pixels still undecided
        if sel.size == 0:
            break
        c, v = reproject(ck, gt, Pt, mats_t, min_w=0.3, sel=sel)            # clean pixels only
        r_ = rf[sel]
        ok = v & np.where(np.isnan(r_), age <= max_age_other, r_ * age <= thr) & ~(gf[sel] & (age > max_age_glossy))
        si = sel[ok]; of[si] = c[ok]; lf[si] = False; n += int(ok.sum())
        pr = v & ~ok
        sp = sel[pr]; pcf[sp] = c[pr]; paf[sp] = age
    new_dirty = (dirty & ~cand) | left
    # leftover holes (thin rims below the tile threshold may never be rendered): refill from same-object pixels,
    # which now include the freshly reused ones
    lh = (st["_hole"] & left) | (st["_contam"] & ~dirty)
    if lh.any():
        out = fill_holes(out, lh, idx)
    return out, new_dirty, n, (probe_c, probe_age)
