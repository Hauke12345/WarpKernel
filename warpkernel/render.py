# SPDX-License-Identifier: GPL-3.0-or-later
"""Keyframe-warp renderer driver.
python3 render.py <scene> sim  [thr] [split]   offline: uses the baseline renders as keyframes / residual pixels,
                                               estimates the render cost from measured timings
python3 render.py <scene> real [thr] [split]   real run, everything rendered and timed

Order (like a hierarchical GOP): render first + last frame as keyframes, then bisect. For an interval (a, b) we
predict its midpoint; if more than `split` of the frame would need re-rendering, the midpoint becomes a keyframe and
both halves are processed, otherwise every frame inside (a, b) is predicted from a and b and only its dirty
rectangles are rendered."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from scenes import load, FRAMES, animated_objects, animated_materials, objects_using, W, H, ROOT
from common import render_layer, read_layer, write_exr, read_rgb, read_gbuf, cam_matrix
from warp import Key, predict, dirty_rects, reuse, RateModel, tm, PAD
import intra

which, mode = sys.argv[1], sys.argv[2]
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
split = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30
tag = sys.argv[5] if len(sys.argv) > 5 else f"{mode}_t{thr}_s{split}"
REAL = mode == "real"
f0, f1 = FRAMES[which]
base = f"{ROOT}/{which}/base"
out = f"{ROOT}/{which}/{tag}"; os.makedirs(out, exist_ok=True)
tmp = f"{out}/_r.exr"

T0 = time.time()
sc = load(which)
moving = animated_objects(sc, f0, f1)
amats = animated_materials(sc, f0, f1)
idx_of = {o.name: o.pass_index for o in sc.objects}
mov_idx = {idx_of[n]: ms for n, ms in moving.items() if n != sc.camera.name}
def glossy_objects(rough_max=0.3, coat_min=0.3, metal_min=0.5):
    """Scene knowledge: objects whose shading depends on the view (low roughness, metal, glass, transmission).
    glossy_objects(0.15, 2.0, 2.0) -> only mirror-like surfaces."""
    out = []
    for o in sc.objects:
        if o.type not in ("MESH", "CURVE", "FONT"):
            continue
        for sl in o.material_slots:
            m = sl.material
            if not m or not m.node_tree:
                continue
            for n in m.node_tree.nodes:
                if n.type == "BSDF_GLASS" or n.type == "BSDF_GLOSSY" or n.type == "BSDF_REFRACTION":
                    out.append(o.pass_index); break
                if n.type == "BSDF_PRINCIPLED":
                    g = lambda k: n.inputs[k].default_value if k in n.inputs else 0.0
                    if g("Roughness") < rough_max or g("Metallic") > metal_min or g("Transmission Weight") > 0.1 or g("Coat Weight") > coat_min:
                        out.append(o.pass_index); break
    return np.array(sorted(set(out)))


GLOSSY = glossy_objects()
MIRROR = np.array([i for i in glossy_objects(0.22, 2.0, 2.0) if i not in mov_idx])   # static mirrors only
amat_objs = {m: np.array([idx_of[o.name] for o in objects_using(m)]) for m in amats}
if REAL:
    sc.frame_set(f0)
    render_layer(sc, "Beauty", tmp, (0, 0, 16, 16), crop=True)
    T_CALL = float(np.median([render_layer(sc, "Beauty", tmp, (16 * i, 0, 16 * i + 16, 16), crop=True) for i in range(1, 4)]))
    T_CALL += 0.03                                # frame change / read / feather bookkeeping per call
else:
    T_CALL = {"interior": 0.045, "outdoor": 0.045, "marble": 0.17}[which]
t_full_est = np.mean([x["beauty"] for x in json.load(open(f"{base}/timing.json"))["frames"]])
os.environ["KF_OVERHEAD_TILES"] = str(T_CALL / t_full_est * (W * H) / 256)
t_setup = time.time() - T0
print(f"mirror objects {len(MIRROR)}"); print(f"setup {t_setup:.1f}s, call overhead {T_CALL:.3f}s = {os.environ['KF_OVERHEAD_TILES']} tiles, glossy objects {len(GLOSSY)}, moving objects {len(mov_idx)}, animated materials {list(amats)}", flush=True)

timing = json.load(open(f"{base}/timing.json"))
T_FULL = np.mean([x["beauty"] for x in timing["frames"]])
T_G = np.mean([x["gbuf"] for x in timing["frames"]])
cams_base = {int(k): np.array(v) for k, v in json.load(open(f"{base}/cams.json")).items()}
cost = dict(key=0.0, gbuf=0.0, residual=0.0, warp=0.0)
calls = dict(key=0, gbuf=0, residual=0)
res_pixels = 0


def mats_at(f):
    return {i: np.array(ms[f - f0], dtype=np.float64) for i, ms in mov_idx.items()}


def get_gbuf(f):
    if REAL:
        sc.frame_set(f)
        cost["gbuf"] += render_layer(sc, "GBuf", tmp); calls["gbuf"] += 1
        return read_layer(tmp, "GBuf"), cam_matrix(sc)
    cost["gbuf"] += T_G; calls["gbuf"] += 1
    return read_gbuf(f"{base}/g_{f:04d}.exr"), cams_base[f]


def make_key(f, g=None, P=None):
    if g is None:
        g, P = get_gbuf(f)
    if INTRA:
        rgb = intra_key(f, g)
    elif REAL:
        sc.frame_set(f)
        sc.cycles.use_light_tree = True
        cost["key"] += render_layer(sc, "Beauty", tmp)
        rgb = read_layer(tmp, "Beauty")["rgb"]
    else:
        cost["key"] += T_FULL
        rgb = read_rgb(f"{base}/f_{f:04d}.exr")
    calls["key"] += 1
    write_exr(f"{out}/f_{f:04d}.exr", rgb)
    return Key(f, rgb, g, P, mats_at(f))


INTRA = os.environ.get("KF_INTRA") == "1"
INTRA_S = 4
key_full_frac = []


def intra_key(f, g):
    """Keyframe with intra-frame adaptive resolution: whole frame at quarter resolution, then full-resolution
    patches only for tiles with object edges, strong lighting gradients, mirrors or emitters."""
    s = INTRA_S
    if REAL:
        sc.frame_set(f)
        sc.cycles.use_light_tree = True
        sc.render.resolution_percentage = 100 // s
        cost["key"] += render_layer(sc, "Beauty", tmp)
        sc.render.resolution_percentage = 100
        L = read_layer(tmp, "Beauty"); lo, lo_idx = L["rgb"], L["idx"]
    else:
        full_b = read_rgb(f"{base}/f_{f:04d}.exr")
        lo = cv2.resize(full_b, (W // s, H // s), interpolation=cv2.INTER_AREA)
        lo_idx = g["idx"][s // 2::s, s // 2::s]
        cost["key"] += T_FULL / (s * s) + T_CALL
    T = 8 * W // 960
    tmask, _ = intra.plan(lo, g, s, T, SPECIAL, 0.05)
    need = intra.expand(tmask, T)
    up = intra.guided_up(lo, lo_idx, g["idx"], s)
    KP = int(os.environ.get("KF_KEYPAD", 3)) * W // 960            # thin denoiser border for keyframe patches
    rects, mask = dirty_rects(need, tile=T, min_px=1, max_rects=400, pad_px=KP)
    full = np.zeros_like(up); have = np.zeros((H, W), bool)
    for (x0, y0, x1, y1) in rects:
        px0, py0, px1, py1 = max(x0 - KP, 0), max(y0 - KP, 0), min(x1 + KP, W), min(y1 + KP, H)
        src = render_rect(f, (px0, py0, px1, py1), False, bucket="key")
        calls["key"] += 1
        full[y0:y1, x0:x1] = src[y0:y1, x0:x1]; have[y0:y1, x0:x1] = True
    tw = time.time()
    img = intra.compose(up, full, have, T, g["idx"])                 # level-matched, faded transitions
    cost["warp"] += time.time() - tw
    key_full_frac.append(round(float(mask.mean()), 3))
    return img


def known_dirty(t, a, b, g):
    """Scene knowledge the warp cannot see: animated material values (e.g. flashing bumpers) changing in (a, b)."""
    m = np.zeros((H, W), bool)
    for name, vals in amats.items():
        va, vb, vt = vals[a - f0], vals[b - f0], vals[t - f0]
        if vt != va or vt != vb:
            m |= np.isin(g["idx"], amat_objs[name])
    if m.any():
        m = cv2.dilate(m.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0     # emitted light spills around
    return m


HALF = os.environ.get("KF_HALF") == "1"
CACHE = os.environ.get("KF_CACHE", "1") == "1"
DRIFT = float(os.environ.get("KF_DRIFT", "0.5"))     # allowed accumulated drift, as a fraction of the threshold
MAXAGE = 30
RATES = RateModel()
EMITTERS = np.array([o.pass_index for o in sc.objects if o.type in ('MESH', 'CURVE', 'FONT') and any(sl.material and sl.material.node_tree and any((n.type == 'EMISSION') or (n.type == 'BSDF_PRINCIPLED' and 'Emission Strength' in n.inputs and n.inputs['Emission Strength'].default_value > 0 and tuple(n.inputs['Emission Color'].default_value)[:3] != (0, 0, 0)) for n in sl.material.node_tree.nodes) for sl in o.material_slots)])
import bpy as _bpy
_TEX = {m.name for m in _bpy.data.materials if m.node_tree and any(n.type.startswith('TEX_') for n in m.node_tree.nodes)}
TEXTURED = np.array([o.pass_index for o in sc.objects if any(sl.material and sl.material.name in _TEX for sl in o.material_slots)])
SPECIAL = np.concatenate([MIRROR, EMITTERS, TEXTURED]).astype(np.int64) if len(MIRROR) + len(EMITTERS) + len(TEXTURED) else np.array([], np.int64)


def render_rect(t, rect, half, bucket="residual"):
    """Real: border render of rect (full or 50% resolution). Sim: crop from the baseline (box-downsampled for half)."""
    px0, py0, px1, py1 = rect
    if REAL:
        sc.render.resolution_percentage = 50 if half else 100
        # (tested: patches without the light tree are cheaper per call but visibly noisier in emitter-heavy scenes)
        dt = render_layer(sc, "Beauty", tmp, rect, crop=True)
        sc.render.resolution_percentage = 100
        cost[bucket] += dt
        c = read_layer(tmp, "Beauty")["rgb"]
        s = 2 if half else 1
        full = np.zeros((H // s, W // s, 3), np.float32)
        hh, ww = min(c.shape[0], (py1 - py0) // s), min(c.shape[1], (px1 - px0) // s)
        full[py0 // s:py0 // s + hh, px0 // s:px0 // s + ww] = c[:hh, :ww]
        return full
    f = T_FULL * (px1 - px0) * (py1 - py0) / (W * H)
    cost[bucket] += (f * 0.25 if half else f) + T_CALL
    full = read_rgb(f"{base}/f_{t:04d}.exr")
    return cv2.resize(full, (W // 2, H // 2), interpolation=cv2.INTER_AREA) if half else full


def finish(t, a, b, pred, dirty, g, stats):
    """Render the dirty rectangles of frame t and merge them into the prediction (feathered).
    KF_HALF: only disocclusion holes are rendered at full resolution; the rest is rendered at half resolution and
    used to correct the prediction's low frequencies:  pred + up(render_half - down(pred))."""
    global res_pixels
    hole = stats["_hole"]
    if HALF:
        full_dirty = dirty & (cv2.dilate(hole.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        rects_f, mask_f = dirty_rects(full_dirty)
        rects_h, mask_h = dirty_rects(dirty & ~mask_f)
    else:
        rects_f, mask_f = dirty_rects(dirty)
        rects_h, mask_h = [], np.zeros((H, W), bool)
    img = pred.copy()
    wmask = np.zeros((H, W), np.float32); src_acc = np.zeros_like(pred)
    if os.environ.get("KF_SAVEPRED"):
        write_exr(f"{out}/p_{t:04d}.exr", pred)
    if REAL:
        sc.frame_set(t)
    pred_low = cv2.resize(pred, (W // 2, H // 2), interpolation=cv2.INTER_AREA) if rects_h else None
    for half, rects in ((False, rects_f), (True, rects_h)):
        for (x0, y0, x1, y1) in rects:
            px0, py0, px1, py1 = max(x0 - PAD, 0), max(y0 - PAD, 0), min(x1 + PAD, W), min(y1 + PAD, H)   # denoiser context
            src = render_rect(t, (px0, py0, px1, py1), half)
            calls["residual"] += 1
            if half:
                lo = src[py0 // 2:py1 // 2, px0 // 2:px1 // 2] - pred_low[py0 // 2:py1 // 2, px0 // 2:px1 // 2]
                corr = cv2.resize(lo, (px1 - px0, py1 - py0), interpolation=cv2.INTER_LINEAR)
                region = np.maximum(pred[py0:py1, px0:px1] + corr, 0)
                res_pixels += (x1 - x0) * (y1 - y0) // 4
            else:
                region = src[py0:py1, px0:px1]
                res_pixels += (x1 - x0) * (y1 - y0)
            # feather: weight 1 inside the dirty rect, fading to 0 across the rendered padding -> no visible seams
            yy, xx = np.mgrid[py0:py1, px0:px1]
            dx = np.maximum(np.maximum(x0 - xx, xx - (x1 - 1)), 0); dy = np.maximum(np.maximum(y0 - yy, yy - (y1 - 1)), 0)
            ramp = np.clip(1 - np.maximum(dx, dy) / float(PAD), 0, 1).astype(np.float32)
            ramp[(xx < x0) & (x0 == 0)] = 1; ramp[(yy < y0) & (y0 == 0)] = 1
            if half:
                ramp = np.minimum(ramp, 0.999)     # full-res pixels win where both overlap
            sub = wmask[py0:py1, px0:px1]; upd = ramp > sub
            src_acc[py0:py1, px0:px1][upd] = region[upd]; sub[upd] = ramp[upd]
    img = img * (1 - wmask[..., None]) + src_acc * wmask[..., None]
    write_exr(f"{out}/f_{t:04d}.exr", img)
    cv2.imwrite(f"{out}/m_{t:04d}.png", (mask_f * 255 + (mask_h & ~mask_f) * 128).astype(np.uint8))
    stats.update(frame=t, a=a, b=b, rects=len(rects_f) + len(rects_h), rendered=round(float(mask_f.mean() + 0.25 * (mask_h & ~mask_f).mean()), 3))
    for kk in [x for x in stats if x.startswith("_")]:
        stats.pop(kk)
    return stats, img, wmask >= 0.999


log = []
keys = {}
gcache = {}
keys[f0] = make_key(f0); keys[f1] = make_key(f1)
log += [dict(frame=f0, key=True), dict(frame=f1, key=True)]
stack = [(f0, f1)]
while stack:
    a, b = stack.pop(0)
    if b - a < 2:
        continue
    m = (a + b) // 2
    g, P = get_gbuf(m)
    tw = time.time()
    pred, dirty, st = predict(m, keys[a], keys[b], g, P, mats_at(m), thr, known_dirty(m, a, b, g), GLOSSY, MIRROR)
    cost["warp"] += time.time() - tw
    _, mask = dirty_rects(dirty & ~st["_gforce"])     # glossy pixels are handled by the patch cache anyway
    frac = float(mask.mean())
    st.pop("_hole", None) if False else None
    print(f"interval {a}-{b}: midpoint {m} needs {frac:.1%}", flush=True)
    if frac > split and b - a > 2:
        keys[m] = make_key(m, g, P)
        log.append(dict(frame=m, key=True, mid_dirty=round(frac, 3)))
        stack += [(a, m), (m, b)]
        continue
    # accept interval: every frame in (a, b) predicted from keys a and b, processed in order so freshly
    # path-traced patches can serve the next frames (patch cache)
    cache = []
    gm = (g, P)
    for t in range(a + 1, b):
        gt_, Pt_ = gm if t == m else get_gbuf(t)
        tw = time.time()
        kd = known_dirty(t, a, b, gt_)
        pred, dirty, st = predict(t, keys[a], keys[b], gt_, Pt_, mats_at(t), thr, kd, GLOSSY, MIRROR)
        nre = 0; probe = None
        if CACHE:
            pred, dirty, nre, probe = reuse(t, cache, gt_, Pt_, mats_at(t), pred, dirty, st, b - a, thr * DRIFT, kd, MIRROR, RATES)
        cost["warp"] += time.time() - tw
        st["reused"] = round(nre / (W * H), 4)
        if os.environ.get("KF_DEBUG") and t == int(os.environ["KF_DEBUG"]):
            np.savez(f"{out}/debug_{t}.npz", hole=st["_hole"], dirty=dirty, pred=pred, both=st["_both"], d=st["_d"],
                     cache_frames=np.array([c.f for c in cache]))
        st2, img, rmask = finish(t, a, b, pred, dirty, gt_, st)
        log.append(st2)
        if CACHE and probe is not None:
            pa = probe[1] > 0
            pm = rmask & pa
            if pm.sum() > 64:      # free probe: how wrong would the cache have been on pixels we rendered anyway?
                err = np.abs(tm(img[pm]) - tm(probe[0][pm])).max(-1)
                ys, xs = np.nonzero(pm)
                for age in np.unique(probe[1][pm]):
                    sel = probe[1][pm] == age
                    RATES.update(ys[sel], xs[sel], err[sel], int(age))
        if CACHE and rmask.any():
            ck = Key(t, img, gt_, Pt_, mats_at(t))
            ck.trust = ck.trust * rmask
            cache = ([c for c in cache if t - c.f < MAXAGE] + [ck])[-8:]
    print(f"  accepted {a}-{b}", flush=True)

wall = time.time() - T0
render_cost = cost["key"] + cost["gbuf"] + cost["residual"]
summary = dict(scene=which, mode=mode, thr=thr, split=split, frames=f1 - f0 + 1,
               keys=sorted(keys), n_keys=len(keys), calls=calls,
               cost={k: round(v, 1) for k, v in cost.items()}, render_cost=round(render_cost, 1),
               wall=round(wall, 1), setup=round(t_setup, 1),
               baseline=timing["beauty_total"], speedup_render=round(timing["beauty_total"] / render_cost, 2),
               key_full_frac=key_full_frac,
               rendered_pixel_frac=round((len(keys) * W * H + res_pixels) / ((f1 - f0 + 1) * W * H), 3),
               log=sorted(log, key=lambda x: x["frame"]))
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "log"}), flush=True)
