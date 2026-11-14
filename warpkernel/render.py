# SPDX-License-Identifier: GPL-3.0-or-later
"""WarpKernel renderer.

Any .blend file:
    python3 render.py --blend scene.blend [--frames 1 120] [--out DIR] [--thr 0.08] [--split 0.3] [--spp N] [--scale 100]
    -> DIR/frames/0001.png ... (scene colour management) + DIR/<name>.mp4 + DIR/summary.json

The three test scenes (benchmark):
    python3 render.py <interior|outdoor|marble> real [thr] [split] [tag]   real run, everything rendered and timed
    python3 render.py <interior|outdoor|marble> sim  [thr] [split] [tag]   offline replay on an existing baseline

Order (like a hierarchical GOP): render first + last frame as keyframes, then bisect. For an interval (a, b) we
predict its midpoint; if more than `split` of the frame would need re-rendering, the midpoint becomes a keyframe and
both halves are processed, otherwise every frame inside (a, b) is predicted from a and b and only its dirty
rectangles are rendered."""
import sys, os, time, json, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

T0 = time.time()
BLEND = "--blend" in sys.argv
if BLEND:
    ap = argparse.ArgumentParser(description="WarpKernel: codec-style frame reuse for Blender Cycles")
    ap.add_argument("--blend", required=True)
    ap.add_argument("--frames", nargs=2, type=int, help="first last (default: the scene's frame range)")
    ap.add_argument("--out", help="output directory (default: out/<blend name>)")
    ap.add_argument("--thr", type=float, default=0.08, help="disagreement threshold (display units, lower = stricter)")
    ap.add_argument("--split", type=float, default=0.30, help="add a keyframe when more than this share of a midpoint is dirty")
    ap.add_argument("--spp", type=int, help="samples per pixel (default: the file's own)")
    ap.add_argument("--scale", type=int, help="resolution percentage (default: the file's own)")
    ap.add_argument("--max-rot", type=float, default=45.0,
                    help="add a keyframe when a visible moving object turns more than this many degrees between keyframes")
    ap.add_argument("--no-video", action="store_true")
    A = ap.parse_args()
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(A.blend))
    _r = bpy.context.scene.render
    _pct = A.scale or _r.resolution_percentage
    os.environ["KF_W"] = str(_r.resolution_x * _pct // 100); os.environ["KF_H"] = str(_r.resolution_y * _pct // 100)

import numpy as np
import cv2
from scenes import load, load_blend, FRAMES, animated_objects, animated_materials, objects_using, deforming_objects, W, H, ROOT
from common import render_layer, read_layer, write_exr, read_rgb, read_gbuf, cam_matrix
from warp import Key, predict, dirty_rects, reuse, RateModel, tm, PAD
import intra

if BLEND:
    which, mode, thr, split = os.path.splitext(os.path.basename(A.blend))[0], "real", A.thr, A.split
    sc, WARN = load_blend(os.path.abspath(A.blend))
    if A.spp:
        sc.cycles.samples = A.spp
    f0, f1 = A.frames if A.frames else (sc.frame_start, sc.frame_end)
    out = os.path.abspath(A.out or os.path.join(os.path.dirname(HERE), "out", which))
    MAX_ROT = A.max_rot or None
    base = os.path.join(out, "_none")
    for w_ in WARN:
        print("WARNING:", w_, flush=True)
else:
    which, mode = sys.argv[1], sys.argv[2]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    split = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30
    WARN = []
    MAX_ROT = float(os.environ.get("KF_MAX_ROT", "0")) or None     # off for the published benchmark runs
    f0, f1 = FRAMES[which]
    base = f"{ROOT}/{which}/base"
    out = f"{ROOT}/{which}/{sys.argv[5] if len(sys.argv) > 5 else f'{mode}_t{thr}_s{split}'}"
    sc = load(which)
REAL = mode == "real"
os.makedirs(out, exist_ok=True)
tmp = f"{out}/_r.exr"
moving = animated_objects(sc, f0, f1)
amats = animated_materials(sc, f0, f1)
idx_of = {o.name: o.pass_index for o in sc.objects}
mov_idx = {idx_of[n]: ms for n, ms in moving.items() if n != sc.camera.name}
DEFORM = np.array([idx_of[n] for n in deforming_objects(sc, f0, f1)], np.int64)
if len(DEFORM):
    print(f"deforming objects (re-rendered every frame): {[o.name for o in sc.objects if o.pass_index in set(DEFORM.tolist())]}", flush=True)
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
HAVE_BASE = os.path.exists(f"{base}/timing.json")


def set_overhead(t_full):
    """Cost of one extra render call, expressed in 16-px tiles of path tracing (drives patch merging)."""
    os.environ["KF_OVERHEAD_TILES"] = str(T_CALL / max(t_full, 1e-3) * (W * H) / 256)


if HAVE_BASE:
    set_overhead(np.mean([x["beauty"] for x in json.load(open(f"{base}/timing.json"))["frames"]]))
t_setup = time.time() - T0
print(f"{W}x{H}, frames {f0}-{f1}, setup {t_setup:.1f}s, call overhead {T_CALL:.3f}s, glossy objects {len(GLOSSY)}, "
      f"mirror objects {len(MIRROR)}, moving objects {len(mov_idx)}, animated materials {list(amats)}", flush=True)

timing = json.load(open(f"{base}/timing.json")) if HAVE_BASE else None
if not REAL:
    assert HAVE_BASE, "sim mode needs a baseline (run baseline.py first)"
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


def turn_angle(i, a, b):
    """Rotation (degrees) of moving object i between frames a and b."""
    Ra, Rb = np.array(mov_idx[i][a - f0])[:3, :3], np.array(mov_idx[i][b - f0])[:3, :3]
    Ra = Ra / np.linalg.norm(Ra, axis=0); Rb = Rb / np.linalg.norm(Rb, axis=0)
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def must_split(a, b):
    """Scene knowledge: if a visible moving object turns a lot between two keyframes, its shading and shadow can change
    and come back (a cube turned 180 degrees looks the same), so the two keyframes may agree by coincidence.
    Force a keyframe in between when any such object (>0.2% of the frame) turns more than MAX_ROT degrees."""
    if not MAX_ROT or b - a < 3:
        return False
    for i in mov_idx:
        area = max(float((keys[a].idx == i).mean()), float((keys[b].idx == i).mean()))
        if area > 0.002 and turn_angle(i, a, b) > MAX_ROT:
            return True
    return False


def known_dirty(t, a, b, g):
    """Scene knowledge the warp cannot see: animated material values (e.g. flashing bumpers) changing in (a, b)."""
    m = np.zeros((H, W), bool)
    if len(DEFORM):
        d = np.isin(g["idx"], DEFORM)
        if d.any():
            m |= cv2.dilate(d.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
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
keys[f0] = make_key(f0)
if not HAVE_BASE:
    set_overhead(cost["key"])            # first keyframe = measured cost of a full frame
keys[f1] = make_key(f1)
log += [dict(frame=f0, key=True), dict(frame=f1, key=True)]
stack = [(f0, f1)]
while stack:
    a, b = stack.pop(0)
    if b - a < 2:
        continue
    m = (a + b) // 2
    g, P = get_gbuf(m)
    if must_split(a, b):
        print(f"interval {a}-{b}: a moving object turns more than {MAX_ROT:g} degrees -> keyframe {m}", flush=True)
        keys[m] = make_key(m, g, P)
        log.append(dict(frame=m, key=True, reason="rotation"))
        stack += [(a, m), (m, b)]
        continue
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
summary = dict(scene=which, mode=mode, thr=thr, split=split, frames=f1 - f0 + 1, first=f0, last=f1,
               width=W, height=H, spp=sc.cycles.samples, blend=os.path.abspath(A.blend) if BLEND else None,
               keys=sorted(keys), n_keys=len(keys), calls=calls,
               cost={k: round(v, 1) for k, v in cost.items()}, render_cost=round(render_cost, 1),
               wall=round(wall, 1), setup=round(t_setup, 1),
               baseline=timing["beauty_total"] if timing else None,
               speedup_render=round(timing["beauty_total"] / render_cost, 2) if timing else None,
               est_every_frame=round(cost["key"] / max(calls["key"], 1) * (f1 - f0 + 1), 1),
               est_speedup=round(cost["key"] / max(calls["key"], 1) * (f1 - f0 + 1) / wall, 2),
               warnings=WARN, deforming=[o.name for o in sc.objects if o.pass_index in set(DEFORM.tolist())],
               key_full_frac=key_full_frac,
               rendered_pixel_frac=round((len(keys) * W * H + res_pixels) / ((f1 - f0 + 1) * W * H), 3),
               log=sorted(log, key=lambda x: x["frame"]))
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "log"}), flush=True)

if BLEND:
    # display-ready frames with the scene's own colour management, plus an mp4
    os.makedirs(f"{out}/frames", exist_ok=True)
    import bpy
    ims = sc.render.image_settings
    if hasattr(ims, "media_type"):
        ims.media_type = "IMAGE"
    ims.file_format = "PNG"; ims.color_depth = "8"; ims.color_mode = "RGB"
    for f in range(f0, f1 + 1):
        im = bpy.data.images.load(f"{out}/f_{f:04d}.exr")
        im.colorspace_settings.name = "Linear Rec.709"
        im.save_render(f"{out}/frames/{f:04d}.png", scene=sc)
        bpy.data.images.remove(im)
    if not A.no_video:
        fps = sc.render.fps / sc.render.fps_base
        mp4 = f"{out}/{which}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-start_number", str(f0),
                        "-i", f"{out}/frames/%04d.png", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
                        "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", mp4 + ".tmp.mp4"], check=True)
        os.replace(mp4 + ".tmp.mp4", mp4)
        print("VIDEO", mp4, flush=True)
    print(f"DONE {f1 - f0 + 1} frames in {wall:.0f} s; rendering every frame would take about "
          f"{summary['est_every_frame']:.0f} s (est. {summary['est_speedup']}x). Check the result with check.py.", flush=True)
