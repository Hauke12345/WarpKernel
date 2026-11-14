# SPDX-License-Identifier: GPL-3.0-or-later
"""Render helpers shared by the baseline and the keyframe-warp renderer."""
import time
import bpy
import numpy as np
import OpenEXR
from scenes import W, H


def render_layer(sc, layer, path, rect=None, crop=False):
    """Render one view layer to a multilayer EXR. rect = (x0, y0, x1, y1) in top-down pixel coords (border render)."""
    for v in sc.view_layers:
        v.use = (v.name == layer)
    r = sc.render
    if rect:
        x0, y0, x1, y1 = rect
        r.use_border = True; r.use_crop_to_border = crop
        r.border_min_x, r.border_max_x = x0 / W, x1 / W
        r.border_min_y, r.border_max_y = 1 - y1 / H, 1 - y0 / H
    else:
        r.use_border = False; r.use_crop_to_border = False
    r.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True, scene=sc.name)
    return time.time() - t


def read_layer(path, layer):
    f = OpenEXR.File(path)
    ch = {}
    for p in f.parts:
        for k, v in p.channels.items():
            if k.startswith(layer + "."):
                ch[k[len(layer) + 1:]] = np.asarray(v.pixels, dtype=np.float32)
    out = {}
    if "Combined" in ch:
        out["rgb"] = ch["Combined"][..., :3]
    if "Position.X" in ch:
        out["pos"] = np.stack([ch["Position.X"], ch["Position.Y"], ch["Position.Z"]], -1)
    if "Object Index.X" in ch:
        out["idx"] = np.rint(ch["Object Index.X"]).astype(np.int32)
    return out


def write_exr(path, rgb, half=True):
    dt = np.float16 if half else np.float32
    chans = {c: np.ascontiguousarray(rgb[..., i].astype(dt)) for i, c in enumerate("RGB")}
    OpenEXR.File({"compression": OpenEXR.NO_COMPRESSION, "type": OpenEXR.scanlineimage}, chans).write(path)


def write_gbuf(path, g):
    chans = {"P.X": g["pos"][..., 0], "P.Y": g["pos"][..., 1], "P.Z": g["pos"][..., 2], "I.X": g["idx"].astype(np.float32)}
    chans = {k: np.ascontiguousarray(v.astype(np.float32)) for k, v in chans.items()}
    OpenEXR.File({"compression": OpenEXR.NO_COMPRESSION, "type": OpenEXR.scanlineimage}, chans).write(path)


def read_gbuf(path):
    f = OpenEXR.File(path)
    c = {k: np.asarray(v.pixels, dtype=np.float32) for k, v in f.parts[0].channels.items()}
    return dict(pos=np.stack([c["P.X"], c["P.Y"], c["P.Z"]], -1), idx=np.rint(c["I.X"]).astype(np.int32))


def read_rgb(path):
    f = OpenEXR.File(path)
    c = f.parts[0].channels
    if "RGB" in c:
        return np.asarray(c["RGB"].pixels, dtype=np.float32)[..., :3]
    return np.stack([np.asarray(c[k].pixels, dtype=np.float32) for k in "RGB"], -1)


def cam_matrix(sc):
    cam = sc.camera
    dg = bpy.context.evaluated_depsgraph_get()
    P = cam.calc_matrix_camera(dg, x=W, y=H) @ cam.matrix_world.inverted()
    return np.array(P, dtype=np.float64)
