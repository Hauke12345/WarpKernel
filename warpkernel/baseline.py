# SPDX-License-Identifier: GPL-3.0-or-later
"""Baseline: render every frame normally (timed). Also stores the cheap G-buffer of every frame (timed separately,
not part of the baseline) so the warper can be developed offline.
python3 baseline.py <scene>"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scenes import load, FRAMES, animated_objects, animated_materials, ROOT
from common import render_layer, read_layer, write_exr, write_gbuf, cam_matrix

which = sys.argv[1]
out = f"{ROOT}/{which}/base"; os.makedirs(out, exist_ok=True)
sc = load(which)
f0, f1 = FRAMES[which]
tmp = f"{out}/_r.exr"
log, cams = [], {}
T0 = time.time()
for f in range(f0, f1 + 1):
    sc.frame_set(f)
    tb = render_layer(sc, "Beauty", tmp)
    write_exr(f"{out}/f_{f:04d}.exr", read_layer(tmp, "Beauty")["rgb"])
    tg = render_layer(sc, "GBuf", tmp)
    write_gbuf(f"{out}/g_{f:04d}.exr", read_layer(tmp, "GBuf"))
    cams[f] = cam_matrix(sc).tolist()
    log.append(dict(frame=f, beauty=round(tb, 2), gbuf=round(tg, 2)))
    print("FRAME", log[-1], flush=True)
json.dump(dict(frames=log, beauty_total=round(sum(l["beauty"] for l in log), 1),
               gbuf_total=round(sum(l["gbuf"] for l in log), 1), wall=round(time.time() - T0, 1)),
          open(f"{out}/timing.json", "w"), indent=1)
json.dump(cams, open(f"{out}/cams.json", "w"))
print("DONE", which)
