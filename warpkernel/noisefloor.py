# SPDX-License-Identifier: GPL-3.0-or-later
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, json
from scenes import load, FRAMES, ROOT
from common import render_layer, read_layer, write_exr
which=sys.argv[1]; f0,f1=FRAMES[which]; f=(f0+f1)//2
sc=load(which); sc.cycles.seed=7; sc.frame_set(f)
tmp=f'{ROOT}/{which}/_nf.exr'
render_layer(sc,'Beauty',tmp); write_exr(f'{ROOT}/{which}/noisefloor_f{f:04d}.exr', read_layer(tmp,'Beauty')['rgb'])
print('done',f)
