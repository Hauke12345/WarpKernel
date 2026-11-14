# SPDX-License-Identifier: GPL-3.0-or-later
"""Codec redundancy of a frame sequence: raw vs all-intra vs normal inter-frame H.264 (same CRF)."""
import sys, os, subprocess, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import read_rgb
from scenes import FRAMES, W, H, ROOT
def disp(x): x=np.maximum(x,0); return np.clip((x/(1+x))**(1/2.2),0,1)
which, run = sys.argv[1], sys.argv[2] if len(sys.argv)>2 else 'base'
f0,f1=FRAMES[which]; d=f'{ROOT}/{which}/{run}'
raw=b''.join((disp(read_rgb(f'{d}/f_{f:04d}.exr'))*255+0.5).astype(np.uint8).tobytes() for f in range(f0,f1+1))
res={'raw_rgb_MB': len(raw)/1e6}
for crf in (18,):
  for name,extra in (('intra',['-g','1']),('inter',['-g','250','-bf','3'])):
    out=f'{d}/_c_{name}.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','30','-i','-',
                    '-c:v','libx264','-preset','medium','-crf',str(crf),'-pix_fmt','yuv420p',*extra,out],input=raw,check=True)
    res[f'{name}_crf{crf}_KB']=round(os.path.getsize(out)/1e3,1)
res['intra/inter']=round(res['intra_crf18_KB']/res['inter_crf18_KB'],2)
res['raw/inter']=round(res['raw_rgb_MB']*1e3/res['inter_crf18_KB'],1)
print(which, run, json.dumps(res))
