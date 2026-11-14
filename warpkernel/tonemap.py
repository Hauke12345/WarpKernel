# SPDX-License-Identifier: GPL-3.0-or-later
"""AgX-tonemap EXRs to PNG: python3 tm.py <scene> exr1 [exr2 ...]"""
import sys, bpy
LOOK = {"interior": ("AgX - Medium High Contrast", -0.3), "outdoor": ("AgX - Medium High Contrast", -0.2), "marble": ("AgX - Punchy", 0.0)}
sc = bpy.context.scene; sc.render.image_settings.file_format = "PNG"; sc.view_settings.view_transform = "AgX"
sc.view_settings.look, sc.view_settings.exposure = LOOK[sys.argv[1]]
for p in sys.argv[2:]:
    im = bpy.data.images.load(p); im.save_render(p.replace(".exr", ".png"), scene=sc); bpy.data.images.remove(im)
