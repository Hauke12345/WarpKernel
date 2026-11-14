# SPDX-License-Identifier: GPL-3.0-or-later
"""Load one of the three test scenes with its reference lighting and a common Cycles setup for the
keyframe-warp experiment.

    sc = load("interior" | "outdoor" | "marble")

Render setup: 960x540, Cycles CPU, 8 spp fixed + OIDN (fast prefilter), no DOF / motion blur / compositor.
Two view layers:
  Beauty  combined (denoised) + object index        -> what the viewer sees
  GBuf    1 spp, no denoise, black emission override: position + object index only (cheap geometry pass)
Every object gets a unique pass_index so the warper knows which matrix moves each pixel."""
import math
import os
import bpy, addon_utils
from mathutils import Vector

addon_utils.enable("cycles")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = {"interior": os.path.join(_REPO, "scenes", "cosy_room.py"), "outdoor": os.path.join(_REPO, "scenes", "village_square.py"),
       "marble": os.path.join(_REPO, "scenes", "marble_coaster.py")}
FRAMES = {"interior": (1, 60), "outdoor": (1, 60), "marble": (20, 79)}
W = int(os.environ.get("KF_W", 960)); H = W * 9 // 16; SPP = 8
S = W / 960                                  # scale factor relative to the 960x540 development setup
ROOT = os.path.join(os.environ.get("WARPKERNEL_OUT", os.path.join(_REPO, "out")), f"{W}x{H}")


def _aim(o, t):
    o.rotation_euler = (Vector(t) - o.location).to_track_quat("-Z", "Y").to_euler()


def _light(sc, name, kind, loc, energy, color, size=0.5, target=None):
    d = bpy.data.lights.new(name, kind)
    d.energy, d.color = energy, color
    if kind == "AREA":
        d.shape = "DISK"; d.size = size
    elif kind in ("POINT", "SPOT"):
        d.shadow_soft_size = size
    o = bpy.data.objects.new(name, d); o.location = loc
    sc.collection.objects.link(o)
    if target:
        _aim(o, target)
    return o


def _lighting(which, sc):
    """Same lighting as the private reference renders of the benchmark."""
    if which == "interior":
        bpy.data.materials["DuskBackdrop"].node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 2.0
        _light(sc, "WindowSky", "AREA", (1.3, 3.2, 1.6), 90, (0.45, 0.58, 1.0), 2.2, target=(1.0, 0.0, 0.8))
        _light(sc, "LampLight", "POINT", (1.4, 1.9, 1.45), 140, (1.0, 0.62, 0.32), 0.08)
        _light(sc, "CeilingFill", "AREA", (0.0, 0.0, 2.7), 8, (1.0, 0.75, 0.5), 2.5, target=(0, 0, 0))
        bg = sc.world.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0.01, 0.012, 0.03, 1)
        sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Medium High Contrast"; sc.view_settings.exposure = -0.3
    elif which == "outdoor":
        sun = bpy.data.lights.new("Sun", "SUN"); sun.energy = 2.6; sun.color = (1.0, 0.6, 0.32); sun.angle = math.radians(1.5)
        so = bpy.data.objects.new("Sun", sun); sc.collection.objects.link(so)
        el, az = math.radians(32), math.radians(-125)
        so.location = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))) * 50
        _aim(so, (0, 0, 0))
        nt = sc.world.node_tree
        sky = nt.nodes.new("ShaderNodeTexSky"); sky.sky_type = "MULTIPLE_SCATTERING"
        for k, v in (("sun_disc", False), ("sun_elevation", el), ("sun_rotation", az - math.pi / 2), ("air_density", 1.5)):
            try:
                setattr(sky, k, v)
            except Exception:
                pass
        bg = nt.nodes["Background"]; bg.inputs[1].default_value = 0.09
        nt.links.new(sky.outputs[0], bg.inputs[0])
        sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Medium High Contrast"; sc.view_settings.exposure = -0.2
    else:   # marble carries its own lights; drop the world haze, keep the look
        w = sc.world
        out = next((n for n in w.node_tree.nodes if n.type == "OUTPUT_WORLD"), None)
        if out and out.inputs["Volume"].links:
            w.node_tree.links.remove(out.inputs["Volume"].links[0])
        for n in ("Bulb", "LEDStrip", "TextWarm", "TextBlue", "TextWhite", "NeonRed"):
            m = bpy.data.materials.get(n)
            if m:
                m.cycles.emission_sampling = "NONE"


def load(which):
    exec(open(SRC[which]).read(), {"__name__": "__kfwarp__"})
    sc = bpy.context.scene
    _lighting(which, sc)
    for o in list(bpy.data.objects):
        if o.type == "LIGHT_PROBE":
            bpy.data.objects.remove(o)
    sc.render.engine = "CYCLES"
    c = sc.cycles
    c.device = "CPU"; c.samples = SPP; c.use_adaptive_sampling = False; c.time_limit = 0.0
    c.use_denoising = True; c.denoiser = "OPENIMAGEDENOISE"
    c.denoising_input_passes = "RGB_ALBEDO_NORMAL"; c.denoising_prefilter = "FAST"
    c.use_animated_seed = False; c.seed = 0
    c.use_light_tree = True
    c.caustics_reflective = c.caustics_refractive = False
    r = sc.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
    r.use_motion_blur = False; r.use_compositing = False; r.use_persistent_data = True
    r.use_border = False; r.use_crop_to_border = False
    sc.camera.data.dof.use_dof = False
    im = r.image_settings
    if hasattr(im, "media_type"):
        im.media_type = "MULTI_LAYER_IMAGE"
    im.file_format = "OPEN_EXR_MULTILAYER"; im.color_depth = "32"; im.exr_codec = "NONE"

    for i, o in enumerate(sorted(bpy.data.objects, key=lambda o: o.name)):
        o.pass_index = i + 1
    vl = sc.view_layers[0]
    vl.name = "Beauty"
    vl.use_pass_combined = True; vl.use_pass_object_index = True; vl.use_pass_z = False
    g = sc.view_layers.new("GBuf")
    g.use_pass_combined = False; g.use_pass_position = True; g.use_pass_object_index = True; g.use_pass_z = False
    g.samples = 1
    g.cycles.use_denoising = False
    blk = bpy.data.materials.new("GBufBlack"); blk.use_nodes = True
    nt = blk.node_tree; nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission"); e.inputs[1].default_value = 0.0
    o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(e.outputs[0], o.inputs[0])
    g.material_override = blk
    g.use_sky = False
    return sc


def animated_objects(sc, f0, f1):
    """Objects whose world matrix changes over [f0, f1] -> {name: [matrix per frame]} (camera included)."""
    names = [o.name for o in sc.objects if o.type in ("MESH", "CURVE", "FONT", "SURFACE", "META", "CAMERA")]
    mats = {n: [] for n in names}
    for f in range(f0, f1 + 1):
        sc.frame_set(f)
        for n in names:
            mats[n].append(sc.objects[n].matrix_world.copy())
    moving = {}
    for n, ms in mats.items():
        a = ms[0]
        if any(max(abs(a[i][j] - m[i][j]) for i in range(4) for j in range(4)) > 1e-6 for m in ms[1:]):
            moving[n] = ms
    return moving


def animated_materials(sc, f0, f1):
    """Material node values driven by f-curves -> {material name: [value tuple per frame]}."""
    out = {}
    for m in bpy.data.materials:
        nt = m.node_tree
        ad = nt.animation_data if nt else None
        if not ad or not ad.action:
            continue
        fcs = []
        try:
            fcs = list(ad.action.fcurves)
        except AttributeError:   # layered actions (Blender 4.4+)
            for layer in ad.action.layers:
                for strip in layer.strips:
                    for cb in strip.channelbags:
                        fcs += list(cb.fcurves)
        if fcs:
            out[m.name] = [tuple(round(fc.evaluate(f), 4) for fc in fcs) for f in range(f0, f1 + 1)]
    return out


def objects_using(mat_name):
    return [o for o in bpy.data.objects if o.type == "MESH" and any(s.material and s.material.name == mat_name for s in o.material_slots)]
