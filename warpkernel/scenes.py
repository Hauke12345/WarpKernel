# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene loading and the common Cycles setup.

    sc = load("interior" | "outdoor" | "marble")       # one of the three procedural test scenes
    sc, warnings = load_blend("my_scene.blend")         # any .blend (resolution from KF_W / KF_H, see render.py)

Render setup: Cycles CPU, fixed samples (8 for the test scenes, the file's own count for a .blend) + OIDN
(fast prefilter), no DOF / motion blur / compositor.
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
W = int(os.environ.get("KF_W", 960)); H = int(os.environ.get("KF_H", W * 9 // 16)); SPP = 8
S = W / 960                                  # scale factor relative to the 960x540 development setup
ROOT = os.path.join(os.environ.get("WARPKERNEL_OUT", os.path.join(_REPO, "out")), f"{W}x{H}")
DEFORM_MODS = {"ARMATURE", "CLOTH", "SOFT_BODY", "FLUID", "DYNAMIC_PAINT", "OCEAN", "WAVE", "NODES", "PARTICLE_SYSTEM",
               "EXPLODE", "MESH_CACHE", "MESH_SEQUENCE_CACHE", "SURFACE_DEFORM", "MESH_DEFORM", "LATTICE", "HOOK",
               "SIMPLE_DEFORM", "DISPLACE", "SHRINKWRAP", "WARP", "CAST", "CURVE"}


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
    return prepare(sc, SPP)


def load_blend(path):
    """Open a .blend (or use it if it is already open) and prepare it. Returns (scene, warnings)."""
    if os.path.abspath(bpy.data.filepath or "") != os.path.abspath(path):
        bpy.ops.wm.open_mainfile(filepath=path)
    sc = bpy.context.scene
    warn = []
    if sc.camera is None:
        raise RuntimeError("the scene has no active camera")
    if sc.camera.data.dof.use_dof:
        warn.append("depth of field is switched off (predicted frames cannot reproduce it)")
    if sc.render.use_motion_blur:
        warn.append("motion blur is switched off (predicted frames cannot reproduce it)")
    ng = getattr(sc, "compositing_node_group", None) or (getattr(sc, "node_tree", None) if getattr(sc, "use_nodes", False) else None)
    plain = {"R_LAYERS", "COMPOSITE", "GROUP_OUTPUT", "GROUP_INPUT", "VIEWER", "REROUTE", "FRAME"}
    if sc.render.use_compositing and ng is not None and any(n.type not in plain for n in ng.nodes):
        warn.append("compositor effects (glare, colour correction...) are not applied; output is the plain render with "
                    "the scene's colour management")
    w = sc.world
    if w and w.node_tree and any(n.type == "OUTPUT_WORLD" and n.inputs["Volume"].links for n in w.node_tree.nodes):
        warn.append("world volume (fog) found: it is view-dependent and will mostly be re-rendered, expect a low speed-up")
    if any(m.node_tree and any(n.type == "OUTPUT_MATERIAL" and n.inputs["Volume"].links for n in m.node_tree.nodes)
           for m in bpy.data.materials):
        warn.append("volume materials found: their look is view-dependent, expect artifacts or re-rendering")
    if sc.render.engine != "CYCLES":
        warn.append(f"render engine was {sc.render.engine}, switched to Cycles")
    spp = sc.cycles.samples if sc.render.engine == "CYCLES" else 64
    return prepare(sc, spp), warn


def prepare(sc, spp):
    """Common setup: Cycles CPU, fixed seed and sample count, OIDN, Beauty + GBuf view layers, unique pass indices."""
    for o in list(bpy.data.objects):
        if o.type == "LIGHT_PROBE":
            bpy.data.objects.remove(o)
    sc.render.engine = "CYCLES"
    c = sc.cycles
    c.device = "CPU"; c.samples = spp; c.use_adaptive_sampling = False; c.time_limit = 0.0
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
    vl = bpy.context.view_layer if bpy.context.view_layer in list(sc.view_layers) else sc.view_layers[0]
    for v in list(sc.view_layers):
        if v != vl:
            sc.view_layers.remove(v)
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
    names = [o.name for o in sc.objects if o.type in ("MESH", "CURVE", "CURVES", "FONT", "SURFACE", "META", "CAMERA", "POINTCLOUD")]
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


def deforming_objects(sc, f0, f1, samples=4):
    """Objects whose shape changes over [f0, f1] (armatures, shape keys, simulations, animated modifiers...).
    Their motion is not rigid, so the warp cannot predict them: they are re-rendered in every frame.
    Detected by comparing the evaluated geometry (object space) at a few frames."""
    import numpy as np
    cand = []
    for o in sc.objects:
        if o.type not in ("MESH", "CURVE", "FONT", "SURFACE", "META", "CURVES", "POINTCLOUD"):
            continue
        sk = getattr(o.data, "shape_keys", None)
        if any(m.type in DEFORM_MODS for m in o.modifiers) or (sk and sk.animation_data) or o.particle_systems or \
                (o.data and o.data.animation_data) or o.type in ("META", "CURVES", "POINTCLOUD"):
            cand.append(o.name)
    if not cand:
        return []
    frames = sorted({f0, f1, *[int(f0 + (f1 - f0) * k / (samples - 1)) for k in range(samples)]})

    def sig(name):
        dg = bpy.context.evaluated_depsgraph_get()
        ob = sc.objects[name].evaluated_get(dg)
        try:
            me = ob.to_mesh()
        except RuntimeError:
            return None
        n = len(me.vertices)
        co = np.zeros(n * 3, np.float32)
        if n:
            me.vertices.foreach_get("co", co)
        ob.to_mesh_clear()
        inst = sum(1 for i in dg.object_instances if i.is_instance and i.parent and i.parent.name == name)
        return n, inst, co[::max(1, n // 2000) * 3].round(4).tobytes() if n else b""
    sigs = {n: [] for n in cand}
    for f in frames:
        sc.frame_set(f)
        for n in cand:
            sigs[n].append(sig(n))
    return [n for n, ss in sigs.items() if any(s_ != ss[0] for s_ in ss[1:])]
