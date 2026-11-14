# SPDX-License-Identifier: GPL-3.0-or-later
"""
MARBLE CHAOS - a procedural marble roller coaster x pinball machine for Blender.

Blender 4.2 or newer (tested on 5.3). Built for EEVEE on an RTX-class GPU.

HOW TO RUN
  1. Open Blender, go to the Scripting tab, open this file, press Run Script.
     (or from a terminal:  blender -P marble_coaster.py )
  2. Select the "IrradianceVolume" object -> Object Data properties -> Bake.
     Takes a few seconds and gives proper bounce light. Re-bake if you move things.
  3. Render > Render Animation (Ctrl+F12). Output goes to //render/ next to the .blend
     (save the .blend first, otherwise it lands next to Blender's temp folder).

Everything below is plain Python - tweak the SETTINGS block and re-run.
Re-running wipes the current scene and rebuilds it.
"""

import bpy, bmesh, math, random
from mathutils import Vector, Matrix, Euler

# ============================== SETTINGS ==============================
RESOLUTION     = (1920, 1080)
RENDER_SAMPLES = 64        # 32 = fast draft, 64 = good, 128 = clean
VIEWPORT_SAMPLES = 16
FPS            = 30
FRAMES         = 360       # 12 s
HAZE           = True      # volumetric haze for light beams (costs ~20-30 %)
MOTION_BLUR    = True
OUTPUT         = "//render/marble_chaos_"
MARBLES        = 16
SEED           = 7
# =====================================================================

random.seed(SEED)
R_MARBLE = 0.06
RAIL_R = 0.008
RAIL_W = 0.045                              # rail half-spacing (lateral)
RAIL_H = math.sqrt((R_MARBLE + RAIL_R) ** 2 - RAIL_W ** 2)   # rail drop below marble centre
DECK_H = 0.078                              # deck below marble centre
DECK_W = 0.07
BASE_TOP = 0.5
FIELD_TOP = 0.56
UP = Vector((0, 0, 1))
BUMPERS = ((-0.8, -0.5), (0.8, -0.5), (0.0, -0.52))
RING_C, RING_R = (0.0, -0.88), 0.16
FLIPPER_PIVOTS = ((-0.44, -1.17), (0.44, -1.17))
# no support rods may land on these (xy centre, radius)
KEEP_OUT = [(b, 0.14) for b in BUMPERS] + [(RING_C, 0.3)] + [(f, 0.38) for f in FLIPPER_PIVOTS]


# ----------------------------------------------------------------------------- utils
def setp(obj, attr, val):
    """Set an attribute if this Blender version has it (keeps the script portable)."""
    try:
        setattr(obj, attr, val)
    except Exception as e:  # noqa
        print(f"[marble] skipped {type(obj).__name__}.{attr}: {e}")


COLL = None


def link(o):
    COLL.objects.link(o)
    return o


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for coll in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.lightprobes, bpy.data.actions, bpy.data.worlds):
        for b in list(coll):
            if b.users == 0:
                coll.remove(b)


# ----------------------------------------------------------------------------- materials
def _nodes(m):
    if not m.node_tree:
        setp(m, "use_nodes", True)
    return m.node_tree


def sock(node, names, val):
    for n in (names if isinstance(names, (list, tuple)) else [names]):
        if n in node.inputs:
            node.inputs[n].default_value = val
            return


def principled(name, color=(0.8, 0.8, 0.8), metal=0.0, rough=0.5, emit=None, strength=0.0, coat=0.0):
    m = bpy.data.materials.new(name)
    nt = _nodes(m)
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    sock(b, "Base Color", (*color, 1))
    sock(b, "Metallic", metal)
    sock(b, "Roughness", rough)
    sock(b, ["Coat Weight", "Clearcoat"], coat)
    if emit:
        sock(b, ["Emission Color", "Emission"], (*emit, 1))
        sock(b, "Emission Strength", strength)
    m.diffuse_color = (*color, 1)
    return m


def emissive(name, color, strength):
    m = bpy.data.materials.new(name)
    nt = _nodes(m)
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    e = nt.nodes.new("ShaderNodeEmission")
    e.name = "Emission"
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], out.inputs["Surface"])
    m.diffuse_color = (*color, 1)
    return m


def wood(name, c1, c2, scale=3.0, rough=0.42, coat=0.35, stretch=(1, 1, 6)):
    m = principled(name, c1, rough=rough, coat=coat)
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = stretch
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "RINGS"
    wave.inputs["Scale"].default_value = scale
    wave.inputs["Distortion"].default_value = 7.0
    wave.inputs["Detail"].default_value = 4.0
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 25.0
    mix = nt.nodes.new("ShaderNodeMath")
    mix.operation = "MULTIPLY_ADD"
    mix.inputs[1].default_value = 0.8
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*c1, 1)
    ramp.color_ramp.elements[1].color = (*c2, 1)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    L = nt.links.new
    L(tc.outputs["Object"], mp.inputs["Vector"])
    L(mp.outputs["Vector"], wave.inputs["Vector"])
    L(mp.outputs["Vector"], noise.inputs["Vector"])
    L(wave.outputs["Fac"], mix.inputs[0])
    L(noise.outputs["Fac"], mix.inputs[2])
    L(mix.outputs[0], ramp.inputs["Fac"])
    L(ramp.outputs["Color"], b.inputs["Base Color"])
    L(wave.outputs["Fac"], bump.inputs["Height"])
    L(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def build_materials():
    M = {}
    M["wood"] = wood("Wood", (0.30, 0.13, 0.05), (0.62, 0.36, 0.16))
    M["wood_light"] = wood("WoodLight", (0.48, 0.25, 0.10), (0.80, 0.52, 0.26), scale=2.0)
    M["wood_dark"] = wood("WoodDark", (0.10, 0.04, 0.02), (0.28, 0.13, 0.06), scale=4, rough=0.3, coat=0.6)
    M["floor"] = wood("Parquet", (0.05, 0.025, 0.015), (0.16, 0.08, 0.04), scale=1.5, rough=0.28,
                      coat=0.4, stretch=(1, 6, 1))
    M["brass"] = principled("Brass", (0.80, 0.53, 0.24), metal=1, rough=0.28)
    M["copper"] = principled("Copper", (0.85, 0.42, 0.25), metal=1, rough=0.3)
    M["steel"] = principled("Steel", (0.86, 0.87, 0.9), metal=1, rough=0.14)
    M["chrome"] = principled("Chrome", (0.95, 0.95, 0.96), metal=1, rough=0.035)
    M["dark"] = principled("DarkMetal", (0.035, 0.035, 0.04), metal=0.8, rough=0.35)
    M["skin"] = principled("DrumSkin", (0.85, 0.78, 0.62), rough=0.6)
    M["curtain"] = principled("Curtain", (0.006, 0.005, 0.01), rough=0.95)
    M["screen"] = principled("Screen", (0.004, 0.006, 0.015), rough=0.12, emit=(0.02, 0.04, 0.12), strength=1.0)
    M["purple"] = emissive("NeonPurple", (0.55, 0.1, 1.0), 10)
    M["blue"] = emissive("NeonBlue", (0.1, 0.38, 1.0), 10)
    M["orange"] = emissive("NeonOrange", (1.0, 0.42, 0.08), 7)
    M["red"] = emissive("NeonRed", (1.0, 0.06, 0.05), 6)
    M["bulb"] = emissive("Bulb", (1.0, 0.6, 0.28), 22)
    M["led"] = emissive("LEDStrip", (1.0, 0.45, 0.15), 10)
    M["text_warm"] = emissive("TextWarm", (1.0, 0.62, 0.25), 14)
    M["text_blue"] = emissive("TextBlue", (0.45, 0.7, 1.0), 10)
    M["text_white"] = emissive("TextWhite", (0.95, 0.92, 1.0), 8)
    M["flipper"] = principled("Flipper", (0.9, 0.85, 0.75), rough=0.25, emit=(1.0, 0.4, 0.1), strength=3)
    M["rubber"] = principled("Rubber", (0.6, 0.05, 0.03), rough=0.5)
    return M


# ----------------------------------------------------------------------------- mesh builder
class MB:
    """Accumulates many primitives into one mesh (one object = fast scene)."""

    def __init__(self):
        self.bm = bmesh.new()

    def _mark(self, ret, smooth):
        faces = {f for v in ret["verts"] for f in v.link_faces}
        for f in faces:
            f.smooth = smooth and len(f.verts) == 4

    def cyl(self, a, b, r, seg=12, r2=None, caps=True):
        a, b = Vector(a), Vector(b)
        d = b - a
        if d.length < 1e-5:
            return
        M = Matrix.Translation((a + b) / 2) @ d.to_track_quat("Z", "Y").to_matrix().to_4x4()
        ret = bmesh.ops.create_cone(self.bm, cap_ends=caps, cap_tris=False, segments=seg,
                                    radius1=r, radius2=r if r2 is None else r2, depth=d.length, matrix=M)
        self._mark(ret, True)

    def box(self, c, size, rot=(0, 0, 0)):
        M = Matrix.Translation(c) @ Euler(rot).to_matrix().to_4x4() @ Matrix.Diagonal((*size, 1))
        ret = bmesh.ops.create_cube(self.bm, size=1.0, matrix=M)
        self._mark(ret, False)

    def sphere(self, c, r, sub=1):
        ret = bmesh.ops.create_icosphere(self.bm, subdivisions=sub, radius=r, matrix=Matrix.Translation(c))
        for f in {f for v in ret["verts"] for f in v.link_faces}:
            f.smooth = True

    def uvsphere(self, c, r, u=16, v=8):
        ret = bmesh.ops.create_uvsphere(self.bm, u_segments=u, v_segments=v, radius=r, matrix=Matrix.Translation(c))
        for f in {f for v in ret["verts"] for f in v.link_faces}:
            f.smooth = True

    def prism(self, poly, z0, z1):
        bot = [self.bm.verts.new((x, y, z0)) for x, y in poly]
        top = [self.bm.verts.new((x, y, z1)) for x, y in poly]
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            self.bm.faces.new((bot[i], bot[j], top[j], top[i]))
        self.bm.faces.new(list(reversed(bot)))
        self.bm.faces.new(top)

    def obj(self, name, mat, loc=(0, 0, 0)):
        me = bpy.data.meshes.new(name)
        self.bm.to_mesh(me)
        self.bm.free()
        o = bpy.data.objects.new(name, me)
        o.location = loc
        if mat:
            me.materials.append(mat)
        return link(o)


def tube(name, pts, r, mat, closed=False, res=2):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = r
    cu.bevel_resolution = res
    cu.use_fill_caps = True
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, v in zip(sp.points, pts):
        p.co = (v[0], v[1], v[2], 1)
    sp.use_cyclic_u = closed
    cu.materials.append(mat)
    return link(bpy.data.objects.new(name, cu))


def rounded_rect(x0, y0, x1, y1, r, seg=8):
    pts = []
    corners = [(x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)]
    for cx, cy, a0 in corners:
        for k in range(seg + 1):
            a = math.radians(a0 + 90 * k / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def text(name, body, size, loc, mat, parent=None, align="CENTER"):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = "CENTER"
    cu.extrude = 0.002
    cu.materials.append(mat)
    o = link(bpy.data.objects.new(name, cu))
    o.location = loc
    o.rotation_euler = (math.radians(90), 0, 0)    # face -Y (towards the camera)
    o.parent = parent
    return o


def empty(name, loc=(0, 0, 0), rot=(0, 0, 0)):
    e = link(bpy.data.objects.new(name, None))
    e.location = loc
    e.rotation_euler = rot
    e.empty_display_size = 0.2
    return e


# ----------------------------------------------------------------------------- animation helpers
def _fcurve(idblock, path, index):
    ad = idblock.animation_data or idblock.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(idblock.name + "Action")
    act = ad.action
    if hasattr(act, "fcurve_ensure_for_datablock"):          # Blender 4.4+
        return act.fcurve_ensure_for_datablock(idblock, path, index=index)
    fc = act.fcurves.find(path, index=index)
    return fc or act.fcurves.new(path, index=index)


def keys(idblock, path, index, frames, values, interp="LINEAR"):
    fc = _fcurve(idblock, path, index)
    fc.keyframe_points.add(len(frames))
    co = []
    for f, v in zip(frames, values):
        co += [f, v]
    fc.keyframe_points.foreach_set("co", co)
    for kp in fc.keyframe_points:
        kp.interpolation = interp
        kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"
    fc.update()


def keys_vec(obj, path, frames, vecs, interp="LINEAR"):
    for i in range(3):
        keys(obj, path, i, frames, [v[i] for v in vecs], interp)


# ----------------------------------------------------------------------------- track geometry
def catmull(P, closed=True, step=0.02):
    """Centripetal Catmull-Rom through control points P=[(x,y,z,tag)], resampled by arc length."""
    pts = [Vector(p[:3]) for p in P]
    tags = [p[3] for p in P]
    n = len(pts)
    dense, dtag = [], []

    def tj(ti, a, b):
        return ti + max((b - a).length, 1e-4) ** 0.5

    segs = n if closed else n - 1
    for i in range(segs):
        p0 = pts[(i - 1) % n] if (closed or i > 0) else pts[0]
        p1, p2 = pts[i], pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if (closed or i + 2 < n) else pts[-1]
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        m = max(4, int((p2 - p1).length / 0.005))
        tag = tags[i] if tags[i] == tags[(i + 1) % n] else ""
        for k in range(m):
            t = t1 + (t2 - t1) * k / m
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            dense.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
            dtag.append(tag)
    # arc-length resample
    out, otag = [dense[0]], [dtag[0]]
    acc = 0.0
    for i in range(1, len(dense) + (1 if closed else 0)):
        a, b = dense[i - 1], dense[i % len(dense)]
        acc += (b - a).length
        if acc >= step:
            out.append(b.copy())
            otag.append(dtag[i % len(dense)])
            acc = 0.0
    if closed and (out[-1] - out[0]).length < step * 0.5:
        out.pop(); otag.pop()
    return out, otag


def frames_for(C, closed=True):
    """Tangent / lateral / track-normal per sample. Lateral prefers horizontal (banking-free),
    falls back to parallel transport where the track is vertical (loops, steep lifts)."""
    n = len(C)
    T = []
    for i in range(n):
        a = C[(i - 1) % n] if (closed or i > 0) else C[i]
        b = C[(i + 1) % n] if (closed or i < n - 1) else C[i]
        T.append((b - a).normalized())
    L = []
    prev = T[0].cross(UP).normalized()
    for i in range(n):
        lt = prev - T[i] * prev.dot(T[i])
        lt = lt.normalized() if lt.length > 1e-6 else prev
        lu = T[i].cross(UP)
        w = lu.length
        if w > 1e-4:
            lu.normalize()
            if lu.dot(lt) < 0:
                lu = -lu
        a = min(1.0, max(0.0, (w - 0.25) / 0.45))
        l = (lt * (1 - a) + lu * a)
        l = (l - T[i] * l.dot(T[i])).normalized()
        L.append(l)
        prev = l
    N = [L[i].cross(T[i]).normalized() for i in range(n)]
    return T, L, N


def build_track(name, P, neon_mat, M, spiral=None, loop=None):
    C, tags = catmull(P)
    T, L, N = frames_for(C)
    n = len(C)

    rail_a = [C[i] + L[i] * RAIL_W - N[i] * RAIL_H for i in range(n)]
    rail_b = [C[i] - L[i] * RAIL_W - N[i] * RAIL_H for i in range(n)]
    tube(f"{name}_RailL", rail_a, RAIL_R, M["brass"], closed=True)
    tube(f"{name}_RailR", rail_b, RAIL_R, M["brass"], closed=True)

    # wooden deck ribbon
    bm = bmesh.new()
    vs = []
    for i in range(n):
        c = C[i] - N[i] * DECK_H
        vs.append((bm.verts.new(c + L[i] * DECK_W), bm.verts.new(c - L[i] * DECK_W)))
    for i in range(n):
        j = (i + 1) % n
        f = bm.faces.new((vs[i][0], vs[i][1], vs[j][1], vs[j][0]))
        f.smooth = True
    me = bpy.data.meshes.new(f"{name}_Deck")
    bm.to_mesh(me)
    bm.free()
    deck = link(bpy.data.objects.new(f"{name}_Deck", me))
    me.materials.append(M["wood"])
    sol = deck.modifiers.new("Thick", "SOLIDIFY")
    sol.thickness = 0.014
    sol.offset = 1.0

    # neon edges under the deck
    # (neon colour per section: neon_mat maps tag -> material)
    col = [neon_mat.get(tags[i], neon_mat[""]) for i in range(n)]
    start = next((i for i in range(n) if col[i] is not col[i - 1]), 0)
    runs, cur = [], [start]
    for k in range(1, n + 1):
        i = (start + k) % n
        cur.append(i)
        if col[i] is not col[cur[0]] or k == n:
            runs.append(cur)
            cur = [i]
    for r_i, run in enumerate(runs):
        mat = col[run[0]]
        for s, side in ((1, "A"), (-1, "B")):
            pts = [C[i] - N[i] * (DECK_H + 0.006) + L[i] * s * (DECK_W + 0.004) for i in run]
            if len(pts) > 1:
                tube(f"{name}_Neon{side}{r_i}", pts, 0.0055, mat, res=1)

    # rail posts, lift chain, supports
    mb = MB()
    for i in range(0, n, 6):
        for s in (1, -1):
            top = C[i] + L[i] * s * RAIL_W - N[i] * RAIL_H
            mb.cyl(top, top - N[i] * (DECK_H - RAIL_H), 0.0045, seg=6)
    for i in range(0, n, 25):          # cross ties on top of deck (sleepers)
        c = C[i] - N[i] * (DECK_H - 0.006)
        rot = Matrix((L[i], T[i], N[i])).transposed().to_euler()
        mb.box(c, (DECK_W * 2.1, 0.02, 0.012), rot)

    sup = MB()
    last = -999
    for i in range(n):
        if tags[i] in ("spiral", "loop") or N[i].z < 0.75:
            continue
        if i - last < 28:
            continue
        last = i
        top = C[i] - N[i] * (DECK_H + 0.01)
        x, y = top.x, top.y
        on_field = -1.16 < x < 1.16 and -1.43 < y < -0.24
        foot = FIELD_TOP if on_field else BASE_TOP
        if any((Vector((x, y)) - Vector(c)).length < rad for c, rad in KEEP_OUT):
            continue
        if top.z - foot > 0.1:
            sup.cyl(top, (x, y, foot), 0.011, seg=8)
            sup.cyl((x, y, foot), (x, y, foot + 0.02), 0.025, seg=10)   # foot plate
    if spiral:
        cx, cy, z0, z1 = spiral
        sup.cyl((cx, cy, BASE_TOP), (cx, cy, z1 + 0.15), 0.032, seg=16)
        sup.sphere((cx, cy, z1 + 0.18), 0.05, 2)
        last = -999
        for i in range(n):
            if tags[i] != "spiral" or i - last < 22:
                continue
            last = i
            d = C[i] - N[i] * (DECK_H + 0.01)
            sup.cyl((cx, cy, d.z - 0.04), d, 0.008, seg=6)
    if loop:
        cx, ym, cz, r = loop
        for s in (-1, 1):
            sup.cyl((cx, ym, cz - r - DECK_H - 0.02), (cx + s * 0.32, ym, BASE_TOP), 0.014, seg=8)
            sup.cyl((cx + s * 0.32, ym, BASE_TOP), (cx + s * 0.32, ym, BASE_TOP + 0.02), 0.03, seg=10)
    mb.obj(f"{name}_Posts", M["brass"])
    sup.obj(f"{name}_Supports", M["brass"])

    # little bulbs along the outside of the deck
    lb = MB()
    for i in range(0, n, 32):
        for s in (1, -1):
            lb.sphere(C[i] - N[i] * (DECK_H + 0.004) + L[i] * s * (DECK_W + 0.016), 0.011, 1)
    lb.obj(f"{name}_Bulbs", M["bulb"])

    # lift chain + lift motor
    lift_idx = [i for i in range(n) if tags[i] == "lift"]
    if lift_idx:
        pts = [C[i] - N[i] * (R_MARBLE + 0.006) for i in lift_idx]
        tube(f"{name}_LiftChain", pts, 0.0045, M["dark"], res=1)
        dogs = MB()
        for i in lift_idx[::5]:
            c = C[i] - N[i] * (R_MARBLE + 0.004)
            rot = Matrix((L[i], T[i], N[i])).transposed().to_euler()
            dogs.box(c, (0.012, 0.018, 0.012), rot)
        dogs.obj(f"{name}_ChainLinks", M["dark"])
    return C, tags


def marble_motion(C, tags, frames, count, g=3.2, v0=0.55, v_lift=0.5):
    """Energy-based speeds along a closed track: slow chain lift, fast drops."""
    n = len(C)
    htop = max(C[i].z for i in range(n) if tags[i] != "lift")
    t_at = [0.0]
    for i in range(1, n + 1):
        a, b = C[i - 1], C[i % n]
        h = (a.z + b.z) / 2
        v = v_lift if tags[i - 1] == "lift" else math.sqrt(v0 ** 2 + 2 * g * max(0.0, htop - h))
        t_at.append(t_at[-1] + (b - a).length / v)
    period = t_at[-1]

    def pos(t):
        t %= period
        lo, hi = 0, n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if t_at[mid] <= t:
                lo = mid
            else:
                hi = mid
        f = (t - t_at[lo]) / max(1e-9, t_at[lo + 1] - t_at[lo])
        return C[lo].lerp(C[(lo + 1) % n], f)

    tracks = []
    for k in range(count):
        off = period * k / count
        tracks.append([pos(off + (f - 1) / FPS) for f in frames])
    return tracks, period


# ----------------------------------------------------------------------------- scene parts
def build_room(M):
    mb = MB()
    mb.box((0, 2, -0.01), (30, 30, 0.02))
    mb.obj("Floor", M["floor"])
    # pleated curtain
    bm = bmesh.new()
    cols, H, W = 160, 7.0, 16.0
    grid = []
    for j in range(2):
        row = []
        for i in range(cols + 1):
            x = -W / 2 + W * i / cols
            y = 3.6 + 0.12 * math.sin(i * 0.9) + 0.05 * math.sin(i * 2.3)
            row.append(bm.verts.new((x, y, H * j)))
        grid.append(row)
    for i in range(cols):
        f = bm.faces.new((grid[0][i], grid[0][i + 1], grid[1][i + 1], grid[1][i]))
        f.smooth = True
    me = bpy.data.meshes.new("Curtain")
    bm.to_mesh(me)
    bm.free()
    link(bpy.data.objects.new("Curtain", me)).data.materials.append(M["curtain"])


def build_base(M):
    outline = rounded_rect(-2.45, -1.58, 2.45, 1.6, 0.45)
    b = MB()
    b.prism(outline, 0.05, BASE_TOP - 0.04)
    b.obj("Base", M["wood_dark"])
    t = MB()
    t.prism(rounded_rect(-2.5, -1.63, 2.5, 1.65, 0.48), BASE_TOP - 0.04, BASE_TOP)
    t.prism(rounded_rect(-2.47, -1.6, 2.47, 1.62, 0.46), 0.0, 0.05)
    t.obj("BaseTrim", M["brass"])
    led = MB()
    led.prism(rounded_rect(-2.44, -1.57, 2.44, 1.59, 0.44), 0.055, 0.075)
    led.obj("BaseLED", M["led"])
    # brass ribs on the base skirt
    ribs = MB()
    per = outline
    for k in range(0, len(per), 3):
        x, y = per[k]
        ribs.box((x * 1.004, y * 1.004 if y > 0 else y * 1.004, 0.27), (0.03, 0.03, 0.36))
    ribs.obj("BaseRibs", M["brass"])
    # bulbs along the top edge
    bulbs = MB()
    rr = rounded_rect(-2.42, -1.55, 2.42, 1.57, 0.42, seg=6)
    acc, prev = 0.0, Vector((*rr[0], 0))
    for i in range(1, len(rr) * 1 + 1):
        p = Vector((*rr[i % len(rr)], 0))
        seglen = (p - prev).length
        steps = max(1, int(seglen / 0.11))
        for s in range(steps):
            q = prev.lerp(p, s / steps)
            bulbs.sphere((q.x, q.y, BASE_TOP + 0.018), 0.017, 1)
        prev = p
    # decorative gears on the front skirt (animated)
    gears = []
    for gx, r, teeth, spd in ((-1.6, 0.17, 14, 1), (-1.22, 0.11, 9, -1.55), (1.25, 0.15, 12, -1),
                              (1.6, 0.1, 8, 1.5), (-0.55, 0.09, 8, 1.2), (0.6, 0.12, 10, -0.9)):
        g = MB()
        g.cyl((0, 0.012, 0), (0, -0.012, 0), r, seg=32)
        g.cyl((0, 0.02, 0), (0, -0.02, 0), r * 0.25, seg=16)
        for k in range(teeth):
            a = 2 * math.pi * k / teeth
            g.box((math.cos(a) * (r + 0.02), 0, math.sin(a) * (r + 0.02)), (0.035, 0.022, 0.03), (0, -a, 0))
        for k in range(4):
            a = k * math.pi / 2 + 0.4
            g.box((math.cos(a) * r * 0.55, -0.014, math.sin(a) * r * 0.55), (r * 0.3, 0.006, r * 0.3), (0, -a, 0))
        o = g.obj(f"Gear{len(gears)}", M["brass"], loc=(gx, -1.6, 0.26))
        gears.append((o, spd))
    return bulbs, gears


def build_playfield(M):
    fx0, fy0, fx1, fy1 = -1.16, -1.43, 1.16, -0.24
    pf = MB()
    pf.prism(rounded_rect(fx0, fy0, fx1, fy1, 0.25), BASE_TOP, FIELD_TOP)
    pf.obj("Playfield", M["wood_light"])
    rim_out = rounded_rect(fx0 - 0.03, fy0 - 0.03, fx1 + 0.03, fy1 + 0.03, 0.27)
    rim = MB()
    rim.prism(rim_out, FIELD_TOP - 0.01, FIELD_TOP + 0.045)
    rim.obj("PlayfieldRim", M["brass"])
    neon = [(x, y, FIELD_TOP + 0.05) for x, y in rounded_rect(fx0 - 0.01, fy0 - 0.01, fx1 + 0.01, fy1 + 0.01, 0.26, 10)]
    tube("PlayfieldNeon", neon, 0.008, M["purple"], closed=True)

    # flippers (pivot at origin, extend along +x / -x)
    flippers = []
    for side, px in ((1, -0.44), (-1, 0.44)):
        f = MB()
        f.cyl((0, 0, -0.01), (0, 0, 0.04), 0.05, seg=24)
        f.cyl((side * 0.34, 0, -0.01), (side * 0.34, 0, 0.04), 0.024, seg=16)
        bm = f.bm
        bmesh.ops.convex_hull(bm, input=bm.verts[:])
        for fa in bm.faces:
            fa.smooth = False
        o = f.obj(f"Flipper{'L' if side > 0 else 'R'}", M["flipper"], loc=(px, -1.17, FIELD_TOP + 0.012))
        rest = math.radians(-28 * side)
        o.rotation_euler.z = rest
        e = MB()
        e.cyl((px, -1.17, FIELD_TOP), (px, -1.17, FIELD_TOP + 0.06), 0.012, seg=10)
        e.obj(f"FlipperPin{len(flippers)}", M["chrome"])
        flippers.append((o, side, rest))

    # slingshots + inlane guides
    deco = MB()
    neon_bits = MB()
    for s in (1, -1):
        tri = [(s * 0.78, -0.86), (s * 0.78, -1.12), (s * 0.58, -1.12)]
        tri = tri if s < 0 else list(reversed(tri))
        deco.prism(tri, FIELD_TOP, FIELD_TOP + 0.05)
        for (x, y) in tri:
            deco.cyl((x, y, FIELD_TOP), (x, y, FIELD_TOP + 0.08), 0.01, seg=8)
        neon_bits.cyl((s * 0.78, -0.86, FIELD_TOP + 0.055), (s * 0.58, -1.12, FIELD_TOP + 0.055), 0.008, seg=8)
        # inlane rail
        a, b = (s * 0.98, -0.78), (s * 0.52, -1.1)
        deco.cyl((*a, FIELD_TOP + 0.035), (*b, FIELD_TOP + 0.035), 0.008, seg=8)
        deco.cyl((*a, FIELD_TOP), (*a, FIELD_TOP + 0.05), 0.012, seg=8)
    deco.obj("Slingshots", M["wood_dark"])
    neon_bits.obj("SlingshotNeon", M["orange"])

    # pop bumpers (each with its own pulsing material)
    bumpers = []
    for k, (x, y) in enumerate(BUMPERS):
        mat = emissive(f"BumperCap{k}", (1.0, 0.1, 0.06), 10)
        bmb = MB()
        bmb.cyl((x, y, FIELD_TOP), (x, y, FIELD_TOP + 0.035), 0.075, seg=24)
        bmb.cyl((x, y, FIELD_TOP + 0.075), (x, y, FIELD_TOP + 0.085), 0.08, seg=24)
        bmb.obj(f"BumperBody{k}", M["chrome"])
        cap = MB()
        cap.cyl((x, y, FIELD_TOP + 0.035), (x, y, FIELD_TOP + 0.075), 0.068, seg=24)
        cap.obj(f"BumperCap{k}", mat)
        bumpers.append(mat)

    # central target ring
    ring_c, ring_r = RING_C, RING_R
    ring = [(ring_c[0] + ring_r * math.cos(a), ring_c[1] + ring_r * math.sin(a), FIELD_TOP + 0.03)
            for a in [2 * math.pi * k / 64 for k in range(64)]]
    tube("TargetRing", ring, 0.012, M["blue"], closed=True)
    ring2 = [(ring_c[0] + 0.1 * math.cos(a), ring_c[1] + 0.1 * math.sin(a), FIELD_TOP + 0.005)
             for a in [2 * math.pi * k / 48 for k in range(48)]]
    tube("TargetRingInner", ring2, 0.006, M["blue"], closed=True)
    posts = MB()
    for k in range(12):
        a = 2 * math.pi * k / 12
        x, y = ring_c[0] + (ring_r + 0.03) * math.cos(a), ring_c[1] + (ring_r + 0.03) * math.sin(a)
        posts.cyl((x, y, FIELD_TOP), (x, y, FIELD_TOP + 0.07), 0.008, seg=8)
    posts.cyl((*RING_C, FIELD_TOP), (*RING_C, FIELD_TOP + 0.12), 0.012, seg=10)
    posts.obj("TargetPosts", M["chrome"])
    tip = MB()
    tip.uvsphere((*RING_C, FIELD_TOP + 0.13), 0.025)
    tip.obj("TargetTip", M["blue"])

    # inserts (little lit discs in the playfield)
    ins = {"orange": MB(), "red": MB(), "blue": MB()}
    spots = [(-0.3, -1.05), (0.3, -1.05), (-0.42, -0.62), (0.42, -0.62), (-0.6, -0.35), (0.6, -0.35),
             (-0.3, -0.33), (0.3, -0.33), (-0.95, -1.0), (0.95, -1.0), (0, -1.08)]
    for k, (x, y) in enumerate(spots):
        key = ("orange", "red", "blue")[k % 3]
        ins[key].cyl((x, y, FIELD_TOP - 0.002), (x, y, FIELD_TOP + 0.003), 0.03, seg=20)
    for key, mb in ins.items():
        mb.obj(f"Inserts_{key}", M[key])
    # orbit lanes around the top of the playfield (orange neon + brass guide)
    for s in (1, -1):
        arc = [(s * (0.25 + 0.75 * math.cos(a)), -0.95 + 0.62 * math.sin(a), FIELD_TOP + 0.035)
               for a in [math.radians(15 + 70 * k / 40) for k in range(41)]]
        tube(f"OrbitNeon{s:+d}", arc, 0.008, M["orange"])
        arc2 = [(x * 1.06, y - 0.03, z + 0.005) for x, y, z in arc]
        tube(f"OrbitRail{s:+d}", arc2, 0.007, M["brass"])
    drain = MB()
    drain.uvsphere((0, -1.38, FIELD_TOP), 0.05)
    drain.obj("DrainLight", M["purple"])
    return flippers, bumpers


def build_frame(M):
    w = MB()
    br = MB()
    back_x = [-2.25, -1.5, -0.75, 0.75, 1.5, 2.25]
    yb = 1.47
    for x in back_x:
        h = 3.42
        w.box((x, yb, (BASE_TOP + h) / 2), (0.07, 0.07, h - BASE_TOP))
    for z in (1.25, 2.05, 2.82, 3.32):
        w.box((0, yb, z), (4.6, 0.06, 0.05))
    for x0, x1 in zip(back_x[:-1], back_x[1:]):
        if abs(x0 + x1) < 0.1:
            continue
        for za, zb in ((1.25, 2.05), (2.05, 2.82)):
            br.cyl((x0, yb - 0.04, za), (x1, yb - 0.04, zb), 0.009, seg=6)
            br.cyl((x1, yb - 0.04, za), (x0, yb - 0.04, zb), 0.009, seg=6)
    for s in (-1, 1):
        cols = [(-0.55, 1.8), (0.45, 2.7)]
        for y, h in cols:
            w.box((s * 2.3, y, (BASE_TOP + h) / 2), (0.07, 0.07, h - BASE_TOP))
            br.sphere((s * 2.3, y, h + 0.04), 0.04, 2)
        w.box((s * 2.3, 0.45, 1.25), (0.06, 2.0, 0.05))
        w.box((s * 2.3, 0.95, 2.05), (0.06, 1.0, 0.05))
        br.cyl((s * 2.3, -0.55, 1.25), (s * 2.3, 0.45, 2.05), 0.009, seg=6)
        br.cyl((s * 2.3, 0.45, 2.05), (s * 2.3, 1.45, 2.82), 0.009, seg=6)
    w.obj("FrameWood", M["wood"])
    br.obj("FrameBrass", M["brass"])
    bulbs = MB()
    for k in range(31):
        x = -2.25 + 4.5 * k / 30
        bulbs.sphere((x, yb - 0.045, 3.35), 0.016, 1)
        bulbs.sphere((x, yb - 0.045, 2.05), 0.013, 1)
    return bulbs


def build_wheel(name, loc, r, M, ring_mat, spokes=12, bulbs_n=24):
    e = empty(name, loc)
    mb = MB()
    for k in range(spokes):
        a = 2 * math.pi * k / spokes
        mb.cyl((0, 0, 0), (r * math.cos(a), 0, r * math.sin(a)), 0.008, seg=6)
        c = Vector((r * 1.04 * math.cos(a), 0, r * 1.04 * math.sin(a)))
        mb.box(c, (0.06, 0.07, 0.04), (0, -a, 0))
    mb.cyl((0, 0.06, 0), (0, -0.06, 0), 0.05, seg=24)
    hub = mb.obj(f"{name}_Spokes", M["brass"])
    hub.parent = e
    for yy in (-0.035, 0.035):
        ring = [(r * math.cos(a), yy, r * math.sin(a)) for a in [2 * math.pi * k / 96 for k in range(96)]]
        o = tube(f"{name}_Rim{yy:+.2f}", ring, 0.012, M["brass"], closed=True)
        o.parent = e
    glow = [(r * 0.86 * math.cos(a), -0.045, r * 0.86 * math.sin(a)) for a in [2 * math.pi * k / 96 for k in range(96)]]
    tube(f"{name}_Glow", glow, 0.009, ring_mat, closed=True).parent = e
    b = MB()
    for k in range(bulbs_n):
        a = 2 * math.pi * (k + 0.5) / bulbs_n
        b.sphere((r * 0.93 * math.cos(a), -0.05, r * 0.93 * math.sin(a)), 0.014, 1)
    b.obj(f"{name}_Bulbs", M["bulb"]).parent = e
    # stand
    st = MB()
    for s in (-1, 1):
        st.cyl((loc[0] + s * r * 0.6, loc[1] + 0.1, BASE_TOP if loc[2] - r < 1.6 else loc[2] - r - 0.4),
               (loc[0], loc[1] + 0.1, loc[2]), 0.016, seg=8)
    st.cyl((loc[0], loc[1] + 0.1, loc[2]), (loc[0], loc[1] - 0.08, loc[2]), 0.02, seg=10)
    st.obj(f"{name}_Stand", M["dark"])
    return e


def build_music(M):
    # organ pipes
    pipes = MB()
    lips = MB()
    for k in range(13):
        x = -0.45 + 0.075 * k
        h = 0.45 + 0.28 * math.sin(math.pi * k / 12)
        pipes.cyl((x, 1.36, 2.76), (x, 1.36, 2.76 + h), 0.03, seg=16)
        lips.box((x, 1.33, 2.85), (0.03, 0.01, 0.025))
    pipes.obj("OrganPipes", M["brass"])
    lips.obj("OrganLips", M["dark"])
    # kick drum
    d = MB()
    d.cyl((1.15, 1.25, 2.3), (1.15, 1.47, 2.3), 0.24, seg=48)
    d.obj("DrumShell", M["wood_dark"])
    s = MB()
    s.cyl((1.15, 1.245, 2.3), (1.15, 1.25, 2.3), 0.235, seg=48)
    s.obj("DrumSkin", M["skin"])
    rim = MB()
    for y in (1.245, 1.475):
        ring = [(1.15 + 0.245 * math.cos(a), y, 2.3 + 0.245 * math.sin(a)) for a in [2 * math.pi * k / 64 for k in range(64)]]
        tube(f"DrumHoop{y}", ring, 0.012, M["brass"], closed=True)
    for k in range(10):
        a = 2 * math.pi * k / 10
        rim.cyl((1.15 + 0.25 * math.cos(a), 1.25, 2.3 + 0.25 * math.sin(a)),
                (1.15 + 0.25 * math.cos(a), 1.47, 2.3 + 0.25 * math.sin(a)), 0.006, seg=6)
    rim.obj("DrumTension", M["chrome"])
    # horn
    h = MB()
    base, mouth = Vector((2.3, 0.95, 2.36)), Vector((2.68, 0.55, 2.46))
    h.cyl(base, base.lerp(mouth, 0.55), 0.03, r2=0.05, seg=24, caps=False)
    h.cyl(base.lerp(mouth, 0.55), mouth, 0.05, r2=0.21, seg=32, caps=False)
    ho = h.obj("Horn", M["brass"])
    ho.modifiers.new("Thick", "SOLIDIFY").thickness = 0.008
    st = MB()
    st.cyl((2.3, 0.95, 2.36), (2.3, 1.45, 2.36), 0.012, seg=8)
    st.obj("HornArm", M["dark"])


def panel(name, loc, yaw, size, lines, M, post_from=None):
    e = empty(name, loc, (0, 0, math.radians(yaw)))
    w, h = size
    p = MB()
    p.box((0, 0.02, 0), (w + 0.06, 0.05, h + 0.06))
    p.obj(f"{name}_Frame", M["wood_dark"]).parent = e
    s = MB()
    s.box((0, -0.008, 0), (w, 0.01, h))
    s.obj(f"{name}_Screen", M["screen"]).parent = e
    b = MB()
    n = 16
    for k in range(n):
        t = k / n
        for (x0, y0, x1, y1) in ((-w / 2 - 0.03, h / 2 + 0.03, w / 2 + 0.03, h / 2 + 0.03),
                                 (-w / 2 - 0.03, -h / 2 - 0.03, w / 2 + 0.03, -h / 2 - 0.03)):
            b.sphere((x0 + (x1 - x0) * t, -0.01, y0), 0.011, 1)
    b.obj(f"{name}_Bulbs", M["bulb"]).parent = e
    for body, sz, z, mat in lines:
        text(f"{name}_T{z:.2f}", body, sz, (0, -0.016, z), M[mat], parent=e)
    if post_from is not None:
        po = MB()
        for sx in (-w / 3, w / 3):
            po.cyl((sx, 0.03, -h / 2 - 0.03), (sx, 0.03, post_from - loc[2]), 0.014, seg=8)
        po.obj(f"{name}_Posts", M["brass"]).parent = e
    return e


def build_signs(M):
    # marquee
    e = empty("Marquee", (-1.25, 1.36, 3.6), (math.radians(-10), 0, 0))
    m = MB()
    m.box((0, 0.03, 0), (1.42, 0.06, 0.4))
    m.obj("Marquee_Board", M["wood_dark"]).parent = e
    t = MB()
    t.box((0, 0.0, 0), (1.46, 0.02, 0.44))
    t.obj("Marquee_Trim", M["brass"]).parent = e
    text("Marquee_Text", "HAUKE", 0.17, (0, -0.02, 0.065), M["text_warm"], parent=e)
    text("Marquee_Sub", "AI EXPERIMENTS", 0.095, (0, -0.02, -0.085), M["text_blue"], parent=e)
    b = MB()
    for k in range(26):
        x = -0.69 + 1.38 * k / 25
        b.sphere((x, -0.015, 0.185), 0.013, 1)
        b.sphere((x, -0.015, -0.185), 0.013, 1)
    for k in range(5):
        z = -0.185 + 0.37 * k / 4
        b.sphere((-0.71, -0.015, z), 0.013, 1)
        b.sphere((0.71, -0.015, z), 0.013, 1)
    b.obj("Marquee_Bulbs", M["bulb"]).parent = e
    st = MB()
    for x in (-1.75, -0.75):
        st.cyl((x, 1.45, 3.32), (x, 1.4, 3.42), 0.015, seg=8)
    st.obj("Marquee_Stand", M["brass"])

    panel("ScorePanel", (-2.78, -0.62, 1.12), -24, (0.62, 0.5), [
        ("MARBLE CHAOS", 0.06, 0.15, "text_blue"),
        ("SCORE", 0.06, 0.04, "text_white"),
        ("1.337.420", 0.12, -0.11, "text_warm"),
    ], M, post_from=0.0)
    panel("NowPlaying", (0.0, 0.12, 0.98), 0, (0.52, 0.26), [
        ("NOW PLAYING", 0.045, 0.07, "text_white"),
        ("GRAVITY WALTZ", 0.075, -0.035, "text_blue"),
    ], M, post_from=BASE_TOP)
    panel("TrackList", (2.8, -0.35, 1.32), 24, (0.56, 0.62), [
        ("TRACK SELECT", 0.055, 0.23, "text_warm"),
        ("1. Gravity Waltz", 0.042, 0.12, "text_white"),
        ("2. Steel Rain", 0.042, 0.04, "text_white"),
        ("3. Loop Theory", 0.042, -0.04, "text_white"),
        ("4. Brass Spiral", 0.042, -0.12, "text_white"),
        ("5. Midnight Run", 0.042, -0.20, "text_white"),
    ], M, post_from=0.0)


def build_reservoir(M):
    r = MB()
    r.prism(rounded_rect(-2.18, -1.48, -1.72, -1.24, 0.04), BASE_TOP, BASE_TOP + 0.07)
    r.obj("Reservoir", M["wood"])
    s = MB()
    for i in range(7):
        for j in range(3):
            s.uvsphere((-2.12 + i * 0.066 + (j % 2) * 0.03, -1.42 + j * 0.065, BASE_TOP + 0.085), 0.03, 16, 8)
    s.obj("ReservoirMarbles", M["chrome"])


# ----------------------------------------------------------------------------- tracks
def circuit():
    """One continuous circuit: chain lift (right) -> run along the back -> wide spiral (left)
    -> run across the front -> vertical loop over the playfield -> sweep round the front -> lift."""
    P = [(2.1, -0.52, 0.84, "lift"), (2.09, 0.22, 1.92, "lift"), (2.06, 0.92, 2.96, "lift"),
         (1.86, 1.16, 3.1, ""), (1.2, 1.17, 3.1, ""), (0.4, 1.16, 3.06, ""), (-0.4, 1.12, 3.02, "")]
    cx, cy, R = -1.18, 0.33, 0.74
    a0, a1 = 75, 990                 # counter-clockwise, 2.5 turns, exits at the front heading +x
    z0, z1 = 2.95, 1.46
    steps = int((a1 - a0) / 20)
    for k in range(steps + 1):
        a = math.radians(a0 + (a1 - a0) * k / steps)
        P.append((cx + R * math.cos(a), cy + R * math.sin(a), z0 + (z1 - z0) * k / steps, "spiral"))
    P += [(-0.55, -0.42, 1.42, ""), (0.0, -0.46, 1.37, ""), (0.5, -0.5, 1.33, "")]
    lx, lz, lr = 0.9, 1.66, 0.35
    for k in range(13):
        th = math.radians(30 * k)
        P.append((lx + lr * math.sin(th), -0.52 - 0.26 * k / 12, lz - lr * math.cos(th), "loop"))
    P += [(1.32, -0.92, 1.12, ""), (1.72, -1.27, 0.92, ""), (2.08, -1.2, 0.84, ""), (2.16, -0.86, 0.83, "")]
    return P, (cx, cy, z1, z0), (lx, -0.65, lz, lr)


# ----------------------------------------------------------------------------- lights, camera, render
def build_lights():
    def light(name, kind, loc, energy, color, size=0.5, aim=None, spot=None):
        ld = bpy.data.lights.new(name, kind)
        ld.energy = energy
        ld.color = color
        if kind == "AREA":
            ld.shape = "DISK"
            ld.size = size
        elif kind in ("POINT", "SPOT"):
            ld.shadow_soft_size = size
        if spot:
            ld.spot_size = math.radians(spot)
            ld.spot_blend = 0.6
        o = link(bpy.data.objects.new(name, ld))
        o.location = loc
        if aim:
            d = Vector(aim) - Vector(loc)
            o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return o

    light("Key", "AREA", (-3.5, -4.2, 5.0), 700, (1.0, 0.64, 0.34), 2.5, aim=(0, 0.2, 1.4))
    light("RimPurple", "AREA", (-3.5, 3.2, 4.2), 260, (0.55, 0.25, 1.0), 2.0, aim=(-0.8, 0.5, 1.8))
    light("RimBlue", "AREA", (3.6, 3.0, 3.8), 220, (0.25, 0.5, 1.0), 2.0, aim=(0.8, 0.5, 1.8))
    light("Fill", "AREA", (3.5, -4.5, 2.2), 110, (1.0, 0.7, 0.45), 3.0, aim=(0, 0, 1.2))
    light("TopBeam", "SPOT", (0.2, 0.6, 6.5), 3200, (1.0, 0.68, 0.38), 0.1, aim=(0, 0, 1.0), spot=26)
    for name, loc, e, c in (("GlowA", (-1.05, 0.35, 2.0), 45, (0.6, 0.2, 1.0)),
                            ("GlowB", (1.0, -0.2, 1.46), 35, (0.2, 0.5, 1.0)),
                            ("GlowField", (0.0, -0.8, 0.95), 70, (1.0, 0.45, 0.2)),
                            ("GlowWheel", (0.0, 0.9, 2.1), 60, (1.0, 0.55, 0.25))):
        light(name, "POINT", loc, e, c, 0.25)


def build_world():
    w = bpy.data.worlds.new("MarbleWorld")
    bpy.context.scene.world = w
    nt = _nodes(w)
    bg = next(n for n in nt.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.003, 0.0025, 0.005, 1)
    bg.inputs["Strength"].default_value = 1.0
    if HAZE:
        out = next(n for n in nt.nodes if n.type == "OUTPUT_WORLD")
        vol = nt.nodes.new("ShaderNodeVolumePrincipled")
        vol.inputs["Density"].default_value = 0.0015
        vol.inputs["Anisotropy"].default_value = 0.35
        vol.inputs["Color"].default_value = (0.9, 0.85, 1.0, 1)
        nt.links.new(vol.outputs[0], out.inputs["Volume"])


def build_camera(frames):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = 32
    cam = link(bpy.data.objects.new("Camera", cd))
    tgt = empty("CameraTarget", (0, 0.1, 1.76))
    con = cam.constraints.new("TRACK_TO")
    con.target = tgt
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    cd.dof.use_dof = True
    cd.dof.focus_object = tgt
    cd.dof.aperture_fstop = 3.2
    keys_vec(cam, "location", [1, FRAMES], [(-1.7, -6.75, 2.6), (1.5, -6.55, 3.15)], "BEZIER")
    keys_vec(tgt, "location", [1, FRAMES], [(-0.15, 0.1, 1.74), (0.15, 0.05, 1.8)], "BEZIER")
    cam.location = (-1.7, -6.75, 2.6)
    bpy.context.scene.camera = cam
    return cam


def build_probes():
    lp = bpy.data.lightprobes.new("IrradianceVolume", "VOLUME")
    o = link(bpy.data.objects.new("IrradianceVolume", lp))
    o.location = (0, 0.05, 1.9)
    o.scale = (2.7, 1.75, 1.95)
    setp(lp, "resolution_x", 14)
    setp(lp, "resolution_y", 10)
    setp(lp, "resolution_z", 10)
    sp = bpy.data.lightprobes.new("ReflectionSphere", "SPHERE")
    o2 = link(bpy.data.objects.new("ReflectionSphere", sp))
    o2.location = (0, -0.4, 1.6)
    setp(sp, "influence_distance", 4.5)


def setup_compositor(scene):
    try:
        if hasattr(scene, "compositing_node_group"):          # Blender 5.0+
            ng = bpy.data.node_groups.new("MarbleComp", "CompositorNodeTree")
            ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
            rl = ng.nodes.new("CompositorNodeRLayers")
            out = ng.nodes.new("NodeGroupOutput")
            scene.compositing_node_group = ng
        else:                                                 # Blender 4.x
            scene.use_nodes = True
            ng = scene.node_tree
            for n in list(ng.nodes):
                ng.nodes.remove(n)
            rl = ng.nodes.new("CompositorNodeRLayers")
            out = ng.nodes.new("CompositorNodeComposite")
        g = ng.nodes.new("CompositorNodeGlare")
        if "Type" in g.inputs:
            g.inputs["Type"].default_value = "Bloom"
            sock(g, "Quality", "High")
            sock(g, "Threshold", 2.0)
            sock(g, "Strength", 0.3)
            sock(g, "Size", 0.55)
            sock(g, "Saturation", 1.1)
        else:
            setp(g, "glare_type", "BLOOM")
            setp(g, "quality", "HIGH")
            setp(g, "threshold", 1.2)
            setp(g, "mix", -0.45)
            setp(g, "size", 8)
        ld = ng.nodes.new("CompositorNodeLensdist")
        sock(ld, "Dispersion", 0.012)
        sock(ld, "Distortion", 0.0)
        ng.links.new(rl.outputs["Image"], g.inputs["Image"])
        ng.links.new(g.outputs["Image"], ld.inputs["Image"])
        ng.links.new(ld.outputs["Image"], out.inputs[0])
        rl.location, g.location, ld.location, out.location = (-400, 0), (-150, 0), (100, 0), (350, 0)
    except Exception as e:
        print("[marble] compositor setup skipped:", e)


def setup_render(scene):
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = eng
            break
        except TypeError:
            pass
    r = scene.render
    r.resolution_x, r.resolution_y = RESOLUTION
    r.resolution_percentage = 100
    r.fps = FPS
    scene.frame_start, scene.frame_end = 1, FRAMES
    setp(r, "use_motion_blur", MOTION_BLUR)
    setp(r, "motion_blur_shutter", 0.5)
    ee = scene.eevee
    setp(ee, "taa_render_samples", RENDER_SAMPLES)
    setp(ee, "taa_samples", VIEWPORT_SAMPLES)
    setp(ee, "use_raytracing", True)
    setp(ee, "ray_tracing_method", "SCREEN")
    rt = ee.ray_tracing_options
    setp(rt, "resolution_scale", "1")
    setp(rt, "use_denoise", True)
    setp(rt, "trace_max_roughness", 0.6)
    setp(rt, "screen_trace_quality", 0.5)
    setp(rt, "screen_trace_thickness", 0.3)
    setp(ee, "use_fast_gi", True)
    setp(ee, "fast_gi_method", "GLOBAL_ILLUMINATION")
    setp(ee, "fast_gi_resolution", "2")
    setp(ee, "fast_gi_step_count", 8)
    setp(ee, "fast_gi_distance", 1.5)
    setp(ee, "use_shadows", True)
    setp(ee, "shadow_ray_count", 2)
    setp(ee, "shadow_step_count", 8)
    setp(ee, "clamp_surface_indirect", 10.0)
    setp(ee, "volumetric_tile_size", "4")
    setp(ee, "volumetric_samples", 64)
    setp(ee, "volumetric_end", 14.0)
    setp(ee, "use_volumetric_shadows", True)
    setp(ee, "gi_diffuse_bounces", 3)
    vs = scene.view_settings
    setp(vs, "view_transform", "AgX")
    setp(vs, "look", "AgX - Punchy")
    setp(vs, "exposure", 0.0)
    im = r.image_settings
    if hasattr(im, "media_type"):
        setp(im, "media_type", "VIDEO")
    setp(im, "file_format", "FFMPEG")
    setp(r.ffmpeg, "format", "MPEG4")
    setp(r.ffmpeg, "codec", "H264")
    setp(r.ffmpeg, "constant_rate_factor", "HIGH")
    setp(r.ffmpeg, "ffmpeg_preset", "GOOD")
    r.filepath = OUTPUT
    setup_compositor(scene)


def setup_viewport():
    if bpy.app.background or not bpy.context.screen:
        return
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            sp = area.spaces.active
            sp.shading.type = "RENDERED"
            sp.region_3d.view_perspective = "CAMERA"


# ----------------------------------------------------------------------------- main
def main():
    global COLL
    clear_scene()
    scene = bpy.context.scene
    COLL = bpy.data.collections.new("MarbleChaos")
    scene.collection.children.link(COLL)
    M = build_materials()
    frames = list(range(1, FRAMES + 1))

    build_room(M)
    base_bulbs, gears = build_base(M)
    flippers, bumpers = build_playfield(M)
    frame_bulbs = build_frame(M)
    tmp = bpy.data.meshes.new("tmp")
    frame_bulbs.bm.to_mesh(tmp)
    frame_bulbs.bm.free()
    base_bulbs.bm.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)
    base_bulbs.obj("Bulbs", M["bulb"])

    build_music(M)
    build_signs(M)
    build_reservoir(M)
    wheel_a = build_wheel("WheelBig", (0.0, 1.18, 2.08), 0.44, M, M["orange"])
    wheel_b = build_wheel("WheelBlue", (1.42, 0.68, 1.98), 0.32, M, M["blue"], spokes=8, bulbs_n=16)

    # marble circuits
    marble_mesh = MB()
    marble_mesh.uvsphere((0, 0, 0), R_MARBLE, 32, 16)
    me = bpy.data.meshes.new("Marble")
    marble_mesh.bm.to_mesh(me)
    marble_mesh.bm.free()
    me.materials.append(M["chrome"])

    P, spiral, loop = circuit()
    neon = {"": M["purple"], "spiral": M["purple"], "loop": M["blue"], "lift": M["orange"]}
    C, T = build_track("Track", P, neon, M, spiral=spiral, loop=loop)
    for tname, C, T in (("", C, T),):
        paths, period = marble_motion(C, T, frames, MARBLES)
        print(f"[marble] track: {len(C)} samples, lap time {period:.1f}s")
        for k, locs in enumerate(paths):
            o = link(bpy.data.objects.new(f"Marble{tname}{k}", me))
            o.location = locs[0]
            keys_vec(o, "location", frames, locs)

    # pinballs circling the target ring
    pb = []
    for k in range(2):
        o = link(bpy.data.objects.new(f"Pinball{k}", me))
        o.scale = (0.5, 0.5, 0.5)
        locs = []
        for f in frames:
            t = (f / FPS) * 0.55 + k * 0.5
            th = 2 * math.pi * t
            rx = 0.36 + 0.04 * math.sin(3 * th)
            ry = 0.21 + 0.02 * math.cos(2 * th)
            locs.append(Vector((rx * math.cos(th), RING_C[1] + ry * math.sin(th), FIELD_TOP + R_MARBLE * 0.5)))
        keys_vec(o, "location", frames, locs)
        pb.append(o)

    # mechanical animation
    keys(wheel_a, "rotation_euler", 1, [1, FRAMES], [0, -FRAMES / FPS * 0.5])
    keys(wheel_b, "rotation_euler", 1, [1, FRAMES], [0, FRAMES / FPS * 0.9])
    for o, spd in gears:
        keys(o, "rotation_euler", 1, [1, FRAMES], [0, spd * FRAMES / FPS * 0.8])
    flick = [30, 78, 121, 170, 214, 262, 300, 345]
    for o, side, rest in flippers:
        fr, val = [1], [rest]
        for j, f in enumerate(flick):
            if (j % 2 == 0) == (side > 0) or j % 3 == 2:
                fr += [f - 1, f + 2, f + 8, f + 14]
                val += [rest, rest + math.radians(52) * side, rest + math.radians(52) * side, rest]
        keys(o, "rotation_euler", 2, fr, val)
    for k, mat in enumerate(bumpers):
        nt = mat.node_tree
        fr, val = [], []
        for f in range(1 + k * 13, FRAMES + 1, 34 + k * 5):
            fr += [f, f + 2, f + 10]
            val += [14, 90, 14]
        keys(nt, 'nodes["Emission"].inputs[1].default_value', 0, fr, val)

    build_lights()
    build_world()
    build_probes()
    build_camera(frames)
    setup_render(scene)
    scene.frame_set(1)
    setup_viewport()
    print("[marble] scene built:", len(COLL.objects), "objects")


main()
