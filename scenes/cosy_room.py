# SPDX-License-Identifier: GPL-3.0-or-later
"""
COSY ROOM - procedural living room at dusk for Blender (Cycles/EEVEE).
This version contains NO lighting: no lamps, black world, no render/engine setup.
Emissive parts of the model (fairy lights, lamp bulb, dusk backdrop behind the window) are included.
Camera: slow dolly, frames 1-60 at 30 fps.
"""
import bpy, bmesh, math, random
from mathutils import Vector, Matrix

random.seed(3)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.curves, bpy.data.worlds):
    for b in list(coll):
        if b.users == 0:
            coll.remove(b)
sc = bpy.context.scene
COL = bpy.data.collections.new("CosyRoom")
sc.collection.children.link(COL)


def P(name, col, rough=0.5, metal=0.0, trans=0.0, emit=None, s=0.0, sss=0.0):
    m = bpy.data.materials.new(name)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*col, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if trans:
        b.inputs["Transmission Weight"].default_value = trans
        b.inputs["IOR"].default_value = 1.45
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = s
    if sss:
        b.inputs["Subsurface Weight"].default_value = sss
    m.diffuse_color = (*col, 1)
    return m


def wood(name, c1, c2, scale=6):
    m = P(name, c1, rough=0.45)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1, 8, 1)
    w = nt.nodes.new("ShaderNodeTexWave"); w.inputs["Scale"].default_value = scale; w.inputs["Distortion"].default_value = 6
    r = nt.nodes.new("ShaderNodeValToRGB"); r.color_ramp.elements[0].color = (*c1, 1); r.color_ramp.elements[1].color = (*c2, 1)
    nt.links.new(tc.outputs["Object"], mp.inputs[0]); nt.links.new(mp.outputs[0], w.inputs[0])
    nt.links.new(w.outputs["Fac"], r.inputs[0]); nt.links.new(r.outputs[0], b.inputs["Base Color"])
    return m


def mesh(name, build, mat, loc=(0, 0, 0), bevel=0.0, smooth=False):
    bm = bmesh.new()
    build(bm)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    o.location = loc
    me.materials.append(mat)
    COL.objects.link(o)
    if bevel:
        m = o.modifiers.new("Bevel", "BEVEL"); m.width = bevel; m.segments = 3
    return o


def box(c, s, rot=0.0):
    return lambda bm: bmesh.ops.create_cube(bm, size=1, matrix=Matrix.Translation(c) @ Matrix.Rotation(rot, 4, "Z") @ Matrix.Diagonal((*s, 1)))


def boxes(lst):
    def f(bm):
        for c, s in lst:
            bmesh.ops.create_cube(bm, size=1, matrix=Matrix.Translation(c) @ Matrix.Diagonal((*s, 1)))
    return f


M = dict(
    floor=wood("FloorOak", (0.25, 0.13, 0.06), (0.45, 0.26, 0.13)),
    wall=P("Plaster", (0.62, 0.55, 0.47), rough=0.9),
    trim=P("WhiteTrim", (0.8, 0.78, 0.74), rough=0.4),
    sofa=P("SofaFabric", (0.12, 0.22, 0.28), rough=0.95),
    cushion=P("CushionMustard", (0.6, 0.38, 0.08), rough=0.95),
    rug=P("Rug", (0.42, 0.14, 0.1), rough=1.0),
    table=wood("WalnutTable", (0.1, 0.05, 0.025), (0.24, 0.12, 0.06), 10),
    glass=P("VaseGlass", (0.9, 0.95, 0.95), rough=0.0, trans=1.0),
    brass=P("Brass", (0.8, 0.55, 0.25), rough=0.25, metal=1.0),
    black=P("BlackMetal", (0.03, 0.03, 0.03), rough=0.35, metal=0.8),
    shade=P("LampShade", (0.85, 0.75, 0.6), rough=0.8, sss=0.3),
    leaf=P("Leaf", (0.06, 0.2, 0.05), rough=0.6),
    pot=P("Terracotta", (0.5, 0.22, 0.12), rough=0.8),
    ceramic=P("MugCeramic", (0.85, 0.85, 0.82), rough=0.2),
    canvas=P("Painting", (0.2, 0.3, 0.45), rough=0.7),
    fairy=P("FairyLights", (1, 0.8, 0.5), emit=(1.0, 0.62, 0.28), s=40.0),
    bulb=P("LampBulb", (1, 0.9, 0.7), emit=(1.0, 0.7, 0.4), s=60.0),
    dusk=P("DuskBackdrop", (0, 0, 0), emit=(0.25, 0.32, 0.6), s=2.0),
    wire=P("Wire", (0.05, 0.05, 0.05), rough=0.6),
)
for i, c in enumerate([(0.5, 0.1, 0.1), (0.1, 0.25, 0.4), (0.7, 0.6, 0.4), (0.15, 0.3, 0.15), (0.35, 0.2, 0.35), (0.8, 0.75, 0.65)]):
    M[f"book{i}"] = P(f"Book{i}", c, rough=0.7)

# room: 6 x 5 x 2.8, back wall at y=2.5, window in back wall
W, D, H = 6.0, 5.0, 2.8
mesh("Floor", box((0, 0, -0.05), (W, D, 0.1)), M["floor"])
mesh("Ceiling", box((0, 0, H + 0.05), (W, D, 0.1)), M["wall"])
mesh("WallLeft", box((-W / 2 - 0.05, 0, H / 2), (0.1, D, H)), M["wall"])
mesh("WallRight", box((W / 2 + 0.05, 0, H / 2), (0.1, D, H)), M["wall"])
wx0, wx1, wz0, wz1 = 0.3, 2.3, 0.8, 2.3            # window opening in the back wall
mesh("WallBack", boxes([((-(W / 2) + (wx0 + W / 2) / 2, D / 2 + 0.05, H / 2), (wx0 + W / 2, 0.1, H)),
                         (((wx1 + W / 2) / 2, D / 2 + 0.05, H / 2), (W / 2 - wx1, 0.1, H)),
                         (((wx0 + wx1) / 2, D / 2 + 0.05, wz0 / 2), (wx1 - wx0, 0.1, wz0)),
                         (((wx0 + wx1) / 2, D / 2 + 0.05, (wz1 + H) / 2), (wx1 - wx0, 0.1, H - wz1))]), M["wall"])
mesh("WindowFrame", boxes([(((wx0 + wx1) / 2, D / 2 + 0.02, wz0), (wx1 - wx0 + 0.1, 0.12, 0.06)),
                            (((wx0 + wx1) / 2, D / 2 + 0.02, wz1), (wx1 - wx0 + 0.1, 0.12, 0.06)),
                            ((wx0, D / 2 + 0.02, (wz0 + wz1) / 2), (0.06, 0.12, wz1 - wz0)),
                            ((wx1, D / 2 + 0.02, (wz0 + wz1) / 2), (0.06, 0.12, wz1 - wz0)),
                            (((wx0 + wx1) / 2, D / 2 + 0.02, (wz0 + wz1) / 2), (0.04, 0.1, wz1 - wz0)),
                            (((wx0 + wx1) / 2, D / 2 + 0.02, (wz0 + wz1) / 2), (wx1 - wx0, 0.1, 0.04))]), M["trim"])
mesh("WindowSill", box(((wx0 + wx1) / 2, D / 2 - 0.08, wz0 - 0.02), (wx1 - wx0 + 0.3, 0.25, 0.04)), M["trim"])
# dusk backdrop outside the window (emissive gradient plane)
bd = mesh("DuskBackdrop", box((1.3, D / 2 + 1.5, 1.6), (8, 0.05, 6)), M["dusk"])
mesh("Skirting", boxes([((0, D / 2 - 0.01, 0.06), (W, 0.03, 0.12)), ((-W / 2 + 0.01, 0, 0.06), (0.03, D, 0.12)),
                         ((W / 2 - 0.01, 0, 0.06), (0.03, D, 0.12))]), M["trim"])
# rug, sofa, cushions
mesh("Rug", box((-0.3, 0.2, 0.005), (3.0, 2.0, 0.01)), M["rug"])
mesh("Sofa", boxes([((-0.6, 1.75, 0.22), (2.4, 0.9, 0.44)), ((-0.6, 2.1, 0.62), (2.4, 0.22, 0.6)),
                     ((-1.85, 1.75, 0.42), (0.2, 0.9, 0.5)), ((0.65, 1.75, 0.42), (0.2, 0.9, 0.5))]), M["sofa"], bevel=0.05)
mesh("Cushions", boxes([((-1.3, 1.9, 0.62), (0.5, 0.18, 0.45)), ((0.05, 1.9, 0.62), (0.5, 0.18, 0.45))]), M["cushion"], bevel=0.06)
# coffee table
mesh("CoffeeTable", boxes([((-0.4, 0.4, 0.42), (1.3, 0.7, 0.05))] +
                           [((-0.4 + sx * 0.58, 0.4 + sy * 0.28, 0.2), (0.05, 0.05, 0.4)) for sx in (-1, 1) for sy in (-1, 1)]), M["table"], bevel=0.01)
mesh("Vase", lambda bm: bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=0.07, radius2=0.1, depth=0.3,
                                              matrix=Matrix.Translation((-0.7, 0.45, 0.6))), M["glass"], smooth=True)
mesh("Mug", lambda bm: bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=0.045, radius2=0.045, depth=0.1,
                                             matrix=Matrix.Translation((-0.1, 0.3, 0.5))), M["ceramic"], smooth=True)
mesh("Candleholder", lambda bm: bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=0.08, radius2=0.05, depth=0.06,
                                                      matrix=Matrix.Translation((0.05, 0.55, 0.48))), M["brass"], smooth=True)
# bookshelf on the left wall with many books
books = {i: [] for i in range(6)}
for shelf in range(4):
    z = 0.35 + shelf * 0.5
    y = -1.6
    while y < 0.4:
        t = random.uniform(0.03, 0.07); h = random.uniform(0.25, 0.4)
        books[random.randrange(6)].append(((-2.82, y + t / 2, z + h / 2 + 0.02), (0.24, t, h)))
        y += t + 0.005
mesh("Shelf", boxes([((-2.82, -0.6, 0.35 + s * 0.5), (0.3, 2.2, 0.03)) for s in range(5)] +
                     [((-2.82, -1.72, 1.35), (0.3, 0.03, 2.1)), ((-2.82, 0.52, 1.35), (0.3, 0.03, 2.1))]), M["table"])
for i, lst in books.items():
    if lst:
        mesh(f"Books{i}", boxes(lst), M[f"book{i}"])
# floor lamp (bulb inside the shade is emissive)
mesh("LampBase", lambda bm: bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=0.18, radius2=0.18, depth=0.03,
                                                  matrix=Matrix.Translation((1.4, 1.9, 0.015))), M["black"])
mesh("LampPole", lambda bm: bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.015, radius2=0.015, depth=1.45,
                                                  matrix=Matrix.Translation((1.4, 1.9, 0.74))), M["black"])
mesh("LampShade", lambda bm: bmesh.ops.create_cone(bm, cap_ends=False, segments=40, radius1=0.25, radius2=0.15, depth=0.3,
                                                   matrix=Matrix.Translation((1.4, 1.9, 1.55))), M["shade"], smooth=True)
mesh("LampBulb", lambda bm: bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.05, matrix=Matrix.Translation((1.4, 1.9, 1.5))), M["bulb"], smooth=True)
# plant
mesh("Pot", lambda bm: bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=0.15, radius2=0.2, depth=0.35,
                                             matrix=Matrix.Translation((2.4, 1.7, 0.175))), M["pot"], smooth=True)


def leaves(bm):
    for k in range(26):
        a = random.uniform(0, 2 * math.pi); tilt = random.uniform(0.3, 1.1); L = random.uniform(0.35, 0.6)
        base = Vector((2.4, 1.7, 0.35))
        tip = base + Vector((math.cos(a) * math.sin(tilt), math.sin(a) * math.sin(tilt), math.cos(tilt))) * L
        d = tip - base
        Mx = Matrix.Translation((base + tip) / 2) @ d.to_track_quat("Z", "Y").to_matrix().to_4x4() @ Matrix.Diagonal((0.06, 0.015, 1, 1))
        bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=1, radius2=0.2, depth=L, matrix=Mx)
mesh("PlantLeaves", leaves, M["leaf"])
# painting on the right wall
mesh("PaintingFrame", box((2.94, 0.0, 1.6), (0.04, 1.2, 0.85)), M["brass"])
mesh("Painting", box((2.92, 0.0, 1.6), (0.03, 1.1, 0.75)), M["canvas"])


# fairy lights: a sagging string across the back wall and along the shelf (many tiny emitters)
def catenary(p0, p1, sag, n):
    return [p0.lerp(p1, k / (n - 1)) - Vector((0, 0, sag * math.sin(math.pi * k / (n - 1)))) for k in range(n)]
strands = [catenary(Vector((-2.9, 2.4, 2.45)), Vector((0.15, 2.4, 2.45)), 0.35, 70),
           catenary(Vector((2.45, 2.4, 2.45)), Vector((2.9, 0.8, 2.45)), 0.25, 30),
           catenary(Vector((-2.66, -1.7, 2.3)), Vector((-2.66, 0.5, 2.3)), 0.3, 50)]


def fairy(bm):
    for st in strands:
        for i, p in enumerate(st):
            bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.012, matrix=Matrix.Translation(p))
mesh("FairyLights", fairy, M["fairy"], smooth=True)
for j, st in enumerate(strands):
    cu = bpy.data.curves.new(f"Wire{j}", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = 0.002
    sp = cu.splines.new("POLY"); sp.points.add(len(st) - 1)
    for p, v in zip(sp.points, st):
        p.co = (v.x, v.y, v.z + 0.012, 1)
    cu.materials.append(M["wire"])
    COL.objects.link(bpy.data.objects.new(f"FairyWire{j}", cu))

# world: black, no light
w = bpy.data.worlds.new("World"); sc.world = w
bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0, 0, 0, 1); bg.inputs[1].default_value = 0.0

# camera: slow dolly from front-right toward the sofa
cd = bpy.data.cameras.new("Camera"); cd.lens = 24
cam = bpy.data.objects.new("Camera", cd); COL.objects.link(cam); sc.camera = cam
tgt = bpy.data.objects.new("CamTarget", None); COL.objects.link(tgt)
con = cam.constraints.new("TRACK_TO"); con.target = tgt; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
for f, cl, tl in ((1, (1.9, -2.2, 1.45), (-0.2, 1.2, 1.1)), (60, (1.3, -2.0, 1.35), (-0.5, 1.4, 1.05))):
    cam.location = cl; cam.keyframe_insert("location", frame=f)
    tgt.location = tl; tgt.keyframe_insert("location", frame=f)
cam.location = (1.9, -2.2, 1.45); tgt.location = (-0.2, 1.2, 1.1)
sc.frame_start, sc.frame_end = 1, 60
sc.render.fps = 30
sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
sc.frame_set(1)
print("[cosy] built", len(COL.objects), "objects")
