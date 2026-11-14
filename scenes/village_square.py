# SPDX-License-Identifier: GPL-3.0-or-later
"""
VILLAGE SQUARE - procedural outdoor scene for Blender (Cycles/EEVEE).
This version contains NO lighting: no sun, no lamps, black world, no render/engine setup.
Lanterns have a faintly emissive glass (part of the model). Camera: slow arc, frames 1-60 at 30 fps.
"""
import bpy, bmesh, math, random
from mathutils import Vector, Matrix

random.seed(11)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.curves, bpy.data.worlds):
    for b in list(coll):
        if b.users == 0:
            coll.remove(b)
sc = bpy.context.scene
COL = bpy.data.collections.new("VillageSquare")
sc.collection.children.link(COL)


def P(name, col, rough=0.6, metal=0.0, trans=0.0, emit=None, s=0.0):
    m = bpy.data.materials.new(name)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*col, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if trans:
        b.inputs["Transmission Weight"].default_value = trans
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = s
    m.diffuse_color = (*col, 1)
    return m


def cobbles(name):
    m = P(name, (0.35, 0.32, 0.28), rough=0.75)
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    v = nt.nodes.new("ShaderNodeTexVoronoi"); v.inputs["Scale"].default_value = 3.5
    r = nt.nodes.new("ShaderNodeValToRGB")
    r.color_ramp.elements[0].position = 0.0; r.color_ramp.elements[0].color = (0.12, 0.1, 0.08, 1)
    r.color_ramp.elements[1].position = 0.12; r.color_ramp.elements[1].color = (0.42, 0.38, 0.33, 1)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Object"], v.inputs["Vector"])
    nt.links.new(v.outputs["Distance"], r.inputs[0]); nt.links.new(r.outputs[0], b.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.4
    nt.links.new(v.outputs["Distance"], bump.inputs["Height"]); nt.links.new(bump.outputs[0], b.inputs["Normal"])
    return m


def mesh(name, build, mat, bevel=0.0, smooth=False):
    bm = bmesh.new(); build(bm)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    o = bpy.data.objects.new(name, me); me.materials.append(mat); COL.objects.link(o)
    if bevel:
        mo = o.modifiers.new("Bevel", "BEVEL"); mo.width = bevel; mo.segments = 2
    return o


def cube(bm, c, s, rot=0.0):
    bmesh.ops.create_cube(bm, size=1, matrix=Matrix.Translation(c) @ Matrix.Rotation(rot, 4, "Z") @ Matrix.Diagonal((*s, 1)))


M = dict(ground=cobbles("Cobbles"), grass=P("Grass", (0.12, 0.25, 0.06), rough=0.95),
         roof=P("RoofTiles", (0.45, 0.14, 0.08), rough=0.7), window=P("WindowGlass", (0.05, 0.07, 0.09), rough=0.05, metal=0.3),
         frame=P("WindowFrames", (0.85, 0.85, 0.8), rough=0.5), door=P("DoorWood", (0.25, 0.12, 0.05), rough=0.6),
         stone=P("FountainStone", (0.55, 0.52, 0.47), rough=0.8), water=P("Water", (0.6, 0.75, 0.8), rough=0.02, trans=1.0),
         bark=P("Bark", (0.15, 0.09, 0.05), rough=0.9), leaves=P("Foliage", (0.1, 0.24, 0.05), rough=0.8),
         iron=P("Iron", (0.04, 0.04, 0.045), rough=0.4, metal=0.9), bench=P("BenchWood", (0.35, 0.2, 0.1), rough=0.7),
         lantern=P("LanternGlass", (1, 0.85, 0.6), rough=0.2, emit=(1.0, 0.7, 0.35), s=3.0),
         hill=P("Hills", (0.15, 0.25, 0.1), rough=1.0))
walls = [(0.85, 0.75, 0.55), (0.75, 0.45, 0.35), (0.9, 0.88, 0.8), (0.55, 0.65, 0.7), (0.8, 0.6, 0.4), (0.7, 0.72, 0.55)]
for i, c in enumerate(walls):
    M[f"wall{i}"] = P(f"Wall{i}", c, rough=0.85)

mesh("Ground", lambda bm: cube(bm, (0, 0, -0.05), (24, 24, 0.1)), M["ground"])
mesh("Meadow", lambda bm: cube(bm, (0, 0, -0.12), (140, 140, 0.1)), M["grass"])
mesh("Hills", lambda bm: [bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r,
                                                      matrix=Matrix.Translation((x, y, -r * 0.75)))
                          for x, y, r in ((-30, 60, 25), (10, 70, 30), (45, 55, 22), (-60, 40, 20))], M["hill"], smooth=True)

# houses around three sides of the square
houses = [((-7, 4, 0), 0), ((-7, -2.5, 0), 0), ((-2, 8, 0), math.pi / 2), ((4, 8, 0), math.pi / 2), ((8, 3, 0), math.pi), ((8, -3.5, 0), math.pi)]
for i, (pos, rot) in enumerate(houses):
    w, d, h = random.uniform(4.5, 5.5), random.uniform(4, 5), random.uniform(4.5, 6.5)
    R = Matrix.Translation(pos) @ Matrix.Rotation(rot, 4, "Z")
    mesh(f"House{i}", lambda bm: cube(bm, (0, 0, h / 2), (d, w, h)), M[f"wall{i % 6}"]).matrix_world = R

    def roof(bm, w=w, d=d, h=h):
        vs = [bm.verts.new(v) for v in ((-d / 2 - 0.3, -w / 2 - 0.3, h), (d / 2 + 0.3, -w / 2 - 0.3, h), (0, -w / 2 - 0.3, h + 2.2),
                                        (-d / 2 - 0.3, w / 2 + 0.3, h), (d / 2 + 0.3, w / 2 + 0.3, h), (0, w / 2 + 0.3, h + 2.2))]
        for f in ((0, 1, 2), (5, 4, 3), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)):
            bm.faces.new([vs[k] for k in f])
    mesh(f"Roof{i}", roof, M["roof"]).matrix_world = R
    winds, frames = [], []
    for fl in range(2 if h < 5.5 else 3):
        for k in range(3):
            y = -w / 2 + w * (k + 0.5) / 3
            z = 1.6 + fl * 1.7
            winds.append(((d / 2 + 0.03, y, z), (0.05, 0.7, 0.95)))
            frames.append(((d / 2 + 0.02, y, z), (0.06, 0.82, 1.07)))
    mesh(f"Windows{i}", lambda bm: [cube(bm, c, s) for c, s in winds], M["window"]).matrix_world = R
    mesh(f"WinFrames{i}", lambda bm: [cube(bm, c, s) for c, s in frames], M["frame"]).matrix_world = R
    mesh(f"Door{i}", lambda bm: cube(bm, (d / 2 + 0.03, 0, 1.05), (0.06, 1.0, 2.1)), M["door"]).matrix_world = R

# fountain in the middle
mesh("FountainBasin", lambda bm: (bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=2.0, radius2=2.0, depth=0.6,
                                                        matrix=Matrix.Translation((0, 1.0, 0.3)))), M["stone"], smooth=True)
mesh("FountainWater", lambda bm: bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=1.85, radius2=1.85, depth=0.05,
                                                       matrix=Matrix.Translation((0, 1.0, 0.55))), M["water"], smooth=True)
mesh("FountainColumn", lambda bm: (bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=0.25, radius2=0.18, depth=1.6,
                                                         matrix=Matrix.Translation((0, 1.0, 1.1))),
                                   bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=0.7, radius2=0.5, depth=0.15,
                                                         matrix=Matrix.Translation((0, 1.0, 1.9)))), M["stone"], smooth=True)


# trees
def tree_parts():
    trunks, crowns = [], []
    for x, y in ((-4.5, 6.5), (5.5, 5.5), (-5.5, 10.5), (6.8, 11.0), (-9, 9), (10, 7), (2, 12.5), (-2.5, 13), (11.5, -1), (-11.5, 1)):
        h = random.uniform(2.5, 3.5)
        trunks.append((x, y, h))
        for k in range(5):
            crowns.append((x + random.uniform(-0.8, 0.8), y + random.uniform(-0.8, 0.8), h + random.uniform(0.3, 1.6), random.uniform(0.9, 1.5)))
    return trunks, crowns
trunks, crowns = tree_parts()
mesh("TreeTrunks", lambda bm: [bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=0.2, radius2=0.12, depth=h,
                                                     matrix=Matrix.Translation((x, y, h / 2))) for x, y, h in trunks], M["bark"], smooth=True)
mesh("TreeCrowns", lambda bm: [bmesh.ops.create_icosphere(bm, subdivisions=2, radius=r, matrix=Matrix.Translation((x, y, z)))
                               for x, y, z, r in crowns], M["leaves"], smooth=True)
# benches and lantern posts
mesh("Benches", lambda bm: [cube(bm, (x, y, z), s, rot) for (x, y, rot) in ((-3.2, -1.0, 0.3), (3.2, -1.0, -0.3))
                            for (z, s) in ((0.45, (1.6, 0.45, 0.06)), (0.8, (1.6, 0.06, 0.4)))], M["bench"])
posts = [(-3, 3.5), (3, 3.5), (-3, -3.5), (3, -3.5)]
mesh("LanternPosts", lambda bm: [bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=0.06, radius2=0.04, depth=3.0,
                                                       matrix=Matrix.Translation((x, y, 1.5))) for x, y in posts], M["iron"])
mesh("Lanterns", lambda bm: [cube(bm, (x, y, 3.15), (0.25, 0.25, 0.35)) for x, y in posts], M["lantern"], bevel=0.02)

# world: black, no light
w = bpy.data.worlds.new("World"); sc.world = w
bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0, 0, 0, 1); bg.inputs[1].default_value = 0.0

# camera: slow arc around the fountain
cd = bpy.data.cameras.new("Camera"); cd.lens = 24
cam = bpy.data.objects.new("Camera", cd); COL.objects.link(cam); sc.camera = cam
tgt = bpy.data.objects.new("CamTarget", None); COL.objects.link(tgt); tgt.location = (0, 2.0, 2.6)
con = cam.constraints.new("TRACK_TO"); con.target = tgt; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
for f, ang in ((1, -105), (60, -80)):
    a = math.radians(ang)
    cam.location = (12 * math.cos(a), 1.0 + 12 * math.sin(a), 3.3)
    cam.keyframe_insert("location", frame=f)
a = math.radians(-105); cam.location = (12 * math.cos(a), 1.0 + 12 * math.sin(a), 3.3)
sc.frame_start, sc.frame_end = 1, 60
sc.render.fps = 30
sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
sc.frame_set(1)
print("[village] built", len(COL.objects), "objects")
