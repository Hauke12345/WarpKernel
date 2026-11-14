# SPDX-License-Identifier: GPL-3.0-or-later
"""Build a small test scene for WarpKernel and save it as a .blend:
    python3 examples/make_example.py examples/example.blend
A rotating cube (rigid motion), a chrome sphere (mirror), a wobbling cylinder (deforming, wave modifier),
a textured floor, two lights and a slow camera dolly. 640x360, 16 samples, frames 1-48 at 24 fps."""
import sys, math
import bpy
from mathutils import Vector

out = sys.argv[-1] if sys.argv[-1].endswith(".blend") else "example.blend"
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def mat(name, color, rough=0.5, metal=0.0, tex=False):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1); p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    if tex:
        c = nt.nodes.new("ShaderNodeTexChecker"); c.inputs["Scale"].default_value = 12
        c.inputs["Color1"].default_value = (*color, 1); c.inputs["Color2"].default_value = (color[0] * 0.55, color[1] * 0.55, color[2] * 0.55, 1)
        nt.links.new(c.outputs["Color"], p.inputs["Base Color"])
    return m


def add(obj, m):
    obj.data.materials.append(m); return obj


bpy.ops.mesh.primitive_plane_add(size=12); add(bpy.context.object, mat("Floor", (0.55, 0.42, 0.3), 0.6, tex=True)).name = "Floor"
bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 4, 3), rotation=(math.radians(90), 0, 0))
add(bpy.context.object, mat("Wall", (0.7, 0.68, 0.64), 0.8)).name = "Wall"
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-1.6, 0.6, 0.7)); cube = bpy.context.object; cube.name = "SpinningCube"
add(cube, mat("Orange", (0.9, 0.35, 0.08), 0.4))
cube.rotation_euler = (0, 0, 0); cube.keyframe_insert("rotation_euler", frame=1)
cube.rotation_euler = (0, math.radians(30), math.radians(180)); cube.keyframe_insert("rotation_euler", frame=48)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.75, location=(0.4, 1.0, 0.75), segments=64, ring_count=32)
s = bpy.context.object; s.name = "ChromeSphere"; add(s, mat("Chrome", (0.9, 0.9, 0.92), 0.03, 1.0)); bpy.ops.object.shade_smooth()
bpy.ops.mesh.primitive_cylinder_add(radius=0.35, depth=1.6, location=(2.0, 0.2, 0.8), vertices=48)
cyl = bpy.context.object; cyl.name = "WobblyCylinder"; add(cyl, mat("Teal", (0.1, 0.55, 0.5), 0.5))
bpy.ops.object.modifier_add(type="SUBSURF"); cyl.modifiers[-1].levels = 2
w = cyl.modifiers.new("Wave", "WAVE"); w.height = 0.12; w.width = 0.6; w.speed = 0.08; w.use_normal = True
sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN")); sun.data.energy = 3.0; sun.data.angle = math.radians(2)
sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(-30)); sc.collection.objects.link(sun)
area = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", "AREA")); area.data.energy = 300; area.data.size = 3
area.location = (-3, -3, 4); area.rotation_euler = (math.radians(45), 0, math.radians(-45)); sc.collection.objects.link(area)
sc.world = bpy.data.worlds.new("World"); sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.45, 0.6, 1)
sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 32
for f, x in ((1, -1.2), (48, 1.2)):
    cam.location = (x, -6.0, 2.2); cam.rotation_euler = ((Vector((0.2, 0.8, 0.7)) - cam.location).to_track_quat("-Z", "Y").to_euler())
    cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_euler", frame=f)
sc.render.engine = "CYCLES"; sc.cycles.samples = 16; sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 640, 360, 100
sc.frame_start, sc.frame_end = 1, 48; sc.render.fps = 24
sc.view_settings.view_transform = "AgX"
bpy.ops.wm.save_as_mainfile(filepath=out)
print("saved", out)
