# WarpKernel: instructions for Claude

WarpKernel renders Blender Cycles animations faster by rendering keyframes, predicting the frames in between from
exact scene geometry, and path-tracing only what changed. Use it when someone wants an animation (not a single image)
rendered faster with Cycles, typically on a CPU-only machine.

## Setup

```bash
pip install -r requirements.txt        # Blender as a Python module (bpy 5.3, Python 3.13), numpy, opencv, numba, ...
```
Run scripts with plain `python3` (bpy is imported as a module, not via `blender --background`).
If `pip install bpy` fails, check the Python version first: the bpy wheel only exists for one Python version per release.

## Workflow for "render this animation faster"

1. **Render the animation:**
   ```bash
   python3 warpkernel/render.py --blend scene.blend --out out/scene
   ```
   Read every `WARNING:` line and the list of deforming objects, and pass them on to the person. Output:
   `out/scene/frames/*.png` (the scene's own colour management), `out/scene/<name>.mp4`, `out/scene/summary.json`.
   Useful options: `--frames 1 120`, `--scale 50` (resolution %, for a quick trial), `--spp N`, `--thr`, `--split`,
   `--max-rot`.
2. **Check it:**
   ```bash
   python3 warpkernel/check.py --run out/scene
   ```
   It renders 3 reference frames (the predicted frames farthest from any keyframe) and writes
   `out/scene/check/compare_*.png` and `crops_*.png`.
3. **Look at the images, not only the numbers.** Read every `compare_*.png` and `crops_*.png`. Look for seams around
   rectangular patches, ghost outlines of objects, stale reflections, smeared shadows, wrong shading on moving objects.
   PSNR/SSIM in `check/report.json` can look fine while these are visible.
4. **If something is wrong:** see "Fixing problems" below, re-render, check again.
5. **Report honestly:** WarpKernel time vs `est_every_frame` from `summary.json` (or the reference render time from
   `check/report.json` × number of frames), the check results, the remaining visible issues, and the warnings.

For a long animation, do a trial first: `--scale 50` with a short `--frames` range, check it, then the full run.

## Rules learned the hard way

- **Never judge quality by PSNR/SSIM alone.** Patch seams, ghost outlines and stale reflections scored fine and were
  only found by looking at zoomed crops.
- **Two keyframes agreeing does not prove a pixel is right.** Mirrors can reflect different things that happen to match,
  and a cube turned 180° looks the same. That is why mirrors always go through the patch cache, and why `--max-rot`
  adds keyframes when an object turns a lot. Don't switch these off to gain speed.
- **Keep the light tree on for patches.** Without it, patches are cheaper but visibly noisier in scenes with many small
  emitters.
- **Lower resolution for keyframes or patches was tried and dropped.** Half-res patches gave wrong reflections, and
  quarter-res keyframes gave no net speed-up plus visible stripes. Don't reintroduce them without a by-eye check.
- **The speed-up depends on the scene.** Calm, mostly diffuse scenes: about 10×. Mirrors, moving chrome, flashing
  lights, fog: about 2×. Say this up front; don't promise "10×".

## Fixing problems

| Symptom in the crops | Try |
|---|---|
| wrong shading or shadow on a moving object | `--max-rot 20` (more keyframes) |
| stale or wrong reflections | `--thr 0.05` |
| patch seams, blotches | `--thr 0.05`; check that the scene's denoiser settings were not changed by hand |
| too slow, mostly re-rendering | look at `summary.json` → `deforming`, mirrors, volumes; those are re-rendered on purpose |
| artifacts everywhere | `--split 0.2` (more keyframes) |

## Not supported (the script warns and switches off)

Depth of field, motion blur, compositor effects. Volumes and fog work but are mostly re-rendered. Deforming objects
(armatures, shape keys, simulations, animated modifiers) are detected automatically and re-rendered every frame.

## Code map

| File | Purpose |
|---|---|
| `warpkernel/render.py` | entry point: keyframes, bisection, patches, output |
| `warpkernel/check.py` | reference frames, comparison and crops for the by-eye check |
| `warpkernel/scenes.py` | scene loading (`load_blend`), Cycles setup, deforming-object detection |
| `warpkernel/warp.py` | reprojection kernel (Numba), prediction, dirty rectangles, patch cache |
| `warpkernel/common.py` | render and EXR helpers |
| `examples/make_example.py` | small test scene with every feature (moving, mirror, deforming) |
| `warpkernel/baseline.py`, `evaluate.py` | benchmark tools for the three test scenes |

Commits in this repository must use the owner's GitHub no-reply address, never a private email.
