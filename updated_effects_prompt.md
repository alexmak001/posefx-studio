# Update Effects — Redesigned Lineup

## Context

The effect renderers in `src/render/effects/` need to be replaced with a better set of 10 effects. Read `src/render/base.py` for the `BaseRenderer` and `RenderContext` interfaces. Read `src/inference/base.py` for `PoseResult` and `MaskResult` dataclass shapes.

## Critical design rule: Head exclusion zone

ALL body effects (effects 1-8) must preserve the person's real face. Implement a shared utility function in `src/render/utils.py`:

```python
def get_head_mask(pose: PoseResult, frame_shape: tuple, scale: float = 1.8) -> np.ndarray:
    """Create a circular mask around the head region using pose keypoints.
    
    Uses nose (0), left_eye (1), right_eye (2), left_ear (3), right_ear (4)
    to estimate head center and radius. Returns a binary mask where True = head region.
    Scale controls how much larger than the detected head to protect.
    """
```

Every body effect uses this to composite: apply the effect to the body, but blend the original camera frame back in wherever the head mask is True. This way the person always sees their real face.

## The 10 effects to implement

Delete all existing effect files in `src/render/effects/` and replace with these. Each effect is its own file.

---

### 1. `neon_wireframe.py` — NeonWireframeRenderer

**Needs:** pose + mask

Body below the neck becomes a glowing neon skeleton on black. The person mask area is blacked out, then bright luminous lines are drawn between COCO keypoints with glowing dots at each joint. Head region shows the real camera face.

**Rendering approach:**
- Start with black frame
- Draw thick lines (4-8px) between connected COCO keypoints in cyan (#00FFFF)
- Draw circles (8-12px radius) at each keypoint
- Create glow by drawing the skeleton on a separate layer, applying `cv2.GaussianBlur` (kernel 21-41px), then additively blending it back
- Composite real face back using head mask

**Bass response:** Line thickness scales from 4px (silent) to 10px (max bass). Glow blur kernel scales from 21 to 51. Color shifts from cyan toward white (#CCFFFF) at high bass.

---

### 2. `motion_trails.py` — MotionTrailsRenderer

**Needs:** mask

Rainbow afterimages that trail behind movement. Store the last 10-14 frames of the person's combined_mask. Each historical mask is drawn as a semi-transparent colored overlay, with hue cycling through the spectrum (oldest = red, newest = violet). Fast movement = long colorful streaks. Still = single body.

**Rendering approach:**
- Maintain a deque of (mask, hue) tuples, max length 10-14
- Each frame, push current mask with the next hue value
- Draw from oldest to newest: for each stored mask, tint the original frame with that hue at decreasing opacity (oldest=0.1, newest=0.5)
- Current frame body is drawn normally on top
- Head region from current frame always shows real face

**Bass response:** Trail length increases from 8 to 14 frames. Colors saturate more (higher alpha). Fade time extends so trails linger longer.

---

### 3. `glitch_body.py` — GlitchBodyRenderer

**Needs:** mask + pose

Body region is sliced into horizontal bands (12-20px strips). Each strip is randomly offset horizontally by 0-15px and given RGB channel separation (red channel shifts left 2-4px, blue shifts right 2-4px). Some strips randomly show noise or scanlines. Face stays clean.

**Rendering approach:**
- Extract body region using mask (exclude head zone)
- Slice into horizontal strips
- Per strip: random horizontal offset, apply channel split using numpy array slicing
- 10-20% of strips replaced with random static noise each frame
- Overlay thin scanline pattern (every 3rd pixel row dimmed 30%)
- Composite back, head region untouched

**Bass response:** Strip offset range increases 3x. More strips go to noise (up to 40%). On heavy bass (>0.8), brief 2-frame full RGB inversion flash across body.

---

### 4. `shadow_clones.py` — ShadowClonesRenderer

**Needs:** mask + pose

3-4 time-delayed copies of your silhouette fan out behind the real body. Each clone is tinted a distinct color (purple, cyan, magenta, gold) and shows the pose from 4-8 frames ago. Real body in front with real face.

**Rendering approach:**
- Store last 8 frames of (combined_mask, frame) pairs in a deque
- Pick 3-4 evenly spaced historical frames
- For each clone: shift the mask slightly left/right (spread effect), tint the frame region with that clone's color at 40% opacity
- Draw clones back-to-front (oldest first), then real body on top
- Only real body gets the head pass-through

**Bass response:** Clones spread further apart horizontally (offset increases). Opacity pulses with beat. On bass >0.8, clones briefly scatter 30-50px outward then snap back over 5 frames.

---

### 5. `energy_aura.py` — EnergyAuraRenderer

**Needs:** mask

Pulsing energy field radiates from body outline. Multiple concentric glow layers expand outward from the silhouette edge in warm colors (gold → orange → magenta). Small particle dots drift upward along the aura edges.

**Rendering approach:**
- Get body contour with `cv2.findContours` on combined_mask
- Draw 4-6 concentric outlines at increasing distances (5px, 15px, 25px, 35px) using `cv2.dilate` then subtract to get rings
- Each ring gets a warm color and decreasing opacity
- Apply `cv2.GaussianBlur` to the ring layer for soft glow
- Maintain a list of 50-100 particle dots that spawn at random contour points and drift upward, fading over 20-30 frames
- Inside body mask: slight warm color grade on the original image
- Head region untouched

**Bass response:** Rings expand further out (multiply distances by 1 + bass_energy). Particle spawn rate scales 2x-5x. Colors shift toward white-hot at high bass.

---

### 6. `halo_wings.py` — HaloWingsRenderer

**Needs:** pose

A glowing halo ring above the head and translucent energy wings from the shoulders. Body and face pass through normally — these are overlaid additions, not replacements.

**Rendering approach:**
- **Halo:** Draw an ellipse above the nose keypoint (offset up by ~head_radius * 1.5). Use `cv2.ellipse` with golden color, 3px thickness. Draw a blurred copy underneath for glow.
- **Wings:** From each shoulder keypoint, draw 3-4 curved `cv2.polylines` paths that arc outward and slightly downward. Wings are semi-transparent (alpha 0.3-0.5), drawn in white/gold. Wing spread angle follows the shoulder-to-elbow angle so wings move with arms.
- Draw on an overlay layer, blend additively onto the original frame

**Bass response:** Wings spread 20% wider. Halo brightens and pulses in radius. At bass >0.7, small glowing dots shed from wingtips and drift downward.

---

### 7. `digital_rain.py` — DigitalRainRenderer

**Needs:** mask + pose

Body region is replaced with Matrix-style falling green characters on black. Characters stream downward in columns that are constrained to within the body mask boundary.

**Rendering approach:**
- Maintain a grid of "raindrop" columns across frame width, each with a position (y), speed, and character
- Each frame: advance each raindrop downward, assign a random character (katakana or ASCII)
- Render characters using `cv2.putText` in green (#00FF41) on a black background
- Mask this rain texture to only show within the person's combined_mask (minus head zone)
- Characters at the "head" of each column are brighter (white-green), trailing chars fade to dark green
- Composite real face back via head mask

**Bass response:** Fall speed increases 2x. Lead characters flash brighter. On bass >0.7, a horizontal glitch sweep (bright scanline) moves top-to-bottom across the body.

---

### 8. `particle_dissolve.py` — ParticleDissolveRenderer

**Needs:** mask

Body silhouette filled with thousands of colored dots. When you move, edge particles scatter outward. Still = solid particle body. Dancing = particles explode off edges and reform.

**Rendering approach:**
- Maintain a pool of 2000-3000 particles, each with (x, y, vx, vy, color, life)
- Each frame: identify pixels inside the mask. Particles inside the mask are attracted toward their nearest mask point (spring force). Particles outside drift and fade.
- When the mask edge moves (detected by comparing current mask to previous), particles near that edge get a velocity kick outward
- Draw each particle as a 2-3px circle. Colors cycle through a warm palette (coral, amber, pink, gold)
- Head region: composite real face back

**Bass response:** All particles get a radial velocity kick outward from body center on each bass pulse (proportional to bass_energy). Heavy hits (>0.8) cause a big burst where particles scatter 30-50px then reform over 10 frames.

---

### 9. `sprite_puppet.py` — SpritePuppetRenderer

**Needs:** pose

Full body replacement with a 2D character. This is avatar mode — face IS replaced.

**Rendering approach:**
- Load a sprite sheet with separate body parts: head, torso, upper_arm_L, upper_arm_R, forearm_L, forearm_R, thigh_L, thigh_R, calf_L, calf_R
- Map each part to its corresponding COCO keypoint pair (e.g., upper_arm_L stretches from left_shoulder to left_elbow)
- For each part: calculate angle and scale from keypoint positions, use `cv2.warpAffine` to position/rotate/scale the PNG
- Draw parts in z-order (back arm → torso → front arm → head)
- If no sprite assets found, fall back to a simple geometric puppet (colored rectangles for limbs)
- Support multiple skins via config (load from `src/assets/sprites/{skin_name}/`)

**Bass response:** Limb segments scale slightly (1.0-1.1x) on beat. Optional particle burst from joint positions on heavy hits.

---

### 10. `passthrough.py` — PassthroughRenderer

**Needs:** nothing

Clean camera feed. Optional subtle vignette (darken corners) for a photo-booth look.

**Rendering approach:**
- Return the original frame
- If `bass_energy > 0`: apply a subtle vignette that pulses slightly with bass (corner darkness from 0.2 to 0.35)

**Bass response:** Minimal — slight vignette pulse only, so photos aren't distractingly processed.

---

## Implementation order

Build them in this order (easiest/most impactful first):

1. `passthrough.py` — trivial, gets the framework working
2. `neon_wireframe.py` — your hero effect, skeleton rendering already exists
3. `energy_aura.py` — mask contour + glow, straightforward
4. `motion_trails.py` — frame buffer + tinting, visually impressive
5. `glitch_body.py` — pixel manipulation, very cool on a TV
6. `digital_rain.py` — character rendering, iconic look
7. `shadow_clones.py` — time-delayed masks, needs frame history
8. `particle_dissolve.py` — particle system, most complex rendering
9. `halo_wings.py` — overlay additions, needs good wing curves
10. `sprite_puppet.py` — needs sprite assets, build geometric fallback first

After each effect, verify it works by cycling to it with the `n` key. Check that:
- The effect renders at ≥15 FPS
- Head region shows the real face (except sprite puppet which is full avatar)
- Bass energy visibly affects the effect (test with audio enabled or manually set bass to 0.5)

## Also update

- `src/engine.py` — register all 10 effects
- `config/demo.yaml` — set `effects.default: "neon_wireframe"`
- `src/render/utils.py` — the shared `get_head_mask()` function and any shared particle/glow utilities
- Delete old effects that are being replaced (robot_skeleton, fire_skeleton, body_glow, particle_fill if they exist)

## Checkpoint

`uv run python -m src.main --config config/demo.yaml` — cycle through all 10 effects with `n` key. Each one should be visually distinct and maintain ≥15 FPS. Face should be visible in effects 1-8. Effect name should show on screen.