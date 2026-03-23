"""Snowfall Custom renderer — user-uploaded images falling across the scene.

Two upload slots for custom images. Images are cleaned (face crop or
background removal) and saved to data/snowfall/ as RGBA PNGs.
"""

import math
import random
from pathlib import Path

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.effects import CONFIDENCE_MIN

_CUSTOM_DIR = Path("data/snowfall")
_MAX_FLAKES = 200


class _FallingSprite:
    __slots__ = ("x", "y", "vx", "vy", "size", "wobble_phase", "sprite_type")

    def __init__(self, x: float, y: float, sprite_type: int, size: int) -> None:
        self.x = x
        self.y = y
        self.sprite_type = sprite_type
        self.size = size
        self.vy = random.uniform(1.0, 3.0)
        self.vx = random.uniform(-0.3, 0.3)
        self.wobble_phase = random.uniform(0, math.pi * 2)

    def update(self, w: int, h: int, bass: float) -> None:
        self.wobble_phase += 0.04
        self.x += self.vx + math.sin(self.wobble_phase) * 0.4
        self.y += self.vy * (1.0 + bass * 0.6)
        if self.x < -20:
            self.x = w + 20
        elif self.x > w + 20:
            self.x = -20


class SnowfallCustomRenderer(BaseRenderer):
    """User-uploaded images falling across the scene, avoiding faces.

    Loads up to 2 custom RGBA sprites from data/snowfall/.
    Shows a placeholder message if no custom images are uploaded.
    Size controllable via snowfall_custom_scale (0.3 - 3.0, default 1.0).
    """

    def __init__(self) -> None:
        self._sprites: list[_FallingSprite] = []
        self._sprite_cache: dict[tuple[int, int], np.ndarray] = {}
        self._src_images: list[np.ndarray] = []
        self._loaded = False
        self._scale = 1.0
        self._density = 1.0
        self._custom_dir_mtime: float = 0.0

    @property
    def snowfall_custom_scale(self) -> float:
        return self._scale

    @snowfall_custom_scale.setter
    def snowfall_custom_scale(self, value: float) -> None:
        self._scale = max(0.3, min(3.0, float(value)))
        self._sprite_cache.clear()

    @property
    def snowfall_custom_density(self) -> float:
        return self._density

    @snowfall_custom_density.setter
    def snowfall_custom_density(self, value: float) -> None:
        self._density = max(0.1, min(3.0, float(value)))

    def _load_assets(self) -> None:
        self._src_images = []
        self._sprite_cache.clear()
        self._sprites.clear()

        if _CUSTOM_DIR.exists():
            images = sorted(
                p for p in _CUSTOM_DIR.iterdir()
                if p.suffix.lower() == ".png" and not p.name.startswith(".")
            )
            for path in images[:2]:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is not None and img.ndim == 3 and img.shape[2] == 4:
                    self._src_images.append(img)
            self._custom_dir_mtime = _CUSTOM_DIR.stat().st_mtime

        self._loaded = True

    def reload_custom_images(self) -> None:
        """Force reload after upload/delete."""
        self._loaded = False

    def _get_sprite(self, sprite_type: int, size: int) -> np.ndarray | None:
        if not self._src_images:
            return None
        key = (sprite_type, size)
        if key in self._sprite_cache:
            return self._sprite_cache[key]
        idx = sprite_type % len(self._src_images)
        src = self._src_images[idx]
        s = max(4, size)
        resized = cv2.resize(src, (s, s), interpolation=cv2.INTER_AREA)
        self._sprite_cache[key] = resized
        return resized

    def _make_sprite(self, w: int, h: int, y_min: float, y_max: float) -> _FallingSprite:
        x = random.uniform(0, w)
        y = random.uniform(y_min, y_max)
        num_types = max(1, len(self._src_images))
        st = random.randint(0, num_types - 1)
        base = random.randint(20, 50)
        sz = max(8, int(base * self._scale))
        return _FallingSprite(x, y, st, sz)

    def render(self, ctx: RenderContext) -> np.ndarray:
        h, w = ctx.frame.shape[:2]
        output = ctx.frame.copy()

        if not self._loaded:
            self._load_assets()

        # Auto-detect new uploads
        if _CUSTOM_DIR.exists():
            try:
                mtime = _CUSTOM_DIR.stat().st_mtime
                if mtime != self._custom_dir_mtime:
                    self._load_assets()
            except OSError:
                pass

        # No custom images — show hint text
        if not self._src_images:
            cv2.putText(output, "Upload custom images in the web app",
                        (w // 2 - 200, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (100, 100, 100), 2, cv2.LINE_AA)
            return output

        target = max(10, int(_MAX_FLAKES * self._density))
        if len(self._sprites) == 0:
            for _ in range(target):
                self._sprites.append(self._make_sprite(w, h, -h, h))
        elif len(self._sprites) < target:
            for _ in range(target - len(self._sprites)):
                self._sprites.append(self._make_sprite(w, h, -40, -5))
        elif len(self._sprites) > target:
            self._sprites = self._sprites[:target]

        bass = ctx.bass_energy

        # Head exclusion zones
        head_zones: list[tuple[float, float, float]] = []
        if ctx.pose and ctx.pose.num_people > 0:
            for person_idx in range(ctx.pose.num_people):
                kpts = ctx.pose.keypoints[person_idx]
                confs = ctx.pose.confidences[person_idx]
                pts_list = []
                for idx in [0, 1, 2, 3, 4]:
                    if confs[idx] > CONFIDENCE_MIN:
                        pts_list.append(kpts[idx])
                if len(pts_list) >= 2:
                    pts = np.array(pts_list)
                    cx = float(np.mean(pts[:, 0]))
                    cy = float(np.mean(pts[:, 1]))
                    spread = float(np.max(np.sqrt(
                        (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
                    )))
                    head_zones.append((cx, cy, max(spread * 1.5, 40.0)))

        for sprite in self._sprites:
            sprite.update(w, h, bass)

            if sprite.y > h + 20:
                new = self._make_sprite(w, h, -40, -5)
                sprite.x = new.x
                sprite.y = new.y
                sprite.vy = new.vy
                sprite.vx = new.vx
                sprite.sprite_type = new.sprite_type
                sprite.size = new.size

            in_head = False
            for cx, cy, rad in head_zones:
                if (sprite.x - cx) ** 2 + (sprite.y - cy) ** 2 < rad * rad:
                    in_head = True
                    break
            if not in_head:
                self._paste_sprite(output, sprite, w, h)

        return output

    def _paste_sprite(self, frame: np.ndarray, sprite: _FallingSprite,
                      w: int, h: int) -> None:
        s = sprite.size
        img = self._get_sprite(sprite.sprite_type, s)
        if img is None:
            return

        half = s // 2
        x1, y1 = int(sprite.x) - half, int(sprite.y) - half
        x2, y2 = x1 + s, y1 + s

        sx1 = max(0, -x1)
        sy1 = max(0, -y1)
        dx1 = max(0, x1)
        dy1 = max(0, y1)
        dx2 = min(w, x2)
        dy2 = min(h, y2)
        sx2 = sx1 + (dx2 - dx1)
        sy2 = sy1 + (dy2 - dy1)

        if dx2 <= dx1 or dy2 <= dy1 or sy2 > s or sx2 > s:
            return

        roi = img[sy1:sy2, sx1:sx2]
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            return

        alpha = roi[:, :, 3:4].astype(np.float32) / 255.0
        bgr = roi[:, :, :3]
        frame[dy1:dy2, dx1:dx2] = (
            bgr * alpha + frame[dy1:dy2, dx1:dx2] * (1.0 - alpha)
        ).astype(np.uint8)

    @property
    def name(self) -> str:
        return "Snowfall Custom"

    @property
    def needs_pose(self) -> bool:
        return True

    @property
    def needs_mask(self) -> bool:
        return False
