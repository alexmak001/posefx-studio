"""Angel & Devil renderer — randomly assigns halo+wings or horns+pitchfork per person."""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.effects import CONFIDENCE_MIN

_ASSETS_DIR = Path("data/assets")
_ASSIGN_DELAY = 1.0  # seconds before assigning angel/devil to a new face


class _WingParticle:
    __slots__ = ("x", "y", "vy", "life", "max_life", "color")

    def __init__(self, x: float, y: float, color: tuple[int, int, int]) -> None:
        self.x = x + random.uniform(-3, 3)
        self.y = y
        self.vy = random.uniform(1.0, 3.0)
        self.max_life = random.randint(15, 25)
        self.life = self.max_life
        self.color = color

    def update(self) -> bool:
        self.y += self.vy
        self.life -= 1
        return self.life > 0


class _PersonTrack:
    __slots__ = ("first_seen", "assigned", "is_angel", "center_x", "center_y")

    def __init__(self, cx: float, cy: float) -> None:
        self.first_seen = time.monotonic()
        self.assigned = False
        self.is_angel = True
        self.center_x = cx
        self.center_y = cy


def _load_asset(name: str) -> np.ndarray | None:
    path = _ASSETS_DIR / name
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img


def _paste_rgba(frame: np.ndarray, sprite: np.ndarray,
                cx: int, cy: int, target_w: int, target_h: int) -> None:
    """Alpha-composite an RGBA sprite centered at (cx, cy) onto frame."""
    if sprite is None or sprite.ndim < 3 or sprite.shape[2] != 4:
        return
    h, w = frame.shape[:2]
    resized = cv2.resize(sprite, (max(1, target_w), max(1, target_h)),
                         interpolation=cv2.INTER_AREA)

    x1 = cx - target_w // 2
    y1 = cy - target_h // 2
    x2 = x1 + target_w
    y2 = y1 + target_h

    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    dx1 = max(0, x1)
    dy1 = max(0, y1)
    dx2 = min(w, x2)
    dy2 = min(h, y2)
    sx2 = sx1 + (dx2 - dx1)
    sy2 = sy1 + (dy2 - dy1)

    if dx2 <= dx1 or dy2 <= dy1:
        return
    if sy2 > resized.shape[0] or sx2 > resized.shape[1]:
        return

    roi = resized[sy1:sy2, sx1:sx2]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return

    alpha = roi[:, :, 3:4].astype(np.float32) / 255.0
    bgr = roi[:, :, :3]
    frame[dy1:dy2, dx1:dx2] = (
        bgr * alpha + frame[dy1:dy2, dx1:dx2] * (1.0 - alpha)
    ).astype(np.uint8)


class HaloWingsRenderer(BaseRenderer):
    """Randomly assigns Angel (halo + wings) or Devil (horns + pitchfork)."""

    def __init__(self) -> None:
        self._particles: list[_WingParticle] = []
        self._tracks: list[_PersonTrack | None] = []
        self._horns_img: np.ndarray | None = None
        self._pitchfork_img: np.ndarray | None = None
        self._loaded = False

    def _load_assets(self) -> None:
        self._horns_img = _load_asset("devil_horns.png")
        self._pitchfork_img = _load_asset("devil_pitchfork.png")
        self._loaded = True

    def render(self, ctx: RenderContext) -> np.ndarray:
        h, w = ctx.frame.shape[:2]
        output = ctx.frame.copy()

        if not self._loaded:
            self._load_assets()

        if not ctx.pose or ctx.pose.num_people == 0:
            self._tracks.clear()
            self._draw_particles(output)
            return output

        bass = ctx.bass_energy
        now = time.monotonic()
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Build current head centers
        current_centers: list[tuple[float, float, float] | None] = []
        for person_idx in range(ctx.pose.num_people):
            kpts = ctx.pose.keypoints[person_idx]
            confs = ctx.pose.confidences[person_idx]
            info = self._get_head_info(kpts, confs)
            current_centers.append(info)

        # Match to existing tracks
        new_tracks: list[_PersonTrack | None] = []
        used_old: set[int] = set()
        for info in current_centers:
            if info is None:
                new_tracks.append(None)
                continue
            cx, cy, _ = info
            best_idx = -1
            best_dist = 150.0
            for i, t in enumerate(self._tracks):
                if i in used_old or t is None:
                    continue
                d = math.sqrt((t.center_x - cx) ** 2 + (t.center_y - cy) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx >= 0:
                track = self._tracks[best_idx]
                track.center_x = cx
                track.center_y = cy
                used_old.add(best_idx)
                new_tracks.append(track)
            else:
                new_tracks.append(_PersonTrack(cx, cy))
        self._tracks = new_tracks

        # Assign after delay
        for track in self._tracks:
            if track is None:
                continue
            if not track.assigned and (now - track.first_seen) >= _ASSIGN_DELAY:
                track.assigned = True
                track.is_angel = random.random() > 0.5

        # Draw effects
        for person_idx in range(ctx.pose.num_people):
            if person_idx >= len(self._tracks) or self._tracks[person_idx] is None:
                continue
            track = self._tracks[person_idx]
            if not track.assigned:
                continue

            kpts = ctx.pose.keypoints[person_idx]
            confs = ctx.pose.confidences[person_idx]
            info = current_centers[person_idx]
            if info is None:
                continue

            if track.is_angel:
                self._draw_halo(overlay, kpts, confs, bass, info, h, w)
                self._draw_wings(overlay, kpts, confs, bass, h, w)
            else:
                self._draw_horns_sprite(output, info, bass)
                self._draw_pitchfork_sprite(output, kpts, confs, info, bass, h, w)

        # Glow pass for drawn elements (halo, wings)
        glow_size = int(15 + bass * 10)
        if glow_size % 2 == 0:
            glow_size += 1
        glow = cv2.GaussianBlur(overlay, (glow_size, glow_size), 0)
        output = cv2.add(output, glow)
        output = cv2.add(output, overlay)

        self._draw_particles(output)
        return output

    def _get_head_info(self, kpts: np.ndarray, confs: np.ndarray
                       ) -> tuple[float, float, float] | None:
        head_pts = []
        for idx in [0, 1, 2, 3, 4]:
            if confs[idx] > CONFIDENCE_MIN:
                head_pts.append(kpts[idx])
        if len(head_pts) < 2:
            return None
        pts = np.array(head_pts)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        spread = float(np.max(np.sqrt(
            (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
        )))
        radius = max(spread * 1.2, 25.0)
        return cx, cy, radius

    # ---- Angel ----

    def _draw_halo(self, layer: np.ndarray, kpts: np.ndarray,
                   confs: np.ndarray, bass: float,
                   head_info: tuple[float, float, float],
                   h: int, w: int) -> None:
        head_x, head_y, head_radius = head_info
        halo_y = int(head_y - head_radius * 1.5)
        halo_x = int(head_x)
        halo_w = int(head_radius * (1.0 + bass * 0.2))
        halo_h = int(head_radius * 0.35)
        brightness = int(180 + bass * 75)
        halo_color = (0, brightness, brightness)
        if 0 < halo_y < h and 0 < halo_x < w:
            cv2.ellipse(layer, (halo_x, halo_y), (halo_w, halo_h),
                        0, 0, 360, halo_color, 3, cv2.LINE_AA)
            cv2.ellipse(layer, (halo_x, halo_y + 2), (halo_w - 2, halo_h - 1),
                        0, 160, 380, halo_color, 2, cv2.LINE_AA)

    def _draw_wings(self, layer: np.ndarray, kpts: np.ndarray,
                    confs: np.ndarray, bass: float, h: int, w: int) -> None:
        for side, sh_idx in [("left", 5), ("right", 6)]:
            if confs[sh_idx] < CONFIDENCE_MIN:
                continue
            sx, sy = float(kpts[sh_idx, 0]), float(kpts[sh_idx, 1])
            wing_dir = -1 if side == "left" else 1
            spread = 1.0 + bass * 0.2
            wing_color = (180, 220, 255)
            for f in range(4):
                angle_offset = (f - 2) * 0.25 * spread
                base_angle = -math.pi / 2 + wing_dir * 0.8 + angle_offset
                length = int((80 + f * 20) * spread)
                points = []
                for t_val in np.linspace(0, 1, 12):
                    px = sx + wing_dir * t_val * length * math.cos(base_angle + t_val * 0.5)
                    py = sy + t_val * length * math.sin(base_angle + t_val * 0.5) * 0.6
                    points.append([int(px), int(py)])
                if len(points) > 2:
                    pts_arr = np.array(points, dtype=np.int32)
                    alpha = 0.6 - f * 0.1
                    feather_color = tuple(int(c * alpha) for c in wing_color)
                    cv2.polylines(layer, [pts_arr], False, feather_color, 2, cv2.LINE_AA)
                    if bass > 0.7 and f == 3 and points:
                        tip = points[-1]
                        if 0 <= tip[0] < w and 0 <= tip[1] < h:
                            for _ in range(2):
                                self._particles.append(
                                    _WingParticle(float(tip[0]), float(tip[1]), (0, 200, 200)))
        if len(self._particles) > 80:
            self._particles = self._particles[-80:]

    # ---- Devil ----

    def _draw_horns_sprite(self, frame: np.ndarray,
                           head_info: tuple[float, float, float],
                           bass: float) -> None:
        if self._horns_img is None:
            return
        head_x, head_y, head_radius = head_info
        # Horns sit on top of the head, centered
        horn_w = int(head_radius * 2.2 * (1.0 + bass * 0.15))
        horn_h = int(head_radius * 1.4 * (1.0 + bass * 0.15))
        horn_cx = int(head_x)
        horn_cy = int(head_y - head_radius * 0.9)
        _paste_rgba(frame, self._horns_img, horn_cx, horn_cy, horn_w, horn_h)

    def _draw_pitchfork_sprite(self, frame: np.ndarray, kpts: np.ndarray,
                               confs: np.ndarray,
                               head_info: tuple[float, float, float],
                               bass: float, h: int, w: int) -> None:
        if self._pitchfork_img is None:
            return
        # Use right wrist (10), fall back to right elbow (8), then right shoulder (6)
        hand_idx = -1
        for idx in [10, 8, 6]:
            if confs[idx] > CONFIDENCE_MIN:
                hand_idx = idx
                break
        if hand_idx < 0:
            return
        hx = int(kpts[hand_idx, 0])
        hy = int(kpts[hand_idx, 1])
        _, _, head_radius = head_info
        fork_h = int(head_radius * 1.8 * (1.0 + bass * 0.1))
        fork_w = int(fork_h * self._pitchfork_img.shape[1] / self._pitchfork_img.shape[0])
        _paste_rgba(frame, self._pitchfork_img, hx, hy, fork_w, fork_h)

        # Spark particles from pitchfork on bass
        if bass > 0.5:
            self._particles.append(
                _WingParticle(float(hx), float(hy - fork_h // 3), (0, 50, 255)))

    def _draw_particles(self, frame: np.ndarray) -> None:
        alive = []
        for p in self._particles:
            if p.update():
                alpha = p.life / p.max_life
                color = tuple(int(c * alpha) for c in p.color)
                cv2.circle(frame, (int(p.x), int(p.y)), 2, color, -1)
                alive.append(p)
        self._particles = alive

    @property
    def name(self) -> str:
        return "Angel & Devil"

    @property
    def needs_pose(self) -> bool:
        return True
