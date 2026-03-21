"""Shadow clones renderer — time-delayed colored silhouette copies."""

from collections import deque

import cv2
import numpy as np

from src.render.base import BaseRenderer, RenderContext
from src.render.utils import composite_head

# Clone colors in BGR
_CLONE_COLORS = [
    (200, 50, 180),   # purple
    (200, 200, 0),    # cyan
    (200, 0, 200),    # magenta
    (0, 200, 230),    # gold
]

_HISTORY_LEN = 12
_NUM_CLONES = 4


class ShadowClonesRenderer(BaseRenderer):
    """Time-delayed colored silhouette copies fanning out behind the body.

    3-4 clones show the pose from previous frames, each tinted a
    distinct color and offset horizontally. Real body in front with
    real face preserved.
    """

    def __init__(self) -> None:
        self._history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=_HISTORY_LEN)
        self._scatter_frames = 0
        self._scatter_offsets: list[int] = [0] * _NUM_CLONES

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render shadow clones effect.

        Args:
            ctx: Current frame data with mask and pose.

        Returns:
            Frame with colored shadow clones behind real body.
        """
        h, w = ctx.frame.shape[:2]
        original = ctx.frame.copy()
        output = ctx.frame.copy()

        bass = ctx.bass_energy

        # Store current frame + mask
        if ctx.mask and ctx.mask.num_people > 0:
            self._history.append((ctx.mask.combined_mask.copy(), ctx.frame.copy()))
        else:
            self._history.append((np.zeros((h, w), dtype=np.uint8), ctx.frame.copy()))

        if len(self._history) < 4:
            return composite_head(output, original, ctx.pose, ctx.frame.shape)

        # Bass scatter: on heavy hits, clones jump outward then snap back
        if bass > 0.8 and self._scatter_frames <= 0:
            self._scatter_frames = 5
            for i in range(_NUM_CLONES):
                direction = -1 if i % 2 == 0 else 1
                self._scatter_offsets[i] = direction * int(30 + bass * 20)

        if self._scatter_frames > 0:
            self._scatter_frames -= 1
            if self._scatter_frames == 0:
                self._scatter_offsets = [0] * _NUM_CLONES

        # Pick evenly spaced historical frames for clones
        history_list = list(self._history)
        indices = np.linspace(0, len(history_list) - 2, _NUM_CLONES, dtype=int)

        # Base horizontal spread: clones fan out from center
        base_spread = int(15 + bass * 25)

        # Draw clones back-to-front (oldest/furthest first)
        for clone_idx, hist_idx in enumerate(indices):
            hist_mask, hist_frame = history_list[hist_idx]
            if not np.any(hist_mask):
                continue

            color = _CLONE_COLORS[clone_idx % len(_CLONE_COLORS)]
            # Spread direction alternates left/right
            direction = -1 if clone_idx % 2 == 0 else 1
            offset_x = direction * base_spread * (clone_idx // 2 + 1)
            offset_x += self._scatter_offsets[clone_idx]

            # Create shifted mask
            shifted_mask = np.zeros_like(hist_mask)
            if offset_x > 0 and offset_x < w:
                shifted_mask[:, offset_x:] = hist_mask[:, :w - offset_x]
            elif offset_x < 0 and abs(offset_x) < w:
                shifted_mask[:, :w + offset_x] = hist_mask[:, -offset_x:]
            else:
                shifted_mask = hist_mask

            if not np.any(shifted_mask):
                continue

            # Tint the historical frame with clone color
            opacity = 0.35 + bass * 0.15
            clone_tint = np.zeros((h, w, 3), dtype=np.uint8)
            clone_tint[shifted_mask > 0] = color

            # Blend: tinted overlay on the output
            mask_bool = shifted_mask > 0
            blended = cv2.addWeighted(
                output, 1.0 - opacity,
                np.where(mask_bool[:, :, np.newaxis], clone_tint, output), opacity,
                0,
            )
            output[mask_bool] = blended[mask_bool]

        # Real body on top (current frame pixels where current mask is)
        if ctx.mask and ctx.mask.num_people > 0:
            current_mask = ctx.mask.combined_mask > 0
            output[current_mask] = original[current_mask]

        return composite_head(output, original, ctx.pose, ctx.frame.shape)

    @property
    def name(self) -> str:
        return "Shadow Clones"

    @property
    def needs_mask(self) -> bool:
        return True

    @property
    def needs_pose(self) -> bool:
        return True
