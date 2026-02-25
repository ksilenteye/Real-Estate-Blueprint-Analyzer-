"""
Renders segmentation overlays: colored room masks, labels, boundaries.
"""
import cv2
import numpy as np
from typing import List

from src.classification.room_types import ClassifiedRoom, ROOM_COLORS, RoomType


class OverlayRenderer:
    def render(
        self,
        base_image: np.ndarray,
        rooms: List[ClassifiedRoom],
        alpha: float = 0.4,
        show_labels: bool = True,
        show_boundaries: bool = True,
    ) -> np.ndarray:
        """
        Draw colored semi-transparent masks over the blueprint,
        with room type labels at each room's centroid.

        Args:
            base_image: BGR (H, W, 3)
            rooms: classified rooms with masks
            alpha: transparency of color overlay
            show_labels: whether to draw room type text
            show_boundaries: whether to draw contour outlines
        Returns:
            BGR image with overlays
        """
        overlay = base_image.copy()
        label_layer = base_image.copy()

        for room in rooms:
            color = room.color
            mask = room.mask.astype(np.uint8)

            # Fill the room region with color
            colored = np.zeros_like(base_image)
            colored[:] = color
            room_region = cv2.bitwise_and(colored, colored, mask=mask)
            inv_mask = cv2.bitwise_not(mask * 255)
            background = cv2.bitwise_and(overlay, overlay, mask=inv_mask // 255)

            # Blend
            overlay = cv2.addWeighted(
                overlay, 1.0,
                room_region, alpha,
                0,
            )

            # Draw contour boundary
            if show_boundaries:
                cv2.drawContours(
                    overlay, [room.contour], -1,
                    color, thickness=2, lineType=cv2.LINE_AA,
                )

            # Draw room label
            if show_labels:
                cx, cy = room.centroid
                label = f"{room.label}"
                conf_str = f"({room.confidence:.0%})"

                # Get text size for background rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                (tw, th), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw background rectangle for readability
                padding = 4
                cv2.rectangle(
                    overlay,
                    (cx - tw // 2 - padding, cy - th - padding),
                    (cx + tw // 2 + padding, cy + padding + baseline),
                    (255, 255, 255),
                    -1,
                )
                cv2.rectangle(
                    overlay,
                    (cx - tw // 2 - padding, cy - th - padding),
                    (cx + tw // 2 + padding, cy + padding + baseline),
                    color,
                    1,
                )

                # Draw text
                cv2.putText(
                    overlay, label,
                    (cx - tw // 2, cy),
                    font, font_scale, (0, 0, 0),
                    thickness, cv2.LINE_AA,
                )

        return overlay

    def render_room_masks_only(
        self, shape: tuple, rooms: List[ClassifiedRoom]
    ) -> np.ndarray:
        """
        Render only the colored room masks without the base image.
        Useful for showing segmentation results on a white background.
        """
        canvas = np.ones(shape, dtype=np.uint8) * 255

        for room in rooms:
            color = room.color
            mask = room.mask.astype(np.uint8)
            canvas[mask > 0] = color

        return canvas
