"""
Combines SAM2 masks with OpenCV structural analysis to produce
definitive room regions.

SAM2 provides pixel-precise region masks.
OpenCV provides structural understanding (walls, lines, geometry).
The two are fused: SAM2 masks that align with OpenCV-detected enclosed
regions are kept; masks that span across walls are split or discarded.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from .sam_segmenter import SAMSegmenter
from .opencv_processor import OpenCVProcessor


@dataclass
class RoomRegion:
    """Represents a single detected room in the blueprint."""
    mask: np.ndarray
    contour: np.ndarray
    bbox: tuple
    area: int
    centroid: tuple
    aspect_ratio: float
    solidity: float
    perimeter: float
    neighbors: List[int] = field(default_factory=list)


class RoomExtractor:
    def __init__(self, sam: SAMSegmenter, opencv: OpenCVProcessor):
        self.sam = sam
        self.opencv = opencv

    def extract(
        self, image_rgb: np.ndarray, image_bgr: np.ndarray
    ) -> List[RoomRegion]:
        """
        Full room extraction pipeline combining SAM2 and OpenCV.

        Args:
            image_rgb: RGB array for SAM2
            image_bgr: BGR array for OpenCV
        Returns:
            List[RoomRegion]
        """
        h, w = image_bgr.shape[:2]
        total_area = h * w

        # Step 1: SAM2 automatic mask generation
        sam_masks = self.sam.generate_masks(image_rgb)

        # Filter to plausible room sizes (between 1% and 60% of total area)
        min_area = int(total_area * 0.01)
        max_area = int(total_area * 0.60)
        sam_masks = self.sam.filter_room_candidates(sam_masks, min_area, max_area)

        # Step 2: OpenCV wall detection and room contours
        wall_mask, lines = self.opencv.detect_walls(image_bgr)
        opencv_contours = self.opencv.find_room_contours(wall_mask)

        # Create masks from OpenCV contours
        opencv_masks = []
        for contour in opencv_contours:
            cmask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cmask, [contour], -1, 255, -1)
            opencv_masks.append(cmask.astype(bool))

        # Step 3: Fuse SAM2 masks with OpenCV rooms
        confirmed_rooms = []
        used_opencv = set()

        for sam_entry in sam_masks:
            sam_mask = sam_entry['segmentation']

            # Check overlap with each OpenCV room
            best_iou = 0.0
            best_idx = -1
            overlapping = []

            for i, cv_mask in enumerate(opencv_masks):
                iou = self._compute_overlap_iou(sam_mask, cv_mask)
                if iou > 0.3:
                    overlapping.append(i)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou > 0.5:
                # Good match with one OpenCV room - use SAM mask (more precise)
                room = self._compute_room_properties(sam_mask, best_idx, opencv_contours)
                if room is not None:
                    confirmed_rooms.append(room)
                    used_opencv.add(best_idx)
            elif len(overlapping) > 1:
                # SAM mask spans multiple rooms - split along walls
                sub_masks = self._split_mask_by_walls(sam_mask, wall_mask)
                for sub_mask in sub_masks:
                    area = int(sub_mask.sum())
                    if area >= min_area:
                        room = self._compute_room_from_mask(sub_mask)
                        if room is not None:
                            confirmed_rooms.append(room)
                for idx in overlapping:
                    used_opencv.add(idx)

        # Step 4: Add OpenCV rooms that had no SAM match (fallback)
        for i, (cv_mask, contour) in enumerate(zip(opencv_masks, opencv_contours)):
            if i not in used_opencv:
                area = int(cv_mask.sum())
                if area >= min_area:
                    room = self._compute_room_properties(cv_mask, i, opencv_contours)
                    if room is not None:
                        confirmed_rooms.append(room)

        # Step 5: If neither SAM nor OpenCV found rooms, use SAM masks directly
        if not confirmed_rooms and sam_masks:
            for sam_entry in sam_masks:
                room = self._compute_room_from_mask(sam_entry['segmentation'])
                if room is not None:
                    confirmed_rooms.append(room)

        # Step 6: Remove duplicate/overlapping rooms
        confirmed_rooms = self._remove_duplicates(confirmed_rooms)

        # Step 7: Build adjacency graph
        self._build_adjacency(confirmed_rooms, wall_mask)

        return confirmed_rooms

    def _compute_overlap_iou(
        self, mask_a: np.ndarray, mask_b: np.ndarray
    ) -> float:
        """Intersection over Union between two binary masks."""
        # Ensure same type
        a = mask_a.astype(bool)
        b = mask_b.astype(bool)
        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union == 0:
            return 0.0
        return float(intersection) / float(union)

    def _split_mask_by_walls(
        self, mask: np.ndarray, wall_mask: np.ndarray
    ) -> List[np.ndarray]:
        """Split a SAM mask that crosses walls into sub-masks."""
        # Subtract wall pixels from the mask
        mask_uint8 = mask.astype(np.uint8) * 255
        wall_dilated = cv2.dilate(wall_mask, np.ones((3, 3), np.uint8), iterations=1)
        subtracted = cv2.bitwise_and(mask_uint8, cv2.bitwise_not(wall_dilated))

        # Find connected components
        num_labels, labels = cv2.connectedComponents(subtracted)

        sub_masks = []
        for label_id in range(1, num_labels):
            sub_mask = (labels == label_id)
            sub_masks.append(sub_mask)

        return sub_masks

    def _compute_room_properties(
        self, mask: np.ndarray, contour_idx: int, contours: List[np.ndarray]
    ) -> Optional[RoomRegion]:
        """Build a RoomRegion from a mask and its corresponding OpenCV contour."""
        mask_bool = mask.astype(bool) if mask.dtype != bool else mask
        mask_uint8 = mask_bool.astype(np.uint8) * 255

        if contour_idx >= 0 and contour_idx < len(contours):
            contour = contours[contour_idx]
        else:
            cnts, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            if not cnts:
                return None
            contour = max(cnts, key=cv2.contourArea)

        return self._build_room_region(mask_bool, contour)

    def _compute_room_from_mask(self, mask: np.ndarray) -> Optional[RoomRegion]:
        """Build a RoomRegion purely from a binary mask."""
        mask_bool = mask.astype(bool) if mask.dtype != bool else mask
        mask_uint8 = mask_bool.astype(np.uint8) * 255

        cnts, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not cnts:
            return None
        contour = max(cnts, key=cv2.contourArea)
        return self._build_room_region(mask_bool, contour)

    def _build_room_region(
        self, mask: np.ndarray, contour: np.ndarray
    ) -> Optional[RoomRegion]:
        """Compute geometric properties and build a RoomRegion."""
        area = int(mask.sum())
        if area == 0:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(min(w, h)) / float(max(w, h) + 1e-6)

        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            cx, cy = x + w // 2, y + h // 2
        else:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / float(hull_area + 1e-6)

        return RoomRegion(
            mask=mask,
            contour=contour,
            bbox=(x, y, w, h),
            area=area,
            centroid=(cx, cy),
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            perimeter=perimeter,
        )

    def _remove_duplicates(
        self, rooms: List[RoomRegion], iou_threshold: float = 0.7
    ) -> List[RoomRegion]:
        """Remove rooms that overlap too much, keeping the larger one."""
        if len(rooms) <= 1:
            return rooms

        # Sort by area descending
        rooms.sort(key=lambda r: r.area, reverse=True)
        keep = []

        for room in rooms:
            is_duplicate = False
            for kept in keep:
                iou = self._compute_overlap_iou(room.mask, kept.mask)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(room)

        return keep

    def _build_adjacency(
        self, rooms: List[RoomRegion], wall_mask: np.ndarray
    ) -> None:
        """
        Two rooms are neighbors if their dilated masks overlap
        through a gap in the wall mask.
        """
        dilate_kernel = np.ones((15, 15), np.uint8)

        for i, room_a in enumerate(rooms):
            room_a.neighbors = []
            mask_a = room_a.mask.astype(np.uint8) * 255
            dilated_a = cv2.dilate(mask_a, dilate_kernel, iterations=1)

            for j, room_b in enumerate(rooms):
                if i == j:
                    continue
                mask_b = room_b.mask.astype(np.uint8) * 255
                dilated_b = cv2.dilate(mask_b, dilate_kernel, iterations=1)

                # Check if dilated regions overlap in non-wall areas
                overlap = cv2.bitwise_and(dilated_a, dilated_b)
                non_wall_overlap = cv2.bitwise_and(
                    overlap, cv2.bitwise_not(wall_mask)
                )
                if non_wall_overlap.sum() > 100:
                    room_a.neighbors.append(j)
