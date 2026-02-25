"""
Heuristic room classifier based on geometric properties
and positional relationships in the floorplan.
"""
from typing import List, Tuple
from collections import Counter

from src.segmentation.room_extractor import RoomRegion
from .room_types import RoomType, ClassifiedRoom
from config import ClassificationConfig


class RoomClassifier:
    def __init__(self, config: ClassificationConfig):
        self.config = config

    def classify_all(
        self, rooms: List[RoomRegion], total_plan_area: int
    ) -> List[ClassifiedRoom]:
        """
        Classify all detected rooms using heuristic rules.

        Priority order:
        1. Corridors (very elongated)
        2. Bathrooms (small, near-square)
        3. Kitchen (medium, typically one per plan)
        4. Living room (largest remaining)
        5. Bedrooms (medium-large, roughly square)
        6. Dining room (medium, adjacent to kitchen/living)
        7. Fallback: UNKNOWN
        """
        if not rooms:
            return []

        # Compute plan bounding box and centroid
        all_centroids = [r.centroid for r in rooms]
        plan_cx = sum(c[0] for c in all_centroids) / len(all_centroids)
        plan_cy = sum(c[1] for c in all_centroids) / len(all_centroids)
        plan_centroid = (plan_cx, plan_cy)

        # First pass: classify each room independently
        classified = []
        for room in rooms:
            area_ratio = room.area / total_plan_area
            room_type, confidence = self._classify_single(
                room, total_plan_area, plan_centroid
            )
            classified.append(ClassifiedRoom(
                room_type=room_type,
                confidence=confidence,
                area=room.area,
                area_ratio=area_ratio,
                bbox=room.bbox,
                centroid=room.centroid,
                mask=room.mask,
                contour=room.contour,
                aspect_ratio=room.aspect_ratio,
                neighbors=room.neighbors,
            ))

        # Second pass: ensure at least one living room exists
        classified = self._ensure_living_room(classified)

        # Third pass: resolve duplicate types where unlikely
        classified = self._resolve_duplicates(classified)

        return classified

    def _classify_single(
        self, room: RoomRegion, total_area: int, plan_centroid: tuple
    ) -> Tuple[RoomType, float]:
        """Classify a single room by geometric heuristics."""
        area_ratio = room.area / total_area

        # --- Corridor: very elongated ---
        if room.aspect_ratio < self.config.corridor_max_aspect_ratio:
            return RoomType.CORRIDOR, 0.85

        # --- Bathroom: small and roughly square ---
        if area_ratio < self.config.bathroom_max_area_ratio:
            if room.aspect_ratio > 0.5:  # Not too elongated
                return RoomType.BATHROOM, 0.75
            else:
                return RoomType.STORAGE, 0.60

        # --- Kitchen: medium area ---
        if (self.config.kitchen_min_area_ratio <= area_ratio
                <= self.config.kitchen_max_area_ratio):
            # Kitchen tends to have moderate solidity
            if room.solidity > 0.7:
                return RoomType.KITCHEN, 0.65

        # --- Large rooms ---
        if area_ratio >= self.config.living_room_min_area_ratio:
            # Large and central = likely living room
            dist_to_center = self._distance_to_center(
                room.centroid, plan_centroid
            )
            if dist_to_center < 0.3:
                return RoomType.LIVING_ROOM, 0.80
            else:
                return RoomType.BEDROOM, 0.60

        # --- Medium rooms ---
        if area_ratio >= self.config.bedroom_min_area_ratio:
            if room.aspect_ratio > 0.6:
                return RoomType.BEDROOM, 0.70
            else:
                return RoomType.HALLWAY, 0.55

        # --- Small remaining ---
        if area_ratio < self.config.kitchen_min_area_ratio:
            return RoomType.STORAGE, 0.50

        return RoomType.UNKNOWN, 0.30

    def _distance_to_center(
        self, room_centroid: tuple, plan_centroid: tuple
    ) -> float:
        """Normalized distance from room centroid to plan centroid (0-1)."""
        dx = room_centroid[0] - plan_centroid[0]
        dy = room_centroid[1] - plan_centroid[1]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        # Normalize by approximate plan size
        max_dist = max(abs(plan_centroid[0]), abs(plan_centroid[1]), 1)
        return min(dist / max_dist, 1.0)

    def _ensure_living_room(
        self, rooms: List[ClassifiedRoom]
    ) -> List[ClassifiedRoom]:
        """If no living room was classified, promote the largest non-kitchen room."""
        types = [r.room_type for r in rooms]
        if RoomType.LIVING_ROOM in types:
            return rooms

        # Find the largest room that isn't a corridor or bathroom
        candidates = [
            (i, r) for i, r in enumerate(rooms)
            if r.room_type not in (RoomType.CORRIDOR, RoomType.BATHROOM, RoomType.STORAGE)
        ]
        if candidates:
            candidates.sort(key=lambda x: x[1].area, reverse=True)
            idx = candidates[0][0]
            rooms[idx].room_type = RoomType.LIVING_ROOM
            rooms[idx].confidence = 0.65

        return rooms

    def _resolve_duplicates(
        self, rooms: List[ClassifiedRoom]
    ) -> List[ClassifiedRoom]:
        """
        Resolve unlikely duplicates:
        - Max 1 kitchen, max 1 living room
        - Convert extras to dining room or bedroom
        """
        type_counts = Counter(r.room_type for r in rooms)

        # Too many kitchens - keep largest, convert rest to dining room
        if type_counts.get(RoomType.KITCHEN, 0) > 1:
            kitchens = [(i, r) for i, r in enumerate(rooms)
                        if r.room_type == RoomType.KITCHEN]
            kitchens.sort(key=lambda x: x[1].area, reverse=True)
            for idx, room in kitchens[1:]:
                rooms[idx].room_type = RoomType.DINING_ROOM
                rooms[idx].confidence *= 0.8

        # Too many living rooms - keep largest, convert rest to bedroom
        if type_counts.get(RoomType.LIVING_ROOM, 0) > 1:
            livings = [(i, r) for i, r in enumerate(rooms)
                       if r.room_type == RoomType.LIVING_ROOM]
            livings.sort(key=lambda x: x[1].area, reverse=True)
            for idx, room in livings[1:]:
                rooms[idx].room_type = RoomType.BEDROOM
                rooms[idx].confidence *= 0.8

        return rooms
