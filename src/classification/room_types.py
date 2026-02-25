"""
Room type definitions and classification data structures.
"""
from enum import Enum
from dataclasses import dataclass, field

import numpy as np


class RoomType(Enum):
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    DINING_ROOM = "dining_room"
    HALLWAY = "hallway"
    CORRIDOR = "corridor"
    BALCONY = "balcony"
    STORAGE = "storage"
    UNKNOWN = "unknown"


# Prompt fragments for diffusion generation per room type
ROOM_PROMPT_FRAGMENTS = {
    RoomType.LIVING_ROOM: "furnished living room with sofa, coffee table, TV unit, bookshelf, rug",
    RoomType.BEDROOM: "furnished bedroom with bed, nightstand, wardrobe, desk, lamp",
    RoomType.KITCHEN: "furnished kitchen with cabinets, countertop, stove, refrigerator, sink",
    RoomType.BATHROOM: "furnished bathroom with bathtub, toilet, sink, mirror, tiles",
    RoomType.DINING_ROOM: "furnished dining room with dining table, chairs, sideboard, chandelier",
    RoomType.HALLWAY: "decorated hallway with console table, coat rack, mirror, rug",
    RoomType.CORRIDOR: "narrow corridor with wall art, lighting",
    RoomType.BALCONY: "furnished balcony with plants, small table, chairs",
    RoomType.STORAGE: "organized storage room with shelves, boxes",
    RoomType.UNKNOWN: "furnished interior room with furniture",
}

# Display colors for segmentation overlay (BGR format)
ROOM_COLORS = {
    RoomType.LIVING_ROOM: (80, 180, 80),
    RoomType.BEDROOM: (180, 100, 80),
    RoomType.KITCHEN: (60, 160, 220),
    RoomType.BATHROOM: (200, 200, 60),
    RoomType.DINING_ROOM: (80, 80, 200),
    RoomType.HALLWAY: (160, 160, 160),
    RoomType.CORRIDOR: (120, 120, 120),
    RoomType.BALCONY: (100, 200, 100),
    RoomType.STORAGE: (140, 140, 100),
    RoomType.UNKNOWN: (100, 100, 100),
}


@dataclass
class ClassifiedRoom:
    """A room region with its classification result."""
    room_type: RoomType
    confidence: float
    area: int
    area_ratio: float
    bbox: tuple
    centroid: tuple
    mask: np.ndarray
    contour: np.ndarray
    aspect_ratio: float = 0.0
    neighbors: list = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.room_type.value.replace("_", " ").title()

    @property
    def furnishing_prompt_fragment(self) -> str:
        return ROOM_PROMPT_FRAGMENTS.get(
            self.room_type, ROOM_PROMPT_FRAGMENTS[RoomType.UNKNOWN]
        )

    @property
    def color(self) -> tuple:
        return ROOM_COLORS.get(self.room_type, ROOM_COLORS[RoomType.UNKNOWN])
