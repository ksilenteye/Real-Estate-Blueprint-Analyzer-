"""
Constructs text prompts for ControlNet-conditioned generation
based on room type and user-selected furnishing style.
"""
from typing import List, Tuple
from collections import Counter

from src.classification.room_types import ClassifiedRoom, RoomType
from config import StyleConfig, DiffusionConfig


class PromptBuilder:
    def __init__(self, style_config: StyleConfig, diffusion_config: DiffusionConfig):
        self.styles = style_config.styles
        self.negative_prompt = diffusion_config.negative_prompt

    def build_full_plan_prompt(
        self, rooms: List[ClassifiedRoom], style: str
    ) -> Tuple[str, str]:
        """
        Build a single prompt for the entire floorplan.
        Combines dominant room types into one cohesive prompt.
        """
        style_desc = self.styles.get(style, self.styles["modern"])

        if not rooms:
            prompt = (
                f"top-down architectural floorplan view of a furnished house interior, "
                f"{style_desc}, professional interior design, "
                f"high quality, detailed, 4k, photorealistic"
            )
            return prompt, self.negative_prompt

        # Count room types for prompt construction
        type_counts = Counter(r.room_type for r in rooms)
        room_descriptions = []

        for room_type, count in type_counts.most_common():
            if room_type == RoomType.UNKNOWN:
                continue
            fragment = rooms[0].furnishing_prompt_fragment
            # Find the actual fragment for this type
            for r in rooms:
                if r.room_type == room_type:
                    fragment = r.furnishing_prompt_fragment
                    break
            if count > 1:
                room_descriptions.append(f"{count} {fragment}s")
            else:
                room_descriptions.append(fragment)

        rooms_text = ", ".join(room_descriptions[:4])  # Limit prompt length

        prompt = (
            f"top-down architectural floorplan view of a furnished house with "
            f"{rooms_text}, {style_desc}, "
            f"professional interior design, high quality, detailed, 4k, "
            f"photorealistic floorplan rendering, clean layout"
        )

        return prompt, self.negative_prompt

    def build_room_prompt(
        self, room: ClassifiedRoom, style: str
    ) -> Tuple[str, str]:
        """
        Build a prompt for a single room crop.
        """
        style_desc = self.styles.get(style, self.styles["modern"])
        room_desc = room.furnishing_prompt_fragment

        prompt = (
            f"top-down architectural view of a {room_desc}, "
            f"{style_desc}, professional interior design, "
            f"high quality, detailed, 4k, photorealistic floorplan view"
        )

        return prompt, self.negative_prompt
