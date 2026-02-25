"""
Orchestrates the ControlNet + Stable Diffusion generation process.
Generates a furnished version of the blueprint while preserving
wall structure through canny edge conditioning.
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Optional

from .model_loader import ModelLoader
from .prompt_builder import PromptBuilder
from src.classification.room_types import ClassifiedRoom
from config import DiffusionConfig


class FurnishingPipeline:
    def __init__(
        self,
        model_loader: ModelLoader,
        prompt_builder: PromptBuilder,
        config: DiffusionConfig,
    ):
        self.model_loader = model_loader
        self.prompt_builder = prompt_builder
        self.config = config

    def generate_full_plan(
        self,
        canny_image: Image.Image,
        rooms: List[ClassifiedRoom],
        style: str,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate a single furnished image for the entire floorplan.

        The canny edge map preserves all wall, door, and window positions.
        The prompt describes the overall interior style.

        Args:
            canny_image: 3-channel canny edge PIL image
            rooms: classified rooms (used for prompt construction)
            style: furnishing style key
            seed: optional random seed for reproducibility
        Returns:
            PIL.Image of the furnished floorplan
        """
        pipe = self.model_loader.load_pipeline()
        prompt, negative_prompt = self.prompt_builder.build_full_plan_prompt(
            rooms, style
        )

        # Resize canny image to match generation dimensions
        canny_resized = canny_image.resize(
            (self.config.generation_width, self.config.generation_height),
            Image.LANCZOS,
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        result = pipe(
            prompt=prompt,
            image=canny_resized,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            controlnet_conditioning_scale=self.config.controlnet_conditioning_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            width=self.config.generation_width,
            height=self.config.generation_height,
        )

        return result.images[0]

    def generate_per_room(
        self,
        original_image: np.ndarray,
        canny_full: np.ndarray,
        rooms: List[ClassifiedRoom],
        style: str,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate furnished output room-by-room, then composite back.

        Each room gets a room-type-specific prompt for more accurate
        furnishing. The results are blended back onto the original layout.
        """
        pipe = self.model_loader.load_pipeline()
        h, w = original_image.shape[:2]
        result_canvas = np.array(
            Image.fromarray(original_image).resize(
                (self.config.generation_width, self.config.generation_height)
            )
        )

        scale_x = self.config.generation_width / w
        scale_y = self.config.generation_height / h

        for room in rooms:
            prompt, negative_prompt = self.prompt_builder.build_room_prompt(
                room, style
            )

            # Crop canny region for this room
            x, y, rw, rh = room.bbox
            sx = int(x * scale_x)
            sy = int(y * scale_y)
            sw = max(int(rw * scale_x), 64)
            sh = max(int(rh * scale_y), 64)

            # Ensure dimensions are multiples of 8
            sw = (sw // 8) * 8 or 64
            sh = (sh // 8) * 8 or 64

            # Crop canny edges for this room
            canny_crop = canny_full[sy:sy + sh, sx:sx + sw]
            if canny_crop.size == 0:
                continue
            canny_pil = Image.fromarray(canny_crop).resize((sw, sh), Image.LANCZOS)

            generator = None
            if seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(seed)

            room_result = pipe(
                prompt=prompt,
                image=canny_pil,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                controlnet_conditioning_scale=self.config.controlnet_conditioning_scale,
                negative_prompt=negative_prompt,
                generator=generator,
                width=sw,
                height=sh,
            )

            # Paste room result back into canvas
            room_img = np.array(room_result.images[0])
            ey = min(sy + sh, result_canvas.shape[0])
            ex = min(sx + sw, result_canvas.shape[1])
            room_img = room_img[:ey - sy, :ex - sx]
            result_canvas[sy:ey, sx:ex] = room_img

        return Image.fromarray(result_canvas)
