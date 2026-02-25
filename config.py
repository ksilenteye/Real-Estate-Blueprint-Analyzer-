"""
Central configuration for the Blueprint-to-Furnished application.
All tunable parameters, model paths, and defaults live here.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).parent


@dataclass
class PreprocessingConfig:
    max_dimension: int = 1024
    denoise_strength: int = 10
    grayscale_threshold: float = 0.15
    border_crop_px: int = 5


@dataclass
class SAMConfig:
    hf_model_id: str = "facebook/sam2.1-hiera-large"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    min_mask_region_area: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class OpenCVConfig:
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 80
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10
    morph_kernel_size: int = 3
    wall_thickness_range: tuple = (3, 30)
    contour_min_area: int = 2000


@dataclass
class ClassificationConfig:
    bathroom_max_area_ratio: float = 0.08
    kitchen_min_area_ratio: float = 0.08
    kitchen_max_area_ratio: float = 0.18
    bedroom_min_area_ratio: float = 0.12
    living_room_min_area_ratio: float = 0.18
    corridor_max_aspect_ratio: float = 0.35
    bathroom_max_aspect_ratio: float = 1.8


@dataclass
class DiffusionConfig:
    sd_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    controlnet_model_id: str = "lllyasviel/sd-controlnet-canny"
    torch_dtype: str = "float16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    negative_prompt: str = (
        "blurry, low quality, distorted, deformed walls, "
        "broken furniture, unrealistic, watermark, text"
    )
    enable_cpu_offload: bool = True
    enable_vae_slicing: bool = True
    generation_width: int = 512
    generation_height: int = 512
    seed: Optional[int] = None

    @property
    def dtype(self) -> torch.dtype:
        return torch.float16 if self.torch_dtype == "float16" else torch.float32


@dataclass
class StyleConfig:
    styles: dict = field(default_factory=lambda: {
        "modern": "modern interior design, clean lines, minimalist furniture, neutral colors, contemporary",
        "minimalist": "minimalist interior, sparse furniture, white walls, zen aesthetic, simple elegant",
        "traditional": "traditional interior design, classic furniture, warm wood tones, ornate details, cozy",
        "scandinavian": "scandinavian interior, light wood, hygge, bright airy space, functional furniture",
        "industrial": "industrial interior design, exposed brick, metal fixtures, concrete, loft style",
    })


@dataclass
class AppConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    opencv: OpenCVConfig = field(default_factory=OpenCVConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
