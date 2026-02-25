"""
Manages loading and unloading of Stable Diffusion + ControlNet models.
Implements lazy loading and GPU memory management.

Memory strategy:
- SAM2 and diffusion models are never loaded simultaneously
- enable_model_cpu_offload() moves submodels to GPU only when needed
- enable_vae_slicing() reduces peak memory during decode
"""
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from config import DiffusionConfig


class ModelLoader:
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self._pipe = None

    def load_pipeline(self) -> StableDiffusionControlNetPipeline:
        """
        Load ControlNet + SD pipeline with memory optimizations.
        Returns cached instance if already loaded.
        """
        if self._pipe is not None:
            return self._pipe

        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet_model_id,
            torch_dtype=self.config.dtype,
        )

        self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.sd_model_id,
            controlnet=controlnet,
            torch_dtype=self.config.dtype,
            safety_checker=None,
        )

        # Use fast UniPC scheduler (fewer steps needed)
        self._pipe.scheduler = UniPCMultistepScheduler.from_config(
            self._pipe.scheduler.config
        )

        # Memory management
        if self.config.enable_cpu_offload and self.config.device == "cuda":
            self._pipe.enable_model_cpu_offload()
        elif self.config.device == "cuda":
            self._pipe = self._pipe.to(self.config.device)
        # For CPU, no special handling needed - it's already there

        if self.config.enable_vae_slicing:
            self._pipe.enable_vae_slicing()

        return self._pipe

    def unload_pipeline(self) -> None:
        """Free all GPU memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None
