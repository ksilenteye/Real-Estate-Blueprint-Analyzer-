"""
Wraps SAM2 automatic mask generation for blueprint segmentation.
Produces masks representing potential rooms and spatial regions.
"""
import numpy as np
import torch
from typing import List, Dict, Any

from config import SAMConfig


class SAMSegmenter:
    def __init__(self, config: SAMConfig):
        self.config = config
        self._predictor = None
        self._mask_generator = None

    def load_model(self) -> None:
        """Lazy-load SAM2 model from HuggingFace. Called once, then cached."""
        if self._mask_generator is not None:
            return

        try:
            from sam2.build_sam import build_sam2_hf
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            model = build_sam2_hf(self.config.hf_model_id)

            self._mask_generator = SAM2AutomaticMaskGenerator(
                model=model,
                points_per_side=self.config.points_per_side,
                pred_iou_thresh=self.config.pred_iou_thresh,
                stability_score_thresh=self.config.stability_score_thresh,
                min_mask_region_area=self.config.min_mask_region_area,
            )
        except ImportError:
            # Fallback: try the newer SAM2 API via transformers
            from transformers import SAM2AutomaticMaskGenerator as TFSAM2AMG
            from transformers import SAM2Model, SAM2Processor

            model = SAM2Model.from_pretrained(self.config.hf_model_id)
            processor = SAM2Processor.from_pretrained(self.config.hf_model_id)

            self._mask_generator = None
            self._model = model.to(self.config.device)
            self._processor = processor
            self._use_transformers = True
            return

        self._use_transformers = False

    def generate_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run automatic mask generation on a blueprint image.

        Args:
            image: RGB numpy array (H, W, 3)
        Returns:
            List of dicts with keys:
              - 'segmentation': (H, W) boolean mask
              - 'area': int
              - 'bbox': [x, y, w, h]
              - 'predicted_iou': float
              - 'stability_score': float
        """
        self.load_model()

        if hasattr(self, '_use_transformers') and self._use_transformers:
            return self._generate_with_transformers(image)

        return self._mask_generator.generate(image)

    def _generate_with_transformers(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback generation using the transformers library SAM2 API."""
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        masks = self._processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )

        results = []
        if len(masks) > 0:
            mask_tensor = masks[0]  # first image
            scores = outputs.iou_scores[0] if hasattr(outputs, 'iou_scores') else None

            for i in range(mask_tensor.shape[0]):
                mask = mask_tensor[i, 0].cpu().numpy().astype(bool)
                area = int(mask.sum())
                if area < self.config.min_mask_region_area:
                    continue

                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                bbox = [int(xs.min()), int(ys.min()),
                        int(xs.max() - xs.min()), int(ys.max() - ys.min())]

                results.append({
                    'segmentation': mask,
                    'area': area,
                    'bbox': bbox,
                    'predicted_iou': float(scores[i].max()) if scores is not None else 0.9,
                    'stability_score': 0.9,
                })

        return results

    def filter_room_candidates(
        self, masks: List[Dict[str, Any]], min_area: int, max_area: int
    ) -> List[Dict[str, Any]]:
        """
        Filter masks to plausible room-sized regions.
        Removes tiny fragments and oversized background masks.
        """
        return [
            m for m in masks
            if min_area <= m['area'] <= max_area
        ]

    def unload_model(self) -> None:
        """Free GPU memory by deleting the model."""
        if self._mask_generator is not None:
            del self._mask_generator
            self._mask_generator = None
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._model = None
        if hasattr(self, '_processor') and self._processor is not None:
            del self._processor
            self._processor = None
        torch.cuda.empty_cache()
