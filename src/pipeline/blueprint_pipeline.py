"""
End-to-end orchestrator that ties all stages together.
This is the single callable that the Gradio UI invokes.

Memory management:
1. Load SAM2, run segmentation, unload SAM2
2. Load SD + ControlNet, run generation (stays loaded for re-runs)
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass

from config import AppConfig
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.edge_detector import EdgeDetector
from src.segmentation.sam_segmenter import SAMSegmenter
from src.segmentation.opencv_processor import OpenCVProcessor
from src.segmentation.room_extractor import RoomExtractor
from src.classification.room_classifier import RoomClassifier
from src.classification.room_types import ClassifiedRoom
from src.diffusion.model_loader import ModelLoader
from src.diffusion.prompt_builder import PromptBuilder
from src.diffusion.furnishing_pipeline import FurnishingPipeline
from src.visualization.overlay_renderer import OverlayRenderer
from src.visualization.result_compositor import ResultCompositor


@dataclass
class PipelineResult:
    original: Image.Image
    segmentation_overlay: Image.Image
    furnished: Image.Image
    rooms: List[ClassifiedRoom]
    room_summary: str


class BlueprintPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.preprocessor = ImagePreprocessor(config.preprocessing)
        self.edge_detector = EdgeDetector(config.opencv)
        self.sam = SAMSegmenter(config.sam)
        self.opencv = OpenCVProcessor(config.opencv)
        self.room_extractor = RoomExtractor(self.sam, self.opencv)
        self.classifier = RoomClassifier(config.classification)
        self.model_loader = ModelLoader(config.diffusion)
        self.prompt_builder = PromptBuilder(config.style, config.diffusion)
        self.furnishing = FurnishingPipeline(
            self.model_loader, self.prompt_builder, config.diffusion
        )
        self.overlay_renderer = OverlayRenderer()
        self.compositor = ResultCompositor()

    def run(
        self,
        input_image: Image.Image,
        style: str = "modern",
        seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Execute the full pipeline: preprocess -> segment -> classify -> furnish.
        """
        # Phase 1: Preprocessing
        if progress_callback:
            progress_callback(0.1, desc="Preprocessing image...")
        image_bgr = self.preprocessor.process(input_image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Phase 2: Segmentation (SAM2 + OpenCV)
        if progress_callback:
            progress_callback(0.2, desc="Running SAM2 segmentation...")
        rooms = self.room_extractor.extract(image_rgb, image_bgr)

        # Unload SAM2 to free GPU memory before diffusion
        self.sam.unload_model()

        # Phase 3: Classification
        if progress_callback:
            progress_callback(0.4, desc="Classifying rooms...")
        total_area = image_bgr.shape[0] * image_bgr.shape[1]
        classified_rooms = self.classifier.classify_all(rooms, total_area)

        # Phase 4: Visualization (segmentation overlay)
        if progress_callback:
            progress_callback(0.5, desc="Rendering segmentation overlay...")
        overlay_bgr = self.overlay_renderer.render(image_bgr, classified_rooms)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        overlay_pil = Image.fromarray(overlay_rgb)

        # Phase 5: Edge detection for ControlNet
        if progress_callback:
            progress_callback(0.6, desc="Detecting edges for ControlNet...")
        edge_map = self.edge_detector.detect(image_bgr)
        canny_pil = self.edge_detector.resize_for_diffusion(
            edge_map,
            self.config.diffusion.generation_width,
            self.config.diffusion.generation_height,
        )

        # Phase 6: Furnished generation
        if progress_callback:
            progress_callback(0.7, desc="Generating furnished plan (this may take a while)...")
        furnished = self.furnishing.generate_full_plan(
            canny_pil, classified_rooms, style, seed
        )

        # Phase 7: Build summary
        room_summary = self._build_summary(classified_rooms)

        if progress_callback:
            progress_callback(1.0, desc="Done!")

        return PipelineResult(
            original=input_image,
            segmentation_overlay=overlay_pil,
            furnished=furnished,
            rooms=classified_rooms,
            room_summary=room_summary,
        )

    def run_segmentation_only(
        self,
        input_image: Image.Image,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[Image.Image, List[ClassifiedRoom], str]:
        """
        Run only preprocessing + segmentation + classification.
        """
        if progress_callback:
            progress_callback(0.1, desc="Preprocessing image...")
        image_bgr = self.preprocessor.process(input_image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if progress_callback:
            progress_callback(0.3, desc="Running segmentation...")
        rooms = self.room_extractor.extract(image_rgb, image_bgr)

        # Free SAM memory
        self.sam.unload_model()

        if progress_callback:
            progress_callback(0.6, desc="Classifying rooms...")
        total_area = image_bgr.shape[0] * image_bgr.shape[1]
        classified_rooms = self.classifier.classify_all(rooms, total_area)

        if progress_callback:
            progress_callback(0.8, desc="Rendering overlay...")
        overlay_bgr = self.overlay_renderer.render(image_bgr, classified_rooms)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        overlay_pil = Image.fromarray(overlay_rgb)

        summary = self._build_summary(classified_rooms)

        if progress_callback:
            progress_callback(1.0, desc="Segmentation complete!")

        return overlay_pil, classified_rooms, summary

    def run_generation_only(
        self,
        input_image: Image.Image,
        classified_rooms: List[ClassifiedRoom],
        style: str,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Image.Image:
        """
        Run only edge detection + diffusion on an already-segmented plan.
        """
        if progress_callback:
            progress_callback(0.1, desc="Preparing edges...")
        image_bgr = self.preprocessor.process(input_image)
        edge_map = self.edge_detector.detect(image_bgr)
        canny_pil = self.edge_detector.resize_for_diffusion(
            edge_map,
            self.config.diffusion.generation_width,
            self.config.diffusion.generation_height,
        )

        if progress_callback:
            progress_callback(0.3, desc="Generating furnished plan...")
        furnished = self.furnishing.generate_full_plan(
            canny_pil, classified_rooms, style, seed
        )

        if progress_callback:
            progress_callback(1.0, desc="Generation complete!")

        return furnished

    def _build_summary(self, rooms: List[ClassifiedRoom]) -> str:
        """Human-readable room summary."""
        if not rooms:
            return "No rooms detected."

        from collections import Counter
        counts = Counter(r.label for r in rooms)
        parts = []
        for label, count in counts.most_common():
            if count > 1:
                parts.append(f"{count}x {label}")
            else:
                parts.append(label)

        summary_lines = [
            f"Detected {len(rooms)} room(s):",
            ", ".join(parts),
            "",
            "Room Details:",
        ]

        for i, room in enumerate(rooms, 1):
            x, y, w, h = room.bbox
            summary_lines.append(
                f"  {i}. {room.label} "
                f"(area: {room.area_ratio:.1%} of plan, "
                f"confidence: {room.confidence:.0%}, "
                f"bbox: [{x},{y},{w},{h}])"
            )

        return "\n".join(summary_lines)
