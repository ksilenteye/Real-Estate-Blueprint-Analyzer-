"""
Produces canny edge maps for ControlNet conditioning.
The edge map preserves the structural lines of the blueprint
which ControlNet uses to maintain spatial fidelity in generation.
"""
import cv2
import numpy as np
from PIL import Image

from config import OpenCVConfig


class EdgeDetector:
    def __init__(self, config: OpenCVConfig):
        self.config = config

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Produce a canny edge map from a preprocessed blueprint.

        Args:
            image: BGR numpy array (H, W, 3)
        Returns:
            3-channel edge image (H, W, 3), white edges on black background.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            blurred,
            self.config.canny_low,
            self.config.canny_high,
        )
        # Stack to 3 channels for ControlNet
        return np.stack([edges, edges, edges], axis=-1)

    def to_pil(self, edge_map: np.ndarray) -> Image.Image:
        """Convert edge numpy array to PIL Image for the ControlNet pipeline."""
        return Image.fromarray(edge_map)

    def resize_for_diffusion(
        self, edge_map: np.ndarray, width: int, height: int
    ) -> Image.Image:
        """Resize the edge map to match diffusion model input dimensions."""
        resized = cv2.resize(edge_map, (width, height), interpolation=cv2.INTER_AREA)
        return Image.fromarray(resized)
