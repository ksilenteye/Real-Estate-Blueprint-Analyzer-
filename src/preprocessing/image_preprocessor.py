"""
Normalizes raw blueprint uploads into a consistent format for
downstream segmentation and edge detection.
"""
import cv2
import numpy as np
from PIL import Image

from config import PreprocessingConfig


class ImagePreprocessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def process(self, image: Image.Image) -> np.ndarray:
        """
        Full preprocessing pipeline.
        Returns: np.ndarray BGR (H, W, 3), resized and cleaned.
        """
        img = self._to_numpy_bgr(image)
        img = self._crop_borders(img)
        img = self._resize(img)
        img = self._denoise(img)
        img = self._enhance_contrast(img)
        img = self._binarize_if_blueprint(img)
        return img

    def _to_numpy_bgr(self, image: Image.Image) -> np.ndarray:
        """Convert PIL image to BGR numpy array."""
        if image.mode == "RGBA":
            # Composite alpha onto white background
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            image = bg
        elif image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize so longest edge <= max_dimension, preserving aspect ratio."""
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self.config.max_dimension:
            return image
        scale = self.config.max_dimension / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _crop_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove thin border artifacts from scanned images."""
        px = self.config.border_crop_px
        if px <= 0:
            return image
        h, w = image.shape[:2]
        if h > 2 * px and w > 2 * px:
            return image[px:h - px, px:w - px]
        return image

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising for scanned/photographed blueprints."""
        if self.config.denoise_strength <= 0:
            return image
        return cv2.fastNlMeansDenoisingColored(
            image, None,
            h=self.config.denoise_strength,
            hForColorComponents=self.config.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """CLAHE on L channel of LAB color space to sharpen faint lines."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _binarize_if_blueprint(self, image: np.ndarray) -> np.ndarray:
        """
        If the image is predominantly monochrome (typical for blueprints),
        apply adaptive thresholding to produce clean black/white lines,
        then convert back to 3-channel.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mean_sat = np.mean(saturation) / 255.0

        if mean_sat < self.config.grayscale_threshold:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=15,
                C=10,
            )
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return image
