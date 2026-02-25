"""
Composes the final output: side-by-side display of
original blueprint, segmentation overlay, and furnished result.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ResultCompositor:
    def compose_side_by_side(
        self,
        original: Image.Image,
        segmentation: Image.Image,
        furnished: Image.Image,
        padding: int = 10,
    ) -> Image.Image:
        """
        Create a horizontal triptych: original | segmented | furnished.
        All images are resized to the same height.
        """
        # Normalize all images to same height
        target_h = min(original.height, segmentation.height, furnished.height)
        images = []
        for img in [original, segmentation, furnished]:
            if img.height != target_h:
                scale = target_h / img.height
                new_w = int(img.width * scale)
                img = img.resize((new_w, target_h), Image.LANCZOS)
            images.append(img)

        total_w = sum(img.width for img in images) + padding * 2
        canvas = Image.new("RGB", (total_w, target_h), (255, 255, 255))

        x_offset = 0
        for img in images:
            canvas.paste(img, (x_offset, 0))
            x_offset += img.width + padding

        return canvas

    def compose_with_labels(
        self,
        original: Image.Image,
        segmentation: Image.Image,
        furnished: Image.Image,
        label_height: int = 30,
        padding: int = 10,
    ) -> Image.Image:
        """
        Like compose_side_by_side but adds text labels above each panel.
        """
        labels = ["Original Blueprint", "Room Segmentation", "Furnished Plan"]

        # Normalize heights
        target_h = min(original.height, segmentation.height, furnished.height)
        images = []
        for img in [original, segmentation, furnished]:
            if img.height != target_h:
                scale = target_h / img.height
                new_w = int(img.width * scale)
                img = img.resize((new_w, target_h), Image.LANCZOS)
            images.append(img)

        total_w = sum(img.width for img in images) + padding * 2
        total_h = target_h + label_height

        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Try to use a readable font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

        x_offset = 0
        for i, (img, label) in enumerate(zip(images, labels)):
            # Draw label
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_x = x_offset + (img.width - text_w) // 2
            draw.text((text_x, 5), label, fill=(0, 0, 0), font=font)

            # Paste image below label
            canvas.paste(img, (x_offset, label_height))
            x_offset += img.width + padding

        return canvas
