"""
OpenCV-based structural analysis of blueprints.
Detects walls using Hough line transform and morphological operations.
Produces a binary wall mask and extracted line segments.
"""
import cv2
import numpy as np
from typing import List, Tuple

from config import OpenCVConfig


class OpenCVProcessor:
    def __init__(self, config: OpenCVConfig):
        self.config = config

    def detect_walls(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect wall structures in a blueprint image.

        Args:
            image: BGR numpy array
        Returns:
            (wall_mask, lines)
            wall_mask: binary (H, W) uint8, 255 = wall
            lines: list of ((x1,y1), (x2,y2)) tuples
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold to binarize - walls become black (0)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=10,
        )

        # Morphological closing to connect small wall gaps
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
        )
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Detect lines using probabilistic Hough transform
        raw_lines = cv2.HoughLinesP(
            closed,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_line_length,
            maxLineGap=self.config.hough_max_line_gap,
        )

        lines = []
        wall_mask = np.zeros(gray.shape, dtype=np.uint8)

        if raw_lines is not None:
            for line in raw_lines:
                x1, y1, x2, y2 = line[0]
                lines.append(((x1, y1), (x2, y2)))
                # Draw wall lines with thickness proportional to detected wall width
                cv2.line(wall_mask, (x1, y1), (x2, y2), 255, thickness=3)

        # Combine Hough lines with the thresholded binary for robust wall mask
        # Dilate the line mask to cover full wall thickness
        wall_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        wall_mask = cv2.dilate(wall_mask, wall_kernel, iterations=1)

        # Also merge in thick dark regions from the original binary
        # (catches walls that aren't perfectly straight lines)
        thick_features = self._detect_thick_features(closed)
        wall_mask = cv2.bitwise_or(wall_mask, thick_features)

        return wall_mask, lines

    def _detect_thick_features(self, binary: np.ndarray) -> np.ndarray:
        """
        Detect thick dark features (walls) using morphological operations.
        Erode to remove thin lines (text, dimensions), then dilate back.
        """
        min_thick, max_thick = self.config.wall_thickness_range

        # Horizontal walls
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_thick))
        h_walls = cv2.erode(binary, h_kernel, iterations=1)
        h_walls = cv2.dilate(h_walls, h_kernel, iterations=1)

        # Vertical walls
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_thick, 1))
        v_walls = cv2.erode(binary, v_kernel, iterations=1)
        v_walls = cv2.dilate(v_walls, v_kernel, iterations=1)

        return cv2.bitwise_or(h_walls, v_walls)

    def find_room_contours(self, wall_mask: np.ndarray) -> List[np.ndarray]:
        """
        Invert the wall mask and find contours of enclosed regions (rooms).

        Returns:
            List of contour arrays, each representing a room boundary.
        """
        # Invert: rooms become white, walls become black
        inverted = cv2.bitwise_not(wall_mask)

        # Clean up small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        # Filter by minimum area and approximate polygons
        room_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.config.contour_min_area:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                room_contours.append(approx)

        return room_contours

    def detect_doors_windows(
        self, image: np.ndarray, wall_mask: np.ndarray
    ) -> dict:
        """
        Detect gaps in walls that represent doors/windows.

        Returns:
            {'doors': List[bbox], 'windows': List[bbox]}
        """
        doors = []
        windows = []

        # Find gaps by looking at connected components of the inverted wall mask
        # along wall lines
        inverted = cv2.bitwise_not(wall_mask)

        # Thin the wall mask to skeleton to find gaps more precisely
        skeleton = cv2.ximgproc.thinning(wall_mask) if hasattr(cv2, 'ximgproc') else wall_mask

        # Find contours in the gap regions near walls
        dilated_walls = cv2.dilate(wall_mask, np.ones((7, 7), np.uint8), iterations=1)
        gap_region = cv2.bitwise_and(inverted, dilated_walls)

        contours, _ = cv2.findContours(
            gap_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w, h) / (min(w, h) + 1e-6)

            if 100 < area < 3000 and aspect > 2:
                # Elongated small gap near wall = door or window
                if area < 800:
                    doors.append((x, y, w, h))
                else:
                    windows.append((x, y, w, h))

        return {'doors': doors, 'windows': windows}
