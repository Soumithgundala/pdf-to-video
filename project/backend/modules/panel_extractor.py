"""
Phase 1: Panel Extraction
Uses OpenCV contour detection to split manga pages into individual panels.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Panel:
    """Represents a single panel with metadata."""
    panel_id: int
    source_page: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    image: Optional[np.ndarray] = None
    path: Optional[Path] = None


class PanelExtractor:
    """Extracts individual panels from manga pages using contour detection."""

    def __init__(
        self,
        min_panel_area: int = 10000,
        max_panel_area: int = 5000000,
        panel_padding: int = 2,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize panel extractor.

        Args:
            min_panel_area: Minimum panel area in pixels
            max_panel_area: Maximum panel area in pixels
            panel_padding: Pixels to pad around detected panels
            output_dir: Directory to save extracted panels
        """
        self.min_panel_area = min_panel_area
        self.max_panel_area = max_panel_area
        self.panel_padding = panel_padding
        self.output_dir = output_dir

    def extract_panels_from_page(
        self,
        page_path: Path,
        page_number: int,
        output_dir: Optional[Path] = None
    ) -> List[Panel]:
        """
        Extract panels from a single manga page.

        Args:
            page_path: Path to the page image
            page_number: Page number for panel ID generation
            output_dir: Directory to save panels

        Returns:
            List of Panel objects
        """
        page_path = Path(page_path)
        save_dir = Path(output_dir) if output_dir else self.output_dir

        if save_dir is None:
            raise ValueError("Output directory must be specified")

        save_dir.mkdir(parents=True, exist_ok=True)

        # Load image
        image = cv2.imread(str(page_path))
        if image is None:
            raise ValueError(f"Failed to load image: {page_path}")

        # Get panel contours
        contours = self._detect_panel_contours(image)

        panels = []

        for idx, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Add padding
            x = max(0, x - self.panel_padding)
            y = max(0, y - self.panel_padding)
            w = min(image.shape[1] - x, w + 2 * self.panel_padding)
            h = min(image.shape[0] - y, h + 2 * self.panel_padding)

            # Extract panel region
            panel_image = image[y:y+h, x:x+w]

            # Create panel ID (global counter handled externally)
            panel_id = idx + 1
            global_id = self._generate_global_id(page_number, idx)

            # Save panel with temporary unique name to avoid naming collisions and file overwrites
            panel_filename = f"temp_page_{page_number}_panel_{idx}.png"
            panel_path = save_dir / panel_filename
            cv2.imwrite(str(panel_path), panel_image)

            panel = Panel(
                panel_id=global_id,
                source_page=page_number,
                bbox=(x, y, w, h),
                image=panel_image,
                path=panel_path
            )
            panels.append(panel)
            logger.info(f"Extracted panel page_{page_number}_idx_{idx} to {panel_path}")

        return panels

    def extract_all_panels(
        self,
        page_paths: List[Path],
        output_dir: Optional[Path] = None
    ) -> List[Panel]:
        """
        Extract panels from multiple manga pages.

        Args:
            page_paths: List of page image paths
            output_dir: Directory to save panels

        Returns:
            List of all extracted Panel objects
        """
        all_panels = []
        panel_counter = 0

        for page_num, page_path in enumerate(page_paths, start=1):
            panels = self.extract_panels_from_page(page_path, page_num, output_dir)

            # Reassign global IDs
            for panel in panels:
                panel_counter += 1
                panel.panel_id = panel_counter

                # Rename file with correct global ID
                old_path = panel.path
                new_filename = f"panel_P{panel_counter}.png"
                new_path = old_path.parent / new_filename
                old_path.rename(new_path)
                panel.path = new_path

            all_panels.extend(panels)

        logger.info(f"Total panels extracted: {len(all_panels)}")
        return all_panels

    def _detect_panel_contours(self, image: np.ndarray) -> List:
        """
        Detect panel contours using OpenCV.

        Strategy:
        1. Convert to grayscale
        2. Apply Gaussian blur
        3. Detect edges using Canny
        4. Find contours
        5. Filter by area and aspect ratio
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours
        valid_contours = []
        image_area = image.shape[0] * image.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_panel_area:
                continue
            if area > self.max_panel_area:
                continue

            # Filter out very thin/wide regions (likely text boxes or artifacts)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Keep panels with reasonable aspect ratios
            if 0.2 < aspect_ratio < 5.0:
                valid_contours.append(contour)

        # Sort contours by reading order (top-to-bottom, left-to-right)
        valid_contours = self._sort_contours_reading_order(valid_contours)

        return valid_contours

    def _sort_contours_reading_order(self, contours: List) -> List:
        """
        Sort contours in Japanese manga reading order
        (top-to-bottom, right-to-left for traditional, or left-to-right for modern)
        """
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h, contour))

        # Sort by y position (top to bottom), then x position (left to right)
        # For traditional manga, you might reverse x order
        bounding_boxes.sort(key=lambda b: (b[1], b[0]))

        return [b[4] for b in bounding_boxes]

    def _generate_global_id(self, page_number: int, local_index: int) -> int:
        """Generate a global panel ID from page and local index."""
        return (page_number - 1) * 10 + local_index + 1
