"""
Phase 1: Panel Extraction
Uses a multi-strategy approach to split manga pages into meaningful artwork panels.

Primary strategy: Horizontal/vertical dark-line projection to detect panel borders.
Fallback strategy: Contour detection (external only) with strict area + dimension filters.
Last-resort: Smart grid split.
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
    """
    Extracts meaningful artwork panels from manga pages.

    Pipeline:
    1. Try projection-profile line detection (finds panel borders as dark stripes).
    2. If that yields < MIN_PANELS, try binary-threshold contour detection.
    3. If still < MIN_PANELS, fall back to a smart grid split.

    Filters applied to every candidate:
    - Minimum area ≥ MIN_AREA_FRACTION of the page.
    - Both width AND height ≥ MIN_DIM_PX pixels.
    - Aspect ratio within [MIN_ASPECT, MAX_ASPECT].
    - Non-max-suppression to remove panels nested inside larger panels.
    """

    # A real manga panel must cover at least this fraction of the page
    MIN_AREA_FRACTION = 0.04      # 4 % of page area
    MAX_AREA_FRACTION = 0.96      # never the whole page
    # Absolute minimum pixel size for width AND height
    MIN_DIM_PX = 250
    # Aspect-ratio limits
    MIN_ASPECT = 0.15
    MAX_ASPECT = 6.5
    # NMS: drop a box when this fraction of it overlaps a larger box
    NMS_OVERLAP_THRESHOLD = 0.35
    # Minimum panels to accept before trying next strategy
    MIN_PANELS = 2
    # Hard cap — keeps the largest N panels per page
    MAX_PANELS = 8

    def __init__(
        self,
        min_panel_area: int = 10000,   # kept for API compatibility
        max_panel_area: int = 5000000,
        panel_padding: int = 4,
        output_dir: Optional[Path] = None,
    ):
        self.panel_padding = panel_padding
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_panels_from_page(
        self,
        page_path: Path,
        page_number: int,
        output_dir: Optional[Path] = None,
    ) -> List[Panel]:
        page_path = Path(page_path)
        save_dir = Path(output_dir) if output_dir else self.output_dir
        if save_dir is None:
            raise ValueError("Output directory must be specified")
        save_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(page_path))
        if image is None:
            raise ValueError(f"Failed to load image: {page_path}")

        bboxes = self._detect_panel_bboxes(image)

        panels = []
        for idx, (x, y, w, h) in enumerate(bboxes):
            panel_image = image[y:y + h, x:x + w]
            panel_filename = f"temp_page_{page_number}_panel_{idx}.png"
            panel_path = save_dir / panel_filename
            cv2.imwrite(str(panel_path), panel_image)

            panel = Panel(
                panel_id=self._generate_global_id(page_number, idx),
                source_page=page_number,
                bbox=(x, y, w, h),
                image=panel_image,
                path=panel_path,
            )
            panels.append(panel)
            logger.info(
                "Extracted panel page_%d_idx_%d  size=(%dx%d)  path=%s",
                page_number, idx, w, h, panel_path,
            )

        return panels

    def extract_all_panels(
        self,
        page_paths: List[Path],
        output_dir: Optional[Path] = None,
    ) -> List[Panel]:
        all_panels = []
        panel_counter = 0

        for page_num, page_path in enumerate(page_paths, start=1):
            panels = self.extract_panels_from_page(page_path, page_num, output_dir)
            for panel in panels:
                panel_counter += 1
                panel.panel_id = panel_counter
                old_path = panel.path
                new_filename = f"panel_P{panel_counter}.png"
                new_path = old_path.parent / new_filename
                old_path.rename(new_path)
                panel.path = new_path
            all_panels.extend(panels)

        logger.info("Total panels extracted: %d", len(all_panels))
        return all_panels

    # ------------------------------------------------------------------
    # Detection pipeline
    # ------------------------------------------------------------------

    def _detect_panel_bboxes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Try multiple strategies; return reading-order bboxes."""
        H, W = image.shape[:2]
        min_area = int(H * W * self.MIN_AREA_FRACTION)
        max_area = int(H * W * self.MAX_AREA_FRACTION)

        # Strategy 1 – projection-based border detection
        bboxes = self._projection_based_detection(image, H, W, min_area)
        logger.info("Projection detection: %d panels", len(bboxes))

        # Strategy 2 – contour-based detection
        if len(bboxes) < self.MIN_PANELS:
            bboxes = self._contour_based_detection(image, H, W, min_area, max_area)
            logger.info("Contour detection: %d panels", len(bboxes))

        # Strategy 3 – grid fallback
        if len(bboxes) < self.MIN_PANELS:
            bboxes = self._grid_fallback(image, H, W)
            logger.info("Grid fallback: %d panels", len(bboxes))

        # Cap to the largest MAX_PANELS
        if len(bboxes) > self.MAX_PANELS:
            bboxes = sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)[: self.MAX_PANELS]

        bboxes.sort(key=lambda b: (b[1], b[0]))
        return bboxes

    # ------------------------------------------------------------------
    # Strategy 1: Projection-profile border detection
    # ------------------------------------------------------------------

    def _projection_based_detection(
        self, image: np.ndarray, H: int, W: int, min_area: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find dark horizontal and vertical bands (panel borders) using
        row/column mean brightness profiles.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Row mean — very dark rows are panel gutters
        row_mean = np.mean(gray, axis=1).astype(np.float32)
        # Col mean — very dark cols are panel gutters
        col_mean = np.mean(gray, axis=0).astype(np.float32)

        # Find gutter rows (dark stripe across almost the full width)
        h_cuts = self._find_cut_positions(row_mean, H, axis="row",
                                           min_frac=0.10, dark_thresh=200)
        # Find gutter cols
        v_cuts = self._find_cut_positions(col_mean, W, axis="col",
                                           min_frac=0.10, dark_thresh=200)

        # Build candidate rectangles from the grid of cuts
        row_ranges = self._cuts_to_ranges(h_cuts, H)
        col_ranges = self._cuts_to_ranges(v_cuts, W)

        bboxes = []
        for (y0, y1) in row_ranges:
            for (x0, x1) in col_ranges:
                w, h = x1 - x0, y1 - y0
                if w * h < min_area:
                    continue
                if w < self.MIN_DIM_PX or h < self.MIN_DIM_PX:
                    continue
                aspect = w / h if h > 0 else 0
                if not (self.MIN_ASPECT < aspect < self.MAX_ASPECT):
                    continue
                # Clip padding
                x = max(0, x0 - self.panel_padding)
                y = max(0, y0 - self.panel_padding)
                w2 = min(W - x, w + 2 * self.panel_padding)
                h2 = min(H - y, h + 2 * self.panel_padding)
                bboxes.append((x, y, w2, h2))

        bboxes = self._non_max_suppression(bboxes)
        return bboxes

    def _find_cut_positions(
        self, profile: np.ndarray, length: int,
        axis: str, min_frac: float, dark_thresh: float
    ) -> List[int]:
        """
        Find positions of dark bands in a 1-D brightness profile.
        Returns center positions of each band, including synthetic boundaries
        at 0 and length.
        """
        # Smooth to reduce noise
        kernel = np.ones(max(3, length // 200)) / max(3, length // 200)
        smoothed = np.convolve(profile, kernel, mode="same")

        # A "dark band" is where mean brightness < dark_thresh
        is_dark = smoothed < dark_thresh

        cuts = [0]
        in_band = False
        band_start = 0
        min_band_width = max(2, int(length * 0.003))

        for i, dark in enumerate(is_dark):
            if dark and not in_band:
                band_start = i
                in_band = True
            elif not dark and in_band:
                band_width = i - band_start
                if band_width >= min_band_width:
                    cuts.append((band_start + i) // 2)
                in_band = False

        cuts.append(length)

        # Deduplicate cuts that are very close together (< 2% of length apart)
        min_gap = max(5, int(length * 0.02))
        deduped = [cuts[0]]
        for c in cuts[1:]:
            if c - deduped[-1] >= min_gap:
                deduped.append(c)
        return deduped

    def _cuts_to_ranges(self, cuts: List[int], length: int) -> List[Tuple[int, int]]:
        """Convert sorted cut positions to (start, end) ranges, filtering tiny slivers."""
        ranges = []
        min_size = int(length * 0.05)
        for i in range(len(cuts) - 1):
            start, end = cuts[i], cuts[i + 1]
            if end - start >= min_size:
                ranges.append((start, end))
        return ranges

    # ------------------------------------------------------------------
    # Strategy 2: Contour-based detection
    # ------------------------------------------------------------------

    def _contour_based_detection(
        self, image: np.ndarray, H: int, W: int, min_area: int, max_area: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Use binary threshold + morphological ops to find panel contours.
        RETR_EXTERNAL ensures we only get outermost shapes (not text inside panels).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # OTSU binary threshold — manga borders are black on white
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological closing: bridge small gaps in panel borders
        close_size = max(5, min(W, H) // 80)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Remove small noise
        open_size = max(3, min(W, H) // 150)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < self.MIN_DIM_PX or h < self.MIN_DIM_PX:
                continue
            aspect = w / h if h > 0 else 0
            if not (self.MIN_ASPECT < aspect < self.MAX_ASPECT):
                continue
            x = max(0, x - self.panel_padding)
            y = max(0, y - self.panel_padding)
            w = min(W - x, w + 2 * self.panel_padding)
            h = min(H - y, h + 2 * self.panel_padding)
            raw_bboxes.append((x, y, w, h))

        return self._non_max_suppression(raw_bboxes)

    # ------------------------------------------------------------------
    # Strategy 3: Grid fallback
    # ------------------------------------------------------------------

    def _grid_fallback(self, image: np.ndarray, H: int, W: int) -> List[Tuple[int, int, int, int]]:
        """Split the page into equal rows (2 or 3) as a last resort."""
        rows = 3 if H > W else 2
        row_h = H // rows
        bboxes = []
        for r in range(rows):
            y0 = r * row_h
            y1 = H if r == rows - 1 else (r + 1) * row_h
            bboxes.append((0, y0, W, y1 - y0))
        logger.info("Grid fallback: %d-row split", rows)
        return bboxes

    # ------------------------------------------------------------------
    # Non-max suppression
    # ------------------------------------------------------------------

    def _non_max_suppression(
        self, bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Remove boxes that are largely contained within a larger box."""
        if not bboxes:
            return []

        sorted_boxes = sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)
        kept = []

        for box in sorted_boxes:
            x1, y1, w1, h1 = box
            dominated = False
            for kx, ky, kw, kh in kept:
                ix1 = max(x1, kx)
                iy1 = max(y1, ky)
                ix2 = min(x1 + w1, kx + kw)
                iy2 = min(y1 + h1, ky + kh)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                box_area = w1 * h1
                if box_area == 0:
                    continue
                if inter / box_area >= self.NMS_OVERLAP_THRESHOLD:
                    dominated = True
                    break
            if not dominated:
                kept.append(box)

        return kept

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_global_id(self, page_number: int, local_index: int) -> int:
        return (page_number - 1) * 10 + local_index + 1
