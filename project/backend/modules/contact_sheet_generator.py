"""
Phase 1: Contact Sheet Generator
Stitches all extracted panels into a single contact sheet with visible ID labels.
"""
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import logging

# Disable Pillow's decompression-bomb guard so large manga panels don't crash
# the process. We downscale every panel before stitching anyway, so memory
# usage stays under control.
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


class ContactSheetGenerator:
    """Generates a contact sheet from extracted panels."""

    def __init__(
        self,
        panel_size: Tuple[int, int] = (300, 400),
        grid_cols: int = 5,
        label_height: int = 30,
        background_color: str = "#1a1a1a",
        label_bg_color: str = "#2a2a2a",
        label_text_color: str = "#ffffff"
    ):
        """
        Initialize contact sheet generator.

        Args:
            panel_size: Target size for each panel in grid (width, height)
            grid_cols: Number of columns in the grid
            label_height: Height of ID label area
            background_color: Background color for contact sheet
            label_bg_color: Background color for labels
            label_text_color: Text color for labels
        """
        self.panel_size = panel_size
        self.grid_cols = grid_cols
        self.label_height = label_height
        self.background_color = background_color
        self.label_bg_color = label_bg_color
        self.label_text_color = label_text_color

    def generate_contact_sheet(
        self,
        panel_dirs: List[Path],
        output_path: Path,
        panels_per_sheet: int = 50
    ) -> List[Path]:
        """
        Generate contact sheet(s) from panel images.

        Args:
            panel_dirs: List of panel image paths
            output_path: Path to save contact sheet (without extension)
            panels_per_sheet: Maximum panels per contact sheet

        Returns:
            List of paths to generated contact sheets
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        contact_sheets = []

        # Split panels into sheets if needed
        total_panels = len(panel_dirs)
        num_sheets = (total_panels + panels_per_sheet - 1) // panels_per_sheet

        for sheet_num in range(num_sheets):
            start_idx = sheet_num * panels_per_sheet
            end_idx = min((sheet_num + 1) * panels_per_sheet, total_panels)
            sheet_panels = panel_dirs[start_idx:end_idx]

            sheet_path = output_path.parent / f"{output_path.stem}_sheet{sheet_num + 1}.png"

            self._create_single_sheet(sheet_panels, sheet_path, start_idx)
            contact_sheets.append(sheet_path)
            logger.info(f"Created contact sheet {sheet_num + 1}/{num_sheets}: {sheet_path}")

        return contact_sheets

    def _create_single_sheet(
        self,
        panel_paths: List[Path],
        output_path: Path,
        id_offset: int = 0
    ) -> None:
        """
        Create a single contact sheet from a list of panel images.
        """
        num_panels = len(panel_paths)

        if num_panels == 0:
            raise ValueError("No panels to create contact sheet")

        # Calculate grid dimensions
        num_rows = (num_panels + self.grid_cols - 1) // self.grid_cols

        # Calculate sheet size
        cell_width = self.panel_size[0]
        cell_height = self.panel_size[1] + self.label_height
        sheet_width = self.grid_cols * cell_width
        sheet_height = num_rows * cell_height

        # Create blank canvas
        sheet = Image.new('RGB', (sheet_width, sheet_height), self.background_color)
        draw = ImageDraw.Draw(sheet)

        # Load font (fallback to default if not available)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except (OSError, IOError):
                font = ImageFont.load_default()

        for idx, panel_path in enumerate(panel_paths):
            panel_id = idx + 1 + id_offset

            # Calculate position
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            x = col * cell_width
            y = row * cell_height

            # Load and resize panel
            try:
                panel_img = Image.open(panel_path)
                # Pre-downscale to at most 400px on the longest side BEFORE
                # stitching.  This keeps the final contact sheet from ballooning
                # into hundreds of millions of pixels for high-res manga scans.
                panel_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                panel_img = self._resize_and_pad(panel_img, self.panel_size)

                # Paste panel
                sheet.paste(panel_img, (x, y))

                # Draw label
                label_rect = (x, y + self.panel_size[1], x + cell_width, y + cell_height)
                draw.rectangle(label_rect, fill=self.label_bg_color)

                # Draw label text
                label_text = f"[P{panel_id}]"
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_x = x + (cell_width - text_width) // 2
                text_y = y + self.panel_size[1] + (self.label_height - 16) // 2
                draw.text((text_x, text_y), label_text, fill=self.label_text_color, font=font)

            except Exception as e:
                logger.error(f"Failed to add panel {panel_id}: {e}")
                # Draw placeholder
                draw.rectangle(
                    (x, y, x + cell_width, y + self.panel_size[1]),
                    fill="#333333",
                    outline="#555555"
                )
                placeholder_text = f"[P{panel_id}] FAILED"
                draw.text((x + 10, y + 10), placeholder_text, fill="#ff0000", font=font)

        # Save contact sheet (avoid optimize=True as it is a CPU bottleneck on large grids)
        sheet.save(output_path, 'PNG')
        logger.info(f"Saved contact sheet: {output_path}")

    def _resize_and_pad(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image to fit target size, maintaining aspect ratio with padding.
        """
        target_width, target_height = target_size

        # Calculate aspect ratios
        original_ratio = image.width / image.height
        target_ratio = target_width / target_height

        if original_ratio > target_ratio:
            # Image is wider, fit to width
            new_width = target_width
            new_height = int(target_width / original_ratio)
        else:
            # Image is taller, fit to height
            new_height = target_height
            new_width = int(target_height * original_ratio)

        # Resize
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create padded image
        padded = Image.new('RGB', target_size, self.background_color)

        # Calculate paste position (centered)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        # Convert to RGB if necessary
        if resized.mode in ('RGBA', 'P'):
            resized = resized.convert('RGB')

        padded.paste(resized, (paste_x, paste_y))

        return padded

    def generate_panel_index_map(self, panel_paths: List[Path]) -> dict:
        """
        Generate a mapping of panel IDs to their file paths.

        Returns:
            Dictionary mapping panel_id (str) to file path (str)
        """
        panel_map = {}
        for idx, path in enumerate(panel_paths, start=1):
            panel_map[f"P{idx}"] = str(path)
        return panel_map
