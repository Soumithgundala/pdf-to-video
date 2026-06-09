"""
Phase 1: Contact Sheet Generator
Stitches all extracted panels into a single contact sheet with highly presentable, card-based layouts and ID badges.
"""
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import logging

# Disable Pillow's decompression-bomb guard so large manga panels don't crash
# the process. We downscale every panel before stitching anyway, so memory
# usage stays under control.
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


class ContactSheetGenerator:
    """Generates a contact sheet from extracted panels with premium card-based styling."""

    def __init__(
        self,
        panel_size: Tuple[int, int] = (300, 400),
        grid_cols: int = 5,
        label_height: int = 40,
        background_color: str = "#0b0f19",    # Sleek dark navy/slate
        label_bg_color: str = "#1e293b",       # Card background (slate-800)
        label_text_color: str = "#ffffff"
    ):
        """
        Initialize contact sheet generator.

        Args:
            panel_size: Target size for each panel in grid (width, height)
            grid_cols: Number of columns in the grid
            label_height: Height of ID label area inside the card
            background_color: Background color for contact sheet
            label_bg_color: Background color for cards
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
        panel_paths: List[Path],
        output_path: Path,
        panels_per_sheet: int = 50
    ) -> List[Path]:
        """
        Generate contact sheet(s) from panel images.

        Args:
            panel_paths: List of panel image paths
            output_path: Path to save contact sheet (without extension)
            panels_per_sheet: Maximum panels per contact sheet

        Returns:
            List of paths to generated contact sheets
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        contact_sheets = []

        # Split panels into sheets if needed
        total_panels = len(panel_paths)
        num_sheets = (total_panels + panels_per_sheet - 1) // panels_per_sheet

        for sheet_num in range(num_sheets):
            start_idx = sheet_num * panels_per_sheet
            end_idx = min((sheet_num + 1) * panels_per_sheet, total_panels)
            sheet_panels = panel_paths[start_idx:end_idx]

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

        # Load fonts
        font = None
        header_font = None
        sub_font = None

        # Try to load Windows-friendly fonts
        for font_name in ["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"]:
            try:
                font = ImageFont.truetype(font_name, 14)
                header_font = ImageFont.truetype(font_name, 28)
                sub_font = ImageFont.truetype(font_name.replace("b.ttf", ".ttf").replace("bd.ttf", ".ttf"), 12)
                break
            except Exception:
                continue

        # Fallbacks for non-Windows/default environments
        if font is None:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
                header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
                sub_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except Exception:
                font = ImageFont.load_default()
                header_font = ImageFont.load_default()
                sub_font = ImageFont.load_default()

        # Calculate grid layout dimensions (adding card margins and borders)
        num_rows = (num_panels + self.grid_cols - 1) // self.grid_cols
        header_height = 100
        
        cell_width = self.panel_size[0] + 20   # 320px width per cell
        cell_height = self.panel_size[1] + self.label_height + 20  # 460px height per cell
        
        sheet_width = self.grid_cols * cell_width
        sheet_height = num_rows * cell_height + header_height

        # Create blank canvas
        sheet = Image.new('RGB', (sheet_width, sheet_height), self.background_color)
        draw = ImageDraw.Draw(sheet)

        # Draw a beautiful header
        draw.rectangle((0, 0, sheet_width, header_height), fill="#0f172a") # Dark Slate background
        draw.rectangle((0, header_height - 2, sheet_width, header_height), fill="#1e293b") # Bottom border line
        
        # Header text
        draw.text((30, 20), "MANGA RECAP", fill="#818cf8", font=header_font)
        draw.text((30, 60), f"Storyboard Panels Directory | Total Panels: {len(panel_paths) + id_offset} | Showing P{id_offset + 1} - P{id_offset + num_panels}", fill="#94a3b8", font=sub_font)
        
        sheet_info = f"Sheet {id_offset // 50 + 1}"
        draw.text((sheet_width - 150, 35), sheet_info, fill="#64748b", font=header_font)

        # Draw cells
        for idx, panel_path in enumerate(panel_paths):
            panel_id = idx + 1 + id_offset

            # Calculate position
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            x = col * cell_width
            y = row * cell_height + header_height

            card_x1 = x + 10
            card_y1 = y + 10
            card_x2 = x + cell_width - 10
            card_y2 = y + cell_height - 10

            # Draw card rounded container with borders
            draw.rounded_rectangle(
                (card_x1, card_y1, card_x2, card_y2),
                radius=12,
                fill=self.label_bg_color,
                outline="#334155",
                width=2
            )

            # Load and resize panel
            try:
                panel_img = Image.open(panel_path)
                # Pre-downscale to fit cleanly inside our card
                panel_img.thumbnail((self.panel_size[0] - 12, self.panel_size[1] - 12), Image.Resampling.LANCZOS)
                
                # Canvas size inside the card for the panel image
                inner_target_size = (cell_width - 32, self.panel_size[1] - 10)
                panel_img = self._resize_and_pad(panel_img, inner_target_size)

                # Paste panel inside card
                paste_x = card_x1 + 6
                paste_y = card_y1 + 6
                sheet.paste(panel_img, (paste_x, paste_y))

                # Draw pill badge overlaying the bottom card area
                pill_w = 120
                pill_h = 26
                pill_x1 = card_x1 + (cell_width - 20 - pill_w) // 2
                pill_y1 = card_y1 + inner_target_size[1] + (self.label_height + 8 - pill_h) // 2
                pill_x2 = pill_x1 + pill_w
                pill_y2 = pill_y1 + pill_h

                draw.rounded_rectangle(
                    (pill_x1, pill_y1, pill_x2, pill_y2),
                    radius=13,
                    fill="#4f46e5"  # Beautiful Indigo-600 pill badge
                )

                # Centered label text inside pill
                label_text = f"PANEL P{panel_id}"
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = pill_x1 + (pill_w - text_width) // 2
                text_y = pill_y1 + (pill_h - text_height) // 2 - 1
                
                draw.text((text_x, text_y), label_text, fill=self.label_text_color, font=font)

            except Exception as e:
                logger.error(f"Failed to add panel {panel_id}: {e}")
                # Draw elegant error placeholder
                draw.rounded_rectangle(
                    (card_x1 + 6, card_y1 + 6, card_x2 - 6, card_y1 + self.panel_size[1] - 4),
                    radius=8,
                    fill="#3f1f1f"
                )
                placeholder_text = f"P{panel_id} FAILED"
                draw.text((card_x1 + 20, card_y1 + 20), placeholder_text, fill="#ff8888", font=font)

        # Save contact sheet (avoid optimize=True as it is a CPU bottleneck on large grids)
        sheet.save(output_path, 'PNG')
        logger.info(f"Saved contact sheet: {output_path}")

    def _resize_and_pad(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image to fit target size, maintaining aspect ratio with Slate padding.
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

        # Padded area uses deep charcoal to contrast panel scans
        padded = Image.new('RGB', target_size, "#090d16")

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
