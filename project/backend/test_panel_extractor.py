"""
Quick smoke-test for the new PanelExtractor.
Tests first 3 pages of the existing job workspace.
"""
import sys
sys.path.insert(0, r"C:\Users\soumi\Downloads\pdftovideo\project\backend")

from pathlib import Path
from modules.panel_extractor import PanelExtractor

pages_dir = Path(r"C:\Users\soumi\Downloads\pdftovideo\project\backend\temp_workspace\e86c07d2-3c73-46c4-89f6-0ff00c9c09a2\pages")
out_dir   = Path(r"C:\Users\soumi\Downloads\pdftovideo\project\backend\temp_workspace\panel_test_output")
out_dir.mkdir(exist_ok=True)

pages = sorted(pages_dir.glob("*.png"))[:5]

extractor = PanelExtractor()

for page_path in pages:
    page_num = int(page_path.stem.split("_")[1])
    panels = extractor.extract_panels_from_page(page_path, page_num, out_dir)
    comparisons = [(f"bbox:{p.bbox[2]}x{p.bbox[3]} -> img:{p.image.shape[1]}x{p.image.shape[0]}") for p in panels]
    print(f"Page {page_num:03d}: {len(panels)} panels -> {comparisons}")

print("\nDone. Check output dir:", out_dir)
