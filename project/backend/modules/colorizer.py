import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

# Add colorization_model to Python path so we can import from it
sys.path.append(str(Path(__file__).parent.parent / "colorization_model"))

class MangaColorizer:
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"MangaColorizer initializing on device: {self.device}")
        
        try:
            from colorizers import siggraph17
            # Load the model with pretrained weights
            self.model = siggraph17(pretrained=True).eval().to(self.device)
            logger.info("Successfully loaded pre-trained SIGGRAPH17 colorizer model.")
        except Exception as e:
            logger.error(f"Failed to load local colorizer weights: {e}")
            self.model = None

    def colorize_image(self, img_path: Path, output_path: Path) -> bool:
        if self.model is None:
            logger.warning("Colorizer model not initialized. Skipping colorization.")
            return False
            
        try:
            from colorizers.util import load_img, preprocess_img, postprocess_tens
            
            # 1. Load image (expects RGB numpy array)
            img = load_img(str(img_path))
            
            # 2. Preprocess to get L channel (original size and 256x256 resized)
            tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
            
            # 3. Transfer to device
            tens_l_rs = tens_l_rs.to(self.device)
            
            # 4. Run model forward pass
            with torch.no_grad():
                out_ab = self.model(tens_l_rs).cpu()
                
            # 5. Postprocess to combine original L channel with predicted ab channels
            out_img = postprocess_tens(tens_l_orig, out_ab)
            
            # 6. Save image using cv2 (out_img is float32 RGB in [0, 1])
            out_img_bgr = (out_img[:, :, ::-1] * 255.0).astype(np.uint8)
            cv2.imwrite(str(output_path), out_img_bgr)
            
            logger.info(f"Colorized: {img_path.name} -> {output_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to colorize image {img_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
