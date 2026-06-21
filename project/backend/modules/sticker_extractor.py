"""
Sticker Extractor Module
Uses rembg to remove backgrounds and extract character stickers from panels.
"""
import cv2
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from rembg import remove
    REMBG_AVAILABLE = True
    logger.info("rembg is successfully imported and available.")
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("rembg is not installed. Character sticker extraction will fall back to using the full panel image.")

def extract_sticker(img: np.ndarray) -> np.ndarray:
    """
    Remove background from the panel image to extract the character sticker as a transparent RGBA image.
    If rembg is not available, returns the original image with an added alpha channel.
    """
    if img is None or img.size == 0:
        return img

    if not REMBG_AVAILABLE:
        # Return original with alpha channel
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    try:
        # rembg.remove handles numpy arrays (H, W, 3) and returns (H, W, 4)
        # Convert BGR (OpenCV default) to RGB for rembg
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sticker_rgb = remove(img_rgb)
        # Convert back to BGRA for OpenCV
        sticker_bgra = cv2.cvtColor(sticker_rgb, cv2.COLOR_RGBA2BGRA)
        return sticker_bgra
    except Exception as e:
        logger.error(f"Failed to remove background using rembg: {e}")
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img


def extract_clean_sticker(img: np.ndarray, focus_box: Optional[List[int]] = None) -> np.ndarray:
    """
    Extract a transparent character sticker from a panel image:
    1. Focus Area Crop: If focus_box is provided, crop image to it with padding.
    2. Background Removal: Remove background using rembg.
    3. Noise & Speech Bubble Filtering: Apply connected components analysis to filter out unwanted regions.
    4. Content Crop: Crop the final sticker to its content bounding box to remove transparent edges.
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    
    # 1. Focus area cropping
    cropped_img = img
    is_valid_focus = False
    if focus_box and len(focus_box) == 4:
        ymin, xmin, ymax, xmax = focus_box
        # Check if focus box is not the entire image (default)
        if not (ymin <= 5 and xmin <= 5 and ymax >= 995 and xmax >= 995):
            is_valid_focus = True

    if is_valid_focus:
        ymin, xmin, ymax, xmax = focus_box
        ymin_px = int(ymin * h / 1000.0)
        xmin_px = int(xmin * w / 1000.0)
        ymax_px = int(ymax * h / 1000.0)
        xmax_px = int(xmax * w / 1000.0)
        
        # Apply 15% padding around the box
        pad_h = int((ymax_px - ymin_px) * 0.15)
        pad_w = int((xmax_px - xmin_px) * 0.15)
        ymin_px = max(0, ymin_px - pad_h)
        ymax_px = min(h, ymax_px + pad_h)
        xmin_px = max(0, xmin_px - pad_w)
        xmax_px = min(w, xmax_px + pad_w)
        
        if ymax_px > ymin_px and xmax_px > xmin_px:
            cropped_img = img[ymin_px:ymax_px, xmin_px:xmax_px]
            logger.info(f"Cropped panel to focus box with padding: [{ymin_px}, {xmin_px}, {ymax_px}, {xmax_px}]")

    # 2. Background removal
    sticker_rgba = extract_sticker(cropped_img)
    if sticker_rgba is None or sticker_rgba.shape[2] != 4:
        # If extraction did not return an alpha channel, just add alpha and return
        if sticker_rgba.shape[2] == 3:
            return cv2.cvtColor(sticker_rgba, cv2.COLOR_BGR2BGRA)
        return sticker_rgba

    # 3. Clean alpha channel & speech bubble removal
    h_crop, w_crop = sticker_rgba.shape[:2]
    total_pixels = h_crop * w_crop
    
    sticker_rgb = sticker_rgba[:, :, :3]
    alpha = sticker_rgba[:, :, 3]
    
    # Grayscale original crop for intensity checks
    if cropped_img.shape[2] == 3:
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped_img
        
    _, alpha_thresh = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(alpha_thresh)
    
    clean_mask = np.zeros_like(alpha)
    valid_components = []
    min_area = max(100, int(total_pixels * 0.001)) # At least 0.1% of crop area
    
    # Compute edge map once to score detail/edge-density
    edges = cv2.Canny(gray, 50, 150)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
            
        comp_mask = (labels == i).astype(np.uint8)
        mean_val = cv2.mean(gray, mask=comp_mask)[0]
        
        # Speech bubble filter (bright pixels in original panel)
        if mean_val > 220:
            logger.info(f"Sticker extractor: Filtered component {i} as speech bubble (mean grayscale={mean_val:.1f})")
            continue
            
        # Compute detail score (edge density) inside this component
        edge_density = cv2.mean(edges, mask=comp_mask)[0]
        
        valid_components.append((i, area, edge_density, comp_mask))
        
    if not valid_components:
        # Fallback: keep the largest component if everything got filtered
        if num_labels > 1:
            largest_idx = 1 + np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)])
            clean_mask = (labels == largest_idx).astype(np.uint8) * 255
            logger.info(f"Sticker extractor: No valid components found, falling back to largest component {largest_idx}")
        else:
            clean_mask = alpha
    else:
        # Keep components that are:
        # 1. At least 10% of the largest valid component's area
        # 2. Or have high detail density (edge_density > 20.0), indicating character face/features
        largest_valid_area = max(comp[1] for comp in valid_components)
        for idx, area, density, comp_mask in valid_components:
            if area >= largest_valid_area * 0.10 or (area >= largest_valid_area * 0.03 and density > 20.0):
                clean_mask = cv2.bitwise_or(clean_mask, comp_mask * 255)
            else:
                logger.info(f"Sticker extractor: Dropped component {idx} (area={area}, detail={density:.2f})")
                
    cleaned_alpha = cv2.bitwise_and(alpha, clean_mask)
    
    # 4. Content Crop: Crop to tight bounding box of content
    non_zero_coords = cv2.findNonZero(cleaned_alpha)
    if non_zero_coords is not None:
        x_c, y_c, w_c, h_c = cv2.boundingRect(non_zero_coords)
        cropped_rgb = sticker_rgb[y_c:y_c+h_c, x_c:x_c+w_c]
        cropped_alpha = cleaned_alpha[y_c:y_c+h_c, x_c:x_c+w_c]
        result_sticker = cv2.merge([cropped_rgb, cropped_alpha])
        logger.info(f"Sticker extractor: Cropped sticker size from {sticker_rgba.shape[:2]} to {result_sticker.shape[:2]}")
        return result_sticker
        
    return sticker_rgba
