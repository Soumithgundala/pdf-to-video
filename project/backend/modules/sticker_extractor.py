"""
Sticker Extractor Module

Creates clean manga/anime-style character stickers by combining rembg
segmentation, focus-box guidance, alpha matte cleanup, and safe cropping.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from rembg import new_session, remove

    REMBG_AVAILABLE = True
    logger.info("rembg is successfully imported and available.")
except ImportError:
    REMBG_AVAILABLE = False
    new_session = None  # type: ignore[assignment]
    remove = None  # type: ignore[assignment]
    logger.warning(
        "rembg is not installed. Character sticker extraction will fall back to the full panel image."
    )

REMBG_MODEL = (os.getenv("MANGA_STICKER_REM_BG_MODEL", "isnet-anime").strip() or "isnet-anime")
REMBG_SESSION_CACHE: Dict[str, object] = {}
REMBG_SESSION_LOCK = threading.Lock()


def _ensure_bgra(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 4:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)


def _get_rembg_session() -> Optional[object]:
    if not REMBG_AVAILABLE:
        return None

    model_candidates = [
        REMBG_MODEL,
        "isnet-anime",
        "birefnet-general",
        "birefnet-general-lite",
        "u2net",
    ]
    last_error: Optional[Exception] = None

    with REMBG_SESSION_LOCK:
        for model_name in model_candidates:
            cached = REMBG_SESSION_CACHE.get(model_name)
            if cached is not None:
                return cached

            try:
                session = new_session(model_name)  # type: ignore[misc]
                REMBG_SESSION_CACHE[model_name] = session
                if model_name != REMBG_MODEL:
                    logger.info("Sticker extractor: using fallback rembg model '%s'.", model_name)
                return session
            except Exception as exc:
                last_error = exc
                logger.warning("Sticker extractor: failed to load rembg model '%s': %s", model_name, exc)

    if last_error:
        logger.warning("Sticker extractor: no rembg session could be created: %s", last_error)
    return None


def _remove_background(
    img: np.ndarray,
    use_alpha_matting: bool = False,
    max_side: int = 1024,
) -> np.ndarray:
    if img is None or img.size == 0:
        return img

    if not REMBG_AVAILABLE:
        return _ensure_bgra(img)

    try:
        orig_h, orig_w = img.shape[:2]
        max_dim = max(orig_h, orig_w)
        scale = 1.0
        if max_dim > max_side and max_dim > 0:
            scale = float(max_side) / float(max_dim)

        proc_img = img
        if scale < 1.0:
            proc_w = max(1, int(round(orig_w * scale)))
            proc_h = max(1, int(round(orig_h * scale)))
            proc_img = cv2.resize(img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        session = _get_rembg_session()
        remove_kwargs = dict(
            alpha_matting=use_alpha_matting,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=8,
        )
        if session is not None:
            remove_kwargs["session"] = session

        sticker = remove(  # type: ignore[misc]
            img_rgb,
            **remove_kwargs,
        )

        if isinstance(sticker, bytes):
            from io import BytesIO
            from PIL import Image

            sticker = np.array(Image.open(BytesIO(sticker)).convert("RGBA"))

        if sticker.ndim == 2:
            alpha = sticker
        elif sticker.shape[2] == 4:
            alpha = sticker[:, :, 3]
        elif sticker.shape[2] == 3:
            alpha = cv2.cvtColor(sticker, cv2.COLOR_RGB2GRAY)
        else:
            return _ensure_bgra(img)

        if scale < 1.0:
            alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        result = _ensure_bgra(img)
        result[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
        return result
    except Exception as exc:
        logger.error("Failed to remove background using rembg: %s", exc)
        return _ensure_bgra(img)


def _focus_box_to_pixels(
    focus_box: Optional[List[int]],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    if not focus_box or len(focus_box) != 4:
        return None

    ymin, xmin, ymax, xmax = focus_box
    if ymin <= 5 and xmin <= 5 and ymax >= 995 and xmax >= 995:
        return None

    fx1 = int(xmin * width / 1000.0)
    fy1 = int(ymin * height / 1000.0)
    fx2 = int(xmax * width / 1000.0)
    fy2 = int(ymax * height / 1000.0)

    fx1 = max(0, min(fx1, width - 1))
    fy1 = max(0, min(fy1, height - 1))
    fx2 = max(fx1 + 1, min(fx2, width))
    fy2 = max(fy1 + 1, min(fy2, height))
    return fx1, fy1, fx2, fy2


def _expand_bounds(
    bounds: Tuple[int, int, int, int],
    width: int,
    height: int,
    pad_ratio: float = 0.22,
    min_pad: int = 32,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bounds
    pad_w = max(min_pad, int((x2 - x1) * pad_ratio))
    pad_h = max(min_pad, int((y2 - y1) * pad_ratio))

    return (
        max(0, x1 - pad_w),
        max(0, y1 - pad_h),
        min(width, x2 + pad_w),
        min(height, y2 + pad_h),
    )


def _auto_focus_rect(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Estimate a likely subject box when the LLM focus box is missing or unhelpful."""
    if img is None or img.size == 0:
        return None

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    edge_map = cv2.Canny(gray, 40, 120)
    sat_map = hsv[:, :, 1]
    val_map = hsv[:, :, 2]

    edge_map = cv2.GaussianBlur(edge_map, (0, 0), 2.0).astype(np.float32) / 255.0
    sat_map = cv2.GaussianBlur(sat_map, (0, 0), 2.0).astype(np.float32) / 255.0
    dark_map = 1.0 - (cv2.GaussianBlur(val_map, (0, 0), 2.0).astype(np.float32) / 255.0)

    saliency = edge_map * 0.6 + sat_map * 0.25 + dark_map * 0.15
    saliency = cv2.GaussianBlur(saliency, (0, 0), 6.0)
    threshold = max(0.16, float(np.percentile(saliency, 82)))
    mask = (saliency >= threshold).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return None

    image_cx = width / 2.0
    image_cy = height / 2.0
    best_score = -1.0
    best_bbox: Optional[Tuple[int, int, int, int]] = None

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])

        if area < max(800, int(height * width * 0.002)):
            continue
        if area > height * width * 0.82:
            continue

        comp_mask = (labels == idx).astype(np.uint8)
        mean_saliency = float(cv2.mean(saliency, mask=comp_mask)[0])
        comp_cx = x + cw / 2.0
        comp_cy = y + ch / 2.0
        distance = ((comp_cx - image_cx) ** 2 + (comp_cy - image_cy) ** 2) ** 0.5
        border_penalty = 1.0
        if x <= 2 or y <= 2 or x + cw >= width - 3 or y + ch >= height - 3:
            border_penalty = 0.72

        score = area * mean_saliency * border_penalty + max(0.0, 1200.0 - distance) * 0.6
        if score > best_score:
            best_score = score
            best_bbox = (x, y, x + cw, y + ch)

    if best_bbox is None:
        return None

    expanded = _expand_bounds(best_bbox, width, height, pad_ratio=0.38, min_pad=48)
    x1, y1, x2, y2 = expanded
    box_w = x2 - x1
    box_h = y2 - y1
    if (box_w * box_h) > height * width * 0.9:
        return None

    shrink = 0.72
    shrink_w = max(64, int(box_w * shrink))
    shrink_h = max(64, int(box_h * shrink))
    center_x = x1 + box_w // 2
    center_y = y1 + box_h // 2
    half_w = shrink_w // 2
    half_h = shrink_h // 2
    tight = (
        max(0, center_x - half_w),
        max(0, center_y - half_h),
        min(width, center_x + half_w),
        min(height, center_y + half_h),
    )
    return tight


def _component_stats(alpha: np.ndarray, gray: np.ndarray, edges: np.ndarray, min_area: int) -> List[dict]:
    _, alpha_mask = cv2.threshold(alpha, 18, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_mask)

    valid_components = []
    total_h, total_w = alpha.shape[:2]

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        comp_mask = (labels == idx).astype(np.uint8)
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])

        mean_val = cv2.mean(gray, mask=comp_mask)[0]
        std_val = cv2.meanStdDev(gray, mask=comp_mask)[1][0][0]
        edge_density = cv2.mean(edges, mask=comp_mask)[0]
        touches_border = x <= 1 or y <= 1 or x + cw >= total_w - 2 or y + ch >= total_h - 2

        valid_components.append(
            {
                "idx": idx,
                "area": area,
                "mask": comp_mask,
                "bbox": (x, y, cw, ch),
                "mean_val": mean_val,
                "std_val": std_val,
                "edge_density": edge_density,
                "touches_border": touches_border,
            }
        )

    return valid_components


def _rank_components(
    components: List[dict],
    focus_rect: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
) -> List[dict]:
    height, width = image_shape
    if not components:
        return []

    for comp in components:
        x, y, cw, ch = comp["bbox"]
        x2 = x + cw
        y2 = y + ch

        border_pixels = 0
        mask = comp["mask"].astype(np.uint8) * 255
        border_pixels += int(mask[0, :].sum() > 0) + int(mask[-1, :].sum() > 0)
        border_pixels += int(mask[:, 0].sum() > 0) + int(mask[:, -1].sum() > 0)

        comp_score = float(comp["area"])
        comp_score += float(comp["edge_density"]) * 180.0

        if focus_rect is not None:
            fx1, fy1, fx2, fy2 = focus_rect
            focus_cx = (fx1 + fx2) / 2.0
            focus_cy = (fy1 + fy2) / 2.0
            ix1 = max(fx1, x)
            iy1 = max(fy1, y)
            ix2 = min(fx2, x2)
            iy2 = min(fy2, y2)
            overlap = 0
            if ix2 > ix1 and iy2 > iy1:
                overlap = (ix2 - ix1) * (iy2 - iy1)
            comp_score += overlap * 2.2

            comp_cx = x + cw / 2.0
            comp_cy = y + ch / 2.0
            distance = ((comp_cx - focus_cx) ** 2 + (comp_cy - focus_cy) ** 2) ** 0.5
            comp_score += max(0.0, 1200.0 - distance) * 2.0
            comp["overlap"] = overlap
            comp["focus_distance"] = distance
        else:
            comp_cx = x + cw / 2.0
            comp_cy = y + ch / 2.0
            image_cx = width / 2.0
            image_cy = height / 2.0
            distance = ((comp_cx - image_cx) ** 2 + (comp_cy - image_cy) ** 2) ** 0.5
            comp_score += max(0.0, 1000.0 - distance) * 1.5
            comp["focus_distance"] = distance
            comp["overlap"] = 0

        if comp["touches_border"]:
            comp_score *= 0.8
        if comp["mean_val"] > 235 and comp["std_val"] < 18:
            comp_score *= 0.5

        comp_score -= border_pixels * 40.0
        comp["score"] = comp_score

    return sorted(components, key=lambda c: c.get("score", 0.0), reverse=True)


def _refine_alpha(alpha: np.ndarray) -> np.ndarray:
    mask = alpha.copy()
    if mask.ndim != 2:
        raise ValueError("Alpha mask must be single-channel.")

    _, mask = cv2.threshold(mask, 18, 255, cv2.THRESH_BINARY)

    min_dim = min(mask.shape[:2])
    kernel_size = max(3, int(round(min_dim * 0.004)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    close_kernel = np.ones((kernel_size + 2, kernel_size + 2), dtype=np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    coverage = float(np.count_nonzero(mask)) / float(mask.size)
    if coverage < 0.6:
        feather_size = max(0.5, min_dim * 0.0008)
        mask = cv2.GaussianBlur(mask, (0, 0), feather_size)
    return mask


def _refine_with_grabcut(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Use GrabCut to clean up a likely foreground mask inside a tight crop."""
    if rgb is None or alpha is None or rgb.size == 0 or alpha.size == 0:
        return alpha

    if alpha.ndim != 2:
        return alpha

    coverage = float(np.count_nonzero(alpha)) / float(alpha.size)
    if coverage < 0.03 or coverage > 0.92:
        return alpha

    mask = np.full(alpha.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    mask[alpha < 18] = cv2.GC_BGD
    mask[(alpha >= 18) & (alpha < 90)] = cv2.GC_PR_BGD
    mask[alpha >= 90] = cv2.GC_PR_FGD

    sure_fg = cv2.erode((alpha > 200).astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)
    mask[sure_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(rgb, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return alpha

    refined = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    refined = cv2.bitwise_and(refined, alpha)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return refined


def extract_sticker(
    img: np.ndarray,
    use_alpha_matting: bool = False,
    max_side: int = 1024,
) -> np.ndarray:
    """
    Remove background from the panel image to extract the character sticker as a transparent RGBA image.
    If rembg is not available, returns the original image with an added alpha channel.
    """
    if img is None or img.size == 0:
        return img

    if not REMBG_AVAILABLE:
        return _ensure_bgra(img)

    sticker = _remove_background(img, use_alpha_matting=use_alpha_matting, max_side=max_side)
    return sticker if sticker is not None else _ensure_bgra(img)


def extract_clean_sticker(img: np.ndarray, focus_box: Optional[List[int]] = None) -> np.ndarray:
    """
    Extract a rectangular crop of the focus area (action/closeup) instead of removing the background.
    This avoids bad rembg stickers and highlights the most exciting parts.
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    focus_rect = _focus_box_to_pixels(focus_box, w, h)
    if focus_rect is None:
        focus_rect = _auto_focus_rect(img)

    if focus_rect is not None:
        # Pad the focus box slightly to keep some context
        fx1, fy1, fx2, fy2 = _expand_bounds(focus_rect, w, h, pad_ratio=0.15, min_pad=24)
        crop_img = img[fy1:fy2, fx1:fx2]
        
        # Ensure BGRA with fully opaque alpha
        rgba = _ensure_bgra(crop_img)
        rgba[:, :, 3] = 255
        logger.info("Sticker extractor: produced rectangular crop instead of transparent sticker.")
        return rgba

    # Fallback to whole image if no focus rect found
    rgba = _ensure_bgra(img)
    rgba[:, :, 3] = 255
    return rgba
