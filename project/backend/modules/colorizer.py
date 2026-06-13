import os
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

class MangaColorizer:
    def __init__(self, mode: str = "stable_diffusion", use_gpu: bool = True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.mode = mode.lower()
        logger.info(f"MangaColorizer initializing on device: {self.device} in mode: {self.mode}")
        
        # Decide torch datatype
        if self.device.type == "cuda":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        self.pipe = None
        self.classic_model = None

        if self.mode == "stable_diffusion":
            try:
                from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
                
                logger.info("Loading ControlNet model (lllyasviel/control_v11p_sd15_canny)...")
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11p_sd15_canny",
                    torch_dtype=self.torch_dtype
                )
                
                logger.info("Loading Base SD 1.5 Anime model (stablediffusionapi/anything-v5)...")
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    "stablediffusionapi/anything-v5",
                    controlnet=controlnet,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None
                )
                
                # Configure scheduler for faster and high-quality generation
                self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
                
                # Move to device (enable_model_cpu_offload will handle shifting between CPU and GPU if enabled)
                if self.device.type == "cuda":
                    logger.info("Enabling memory optimizations for GPU...")
                    self.pipe.enable_attention_slicing()
                    try:
                        self.pipe.enable_model_cpu_offload()
                    except Exception as offload_err:
                        logger.warning(f"Failed to enable model CPU offload: {offload_err}. Loading directly to device.")
                        self.pipe.to(self.device)
                else:
                    self.pipe.to(self.device)
                    
                logger.info("Successfully loaded Stable Diffusion ControlNet colorizer.")
            except Exception as e:
                logger.error(f"Failed to load local colorizer weights/models: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.pipe = None
        elif self.mode == "classic":
            try:
                from colorization_model.colorizers import siggraph17
                logger.info("Loading Classic CNN Colorizer model (siggraph17)...")
                self.classic_model = siggraph17(pretrained=True).eval()
                if self.device.type == "cuda":
                    self.classic_model.cuda()
                logger.info("Successfully loaded Classic CNN Colorizer model.")
            except Exception as e:
                logger.error(f"Failed to load Classic CNN Colorizer model: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.classic_model = None
        elif self.mode in ("none", "bw"):
            logger.info("Colorizer initialized in 'none' mode. Colorization will be skipped.")

    def colorize_image(self, img_path: Path, output_path: Path, prompt: str = None) -> bool:
        if self.mode in ("none", "bw"):
            logger.info(f"Skipping colorization for {img_path.name} (B&W mode)")
            return True

        if self.mode == "stable_diffusion" and self.pipe is None:
            logger.warning("Stable Diffusion colorizer model not initialized. Skipping colorization.")
            return False

        if self.mode == "classic" and self.classic_model is None:
            logger.warning("Classic CNN colorizer model not initialized. Skipping colorization.")
            return False

        try:
            if self.mode == "stable_diffusion":
                # Fallback prompt if none provided
                if not prompt:
                    prompt = "colored manga panel, anime style, highly detailed"
                
                # 1. Load original image in BGR
                orig_bgr = cv2.imread(str(img_path))
                if orig_bgr is None:
                    logger.error(f"Failed to load image from {img_path}")
                    return False
                    
                orig_h, orig_w = orig_bgr.shape[:2]
                
                # 2. Convert original to Lab color space and extract L channel (luminance / line-art)
                orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2Lab)
                orig_l = orig_lab[:, :, 0]
                
                # 3. Resize for Stable Diffusion processing (512x512 is standard/efficient)
                sd_size = (512, 512)
                resized_bgr = cv2.resize(orig_bgr, sd_size, interpolation=cv2.INTER_AREA)
                
                # 4. Extract Canny edges from resized grayscale image (white lines on black bg)
                resized_gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(resized_gray, 50, 150)
                edges_3ch = np.stack([edges, edges, edges], axis=-1)
                control_image = Image.fromarray(edges_3ch)
                
                # 5. Run pipeline
                logger.info(f"Colorizing panel using prompt: '{prompt}'")
                generator = torch.Generator().manual_seed(42)
                
                result_img = self.pipe(
                    prompt=prompt,
                    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, grayscale, black and white, sketch",
                    image=control_image,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=1.0,
                    generator=generator
                ).images[0]
                
                # 6. Convert PIL output to BGR OpenCV array
                result_bgr = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
                
                # 7. Resize output BGR back to original size
                result_resized = cv2.resize(result_bgr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                
                # 8. Convert back-resized image to Lab
                result_lab = cv2.cvtColor(result_resized, cv2.COLOR_BGR2Lab)
                
                # 9. Blend: combine original L (preserves sharp lines) with predicted ab channels (colors)
                merged_lab = np.zeros_like(orig_lab)
                merged_lab[:, :, 0] = orig_l
                merged_lab[:, :, 1] = result_lab[:, :, 1]
                merged_lab[:, :, 2] = result_lab[:, :, 2]
                
                # 10. Convert Lab to BGR and write
                colored_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_Lab2BGR)
                cv2.imwrite(str(output_path), colored_bgr)
                
                logger.info(f"Successfully colorized and saved: {output_path.name}")
                return True

            elif self.mode == "classic":
                # Use SIGGRAPH 2017 classic CNN colorizer
                from colorization_model.colorizers import load_img, preprocess_img, postprocess_tens
                
                # Preprocess image
                img_rgb = load_img(str(img_path))
                tens_l_orig, tens_l_rs = preprocess_img(img_rgb, HW=(256, 256))
                
                if self.device.type == "cuda":
                    tens_l_rs = tens_l_rs.cuda()
                
                # Run forward pass through the model
                with torch.no_grad():
                    out_ab = self.classic_model(tens_l_rs).cpu()
                
                # Postprocess
                out_img_rgb = postprocess_tens(tens_l_orig, out_ab)
                
                # Convert float RGB to uint8 BGR for saving with OpenCV
                out_img_rgb_uint8 = (out_img_rgb * 255).astype(np.uint8)
                out_img_bgr = cv2.cvtColor(out_img_rgb_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), out_img_bgr)
                
                logger.info(f"Successfully colorized using Classic CNN and saved: {output_path.name}")
                return True

        except Exception as e:
            logger.error(f"Failed to colorize image {img_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
