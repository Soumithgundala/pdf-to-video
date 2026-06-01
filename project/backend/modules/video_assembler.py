"""
Phase 4: Video Assembly
Uses MoviePy to create polished videos with Ken Burns effects.
"""
import random
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image
import logging
import cv2

logger = logging.getLogger(__name__)

import config as app_config

os.environ.setdefault("IMAGEIO_FFMPEG_EXE", app_config.FFMPEG_BINARY)

try:
    # MoviePy v2
    from moviepy import (
        ImageClip, AudioFileClip, CompositeAudioClip,
        concatenate_videoclips, concatenate_audioclips, ColorClip,
        TextClip, CompositeVideoClip
    )
except ImportError as e:
    logger.warning(f"Could not import moviepy: {e}")
    ImageClip = None
    AudioFileClip = None
    CompositeAudioClip = None
    concatenate_videoclips = None
    concatenate_audioclips = None
    ColorClip = None
    TextClip = None
    CompositeVideoClip = None


@dataclass
class VideoPartConfig:
    """Configuration for a single video part."""
    part_number: int
    voiceover_path: Path
    audio_duration_ms: int
    panels: List[Path]
    background_music_path: Optional[Path]
    output_path: Path


@dataclass
class GeneratedVideo:
    """Result of video generation."""
    part_number: int
    video_path: Path
    duration_seconds: float


class KenBurnsEffect:
    """Generate Ken Burns (pan/zoom) effects for images."""

    def __init__(
        self,
        zoom_range: Tuple[float, float] = (1.0, 1.2),
        pan_intensity: float = 0.1
    ):
        """
        Initialize Ken Burns effect generator.

        Args:
            zoom_range: (min_zoom, max_zoom) for zoom effect
            pan_intensity: How much to pan (0.0 to 1.0)
        """
        self.zoom_range = zoom_range
        self.pan_intensity = pan_intensity

    def get_random_effect(self) -> dict:
        """Get random Ken Burns parameters."""
        zoom = random.uniform(*self.zoom_range)
        pan_x = random.uniform(-self.pan_intensity, self.pan_intensity)
        pan_y = random.uniform(-self.pan_intensity, self.pan_intensity)

        # Determine direction (positive or negative zoom)
        if random.random() > 0.5:
            # Zoom in
            start_zoom = 1.0
            end_zoom = zoom
        else:
            # Zoom out
            start_zoom = zoom
            end_zoom = 1.0

        return {
            "start_zoom": start_zoom,
            "end_zoom": end_zoom,
            "start_x": pan_x,
            "start_y": pan_y,
            "end_x": -pan_x,
            "end_y": -pan_y
        }


class VideoAssembler:
    """Assembles final videos with effects and audio."""
    _nvenc_available = None

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        resolution: Tuple[int, int] = (app_config.VIDEO_WIDTH, app_config.VIDEO_HEIGHT),
        fps: int = app_config.VIDEO_FPS,
        background_music_volume: float = app_config.BACKGROUND_MUSIC_VOLUME
    ):
        """
        Initialize Video Assembler.

        Args:
            output_dir: Directory to save videos
            resolution: Output resolution (width, height)
            fps: Frames per second
            background_music_volume: Volume for background music (0.0 to 1.0)
        """
        self.output_dir = output_dir or Path(app_config.WORKSPACE_DIR) / "videos"
        self.resolution = resolution
        self.fps = fps
        self.background_music_volume = background_music_volume

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ken_burns = KenBurnsEffect()
        logger.info("MoviePy ffmpeg binary: %s", app_config.FFMPEG_BINARY)

        # Check NVENC capability once
        if VideoAssembler._nvenc_available is None:
            VideoAssembler._nvenc_available = self._test_nvenc()

        if VideoAssembler._nvenc_available:
            self.default_codec = app_config.VIDEO_CODEC
            self.default_preset = app_config.VIDEO_PRESET
            logger.info("Using NVIDIA hardware encoding (h264_nvenc)")
        else:
            self.default_codec = app_config.VIDEO_FALLBACK_CODEC
            self.default_preset = app_config.VIDEO_FALLBACK_PRESET
            logger.info("NVIDIA hardware encoding not available. Using CPU software encoding (libx264)")

    def _test_nvenc(self) -> bool:
        """Test if h264_nvenc is actually usable on this hardware."""
        import tempfile
        import subprocess
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_out = Path(tmpdir) / "test_nvenc.mp4"
            try:
                # Generate a 1-second 64x64 video using h264_nvenc
                cmd = [
                    app_config.FFMPEG_BINARY,
                    "-y",
                    "-f", "lavfi",
                    "-i", "color=c=black:s=64x64:d=1",
                    "-c:v", "h264_nvenc",
                    str(temp_out)
                ]
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
                return result.returncode == 0
            except Exception:
                return False

    def assemble_video(
        self,
        config: VideoPartConfig
    ) -> GeneratedVideo:
        """
        Assemble a single video part.

        Args:
            config: VideoPartConfig with all necessary paths and settings

        Returns:
            GeneratedVideo with result
        """
        logger.info(f"Assembling video part {config.part_number}...")

        # Calculate timing for each panel
        panels_count = len(config.panels)
        panel_durations = self._calculate_panel_durations(
            config.audio_duration_ms,
            panels_count
        )

        # Create video clips for each panel
        video_clips = []
        current_time = 0

        for i, (panel_path, duration_ms) in enumerate(zip(config.panels, panel_durations)):
            duration_seconds = duration_ms / 1000

            clip = self._create_panel_clip(
                panel_path,
                duration_seconds,
                start_time=current_time
            )
            video_clips.append(clip)
            current_time += duration_seconds

        # Concatenate all clips
        final_video = concatenate_videoclips(video_clips, method="compose")

        # Add audio
        final_video = self._add_audio(
            final_video,
            config.voiceover_path,
            config.background_music_path
        )

        # Pad if needed to meet minimum duration
        total_duration = final_video.duration
        if total_duration < app_config.TARGET_AUDIO_DURATION_SECONDS:
            final_video = self._add_end_card(
                final_video,
                config.part_number,
                app_config.TARGET_AUDIO_DURATION_SECONDS
            )

        # Write output using the cached default codec (NVENC or CPU)
        output_path = config.output_path
        tmp_output_path = output_path.with_suffix(".tmp.mp4")

        if tmp_output_path.exists():
            tmp_output_path.unlink()

        try:
            logger.info(
                "Encoding video part %s with %s...",
                config.part_number,
                self.default_codec
            )
            self._write_videofile_with_watchdog(
                final_video,
                tmp_output_path,
                part_number=config.part_number,
                codec=self.default_codec,
                preset=self.default_preset
            )
            self._validate_video_file(tmp_output_path)
            tmp_output_path.replace(output_path)
        except Exception as encode_error:
            logger.warning(
                "Encoding failed for video part %s using %s: %s",
                config.part_number,
                self.default_codec,
                encode_error
            )
            if tmp_output_path.exists():
                tmp_output_path.unlink()

            fallback_codec = app_config.VIDEO_FALLBACK_CODEC
            fallback_preset = app_config.VIDEO_FALLBACK_PRESET
            
            if self.default_codec == fallback_codec:
                raise encode_error

            logger.info(
                "Retrying video part %s with %s software encoding...",
                config.part_number,
                fallback_codec
            )
            self._write_videofile_with_watchdog(
                final_video,
                tmp_output_path,
                part_number=config.part_number,
                codec=fallback_codec,
                preset=fallback_preset
            )
            self._validate_video_file(tmp_output_path)
            tmp_output_path.replace(output_path)

        total_duration = final_video.duration

        # Cleanup
        final_video.close()

        logger.info(f"Video part {config.part_number} saved: {output_path}")

        return GeneratedVideo(
            part_number=config.part_number,
            video_path=output_path,
            duration_seconds=total_duration
        )

    def _validate_video_file(self, video_path: Path) -> None:
        """Raise if ffprobe cannot read the newly encoded MP4."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            logger.warning("ffprobe not found; skipping encoded video validation")
            return

        if result.returncode != 0 or not result.stdout.strip():
            raise ValueError(
                f"Encoded video is not playable: {video_path} {result.stderr.strip()}"
            )

    def _write_videofile_with_watchdog(
        self,
        video,
        output_path: Path,
        *,
        part_number: int,
        codec: str,
        preset: str
    ) -> None:
        stop_event = threading.Event()
        watchdog = threading.Thread(
            target=self._watch_encode_progress,
            args=(output_path, part_number, codec, stop_event),
            name=f"video-encode-watchdog-part-{part_number}",
            daemon=True
        )
        watchdog.start()

        try:
            video.write_videofile(
                str(output_path),
                fps=self.fps,
                codec=codec,
                audio_codec='aac',
                bitrate=app_config.VIDEO_BITRATE,
                preset=preset,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                logger=None
            )
        finally:
            stop_event.set()
            watchdog.join(timeout=1)

    def _watch_encode_progress(
        self,
        output_path: Path,
        part_number: int,
        codec: str,
        stop_event: threading.Event
    ) -> None:
        timeout_seconds = app_config.VIDEO_ENCODE_STALL_TIMEOUT_SECONDS
        check_seconds = app_config.VIDEO_ENCODE_STALL_CHECK_SECONDS
        last_size = -1
        last_progress_time = time.monotonic()

        while not stop_event.wait(check_seconds):
            current_size = output_path.stat().st_size if output_path.exists() else 0
            if current_size != last_size:
                last_size = current_size
                last_progress_time = time.monotonic()
                logger.info(
                    "Encoding part %s with %s: %.2f MB written",
                    part_number,
                    codec,
                    current_size / (1024 * 1024)
                )
                continue

            stalled_for = time.monotonic() - last_progress_time
            if stalled_for >= timeout_seconds:
                if codec == "libx264":
                    logger.warning(
                        "Encoding part %s with %s stalled for %.0fs at %s bytes. Skipping crash because it is software encoding (Windows caching).",
                        part_number,
                        codec,
                        stalled_for,
                        current_size
                    )
                    last_progress_time = time.monotonic()
                else:
                    logger.critical(
                        "Encoding part %s with %s stalled for %.0fs at %s bytes. Crashing backend.",
                        part_number,
                        codec,
                        stalled_for,
                        current_size
                    )
                    os._exit(1)

    def _calculate_panel_durations(
        self,
        total_duration_ms: int,
        num_panels: int,
        min_duration_ms: int = 3000,
        max_duration_ms: int = 10000
    ) -> List[int]:
        """Calculate duration for each panel."""
        base_duration = total_duration_ms // num_panels
        base_duration = max(min_duration_ms, min(base_duration, max_duration_ms))

        durations = [base_duration] * num_panels
        total = sum(durations)
        diff = total_duration_ms - total

        # Distribute difference
        for i in range(abs(diff) // 100):
            if diff > 0:
                durations[i % num_panels] += 100
            else:
                durations[i % num_panels] = max(1000, durations[i % num_panels] - 100)

        return durations

    def _create_panel_clip(
        self,
        panel_path: Path,
        duration: float,
        start_time: float = 0
    ) -> ImageClip:
        """Create a video clip for a single panel with Ken Burns effect."""
        # Load image and scale to cover size once using OpenCV
        img = cv2.imread(str(panel_path))
        if img is None:
            raise ValueError(f"Failed to load image: {panel_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h_orig, w_orig = img.shape[:2]
        scale = self._get_resize_factor((w_orig, h_orig))
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create base clip
        clip = ImageClip(img_resized, duration=duration)

        # Apply Ken Burns effect
        effect_params = self.ken_burns.get_random_effect()
        target_w, target_h = self.resolution

        # Apply zoom effect
        def zoom_effect(get_frame, t):
            frame = get_frame(t)
            progress = t / duration

            current_zoom = effect_params["start_zoom"] + (
                effect_params["end_zoom"] - effect_params["start_zoom"]
            ) * progress

            h, w = frame.shape[:2]
            new_h, new_w = int(h / current_zoom), int(w / current_zoom)
            
            new_h = min(new_h, h)
            new_w = min(new_w, w)
            
            y_start = int((h - new_h) * (0.5 + effect_params["start_y"] * progress))
            x_start = int((w - new_w) * (0.5 + effect_params["start_x"] * progress))
            
            y_start = max(0, min(y_start, h - new_h))
            x_start = max(0, min(x_start, w - new_w))

            cropped = frame[
                y_start:y_start + new_h,
                x_start:x_start + new_w
            ]

            if cropped.shape[:2] != (h, w):
                cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            return cropped

        clip = clip.transform(zoom_effect)
        clip.size = (target_w, target_h) # Set clip size to final target size

        # Set start time
        clip = clip.with_start(start_time)

        return clip

    def _get_resize_factor(self, img_size: Tuple[int, int]) -> float:
        """Calculate resize factor to fit video resolution."""
        target_w, target_h = self.resolution
        img_w, img_h = img_size

        # Calculate scale to cover target resolution
        scale_w = target_w / img_w
        scale_h = target_h / img_h

        # Use larger scale to ensure coverage
        return max(scale_w, scale_h)

    def _add_audio(
        self,
        video,
        voiceover_path: Path,
        background_music_path: Optional[Path]
    ):
        """Add voiceover and background music to video."""
        # Load voiceover
        voiceover = AudioFileClip(str(voiceover_path))

        audio_clips = [voiceover]

        # Add background music if provided
        if background_music_path and background_music_path.exists():
            bg_music = AudioFileClip(str(background_music_path))
            bg_music = bg_music.with_volume_scaled(self.background_music_volume)

            # Loop or trim to match video duration
            if bg_music.duration < video.duration:
                # Loop music
                num_loops = int(video.duration / bg_music.duration) + 1
                bg_music = concatenate_audioclips([bg_music] * num_loops)

            bg_music = bg_music.subclipped(0, video.duration)
            audio_clips.append(bg_music)

        # Combine audio
        final_audio = CompositeAudioClip(audio_clips)
        final_audio = final_audio.with_duration(video.duration)

        return video.with_audio(final_audio)

    def _add_end_card(
        self,
        video,
        current_part: int,
        target_duration: float
    ):
        """Add end card with call to action."""
        remaining_duration = target_duration - video.duration

        if remaining_duration <= 0:
            return video

        # Create end card
        next_part = current_part + 1 if current_part < 4 else 1
        text = f"Like for Part {next_part}"

        # Create text clip
        end_card = ColorClip(
            size=self.resolution,
            color=(10, 20, 40),
            duration=remaining_duration
        )

        try:
            text_clip = TextClip(
                text=text,
                font_size=60,
                color='white',
                bg_color=None,
                size=self.resolution
            ).with_duration(remaining_duration)
            end_card = CompositeVideoClip([end_card, text_clip])
        except Exception:
            # Fallback if textclip fails
            pass

        # Concatenate
        return concatenate_videoclips([video, end_card], method="compose")

    def assemble_all_videos(
        self,
        configs: List[VideoPartConfig]
    ) -> List[GeneratedVideo]:
        """Assemble all video parts."""
        results = []

        for video_config in configs:
            result = self.assemble_video(video_config)
            results.append(result)

        return results
