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
from typing import List, Tuple, Optional, Dict
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
    import moviepy.video.fx as vfx
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
    vfx = None

from typing import Any

DEFAULT_ASSETS = {
    "music": {
        "sad_violin": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3",
        "upbeat_adventure": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        "dramatic": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
        "binks_brew": "",
        "drum_of_libration": "",
        "yo_ho_ho_ho": ""
    },
    "sfx": {
        "yohoho": "https://www.myinstants.com/media/sounds/yo-ho-ho.mp3",
        "sword_clash": "https://www.soundboard.com/handler/DownLoadTrack.ashx?cliptrackid=274987"
    }
}

def download_default_asset(asset_type: str, asset_name: str) -> Optional[Path]:
    import urllib.request
    
    # Base directory for local assets
    base_assets_dir = Path(__file__).parent.parent / "assets"
    target_dir = base_assets_dir / asset_type
    target_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = target_dir / f"{asset_name}.mp3"
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path
        
    url = DEFAULT_ASSETS.get(asset_type, {}).get(asset_name)
    if not url:
        return None
        
    logger.info(f"Downloading default asset {asset_name} from {url}...")
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response, open(target_path, 'wb') as out_file:
            out_file.write(response.read())
        logger.info(f"Successfully downloaded {asset_name} to {target_path}")
        return target_path
    except Exception as e:
        logger.warning(f"Failed to download default asset {asset_name} from {url}: {e}. Proceeding without this asset.")
        if target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass
        return None


@dataclass
class VideoPartConfig:
    """Configuration for a single video part."""
    part_number: int
    voiceover_path: Path
    audio_duration_ms: int
    panels: List[Path]
    background_music_path: Optional[Path]
    output_path: Path
    focus_areas: Optional[Dict[str, List[int]]] = None
    script_segments: Optional[List[Dict[str, str]]] = None
    word_boundaries: Optional[List[Dict[str, Any]]] = None
    music_mood: Optional[str] = None
    sound_effects: Optional[List[Dict[str, str]]] = None


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
                # Generate a 1-second 320x240 video using h264_nvenc
                cmd = [
                    app_config.FFMPEG_BINARY,
                    "-y",
                    "-f", "lavfi",
                    "-i", "color=c=black:s=320x240:d=1",
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
        Assemble a single video part with dynamic timing, captions, and SFX/music.
        """
        logger.info(f"Assembling video part {config.part_number}...")

        # Resolve background music path based on music_mood if not provided
        bg_music_path = config.background_music_path
        if not bg_music_path and config.music_mood:
            bg_music_path = download_default_asset("music", config.music_mood)

        # Download SFX assets
        sfx_list = []
        if config.sound_effects:
            for sfx in config.sound_effects:
                sfx_name = sfx.get("effect")
                panel_id = sfx.get("panel_id")
                if sfx_name and panel_id:
                    sfx_path = download_default_asset("sfx", sfx_name)
                    if sfx_path:
                        sfx_list.append((sfx_path, panel_id))

        # Calculate precise video duration based on the end of the last word (plus 0.3s padding for natural cutoff)
        # This completely eliminates trailing silence ("dead air") from the video.
        if config.word_boundaries:
            last_word_end = config.word_boundaries[-1]["start"] + config.word_boundaries[-1]["duration"]
            target_duration = last_word_end + 0.3
            logger.info(f"Target video duration set to {target_duration:.2f}s based on final word boundary (raw audio duration: {config.audio_duration_ms/1000.0:.2f}s)")
        else:
            target_duration = config.audio_duration_ms / 1000.0

        # Calculate timing for each panel
        panels_count = len(config.panels)
        
        # Check if we have word boundaries and segments to calculate dynamic durations
        if config.word_boundaries and config.script_segments:
            # We calculate dynamic durations in seconds, then convert to ms
            durations_sec = self._calculate_dynamic_panel_durations(config, target_duration)
            panel_durations = [int(d * 1000) for d in durations_sec]
        else:
            panel_durations = self._calculate_panel_durations(
                int(target_duration * 1000),
                panels_count
            )

        # Create video clips for each panel
        video_clips = []
        current_time = 0.0
        transition_duration = 0.3  # 300 ms crossfade transition
        panel_start_times = {}

        for i, (panel_path, duration_ms) in enumerate(zip(config.panels, panel_durations)):
            duration_seconds = duration_ms / 1000

            # Extend clip duration to account for transition overlaps
            clip_duration = duration_seconds
            if i < panels_count - 1 and panels_count > 1:
                clip_duration += transition_duration

            # Extract panel ID from filename (e.g. "P1" from "panel_P1.png")
            import re
            m = re.search(r'panel_(P\d+)', panel_path.name)
            panel_id = m.group(1) if m else f"P{i+1}"
            
            panel_start_times[panel_id] = current_time
            
            focus_box = None
            if config.focus_areas and panel_id in config.focus_areas:
                focus_box = config.focus_areas[panel_id]

            clip = self._create_panel_clip(
                panel_path,
                clip_duration,
                start_time=current_time,
                focus_box=focus_box
            )
            
            if i > 0 and vfx is not None:
                clip = clip.with_effects([vfx.CrossFadeIn(transition_duration)])

            video_clips.append(clip)
            
            # Next clip starts with transition_duration overlap
            current_time += duration_seconds

        # Concatenate all clips with transition overlaps
        if vfx is not None and panels_count > 1:
            final_video = concatenate_videoclips(video_clips, method="compose", padding=-transition_duration)
        else:
            final_video = concatenate_videoclips(video_clips, method="compose")

        # Mix sound effects with their calculated start times
        mixed_sfx = []
        for sfx_path, panel_id in sfx_list:
            if panel_id in panel_start_times:
                mixed_sfx.append((sfx_path, panel_start_times[panel_id]))

        # Add audio
        final_video = self._add_audio_enhanced(
            final_video,
            config.voiceover_path,
            bg_music_path,
            mixed_sfx
        )

        # Add captions/subtitles if word boundaries are present
        if config.word_boundaries:
            logger.info("Adding dynamic auto-captions to video...")
            caption_clips = self._create_caption_clips(config)
            if caption_clips:
                final_video = CompositeVideoClip([final_video] + caption_clips)

        # Add Outro Call to Action Overlay (Like for Part X) in the last 4 seconds
        target_w, target_h = self.resolution
        outro_duration = min(4.0, final_video.duration)
        if outro_duration > 0:
            next_part = config.part_number + 1 if config.part_number < 3 else 1
            outro_words = [{"word": "Like"}, {"word": "for"}, {"word": f"Part {next_part}"}]
            try:
                outro_clip = self._create_rich_text_clip(
                    phrase_words=outro_words,
                    active_idx=2,
                    font_name="arialbd",
                    font_size=65,
                    stroke_width=4.0,
                    stroke_color=(0, 0, 0, 255),
                    size=(target_w - 100, 200)
                ).with_duration(outro_duration).with_start(final_video.duration - outro_duration).with_position(("center", 300))
                
                final_video = CompositeVideoClip([final_video, outro_clip])
                logger.info(f"Added outro call to action: Like for Part {next_part}")
            except Exception as outro_err:
                logger.warning(f"Failed to create outro TextClip: {outro_err}")

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
            temp_audio_path = output_path.parent / f"part_{part_number}_temp_audio.mp4"
            video.write_videofile(
                str(output_path),
                fps=self.fps,
                codec=codec,
                audio_codec='aac',
                temp_audiofile=str(temp_audio_path),
                bitrate=app_config.VIDEO_BITRATE,
                preset=preset,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                logger=None
            )
        finally:
            stop_event.set()
            watchdog.join(timeout=1)

    def _kill_child_processes(self) -> None:
        """Find and kill all child processes (like ffmpeg) of the current Python process."""
        import os
        import subprocess
        current_pid = os.getpid()
        logger.info("Attempting to kill child processes of parent PID %s", current_pid)
        try:
            # Query child processes using wmic
            cmd = f"wmic process where (ParentProcessId={current_pid}) get ProcessId"
            output = subprocess.check_output(cmd, shell=True, text=True)
            pids = []
            for line in output.splitlines():
                val = line.strip()
                if val.isdigit():
                    pids.append(int(val))
            
            for pid in pids:
                logger.warning("Terminating stalled child process PID %s", pid)
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
        except Exception as e:
            logger.error("Failed to kill child processes via wmic: %s", e)
            # Fallback: kill all ffmpeg.exe processes
            try:
                subprocess.run("taskkill /F /IM ffmpeg.exe", shell=True, capture_output=True)
            except Exception as e2:
                logger.error("Fallback taskkill failed: %s", e2)

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
                        "Encoding part %s with %s stalled for %.0fs at %s bytes. Killing ffmpeg subprocess to abort job gracefully.",
                        part_number,
                        codec,
                        stalled_for,
                        current_size
                    )
                    self._kill_child_processes()
                    # Sleep briefly to ensure taskkill took effect
                    time.sleep(2)

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

    def _calculate_dynamic_panel_durations(self, config: VideoPartConfig, target_duration: float) -> List[float]:
        panels_count = len(config.panels)
        if panels_count == 0:
            return []

        # Fallback to even distribution if no word boundaries
        if not config.word_boundaries or not config.script_segments:
            logger.info("No word boundaries or script segments. Falling back to even panel duration distribution.")
            durations_ms = self._calculate_panel_durations(
                int(target_duration * 1000),
                panels_count
            )
            return [d / 1000.0 for d in durations_ms]

        # Map panel files to panel IDs
        import re
        panel_id_to_idx = {}
        for idx, panel_path in enumerate(config.panels):
            m = re.search(r'panel_(P\d+)', panel_path.name)
            pid = m.group(1) if m else f"P{idx+1}"
            panel_id_to_idx[pid] = idx

        durations = [0.0] * panels_count
        words = config.word_boundaries
        word_idx = 0
        total_words = len(words)
        current_start = 0.0
        
        for seg_idx, segment in enumerate(config.script_segments):
            pid = segment.get("panel_id")
            text = segment.get("text", "")
            p_idx = panel_id_to_idx.get(pid, seg_idx)
            if p_idx >= panels_count:
                p_idx = panels_count - 1
                
            seg_word_count = len(text.split())
            if seg_word_count == 0:
                continue
                
            seg_words = words[word_idx:word_idx + seg_word_count]
            word_idx += seg_word_count
            
            if seg_words:
                last_word = seg_words[-1]
                current_end = last_word["start"] + last_word["duration"]
            else:
                current_end = current_start + 4.0
                
            if seg_idx == len(config.script_segments) - 1 and total_words > 0:
                last_w = words[-1]
                current_end = max(current_end, last_w["start"] + last_w["duration"])
                
            durations[p_idx] = max(1.5, current_end - current_start)
            current_start = current_end
            
        for idx in range(panels_count):
            if durations[idx] == 0.0:
                durations[idx] = 4.0
                
        sum_durations = sum(durations)
        if sum_durations > 0:
            scale = target_duration / sum_durations
            durations = [d * scale for d in durations]
                
        logger.info(f"Dynamic panel durations (seconds): {durations}")
        return durations

    def _group_words_into_phrases(self, word_boundaries: List[Dict[str, Any]], max_words: int = 4) -> List[List[Dict[str, Any]]]:
        if not word_boundaries:
            return []
        phrases = []
        current_phrase = []
        for word in word_boundaries:
            if not current_phrase:
                current_phrase.append(word)
                continue
                
            last_word = current_phrase[-1]
            pause = word["start"] - (last_word["start"] + last_word["duration"])
            
            if len(current_phrase) >= max_words or pause > 0.3:
                phrases.append(current_phrase)
                current_phrase = [word]
            else:
                current_phrase.append(word)
        if current_phrase:
            phrases.append(current_phrase)
        return phrases

    def _create_rich_text_clip(
        self,
        phrase_words: List[Dict[str, Any]],
        active_idx: Optional[int],
        font_name: str,
        font_size: int,
        stroke_width: float,
        stroke_color: tuple,
        size: Tuple[int, int]
    ) -> ImageClip:
        """Create a transparent ImageClip of text with a highlighted active word using Pillow."""
        from PIL import Image, ImageDraw, ImageFont
        
        width, height = size
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Resolve font
        try:
            font = ImageFont.truetype(font_name, font_size)
        except Exception:
            try:
                font = ImageFont.truetype(f"{font_name}.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
                
        # Calculate space width
        space_bbox = draw.textbbox((0, 0), " ", font=font)
        space_w = space_bbox[2] - space_bbox[0]
        
        # Calculate sizes of each word
        word_sizes = []
        total_w = 0
        max_h = 0
        for w in phrase_words:
            text = w["word"]
            bbox = draw.textbbox((0, 0), text, font=font)
            w_width = bbox[2] - bbox[0]
            w_height = bbox[3] - bbox[1]
            word_sizes.append((w_width, w_height))
            total_w += w_width
            if w_height > max_h:
                max_h = w_height
                
        total_w += space_w * (len(phrase_words) - 1)
        
        # Centering coordinates
        x = (width - total_w) // 2
        y = (height - max_h) // 2
        
        # Draw each word with its corresponding color
        for idx, w in enumerate(phrase_words):
            text = w["word"]
            w_w, w_h = word_sizes[idx]
            
            if active_idx is not None and idx == active_idx:
                color = (250, 204, 21, 255)  # Bright yellow
            else:
                color = (255, 255, 255, 255)  # White
                
            draw.text(
                (x, y),
                text,
                font=font,
                fill=color,
                stroke_width=int(stroke_width),
                stroke_fill=stroke_color
            )
            x += w_w + space_w
            
        return ImageClip(np.array(img), transparent=True)

    def _create_caption_clips(self, config: VideoPartConfig) -> List[Any]:
        if not config.word_boundaries:
            return []
            
        phrases = self._group_words_into_phrases(config.word_boundaries)
        caption_clips = []
        target_w, target_h = self.resolution
        
        for phrase in phrases:
            phrase_start = phrase[0]["start"]
            phrase_end = phrase[-1]["start"] + phrase[-1]["duration"]
            n_words = len(phrase)
            
            for idx, active_word in enumerate(phrase):
                win_start = phrase_start if idx == 0 else active_word["start"]
                win_end = phrase_end if idx == n_words - 1 else phrase[idx+1]["start"]
                win_dur = win_end - win_start
                if win_dur <= 0:
                    continue
                    
                try:
                    tc = self._create_rich_text_clip(
                        phrase_words=phrase,
                        active_idx=idx,
                        font_name="arialbd",
                        font_size=55,
                        stroke_width=3.5,
                        stroke_color=(0, 0, 0, 255),
                        size=(target_w - 100, 300)
                    ).with_duration(win_dur).with_start(win_start).with_position(("center", target_h - 450))
                    
                    caption_clips.append(tc)
                except Exception as e:
                    logger.warning(f"Failed to create caption TextClip: {e}")
                    
        return caption_clips

    def _create_panel_clip(
        self,
        panel_path: Path,
        duration: float,
        start_time: float = 0,
        focus_box: Optional[List[int]] = None
    ) -> ImageClip:
        """Create a video clip for a single panel with Ken Burns effect centered on the focus box."""
        # Load image and scale to cover size once using OpenCV
        img = cv2.imread(str(panel_path))
        if img is None:
            raise ValueError(f"Failed to load image: {panel_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h_orig, w_orig = img.shape[:2]
        
        # Calculate pixel coordinates for the focus center in original image
        if focus_box and len(focus_box) == 4:
            ymin, xmin, ymax, xmax = focus_box
            ymin_px = int(ymin * h_orig / 1000)
            xmin_px = int(xmin * w_orig / 1000)
            ymax_px = int(ymax * h_orig / 1000)
            xmax_px = int(xmax * w_orig / 1000)
            
            # Clamp focus box pixels to image bounds
            ymin_px = max(0, min(ymin_px, h_orig))
            xmin_px = max(0, min(xmin_px, w_orig))
            ymax_px = max(ymin_px, min(ymax_px, h_orig))
            xmax_px = max(xmin_px, min(xmax_px, w_orig))
            
            center_x = (xmin_px + xmax_px) // 2
            center_y = (ymin_px + ymax_px) // 2
        else:
            center_x = w_orig // 2
            center_y = h_orig // 2

        scale = self._get_resize_factor((w_orig, h_orig))
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create base clip
        clip = ImageClip(img_resized, duration=duration)

        # Apply Ken Burns effect
        effect_params = self.ken_burns.get_random_effect()
        target_w, target_h = self.resolution

        # Project center coordinates to the resized image
        resized_center_x = int(center_x * scale)
        resized_center_y = int(center_y * scale)

        # Apply zoom effect
        def zoom_effect(get_frame, t):
            frame = get_frame(t)
            progress = t / duration

            # Quadratic ease-in-out curve for organic acceleration/deceleration
            eased_progress = 2 * progress * progress if progress < 0.5 else 1 - (-2 * progress + 2)**2 / 2

            current_zoom = effect_params["start_zoom"] + (
                effect_params["end_zoom"] - effect_params["start_zoom"]
            ) * eased_progress

            h, w = frame.shape[:2]
            new_h, new_w = int(h / current_zoom), int(w / current_zoom)
            
            new_h = min(new_h, h)
            new_w = min(new_w, w)
            
            # Calculate base crop start to center on the focus center
            x_start_base = resized_center_x - new_w // 2
            y_start_base = resized_center_y - new_h // 2
            
            # Apply subtle pan relative to the focus center (limited to 10% of remaining slack)
            pan_limit_x = int((w - new_w) * 0.1)
            pan_limit_y = int((h - new_h) * 0.1)
            
            pan_x = int(effect_params["start_x"] * pan_limit_x * (1 - 2 * eased_progress))
            pan_y = int(effect_params["start_y"] * pan_limit_y * (1 - 2 * eased_progress))
            
            x_start = x_start_base + pan_x
            y_start = y_start_base + pan_y
            
            # Clamp crop window to valid frame bounds
            x_start = max(0, min(x_start, w - new_w))
            y_start = max(0, min(y_start, h - new_h))

            cropped = frame[
                y_start:y_start + new_h,
                x_start:x_start + new_w
            ]

            if cropped.shape[:2] != (target_h, target_w):
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

    def _add_audio_enhanced(
        self,
        video,
        voiceover_path: Path,
        background_music_path: Optional[Path],
        sfx_list: List[Tuple[Path, float]]
    ):
        """Add voiceover, background music with dynamic ducking, and sound effects to video."""
        # Load voiceover and trim to match video duration (removing trailing dead silence)
        voiceover = AudioFileClip(str(voiceover_path))
        if voiceover.duration > video.duration:
            voiceover = voiceover.subclipped(0, video.duration)

        audio_clips = [voiceover]

        # Add background music if provided
        if background_music_path and background_music_path.exists():
            bg_music = AudioFileClip(str(background_music_path))

            # Loop or trim to match video duration
            if bg_music.duration < video.duration:
                # Loop music
                num_loops = int(video.duration / bg_music.duration) + 1
                bg_music = concatenate_audioclips([bg_music] * num_loops)

            bg_music = bg_music.subclipped(0, video.duration)

            # Apply dynamic audio ducking
            try:
                # Set up low-resolution sampling of voiceover volume
                fps = 1000  # 1kHz is plenty for volume envelope
                voice_data = voiceover.to_soundarray(fps=fps)
                
                # Take absolute maximum across channels
                if len(voice_data.shape) > 1 and voice_data.shape[1] > 1:
                    voice_data = np.abs(voice_data).max(axis=1)
                else:
                    voice_data = np.abs(voice_data).flatten()
                
                # Calculate smoothed envelope using attack/decay tracking
                envelope = np.zeros_like(voice_data)
                current_max = 0.0
                attack_decay = 0.1  # seconds
                decay_factor = np.exp(-1.0 / (fps * attack_decay))
                
                # Forward decay pass
                for idx in range(len(voice_data)):
                    current_max = max(voice_data[idx], current_max * decay_factor)
                    envelope[idx] = current_max
                    
                # Backward attack pass
                current_max = 0.0
                for idx in range(len(voice_data) - 1, -1, -1):
                    current_max = max(envelope[idx], current_max * decay_factor)
                    envelope[idx] = current_max

                # Normalize envelope to [0.0, 1.0]
                max_env = envelope.max()
                if max_env > 0:
                    envelope = envelope / max_env
                
                # Map envelope to ducking levels
                normal_vol = self.background_music_volume
                ducked_vol = max(0.01, normal_vol * 0.2)  # duck to 20% of normal volume
                
                threshold = 0.05
                ducking_levels = np.where(envelope > threshold, ducked_vol, normal_vol)
                
                # Smooth ducking levels to avoid volume pops
                smooth_window = int(0.15 * fps)
                if smooth_window > 1:
                    ducking_levels = np.convolve(
                        ducking_levels, 
                        np.ones(smooth_window) / smooth_window, 
                        mode='same'
                    )
                
                # Define volume scaling function
                def vol_func(t):
                    if isinstance(t, np.ndarray):
                        indices = (t * fps).astype(int)
                        indices = np.clip(indices, 0, len(ducking_levels) - 1)
                        return ducking_levels[indices]
                    else:
                        idx = int(t * fps)
                        idx = max(0, min(idx, len(ducking_levels) - 1))
                        return ducking_levels[idx]

                # Apply the volume scaling function using fl()
                def fl_audio(gf, t):
                    factor = vol_func(t)
                    frame = gf(t)
                    if isinstance(t, np.ndarray):
                        if len(frame.shape) > 1 and frame.shape[1] > 1:
                            return factor[:, None] * frame
                        return factor * frame
                    return factor * frame
                
                bg_music = bg_music.fl(fl_audio)
                logger.info("Dynamic audio ducking applied to background music.")
            except Exception as duck_err:
                logger.warning(f"Failed to apply dynamic audio ducking: {duck_err}. Falling back to flat volume.")
                bg_music = bg_music.with_volume_scaled(self.background_music_volume)

            audio_clips.append(bg_music)

        # Add sound effects
        if sfx_list:
            for sfx_path, sfx_start in sfx_list:
                try:
                    sfx_clip = AudioFileClip(str(sfx_path))
                    # Clamp/trim sound effect if it exceeds video duration
                    if sfx_start < video.duration:
                        sfx_clip = sfx_clip.with_start(sfx_start)
                        audio_clips.append(sfx_clip)
                        logger.info(f"Mixed sound effect {sfx_path.name} at t={sfx_start}s")
                except Exception as sfx_err:
                    logger.warning(f"Failed to mix sound effect {sfx_path.name}: {sfx_err}")

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
        next_part = current_part + 1 if current_part < 3 else 1
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
