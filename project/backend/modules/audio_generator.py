"""
Phase 3: Audio Generation
Uses Kokoro-82M for high-quality, local text-to-speech voiceovers.
"""
import asyncio
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import logging
import threading
import json
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded singleton for the Kokoro pipeline
_kokoro_pipeline = None
_kokoro_lock = threading.Lock()


def _get_kokoro_pipeline(lang_code: str = "a"):
    """Get or create the singleton Kokoro pipeline instance."""
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        with _kokoro_lock:
            if _kokoro_pipeline is None:
                try:
                    from kokoro import KPipeline
                    logger.info("Loading Kokoro-82M TTS pipeline (lang_code=%s)...", lang_code)
                    _kokoro_pipeline = KPipeline(lang_code=lang_code)
                    logger.info("Kokoro-82M pipeline loaded successfully")
                except ImportError:
                    raise ImportError(
                        "kokoro package not installed. Run: pip install kokoro soundfile"
                    )
    return _kokoro_pipeline


@dataclass
class GeneratedAudio:
    """Represents generated audio with metadata."""
    part_number: int
    audio_path: Path
    duration_ms: int
    script: str
    word_boundaries: Optional[List[dict]] = None

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000


# -----------------------------------------------------------------
# Voice catalogue
# -----------------------------------------------------------------

KOKORO_VOICES = {
    # American English — Female
    "af_alloy":   {"name": "Alloy",   "gender": "female", "accent": "American"},
    "af_aoede":   {"name": "Aoede",   "gender": "female", "accent": "American"},
    "af_bella":   {"name": "Bella",   "gender": "female", "accent": "American"},
    "af_heart":   {"name": "Heart",   "gender": "female", "accent": "American"},
    "af_jessica": {"name": "Jessica", "gender": "female", "accent": "American"},
    "af_kore":    {"name": "Kore",    "gender": "female", "accent": "American"},
    "af_nicole":  {"name": "Nicole",  "gender": "female", "accent": "American"},
    "af_nova":    {"name": "Nova",    "gender": "female", "accent": "American"},
    "af_river":   {"name": "River",   "gender": "female", "accent": "American"},
    "af_sarah":   {"name": "Sarah",   "gender": "female", "accent": "American"},
    "af_sky":     {"name": "Sky",     "gender": "female", "accent": "American"},
    # American English — Male
    "am_adam":    {"name": "Adam",    "gender": "male", "accent": "American"},
    "am_echo":    {"name": "Echo",    "gender": "male", "accent": "American"},
    "am_eric":    {"name": "Eric",    "gender": "male", "accent": "American"},
    "am_fenrir":  {"name": "Fenrir",  "gender": "male", "accent": "American"},
    "am_liam":    {"name": "Liam",    "gender": "male", "accent": "American"},
    "am_michael": {"name": "Michael", "gender": "male", "accent": "American"},
    "am_onyx":    {"name": "Onyx",    "gender": "male", "accent": "American"},
    "am_puck":    {"name": "Puck",    "gender": "male", "accent": "American"},
    # British English — Female
    "bf_alice":    {"name": "Alice",    "gender": "female", "accent": "British"},
    "bf_emma":     {"name": "Emma",     "gender": "female", "accent": "British"},
    "bf_isabella": {"name": "Isabella", "gender": "female", "accent": "British"},
    "bf_lily":     {"name": "Lily",     "gender": "female", "accent": "British"},
    # British English — Male
    "bm_daniel": {"name": "Daniel", "gender": "male", "accent": "British"},
    "bm_fable":  {"name": "Fable",  "gender": "male", "accent": "British"},
    "bm_george": {"name": "George", "gender": "male", "accent": "British"},
    "bm_lewis":  {"name": "Lewis",  "gender": "male", "accent": "British"},
}

SAMPLE_TEXT = "In a world of manga, every panel tells a story. Heroes rise, villains fall, and adventure awaits."

# Legacy compat alias
RECOMMENDED_VOICES = {
    "male": [v for v, m in KOKORO_VOICES.items() if m["gender"] == "male"],
    "female": [v for v, m in KOKORO_VOICES.items() if m["gender"] == "female"],
    "narrator": ["am_michael", "am_adam", "bm_george"],
}


class AudioGenerator:
    """Generates voiceover audio using Kokoro-82M."""

    def __init__(
        self,
        voice: str = "am_michael",
        output_dir: Optional[Path] = None,
        speed: float = 1.0,
    ):
        """
        Initialize Audio Generator.

        Args:
            voice: Kokoro voice ID (e.g. 'am_michael', 'af_heart')
            output_dir: Directory to save audio files
            speed: Speech speed multiplier (1.0 = normal)
        """
        import config as _cfg
        self.voice = voice
        self.output_dir = output_dir or Path(_cfg.WORKSPACE_DIR) / "audio"
        self.speed = speed

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine lang_code from voice prefix
        self._lang_code = self._voice_to_lang(voice)

    @staticmethod
    def _voice_to_lang(voice_id: str) -> str:
        """Map voice ID prefix to Kokoro lang_code."""
        prefix = voice_id[0] if voice_id else "a"
        mapping = {
            "a": "a",   # American English
            "b": "b",   # British English
            "j": "j",   # Japanese
            "z": "z",   # Mandarin Chinese
            "f": "f",   # French
            "h": "h",   # Hindi
            "i": "i",   # Italian
            "e": "e",   # Spanish
            "p": "p",   # Brazilian Portuguese
            "k": "k",   # Korean
        }
        return mapping.get(prefix, "a")

    def _generate_wav(self, text: str, voice: str, output_path: Path) -> Path:
        """Generate a WAV file from text using Kokoro."""
        import soundfile as sf

        pipeline = _get_kokoro_pipeline(self._lang_code)

        # Kokoro yields chunks — concatenate them into one array
        audio_chunks = []
        for _gs, _ps, audio_chunk in pipeline(text, voice=voice, speed=self.speed):
            if audio_chunk is not None:
                audio_chunks.append(audio_chunk)

        if not audio_chunks:
            raise RuntimeError(f"Kokoro produced no audio for voice={voice}")

        full_audio = np.concatenate(audio_chunks)
        sf.write(str(output_path), full_audio, 24000)
        return output_path

    async def generate_audio(
        self,
        script: str,
        part_number: int,
        output_filename: Optional[str] = None
    ) -> GeneratedAudio:
        """
        Generate audio for a script.

        Args:
            script: The voiceover text to convert
            part_number: Video part number
            output_filename: Custom filename (optional)

        Returns:
            GeneratedAudio object with path and duration
        """
        if output_filename is None:
            output_filename = f"part_{part_number}_voiceover.wav"

        output_path = self.output_dir / output_filename

        logger.info(f"Generating Kokoro audio for part {part_number} (voice={self.voice})...")

        # Run synthesis in a thread so we don't block the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._generate_wav,
            script,
            self.voice,
            output_path,
        )

        # Get audio duration
        duration_ms = self._get_audio_duration(output_path)

        # Estimate word boundaries (Kokoro doesn't produce native boundaries)
        words = self._estimate_word_boundaries(script, duration_ms)

        # Save word boundaries to JSON for caching
        words_path = output_path.with_name(f"part_{part_number}_words.json")
        try:
            with open(words_path, "w", encoding="utf-8") as f:
                json.dump(words, f, indent=2)
            logger.info(f"Saved {len(words)} estimated word boundaries to {words_path}")
        except Exception as write_err:
            logger.warning(f"Could not save word boundaries: {write_err}")

        logger.info(f"Generated audio for part {part_number}: {output_path} ({duration_ms}ms)")

        return GeneratedAudio(
            part_number=part_number,
            audio_path=output_path,
            duration_ms=duration_ms,
            script=script,
            word_boundaries=words,
        )

    def generate_audio_sync(
        self,
        script: str,
        part_number: int,
        output_filename: Optional[str] = None
    ) -> GeneratedAudio:
        """Synchronous wrapper for audio generation."""
        return asyncio.run(self.generate_audio(script, part_number, output_filename))

    async def generate_all_audio(
        self,
        scripts: List[Tuple[int, str]]
    ) -> List[GeneratedAudio]:
        """
        Generate audio for multiple scripts sequentially.

        Args:
            scripts: List of (part_number, script) tuples

        Returns:
            List of GeneratedAudio objects
        """
        results = []
        for part_num, script in scripts:
            result = await self.generate_audio(script, part_num)
            results.append(result)

        results.sort(key=lambda x: x.part_number)
        return results

    def _get_audio_duration(self, audio_path: Path) -> int:
        """
        Get audio duration in milliseconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in milliseconds
        """
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            return int(info.duration * 1000)
        except Exception as e:
            logger.warning(f"Failed to get duration with soundfile: {e}")
            try:
                from mutagen.mp3 import MP3
                audio = MP3(str(audio_path))
                return int(audio.info.length * 1000)
            except Exception:
                pass
            # Last-resort fallback: WAV at 24kHz, 16-bit mono
            file_size = audio_path.stat().st_size
            estimated_duration = (file_size - 44) / (24000 * 2)
            return max(int(estimated_duration * 1000), 0)

    @staticmethod
    def _estimate_word_boundaries(script: str, duration_ms: int) -> List[dict]:
        """Estimate word boundaries by distributing evenly across duration."""
        words = script.split()
        if not words:
            return []
        total_sec = duration_ms / 1000.0
        word_dur = total_sec / len(words)
        boundaries = []
        current_time = 0.0
        for w in words:
            clean_word = w.strip(".,!?;:()\"'-")
            boundaries.append({
                "word": clean_word,
                "start": current_time,
                "duration": word_dur,
            })
            current_time += word_dur
        return boundaries

    @staticmethod
    def generate_sample(
        voice_id: str,
        output_path: Path,
        text: Optional[str] = None,
        speed: float = 1.0,
    ) -> Path:
        """
        Generate a short audio sample for a voice.

        Args:
            voice_id: Kokoro voice ID
            output_path: Where to save the sample WAV
            text: Sample text (uses default if None)
            speed: Speed multiplier

        Returns:
            Path to the generated sample file
        """
        import soundfile as sf

        sample_text = text or SAMPLE_TEXT
        lang_code = AudioGenerator._voice_to_lang(voice_id)
        pipeline = _get_kokoro_pipeline(lang_code)

        audio_chunks = []
        for _gs, _ps, audio_chunk in pipeline(sample_text, voice=voice_id, speed=speed):
            if audio_chunk is not None:
                audio_chunks.append(audio_chunk)

        if not audio_chunks:
            raise RuntimeError(f"Kokoro produced no audio for voice={voice_id}")

        full_audio = np.concatenate(audio_chunks)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), full_audio, 24000)

        logger.info(f"Generated sample for voice {voice_id}: {output_path}")
        return output_path

    def list_available_voices(self) -> List[dict]:
        """List available Kokoro voices with metadata."""
        result = []
        for voice_id, meta in KOKORO_VOICES.items():
            result.append({
                "id": voice_id,
                "name": meta["name"],
                "gender": meta["gender"],
                "accent": meta["accent"],
            })
        return result

    @staticmethod
    def calculate_image_timing(
        audio_duration_ms: int,
        num_panels: int,
        min_duration_ms: int = 3000,
        max_duration_ms: int = 10000
    ) -> List[int]:
        """
        Calculate how long each panel should display based on audio duration.

        Args:
            audio_duration_ms: Total audio duration in milliseconds
            num_panels: Number of panels to display
            min_duration_ms: Minimum display time per panel
            max_duration_ms: Maximum display time per panel

        Returns:
            List of durations in milliseconds for each panel
        """
        # Calculate base duration per panel
        base_duration = audio_duration_ms // num_panels

        # Clamp to min/max
        if base_duration < min_duration_ms:
            base_duration = min_duration_ms
        elif base_duration > max_duration_ms:
            base_duration = max_duration_ms

        # Distribute duration across panels
        total_assigned = base_duration * num_panels
        remaining = audio_duration_ms - total_assigned

        durations = [base_duration] * num_panels

        # Distribute remaining milliseconds
        for i in range(abs(remaining) // 100):
            if remaining > 0:
                durations[i % num_panels] += 100
            else:
                durations[i % num_panels] -= 100

        return durations
