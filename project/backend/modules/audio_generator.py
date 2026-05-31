"""
Phase 3: Audio Generation
Uses edge-tts for high-quality, free text-to-speech voiceovers.
"""
import asyncio
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import mutagen
from mutagen.mp3 import MP3
import logging

logger = logging.getLogger(__name__)

try:
    import edge_tts
except ImportError:
    edge_tts = None

import config


@dataclass
class GeneratedAudio:
    """Represents generated audio with metadata."""
    part_number: int
    audio_path: Path
    duration_ms: int
    script: str

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000


class AudioGenerator:
    """Generates voiceover audio using edge-tts."""

    def __init__(
        self,
        voice: str = "en-US-ChristopherNeural",
        output_dir: Optional[Path] = None,
        rate: str = "+0%",
        volume: str = "+0%"
    ):
        """
        Initialize Audio Generator.

        Args:
            voice: Edge-tts voice name (default: Christopher - natural male voice)
            output_dir: Directory to save audio files
            rate: Speech rate adjustment (e.g., "+20%", "-10%")
            volume: Volume adjustment
        """
        if edge_tts is None:
            raise ImportError("edge-tts package not installed")

        self.voice = voice
        self.output_dir = output_dir or Path(config.WORKSPACE_DIR) / "audio"
        self.rate = rate
        self.volume = volume

        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            output_filename = f"part_{part_number}_voiceover.mp3"

        output_path = self.output_dir / output_filename

        logger.info(f"Generating audio for part {part_number}...")

        # Generate audio with retries for transient network/DNS failures.
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                communicate = edge_tts.Communicate(
                    text=script,
                    voice=self.voice,
                    rate=self.rate,
                    volume=self.volume
                )
                await communicate.save(str(output_path))
                break
            except Exception as e:
                if attempt == max_attempts:
                    logger.error(
                        "Audio generation failed for part %s after %s attempts: %s",
                        part_number,
                        max_attempts,
                        e
                    )
                    raise

                delay_seconds = 2 ** (attempt - 1)
                logger.warning(
                    "Audio generation failed for part %s on attempt %s/%s: %s. Retrying in %ss...",
                    part_number,
                    attempt,
                    max_attempts,
                    e,
                    delay_seconds
                )
                await asyncio.sleep(delay_seconds)

        # Get audio duration
        duration_ms = self._get_audio_duration(output_path)

        logger.info(f"Generated audio for part {part_number}: {output_path} ({duration_ms}ms)")

        return GeneratedAudio(
            part_number=part_number,
            audio_path=output_path,
            duration_ms=duration_ms,
            script=script
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
        Generate audio for multiple scripts in parallel.

        Args:
            scripts: List of (part_number, script) tuples

        Returns:
            List of GeneratedAudio objects
        """
        tasks = [
            self.generate_audio(script, part_num)
            for part_num, script in scripts
        ]

        results = await asyncio.gather(*tasks)

        # Sort by part number
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
            audio = MP3(str(audio_path))
            duration_seconds = audio.info.length
            return int(duration_seconds * 1000)
        except Exception as e:
            logger.warning(f"Failed to get duration with mutagen: {e}")
            # Fallback: estimate from file size
            # Average bit rate for edge-tts is ~48kbps
            file_size = audio_path.stat().st_size
            estimated_duration = (file_size * 8) / 48000  # 48kbps
            return int(estimated_duration * 1000)

    def list_available_voices(self) -> List[str]:
        """List available edge-tts voices."""
        result = subprocess.run(
            ["edge-tts", "--list-voices"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Failed to list voices: {result.stderr}")
            return []

        # Parse voice names from output
        voices = []
        for line in result.stdout.split('\n'):
            if 'Name:' in line:
                name = line.split('Name:')[1].strip()
                voices.append(name)

        return voices

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


# Available high-quality English voices
RECOMMENDED_VOICES = {
    "male": [
        "en-US-ChristopherNeural",
        "en-US-EricNeural",
        "en-US-GuyNeural",
        "en-GB-RyanNeural",
        "en-GB-ThomasNeural",
    ],
    "female": [
        "en-US-AriaNeural",
        "en-US-JennyNeural",
        "en-GB-SoniaNeural",
        "en-GB-MiaNeural",
    ],
    "narrator": [
        "en-US-ChristopherNeural",
        "en-GB-RyanNeural",
    ]
}
