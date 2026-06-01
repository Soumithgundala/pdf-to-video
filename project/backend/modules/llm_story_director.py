"""
Phase 2: LLM Story Director
Uses Vision-Language Model to analyze manga and create video scripts.
"""
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import logging
import re

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

import config


@dataclass
class VideoScript:
    """Structured output for a single video part."""
    part_number: int
    script: str
    selected_panels: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoScript':
        return cls(
            part_number=data['part_number'],
            script=data['script'],
            selected_panels=data['selected_panels']
        )


@dataclass
class StoryAnalysis:
    """Complete story analysis for all 4 parts."""
    parts: List[VideoScript]
    total_panels_selected: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'parts': [p.to_dict() for p in self.parts],
            'total_panels_selected': self.total_panels_selected
        }


class LLMStoryDirector:
    """Analyzes manga chapter and creates video scripts using VLM."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        max_retries: int = 3
    ):
        """
        Initialize LLM Story Director.

        Args:
            provider: "openai" for GPT-4o or "google" for Gemini 1.5 Pro
            model: Specific model to use (optional, uses default if not provided)
            max_retries: Maximum retry attempts for JSON parsing failures
        """
        self.provider = provider
        self.max_retries = max_retries

        if provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package not installed")
            self.client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
            self.model = model or "gpt-4o"
        elif provider == "google":
            if genai is None:
                raise ImportError("google-genai package not installed")
            self.client = genai.Client(api_key=config.GOOGLE_API_KEY) if config.GOOGLE_API_KEY else None
            self.model = model or "gemini-2.5-flash"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def analyze_manga(
        self,
        page_images: List[Path],
        contact_sheet: Path,
        total_panels: int
    ) -> StoryAnalysis:
        """
        Analyze manga chapter and generate scripts for 4 video parts.

        Args:
            page_images: List of paths to page images
            contact_sheet: Path to contact sheet image
            total_panels: Total number of extracted panels

        Returns:
            StoryAnalysis object with scripts for 4 parts

        Raises:
            ValueError: If analysis fails after all retries
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting story analysis (attempt {attempt + 1}/{self.max_retries})")

                if self.provider == "openai":
                    result = self._analyze_with_openai(page_images, contact_sheet, total_panels)
                else:
                    result = self._analyze_with_gemini(page_images, contact_sheet, total_panels)

                # Verify minimum panels per part
                self._validate_result(result, total_panels)

                logger.info(f"Successfully analyzed manga into {len(result.parts)} video parts")
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to parse LLM response after {self.max_retries} attempts")

            except Exception as e:
                logger.error(f"Analysis failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise

        raise ValueError("Story analysis failed")

    def _encode_image(self, image_path: Path) -> str:
        """Encode an image to base64 string."""
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _analyze_with_openai(
        self,
        page_images: List[Path],
        contact_sheet: Path,
        total_panels: int
    ) -> StoryAnalysis:
        """Analyze with OpenAI GPT-4o Vision."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        # Encode contact sheet
        base64_image = self._encode_image(contact_sheet)

        # Build messages for Chat Completion API with Vision
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{self._get_user_prompt_text(total_panels)}\n\nHere is the contact sheet containing all extracted panels labeled P1, P2, etc."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        if not response_text:
            raise ValueError("Empty response received from OpenAI API")

        return self._parse_response(response_text)

    def _analyze_with_gemini(
        self,
        page_images: List[Path],
        contact_sheet: Path,
        total_panels: int
    ) -> StoryAnalysis:
        """Analyze with Google Gemini 1.5 Flash."""
        if not self.client:
            raise ValueError("Google API key not configured")

        from PIL import Image
        try:
            img = Image.open(contact_sheet)
        except Exception as e:
            logger.error(f"Failed to open contact sheet: {e}")
            raise ValueError(f"Failed to load contact sheet image: {e}")

        # Build the prompt
        prompt = f"""{self._get_system_prompt()}

{self._get_user_prompt_text(total_panels)}

The manga has {total_panels} panels. Refer to the attached contact sheet showing all extracted panels labeled P1, P2, etc.
Generate ONLY valid JSON output matching the required schema."""

        # Use the new google-genai SDK
        # We pass the text prompt and the PIL Image object in contents list
        # And we set response_mime_type="application/json" to ensure valid JSON output
        config_params = {}
        if types is not None:
            config_params["config"] = types.GenerateContentConfig(
                response_mime_type="application/json"
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, img],
            **config_params
        )

        response_text = response.text
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> StoryAnalysis:
        """Parse LLM JSON response into StoryAnalysis."""
        # Extract JSON if wrapped in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)

        data = json.loads(response_text)

        parts = []
        for part_data in data.get('parts', []):
            script = VideoScript(
                part_number=part_data['part_number'],
                script=part_data['script'],
                selected_panels=part_data['selected_panels']
            )
            parts.append(script)

        total_selected = sum(len(p.selected_panels) for p in parts)

        return StoryAnalysis(parts=parts, total_panels_selected=total_selected)

    def _validate_result(self, result: StoryAnalysis, total_panels: int) -> None:
        """Validate that result meets requirements."""
        if len(result.parts) != 4:
            raise ValueError(f"Expected 4 video parts, got {len(result.parts)}")

        for part in result.parts:
            num_panels = len(part.selected_panels)
            if num_panels < 5:
                raise ValueError(f"Part {part.part_number} has only {num_panels} panels, minimum is 5")
            if num_panels > 7:
                raise ValueError(f"Part {part.part_number} has {num_panels} panels, maximum is 7")

            # Validate panel IDs exist
            for panel_id in part.selected_panels:
                if not panel_id.startswith('P'):
                    raise ValueError(f"Invalid panel ID format: {panel_id}")
                try:
                    panel_num = int(panel_id[1:])
                    if panel_num < 1 or panel_num > total_panels:
                        raise ValueError(f"Panel ID {panel_id} out of range (1-{total_panels})")
                except ValueError:
                    raise ValueError(f"Invalid panel ID: {panel_id}")

            # Validate script word count
            word_count = len(part.script.split())
            if word_count < 50 or word_count > 90:
                logger.warning(f"Part {part.part_number} script has {word_count} words (target: 65-75)")

    def _get_system_prompt(self) -> str:
        """Get system prompt for the LLM."""
        return """You are a professional manga story analyst and video script writer. Your task is to analyze manga chapters and create engaging video recap scripts.

You must:
1. Read and understand the manga narrative from the provided page images
2. Divide the story into exactly 4 parts
3. Write engaging voiceover scripts for each part (65-75 words, ~35-40 seconds spoken)
4. Select 5-7 panels from the contact sheet that best illustrate each script

Output a valid JSON object with this exact structure:
{
  "parts": [
    {
      "part_number": 1,
      "script": "The voiceover text for this part...",
      "selected_panels": ["P1", "P2", "P3", "P4", "P5"]
    },
    ...
  ]
}

CRITICAL RULES:
- Each part must have exactly 5-7 panels selected
- Panel IDs must match IDs shown on the contact sheet (format: P1, P2, P3, etc.)
- Each script should be 65-75 words for natural pacing
- Scripts should tell a cohesive narrative when combined
- Select panels that visually represent the script content
- DO NOT select the same panel multiple times across parts
- Each part should cover a distinct segment of the story"""

    def _get_user_prompt_text(self, total_panels: int) -> str:
        """Get user prompt text."""
        return f"""Analyze this manga chapter and create a 4-part video recap script.

The contact sheet shows {total_panels} extracted panels with IDs P1 through P{total_panels}.

For each of the 4 parts:
1. Write an engaging voiceover script (65-75 words)
2. Select 5-7 panels that best illustrate the script
3. Ensure the parts form a complete narrative of the chapter

Output valid JSON only."""
