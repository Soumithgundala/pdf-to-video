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
    panel_focus_areas: Dict[str, List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'parts': [p.to_dict() for p in self.parts],
            'total_panels_selected': self.total_panels_selected,
            'panel_focus_areas': self.panel_focus_areas or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoryAnalysis':
        parts = [VideoScript.from_dict(p) for p in data.get('parts', [])]
        return cls(
            parts=parts,
            total_panels_selected=data.get('total_panels_selected', 0),
            panel_focus_areas=data.get('panel_focus_areas', {})
        )


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
        total_panels: int,
        panels_pdf_path: Optional[Path] = None
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
                    result = self._analyze_with_gemini(page_images, contact_sheet, total_panels, panels_pdf_path)

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
        total_panels: int,
        panels_pdf_path: Optional[Path] = None
    ) -> StoryAnalysis:
        """Analyze with Google Gemini using PDF upload and contact sheet fallback."""
        if not self.client:
            raise ValueError("Google API key not configured")

        contents = []
        uploaded_file = None

        if panels_pdf_path and panels_pdf_path.exists():
            try:
                logger.info(f"Uploading panels PDF to Gemini: {panels_pdf_path}")
                uploaded_file = self.client.files.upload(file=panels_pdf_path)
                contents.append(uploaded_file)
                logger.info(f"Successfully uploaded panels PDF. File URI: {uploaded_file.uri}")
            except Exception as upload_err:
                logger.warning(f"Failed to upload panels PDF to Gemini: {upload_err}. Falling back to contact sheet image.")
        
        if not contents:
            from PIL import Image
            try:
                img = Image.open(contact_sheet)
                contents.append(img)
            except Exception as e:
                logger.error(f"Failed to open contact sheet: {e}")
                raise ValueError(f"Failed to load contact sheet image: {e}")

        # Build the prompt
        prompt = f"""{self._get_system_prompt()}

{self._get_user_prompt_text(total_panels)}

The manga has {total_panels} panels. Refer to the uploaded document (where each page represents a panel labeled sequentially P1, P2, P3... up to P{total_panels}) or contact sheet.
Generate ONLY valid JSON output matching the required schema."""

        contents.append(prompt)

        config_params = {}
        if types is not None:
            config_params["config"] = types.GenerateContentConfig(
                response_mime_type="application/json"
            )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                **config_params
            )
            response_text = response.text
        finally:
            if uploaded_file:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                    logger.info("Deleted temporary panels PDF from Gemini storage.")
                except Exception as delete_err:
                    logger.warning(f"Could not delete uploaded file {uploaded_file.name}: {delete_err}")

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
        focus_areas = data.get('panel_focus_areas', {})

        return StoryAnalysis(
            parts=parts,
            total_panels_selected=total_selected,
            panel_focus_areas=focus_areas
        )

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
            if word_count < 90 or word_count > 160:
                logger.warning(f"Part {part.part_number} script has {word_count} words (target: 110-140)")

        # Validate focus areas
        if result.panel_focus_areas:
            for pid, box in list(result.panel_focus_areas.items()):
                if not isinstance(box, list) or len(box) != 4:
                    logger.warning(f"Invalid focus box format for {pid}: {box}. Resetting to default.")
                    result.panel_focus_areas[pid] = [0, 0, 1000, 1000]
                else:
                    try:
                        result.panel_focus_areas[pid] = [int(v) for v in box]
                    except ValueError:
                        logger.warning(f"Non-integer value in focus box for {pid}: {box}")
                        result.panel_focus_areas[pid] = [0, 0, 1000, 1000]

    def _get_system_prompt(self) -> str:
        """Get system prompt for the LLM."""
        return """You are a master manga recap narrator and expert story analyst specializing in One Piece. Your task is to write highly dramatic, cinematic, and deeply engaging recap scripts tailored for passionate, dedicated manga readers.

You must:
1. Read and understand the manga narrative from the provided pages.
2. Divide the story into exactly 4 sequential parts.
3. Write a dramatic, lore-rich voiceover script for each part (110-140 words).
4. Select 5-7 panels from the manga pages that best illustrate each script.
5. Identify the primary visual focus area (character's face, action scene, main subject) for EACH panel.

CRITICAL CONTENT & TONE RULES:
- Target a smart, dedicated audience. Assume they know all the lore, terms (Haki, Devil Fruits, Will of D, etc.), and characters.
- Explain the story deeply: describe the specific actions, character emotions, dialogue impact, and combat details in the panels.
- Tone should be high-energy, exciting, and full of suspense, like an epic YouTube manga recap.
- ABSOLUTELY DO NOT mention meta phrases like "part 1", "part 2", "in this part", "this is part...", "this video", "first", "next", or make any reference to the division of the video/chapters. Each script must read like a seamless, continuous, deep-dive narrative, flowing naturally into the next part as if it were one single video.
- Avoid generic summaries; explain the actual events, character dialogue, and action sequences in the panels.

Output a valid JSON object with this exact structure:
{
  "panel_focus_areas": {
    "P1": [ymin, xmin, ymax, xmax],
    "P2": [ymin, xmin, ymax, xmax],
    ...
  },
  "parts": [
    {
      "part_number": 1,
      "script": "The deep-dive voiceover text for this part...",
      "selected_panels": ["P1", "P2", "P3", "P4", "P5"]
    },
    ...
  ]
}

CRITICAL FORMAT RULES:
- panel_focus_areas: For each panel P1, P2, ... up to P{total_panels}, detect the primary visual content/character/subject that must be visible in the video. Output the normalized bounding box [ymin, xmin, ymax, xmax] from 0 to 1000 (0 is top/left, 1000 is bottom/right).
- Each part must have exactly 5-7 panels selected in reading order.
- Panel IDs must match the sequential panel IDs (format: P1, P2, P3, etc.).
- Each script should be 110-140 words for natural pacing (~60-70 seconds spoken).
- DO NOT select the same panel multiple times across parts.
- Each part must cover a distinct segment of the story in sequence."""

    def _get_user_prompt_text(self, total_panels: int) -> str:
        """Get user prompt text."""
        return f"""Analyze this manga chapter and create a 4-part video recap script.

The document contains {total_panels} sequential panels with IDs P1 through P{total_panels}.

For each of the 4 parts:
1. Write a deep-dive, dramatic voiceover script (110-140 words) explaining the details of the actions, character expressions, dialogue, and narrative.
2. Select 5-7 panels that best illustrate the script.
3. For all {total_panels} panels, specify their primary focus area coordinates [ymin, xmin, ymax, xmax] in the panel_focus_areas dictionary.
4. Ensure the parts form a seamless, continuous story flow. Never mention "part 1", "part 2", or any part numbers.
5. Keep the tone engaging and targeted at seasoned One Piece manga readers.

Output valid JSON only."""
