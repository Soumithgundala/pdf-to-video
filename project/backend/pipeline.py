"""
Manga Recap Video Pipeline Orchestrator
Coordinates all phases from PDF to final videos.
"""
import shutil
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import json

import config
from modules import (
    PDFProcessor,
    PanelExtractor,
    ContactSheetGenerator,
    LLMStoryDirector,
    AudioGenerator,
    VideoAssembler,
    StoryAnalysis,
    VideoPartConfig,
)

logger = logging.getLogger(__name__)


class MangaPipeline:
    """Main pipeline orchestrator for manga video generation."""

    def __init__(
        self,
        workspace_base: Optional[Path] = None,
        llm_provider: str = "openai"
    ):
        """
        Initialize the pipeline.

        Args:
            workspace_base: Base directory for all workspaces
            llm_provider: "openai" or "google" for LLM
        """
        self.workspace_base = workspace_base or Path(config.WORKSPACE_DIR)
        self.llm_provider = llm_provider

        # Will be set when processing starts
        self.job_id: Optional[str] = None
        self.job_workspace: Optional[Path] = None

    def process(
        self,
        pdf_path: Path,
        job_id: str,
        background_music_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Process a manga PDF through all pipeline phases.

        Args:
            pdf_path: Path to the PDF file
            job_id: Unique job identifier
            background_music_path: Optional background music file

        Returns:
            Dictionary with results and paths
        """
        self.job_id = job_id
        self.job_workspace = self.workspace_base / job_id

        # Clean existing workspace
        if self.job_workspace.exists():
            shutil.rmtree(self.job_workspace)

        # Create workspace directories
        (self.job_workspace / "pages").mkdir(parents=True)
        (self.job_workspace / "panels").mkdir(parents=True)
        (self.job_workspace / "contact_sheets").mkdir(parents=True)
        (self.job_workspace / "audio").mkdir(parents=True)
        (self.job_workspace / "videos").mkdir(parents=True)

        results = {
            "job_id": job_id,
            "workspace": str(self.job_workspace),
            "status": "processing",
            "phases": {}
        }

        try:
            self._write_status("processing", 0.05, "Preparing workspace")

            # Phase 1: PDF Processing & Panel Extraction
            logger.info("=" * 50)
            logger.info("PHASE 1: PDF Processing & Panel Extraction")
            logger.info("=" * 50)

            page_paths, panel_paths, contact_sheet_paths = self._run_phase_1(pdf_path)

            results["phases"]["phase_1"] = {
                "status": "completed",
                "pages": len(page_paths),
                "panels": len(panel_paths),
                "contact_sheets": [str(p) for p in contact_sheet_paths]
            }
            self._write_status("processing", 0.25, "PDF converted and panels extracted")

            # Phase 2: LLM Story Director
            logger.info("=" * 50)
            logger.info("PHASE 2: LLM Story Director")
            logger.info("=" * 50)

            story_analysis = self._run_phase_2(page_paths, contact_sheet_paths, len(panel_paths))

            results["phases"]["phase_2"] = {
                "status": "completed",
                "story_parts": len(story_analysis.parts),
                "total_panels_selected": story_analysis.total_panels_selected
            }
            self._write_status("processing", 0.5, "Story analysis complete")

            # Phase 3: Audio Generation
            logger.info("=" * 50)
            logger.info("PHASE 3: Audio Generation")
            logger.info("=" * 50)

            audio_results = self._run_phase_3(story_analysis)

            results["phases"]["phase_3"] = {
                "status": "completed",
                "audio_files": [
                    {
                        "part": a.part_number,
                        "path": str(a.audio_path),
                        "duration_ms": a.duration_ms
                    }
                    for a in audio_results
                ]
            }
            self._write_status("processing", 0.75, "Audio generation complete")

            # Phase 4: Video Assembly
            logger.info("=" * 50)
            logger.info("PHASE 4: Video Assembly")
            logger.info("=" * 50)

            video_results = self._run_phase_4(
                story_analysis,
                audio_results,
                panel_paths,
                background_music_path
            )

            results["phases"]["phase_4"] = {
                "status": "completed",
                "videos": [
                    {
                        "part": v.part_number,
                        "path": str(v.video_path),
                        "duration_seconds": v.duration_seconds
                    }
                    for v in video_results
                ]
            }

            results["status"] = "completed"
            results["video_paths"] = [str(v.video_path) for v in video_results]
            self._write_status("completed", 1.0, "Pipeline complete")

            logger.info("=" * 50)
            logger.info("PIPELINE COMPLETE!")
            logger.info(f"Generated {len(video_results)} videos in {self.job_workspace}")
            logger.info("=" * 50)

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)

            self._write_status("failed", 0.0, str(e), error=str(e))

            raise

    def _run_phase_1(
        self,
        pdf_path: Path
    ) -> tuple[List[Path], List[Path], List[Path]]:
        """Execute Phase 1: PDF processing and panel extraction."""

        # PDF to Pages
        pdf_processor = PDFProcessor(
            output_dir=self.job_workspace / "pages",
            dpi=config.PDF_DPI
        )
        page_paths = pdf_processor.convert_pdf_to_pages(pdf_path)

        # Panel Extraction
        panel_extractor = PanelExtractor(
            output_dir=self.job_workspace / "panels"
        )
        panel_paths = []
        for i, page_path in enumerate(page_paths, start=1):
            self._write_status(
                "processing",
                0.05 + 0.15 * (i / len(page_paths)),
                f"Extracting panels from page {i}/{len(page_paths)}"
            )
            panels = panel_extractor.extract_panels_from_page(
                page_path, i, self.job_workspace / "panels"
            )
            panel_paths.extend([p.path for p in panels])

        # Renumber panels globally
        for i, panel_path in enumerate(panel_paths, start=1):
            new_name = f"panel_P{i}.png"
            new_path = panel_path.parent / new_name
            panel_path.rename(new_path)

        # Reload updated paths
        panel_paths = sorted(
            (self.job_workspace / "panels").glob("panel_P*.png"),
            key=lambda p: int(p.stem.split('P')[1])
        )

        # Contact Sheet Generation
        self._write_status(
            "processing",
            0.20,
            "Generating panel contact sheets"
        )
        contact_gen = ContactSheetGenerator()
        contact_sheet_paths = contact_gen.generate_contact_sheet(
            panel_paths,
            self.job_workspace / "contact_sheets" / "contact_sheet"
        )

        return page_paths, panel_paths, contact_sheet_paths

    def _run_phase_2(
        self,
        page_paths: List[Path],
        contact_sheet_paths: List[Path],
        total_panels: int
    ) -> StoryAnalysis:
        """Execute Phase 2: LLM story analysis."""

        try:
            llm_director = LLMStoryDirector(provider=self.llm_provider)
            story_analysis = llm_director.analyze_manga(
                page_paths,
                contact_sheet_paths[0],
                total_panels
            )
        except Exception as e:
            logger.warning(f"LLM story analysis or initialization failed: {e}")
            logger.warning("Falling back to mock story analysis")
            story_analysis = self._mock_story_analysis(total_panels)

        # Save analysis to JSON
        analysis_path = self.job_workspace / "story_analysis.json"
        try:
            with open(analysis_path, 'w') as f:
                json.dump(story_analysis.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save story_analysis.json: {e}")

        return story_analysis

    def _run_phase_3(
        self,
        story_analysis: StoryAnalysis
    ) -> list:
        """Execute Phase 3: Audio generation."""

        audio_gen = AudioGenerator(
            output_dir=self.job_workspace / "audio"
        )

        scripts = [
            (part.part_number, part.script)
            for part in story_analysis.parts
        ]

        audio_results = asyncio.run(audio_gen.generate_all_audio(scripts))

        return audio_results

    def _run_phase_4(
        self,
        story_analysis: StoryAnalysis,
        audio_results: list,
        panel_paths: List[Path],
        background_music_path: Optional[Path]
    ) -> list:
        """Execute Phase 4: Video assembly."""

        video_assembler = VideoAssembler(
            output_dir=self.job_workspace / "videos"
        )

        # Build panel lookup
        panel_lookup = {f"P{i+1}": p for i, p in enumerate(panel_paths)}

        # Build video configs
        video_configs = []

        for part, audio in zip(story_analysis.parts, audio_results):
            # Get panel paths for this part
            part_panel_paths = [
                panel_lookup[pid]
                for pid in part.selected_panels
                if pid in panel_lookup
            ]

            video_config = VideoPartConfig(
                part_number=part.part_number,
                voiceover_path=audio.audio_path,
                audio_duration_ms=audio.duration_ms,
                panels=part_panel_paths,
                background_music_path=background_music_path,
                output_path=self.job_workspace / "videos" / f"part_{part.part_number}.mp4"
            )
            video_configs.append(video_config)

        video_results = []
        for i, config in enumerate(video_configs, start=1):
            self._write_status(
                "processing",
                0.75 + 0.20 * ((i - 1) / len(video_configs)),
                f"Assembling video part {i}/{len(video_configs)}"
            )
            result = video_assembler.assemble_video(config)
            video_results.append(result)

        return video_results

    def _mock_story_analysis(self, total_panels: int) -> StoryAnalysis:
        """Generate mock story analysis when LLM is not available."""
        from modules import VideoScript

        panels_per_part = total_panels // 4
        remainder = total_panels % 4

        parts = []
        current_panel = 1

        for i in range(4):
            num_panels = panels_per_part + (1 if i < remainder else 0)
            num_panels = min(num_panels, 7)
            num_panels = max(num_panels, 5)

            end_panel = min(current_panel + num_panels - 1, total_panels)
            selected = [f"P{j}" for j in range(current_panel, end_panel + 1)]

            if len(selected) < 5:
                # Pad with nearby panels
                needed = 5 - len(selected)
                for j in range(1, needed + 1):
                    if end_panel + j <= total_panels:
                        selected.append(f"P{end_panel + j}")

            selected = selected[:7]  # Max 7 panels

            script = f"This is part {i+1} of the manga recap. The story continues with exciting developments as our protagonist faces new challenges. Watch as the narrative unfolds across these carefully selected panels that capture the essence of this chapter."

            parts.append(VideoScript(
                part_number=i + 1,
                script=script,
                selected_panels=selected
            ))

            current_panel = end_panel + 1

        return StoryAnalysis(parts=parts, total_panels_selected=sum(len(p.selected_panels) for p in parts))

    def cleanup(self):
        """Clean up workspace."""
        if self.job_workspace and self.job_workspace.exists():
            logger.info(f"Cleaning up workspace: {self.job_workspace}")
            shutil.rmtree(self.job_workspace)

    def _write_status(
        self,
        status: str,
        progress: float,
        message: str,
        *,
        error: Optional[str] = None
    ) -> None:
        """Persist a lightweight status file for filesystem-based polling."""
        if not self.job_workspace or not self.job_id:
            return

        payload = {
            "job_id": self.job_id,
            "status": status,
            "progress": progress,
            "message": message,
        }
        if error:
            payload["error_message"] = error

        try:
            status_file = self.job_workspace / "status.json"
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as write_err:
            logger.warning(f"Could not write status.json: {write_err}")
