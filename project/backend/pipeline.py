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

        # Phase tracking states
        self.phase_1_status = "pending"
        self.phase_1_progress = 0.0
        self.phase_1_message = "Pending"
        self.phase_2_status = "pending"
        self.phase_2_progress = 0.0
        self.phase_2_message = "Pending"
        self.phase_3_status = "pending"
        self.phase_3_progress = 0.0
        self.phase_3_message = "Pending"
        self.phase_4_status = "pending"
        self.phase_4_progress = 0.0
        self.phase_4_message = "Pending"

    def process(
        self,
        pdf_path: Path,
        job_id: str,
        background_music_path: Optional[Path] = None,
        cached_job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a manga PDF through all pipeline phases.

        Args:
            pdf_path: Path to the PDF file
            job_id: Unique job identifier
            background_music_path: Optional background music file
            cached_job_id: Optional cached job ID to reuse assets from

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
            if cached_job_id:
                logger.info(f"Fast-tracking job {job_id} by copying cached assets from job {cached_job_id}...")
                self._write_status(
                    "processing",
                    0.70,
                    "Loading cached pages, panels, and voiceover audio",
                    active_phase="phase_4",
                    phase_status="processing",
                    phase_progress=0.0,
                    phase_message="Loading cached assets"
                )
                
                # Mark previous phases as completed
                self.phase_1_status = "completed"
                self.phase_1_progress = 1.0
                self.phase_1_message = "PDF converted and panels extracted (cached)"
                self.phase_2_status = "completed"
                self.phase_2_progress = 1.0
                self.phase_2_message = "Story analysis complete (cached)"
                self.phase_3_status = "completed"
                self.phase_3_progress = 1.0
                self.phase_3_message = "Audio generation complete (cached)"
                
                panel_paths, story_analysis, audio_results = self.load_cached_assets(cached_job_id)
                
                results["phases"]["phase_1"] = {
                    "status": "completed",
                    "pages": 0,
                    "panels": len(panel_paths),
                    "contact_sheets": []
                }
                results["phases"]["phase_2"] = {
                    "status": "completed",
                    "story_parts": len(story_analysis.parts),
                    "total_panels_selected": story_analysis.total_panels_selected
                }
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
            else:
                self._write_status(
                    "processing",
                    0.05,
                    "Preparing workspace",
                    active_phase="phase_1",
                    phase_status="processing",
                    phase_progress=0.0,
                    phase_message="Preparing workspace"
                )

                # Phase 1: PDF Processing & Panel Extraction
                logger.info("=" * 50)
                logger.info("PHASE 1: PDF Processing & Panel Extraction")
                logger.info("=" * 50)

                page_paths, panel_paths, contact_sheet_paths, panels_pdf_path = self._run_phase_1(pdf_path)

                results["phases"]["phase_1"] = {
                    "status": "completed",
                    "pages": len(page_paths),
                    "panels": len(panel_paths),
                    "contact_sheets": [str(p) for p in contact_sheet_paths]
                }
                self._write_status(
                    "processing",
                    0.25,
                    "PDF converted and panels extracted",
                    active_phase="phase_1",
                    phase_status="completed",
                    phase_progress=1.0,
                    phase_message="PDF converted and panels extracted"
                )

                # Phase 2: LLM Story Director
                logger.info("=" * 50)
                logger.info("PHASE 2: LLM Story Director")
                logger.info("=" * 50)

                self._write_status(
                    "processing",
                    0.25,
                    "Starting story analysis",
                    active_phase="phase_2",
                    phase_status="processing",
                    phase_progress=0.0,
                    phase_message="Starting story analysis"
                )

                story_analysis = self._run_phase_2(page_paths, contact_sheet_paths, len(panel_paths), panels_pdf_path)

                results["phases"]["phase_2"] = {
                    "status": "completed",
                    "story_parts": len(story_analysis.parts),
                    "total_panels_selected": story_analysis.total_panels_selected
                }
                self._write_status(
                    "processing",
                    0.5,
                    "Story analysis complete",
                    active_phase="phase_2",
                    phase_status="completed",
                    phase_progress=1.0,
                    phase_message="Story analysis complete"
                )

                # Phase 3: Audio Generation
                logger.info("=" * 50)
                logger.info("PHASE 3: Audio Generation")
                logger.info("=" * 50)

                self._write_status(
                    "processing",
                    0.5,
                    "Starting audio generation",
                    active_phase="phase_3",
                    phase_status="processing",
                    phase_progress=0.0,
                    phase_message="Starting audio generation"
                )

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
                self._write_status(
                    "processing",
                    0.75,
                    "Audio generation complete",
                    active_phase="phase_3",
                    phase_status="completed",
                    phase_progress=1.0,
                    phase_message="Audio generation complete"
                )

            # Phase 4: Video Assembly
            logger.info("=" * 50)
            logger.info("PHASE 4: Video Assembly")
            logger.info("=" * 50)

            self._write_status(
                "processing",
                0.75,
                "Starting video assembly",
                active_phase="phase_4",
                phase_status="processing",
                phase_progress=0.0,
                phase_message="Starting video assembly"
            )

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
            self._write_status(
                "completed",
                1.0,
                "Pipeline complete"
            )

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
    ) -> tuple[List[Path], List[Path], List[Path], Optional[Path]]:
        """Execute Phase 1: PDF processing and panel extraction."""

        # PDF to Pages
        self._write_status(
            "processing",
            0.05,
            "Converting PDF to page images",
            active_phase="phase_1",
            phase_status="processing",
            phase_progress=0.05,
            phase_message="Converting PDF to page images"
        )
        pdf_processor = PDFProcessor(
            output_dir=self.job_workspace / "pages",
            dpi=config.PDF_DPI
        )
        page_paths = pdf_processor.convert_pdf_to_pages(pdf_path)
        if len(page_paths) > 1:
            logger.info("Skipping page 1 (first page) as it is the landing/marketing page.")
            page_paths = page_paths[1:]

        # Panel Extraction
        panel_extractor = PanelExtractor(
            output_dir=self.job_workspace / "panels"
        )
        panel_paths = []
        for i, page_path in enumerate(page_paths, start=1):
            page_progress = i / len(page_paths)
            self._write_status(
                "processing",
                0.05 + 0.15 * page_progress,
                f"Extracting panels from page {i}/{len(page_paths)}",
                active_phase="phase_1",
                phase_status="processing",
                phase_progress=0.05 + 0.85 * page_progress,
                phase_message=f"Extracting panels from page {i}/{len(page_paths)}"
            )
            panels = panel_extractor.extract_panels_from_page(
                page_path, i, self.job_workspace / "panels"
            )
            panel_paths.extend([p.path for p in panels])

        # Rename to intermediate unique UUID names first to avoid collisions
        temp_renamed = []
        import uuid
        for panel_path in panel_paths:
            temp_name = f"rename_temp_{uuid.uuid4().hex}.png"
            temp_path = panel_path.parent / temp_name
            panel_path.rename(temp_path)
            temp_renamed.append(temp_path)

        # Renumber panels globally from intermediate names
        for i, temp_path in enumerate(temp_renamed, start=1):
            new_name = f"panel_P{i}.png"
            new_path = temp_path.parent / new_name
            if new_path.exists():
                new_path.unlink()
            temp_path.rename(new_path)

        # Reload updated paths
        panel_paths = sorted(
            (self.job_workspace / "panels").glob("panel_P*.png"),
            key=lambda p: int(p.stem.split('P')[1])
        )

        # Contact Sheet Generation
        self._write_status(
            "processing",
            0.20,
            "Generating panel contact sheets",
            active_phase="phase_1",
            phase_status="processing",
            phase_progress=0.95,
            phase_message="Generating panel contact sheets"
        )
        contact_gen = ContactSheetGenerator()
        contact_sheet_paths = contact_gen.generate_contact_sheet(
            panel_paths,
            self.job_workspace / "contact_sheets" / "contact_sheet"
        )

        # Compile panels to PDF
        panels_pdf_path = self.job_workspace / "panels" / "extracted_panels.pdf"
        try:
            from PIL import Image
            images = [Image.open(p).convert('RGB') for p in panel_paths]
            if images:
                images[0].save(panels_pdf_path, save_all=True, append_images=images[1:])
                logger.info(f"Successfully compiled {len(panel_paths)} panels into PDF: {panels_pdf_path}")
            else:
                logger.warning("No panels found to compile to PDF")
                panels_pdf_path = None
        except Exception as pdf_err:
            logger.error(f"Failed to compile panels to PDF: {pdf_err}")
            panels_pdf_path = None

        return page_paths, panel_paths, contact_sheet_paths, panels_pdf_path

    def _run_phase_2(
        self,
        page_paths: List[Path],
        contact_sheet_paths: List[Path],
        total_panels: int,
        panels_pdf_path: Optional[Path] = None
    ) -> StoryAnalysis:
        """Execute Phase 2: LLM story analysis."""
        self._write_status(
            "processing",
            0.30,
            "Analyzing story structure with LLM",
            active_phase="phase_2",
            phase_status="processing",
            phase_progress=0.2,
            phase_message="Analyzing story structure with LLM"
        )
        try:
            llm_director = LLMStoryDirector(provider=self.llm_provider)
            story_analysis = llm_director.analyze_manga(
                page_paths,
                contact_sheet_paths[0],
                total_panels,
                panels_pdf_path=panels_pdf_path
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
            voice=config.TTS_VOICE,
            rate=config.TTS_RATE,
            output_dir=self.job_workspace / "audio"
        )

        scripts = [
            (part.part_number, part.script)
            for part in story_analysis.parts
        ]

        audio_results = []
        total_parts = len(scripts)
        for idx, (part_number, script) in enumerate(scripts, start=1):
            self._write_status(
                "processing",
                0.50 + 0.25 * ((idx - 1) / total_parts),
                f"Generating voiceover for part {part_number}/{total_parts}",
                active_phase="phase_3",
                phase_status="processing",
                phase_progress=(idx - 1) / total_parts,
                phase_message=f"Generating voiceover for part {part_number}/{total_parts}"
            )
            result = audio_gen.generate_audio_sync(script, part_number)
            audio_results.append(result)

        audio_results.sort(key=lambda x: x.part_number)

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
                output_path=self.job_workspace / "videos" / f"part_{part.part_number}.mp4",
                focus_areas=getattr(story_analysis, "panel_focus_areas", {})
            )
            video_configs.append(video_config)

        video_results = []
        for i, config in enumerate(video_configs, start=1):
            self._write_status(
                "processing",
                0.75 + 0.20 * ((i - 1) / len(video_configs)),
                f"Assembling video part {i}/{len(video_configs)}",
                active_phase="phase_4",
                phase_status="processing",
                phase_progress=(i - 1) / len(video_configs),
                phase_message=f"Assembling video part {i}/{len(video_configs)}"
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

            script = "The crew navigates the treacherous waters of the Grand Line, where tension escalates as mysterious shadows loom ahead. The straw hats brace themselves for a major confrontation, analyzing every detail and movement of their adversaries, setting the stage for a dramatic clash of ideals and power."

            parts.append(VideoScript(
                part_number=i + 1,
                script=script,
                selected_panels=selected
            ))

            current_panel = end_panel + 1

        return StoryAnalysis(parts=parts, total_panels_selected=sum(len(p.selected_panels) for p in parts))

    def load_cached_assets(
        self,
        cached_job_id: str
    ) -> tuple[List[Path], StoryAnalysis, list]:
        """Copy cached panels, audio, and story analysis JSON into current workspace and load objects."""
        cached_workspace = self.workspace_base / cached_job_id

        # 1. Copy panels directory
        shutil.copytree(
            cached_workspace / "panels",
            self.job_workspace / "panels",
            dirs_exist_ok=True
        )

        # 2. Copy audio directory
        shutil.copytree(
            cached_workspace / "audio",
            self.job_workspace / "audio",
            dirs_exist_ok=True
        )

        # 3. Copy story_analysis.json
        shutil.copy2(
            cached_workspace / "story_analysis.json",
            self.job_workspace / "story_analysis.json"
        )

        # 4. Load panel paths
        panel_paths = sorted(
            (self.job_workspace / "panels").glob("panel_P*.png"),
            key=lambda p: int(p.stem.split('P')[1]) if p.stem.split('P')[1].isdigit() else 0
        )

        # 5. Load story analysis object
        with open(self.job_workspace / "story_analysis.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            story_analysis = StoryAnalysis.from_dict(data)

        # 6. Reconstruct Audio Results
        from modules.audio_generator import GeneratedAudio
        from mutagen.mp3 import MP3
        audio_results = []
        for part in story_analysis.parts:
            audio_path = self.job_workspace / "audio" / f"part_{part.part_number}_voiceover.mp3"
            try:
                audio = MP3(str(audio_path))
                duration_ms = int(audio.info.length * 1000)
            except Exception:
                duration_ms = 40000
            audio_results.append(GeneratedAudio(
                part_number=part.part_number,
                audio_path=audio_path,
                duration_ms=duration_ms,
                script=part.script
            ))

        return panel_paths, story_analysis, audio_results

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
        active_phase: Optional[str] = None,
        phase_status: Optional[str] = None,
        phase_progress: Optional[float] = None,
        phase_message: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Persist a lightweight status file for filesystem-based polling."""
        if active_phase:
            if phase_status is not None:
                setattr(self, f"{active_phase}_status", phase_status)
            if phase_progress is not None:
                setattr(self, f"{active_phase}_progress", phase_progress)
            if phase_message is not None:
                setattr(self, f"{active_phase}_message", phase_message)

        # Build payload
        payload = {
            "job_id": self.job_id,
            "status": status,
            "progress": progress,
            "message": message,
            "phase_1_status": "completed" if status == "completed" else self.phase_1_status,
            "phase_1_progress": 1.0 if status == "completed" else self.phase_1_progress,
            "phase_1_message": self.phase_1_message if status != "completed" else "PDF converted and panels extracted",
            "phase_2_status": "completed" if status == "completed" else self.phase_2_status,
            "phase_2_progress": 1.0 if status == "completed" else self.phase_2_progress,
            "phase_2_message": self.phase_2_message if status != "completed" else "Story analysis complete",
            "phase_3_status": "completed" if status == "completed" else self.phase_3_status,
            "phase_3_progress": 1.0 if status == "completed" else self.phase_3_progress,
            "phase_3_message": self.phase_3_message if status != "completed" else "Audio generation complete",
            "phase_4_status": "completed" if status == "completed" else self.phase_4_status,
            "phase_4_progress": 1.0 if status == "completed" else self.phase_4_progress,
            "phase_4_message": self.phase_4_message if status != "completed" else "Video assembly complete",
        }

        if error:
            payload["error_message"] = error
            # If there's an active phase that failed, set it
            if active_phase:
                setattr(self, f"{active_phase}_status", "failed")
                setattr(self, f"{active_phase}_message", f"Failed: {error}")
            else:
                for p in ["phase_1", "phase_2", "phase_3", "phase_4"]:
                    if getattr(self, f"{p}_status") == "processing":
                        setattr(self, f"{p}_status", "failed")
                        setattr(self, f"{p}_message", f"Failed: {error}")
            
            # Refresh payload values with updated properties
            payload.update({
                "phase_1_status": self.phase_1_status,
                "phase_1_message": self.phase_1_message,
                "phase_2_status": self.phase_2_status,
                "phase_2_message": self.phase_2_message,
                "phase_3_status": self.phase_3_status,
                "phase_3_message": self.phase_3_message,
                "phase_4_status": self.phase_4_status,
                "phase_4_message": self.phase_4_message,
            })

        if not self.job_workspace or not self.job_id:
            return

        try:
            status_file = self.job_workspace / "status.json"
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as write_err:
            logger.warning(f"Could not write status.json: {write_err}")
