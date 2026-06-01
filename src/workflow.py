"""ICL Workflow Orchestrator."""

import os
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import logging

import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core_types import WorkflowState, ToolResult, FieldComplexity
from tools import (
    ToolExtractInitialMask,
    ToolGenerateSpikeMask,
    ToolMergeAndDilateMask,
    ToolInterpolateAndSubtract,
    ToolEvaluateFieldComplexity,
    ToolExcludeRegionFromMask,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ICLWorkflow")


@dataclass
class WorkflowConfig:
    """Configuration for ICL workflow."""
    # Detection parameters
    detect_thresh: float = 1.5
    min_area: int = 5

    # Spike parameters
    spike_width: Optional[float] = None
    spike_length: Optional[float] = None
    rotation_angle: Optional[float] = None

    # Dilation parameters
    dilation_factor: float = 1.5
    kernel_shape: str = "disk"

    # Interpolation parameters
    interpolation_method: str = "rbf"

    # Auto-mode thresholds
    auto_mode_threshold: float = 20.0
    semi_auto_threshold: float = 50.0

    # Output directory
    output_dir: str = "./outputs"


class ICLWorkflow:
    """
    Orchestrates the ICL subtraction workflow with HITL support.

    Supports three modes based on field complexity:
    - AUTO: Minimal human intervention (complexity < 20, no bright stars)
    - SEMI_AUTO: Review after final step (complexity < 50)
    - MANUAL: Human input at each step (complexity >= 50)
    """

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.state = WorkflowState()
        self._human_callback: Optional[Callable[[str, ToolResult], bool]] = None

        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

    def register_human_callback(self, callback: Callable[[str, ToolResult], bool]):
        """
        Register a callback function for human-in-the-loop decisions.

        The callback receives (step_name, tool_result) and returns True to approve, False to reject.
        """
        self._human_callback = callback

    def run(self, image_fits_path: str, instrument: str,
            center_x: Optional[float] = None,
            center_y: Optional[float] = None,
            mode: Optional[str] = None) -> WorkflowState:
        """
        Run the complete ICL workflow.

        Args:
            image_fits_path: Path to input FITS image
            instrument: Telescope instrument (CSST, Euclid, HST, JWST)
            center_x: Lens galaxy center X (auto-detected if None)
            center_y: Lens galaxy center Y (auto-detected if None)
            mode: "auto", "semi_auto", "manual", or None for auto-detect

        Returns:
            Final workflow state
        """
        self.state.image_path = image_fits_path

        # Step 0: Evaluate field complexity
        logger.info("=" * 60)
        logger.info("Step 0: Evaluating field complexity...")
        logger.info("=" * 60)

        complexity_result = ToolEvaluateFieldComplexity.execute(
            image_fits_path,
            detect_thresh=self.config.detect_thresh,
            output_dir=self.config.output_dir
        )

        self._log_result(complexity_result)

        if complexity_result.state_update.get('field_complexity'):
            self.state.field_complexity = complexity_result.state_update['field_complexity']

        # Determine mode
        if mode is None:
            if self.state.field_complexity:
                if self.state.field_complexity.complexity_score < self.config.auto_mode_threshold:
                    mode = "auto"
                elif self.state.field_complexity.complexity_score < self.config.semi_auto_threshold:
                    mode = "semi_auto"
                else:
                    mode = "manual"
            else:
                mode = "manual"

        logger.info(f"Workflow mode: {mode.upper()}")

        # Step 1: Initial mask extraction
        logger.info("=" * 60)
        logger.info("Step 1: Extracting initial mask...")
        logger.info("=" * 60)

        result1 = ToolExtractInitialMask.execute(
            image_fits_path,
            detect_thresh=self.config.detect_thresh,
            min_area=self.config.min_area,
            output_dir=self.config.output_dir,
            center_x=center_x,
            center_y=center_y,
        )

        self._log_result(result1)

        if result1.success:
            self.state.initial_mask_path = result1.output_path
            self.state.step_1_approved = self._handle_human_input("Step 1: Initial Mask", result1, mode)

            if not self.state.step_1_approved:
                logger.warning("Step 1 rejected - workflow may need parameter adjustment")
                # In real implementation, could loop back with adjusted parameters

        # Step 2: Spike mask generation
        logger.info("=" * 60)
        logger.info("Step 2: Generating spike mask...")
        logger.info("=" * 60)

        if result1.success and result1.metrics:
            cx = center_x or result1.metrics.get('center_x')
            cy = center_y or result1.metrics.get('center_y')
        else:
            cx, cy = center_x, center_y

        result2 = ToolGenerateSpikeMask.execute(
            image_fits_path,
            instrument=instrument,
            center_x=cx,
            center_y=cy,
            spike_width=self.config.spike_width,
            spike_length=self.config.spike_length,
            rotation_angle=self.config.rotation_angle,
            output_dir=self.config.output_dir,
        )

        self._log_result(result2)

        if result2.success:
            self.state.spike_mask_path = result2.output_path
            self.state.step_2_approved = self._handle_human_input("Step 2: Spike Mask", result2, mode)

        # Step 3: Merge and dilate masks
        logger.info("=" * 60)
        logger.info("Step 3: Merging and dilating masks...")
        logger.info("=" * 60)

        if self.state.initial_mask_path and self.state.spike_mask_path:
            result3 = ToolMergeAndDilateMask.execute(
                self.state.initial_mask_path,
                self.state.spike_mask_path,
                dilation_factor=self.config.dilation_factor,
                kernel_shape=self.config.kernel_shape,
                image_fits_path=image_fits_path,
                output_dir=self.config.output_dir,
            )

            self._log_result(result3)

            if result3.success:
                self.state.final_mask_path = result3.output_path
                self.state.step_3_approved = self._handle_human_input("Step 3: Mask Dilation", result3, mode)

        # Step 4: Interpolate and subtract
        logger.info("=" * 60)
        logger.info("Step 4: ICL interpolation and subtraction...")
        logger.info("=" * 60)

        if self.state.final_mask_path:
            result4 = ToolInterpolateAndSubtract.execute(
                image_fits_path,
                self.state.final_mask_path,
                method=self.config.interpolation_method,
                output_dir=self.config.output_dir,
            )

            self._log_result(result4)

            if result4.success:
                self.state.icl_model_path = result4.state_update.get('icl_model_path')
                self.state.residual_path = result4.output_path
                self.state.quality_metrics = result4.state_update.get('quality_metrics')
                self.state.step_4_approved = self._handle_human_input("Step 4: ICL Subtraction", result4, mode)

        logger.info("=" * 60)
        logger.info("Workflow complete!")
        logger.info(f"Residual image: {self.state.residual_path}")
        logger.info("=" * 60)

        return self.state

    def _handle_human_input(self, step_name: str, result: ToolResult, mode: str) -> bool:
        """
        Handle human-in-the-loop decision.

        In AUTO mode, skips human input unless quality checks fail.
        In SEMI_AUTO mode, only requires input at final step.
        In MANUAL mode, always requires human input.
        """
        if mode == "auto":
            # In auto mode, only check if quality metrics indicate failure
            if result.metrics and not result.metrics.get('passed', True):
                logger.warning(f"Quality check failed at {step_name}")
                return self._invoke_human_callback(step_name, result)
            return True

        elif mode == "semi_auto":
            # Only require input at Step 4
            if "Step 4" in step_name:
                return self._invoke_human_callback(step_name, result)
            return True

        else:  # manual mode
            return self._invoke_human_callback(step_name, result)

    def _invoke_human_callback(self, step_name: str, result: ToolResult) -> bool:
        """Invoke the registered human callback or default to True."""
        if self._human_callback:
            try:
                return self._human_callback(step_name, result)
            except Exception as e:
                logger.error(f"Human callback failed: {e}")
                return False
        else:
            # Default behavior: log and approve
            logger.info(f"[HITL] {step_name}: Auto-approved (no callback registered)")
            logger.info(f"  Visualization: {result.visualization_path}")
            return True

    def _log_result(self, result: ToolResult):
        """Log tool execution result."""
        status = "✓" if result.success else "✗"
        logger.info(f"{status} {result.message}")
        if result.output_path:
            logger.info(f"  Output: {result.output_path}")
        if result.visualization_path:
            logger.info(f"  Visualization: {result.visualization_path}")
        if result.metrics:
            logger.debug(f"  Metrics: {result.metrics}")

    def step_1_exclude_regions(self, regions: list) -> ToolResult:
        """
        Helper method to exclude regions from initial mask (Step 1 correction).

        Args:
            regions: List of {x, y, radius, shape} dictionaries

        Returns:
            Updated mask result
        """
        if not self.state.initial_mask_path:
            return ToolResult(
                success=False,
                message="No initial mask available - run Step 1 first"
            )

        result = ToolExcludeRegionFromMask.execute(
            self.state.initial_mask_path,
            regions,
            self.config.output_dir
        )

        if result.success:
            self.state.initial_mask_path = result.output_path

        return result

    def get_summary(self) -> str:
        """Get a summary of the workflow execution."""
        lines = [
            "=" * 60,
            "ICL Workflow Summary",
            "=" * 60,
            f"Input Image: {self.state.image_path}",
            f"Workflow Mode: {self.state.field_complexity.recommended_auto_mode if self.state.field_complexity else 'Unknown'}",
            "",
            "Steps Completed:",
            f"  Step 1 (Initial Mask): {'✓' if self.state.initial_mask_path else '✗'} "
            f"(Approved: {'✓' if self.state.step_1_approved else '✗'})",
            f"  Step 2 (Spike Mask): {'✓' if self.state.spike_mask_path else '✗'} "
            f"(Approved: {'✓' if self.state.step_2_approved else '✗'})",
            f"  Step 3 (Mask Merge): {'✓' if self.state.final_mask_path else '✗'} "
            f"(Approved: {'✓' if self.state.step_3_approved else '✗'})",
            f"  Step 4 (ICL Subtraction): {'✓' if self.state.residual_path else '✗'} "
            f"(Approved: {'✓' if self.state.step_4_approved else '✗'})",
            "",
            "Output Files:",
        ]

        if self.state.initial_mask_path:
            lines.append(f"  Initial Mask: {self.state.initial_mask_path}")
        if self.state.spike_mask_path:
            lines.append(f"  Spike Mask: {self.state.spike_mask_path}")
        if self.state.final_mask_path:
            lines.append(f"  Final Mask: {self.state.final_mask_path}")
        if self.state.icl_model_path:
            lines.append(f"  ICL Model: {self.state.icl_model_path}")
        if self.state.residual_path:
            lines.append(f"  Residual: {self.state.residual_path}")

        if self.state.quality_metrics:
            lines.extend([
                "",
                "Quality Metrics:",
                f"  Overall Score: {self.state.quality_metrics.overall_score:.1f}/100",
                f"  Flux Conservation: {self.state.quality_metrics.flux_conservation_score:.1f}/100",
                f"  Shape Preservation: {self.state.quality_metrics.shape_preservation_score:.1f}/100",
                f"  Negative Pixel Ratio: {self.state.quality_metrics.negative_pixel_ratio:.4f}",
                f"  Ellipticity RMSE: {self.state.quality_metrics.ellipticity_rmse:.4f}",
                f"  Final Result: {'PASSED' if self.state.quality_metrics.passed else 'FAILED'}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)
