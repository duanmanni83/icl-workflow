"""Type definitions for ICL Workflow."""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, List, Dict, Any, Union
from enum import Enum
import numpy as np


class InstrumentType(str, Enum):
    """Supported telescope instruments."""
    CSST = "CSST"
    EUCLID = "Euclid"
    HST = "HST"
    JWST = "JWST"
    CUSTOM = "Custom"


class InterpolationMethod(str, Enum):
    """Supported interpolation methods."""
    RBF = "rbf"
    CLOUGH_TOCHER = "clough-tocher"
    NEAREST = "nearest"
    LINEAR = "linear"


class KernelShape(str, Enum):
    """Dilation kernel shapes."""
    BOX = "box"
    DISK = "disk"


@dataclass
class MaskRegion:
    """Defines a region to exclude from mask."""
    x: float
    y: float
    radius: float
    shape: Literal["circle", "box", "polygon"] = "circle"
    vertices: Optional[List[Tuple[float, float]]] = None


@dataclass
class SpikeParameters:
    """Parameters for diffraction spike modeling."""
    center_x: float
    center_y: float
    spike_width: float = 3.0
    spike_length: float = 100.0
    rotation_angle: float = 0.0
    num_spikes: int = 4


@dataclass
class QualityMetrics:
    """Quality assessment metrics for ICL subtraction."""
    # Flux conservation metrics
    negative_pixel_ratio: float = 0.0
    mean_residual: float = 0.0
    std_residual: float = 0.0
    flux_conservation_score: float = 0.0

    # Ellipticity metrics
    ellipticity_rmse: float = 0.0
    ellipticity_bias_x: float = 0.0
    ellipticity_bias_y: float = 0.0
    shape_preservation_score: float = 0.0

    # Overall quality score
    overall_score: float = 0.0
    passed: bool = False


@dataclass
class FieldComplexity:
    """Field complexity assessment."""
    num_bright_stars: int = 0
    num_faint_stars: int = 0
    lens_galaxy_surface_brightness: float = 0.0
    background_complexity: float = 0.0
    recommended_auto_mode: bool = True
    complexity_score: float = 0.0


@dataclass
class WorkflowState:
    """Tracks the current state of the workflow."""
    step: int = 0
    image_path: Optional[str] = None
    initial_mask_path: Optional[str] = None
    spike_mask_path: Optional[str] = None
    final_mask_path: Optional[str] = None
    icl_model_path: Optional[str] = None
    residual_path: Optional[str] = None

    # Parameters used
    detection_threshold: float = 1.5
    dilation_factor: float = 1.5
    interpolation_method: InterpolationMethod = InterpolationMethod.RBF

    # Quality metrics
    quality_metrics: Optional[QualityMetrics] = None
    field_complexity: Optional[FieldComplexity] = None

    # Human decision flags
    step_1_approved: bool = False
    step_2_approved: bool = False
    step_3_approved: bool = False
    step_4_approved: bool = False


@dataclass
class ToolResult:
    """Standard result from tool execution."""
    success: bool
    message: str
    output_path: Optional[str] = None
    visualization_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    requires_human_input: bool = False
    state_update: Dict[str, Any] = field(default_factory=dict)