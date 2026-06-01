"""ICL Workflow: MCP-based Strong Lens ICL Subtraction."""

__version__ = "0.1.0"

from core_types import (
    WorkflowState,
    ToolResult,
    InstrumentType,
    InterpolationMethod,
    KernelShape,
    MaskRegion,
    SpikeParameters,
    QualityMetrics,
    FieldComplexity,
)
from mcp_server import ICLWorkflowServer
from workflow import ICLWorkflow, WorkflowConfig
from tools import (
    ToolExtractInitialMask,
    ToolGenerateSpikeMask,
    ToolMergeAndDilateMask,
    ToolInterpolateAndSubtract,
    ToolEvaluateFieldComplexity,
    ToolExcludeRegionFromMask,
)
from utils import FITSHandler, Visualization, Metrics

__all__ = [
    "WorkflowState",
    "ToolResult",
    "InstrumentType",
    "InterpolationMethod",
    "KernelShape",
    "MaskRegion",
    "SpikeParameters",
    "QualityMetrics",
    "FieldComplexity",
    "ICLWorkflowServer",
    "ICLWorkflow",
    "WorkflowConfig",
    "ToolExtractInitialMask",
    "ToolGenerateSpikeMask",
    "ToolMergeAndDilateMask",
    "ToolInterpolateAndSubtract",
    "ToolEvaluateFieldComplexity",
    "ToolExcludeRegionFromMask",
    "FITSHandler",
    "Visualization",
    "Metrics",
]
