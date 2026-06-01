"""MCP Server Implementation for ICL Workflow.

This module provides a Model Context Protocol (MCP) compatible server
that exposes the ICL workflow tools for integration with MCP clients.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import asdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core_types import WorkflowState, ToolResult
from workflow import ICLWorkflow, WorkflowConfig
from tools import (
    ToolExtractInitialMask,
    ToolGenerateSpikeMask,
    ToolMergeAndDilateMask,
    ToolInterpolateAndSubtract,
    ToolEvaluateFieldComplexity,
    ToolExcludeRegionFromMask,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ICLWorkflow.MCP")


class ICLWorkflowServer:
    """
    MCP Server for ICL Workflow.

    Provides tools for ICL subtraction with human-in-the-loop support.
    Can run as a standalone server or be integrated into larger MCP deployments.
    """

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.workflow = ICLWorkflow(self.config)

    def get_capabilities(self) -> Dict[str, Any]:
        """Return MCP server capabilities."""
        return {
            "name": "icl-workflow-server",
            "version": "0.1.0",
            "description": "MCP Server for Strong Lens ICL Subtraction Workflow",
            "tools": [
                {
                    "name": "tool_evaluate_field_complexity",
                    "description": "Evaluate field complexity to determine workflow mode",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "image_fits_path": {"type": "string"},
                            "detect_thresh": {"type": "number", "default": 2.0},
                            "bright_star_thresh": {"type": "number", "default": 1000.0},
                        },
                        "required": ["image_fits_path"]
                    }
                },
                {
                    "name": "tool_extract_initial_mask",
                    "description": "Step 1: Extract initial galaxy mask using SEP",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "image_fits_path": {"type": "string"},
                            "detect_thresh": {"type": "number", "default": 1.5},
                            "min_area": {"type": "integer", "default": 5},
                            "center_x": {"type": "number"},
                            "center_y": {"type": "number"},
                            "check_region_size": {"type": "number", "default": 100},
                        },
                        "required": ["image_fits_path"]
                    }
                },
                {
                    "name": "tool_exclude_region_from_mask",
                    "description": "Exclude specified regions from mask (for arc preservation)",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "mask_path": {"type": "string"},
                            "regions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "x": {"type": "number"},
                                        "y": {"type": "number"},
                                        "radius": {"type": "number"},
                                        "shape": {"type": "string", "enum": ["circle", "box", "polygon"]},
                                        "vertices": {"type": "array"}
                                    }
                                }
                            }
                        },
                        "required": ["mask_path", "regions"]
                    }
                },
                {
                    "name": "tool_generate_spike_mask",
                    "description": "Step 2: Generate diffraction spike mask",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "image_fits_path": {"type": "string"},
                            "instrument": {"type": "string", "enum": ["CSST", "Euclid", "HST", "JWST"]},
                            "center_x": {"type": "number"},
                            "center_y": {"type": "number"},
                            "spike_width": {"type": "number"},
                            "spike_length": {"type": "number"},
                            "rotation_angle": {"type": "number"},
                        },
                        "required": ["image_fits_path", "instrument", "center_x", "center_y"]
                    }
                },
                {
                    "name": "tool_merge_and_dilate_mask",
                    "description": "Step 3: Merge masks and apply dilation",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "seg_map_path": {"type": "string"},
                            "spike_mask_path": {"type": "string"},
                            "dilation_factor": {"type": "number", "default": 1.5},
                            "kernel_shape": {"type": "string", "enum": ["box", "disk"], "default": "disk"},
                            "image_fits_path": {"type": "string"},
                        },
                        "required": ["seg_map_path", "spike_mask_path"]
                    }
                },
                {
                    "name": "tool_interpolate_and_subtract",
                    "description": "Step 4: Interpolate ICL and subtract from image",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "image_fits_path": {"type": "string"},
                            "final_master_mask_path": {"type": "string"},
                            "method": {"type": "string", "enum": ["rbf", "clough-tocher", "linear", "nearest"], "default": "rbf"},
                        },
                        "required": ["image_fits_path", "final_master_mask_path"]
                    }
                },
                {
                    "name": "run_full_workflow",
                    "description": "Run complete ICL subtraction workflow",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "image_fits_path": {"type": "string"},
                            "instrument": {"type": "string"},
                            "center_x": {"type": "number"},
                            "center_y": {"type": "number"},
                            "mode": {"type": "string", "enum": ["auto", "semi_auto", "manual"]},
                        },
                        "required": ["image_fits_path", "instrument"]
                    }
                }
            ],
            "resources": [
                {
                    "uri": "workflow://state",
                    "name": "Current Workflow State",
                    "description": "Current state of the ICL workflow execution"
                },
                {
                    "uri": "workflow://config",
                    "name": "Workflow Configuration",
                    "description": "Current workflow configuration parameters"
                }
            ]
        }

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with given parameters."""
        logger.info(f"Executing tool: {tool_name}")

        result: ToolResult

        if tool_name == "tool_evaluate_field_complexity":
            result = ToolEvaluateFieldComplexity.execute(
                params["image_fits_path"],
                detect_thresh=params.get("detect_thresh", 2.0),
                bright_star_thresh=params.get("bright_star_thresh", 1000.0),
                output_dir=self.config.output_dir
            )

        elif tool_name == "tool_extract_initial_mask":
            result = ToolExtractInitialMask.execute(
                params["image_fits_path"],
                detect_thresh=params.get("detect_thresh", 1.5),
                min_area=params.get("min_area", 5),
                output_dir=self.config.output_dir,
                center_x=params.get("center_x"),
                center_y=params.get("center_y"),
                check_region_size=params.get("check_region_size", 100)
            )

        elif tool_name == "tool_exclude_region_from_mask":
            result = ToolExcludeRegionFromMask.execute(
                params["mask_path"],
                params["regions"],
                output_dir=self.config.output_dir
            )

        elif tool_name == "tool_generate_spike_mask":
            result = ToolGenerateSpikeMask.execute(
                params["image_fits_path"],
                params["instrument"],
                params["center_x"],
                params["center_y"],
                spike_width=params.get("spike_width"),
                spike_length=params.get("spike_length"),
                rotation_angle=params.get("rotation_angle"),
                output_dir=self.config.output_dir
            )

        elif tool_name == "tool_merge_and_dilate_mask":
            result = ToolMergeAndDilateMask.execute(
                params["seg_map_path"],
                params["spike_mask_path"],
                dilation_factor=params.get("dilation_factor", 1.5),
                kernel_shape=params.get("kernel_shape", "disk"),
                image_fits_path=params.get("image_fits_path"),
                output_dir=self.config.output_dir
            )

        elif tool_name == "tool_interpolate_and_subtract":
            result = ToolInterpolateAndSubtract.execute(
                params["image_fits_path"],
                params["final_master_mask_path"],
                method=params.get("method", "rbf"),
                output_dir=self.config.output_dir
            )

        elif tool_name == "run_full_workflow":
            final_state = self.workflow.run(
                params["image_fits_path"],
                params["instrument"],
                center_x=params.get("center_x"),
                center_y=params.get("center_y"),
                mode=params.get("mode")
            )
            return {
                "success": True,
                "state": self._serialize_state(final_state),
                "summary": self.workflow.get_summary()
            }

        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }

        # Convert ToolResult to dict
        return {
            "success": result.success,
            "message": result.message,
            "output_path": result.output_path,
            "visualization_path": result.visualization_path,
            "metrics": result.metrics,
            "requires_human_input": result.requires_human_input
        }

    def get_resource(self, uri: str) -> Dict[str, Any]:
        """Get a resource by URI."""
        if uri == "workflow://state":
            return {
                "uri": uri,
                "data": self._serialize_state(self.workflow.state)
            }
        elif uri == "workflow://config":
            return {
                "uri": uri,
                "data": {
                    "detect_thresh": self.config.detect_thresh,
                    "min_area": self.config.min_area,
                    "dilation_factor": self.config.dilation_factor,
                    "kernel_shape": self.config.kernel_shape,
                    "interpolation_method": self.config.interpolation_method,
                }
            }
        else:
            return {"error": f"Unknown resource: {uri}"}

    def _serialize_state(self, state: WorkflowState) -> Dict[str, Any]:
        """Serialize workflow state to dictionary."""
        result = {
            "step": state.step,
            "image_path": state.image_path,
            "initial_mask_path": state.initial_mask_path,
            "spike_mask_path": state.spike_mask_path,
            "final_mask_path": state.final_mask_path,
            "icl_model_path": state.icl_model_path,
            "residual_path": state.residual_path,
            "detection_threshold": state.detection_threshold,
            "dilation_factor": state.dilation_factor,
            "interpolation_method": state.interpolation_method,
            "step_1_approved": state.step_1_approved,
            "step_2_approved": state.step_2_approved,
            "step_3_approved": state.step_3_approved,
            "step_4_approved": state.step_4_approved,
        }

        if state.quality_metrics:
            result["quality_metrics"] = {
                "overall_score": state.quality_metrics.overall_score,
                "passed": state.quality_metrics.passed,
                "negative_pixel_ratio": state.quality_metrics.negative_pixel_ratio,
                "mean_residual": state.quality_metrics.mean_residual,
                "std_residual": state.quality_metrics.std_residual,
                "flux_conservation_score": state.quality_metrics.flux_conservation_score,
                "ellipticity_rmse": state.quality_metrics.ellipticity_rmse,
                "shape_preservation_score": state.quality_metrics.shape_preservation_score,
            }

        if state.field_complexity:
            result["field_complexity"] = {
                "num_bright_stars": state.field_complexity.num_bright_stars,
                "num_faint_stars": state.field_complexity.num_faint_stars,
                "complexity_score": state.field_complexity.complexity_score,
                "recommended_auto_mode": state.field_complexity.recommended_auto_mode,
            }

        return result

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        method = request.get("method")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": self.get_capabilities()
            }

        elif method == "tools/call":
            tool_name = request.get("params", {}).get("name")
            tool_params = request.get("params", {}).get("arguments", {})
            result = self.execute_tool(tool_name, tool_params)
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": result
            }

        elif method == "resources/read":
            uri = request.get("params", {}).get("uri")
            result = self.get_resource(uri)
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": result
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
