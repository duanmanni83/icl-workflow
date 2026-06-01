#!/usr/bin/env python3
"""Example usage of ICL Workflow."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflow import ICLWorkflow, WorkflowConfig


def example_simple():
    """Simple example with auto mode."""
    print("="*60)
    print("Example: Simple ICL Workflow (Auto Mode)")
    print("="*60)

    config = WorkflowConfig(
        output_dir="./outputs/example_simple",
        detect_thresh=1.5,
        dilation_factor=1.5,
        interpolation_method="rbf",
    )

    workflow = ICLWorkflow(config)

    # Simulate HITL callback (auto-approve everything)
    workflow.register_human_callback(lambda step, result: True)

    # This would work with a real FITS file:
    # state = workflow.run(
    #     "path/to/image.fits",
    #     instrument="CSST",
    #     mode="auto"
    # )

    print("Configuration:")
    print(f"  Detection threshold: {config.detect_thresh}")
    print(f"  Dilation factor: {config.dilation_factor}")
    print(f"  Interpolation: {config.interpolation_method}")
    print("\nTo run with real data:")
    print('  state = workflow.run("image.fits", "CSST", mode="auto")')


def example_manual_hitl():
    """Example with manual human-in-the-loop."""
    print("\n" + "="*60)
    print("Example: Manual HITL Mode")
    print("="*60)

    config = WorkflowConfig(
        output_dir="./outputs/example_manual",
        detect_thresh=1.5,
        dilation_factor=2.0,
    )

    workflow = ICLWorkflow(config)

    # Interactive HITL callback
    def interactive_hitl(step_name: str, result):
        print(f"\n[HITL] {step_name}")
        print(f"  Message: {result.message}")
        print(f"  Output: {result.output_path}")
        print(f"  Viz: {result.visualization_path}")

        if result.metrics:
            print(f"  Metrics:")
            for key, value in result.metrics.items():
                print(f"    {key}: {value}")

        # In real usage, human would review visualization
        # For demo, just ask
        return input("  Approve? (y/n): ").strip().lower() == 'y'

    workflow.register_human_callback(interactive_hitl)

    print("HITL callback registered.")
    print("Each step will pause for human approval.")


def example_step_by_step():
    """Example of running workflow step by step."""
    print("\n" + "="*60)
    print("Example: Step-by-Step Execution")
    print("="*60)

    code = '''
from tools import (
    ToolExtractInitialMask,
    ToolGenerateSpikeMask,
    ToolMergeAndDilateMask,
    ToolInterpolateAndSubtract,
)

# Step 1: Initial mask
result1 = ToolExtractInitialMask.execute(
    "image.fits",
    detect_thresh=1.5,
    output_dir="./outputs"
)
print(f"Step 1: {result1.message}")

# Human reviews visualization at result1.visualization_path
# If arcs are masked, exclude regions:
from tools import ToolExcludeRegionFromMask
result_fix = ToolExcludeRegionFromMask.execute(
    result1.output_path,
    regions=[{"x": 100, "y": 100, "radius": 20, "shape": "circle"}],
    output_dir="./outputs"
)

# Step 2: Spike mask
result2 = ToolGenerateSpikeMask.execute(
    "image.fits",
    instrument="CSST",
    center_x=result1.metrics['center_x'],
    center_y=result1.metrics['center_y'],
    output_dir="./outputs"
)

# Step 3: Merge and dilate
result3 = ToolMergeAndDilateMask.execute(
    result_fix.output_path,  # or result1.output_path if no fix needed
    result2.output_path,
    dilation_factor=1.5,
    output_dir="./outputs"
)

# Step 4: ICL subtraction
result4 = ToolInterpolateAndSubtract.execute(
    "image.fits",
    result3.output_path,
    method="rbf",
    output_dir="./outputs"
)

print(f"Quality Score: {result4.metrics['overall_score']}")
'''
    print(code)


if __name__ == "__main__":
    example_simple()
    example_manual_hitl()
    example_step_by_step()
