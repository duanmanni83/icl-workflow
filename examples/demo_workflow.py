#!/usr/bin/env python3
"""
Demonstration of complete ICL workflow.

This script demonstrates the full workflow with a synthetic test image.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from workflow import ICLWorkflow, WorkflowConfig
from tools import (
    ToolExtractInitialMask,
    ToolGenerateSpikeMask,
    ToolMergeAndDilateMask,
    ToolInterpolateAndSubtract,
    ToolEvaluateFieldComplexity,
)
from utils import FITSHandler, Visualization


def demo_step_by_step():
    """Demonstrate step-by-step workflow execution."""
    print("=" * 70)
    print("ICL Workflow Demonstration")
    print("=" * 70)

    # Use test image or generate one
    test_image = Path(__file__).parent / "test_image.fits"

    if not test_image.exists():
        print("\nGenerating test image...")
        import subprocess
        subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "generate_test_data.py"),
            "--output", str(test_image)
        ])

    print(f"\nInput image: {test_image}")

    output_dir = Path(__file__).parent.parent / "outputs" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Field Complexity Assessment
    print("\n" + "-" * 70)
    print("Step 0: Field Complexity Assessment")
    print("-" * 70)

    result0 = ToolEvaluateFieldComplexity.execute(
        str(test_image),
        output_dir=str(output_dir)
    )

    print(f"Result: {result0.message}")
    print(f"Metrics:")
    for key, value in result0.metrics.items():
        print(f"  {key}: {value}")

    # Step 1: Initial Mask Extraction
    print("\n" + "-" * 70)
    print("Step 1: Initial Mask Extraction")
    print("-" * 70)

    result1 = ToolExtractInitialMask.execute(
        str(test_image),
        detect_thresh=1.5,
        min_area=5,
        output_dir=str(output_dir),
        center_x=None,  # Auto-detect
        center_y=None,
        check_region_size=100
    )

    print(f"Result: {result1.message}")
    print(f"Detected {result1.metrics['num_objects']} objects")
    print(f"Visualization: {result1.visualization_path}")

    if result1.metrics.get('arc_warning'):
        print("\n⚠️  WARNING: Potential arc contamination detected!")
        print("   In production, user would review and potentially exclude regions.")

    # Step 2: Spike Mask Generation
    print("\n" + "-" * 70)
    print("Step 2: Diffraction Spike Mask Generation")
    print("-" * 70)

    cx = result1.metrics['center_x']
    cy = result1.metrics['center_y']
    print(f"Using center: ({cx:.1f}, {cy:.1f})")

    result2 = ToolGenerateSpikeMask.execute(
        str(test_image),
        instrument="CSST",
        center_x=cx,
        center_y=cy,
        output_dir=str(output_dir)
    )

    print(f"Result: {result2.message}")
    print(f"Visualization: {result2.visualization_path}")

    # Step 3: Merge and Dilate
    print("\n" + "-" * 70)
    print("Step 3: Mask Merging and Dilation")
    print("-" * 70)

    result3 = ToolMergeAndDilateMask.execute(
        result1.output_path,
        result2.output_path,
        dilation_factor=1.5,
        kernel_shape="disk",
        image_fits_path=str(test_image),
        output_dir=str(output_dir)
    )

    print(f"Result: {result3.message}")
    print(f"Coverage: {result3.metrics['original_coverage']:.1f}% -> "
          f"{result3.metrics['dilated_coverage']:.1f}%")
    print(f"Visualization: {result3.visualization_path}")

    # Step 4: ICL Interpolation and Subtraction
    print("\n" + "-" * 70)
    print("Step 4: ICL Interpolation and Subtraction")
    print("-" * 70)

    result4 = ToolInterpolateAndSubtract.execute(
        str(test_image),
        result3.output_path,
        method="rbf",
        output_dir=str(output_dir)
    )

    print(f"Result: {result4.message}")
    print(f"\nQuality Metrics:")
    print(f"  Overall Score: {result4.metrics['overall_score']:.1f}/100")
    print(f"  Flux Conservation: {result4.metrics['flux_conservation_score']:.1f}/100")
    print(f"  Shape Preservation: {result4.metrics['shape_preservation_score']:.1f}/100")
    print(f"  Negative Pixel Ratio: {result4.metrics['negative_pixel_ratio']:.4f}")
    print(f"  Ellipticity RMSE: {result4.metrics['ellipticity_rmse']:.4f}")
    print(f"  Status: {'✅ PASSED' if result4.metrics['passed'] else '❌ FAILED'}")
    print(f"\nVisualization: {result4.visualization_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\nOutput files in {output_dir}:")
    print(f"  - seg_map_initial.fits: Initial segmentation mask")
    print(f"  - spike_mask.fits: Diffraction spike mask")
    print(f"  - final_master_mask.fits: Combined and dilated mask")
    print(f"  - icl_model.fits: Estimated ICL model")
    print(f"  - clean_science_residual.fits: Clean science image")

    print("\nVisualizations:")
    print(f"  - step1_mask_overlay.png: Step 1 review")
    print(f"  - step2_spike_mask.png: Step 2 review")
    print(f"  - step3_dilation.png: Step 3 review")
    print(f"  - step4_residual_analysis.png: Final quality analysis")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


def demo_full_workflow():
    """Demonstrate using the high-level workflow API."""
    print("\n" + "=" * 70)
    print("Full Workflow API Demonstration")
    print("=" * 70)

    test_image = Path(__file__).parent / "test_image.fits"

    if not test_image.exists():
        print("\nGenerating test image...")
        import subprocess
        subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "generate_test_data.py"),
            "--output", str(test_image)
        ])

    config = WorkflowConfig(
        output_dir=str(Path(__file__).parent.parent / "outputs" / "demo_api"),
        detect_thresh=1.5,
        dilation_factor=1.5,
        interpolation_method="rbf"
    )

    workflow = ICLWorkflow(config)

    # Simple auto-approve callback
    approval_count = [0]
    def auto_callback(step_name, result):
        approval_count[0] += 1
        print(f"[HITL] {step_name}: Auto-approved")
        return True

    workflow.register_human_callback(auto_callback)

    print("\nRunning full workflow...")
    state = workflow.run(
        str(test_image),
        instrument="CSST",
        mode="auto"
    )

    print(f"\n{workflow.get_summary()}")
    print(f"\nTotal human input checkpoints: {approval_count[0]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ICL Workflow Demo")
    parser.add_argument("--full", action="store_true", help="Run full workflow API demo")
    parser.add_argument("--step", action="store_true", help="Run step-by-step demo")

    args = parser.parse_args()

    if not args.full and not args.step:
        # Run both
        demo_step_by_step()
        demo_full_workflow()
    else:
        if args.step:
            demo_step_by_step()
        if args.full:
            demo_full_workflow()
