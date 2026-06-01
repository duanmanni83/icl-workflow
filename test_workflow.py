#!/usr/bin/env python3
"""
Direct test script for ICL workflow.
Run from the project root directory.
"""

import sys
from pathlib import Path
import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from workflow import ICLWorkflow, WorkflowConfig
from tools import (
    ToolExtractInitialMask,
    ToolGenerateSpikeMask,
    ToolMergeAndDilateMask,
    ToolInterpolateAndSubtract,
    ToolEvaluateFieldComplexity,
)
from utils import FITSHandler, Visualization


def test_step_by_step():
    """Run step-by-step workflow test."""
    print("=" * 70)
    print("ICL Workflow Test - Step by Step")
    print("=" * 70)

    # Use existing test image or create one
    test_image = Path(__file__).parent / "examples" / "test_image.fits"

    if not test_image.exists():
        print("\nGenerating synthetic test image...")
        from astropy.io import fits

        size = 512
        np.random.seed(42)

        # Create synthetic galaxy cluster image
        image = np.zeros((size, size), dtype=np.float64)
        cy, cx = size // 2, size // 2
        y, x = np.indices((size, size))

        # ICL background
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        icl = 5.0 * np.exp(-r / 100) + 0.5 * np.sin(x / 30) * np.cos(y / 30) * np.exp(-r / 150)
        image += icl

        # Central lens galaxy
        lens = 50.0 * np.exp(-((x - cx)**2 / (2 * 15**2) + (y - cy)**2 / (2 * 12**2)))
        image += lens

        # Gravitational arc
        arc_angle = 45
        arc_dist = 35
        for offset in [0, 180]:
            angle = np.deg2rad(arc_angle + offset)
            arc_cx = cx + arc_dist * np.cos(angle)
            arc_cy = cy + arc_dist * np.sin(angle)
            arc = 8.0 * np.exp(-((x - arc_cx)**2 / (2 * 20**2) + (y - arc_cy)**2 / (2 * 3**2)))
            image += arc

        # Diffraction spikes
        for angle_deg in [0, 45, 90, 135]:
            angle = np.deg2rad(angle_deg)
            for dist in range(15, 150):
                for w in range(-2, 3):
                    sx = int(cx + dist * np.cos(angle) + w * np.sin(angle))
                    sy = int(cy + dist * np.sin(angle) - w * np.cos(angle))
                    if 0 <= sx < size and 0 <= sy < size:
                        image[sy, sx] += 3.0 * np.exp(-dist / 50)

        # Background sources
        for _ in range(20):
            bx = np.random.randint(50, size - 50)
            by = np.random.randint(50, size - 50)
            if np.sqrt((bx - cx)**2 + (by - cy)**2) > 80:
                source = 5.0 * np.exp(-((x - bx)**2 + (y - by)**2) / (2 * 3**2))
                image += source

        # Noise
        image += np.random.normal(0, 0.1, (size, size))
        image = np.maximum(image, 0.001)

        # Create FITS
        header = fits.Header()
        header['CRPIX1'] = cx
        header['CRPIX2'] = cy
        header['CRVAL1'] = 150.0
        header['CRVAL2'] = 2.0
        header['CDELT1'] = -0.0001
        header['CDELT2'] = 0.0001
        header['CTYPE1'] = 'RA---TAN'
        header['CTYPE2'] = 'DEC--TAN'
        header['TELESCOP'] = 'CSST'
        header['FILTER'] = 'F150W'

        hdu = fits.PrimaryHDU(image, header=header)
        hdu.writeto(test_image, overwrite=True)
        print(f"Created: {test_image}")

    output_dir = Path(__file__).parent / "outputs" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTest image: {test_image}")
    print(f"Output directory: {output_dir}")

    # Step 0: Field Complexity
    print("\n" + "-" * 70)
    print("Step 0: Field Complexity Assessment")
    print("-" * 70)

    result0 = ToolEvaluateFieldComplexity.execute(
        str(test_image),
        output_dir=str(output_dir)
    )
    print(f"✓ {result0.message}")
    print(f"  Complexity Score: {result0.metrics.get('complexity_score', 'N/A')}")
    print(f"  Recommended Mode: {'AUTO' if result0.metrics.get('recommended_auto_mode') else 'MANUAL'}")

    # Step 1: Initial Mask
    print("\n" + "-" * 70)
    print("Step 1: Initial Mask Extraction")
    print("-" * 70)

    result1 = ToolExtractInitialMask.execute(
        str(test_image),
        detect_thresh=1.5,
        min_area=5,
        output_dir=str(output_dir),
        check_region_size=100
    )
    print(f"✓ {result1.message}")
    print(f"  Objects detected: {result1.metrics.get('num_objects', 'N/A')}")
    print(f"  Visualization: {result1.visualization_path}")

    if result1.metrics.get('arc_warning'):
        print("  ⚠️  Warning: Potential arc contamination!")

    cx = result1.metrics.get('center_x')
    cy = result1.metrics.get('center_y')

    # Step 2: Spike Mask
    print("\n" + "-" * 70)
    print("Step 2: Diffraction Spike Mask")
    print("-" * 70)

    result2 = ToolGenerateSpikeMask.execute(
        str(test_image),
        instrument="CSST",
        center_x=cx,
        center_y=cy,
        output_dir=str(output_dir)
    )
    print(f"✓ {result2.message}")
    print(f"  Instrument: {result2.metrics.get('instrument')}")
    print(f"  Num spikes: {result2.metrics.get('num_spikes')}")
    print(f"  Visualization: {result2.visualization_path}")

    # Step 3: Merge and Dilate
    print("\n" + "-" * 70)
    print("Step 3: Merge and Dilate Masks")
    print("-" * 70)

    result3 = ToolMergeAndDilateMask.execute(
        result1.output_path,
        result2.output_path,
        dilation_factor=1.5,
        kernel_shape="disk",
        image_fits_path=str(test_image),
        output_dir=str(output_dir)
    )
    print(f"✓ {result3.message}")
    print(f"  Original coverage: {result3.metrics.get('original_coverage', 0):.1f}%")
    print(f"  Dilated coverage: {result3.metrics.get('dilated_coverage', 0):.1f}%")
    print(f"  Visualization: {result3.visualization_path}")

    # Step 4: ICL Subtraction
    print("\n" + "-" * 70)
    print("Step 4: ICL Interpolation and Subtraction")
    print("-" * 70)

    result4 = ToolInterpolateAndSubtract.execute(
        str(test_image),
        result3.output_path,
        method="rbf",
        output_dir=str(output_dir)
    )
    print(f"✓ {result4.message}")

    metrics = result4.metrics
    print(f"\n  Quality Assessment:")
    print(f"    Overall Score: {metrics.get('overall_score', 0):.1f}/100")
    print(f"    Flux Conservation: {metrics.get('flux_conservation_score', 0):.1f}/100")
    print(f"    Shape Preservation: {metrics.get('shape_preservation_score', 0):.1f}/100")
    print(f"    Negative Pixel Ratio: {metrics.get('negative_pixel_ratio', 0):.4f}")
    print(f"    Ellipticity RMSE: {metrics.get('ellipticity_rmse', 0):.4f}")
    print(f"    Status: {'✅ PASSED' if metrics.get('passed') else '❌ FAILED'}")
    print(f"    Visualization: {result4.visualization_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nOutput files in {output_dir}:")
    print("  - seg_map_initial.fits: Initial segmentation mask")
    print("  - spike_mask.fits: Diffraction spike mask")
    print("  - final_master_mask.fits: Combined mask")
    print("  - icl_model.fits: ICL model")
    print("  - clean_science_residual.fits: Final cleaned image")
    print("\nVisualizations:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")

    print("\n" + "=" * 70)
    print("✓ Test completed successfully!")
    print("=" * 70)

    return result4.success


if __name__ == "__main__":
    success = test_step_by_step()
    sys.exit(0 if success else 1)
