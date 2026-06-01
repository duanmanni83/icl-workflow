"""ICL Workflow Tools - MCP Tool Implementations."""

import numpy as np
from scipy import ndimage
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
from typing import Optional, List, Tuple, Dict, Any
import logging
import sep

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core_types import (
    ToolResult, InstrumentType, InterpolationMethod, KernelShape,
    SpikeParameters, MaskRegion, FieldComplexity
)
from utils import FITSHandler, Visualization, Metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ICLWorkflow")


class ToolExtractInitialMask:
    """
    Step 1: Extract initial galaxy mask using SEP (Source Extractor Python).

    Human Decision Point: Check if gravitational arcs are incorrectly masked.
    """

    @staticmethod
    def execute(
        image_fits_path: str,
        detect_thresh: float = 1.5,
        min_area: int = 5,
        output_dir: str = "./outputs",
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        check_region_size: float = 100,
    ) -> ToolResult:
        """
        Extract initial mask from FITS image using SEP.

        Args:
            image_fits_path: Path to input FITS image
            detect_thresh: Detection threshold in sigma (default: 1.5)
            min_area: Minimum pixel area for detection (default: 5)
            output_dir: Output directory for results
            center_x: X coordinate of lens galaxy center (optional)
            center_y: Y coordinate of lens galaxy center (optional)
            check_region_size: Size of region to visualize for HITL

        Returns:
            ToolResult with mask path and visualization
        """
        try:
            # Read FITS
            logger.info(f"Reading FITS: {image_fits_path}")
            data, wcs, header = FITSHandler.read_fits(image_fits_path)

            # Handle NaN/Inf
            data_clean = np.copy(data)
            data_clean[~np.isfinite(data_clean)] = 0

            # SEP background estimation
            logger.info("Estimating background with SEP...")
            bkg = sep.Background(data_clean)
            data_sub = data_clean - bkg

            # Object extraction
            logger.info(f"Extracting objects (thresh={detect_thresh}, min_area={min_area})...")
            objects = sep.extract(data_sub, detect_thresh, err=bkg.globalrms, minarea=min_area)
            logger.info(f"Detected {int(len(objects))} objects")

            # Create segmentation map (must be uint8 for sep)
            seg_map = np.zeros_like(data, dtype=np.uint8)
            for i, obj in enumerate(objects):
                # Simple ellipse approximation for each object
                sep.mask_ellipse(seg_map, obj['x'], obj['y'], obj['a'], obj['b'],
                                obj['theta'], r=2.0)

            # Binary mask (1 = masked, 0 = good)
            mask = (seg_map > 0).astype(np.uint8)

            # Determine center if not provided
            if center_x is None or center_y is None:
                # Use brightest object or image center
                if len(objects) > 0:
                    brightest = np.argmax(objects['flux'])
                    center_x = float(objects[brightest]['x'])
                    center_y = float(objects[brightest]['y'])
                else:
                    center_y, center_x = np.array(data.shape) / 2

            # Save mask
            import os
            os.makedirs(output_dir, exist_ok=True)
            mask_path = os.path.join(output_dir, "seg_map_initial.fits")
            FITSHandler.write_fits(mask, mask_path, header)

            # Generate visualization for HITL
            viz_path = os.path.join(output_dir, "step1_mask_overlay.png")
            Visualization.create_mask_overlay(
                data, mask, center_x, center_y, check_region_size,
                title=f"Step 1: Initial Mask (thresh={detect_thresh}, n_obj={int(len(objects))})",
                output_path=viz_path
            )

            # Check for potential arc contamination
            arc_warning = False
            if len(objects) > 0:
                # Check for elongated objects that might be arcs
                elongation = objects['a'] / np.maximum(objects['b'], 1e-10)
                elongated = elongation > 3.0
                if np.any(elongated):
                    arc_warning = True

            message = f"Extracted initial mask with {int(len(objects))} objects. "
            if arc_warning:
                message += "WARNING: Detected elongated objects that might be gravitational arcs. Please verify!"
            else:
                message += "Please review the mask overlay to verify no arcs were incorrectly masked."

            return ToolResult(
                success=True,
                message=message,
                output_path=mask_path,
                visualization_path=viz_path,
                metrics={
                    'num_objects': int(len(objects)),
                    'detection_threshold': detect_thresh,
                    'arc_warning': bool(arc_warning),
                    'center_x': float(center_x),
                    'center_y': float(center_y),
                },
                requires_human_input=True,
                state_update={
                    'step': 1,
                    'initial_mask_path': mask_path,
                    'detection_threshold': detect_thresh,
                }
            )

        except Exception as e:
            import traceback
            logger.error(f"Error in ToolExtractInitialMask: {str(e)}")
            logger.error(traceback.format_exc())
            return ToolResult(
                success=False,
                message=f"Failed to extract initial mask: {str(e)}",
                requires_human_input=False
            )


class ToolExcludeRegionFromMask:
    """
    Helper tool to manually exclude regions from mask.
    Used when gravitational arcs are incorrectly masked.
    """

    @staticmethod
    def execute(
        mask_path: str,
        regions: List[Dict[str, Any]],  # List of {x, y, radius, shape}
        output_dir: str = "./outputs",
    ) -> ToolResult:
        """
        Exclude specified regions from the mask.

        Args:
            mask_path: Path to input mask FITS
            regions: List of regions to exclude (unmask)
            output_dir: Output directory

        Returns:
            ToolResult with updated mask
        """
        try:
            mask, wcs, header = FITSHandler.read_fits(mask_path)
            mask = mask.astype(np.uint8)

            for region in regions:
                x = region['x']
                y = region['y']
                radius = region.get('radius', 10)
                shape = region.get('shape', 'circle')

                if shape == 'circle':
                    y_idx, x_idx = np.indices(mask.shape)
                    dist = np.sqrt((x_idx - x)**2 + (y_idx - y)**2)
                    mask[dist < radius] = 0
                elif shape == 'box':
                    y_min = int(max(0, y - radius))
                    y_max = int(min(mask.shape[0], y + radius))
                    x_min = int(max(0, x - radius))
                    x_max = int(min(mask.shape[1], x + radius))
                    mask[y_min:y_max, x_min:x_max] = 0
                elif shape == 'polygon' and 'vertices' in region:
                    from matplotlib.path import Path
                    vertices = region['vertices']
                    y_idx, x_idx = np.indices(mask.shape)
                    points = np.column_stack((x_idx.ravel(), y_idx.ravel()))
                    path = Path(vertices)
                    inside = path.contains_points(points).reshape(mask.shape)
                    mask[inside] = 0

            import os
            output_path = os.path.join(output_dir, "seg_map_initial_excluded.fits")
            FITSHandler.write_fits(mask, output_path, header)

            return ToolResult(
                success=True,
                message=f"Excluded {len(regions)} regions from mask",
                output_path=output_path,
                state_update={'initial_mask_path': output_path}
            )

        except Exception as e:
            logger.error(f"Error in ToolExcludeRegionFromMask: {str(e)}")
            return ToolResult(
                success=False,
                message=f"Failed to exclude regions: {str(e)}"
            )


class ToolGenerateSpikeMask:
    """
    Step 2: Generate diffraction spike mask.

    Human Decision Point: Verify geometric alignment of spike mask.
    """

    # Instrument-specific spike parameters
    INSTRUMENT_PARAMS = {
        InstrumentType.CSST: {
            'num_spikes': 4,
            'default_width': 3.0,
            'default_length': 150.0,
            'default_rotation': 0.0,
        },
        InstrumentType.EUCLID: {
            'num_spikes': 6,
            'default_width': 2.5,
            'default_length': 120.0,
            'default_rotation': 0.0,
        },
        InstrumentType.HST: {
            'num_spikes': 4,
            'default_width': 2.0,
            'default_length': 100.0,
            'default_rotation': 45.0,  # HST has rotated spikes
        },
        InstrumentType.JWST: {
            'num_spikes': 6,
            'default_width': 2.0,
            'default_length': 80.0,
            'default_rotation': 0.0,
        },
    }

    @staticmethod
    def execute(
        image_fits_path: str,
        instrument: str,
        center_x: float,
        center_y: float,
        spike_width: Optional[float] = None,
        spike_length: Optional[float] = None,
        rotation_angle: Optional[float] = None,
        output_dir: str = "./outputs",
    ) -> ToolResult:
        """
        Generate diffraction spike mask.

        Args:
            image_fits_path: Path to input FITS
            instrument: Telescope instrument name
            center_x: Lens galaxy center X
            center_y: Lens galaxy center Y
            spike_width: Override default spike width
            spike_length: Override default spike length
            rotation_angle: Override default rotation
            output_dir: Output directory

        Returns:
            ToolResult with spike mask
        """
        try:
            # Get instrument parameters
            instr_type = InstrumentType(instrument)
            params = ToolGenerateSpikeMask.INSTRUMENT_PARAMS.get(
                instr_type,
                ToolGenerateSpikeMask.INSTRUMENT_PARAMS[InstrumentType.CSST]
            )

            # Use defaults if not specified
            width = spike_width if spike_width is not None else params['default_width']
            length = spike_length if spike_length is not None else params['default_length']
            rotation = rotation_angle if rotation_angle is not None else params['default_rotation']
            num_spikes = params['num_spikes']

            # Read image for shape reference
            data, wcs, header = FITSHandler.read_fits(image_fits_path)

            # Generate spike mask
            mask = np.zeros_like(data, dtype=np.uint8)

            y_idx, x_idx = np.indices(data.shape)

            for i in range(num_spikes):
                angle_deg = rotation + i * (180 / num_spikes)
                angle_rad = np.deg2rad(angle_deg)

                # Vector from center in spike direction
                dx = x_idx - center_x
                dy = y_idx - center_y

                # Project onto spike direction
                parallel = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
                perpendicular = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)

                # Spike condition: within width perpendicular, within length parallel
                in_spike = (
                    (parallel > 0) & (parallel < length) &
                    (np.abs(perpendicular) < width / 2)
                )
                mask[in_spike] = 1

            # Save mask
            import os
            os.makedirs(output_dir, exist_ok=True)
            mask_path = os.path.join(output_dir, "spike_mask.fits")
            FITSHandler.write_fits(mask, mask_path, header)

            # Generate visualization
            viz_path = os.path.join(output_dir, "step2_spike_mask.png")
            Visualization.create_spike_visualization(
                data, mask, center_x, center_y,
                {
                    'num_spikes': num_spikes,
                    'rotation_angle': rotation,
                    'spike_width': width,
                    'spike_length': length,
                },
                title=f"Step 2: Spike Mask ({instrument}, width={width:.1f})",
                output_path=viz_path
            )

            return ToolResult(
                success=True,
                message=f"Generated spike mask for {instrument}. Please verify alignment matches actual spike angles.",
                output_path=mask_path,
                visualization_path=viz_path,
                metrics={
                    'instrument': instrument,
                    'num_spikes': num_spikes,
                    'spike_width': width,
                    'spike_length': length,
                    'rotation_angle': rotation,
                },
                requires_human_input=True,
                state_update={
                    'step': 2,
                    'spike_mask_path': mask_path,
                }
            )

        except Exception as e:
            logger.error(f"Error in ToolGenerateSpikeMask: {str(e)}")
            return ToolResult(
                success=False,
                message=f"Failed to generate spike mask: {str(e)}",
                requires_human_input=False
            )


class ToolMergeAndDilateMask:
    """
    Step 3: Merge masks and apply dilation.

    Human Decision Point: Check dilation doesn't swallow background sources.
    """

    @staticmethod
    def execute(
        seg_map_path: str,
        spike_mask_path: str,
        dilation_factor: float = 1.5,
        kernel_shape: str = "disk",
        image_fits_path: Optional[str] = None,
        output_dir: str = "./outputs",
    ) -> ToolResult:
        """
        Merge segmentation and spike masks, then dilate.

        Args:
            seg_map_path: Path to initial segmentation mask
            spike_mask_path: Path to spike mask
            dilation_factor: Dilation factor (default: 1.5)
            kernel_shape: "box" or "disk" (default: disk)
            image_fits_path: Optional original image for visualization
            output_dir: Output directory

        Returns:
            ToolResult with final merged mask
        """
        try:
            # Read masks
            seg_mask, _, header = FITSHandler.read_fits(seg_map_path)
            spike_mask, _, _ = FITSHandler.read_fits(spike_mask_path)

            # Ensure binary
            seg_mask = (seg_mask > 0).astype(np.uint8)
            spike_mask = (spike_mask > 0).astype(np.uint8)

            # Merge (union)
            merged = np.maximum(seg_mask, spike_mask)

            # Calculate dilation iterations based on factor
            # Approximate: 1 iteration ≈ 1 pixel expansion
            iterations = max(1, int(dilation_factor))

            # Create kernel
            if kernel_shape == "disk":
                kernel = ndimage.generate_binary_structure(2, 2)  # Disk-like
            else:
                kernel = np.ones((3, 3), dtype=bool)  # Box

            # Dilate
            logger.info(f"Applying dilation (factor={dilation_factor}, shape={kernel_shape})...")
            dilated = ndimage.binary_dilation(merged, structure=kernel,
                                             iterations=iterations).astype(np.uint8)

            # Save
            import os
            os.makedirs(output_dir, exist_ok=True)
            mask_path = os.path.join(output_dir, "final_master_mask.fits")
            FITSHandler.write_fits(dilated, mask_path, header)

            # Visualization
            viz_path = None
            if image_fits_path:
                image, _, _ = FITSHandler.read_fits(image_fits_path)
                viz_path = os.path.join(output_dir, "step3_dilation.png")
                Visualization.create_dilation_comparison(
                    merged, dilated, image, viz_path
                )

            # Estimate coverage change
            orig_coverage = np.sum(merged) / merged.size * 100
            new_coverage = np.sum(dilated) / dilated.size * 100

            return ToolResult(
                success=True,
                message=(f"Merged and dilated masks. "
                        f"Coverage: {orig_coverage:.1f}% → {new_coverage:.1f}%. "
                        f"Verify dilation hasn't swallowed distant background sources."),
                output_path=mask_path,
                visualization_path=viz_path,
                metrics={
                    'dilation_factor': dilation_factor,
                    'kernel_shape': kernel_shape,
                    'original_coverage': orig_coverage,
                    'dilated_coverage': new_coverage,
                },
                requires_human_input=True,
                state_update={
                    'step': 3,
                    'final_mask_path': mask_path,
                    'dilation_factor': dilation_factor,
                }
            )

        except Exception as e:
            logger.error(f"Error in ToolMergeAndDilateMask: {str(e)}")
            return ToolResult(
                success=False,
                message=f"Failed to merge and dilate mask: {str(e)}",
                requires_human_input=False
            )


class ToolInterpolateAndSubtract:
    """
    Step 4: Interpolate ICL model and subtract from image.

    Human Decision Point: Verify physical fidelity - flux conservation and shape preservation.
    """

    @staticmethod
    def execute(
        image_fits_path: str,
        final_master_mask_path: str,
        method: str = "rbf",
        output_dir: str = "./outputs",
        background_sources: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> ToolResult:
        """
        Interpolate ICL and subtract from image.

        Args:
            image_fits_path: Path to original image
            final_master_mask_path: Path to final merged mask
            method: Interpolation method ("rbf", "clough-tocher", "linear", "nearest")
            output_dir: Output directory
            background_sources: Optional list of background source regions for shape metrics

        Returns:
            ToolResult with ICL model and residual image
        """
        try:
            # Read data
            image, wcs, header = FITSHandler.read_fits(image_fits_path)
            mask, _, _ = FITSHandler.read_fits(final_master_mask_path)
            mask = (mask > 0).astype(bool)

            logger.info(f"Running ICL interpolation with method: {method}")

            # Prepare data for interpolation
            # Good pixels = not masked
            good_pixels = ~mask

            # Subsample for performance (use every Nth pixel)
            subsample = 4
            y_coords, x_coords = np.indices(image.shape)

            y_good = y_coords[good_pixels][::subsample]
            x_good = x_coords[good_pixels][::subsample]
            values_good = image[good_pixels][::subsample]

            # Remove any NaN/Inf from training data
            valid = np.isfinite(values_good)
            y_good = y_good[valid]
            x_good = x_good[valid]
            values_good = values_good[valid]

            logger.info(f"Using {len(values_good)} points for interpolation")

            # Build interpolator
            points = np.column_stack((x_good, y_good))

            if method == "rbf":
                # Radial Basis Function interpolation
                interpolator = RBFInterpolator(
                    points, values_good,
                    kernel='thin_plate_spline',
                    smoothing=0.1
                )
            elif method == "clough-tocher":
                # Triangular mesh interpolation
                interpolator = CloughTocher2DInterpolator(
                    points, values_good,
                    fill_value=0
                )
            elif method == "linear":
                interpolator = LinearNDInterpolator(
                    points, values_good,
                    fill_value=0
                )
            else:  # nearest
                from scipy.interpolate import NearestNDInterpolator
                interpolator = NearestNDInterpolator(points, values_good)

            # Evaluate on full grid
            logger.info("Evaluating interpolator on full grid...")
            full_points = np.column_stack((x_coords.ravel(), y_coords.ravel()))

            # Process in chunks to avoid memory issues
            chunk_size = 100000
            icl_model = np.zeros_like(image)

            for i in range(0, len(full_points), chunk_size):
                chunk = full_points[i:i+chunk_size]
                icl_model.ravel()[i:i+chunk_size] = interpolator(chunk)

            # Subtract
            residual = image - icl_model

            # Save outputs
            import os
            os.makedirs(output_dir, exist_ok=True)

            icl_path = os.path.join(output_dir, "icl_model.fits")
            residual_path = os.path.join(output_dir, "clean_science_residual.fits")

            FITSHandler.write_fits(icl_model, icl_path, header)
            FITSHandler.write_fits(residual, residual_path, header)

            # Calculate quality metrics
            logger.info("Calculating quality metrics...")
            flux_metrics = Metrics.calculate_flux_metrics(image, residual, mask)
            ellip_metrics = Metrics.calculate_ellipticity_metrics(image, residual, background_sources)
            all_metrics = Metrics.calculate_overall_quality(flux_metrics, ellip_metrics)

            # Generate comprehensive visualization
            viz_path = os.path.join(output_dir, "step4_residual_analysis.png")
            Visualization.create_residual_analysis(
                image, residual, icl_model, all_metrics, viz_path
            )

            # Determine if passed
            passed = all_metrics['passed']

            message = (
                f"ICL subtraction complete.\n"
                f"Overall Quality Score: {all_metrics['overall_score']:.1f}/100\n"
                f"Flux Conservation: {all_metrics['flux_conservation_score']:.1f}/100\n"
                f"Shape Preservation: {all_metrics['shape_preservation_score']:.1f}/100\n"
            )

            if passed:
                message += "✓ Quality checks PASSED. Ready for science analysis."
            else:
                message += "✗ Quality checks FAILED. Review residual images and consider adjusting parameters."

            return ToolResult(
                success=True,
                message=message,
                output_path=residual_path,
                visualization_path=viz_path,
                metrics=all_metrics,
                requires_human_input=True,  # Always require human approval for final step
                state_update={
                    'step': 4,
                    'icl_model_path': icl_path,
                    'residual_path': residual_path,
                    'interpolation_method': method,
                    'quality_metrics': all_metrics,
                }
            )

        except Exception as e:
            logger.error(f"Error in ToolInterpolateAndSubtract: {str(e)}")
            return ToolResult(
                success=False,
                message=f"Failed to interpolate and subtract: {str(e)}",
                requires_human_input=False
            )


class ToolEvaluateFieldComplexity:
    """
    Pre-step: Evaluate field complexity to determine workflow mode.

    Returns recommendation for automatic vs manual processing.
    """

    @staticmethod
    def execute(
        image_fits_path: str,
        detect_thresh: float = 2.0,
        bright_star_thresh: float = 1000.0,
        output_dir: str = "./outputs",
    ) -> ToolResult:
        """
        Evaluate field complexity for confidence routing.

        Args:
            image_fits_path: Path to input FITS
            detect_thresh: Threshold for source detection
            bright_star_thresh: Flux threshold for bright star classification
            output_dir: Output directory

        Returns:
            ToolResult with complexity assessment and recommendations
        """
        try:
            data, wcs, header = FITSHandler.read_fits(image_fits_path)
            data_clean = np.copy(data)
            data_clean[~np.isfinite(data_clean)] = 0

            # SEP extraction
            bkg = sep.Background(data_clean)
            data_sub = data_clean - bkg
            objects = sep.extract(data_sub, detect_thresh, err=bkg.globalrms)

            # Count bright vs faint stars
            bright_stars = np.sum(objects['flux'] > bright_star_thresh)
            faint_stars = len(objects) - bright_stars

            # Estimate lens galaxy surface brightness (brightest central source)
            if len(objects) > 0:
                brightest_idx = np.argmax(objects['flux'])
                lens_sb = objects[brightest_idx]['flux'] / (np.pi * objects[brightest_idx]['a'] * objects[brightest_idx]['b'])
            else:
                lens_sb = 0

            # Background complexity: number density of sources
            area_pixels = data.shape[0] * data.shape[1]
            source_density = len(objects) / area_pixels * 1e6  # per Mpixel

            # Calculate complexity score
            complexity_score = (
                bright_stars * 10 +
                source_density * 0.1 +
                np.log10(lens_sb + 1) * 5
            )

            # Determine recommendation
            if complexity_score < 20 and bright_stars == 0:
                recommended_auto = True
                message = (
                    "Field complexity: LOW\n"
                    f"  Bright stars: {bright_stars}, Faint stars: {faint_stars}\n"
                    f"  Source density: {source_density:.1f}/Mpix\n"
                    "Recommendation: AUTO mode - can process with default parameters."
                )
            elif complexity_score < 50:
                recommended_auto = True
                message = (
                    "Field complexity: MEDIUM\n"
                    f"  Bright stars: {bright_stars}, Faint stars: {faint_stars}\n"
                    f"  Source density: {source_density:.1f}/Mpix\n"
                    "Recommendation: SEMI-AUTO mode - review after Step 4."
                )
            else:
                recommended_auto = False
                message = (
                    "Field complexity: HIGH\n"
                    f"  Bright stars: {bright_stars}, Faint stars: {faint_stars}\n"
                    f"  Source density: {source_density:.1f}/Mpix\n"
                    "Recommendation: MANUAL mode - human input required at each step."
                )

            complexity = FieldComplexity(
                num_bright_stars=int(bright_stars),
                num_faint_stars=int(faint_stars),
                lens_galaxy_surface_brightness=float(lens_sb),
                background_complexity=float(source_density),
                recommended_auto_mode=recommended_auto,
                complexity_score=float(complexity_score)
            )

            return ToolResult(
                success=True,
                message=message,
                metrics={
                    'num_bright_stars': bright_stars,
                    'num_faint_stars': faint_stars,
                    'lens_surface_brightness': float(lens_sb),
                    'source_density': float(source_density),
                    'complexity_score': float(complexity_score),
                    'recommended_auto_mode': recommended_auto,
                },
                requires_human_input=not recommended_auto,
                state_update={
                    'field_complexity': complexity,
                }
            )

        except Exception as e:
            logger.error(f"Error in ToolEvaluateFieldComplexity: {str(e)}")
            return ToolResult(
                success=False,
                message=f"Failed to evaluate field complexity: {str(e)}"
            )
