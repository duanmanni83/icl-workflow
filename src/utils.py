"""Utility functions for FITS handling, visualization, and metrics."""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LogNorm, SymLogNorm
from scipy import ndimage
from skimage import measure
from typing import Optional, Tuple, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ICLWorkflow")


class FITSHandler:
    """Handle FITS file I/O and basic operations."""

    @staticmethod
    def read_fits(fits_path: str, ext: int = 0) -> Tuple[np.ndarray, Optional[WCS], fits.Header]:
        """Read FITS file and return data, WCS, and header."""
        with fits.open(fits_path) as hdul:
            data = hdul[ext].data.astype(np.float64)
            header = hdul[ext].header
            try:
                wcs = WCS(header)
            except Exception:
                wcs = None
        return data, wcs, header

    @staticmethod
    def write_fits(data: np.ndarray, output_path: str, header: Optional[fits.Header] = None) -> None:
        """Write data to FITS file."""
        hdu = fits.PrimaryHDU(data, header=header)
        hdu.writeto(output_path, overwrite=True)
        logger.info(f"Written FITS: {output_path}")

    @staticmethod
    def get_cutout(data: np.ndarray, center_x: float, center_y: float,
                   size: float, wcs: Optional[WCS] = None) -> Tuple[np.ndarray, slice, slice]:
        """Extract a cutout region from the image."""
        x_min = int(max(0, center_x - size / 2))
        x_max = int(min(data.shape[1], center_x + size / 2))
        y_min = int(max(0, center_y - size / 2))
        y_max = int(min(data.shape[0], center_y + size / 2))

        cutout = data[y_min:y_max, x_min:x_max]
        return cutout, slice(y_min, y_max), slice(x_min, x_max)


class Visualization:
    """Generate visualization for human-in-the-loop decisions."""

    @staticmethod
    def create_mask_overlay(image: np.ndarray, mask: np.ndarray,
                           center_x: Optional[float] = None,
                           center_y: Optional[float] = None,
                           size: float = 100,
                           title: str = "Mask Overlay",
                           output_path: Optional[str] = None) -> str:
        """Create pseudo-color image with mask contour overlay."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Determine cutout region
        if center_x is None:
            center_x = image.shape[1] / 2
        if center_y is None:
            center_y = image.shape[0] / 2

        y_slice = slice(int(max(0, center_y - size/2)), int(min(image.shape[0], center_y + size/2)))
        x_slice = slice(int(max(0, center_x - size/2)), int(min(image.shape[1], center_x + size/2)))

        img_cutout = image[y_slice, x_slice]
        mask_cutout = mask[y_slice, x_slice]

        # Log-stretched image
        ax1 = axes[0]
        vmin, vmax = np.percentile(img_cutout[~np.isnan(img_cutout)], [1, 99])
        im1 = ax1.imshow(img_cutout, norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), cmap='gray', origin='lower')
        ax1.set_title("Original Image (Log Scale)")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # Image with mask contours
        ax2 = axes[1]
        ax2.imshow(img_cutout, norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), cmap='gray', origin='lower')
        ax2.contour(mask_cutout, levels=[0.5], colors='red', linewidths=2)
        ax2.set_title("With Mask Contours")

        # Mask only
        ax3 = axes[2]
        im3 = ax3.imshow(mask_cutout, cmap='Reds', alpha=0.7, origin='lower')
        ax3.set_title("Mask")
        plt.colorbar(im3, ax=ax3, fraction=0.046)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path is None:
            output_path = "mask_overlay.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    @staticmethod
    def create_spike_visualization(image: np.ndarray, spike_mask: np.ndarray,
                                   center_x: float, center_y: float,
                                   spike_params: Dict[str, Any],
                                   title: str = "Spike Mask Verification",
                                   output_path: Optional[str] = None) -> str:
        """Create visualization for spike mask alignment check."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Full image with spike overlay
        ax1 = axes[0]
        vmin, vmax = np.percentile(image[~np.isnan(image)], [1, 99])
        ax1.imshow(image, norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), cmap='gray', origin='lower')

        # Draw spike center and angles
        ax1.plot(center_x, center_y, 'r+', markersize=20, markeredgewidth=2)

        num_spikes = spike_params.get('num_spikes', 4)
        rotation = spike_params.get('rotation_angle', 0)
        spike_length = spike_params.get('spike_length', 100)

        for i in range(num_spikes):
            angle = rotation + i * (180 / num_spikes)
            rad = np.deg2rad(angle)
            x_end = center_x + spike_length * np.cos(rad)
            y_end = center_y + spike_length * np.sin(rad)
            ax1.plot([center_x, x_end], [center_y, y_end], 'r--', alpha=0.7, linewidth=1)

        ax1.set_title("Full Image with Spike Geometry")
        ax1.set_xlim(0, image.shape[1])
        ax1.set_ylim(0, image.shape[0])

        # Zoomed view
        ax2 = axes[1]
        size = 150
        y_slice = slice(int(max(0, center_y - size/2)), int(min(image.shape[0], center_y + size/2)))
        x_slice = slice(int(max(0, center_x - size/2)), int(min(image.shape[1], center_x + size/2)))

        img_cutout = image[y_slice, x_slice]
        mask_cutout = spike_mask[y_slice, x_slice]

        ax2.imshow(img_cutout, norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), cmap='gray', origin='lower')
        ax2.contour(mask_cutout, levels=[0.5], colors='lime', linewidths=1.5, alpha=0.8)

        # Draw spike boundaries
        for i in range(num_spikes):
            angle = rotation + i * (180 / num_spikes)
            rad = np.deg2rad(angle)
            width = spike_params.get('spike_width', 3)

            # Draw width boundaries
            for offset in [-width/2, width/2]:
                x_end = size/2 + (spike_length - size/2) * np.cos(rad) + offset * np.sin(rad)
                y_end = size/2 + (spike_length - size/2) * np.sin(rad) - offset * np.cos(rad)
                ax2.plot([size/2, x_end], [size/2, y_end], 'y--', alpha=0.5, linewidth=0.8)

        ax2.set_title("Zoomed View: Mask Alignment Check")

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path is None:
            output_path = "spike_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    @staticmethod
    def create_dilation_comparison(original_mask: np.ndarray, dilated_mask: np.ndarray,
                                   image: np.ndarray,
                                   output_path: Optional[str] = None) -> str:
        """Create before/after comparison for dilation step."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        center_y, center_x = np.array(original_mask.shape) // 2
        size = 200
        y_slice = slice(int(max(0, center_y - size/2)), int(min(original_mask.shape[0], center_y + size/2)))
        x_slice = slice(int(max(0, center_x - size/2)), int(min(original_mask.shape[1], center_x + size/2)))

        img_cutout = image[y_slice, x_slice] if image is not None else None
        orig_cutout = original_mask[y_slice, x_slice]
        dil_cutout = dilated_mask[y_slice, x_slice]

        # Original mask
        ax1 = axes[0]
        if img_cutout is not None:
            vmin, vmax = np.percentile(img_cutout[~np.isnan(img_cutout)], [1, 99])
            ax1.imshow(img_cutout, norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), cmap='gray', origin='lower')
        ax1.contour(orig_cutout, levels=[0.5], colors='red', linewidths=2)
        ax1.set_title("Original Mask Boundary")

        # Dilated mask
        ax2 = axes[1]
        if img_cutout is not None:
            ax2.imshow(img_cutout, norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), cmap='gray', origin='lower')
        ax2.contour(dil_cutout, levels=[0.5], colors='blue', linewidths=2)
        ax2.set_title("Dilated Mask Boundary")

        # Comparison overlay
        ax3 = axes[2]
        if img_cutout is not None:
            ax3.imshow(img_cutout, norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), cmap='gray', origin='lower', alpha=0.6)
        ax3.contour(orig_cutout, levels=[0.5], colors='red', linewidths=2, label='Original')
        ax3.contour(dil_cutout, levels=[0.5], colors='blue', linewidths=2, linestyle='--', label='Dilated')
        ax3.set_title("Comparison (Red=Original, Blue=Dilated)")
        ax3.legend()

        plt.suptitle("Mask Dilation Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path is None:
            output_path = "dilation_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    @staticmethod
    def create_residual_analysis(original: np.ndarray, residual: np.ndarray,
                                 icl_model: np.ndarray,
                                 metrics: Dict[str, float],
                                 output_path: Optional[str] = None) -> str:
        """Create comprehensive residual analysis visualization."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Row 1: Original, ICL Model, Residual
        ax1 = fig.add_subplot(gs[0, 0])
        vmin, vmax = np.percentile(original[~np.isnan(original)], [1, 99])
        ax1.imshow(original, norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), cmap='gray', origin='lower')
        ax1.set_title("Original Image")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(icl_model, norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), cmap='inferno', origin='lower')
        ax2.set_title("ICL Model")

        ax3 = fig.add_subplot(gs[0, 2])
        res_vmax = np.percentile(np.abs(residual[~np.isnan(residual)]), 95)
        ax3.imshow(residual, vmin=-res_vmax, vmax=res_vmax, cmap='RdBu_r', origin='lower')
        ax3.set_title("Residual (Cleaned)")

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        metric_text = f"""
Quality Metrics:

Flux Conservation:
  Negative Pixel Ratio: {metrics.get('negative_pixel_ratio', 0):.4f}
  Mean Residual: {metrics.get('mean_residual', 0):.4f}
  Std Residual: {metrics.get('std_residual', 0):.4f}
  Flux Score: {metrics.get('flux_conservation_score', 0):.2f}

Shape Preservation:
  Ellipticity RMSE: {metrics.get('ellipticity_rmse', 0):.4f}
  e1 Bias: {metrics.get('ellipticity_bias_x', 0):.4f}
  e2 Bias: {metrics.get('ellipticity_bias_y', 0):.4f}
  Shape Score: {metrics.get('shape_preservation_score', 0):.2f}

Overall Score: {metrics.get('overall_score', 0):.2f}/100
Passed: {'✓' if metrics.get('passed', False) else '✗'}
"""
        ax4.text(0.1, 0.5, metric_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Row 2: Histograms
        ax5 = fig.add_subplot(gs[1, :2])
        res_flat = residual[~np.isnan(residual)].flatten()
        ax5.hist(res_flat, bins=100, range=(-res_vmax, res_vmax), alpha=0.7, color='steelblue', edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel("Residual Value")
        ax5.set_ylabel("Pixel Count")
        ax5.set_title("Residual Pixel Distribution")

        ax6 = fig.add_subplot(gs[1, 2:])
        # Radial profile of residual
        center = np.array(residual.shape) // 2
        y, x = np.indices(residual.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r_flat = r.flatten()
        res_vals = residual.flatten()

        # Binned radial profile
        bins = np.linspace(0, min(residual.shape) / 2, 50)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        radial_mean = []
        for i in range(len(bins) - 1):
            mask = (r_flat >= bins[i]) & (r_flat < bins[i+1])
            if np.any(mask):
                radial_mean.append(np.nanmean(res_vals[mask]))
            else:
                radial_mean.append(0)

        ax6.plot(bin_centers, radial_mean, 'b-', linewidth=2)
        ax6.axhline(0, color='red', linestyle='--', linewidth=1)
        ax6.set_xlabel("Radius (pixels)")
        ax6.set_ylabel("Mean Residual")
        ax6.set_title("Radial Profile of Residual")

        # Row 3: Zoomed regions
        # Show central region where lens is
        center_y, center_x = np.array(residual.shape) // 2
        size = 100
        y_slice = slice(int(center_y - size/2), int(center_y + size/2))
        x_slice = slice(int(center_x - size/2), int(center_x + size/2))

        ax7 = fig.add_subplot(gs[2, 0])
        ax7.imshow(original[y_slice, x_slice], norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), cmap='gray', origin='lower')
        ax7.set_title("Original: Center Region")

        ax8 = fig.add_subplot(gs[2, 1])
        ax8.imshow(residual[y_slice, x_slice], vmin=-res_vmax, vmax=res_vmax, cmap='RdBu_r', origin='lower')
        ax8.set_title("Residual: Center Region")

        # Background region check
        bg_y, bg_x = size, size
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.imshow(original[bg_y:bg_y+size, bg_x:bg_x+size], norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), cmap='gray', origin='lower')
        ax9.set_title("Original: Background Region")

        ax10 = fig.add_subplot(gs[2, 3])
        ax10.imshow(residual[bg_y:bg_y+size, bg_x:bg_x+size], vmin=-res_vmax, vmax=res_vmax, cmap='RdBu_r', origin='lower')
        ax10.set_title("Residual: Background Region")

        plt.suptitle("ICL Subtraction Quality Analysis", fontsize=16, fontweight='bold')

        if output_path is None:
            output_path = "residual_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path


class Metrics:
    """Calculate quality metrics for ICL subtraction."""

    @staticmethod
    def calculate_flux_metrics(original: np.ndarray, residual: np.ndarray,
                               mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate flux conservation metrics."""
        if mask is not None:
            res_roi = residual[mask > 0]
        else:
            # Focus on central region
            cy, cx = np.array(residual.shape) // 2
            y_slice = slice(int(cy - 50), int(cy + 50))
            x_slice = slice(int(cx - 50), int(cx + 50))
            res_roi = residual[y_slice, x_slice]

        res_roi = res_roi[~np.isnan(res_roi)]

        negative_pixels = np.sum(res_roi < 0)
        total_pixels = len(res_roi)
        negative_pixel_ratio = negative_pixels / total_pixels if total_pixels > 0 else 0

        mean_residual = np.mean(res_roi)
        std_residual = np.std(res_roi)

        # Flux conservation score: penalize negative over-subtraction
        # Ideal: mean ~ 0, low std, low negative ratio
        flux_score = max(0, 100 - abs(mean_residual) * 100 - std_residual * 50 - negative_pixel_ratio * 100)

        return {
            'negative_pixel_ratio': negative_pixel_ratio,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'flux_conservation_score': flux_score,
        }

    @staticmethod
    def calculate_ellipticity_metrics(original: np.ndarray, residual: np.ndarray,
                                      background_sources: Optional[List[Tuple[int, int, int, int]]] = None) -> Dict[str, float]:
        """Calculate ellipticity preservation metrics using background sources."""
        # If no sources provided, detect some in background
        if background_sources is None:
            # Simple detection on residual to find sources
            from scipy.ndimage import label, center_of_mass
            labeled, num_features = label(residual > np.percentile(residual[~np.isnan(residual)], 95))
            background_sources = []
            for i in range(1, min(num_features + 1, 10)):
                coords = np.argwhere(labeled == i)
                if len(coords) > 0:
                    y_mean, x_mean = coords.mean(axis=0)
                    background_sources.append((int(x_mean) - 10, int(y_mean) - 10, 20, 20))

        ellipticities_orig = []
        ellipticities_res = []

        for (x, y, w, h) in background_sources:
            if x < 0 or y < 0 or x + w >= original.shape[1] or y + h >= original.shape[0]:
                continue

            cutout_orig = original[y:y+h, x:x+w]
            cutout_res = residual[y:y+h, x:x+w]

            if np.any(~np.isfinite(cutout_orig)) or np.any(~np.isfinite(cutout_res)):
                continue

            # Calculate second moments
            e1_orig, e2_orig = Metrics._calculate_ellipticity(cutout_orig)
            e1_res, e2_res = Metrics._calculate_ellipticity(cutout_res)

            if e1_orig is not None:
                ellipticities_orig.append((e1_orig, e2_orig))
                ellipticities_res.append((e1_res, e2_res))

        if len(ellipticities_orig) > 0:
            ellipticities_orig = np.array(ellipticities_orig)
            ellipticities_res = np.array(ellipticities_res)

            rmse = np.sqrt(np.mean((ellipticities_orig - ellipticities_res)**2))
            bias_x = np.mean(ellipticities_res[:, 0] - ellipticities_orig[:, 0])
            bias_y = np.mean(ellipticities_res[:, 1] - ellipticities_orig[:, 1])

            # Shape preservation score
            shape_score = max(0, 100 - rmse * 1000)
        else:
            rmse = 0.0
            bias_x = 0.0
            bias_y = 0.0
            shape_score = 50.0  # Neutral if no sources

        return {
            'ellipticity_rmse': rmse,
            'ellipticity_bias_x': bias_x,
            'ellipticity_bias_y': bias_y,
            'shape_preservation_score': shape_score,
        }

    @staticmethod
    def _calculate_ellipticity(image: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Calculate ellipticity (e1, e2) from image moments."""
        if np.any(~np.isfinite(image)):
            return None, None

        # Normalize
        img = image - np.min(image)
        total_flux = np.sum(img)

        if total_flux <= 0:
            return None, None

        y, x = np.indices(img.shape)

        # First moments (centroid)
        x_c = np.sum(x * img) / total_flux
        y_c = np.sum(y * img) / total_flux

        # Second moments
        x_shifted = x - x_c
        y_shifted = y - y_c

        Qxx = np.sum(x_shifted**2 * img) / total_flux
        Qyy = np.sum(y_shifted**2 * img) / total_flux
        Qxy = np.sum(x_shifted * y_shifted * img) / total_flux

        # Ellipticity components
        denominator = Qxx + Qyy + 2 * np.sqrt(Qxx * Qyy - Qxy**2)
        if denominator == 0:
            return None, None

        e1 = (Qxx - Qyy) / denominator
        e2 = (2 * Qxy) / denominator

        return e1, e2

    @staticmethod
    def calculate_overall_quality(flux_metrics: Dict[str, float],
                                  ellipticity_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Combine all metrics into overall quality assessment."""
        overall_score = (flux_metrics['flux_conservation_score'] +
                        ellipticity_metrics['shape_preservation_score']) / 2

        # Pass criteria
        passed = (
            overall_score >= 70 and
            flux_metrics['negative_pixel_ratio'] < 0.3 and
            ellipticity_metrics['ellipticity_rmse'] < 0.1
        )

        return {
            'overall_score': overall_score,
            'passed': passed,
            **flux_metrics,
            **ellipticity_metrics,
        }