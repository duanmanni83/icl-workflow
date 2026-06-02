#!/usr/bin/env python3
"""
Generate realistic synthetic galaxy cluster data for ICL workflow testing.

This simulates a MACS0416-like strong lensing cluster with:
- Realistic ICL profile (Sersic + power law)
- Multiple bright cluster galaxies (BCGs)
- Strong lensing arcs
- Realistic diffraction spikes
- Background galaxy population
- Realistic noise and PSF
"""

import numpy as np
from astropy.io import fits
from astropy.modeling.models import Sersic2D
import argparse


def create_realistic_psf(size, fwhm_pixels=3.0):
    """Create a realistic PSF (Moffat profile)."""
    from astropy.modeling.models import Moffat2D
    y, x = np.indices((size, size))
    center = size // 2
    moffat = Moffat2D(amplitude=1.0, x_0=center, y_0=center,
                     gamma=fwhm_pixels/2, alpha=4.7)
    psf = moffat(x, y)
    return psf / psf.sum()


def create_icl_model(shape, center, r_eff=100, n=0.5, ellip=0.3):
    """
    Create ICL model with Sersic profile + power law.

    Real ICL typically has:
    - Low Sersic index (n ~ 0.5-1.5)
    - Large effective radius (50-200 kpc)
    - Elliptical shape
    """
    y, x = np.indices(shape)

    # Rotate coordinates for ellipticity
    angle = np.deg2rad(45)
    x_rot = (x - center[1]) * np.cos(angle) + (y - center[0]) * np.sin(angle)
    y_rot = -(x - center[1]) * np.sin(angle) + (y - center[0]) * np.cos(angle)

    # Sersic profile (low n for flat ICL)
    b_n = 1.999 * n - 0.327  # Approximation for Sersic b parameter
    r = np.sqrt(x_rot**2 + (y_rot / (1 - ellip))**2)

    # Combine Sersic + power law for extended wings
    sersic = np.exp(-b_n * ((r / r_eff)**(1/n) - 1))
    power_law = (r / r_eff + 1)**(-1.5)

    # Weighted combination
    icl = 0.7 * sersic + 0.3 * power_law

    return icl


def create_bcg(shape, center, mag_diff=0):
    """Create Brightest Cluster Galaxy."""
    y, x = np.indices(shape)

    # Main BCG: Large elliptical galaxy
    sersic = Sersic2D(
        amplitude=100.0 * 10**(-0.4 * mag_diff),
        x_0=center[1], y_0=center[0],
        r_eff=15, n=4.0,
        ellip=0.4, theta=np.deg2rad(30)
    )

    return sersic(x, y)


def create_lensing_arc(shape, center, distance=50, angle=45,
                       length=40, width=5, curvature=0.3):
    """
    Create realistic strong lensing arc.

    Arcs are typically:
    - Tangentially aligned around the cluster center
    - Curved (radius of curvature related to lens mass)
    - Often in pairs (counter-images)
    """
    y, x = np.indices(shape)
    arc = np.zeros(shape)

    angle_rad = np.deg2rad(angle)

    # Arc center position
    arc_x = center[1] + distance * np.cos(angle_rad)
    arc_y = center[0] + distance * np.sin(angle_rad)

    # Create curved arc
    for t in np.linspace(-length/2, length/2, 100):
        # Curved path
        curve_offset = curvature * t**2 / length

        pos_x = arc_x + t * np.cos(angle_rad) - curve_offset * np.sin(angle_rad)
        pos_y = arc_y + t * np.sin(angle_rad) + curve_offset * np.cos(angle_rad)

        # Gaussian profile perpendicular to arc
        dx = x - pos_x
        dy = y - pos_y

        # Distance perpendicular to arc direction
        perp_dist = dx * np.sin(angle_rad) - dy * np.cos(angle_rad)
        parallel_dist = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)

        # Only add flux near the arc path
        mask = (np.abs(perp_dist) < width * 2) & (np.abs(parallel_dist) < length)
        arc += 15.0 * np.exp(-(perp_dist**2) / (2 * width**2)) * mask

    return arc


def create_diffraction_spikes_hst(shape, center, n_spikes=4):
    """Create HST-style diffraction spikes (cross pattern)."""
    y, x = np.indices(shape)
    spikes = np.zeros(shape)

    # HST ACS/WFC3 has 4 spikes at 45-degree increments
    for i in range(n_spikes):
        angle = np.deg2rad(22.5 + i * 45)  # Offset from detector axes

        # Vector from center in spike direction
        dx = x - center[1]
        dy = y - center[0]

        # Project onto spike direction
        parallel = dx * np.cos(angle) + dy * np.sin(angle)
        perpendicular = -dx * np.sin(angle) + dy * np.cos(angle)

        # Spike profile: narrow in perpendicular, power law in parallel
        spike_width = 2.0
        spike_length = 200

        in_spike = (parallel > 0) & (parallel < spike_length) & \
                   (np.abs(perpendicular) < spike_width)

        # Intensity falls as power law
        intensity = np.where(in_spike,
                           5.0 * (parallel / 10 + 1)**(-0.8),
                           0)

        spikes += intensity

    return spikes


def create_background_galaxies(shape, center, n_galaxies=50):
    """Create population of background galaxies."""
    galaxies = np.zeros(shape)
    cy, cx = center

    np.random.seed(42)

    for i in range(n_galaxies):
        # Random position (avoid center)
        while True:
            x = np.random.uniform(50, shape[1] - 50)
            y = np.random.uniform(50, shape[0] - 50)
            if np.sqrt((x - cx)**2 + (y - cy)**2) > 100:
                break

        # Random properties
        size = np.random.uniform(3, 8)
        ellip = np.random.uniform(0, 0.6)
        angle = np.random.uniform(0, 180)
        mag = np.random.uniform(1, 10)

        # Create Sersic galaxy
        sersic = Sersic2D(
            amplitude=mag, x_0=x, y_0=y,
            r_eff=size, n=np.random.uniform(0.5, 4.0),
            ellip=ellip, theta=np.deg2rad(angle)
        )

        y_idx, x_idx = np.indices(shape)
        galaxies += sersic(x_idx, y_idx)

    return galaxies


def generate_realistic_cluster(
    size=1024,
    output_path="realistic_cluster.fits",
    cluster_type="MACS0416"
):
    """Generate realistic cluster image."""
    print(f"Generating realistic {cluster_type}-like cluster...")
    print(f"  Image size: {size}x{size}")

    # Initialize image
    image = np.zeros((size, size), dtype=np.float64)
    center = (size // 2, size // 2)

    # 1. ICL (Intra-Cluster Light)
    print("  Adding ICL component...")
    icl = create_icl_model((size, size), center, r_eff=120, n=0.8, ellip=0.25)
    image += icl * 50  # Scale to reasonable flux

    # 2. Main BCG
    print("  Adding BCG...")
    bcg = create_bcg((size, size), center)
    image += bcg

    # 3. Secondary BCG (for MACS0416-like binary BCG)
    if cluster_type == "MACS0416":
        print("  Adding secondary BCG...")
        offset = 30
        bcg2_center = (center[0] + offset, center[1] - offset)
        bcg2 = create_bcg((size, size), bcg2_center, mag_diff=1.5)
        image += bcg2

    # 4. Strong lensing arcs
    print("  Adding strong lensing arcs...")
    arc1 = create_lensing_arc((size, size), center, distance=60,
                              angle=30, length=50, width=6, curvature=0.4)
    arc2 = create_lensing_arc((size, size), center, distance=55,
                              angle=210, length=45, width=5, curvature=0.35)
    image += arc1 + arc2

    # 5. Diffraction spikes (from bright stars in field)
    print("  Adding diffraction spikes...")
    spikes = create_diffraction_spikes_hst((size, size), center, n_spikes=4)
    image += spikes

    # 6. Background galaxies
    print("  Adding background galaxies...")
    bg_gals = create_background_galaxies((size, size), center, n_galaxies=80)
    image += bg_gals

    # 7. Add realistic noise
    print("  Adding noise...")
    # Read noise + Poisson noise
    read_noise = 5.0
    poisson_factor = 0.1

    noise = np.random.normal(0, read_noise, (size, size))
    poisson_noise = np.random.poisson(image * poisson_factor) / poisson_factor - image
    image += noise + poisson_noise

    # Ensure positive
    image = np.maximum(image, 0.01)

    # 8. Apply slight smoothing (PSF effect)
    from scipy.ndimage import gaussian_filter
    image = gaussian_filter(image, sigma=1.5)

    # Create FITS header with realistic WCS
    header = fits.Header()
    header['SIMPLE'] = True
    header['BITPIX'] = -64
    header['NAXIS'] = 2
    header['NAXIS1'] = size
    header['NAXIS2'] = size

    # WCS for MACS0416 region
    header['CRPIX1'] = size / 2
    header['CRPIX2'] = size / 2
    header['CRVAL1'] = 64.0375  # RA of MACS0416
    header['CRVAL2'] = -24.0661  # Dec of MACS0416
    header['CDELT1'] = -1.8056e-05  # 0.065 arcsec/pixel in degrees
    header['CDELT2'] = 1.8056e-05
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'

    # Instrument info (simulating HST WFC3/IR)
    header['TELESCOP'] = 'HST'
    header['INSTRUME'] = 'WFC3/IR'
    header['FILTER'] = 'F105W'
    header['EXPTIME'] = 2000.0
    header['PHOTFLAM'] = 1.5e-20
    header['PHOTZPT'] = -21.1
    header['ORIENTAT'] = 0.0

    # Object info
    header['OBJECT'] = cluster_type
    header['RA_TARG'] = 64.0375
    header['DEC_TARG'] = -24.0661

    print(f"  Writing FITS file: {output_path}")
    hdu = fits.PrimaryHDU(image.astype(np.float64), header=header)
    hdu.writeto(output_path, overwrite=True)

    print(f"\n✓ Created realistic cluster image:")
    print(f"  Size: {size}x{size} pixels")
    print(f"  Pixel scale: 0.065 arcsec/pixel")
    print(f"  Field of view: {size * 0.065:.1f} arcsec")
    print(f"  Flux range: {image.min():.2e} to {image.max():.2e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic galaxy cluster for ICL testing"
    )
    parser.add_argument("--size", "-s", type=int, default=1024,
                       help="Image size in pixels (default: 1024)")
    parser.add_argument("--output", "-o",
                       default="data/clash/realistic_cluster.fits",
                       help="Output path")
    parser.add_argument("--type", "-t", default="MACS0416",
                       choices=["MACS0416", "ABELL2744", "GENERIC"],
                       help="Cluster type to simulate")

    args = parser.parse_args()

    generate_realistic_cluster(
        size=args.size,
        output_path=args.output,
        cluster_type=args.type
    )


if __name__ == "__main__":
    main()
