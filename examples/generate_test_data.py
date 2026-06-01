#!/usr/bin/env python3
"""Generate synthetic test data for ICL workflow demonstration."""

import numpy as np
from astropy.io import fits
from pathlib import Path
import argparse


def create_elliptical_gaussian(shape, center, sigma_x, sigma_y, angle, amplitude=1.0):
    """Create an elliptical Gaussian."""
    y, x = np.indices(shape)
    angle_rad = np.deg2rad(angle)

    # Rotate coordinates
    x_rot = (x - center[1]) * np.cos(angle_rad) + (y - center[0]) * np.sin(angle_rad)
    y_rot = -(x - center[1]) * np.sin(angle_rad) + (y - center[0]) * np.cos(angle_rad)

    gaussian = amplitude * np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
    return gaussian


def generate_synthetic_image(
    size=512,
    add_lensing_arc=True,
    add_diffraction_spikes=True,
    add_background_sources=True,
    noise_level=0.1,
    output_path="test_image.fits"
):
    """
    Generate a synthetic astronomical image with ICL-like features.

    Components:
    - Smooth ICL background (gradient + polynomial)
    - Central lens galaxy
    - Gravitational arc (to be preserved)
    - Diffraction spikes
    - Background sources
    - Noise
    """
    print(f"Generating synthetic image ({size}x{size})...")

    # Initialize image
    image = np.zeros((size, size), dtype=np.float64)

    # Center
    cy, cx = size // 2, size // 2

    # 1. ICL Background (smooth gradient + polynomial)
    print("  Adding ICL background...")
    y, x = np.indices((size, size))

    # Large-scale gradient
    gradient = 0.001 * (x + y) + 0.0001 * (x * y / size)

    # Polynomial background (simulating ICL)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    icl = 5.0 * np.exp(-r / 100) + 0.5 * np.exp(-r / 200)

    # Add some structure to ICL (asymmetry)
    icl += 0.5 * np.sin(x / 30) * np.cos(y / 30) * np.exp(-r / 150)

    image += gradient + icl

    # 2. Central Lens Galaxy
    print("  Adding central lens galaxy...")
    lens_galaxy = create_elliptical_gaussian(
        (size, size),
        center=(cy, cx),
        sigma_x=15,
        sigma_y=12,
        angle=30,
        amplitude=50.0
    )
    image += lens_galaxy

    # 3. Gravitational Arc (elongated feature to be preserved)
    if add_lensing_arc:
        print("  Adding gravitational arc...")
        # Arc on one side of the lens
        arc_angle = 45  # degrees
        arc_distance = 35

        arc_centers = [
            (cy + arc_distance * np.sin(np.deg2rad(arc_angle)),
             cx + arc_distance * np.cos(np.deg2rad(arc_angle))),
            (cy + arc_distance * np.sin(np.deg2rad(arc_angle + 180)),
             cx + arc_distance * np.cos(np.deg2rad(arc_angle + 180)))
        ]

        for arc_cy, arc_cx in arc_centers:
            arc = create_elliptical_gaussian(
                (size, size),
                center=(arc_cy, arc_cx),
                sigma_x=20,
                sigma_y=3,
                angle=arc_angle,
                amplitude=8.0
            )
            image += arc

    # 4. Diffraction Spikes (4 for CSST-like)
    if add_diffraction_spikes:
        print("  Adding diffraction spikes...")
        spike_length = 150
        spike_width = 2

        for angle in [0, 45, 90, 135]:
            angle_rad = np.deg2rad(angle)

            for dist in range(15, spike_length):
                for width_offset in range(-spike_width, spike_width + 1):
                    sx = int(cx + dist * np.cos(angle_rad) + width_offset * np.sin(angle_rad))
                    sy = int(cy + dist * np.sin(angle_rad) - width_offset * np.cos(angle_rad))

                    if 0 <= sx < size and 0 <= sy < size:
                        # Intensity falls off with distance
                        intensity = 3.0 * np.exp(-dist / 50)
                        image[sy, sx] += intensity

    # 5. Background Sources
    if add_background_sources:
        print("  Adding background sources...")
        np.random.seed(42)
        n_sources = 30

        for _ in range(n_sources):
            # Random position (avoiding center)
            while True:
                by = np.random.randint(50, size - 50)
                bx = np.random.randint(50, size - 50)
                if np.sqrt((by - cy)**2 + (bx - cx)**2) > 80:
                    break

            source = create_elliptical_gaussian(
                (size, size),
                center=(by, bx),
                sigma_x=np.random.uniform(2, 5),
                sigma_y=np.random.uniform(2, 5),
                angle=np.random.uniform(0, 180),
                amplitude=np.random.uniform(1, 5)
            )
            image += source

    # 6. Add Noise
    print("  Adding noise...")
    noise = np.random.normal(0, noise_level, (size, size))
    image += noise

    # Ensure positive values
    image = np.maximum(image, 0.001)

    # Add WCS header
    header = fits.Header()
    header['SIMPLE'] = True
    header['BITPIX'] = -64
    header['NAXIS'] = 2
    header['NAXIS1'] = size
    header['NAXIS2'] = size

    # Basic WCS
    header['CRPIX1'] = cx
    header['CRPIX2'] = cy
    header['CRVAL1'] = 150.0
    header['CRVAL2'] = 2.0
    header['CDELT1'] = -0.0001  # ~0.36 arcsec/pixel
    header['CDELT2'] = 0.0001
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'

    # Instrument info
    header['TELESCOP'] = 'CSST'
    header['INSTRUME'] = 'ICL-TEST'
    header['FILTER'] = 'F150W'
    header['EXPTIME'] = 1000.0

    # Save
    hdu = fits.PrimaryHDU(image, header=header)
    hdu.writeto(output_path, overwrite=True)
    print(f"Saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate test FITS image")
    parser.add_argument("--size", "-s", type=int, default=512, help="Image size")
    parser.add_argument("--output", "-o", default="test_image.fits", help="Output path")
    parser.add_argument("--no-arc", action="store_true", help="Don't add gravitational arc")
    parser.add_argument("--no-spikes", action="store_true", help="Don't add diffraction spikes")
    parser.add_argument("--noise", "-n", type=float, default=0.1, help="Noise level")

    args = parser.parse_args()

    generate_synthetic_image(
        size=args.size,
        add_lensing_arc=not args.no_arc,
        add_diffraction_spikes=not args.no_spikes,
        noise_level=args.noise,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
