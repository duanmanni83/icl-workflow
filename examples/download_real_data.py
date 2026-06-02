#!/usr/bin/env python3
"""
Download real Hubble CLASH/Frontier Fields data using astroquery.

This script uses astroquery.mast to download real strong lensing cluster data.
"""

import sys
from pathlib import Path

def download_clash_data(target="MACS0416", output_dir="./data/clash"):
    """Download CLASH data using astroquery."""
    try:
        from astroquery.mast import Observations
        from astropy.io import fits
        import numpy as np
    except ImportError as e:
        print(f"Error: Required package not installed: {e}")
        print("\nPlease install:")
        print("  pip install astroquery astropy")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Downloading CLASH data for {target}")
    print("=" * 70)

    # Query for observations
    print("\nQuerying MAST archive...")

    # Try CLASH first (don't filter by provenance_name)
    obs_table = Observations.query_criteria(
        obs_collection="HST",
        target_name=target
    )

    if len(obs_table) == 0:
        print(f"No HST data found for {target}")
        return None

    print(f"Found {len(obs_table)} observations")

    if len(obs_table) == 0:
        print("No data found. Please check the target name.")
        return None

    # Get data products
    print("Getting data products...")
    data_products = Observations.get_product_list(obs_table[:10])
    print(f"Total products: {len(data_products)}")

    # Filter for drizzled images and preferred filters
    print("Filtering for science images...")

    # Look for DRZ/DRC files (drizzled images)
    mask = (
        (data_products['productSubGroupDescription'] == 'DRZ') |
        (data_products['productSubGroupDescription'] == 'DRC')
    )

    # Prefer specific filters for ICL studies
    preferred_filters = ['F814W', 'F606W', 'F105W', 'F125W', 'F140W', 'F160W']
    for filt in preferred_filters:
        mask = mask | data_products['productFilename'].str.contains(filt, case=False, na=False)

    filtered = data_products[mask]
    print(f"Filtered to {len(filtered)} products")

    if len(filtered) == 0:
        print("No suitable images found.")
        return None

    # Show available products
    print("\nAvailable products:")
    for i, row in enumerate(filtered[:10]):
        size_mb = row['size'] / 1024 / 1024 if row['size'] else 0
        print(f"  {i+1}. {row['productFilename']} ({size_mb:.1f} MB)")

    # Download the first suitable product
    to_download = filtered[:1]
    print(f"\nDownloading: {to_download[0]['productFilename']}")
    print("This may take a few minutes for large files...")

    import time
    start_time = time.time()

    manifest = Observations.download_products(
        to_download,
        download_dir=str(output_path),
        cache=True
    )

    elapsed = time.time() - start_time
    print(f"Download completed in {elapsed:.1f} seconds")

    # Find the downloaded file
    if len(manifest) > 0 and manifest['Status'][0] == 'COMPLETE':
        local_path = manifest['Local Path'][0]
        print(f"\n✓ Saved to: {local_path}")

        # Verify it's a valid FITS file
        try:
            with fits.open(local_path) as hdul:
                print(f"\nFITS file info:")
                hdul.info()

                # Get image dimensions
                for hdu in hdul:
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        print(f"\nImage shape: {hdu.data.shape}")
                        print(f"Data type: {hdu.data.dtype}")
                        if hdu.data.size > 0:
                            print(f"Data range: {np.nanmin(hdu.data):.2e} to {np.nanmax(hdu.data):.2e}")
                        break

                # Show header info
                if 'FILTER' in hdul[0].header:
                    print(f"Filter: {hdul[0].header['FILTER']}")
                if 'EXPTIME' in hdul[0].header:
                    print(f"Exposure time: {hdul[0].header['EXPTIME']}s")
                if 'INSTRUME' in hdul[0].header:
                    print(f"Instrument: {hdul[0].header['INSTRUME']}")

        except Exception as e:
            print(f"Warning: Could not read FITS file: {e}")

        # Create a symlink with simpler name
        simple_name = output_path / f"{target.lower()}_hst_icl.fits"

        import shutil
        if Path(local_path).exists():
            shutil.copy(local_path, simple_name)
            print(f"\n✓ Copied to: {simple_name}")
            return str(simple_name)
    else:
        print("Download failed or incomplete.")
        print(f"Status: {manifest['Status'][0] if len(manifest) > 0 else 'Unknown'}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download real Hubble CLASH/HFF data for ICL testing"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="MACS0416",
        choices=["MACS0416", "ABELL2744", "MACS0717", "MACS1149", "RXJ1347"],
        help="Target cluster name (default: MACS0416)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/clash",
        help="Output directory (default: ./data/clash)"
    )

    args = parser.parse_args()

    print(f"\nTarget: {args.target}")
    print(f"Output: {args.output}")
    print("")

    result = download_clash_data(args.target, args.output)

    if result:
        print("\n" + "=" * 70)
        print("Download successful!")
        print("=" * 70)
        print(f"\nTo run ICL workflow:")
        print(f"  python test_workflow.py --fits {result}")
        return 0
    else:
        print("\n" + "=" * 70)
        print("Download failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
