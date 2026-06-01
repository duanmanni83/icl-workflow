#!/usr/bin/env python3
"""
Test ICL workflow with real Hubble data from MAST.

This script provides two options:
1. Download real CLASH/HFF data using astroquery (large files, slow)
2. Use the synthetic data generator (fast, for testing)
"""

import sys
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_astroquery():
    """Check if astroquery is installed."""
    try:
        from astroquery.mast import Observations
        return True
    except ImportError:
        return False


def download_hubble_sample(target="MACS0416", output_dir="./data"):
    """
    Download a sample Hubble image using astroquery.

    Args:
        target: Target cluster name (e.g., 'MACS0416', 'ABELL2744')
        output_dir: Where to save the data
    """
    if not check_astroquery():
        print("Error: astroquery not installed.")
        print("Install with: pip install astroquery")
        return None

    from astroquery.mast import Observations
    from astropy.io import fits
    import time

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Searching MAST for {target}...")
    print(f"{'='*70}")

    # Query for CLASH observations
    print("\nQuerying MAST archive...")
    obs_table = Observations.query_criteria(
        provenance_name="CLASH",
        target_name=target,
        obs_collection="HST"
    )

    if len(obs_table) == 0:
        print(f"No CLASH data found for {target}, trying HFF...")
        obs_table = Observations.query_criteria(
            provenance_name="HFF",
            target_name=target,
            obs_collection="HST"
        )

    print(f"Found {len(obs_table)} observations")

    if len(obs_table) == 0:
        print("No data found. Please try a different target.")
        return None

    # Get products for first few observations
    print("Getting data products...")
    data_products = Observations.get_product_list(obs_table[:5])
    print(f"Total products: {len(data_products)}")

    # Filter for drizzled science images
    print("Filtering for science images...")

    # Look for DRZ/DRC files (drizzled images)
    mask = (
        (data_products['productSubGroupDescription'] == 'DRZ') |
        (data_products['productSubGroupDescription'] == 'DRC')
    )

    # Prefer F814W or F606W filters (good for ICL studies)
    preferred_filters = ['F814W', 'F606W', 'F105W', 'F125W']
    filter_mask = mask.copy()
    for filt in preferred_filters:
        filter_mask = filter_mask | data_products['productFilename'].str.contains(
            filt, case=False, na=False
        )

    filtered = data_products[filter_mask]

    print(f"Filtered to {len(filtered)} products")

    if len(filtered) == 0:
        print("No suitable images found.")
        return None

    # Show available products
    print("\nAvailable products:")
    for i, row in enumerate(filtered[:10]):
        print(f"  {i+1}. {row['productFilename']} ({row['size']/1024/1024:.1f} MB)")

    # Download the first suitable product
    to_download = filtered[:1]
    print(f"\nDownloading: {to_download[0]['productFilename']}")
    print("(This may take a few minutes for large files...)")

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
        print(f"Saved to: {local_path}")

        # Create a symlink/copy with simpler name
        simple_name = output_path / f"{target.lower()}_hst_icl.fits"

        import shutil
        shutil.copy(local_path, simple_name)
        print(f"Copied to: {simple_name}")

        # Show file info
        with fits.open(simple_name) as hdul:
            print("\nFITS file info:")
            hdul.info()
            if 'SCI' in hdul:
                data = hdul['SCI'].data
            elif len(hdul) > 1:
                data = hdul[1].data
            else:
                data = hdul[0].data
            print(f"\nImage shape: {data.shape}")
            print(f"Data range: {data.min():.2e} to {data.max():.2e}")

        return str(simple_name)
    else:
        print("Download failed or incomplete.")
        return None


def run_workflow_test(fits_path, mode="semi_auto"):
    """Run ICL workflow on the downloaded image."""
    from workflow import ICLWorkflow, WorkflowConfig

    print(f"\n{'='*70}")
    print(f"Running ICL Workflow")
    print(f"{'='*70}")
    print(f"Input: {fits_path}")
    print(f"Mode: {mode}")

    config = WorkflowConfig(
        output_dir=str(Path(fits_path).parent / "icl_output"),
        detect_thresh=1.5,
        dilation_factor=1.5,
        interpolation_method="rbf"
    )

    workflow = ICLWorkflow(config)

    # Track approvals
    approvals = []
    def callback(step, result):
        print(f"\n[HITL] {step}")
        print(f"  {result.message[:100]}...")
        if result.visualization_path:
            print(f"  Viz: {result.visualization_path}")
        approvals.append(step)
        # Auto-approve for demo
        return True

    workflow.register_human_callback(callback)

    # Run workflow
    state = workflow.run(
        fits_path,
        instrument="HST",
        mode=mode
    )

    print(f"\n{workflow.get_summary()}")
    print(f"\nCheckpoints reviewed: {len(approvals)}")

    return state


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test ICL workflow with real or synthetic Hubble data"
    )
    parser.add_argument(
        "--download",
        choices=["MACS0416", "ABELL2744", "MACS0717", "MACS1149", "auto"],
        help="Download real Hubble data for specified cluster"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic test data instead of downloading"
    )
    parser.add_argument(
        "--fits",
        help="Path to existing FITS file to process"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "semi_auto", "manual"],
        default="semi_auto",
        help="Workflow mode"
    )
    parser.add_argument(
        "--output",
        default="./data",
        help="Output directory"
    )

    args = parser.parse_args()

    # Determine data source
    if args.fits:
        # Use provided FITS file
        fits_path = args.fits
        if not Path(fits_path).exists():
            print(f"Error: File not found: {fits_path}")
            return 1

    elif args.download:
        # Download real data
        if args.download == "auto":
            target = "MACS0416"  # Default
        else:
            target = args.download

        fits_path = download_hubble_sample(target, args.output)
        if fits_path is None:
            print("\nDownload failed. Falling back to synthetic data...")
            args.synthetic = True

    if args.synthetic or (not args.fits and not args.download):
        # Use synthetic data
        print("\nUsing synthetic test data...")
        synth_script = Path(__file__).parent / "generate_test_data.py"
        fits_path = Path(args.output) / "synthetic_cluster.fits"

        subprocess.run([
            sys.executable,
            str(synth_script),
            "--output", str(fits_path),
            "--size", "512"
        ])

        if not fits_path.exists():
            print("Error: Failed to generate synthetic data")
            return 1

    # Run workflow
    if fits_path and Path(fits_path).exists():
        run_workflow_test(fits_path, args.mode)
        print(f"\n✓ Test completed successfully!")
        print(f"Results saved to: {Path(fits_path).parent / 'icl_output'}")
        return 0
    else:
        print("Error: No valid input file")
        return 1


if __name__ == "__main__":
    sys.exit(main())
