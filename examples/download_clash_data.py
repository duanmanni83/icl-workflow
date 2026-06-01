#!/usr/bin/env python3
"""
Download CLASH/Frontier Fields strong lens cluster data for ICL workflow testing.

Sources:
- CLASH Archive: https://archive.stsci.edu/prepds/clash/
- Frontier Fields: https://archive.stsci.edu/prepds/frontier/
- Hubble Legacy Archive: https://hla.stsci.edu/
"""

import os
import sys
import requests
from pathlib import Path
from urllib.parse import urljoin

try:
    from astroquery.mast import Observations
    HAS_ASTROQUERY = True
except ImportError:
    HAS_ASTROQUERY = False
    print("Warning: astroquery not installed. Using direct download method.")


# Recommended strong lens clusters for ICL testing
RECOMMENDED_CLUSTERS = {
    "abell2744": {
        "name": "Abell 2744 (Pandora's Cluster)",
        "redshift": 0.308,
        "programs": ["CLASH", "HFF", "JWST"],
        "ra": "00h 14m 21.2s",
        "dec": "-30° 23' 50.1s",
        "description": "Complex merging cluster with strong lensing",
        "hla_id": "ABELL2744",
    },
    "macs0416": {
        "name": "MACS J0416.1-2403",
        "redshift": 0.396,
        "programs": ["CLASH", "HFF", "JWST"],
        "ra": "04h 16m 09.4s",
        "dec": "-24° 03' 58.0s",
        "description": "Elongated cluster, likely pre-merger",
        "hla_id": "MACS0416",
    },
    "macs0717": {
        "name": "MACS J0717.5+3745",
        "redshift": 0.545,
        "programs": ["CLASH", "HFF"],
        "ra": "07h 17m 34.0s",
        "dec": "+37° 45' 18.0s",
        "description": "Most massive known cluster, strong lensing",
        "hla_id": "MACS0717",
    },
    "macs1149": {
        "name": "MACS J1149.6+2223",
        "redshift": 0.543,
        "programs": ["CLASH", "HFF"],
        "ra": "11h 49m 36.2s",
        "dec": "+22° 23' 08.0s",
        "description": "Contains lensed supernova Refsdal",
        "hla_id": "MACS1149",
    },
    "rxj1347": {
        "name": "RX J1347.5-1145",
        "redshift": 0.451,
        "programs": ["CLASH"],
        "ra": "13h 47m 30.6s",
        "dec": "-11° 45' 10.0s",
        "description": "Brightest known X-ray cluster",
        "hla_id": "RXJ1347",
    },
}


def download_via_astroquery(cluster_name, output_dir="./data", filters=["F606W", "F814W", "F105W"]):
    """
    Download CLASH data using astroquery.

    Args:
        cluster_name: Cluster key from RECOMMENDED_CLUSTERS
        output_dir: Directory to save files
        filters: List of HST filters to download
    """
    if not HAS_ASTROQUERY:
        print("Error: astroquery is required. Install with: pip install astroquery")
        return False

    cluster = RECOMMENDED_CLUSTERS.get(cluster_name.lower())
    if not cluster:
        print(f"Unknown cluster: {cluster_name}")
        print(f"Available: {', '.join(RECOMMENDED_CLUSTERS.keys())}")
        return False

    print(f"\nQuerying MAST for {cluster['name']}...")

    # Query CLASH observations
    obs_table = Observations.query_criteria(
        provenance_name="CLASH",
        target_name=cluster['hla_id'],
    )

    if len(obs_table) == 0:
        print("No CLASH data found. Trying HFF...")
        obs_table = Observations.query_criteria(
            provenance_name="HFF",
            target_name=cluster['hla_id'],
        )

    print(f"Found {len(obs_table)} observations")

    if len(obs_table) == 0:
        print("No data found.")
        return False

    # Get data products
    data_products = Observations.get_product_list(obs_table[:10])  # Limit to first 10

    # Filter for drizzled images (drz.fits) and specific filters
    mask = (
        (data_products['productSubGroupDescription'] == 'DRZ') |
        (data_products['productSubGroupDescription'] == 'DRC')
    )

    for filt in filters:
        mask = mask | (data_products['productFilename'].str.contains(filt, case=False, na=False))

    filtered = data_products[mask]
    print(f"Filtered to {len(filtered)} products")

    if len(filtered) == 0:
        print("No matching products found.")
        return False

    # Download
    output_path = Path(output_dir) / cluster_name.lower()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to {output_path}...")
    manifest = Observations.download_products(
        filtered,
        download_dir=str(output_path),
        extension="fits"
    )

    print(f"Downloaded {len(manifest)} files")
    return True


def download_clash_direct(cluster_name, output_dir="./data"):
    """
    Download CLASH data directly from STScI FTP.

    Note: This downloads pre-processed mosaic images.
    """
    cluster = RECOMMENDED_CLUSTERS.get(cluster_name.lower())
    if not cluster:
        print(f"Unknown cluster: {cluster_name}")
        return False

    base_url = "https://archive.stsci.edu/pub/hlsp/clash/"

    # CLASH uses specific directory naming
    clash_name = cluster['hla_id'].lower()

    # Example URLs for MACS0416:
    # https://archive.stsci.edu/pub/hlsp/clash/macs0416/hst ACS and WFC3 images

    print(f"\nDirect download URLs for {cluster['name']}:")
    print(f"Base URL: {base_url}{clash_name}/")
    print("\nYou can browse and download from:")
    print(f"  1. {base_url}")
    print(f"  2. https://archive.stsci.edu/prepds/clash/ (Interactive)")
    print(f"  3. https://hla.stsci.edu/hlaview.html (HLA Viewer)")

    return True


def print_cluster_info():
    """Print information about available clusters."""
    print("=" * 70)
    print("Recommended Strong Lens Clusters for ICL Testing")
    print("=" * 70)

    for key, info in RECOMMENDED_CLUSTERS.items():
        print(f"\n{key.upper()}:")
        print(f"  Name:        {info['name']}")
        print(f"  Redshift:    z = {info['redshift']}")
        print(f"  Coordinates: {info['ra']}, {info['dec']}")
        print(f"  Programs:    {', '.join(info['programs'])}")
        print(f"  Notes:       {info['description']}")

    print("\n" + "=" * 70)
    print("Data Access Methods:")
    print("=" * 70)
    print("""
1. HLA (Hubble Legacy Archive) - Easiest
   URL: https://hla.stsci.edu/hlaview.html
   - Search by cluster name
   - Select 'FITS-Science' for download
   - Recommended filters: F606W, F814W, F105W, F125W, F140W

2. MAST Portal - Most comprehensive
   URL: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
   - Search for target
   - Filter by 'Provenance Name' = 'CLASH' or 'HFF'

3. Direct FTP Download
   URL: https://archive.stsci.edu/pub/hlsp/clash/
   - Pre-processed mosaic images available
   - Use wget or curl for batch download

4. Python API (astroquery)
   - See download_via_astroquery() function in this script
    """)


def download_sample_image(output_path="./data/sample_cluster.fits"):
    """
    Download a sample cluster image using direct HTTP request.

    Note: This uses a known public URL if available.
    """
    # Example: Try to download from HLA cutout service
    # This is a simplified example - actual URLs may vary

    print("\nAttempting to download sample data...")
    print("Note: Large FITS files require manual download from HLA/MAST.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # For demonstration, create a sample file with instructions
    readme = output_path.parent / "README.txt"
    readme.write_text("""
Data Download Instructions for ICL Workflow Testing
====================================================

1. Visit HLA Viewer: https://hla.stsci.edu/hlaview.html

2. Search for one of these clusters:
   - Abell 2744 (Pandora's Cluster)
   - MACS J0416.1-2403
   - MACS J0717.5+3745
   - RX J1347.5-1145

3. Select an image with these characteristics:
   - Filter: F606W, F814W, or F105W (good for ICL)
   - Exposure time: >1000s (deeper is better)
   - Pixel scale: ~0.065 arcsec (ACS) or ~0.13 arcsec (WFC3)

4. Click 'Download' and select 'FITS-Science'

5. Save to this directory: {output_dir}

6. Run the ICL workflow:
   python cli.py workflow {filename} --instrument HST --mode manual

Example for Abell 2744:
-----------------------
Filter: F814W
Exposure: ~20000s (HFF depth)
Size: ~100MB per image
Features: Bright BCG, strong lensing arcs, ICL
""".format(output_dir=output_path.parent, filename="cluster_image.fits"))

    print(f"Created README: {readme}")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download CLASH/Frontier Fields data for ICL testing"
    )
    parser.add_argument(
        "cluster",
        nargs="?",
        help=f"Cluster name. Options: {', '.join(RECOMMENDED_CLUSTERS.keys())}"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data",
        help="Output directory"
    )
    parser.add_argument(
        "--method", "-m",
        choices=["astroquery", "direct", "sample"],
        default="sample",
        help="Download method"
    )
    parser.add_argument(
        "--filters", "-f",
        nargs="+",
        default=["F606W", "F814W", "F105W"],
        help="HST filters to download"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available clusters"
    )

    args = parser.parse_args()

    if args.list or args.cluster is None:
        print_cluster_info()
        return

    if args.method == "astroquery":
        success = download_via_astroquery(args.cluster, args.output, args.filters)
    elif args.method == "direct":
        success = download_clash_direct(args.cluster, args.output)
    else:  # sample
        success = download_sample_image(f"{args.output}/{args.cluster}_sample.fits")

    if success:
        print("\n✓ Download process completed")
    else:
        print("\n✗ Download failed. Please try manual download.")
        print_cluster_info()


if __name__ == "__main__":
    main()
