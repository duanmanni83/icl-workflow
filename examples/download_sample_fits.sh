#!/bin/bash
# Download sample CLASH/HFF FITS files for ICL workflow testing
# Data source: STScI HLSP Archive (https://archive.stsci.edu/prepds/clash/)

set -e

# Output directory
OUTPUT_DIR="${1:-./data}"
mkdir -p "$OUTPUT_DIR"

echo "=================================="
echo "CLASH Sample Data Download Script"
echo "=================================="
echo ""
echo "Target directory: $OUTPUT_DIR"
echo ""

# Base URL for CLASH data
BASE_URL="https://archive.stsci.edu/missions/hlsp/clash"

# Small test data (MACS0416 WFC3/IR F105W drizzled image)
# ~150MB file - good for testing ICL workflow
SAMPLE_FILE="macs0416_bcgs_065mas_hst_wfc3ir_f105w_drz.fits"
SAMPLE_URL="${BASE_URL}/macs0416/data/hst/scale_65mas/macs0416_bcgs_065mas_hst_wfc3ir_f105w_v1_drz.fits"

echo "Downloading sample CLASH data..."
echo "  Cluster: MACS J0416.1-2403"
echo "  Filter: F105W (WFC3/IR)"
echo "  Scale: 65mas/pixel"
echo "  Note: File size ~150MB, may take 1-2 minutes"
echo ""

# Download using wget or curl
if command -v wget &> /dev/null; then
    echo "Using wget..."
    wget -c --show-progress -O "${OUTPUT_DIR}/${SAMPLE_FILE}" "$SAMPLE_URL" || {
        echo "wget failed, trying curl..."
        curl -L -o "${OUTPUT_DIR}/${SAMPLE_FILE}" -C - "$SAMPLE_URL"
    }
elif command -v curl &> /dev/null; then
    echo "Using curl..."
    curl -L -o "${OUTPUT_DIR}/${SAMPLE_FILE}" -C - "$SAMPLE_URL"
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

echo ""
echo "✓ Download complete: ${OUTPUT_DIR}/${SAMPLE_FILE}"
echo ""

# Verify the file
if command -v fitsinfo &> /dev/null; then
    echo "FITS file info:"
    fitsinfo "${OUTPUT_DIR}/${SAMPLE_FILE}" 2>/dev/null || echo "  (could not read FITS info)"
fi

echo ""
echo "To run ICL workflow on this image:"
echo "  python ../cli.py workflow ${OUTPUT_DIR}/${SAMPLE_FILE} --instrument HST --mode semi_auto"
echo ""
echo "Or use Python API:"
echo "  python -c \""
echo "    from src.workflow import ICLWorkflow, WorkflowConfig"
echo "    workflow = ICLWorkflow(WorkflowConfig(output_dir='./icl_output'))"
echo "    workflow.run('${OUTPUT_DIR}/${SAMPLE_FILE}', instrument='HST', mode='semi_auto')"
echo "  \""
echo ""

# Alternative download URLs for other clusters
cat << 'EOF' > "${OUTPUT_DIR}/MORE_DATA.txt"
More CLASH Data URLs (copy/paste to browser or use wget):
============================================================

Abell 2744 (Pandora's Cluster):
  https://archive.stsci.edu/missions/hlsp/clash/abell2744/data/hst/

MACS J0717.5+3745:
  https://archive.stsci.edu/missions/hlsp/clash/macs0717/data/hst/

MACS J1149.6+2223:
  https://archive.stsci.edu/missions/hlsp/clash/macs1149/data/hst/

RX J1347.5-1145:
  https://archive.stsci.edu/missions/hlsp/clash/rxj1347/data/hst/

Recommended filters for ICL studies:
  - F606W (ACS/WFC): Good for optical ICL
  - F814W (ACS/WFC): Deep optical, common
  - F105W (WFC3/IR): Near-infrared ICL
  - F125W (WFC3/IR): Near-infrared ICL
  - F140W (WFC3/IR): Near-infrared ICL
  - F160W (WFC3/IR): Near-infrared ICL

File naming convention:
  [cluster]_bcgs_065mas_hst_[instrument]_[filter]_v1_drz.fits

File types:
  - drz.fits: Drizzled science image (primary data product)
  - wht.fits: Weight map (inverse variance)
  - ctx.fits: Context image (data quality)

All CLASH data are public and free to use.
Citation: Postman et al. 2012 (ApJS, 199, 25)
EOF

echo "Created: ${OUTPUT_DIR}/MORE_DATA.txt"
echo ""
echo "Done!"
