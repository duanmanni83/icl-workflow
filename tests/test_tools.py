"""Unit tests for ICL Workflow tools."""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from types import InstrumentType, InterpolationMethod, KernelShape
from utils import FITSHandler, Metrics, Visualization
from tools import (
    ToolExtractInitialMask,
    ToolGenerateSpikeMask,
    ToolMergeAndDilateMask,
    ToolInterpolateAndSubtract,
    ToolEvaluateFieldComplexity,
)


class TestFITSHandler(unittest.TestCase):
    """Test FITS I/O operations."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.fits"

        # Create test data
        self.test_data = np.random.randn(100, 100).astype(np.float64)
        from astropy.io import fits
        hdu = fits.PrimaryHDU(self.test_data)
        hdu.writeto(self.test_file, overwrite=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_read_fits(self):
        """Test FITS reading."""
        data, wcs, header = FITSHandler.read_fits(str(self.test_file))
        np.testing.assert_array_almost_equal(data, self.test_data)
        self.assertIsNotNone(header)

    def test_write_fits(self):
        """Test FITS writing."""
        output_file = Path(self.temp_dir) / "output.fits"
        test_output = np.ones((50, 50))
        FITSHandler.write_fits(test_output, str(output_file))
        self.assertTrue(output_file.exists())


class TestMetrics(unittest.TestCase):
    """Test quality metrics calculation."""

    def test_flux_metrics(self):
        """Test flux conservation metrics."""
        original = np.ones((100, 100))
        # Perfect residual (mean = 0)
        residual_good = np.random.randn(100, 100) * 0.1
        metrics = Metrics.calculate_flux_metrics(original, residual_good)

        self.assertIn('negative_pixel_ratio', metrics)
        self.assertIn('flux_conservation_score', metrics)
        self.assertGreater(metrics['flux_conservation_score'], 50)

    def test_ellipticity_calculation(self):
        """Test ellipticity calculation."""
        # Create elliptical Gaussian
        y, x = np.indices((50, 50))
        center = 25
        sigma_x, sigma_y = 5, 3  # Elliptical
        gaussian = np.exp(-((x - center)**2 / (2 * sigma_x**2) +
                           (y - center)**2 / (2 * sigma_y**2)))

        e1, e2 = Metrics._calculate_ellipticity(gaussian)
        self.assertIsNotNone(e1)
        self.assertIsNotNone(e2)


class TestToolEvaluateFieldComplexity(unittest.TestCase):
    """Test field complexity evaluation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.fits"

        # Create test image with some sources
        data = np.random.randn(100, 100) * 0.1
        # Add bright source
        y, x = np.indices((100, 100))
        data += 10 * np.exp(-((x - 50)**2 + (y - 50)**2) / 20)

        from astropy.io import fits
        hdu = fits.PrimaryHDU(data.astype(np.float64))
        hdu.writeto(self.test_file, overwrite=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_execute(self):
        """Test field complexity evaluation."""
        result = ToolEvaluateFieldComplexity.execute(
            str(self.test_file),
            output_dir=self.temp_dir
        )
        self.assertTrue(result.success)
        self.assertIn('complexity_score', result.metrics)
        self.assertIn('recommended_auto_mode', result.metrics)


class TestToolGenerateSpikeMask(unittest.TestCase):
    """Test spike mask generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.fits"

        # Create test image
        data = np.random.randn(200, 200) * 0.1

        from astropy.io import fits
        hdu = fits.PrimaryHDU(data.astype(np.float64))
        hdu.writeto(self.test_file, overwrite=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_csst_spikes(self):
        """Test CSST spike mask generation."""
        result = ToolGenerateSpikeMask.execute(
            str(self.test_file),
            instrument="CSST",
            center_x=100,
            center_y=100,
            output_dir=self.temp_dir
        )
        self.assertTrue(result.success)
        self.assertEqual(result.metrics['num_spikes'], 4)
        self.assertEqual(result.metrics['instrument'], 'CSST')

    def test_euclid_spikes(self):
        """Test Euclid spike mask generation."""
        result = ToolGenerateSpikeMask.execute(
            str(self.test_file),
            instrument="Euclid",
            center_x=100,
            center_y=100,
            output_dir=self.temp_dir
        )
        self.assertTrue(result.success)
        self.assertEqual(result.metrics['num_spikes'], 6)


class TestToolMergeAndDilateMask(unittest.TestCase):
    """Test mask merging and dilation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create test masks
        self.seg_mask = np.zeros((100, 100), dtype=np.uint8)
        self.seg_mask[40:60, 40:60] = 1

        self.spike_mask = np.zeros((100, 100), dtype=np.uint8)
        self.spike_mask[50, :] = 1

        from astropy.io import fits
        self.seg_file = Path(self.temp_dir) / "seg.fits"
        self.spike_file = Path(self.temp_dir) / "spike.fits"

        fits.PrimaryHDU(self.seg_mask).writeto(self.seg_file, overwrite=True)
        fits.PrimaryHDU(self.spike_mask).writeto(self.spike_file, overwrite=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_merge_and_dilate(self):
        """Test mask merging."""
        result = ToolMergeAndDilateMask.execute(
            str(self.seg_file),
            str(self.spike_file),
            dilation_factor=2,
            kernel_shape="disk",
            output_dir=self.temp_dir
        )
        self.assertTrue(result.success)
        self.assertIn('original_coverage', result.metrics)
        self.assertIn('dilated_coverage', result.metrics)
        # Dilated coverage should be larger
        self.assertGreater(
            result.metrics['dilated_coverage'],
            result.metrics['original_coverage']
        )


class TestToolInterpolateAndSubtract(unittest.TestCase):
    """Test ICL interpolation and subtraction."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create test image with smooth background
        y, x = np.indices((100, 100))
        self.image = (x + y) * 0.1 + np.random.randn(100, 100) * 0.01

        # Create mask covering center
        self.mask = np.zeros((100, 100), dtype=np.uint8)
        self.mask[40:60, 40:60] = 1

        from astropy.io import fits
        self.img_file = Path(self.temp_dir) / "image.fits"
        self.mask_file = Path(self.temp_dir) / "mask.fits"

        fits.PrimaryHDU(self.image.astype(np.float64)).writeto(self.img_file, overwrite=True)
        fits.PrimaryHDU(self.mask).writeto(self.mask_file, overwrite=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_interpolation_methods(self):
        """Test different interpolation methods."""
        for method in ['rbf', 'linear', 'nearest']:
            result = ToolInterpolateAndSubtract.execute(
                str(self.img_file),
                str(self.mask_file),
                method=method,
                output_dir=self.temp_dir
            )
            self.assertTrue(result.success)
            self.assertIn('overall_score', result.metrics)
            self.assertIn('passed', result.metrics)


if __name__ == '__main__':
    unittest.main()
