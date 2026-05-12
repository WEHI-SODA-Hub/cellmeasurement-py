"""Tests for downsampling utilities."""

import numpy as np

from cellmeasurement.io.mask_io import maybe_downsample


class TestMaybeDownsample:
    """Tests for the maybe_downsample function."""

    def test_no_downsampling_factor_1(self):
        """Test that downsample_factor <= 1.0 returns original arrays."""
        image = np.random.rand(3, 256, 256).astype(np.float32)
        nuc = np.random.randint(0, 100, (256, 256), dtype=np.int64)
        cell = np.random.randint(0, 50, (256, 256), dtype=np.int64)

        image_ds, nuc_ds, cell_ds = maybe_downsample(image, nuc, cell, 1.0)

        assert image_ds is image  # Should return same object
        assert nuc_ds is nuc
        assert cell_ds is cell

    def test_no_downsampling_factor_below_threshold(self):
        """Test that downsample_factor rounding to < 2 returns original arrays."""
        image = np.random.rand(3, 256, 256).astype(np.float32)
        nuc = np.random.randint(0, 100, (256, 256), dtype=np.int64)
        cell = np.random.randint(0, 50, (256, 256), dtype=np.int64)

        # 1.4 rounds to 1, should not downsample
        image_ds, nuc_ds, cell_ds = maybe_downsample(image, nuc, cell, 1.4)

        assert image_ds is image
        assert nuc_ds is nuc
        assert cell_ds is cell

    def test_downsampling_factor_2(self):
        """Test downsampling by factor 2."""
        image = np.ones((3, 100, 100), dtype=np.float32) * 5.0
        nuc = np.zeros((100, 100), dtype=np.int64)
        cell = np.zeros((100, 100), dtype=np.int64)

        # Set up a simple pattern: nuc label 1 in top-left quadrant
        nuc[:50, :50] = 1
        cell[:75, :75] = 1

        image_ds, nuc_ds, cell_ds = maybe_downsample(image, nuc, cell, 2.0)

        assert image_ds.shape == (3, 50, 50)
        assert nuc_ds.shape == (50, 50)
        assert cell_ds.shape == (50, 50)
        assert image_ds.dtype == np.float32
        assert nuc_ds.dtype == np.int64
        assert cell_ds.dtype == np.int64

        # Image should preserve intensity (mean over 2x2 blocks)
        np.testing.assert_allclose(image_ds, 5.0)

        # Masks should preserve max label per block
        assert nuc_ds[0, 0] == 1  # Top-left has label 1
        assert cell_ds[0, 0] == 1

    def test_downsampling_factor_4(self):
        """Test downsampling by factor 4."""
        image = np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4)
        nuc = np.ones((4, 4), dtype=np.int64) * 5
        cell = np.ones((4, 4), dtype=np.int64) * 10

        image_ds, nuc_ds, cell_ds = maybe_downsample(image, nuc, cell, 4.0)

        # 4x4 image downsamples to 1x1
        assert image_ds.shape == (1, 1, 1)
        assert nuc_ds.shape == (1, 1)
        assert cell_ds.shape == (1, 1)

        # Image should be mean of all elements: (1+2+...+16)/16 = 8.5
        assert image_ds[0, 0, 0] == 8.5
        assert nuc_ds[0, 0] == 5
        assert cell_ds[0, 0] == 10

    def test_downsampling_crops_to_divisible_shape(self):
        """Test that downsampling handles non-divisible dimensions correctly."""
        image = np.ones((3, 103, 107), dtype=np.float32)  # Not divisible by 2
        nuc = np.ones((103, 107), dtype=np.int64)
        cell = np.ones((103, 107), dtype=np.int64)

        image_ds, nuc_ds, cell_ds = maybe_downsample(image, nuc, cell, 2.0)

        # Should crop to (102, 106) then downsample to (51, 53)
        assert image_ds.shape == (3, 51, 53)
        assert nuc_ds.shape == (51, 53)
        assert cell_ds.shape == (51, 53)

    def test_downsampling_with_none_nuc_mask(self):
        """Test downsampling when nuclear mask is None."""
        image = np.ones((3, 100, 100), dtype=np.float32)
        nuc = None
        cell = np.ones((100, 100), dtype=np.int64)

        image_ds, nuc_ds, cell_ds = maybe_downsample(image, nuc, cell, 2.0)

        assert image_ds.shape == (3, 50, 50)
        assert nuc_ds is None
        assert cell_ds.shape == (50, 50)

    def test_downsampling_preserves_dtype(self):
        """Test that downsampling preserves dtype."""
        image = np.random.rand(3, 100, 100).astype(np.float32)
        nuc = np.random.randint(0, 100, (100, 100), dtype=np.int64)
        cell = np.random.randint(0, 50, (100, 100), dtype=np.int64)

        image_ds, nuc_ds, cell_ds = maybe_downsample(image, nuc, cell, 2.0)

        assert image_ds.dtype == np.float32
        assert nuc_ds.dtype == np.int64
        assert cell_ds.dtype == np.int64

    def test_downsampling_label_preservation(self):
        """Test that downsampling preserves label values with block_reduce max."""
        # Create an image with distinct labels in different quadrants
        cell = np.zeros((100, 100), dtype=np.int64)
        cell[:50, :50] = 1
        cell[:50, 50:] = 2
        cell[50:, :50] = 3
        cell[50:, 50:] = 4

        image = np.ones((1, 100, 100), dtype=np.float32)
        nuc = None

        image_ds, nuc_ds, cell_ds = maybe_downsample(image, nuc, cell, 2.0)

        # Each 2x2 block in original should map to a single cell in downsampled
        assert cell_ds.shape == (50, 50)

        # Top-left should have label 1, top-right label 2, etc.
        assert cell_ds[0, 0] == 1  # top-left
        assert cell_ds[0, 25] == 2  # top-right
        assert cell_ds[25, 0] == 3  # bottom-left
        assert cell_ds[25, 25] == 4  # bottom-right

    def test_downsampling_very_small_image_zero_size(self):
        """Test that very small images that would downsample to zero are returned as-is."""
        image = np.ones((3, 1, 1), dtype=np.float32)
        nuc = np.ones((1, 1), dtype=np.int64)
        cell = np.ones((1, 1), dtype=np.int64)

        image_ds, nuc_ds, cell_ds = maybe_downsample(image, nuc, cell, 10.0)

        # Should not downsample since divisible size would be zero
        assert image_ds is image
        assert nuc_ds is nuc
        assert cell_ds is cell

    def test_downsampling_invalid_factor(self):
        """Test that invalid downsample factors raise errors."""
        image = np.ones((3, 100, 100), dtype=np.float32)
        nuc = np.ones((100, 100), dtype=np.int64)
        cell = np.ones((100, 100), dtype=np.int64)

        # Negative factor should not raise during maybe_downsample
        # (validation happens at CLI level)
        # But zero or negative values still return original arrays
        image_ds, nuc_ds, cell_ds = maybe_downsample(image, nuc, cell, 0.5)
        assert image_ds is image  # No downsampling for factor <= 1

    def test_downsampling_multiple_channels(self):
        """Test downsampling with multiple channels."""
        image = np.random.rand(10, 100, 100).astype(np.float32)
        nuc = np.random.randint(0, 50, (100, 100), dtype=np.int64)
        cell = np.random.randint(0, 30, (100, 100), dtype=np.int64)

        image_ds, nuc_ds, cell_ds = maybe_downsample(image, nuc, cell, 2.0)

        assert image_ds.shape == (10, 50, 50)
        assert image_ds.dtype == np.float32
        assert nuc_ds.shape == (50, 50)
        assert cell_ds.shape == (50, 50)
