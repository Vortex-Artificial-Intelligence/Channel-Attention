from typing import List, Tuple
import unittest

import torch

from channel_attention import (
    SEAttention,
    SpatialAttention,
    ChannelAttention,
    ConvBlockAttention,
)


def generate_test_inputs(
    batch_size: List[int], n_channels: int, seq_len: int, height: int, width: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate random test inputs for time series and image data.
    """
    time_series_inputs = [
        torch.randn(batch, n_channels, seq_len) for batch in batch_size
    ]
    image_inputs = [
        torch.randn(batch, n_channels, height, width) for batch in batch_size
    ]
    return time_series_inputs, image_inputs


class TestAttention(unittest.TestCase):
    batch_size = [1, 4, 16]
    n_channels = 64
    seq_len = 128
    height = 128
    width = 128

    def test_SEAttention(self):
        """Test SEAttention module for both 1D and 2D inputs."""
        time_series_inputs, image_inputs = generate_test_inputs(
            self.batch_size, self.n_channels, self.seq_len, self.height, self.width
        )

        # Test SEAttention for time series (1D)
        se_attention_1d = SEAttention(n_dims=1, n_channels=self.n_channels)
        for x in time_series_inputs:
            output = se_attention_1d(x)
            self.assertEqual(output.shape, x.shape)

        # Test SEAttention for images (2D)
        se_attention_2d = SEAttention(n_dims=2, n_channels=self.n_channels)
        for x in image_inputs:
            output = se_attention_2d(x)
            self.assertEqual(output.shape, x.shape)

    def test_SpatialAttention(self):
        """Test SpatialAttention module for both 1D and 2D inputs."""
        time_series_inputs, image_inputs = generate_test_inputs(
            self.batch_size, self.n_channels, self.seq_len, self.height, self.width
        )

        # Test SpatialAttention for time series (1D)
        sam_1d = SpatialAttention(n_dims=1)
        for x in time_series_inputs:
            output = sam_1d(x)
            self.assertEqual(output.shape, x.shape)

        # Test SpatialAttention for images (2D)
        sam_2d = SpatialAttention(n_dims=2)
        for x in image_inputs:
            output = sam_2d(x)
            self.assertEqual(output.shape, x.shape)

    def test_ChannelAttention(self):
        """Test ChannelAttention module for both 1D and 2D inputs."""
        time_series_inputs, image_inputs = generate_test_inputs(
            self.batch_size, self.n_channels, self.seq_len, self.height, self.width
        )

        # Test ChannelAttention for time series (1D)
        cam_1d = ChannelAttention(n_dims=1, n_channels=self.n_channels)
        for x in time_series_inputs:
            output = cam_1d(x)
            self.assertEqual(output.shape, x.shape)

        # Test ChannelAttention for images (2D)
        cam_2d = ChannelAttention(n_dims=2, n_channels=self.n_channels)
        for x in image_inputs:
            output = cam_2d(x)
            self.assertEqual(output.shape, x.shape)

    def test_ConvBlockAttention(self):
        """Test ConvBlockAttention module for both 1D and 2D inputs."""
        time_series_inputs, image_inputs = generate_test_inputs(
            self.batch_size, self.n_channels, self.seq_len, self.height, self.width
        )

        # Test ConvBlockAttention for time series (1D)
        cbam_1d = ConvBlockAttention(n_dims=1, n_channels=self.n_channels)
        for x in time_series_inputs:
            output = cbam_1d(x)
            self.assertEqual(output.shape, x.shape)

        # Test ConvBlockAttention for images (2D)
        cbam_2d = ConvBlockAttention(n_dims=2, n_channels=self.n_channels)
        for x in image_inputs:
            output = cbam_2d(x)
            self.assertEqual(output.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
