import unittest


class TestChannelAttention(unittest.TestCase):

    def test_se_attention_1d(self):
        import torch
        from channel_attention.squeeze_excitation import SEAttention1D

        batch_size = 4
        channels = 8
        length = 16
        reduction = 2

        x = torch.randn(batch_size, channels, length)
        se_block = SEAttention1D(channels, reduction)
        output = se_block(x)

        self.assertEqual(output.shape, x.shape)

    def test_se_attention_2d(self):
        import torch
        from channel_attention.squeeze_excitation import SEAttention2D

        batch_size = 4
        channels = 8
        height = 16
        width = 16
        reduction = 2

        x = torch.randn(batch_size, channels, height, width)
        se_block = SEAttention2D(channels, reduction)
        output = se_block(x)

        self.assertEqual(output.shape, x.shape)
