import torch.nn as nn
from omegaconf import DictConfig

from src.models.layers.temporal import Bottleneck3D, TemporalBlock


class TemporalModel(nn.Module):
    def __init__(
            self, cfg_temporal_model : DictConfig, in_channels, receptive_field, input_shape):
        super().__init__()

        self.cfg_temporal_model = cfg_temporal_model
        self.receptive_field = receptive_field
        n_temporal_layers = receptive_field - 1

        h, w = input_shape
        modules = []

        block_in_channels = in_channels
        block_out_channels = self.cfg_temporal_model.start_out_channels

        self.use_pyramid_pooling = self.cfg_temporal_model.use_pyramid_pooling
        for _ in range(n_temporal_layers):
            if self.use_pyramid_pooling:
                self.use_pyramid_pooling = True
                pool_sizes = [(2, h, w)]
            else:
                self.use_pyramid_pooling = False
                pool_sizes = None
            temporal = TemporalBlock(
                block_in_channels,
                block_out_channels,
                use_pyramid_pooling=self.use_pyramid_pooling,
                pool_sizes=pool_sizes,
            )
            spatial = [
                Bottleneck3D(block_out_channels, block_out_channels, kernel_size=(1, 3, 3))
                for _ in range(self.cfg_temporal_model.inbetween_layers)
            ]
            temporal_spatial_layers = nn.Sequential(temporal, *spatial)
            modules.extend(temporal_spatial_layers)

            block_in_channels = block_out_channels
            block_out_channels += self.cfg_temporal_model.extra_in_channels

        self.out_channels = block_in_channels

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch, time - receptive_field + 1, C, H, W).

        This function takes an input tensor of shape (batch, time, C, H, W) and performs the following steps:
        1. Reshapes the input tensor to (batch, C, time, H, W) using the `permute` method.
        2. Passes the reshaped tensor through the model.
        3. Reshapes the output tensor back to (batch, C, time, H, W) using the `permute` method.
        4. Slices the output tensor to exclude the first `receptive_field - 1` time steps.

        Note: The `receptive_field` attribute is used to determine the number of time steps to exclude from the output tensor.
        """
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x[:, (self.receptive_field - 1):]


class TemporalModelIdentity(nn.Module):
    def __init__(self, in_channels, receptive_field):
        super().__init__()
        self.receptive_field = receptive_field
        self.out_channels = in_channels

    def forward(self, x):
        return x[:, (self.receptive_field - 1):]
