import torch.nn as nn

from src.models.layers.temporal import Bottleneck3D, TemporalBlock


class TemporalModel(nn.Module):
    def __init__(
            self, in_channels, receptive_field, input_shape, start_out_channels=64, extra_in_channels=0,
            n_spatial_layers_between_temporal_layers=0, use_pyramid_pooling=True):
        """
        Initializes a TemporalModel object.

        Args:
            in_channels (int): The number of input channels.
            receptive_field (int): The receptive field of the model.
            input_shape (tuple): The shape of the input tensor.
            start_out_channels (int, optional): The number of output channels for the first temporal layer. Defaults to 64.
            extra_in_channels (int, optional): The number of extra input channels added to each temporal layer. Defaults to 0.
            n_spatial_layers_between_temporal_layers (int, optional): The number of spatial layers between each temporal layer. Defaults to 0.
            use_pyramid_pooling (bool, optional): Whether to use pyramid pooling. Defaults to True.

        Description:
            This function initializes a TemporalModel object. It creates a sequence of temporal and spatial layers based on the provided parameters. The number of temporal layers is determined by the receptive field of the model. Each temporal layer is followed by a sequence of spatial layers. The number of spatial layers is determined by the `n_spatial_layers_between_temporal_layers` parameter. The output channels of each temporal layer are incremented by the `extra_in_channels` parameter. The final number of output channels is stored in the `out_channels` attribute. The sequence of temporal and spatial layers is stored in the `model` attribute.
        """
        self.receptive_field = receptive_field
        n_temporal_layers = receptive_field - 1

        h, w = input_shape
        modules = []

        block_in_channels = in_channels
        block_out_channels = start_out_channels

        for _ in range(n_temporal_layers):
            if use_pyramid_pooling:
                use_pyramid_pooling = True
                pool_sizes = [(2, h, w)]
            else:
                use_pyramid_pooling = False
                pool_sizes = None
            temporal = TemporalBlock(
                block_in_channels,
                block_out_channels,
                use_pyramid_pooling=use_pyramid_pooling,
                pool_sizes=pool_sizes,
            )
            spatial = [
                Bottleneck3D(block_out_channels, block_out_channels, kernel_size=(1, 3, 3))
                for _ in range(n_spatial_layers_between_temporal_layers)
            ]
            temporal_spatial_layers = nn.Sequential(temporal, *spatial)
            modules.extend(temporal_spatial_layers)

            block_in_channels = block_out_channels
            block_out_channels += extra_in_channels

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
