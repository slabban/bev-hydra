import torch.nn as nn
from omegaconf import DictConfig
from efficientnet_pytorch import EfficientNet

from src.models.layers.convolutions import UpsamplingConcat


class Encoder(nn.Module):
    def __init__(self, encoder_cfg : DictConfig, D):
        super().__init__()
        self.D = D
        self.C = encoder_cfg.out_channels
        self.use_depth_distribution = encoder_cfg.use_depth_distribution
        self.downsample = encoder_cfg.downsample
        self.version = encoder_cfg.name.split('-')[1]

        self.backbone = EfficientNet.from_pretrained(encoder_cfg.NAME)
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        if self.use_depth_distribution:
            self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C + self.D, kernel_size=1, padding=0)
        else:
            self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']

        x = self.upsampling_layer(input_1, input_2)
        return x

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, depth, height, width) if use_depth_distribution is True,
                          otherwise (batch_size, channels, depth, 1, height, width).

        This function first applies the depth_layer to the input tensor x. Then, if use_depth_distribution is True, it calculates the softmax of the first self.D channels of x,
        multiplies it element-wise with the remaining channels, and returns the result. If use_depth_distribution is False, it unsqueezes the last dimension of x,
        repeats it self.D times along the depth dimension, and returns the result.
        """
        x = self.depth_layer(x)  # feature and depth head

        if self.use_depth_distribution:
            depth = x[:, : self.D].softmax(dim=1)
            x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)  # outer product depth and features
        else:
            x = x.unsqueeze(2).repeat(1, 1, self.D, 1, 1)

        return x
