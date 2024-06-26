import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.components.encoder import Encoder
from src.models.components.temporal_model import TemporalModel
from src.models.components.decoder import Decoder
from src.utils.network import pack_sequence_dim, unpack_sequence_dim, set_bn_momentum
from src.utils.geometry import (
    cumulative_warp_features,
    calculate_birds_eye_view_parameters,
    VoxelsSumming,
)


class Damp(nn.Module):
    def __init__(self, model_cfg : DictConfig, common_cfg : DictConfig):
        super().__init__()

        self.model_cfg = model_cfg
        self.common_cfg = common_cfg


        bev_resolution, bev_start_position, bev_dimension = (
            calculate_birds_eye_view_parameters(
                self.common_cfg.lift.x_bound, self.common_cfg.lift.y_bound, self.common_cfg.lift.z_bound
            )
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        self.encoder_downsample = self.model_cfg.encoder.downsample
        self.encoder_out_channels = self.model_cfg.encoder.out_channels
        self.receptive_field = self.common_cfg.receptive_field

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.common_cfg.lift.x_bound[1], self.common_cfg.lift.y_bound[1])

        self.encoder = Encoder(encoder_cfg=self.model_cfg.encoder, D=self.depth_channels)


        # Temporal model
        temporal_in_channels = self.encoder_out_channels + 6
        self.temporal_model = TemporalModel(self.model_cfg.temporal_model,
            in_channels=temporal_in_channels,
            receptive_field=self.receptive_field,
            input_shape=self.bev_size
        )

        self.future_pred_in_channels = self.temporal_model.out_channels

        # Decoder
        self.decoder = Decoder(
            in_channels=self.future_pred_in_channels,
            n_classes=len(self.common_cfg.semantic_segmentation.weights),
            predict_future_flow=False,
        )

        set_bn_momentum(self, self.model_cfg.bn_momentum)

    def create_frustum(self):
        """
        Create a frustum grid in the image plane.

        This function creates a grid in the image plane by dividing the image dimensions by the encoder downsample factor.
        It then creates a depth grid by using the `torch.arange` function to generate a range of values between the minimum and maximum depth bounds.
        The depth grid is reshaped and expanded to match the dimensions of the frustum grid.
        Next, x and y grids are created using the `torch.linspace` function to generate evenly spaced values between 0 and the corresponding dimension minus 1.
        The x and y grids are reshaped and expanded to match the dimensions of the frustum grid.
        Finally, the x, y, and depth grids are stacked together to create the frustum grid, which contains data points in the image: left-right, top-bottom, depth.

        Returns:
            torch.nn.Parameter: The frustum grid as a parameter with requires_grad set to False.
        """
        h, w = self.common_cfg.image.final_dim
        downsampled_h, downsampled_w = (
            h // self.encoder_downsample,
            w // self.encoder_downsample,
        )

        # Depth grid
        depth_grid = torch.arange(*self.common_cfg.lift.d_bound, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(
            n_depth_slices, downsampled_h, downsampled_w
        )
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(
            n_depth_slices, downsampled_h, downsampled_w
        )

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def add_future_egomotions_to_features(self, x, future_egomotion):
        b, s, c = future_egomotion.shape
        h, w = x.shape[-2:]
        future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(b, s, c, h, w)
        # at time 0, no egomotion so feed zero vector
        future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                                future_egomotions_spatial[:, :(self.receptive_field-1)]], dim=1)
        x = torch.cat([x, future_egomotions_spatial], dim=-3)
        return x

    def forward(
        self,
        image,
        intrinsics,
        extrinsics,
        future_egomotion,
    ):
        """
        Forward pass of the model.

        Args:
            image (torch.Tensor): The input image tensor of shape (B, N, C, H, W).
            intrinsics (torch.Tensor): The intrinsics tensor of shape (B, N, 3, 3).
            extrinsics (torch.Tensor): The extrinsics tensor of shape (B, N, 4, 4).
            future_egomotion (torch.Tensor): The future egomotion tensor of shape (B, N, 3).
            
            In the context of NuScences, the inputs have the following shapes:
            The Image shape is 'torch.Size([1, 3, 6, 3, 224, 480])
            The Intrinsics shape is 'torch.Size([1, 3, 6, 3, 3])
            The Extrinsics shape is 'torch.Size([1, 3, 6, 4, 4])
            The Future Egomotions shape is 'torch.Size([1, 3, 6])

        Returns:
            dict: A dictionary containing the output tensor of shape (B, N, C, H, W).
        """

        # Only process features from the past and present
        image = image[:, : self.receptive_field].contiguous()
        intrinsics = intrinsics[:, : self.receptive_field].contiguous()
        extrinsics = extrinsics[:, : self.receptive_field].contiguous()
        future_egomotion = future_egomotion[:, : self.receptive_field].contiguous()

        # Lifting features and project to bird's-eye view
        # output tensor shape (B, S, C, H, W), where H and W are the grid sizes
        # The example output shape is 'torch.Size([1, 3, 64, 200, 200])'
        x = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics)

        # Warp past features to the present's reference frame
        # This does not change the shape 
        x = cumulative_warp_features(
            x.clone(), future_egomotion,
            mode='bilinear', spatial_extent=self.spatial_extent,
        )

        # Add future egomotions to features
        # The example output shape is 'torch.Size([1, 3, 70, 200, 200])'
        x = self.add_future_egomotions_to_features(x, future_egomotion)

        # temporal model
        # The final shape is torch.Size([1, 1, 64, 200, 200])
        # Note that the second dimension is the temporal dimension that is determined by
        # time - receptive_field + 1, since time == receptive_field by default, this final
        # size is 1
        states = self.temporal_model(x)

        # decoder
        # this final output is a dictionary containing
        # Segmentation, Instance Center, Instance Offset, Instance Flow
        # The second dimension are the present and futures predictions, which is just the present
        # Segmentation: torch.Size([1, 1, 2, 200, 200])
        # Instance Center: torch.Size([1, 1, 1, 200, 200])
        # Instance Offset: torch.Size([1, 1, 2, 200, 200])
        # Instance Flow: torch.Size([1, 1, 2, 200, 200])
        bev_output = self.decoder(states[:, -1:])



        # TODO: remove complex output, keeping for backward compatibility
        output = {}
        output = {**output, **bev_output}

        return output

    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features."""
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Camera to ego reference frame
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = (
            combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        )
        points += translation.view(B, N, 1, 1, 1, 3)

        # The 3 dimensions in the ego reference frame are: (forward, sides, height)
        return points

    def encoder_forward(self, x):
        """
        Apply the encoder forward pass to the input tensor.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, n_cameras, channels, height, width).

        Returns:
            torch.Tensor: The output tensor with shape (batch_size, n_cameras, depth, height, width, channels).
        """

        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        x = self.encoder(x) # with depth, the shape is (batch_size, channels, depth, height, width)
        x = x.view(b, n, *x.shape[1:]) # shape is (batch_size, n_cameras, channels, depth, height, width)
        x = x.permute(0, 1, 3, 4, 5, 2) # shape is (batch_size, n_cameras, depth, height, width, channels)

        return x

    def projection_to_birds_eye_view(self, x, geometry):
        """Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
        # batch, n_cameras, depth, height, width, channels
        batch, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, c, self.bev_dimension[0], self.bev_dimension[1]),
            dtype=torch.float,
            device=x.device,
        )

        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            # flatten x
            x_b = x[b].reshape(N, c)

            # Convert positions to integer indices
            geometry_b = (
                geometry[b] - (self.bev_start_position - self.bev_resolution / 2.0)
            ) / self.bev_resolution
            geometry_b = geometry_b.view(N, 3).long()

            # Mask out points that are outside the considered spatial extent.
            mask = (
                (geometry_b[:, 0] >= 0)
                & (geometry_b[:, 0] < self.bev_dimension[0])
                & (geometry_b[:, 1] >= 0)
                & (geometry_b[:, 1] < self.bev_dimension[1])
                & (geometry_b[:, 2] >= 0)
                & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            # Sort tensors so that those within the same voxel are consecutives.
            ranks = (
                geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                + geometry_b[:, 1] * (self.bev_dimension[2])
                + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = (
                x_b[ranks_indices],
                geometry_b[ranks_indices],
                ranks[ranks_indices],
            )

            # Project to bird's-eye view by summing voxels.
            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)

            bev_feature = torch.zeros(
                (
                    self.bev_dimension[2],
                    self.bev_dimension[0],
                    self.bev_dimension[1],
                    c,
                ),
                device=x_b.device,
            )
            bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

            # Put channel in second position and remove z dimension
            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)

            output[b] = bev_feature 
        return output

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics):
        """
        Calculate the bird's eye view features for the given input images.

        Args:
            x (torch.Tensor): The input images with shape (batch_size, sequence_length, number_of_cameras, channels, height, width).
            intrinsics (torch.Tensor): The intrinsic parameters of the cameras with shape (batch_size, sequence_length, number_of_cameras, 3, 3).
            extrinsics (torch.Tensor): The extrinsic parameters of the cameras with shape (batch_size, sequence_length, number_of_cameras, 4, 4).

        Returns:
            torch.Tensor: The bird's-eye view features with shape (batch_size, sequence_length, channels, 200, 200).
        """
        b, s, n, c, h, w = x.shape
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)

        geometry = self.get_geometry(intrinsics, extrinsics)
        x = self.encoder_forward(x) # (batch_size, n_cameras, depth, height, width, channels)
        x = self.projection_to_birds_eye_view(x, geometry)  # (batch, c, self.bev_dimension[0], self.bev_dimension[1]) 
        x = unpack_sequence_dim(x, b, s) # (batch, s, c, self.bev_dimension[0], self.bev_dimension[1])
        return x

