import torch
import torch.nn as nn

# from lightning import LightningModule

from src.models.components.encoder import Encoder

# from src.models.components.temporal_model import TemporalModelIdentity, TemporalModel
# from src.models.components.distributions import DistributionModule
# from src.models.components.future_prediction import FuturePrediction
# from src.models.components.decoder import Decoder
from src.utils.network import pack_sequence_dim, unpack_sequence_dim, set_bn_momentum
from src.utils.geometry import (
    cumulative_warp_features,
    calculate_birds_eye_view_parameters,
    VoxelsSumming,
)


class Damp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        bev_resolution, bev_start_position, bev_dimension = (
            calculate_birds_eye_view_parameters(
                self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND
            )
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.encoder_downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE
        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape

        # Encoder
        self.encoder = Encoder(cfg=self.cfg.MODEL.ENCODER, D=self.depth_channels)

        set_bn_momentum(self, self.cfg.MODEL.BN_MOMENTUM)

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
        h, w = self.cfg.IMAGE.FINAL_DIM
        downsampled_h, downsampled_w = (
            h // self.encoder_downsample,
            w // self.encoder_downsample,
        )

        # Depth grid
        depth_grid = torch.arange(*self.cfg.LIFT.D_BOUND, dtype=torch.float)
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

        Returns:
            dict: A dictionary containing the output tensor of shape (B, N, C, H, W).
        """

        # Only process features from the past and present
        image = image[:, : self.receptive_field].contiguous()
        intrinsics = intrinsics[:, : self.receptive_field].contiguous()
        extrinsics = extrinsics[:, : self.receptive_field].contiguous()
        future_egomotion = future_egomotion[:, : self.receptive_field].contiguous()

        # Lifting features and project to bird's-eye view
        x = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics)

        output = x

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
        # batch, n_cameras, channels, height, width
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x = self.encoder(x)
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)

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
            torch.Tensor: The bird's-eye view features with shape (batch_size, sequence_length, channels, height, width).
        """
        b, s, n, c, h, w = x.shape
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)

        geometry = self.get_geometry(intrinsics, extrinsics)
        x = self.encoder_forward(x)
        x = self.projection_to_birds_eye_view(x, geometry)
        x = unpack_sequence_dim(x, b, s)
        return x

    def distribution_forward(
        self, present_features, future_distribution_inputs=None, noise=None
    ):
        """
        Parameters
        ----------
            present_features: 5-D output from dynamics module with shape (b, 1, c, h, w)
            future_distribution_inputs: 5-D tensor containing labels shape (b, s, cfg.PROB_FUTURE_DIM, h, w)
            noise: a sample from a (0, 1) gaussian with shape (b, s, latent_dim). If None, will sample in function

        Returns
        -------
            sample: sample taken from present/future distribution, broadcast to shape (b, s, latent_dim, h, w)
            present_distribution_mu: shape (b, s, latent_dim)
            present_distribution_log_sigma: shape (b, s, latent_dim)
            future_distribution_mu: shape (b, s, latent_dim)
            future_distribution_log_sigma: shape (b, s, latent_dim)
        """
        b, s, _, h, w = present_features.size()
        assert s == 1

        present_mu, present_log_sigma = self.present_distribution(present_features)

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            # Concatenate future labels to z_t
            future_features = (
                future_distribution_inputs[:, 1:].contiguous().view(b, 1, -1, h, w)
            )
            future_features = torch.cat([present_features, future_features], dim=2)
            future_mu, future_log_sigma = self.future_distribution(future_features)

        if noise is None:
            if self.training:
                noise = torch.randn_like(present_mu)
            else:
                noise = torch.zeros_like(present_mu)
        if self.training:
            mu = future_mu
            sigma = torch.exp(future_log_sigma)
        else:
            mu = present_mu
            sigma = torch.exp(present_log_sigma)
        sample = mu + sigma * noise

        # Spatially broadcast sample to the dimensions of present_features
        sample = sample.view(b, s, self.latent_dim, 1, 1).expand(
            b, s, self.latent_dim, h, w
        )

        output_distribution = {
            "present_mu": present_mu,
            "present_log_sigma": present_log_sigma,
            "future_mu": future_mu,
            "future_log_sigma": future_log_sigma,
        }

        return sample, output_distribution
