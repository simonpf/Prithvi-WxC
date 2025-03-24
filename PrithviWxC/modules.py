"""
PrithviWxC.modules
==================

Provides neural-network modules for the PrithviWxC model.
"""
from importlib.metadata import version
from typing import Callable, Optional, Tuple, Union

import torch
if version("torch") > "2.3.0":
    from torch.nn.attention import SDPBackend, sdpa_kernel
from torch import nn
import torch.nn.functional as F


class LayerNormFirst(nn.Module):
    """
    Layer norm performed along the dimension 1 for image data in channels-first format.
    """
    def __init__(self, n_channels: int, eps: float = 1e-5):
        """
        Args:
            n_channels: The number of channels in the input.
            eps: Epsilon added to variance to avoid numerical issues. """
        super().__init__()
        self.n_channels = n_channels
        self.scaling = nn.Parameter(torch.ones(n_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(n_channels), requires_grad=True)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to x.
        """
        dtype = x.dtype
        mu = x.mean(1, keepdim=True)
        x_n = (x - mu).to(dtype=torch.float32)
        var = x_n.pow(2).mean(1, keepdim=True)
        x_n = x_n / torch.sqrt(var + self.eps)
        shape_ext = (self.n_channels,) + (1,) * (x_n.dim() - 2)
        x = self.scaling.reshape(shape_ext) * x_n.to(dtype=dtype) + self.bias.reshape(shape_ext)
        return x


class Reflect(nn.Module):
    """
    Pad input by reflecting the input tensor.
    """
    def __init__(self, pad: Union[int, Tuple[int]]):
        """
        Instantiates a padding layer.

        Args:
            pad: N-tuple defining the padding added to the n-last dimensions
                of the tensor. If an int, the same padding will be added to the
                two last dimensions of the tensor.
        """
        super().__init__()
        if isinstance(pad, int):
            pad = (pad,) * 2

        full_pad = []
        for n_elems in pad:
            if isinstance(n_elems, (tuple, list)):
                full_pad += [n_elems[0], n_elems[1]]
            elif isinstance(n_elems, int):
                full_pad += [n_elems, n_elems]
            else:
                raise ValueError(
                    "Expected elements of pad tuple to be tuples of integers or integers. "
                    "Got %s.", type(n_elems)
                )

        full_pad = tuple(full_pad[::-1])
        self.pad = full_pad


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add padding to tensor.

        Args:
            x: The input tensor.

        Return:
            The padded tensor.
        """
        return nn.functional.pad(x, self.pad, "reflect")


class InvertedBottleneckBlock(nn.Module):
    """
    Inverted-bottleneck block is used in MobileNet and Efficient net where it is referred
    to as MBConv
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion_factor: int = 4,
            kernel_size: int = 3,
            activation_factory: Callable[[], nn.Module] = nn.GELU,
            normalization_factory: Callable[[int], nn.Module] = LayerNormFirst,
            padding: Optional[Tuple[int]] = None,
            padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
            downsample: Optional[int] = None,
            fused: bool = False,
            stochastic_depth: Optional[float] = None,
    ):
        """
        Args:
            in_channels: The number of channels in the input tensor.
            out_channels: The number of channels in the output.
            expansion_factor: The number of channels in the inverted bottleneck is calculated by
                multiplying 'out_channels' with this expansion factor.
            kernel_size: The kernel size to use for spatial mixing.
            activation_factory: A factory functional to create the activation layers.
            normalization_factory: A factory functional to create the normalization layers.
            padding: The padding to apply before the spatial convolutions.
            padding_factory: A factory functional to create the padding layer.
            downsample: The downsampling to apply in the layer.
            fused: Whether or not to fuse the first two convolution layers.
            stochastic_depth: The probabilistic depth of the layer, i.e., the probability that the
                layer is applied to the input.
        """
        super().__init__()
        self.act = activation_factory()
        act = activation_factory()

        hidden_channels = out_channels * expansion_factor
        self.stochastic_depth = stochastic_depth

        stride = (1, 1)
        if downsample is not None:
            if isinstance(downsample, int):
                downsample = (downsample,) * 2
            if max(downsample) > 1:
                stride = downsample


        if in_channels == out_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=stride,
                    stride=stride
                ),
                LayerNormFirst(out_channels)
            )

        padding = (kernel_size // 2,) * 2
        if 1 < max(stride):
            padding = 0

        blocks = []
        if not fused:
            blocks += [
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
                normalization_factory(hidden_channels),
                act
            ]


            blocks += [
                padding_factory(padding),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size if max(stride) < 2 else stride,
                    stride=stride,
                    groups=hidden_channels,
                ),
                normalization_factory(hidden_channels),
                act
            ]
        else:

            blocks += [
                padding_factory(padding),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                normalization_factory(hidden_channels),
                act
            ]

        blocks += [
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            normalization_factory(out_channels),
            act
        ]
        self.body = nn.Sequential(*blocks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate input through layer.
        """
        shortcut = self.projection(x)
        return shortcut + self.body(x)


class ObservationEncoder(nn.Module):
    """
    Convolutional encoder used to encode satellite observations and meta data.

    The observation encoder downsamples and encodes all satellite observation layers separately.
    """
    def __init__(
            self,
            n_meta_features: int,
            obs_patch_size: Tuple[int, int] = (4, 4),
            channels: Tuple[int, int] = (16, 32, 64)
    ):
        """
        Observation encoder for PrithviWxC model.
        """
        super().__init__()
        self.n_meta_features = n_meta_features

        patching = tuple([sze // 2 for sze in obs_patch_size])
        channels_s1, channels_s2, channels_s3 = channels
        self.channels = channels
        self.patch_height, self.patch_width = obs_patch_size

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(1, channels_s1, kernel_size=patching, stride=patching),
            LayerNormFirst(channels_s1),
            InvertedBottleneckBlock(channels_s1, channels_s2, expansion_factor=1, downsample=(2, 2)),
            InvertedBottleneckBlock(channels_s2, channels_s3, expansion_factor=2),
            InvertedBottleneckBlock(channels_s3, channels_s3),
        )
        self.meta_encoder = nn.Sequential(
            nn.Conv2d(n_meta_features, channels_s1, kernel_size=patching, stride=patching),
            LayerNormFirst(channels_s1),
            InvertedBottleneckBlock(channels_s1, channels_s2, expansion_factor=1, downsample=(2, 2)),
            InvertedBottleneckBlock(channels_s2, channels_s3, expansion_factor=2),
            InvertedBottleneckBlock(channels_s3, channels_s3),
        )
        self.pos_encoder = nn.AvgPool2d(kernel_size=patching, stride=patching)
        self.mask_encoder = nn.MaxPool2d(kernel_size=obs_patch_size, stride=obs_patch_size)


    def forward(
            self,
            obs: torch.Tensor,
            obs_mask: torch.Tensor,
            meta: torch.Tensor,
            pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observations, observation mask, meta data, and position encoding.

        The observation tensor is expected to have shape [B x T x GY x GX x O x 1 x LY x LX], where:
            - B: is the batch dimension
            - T: is the time dimension
            - GY: Is the number of vertical (meridional) mask units.
            - GX: Is the number of horizontal (zonal) mask units.
            - O: Is the number of observation layers.
            - 1: Is just the channels dimension, which is 1 by definition for the observation layers.
            - LY: Is the vertical (meridional) dimension of the observation patches.
            - LX: Is the horizontal (zonal) dimension of the observation patches.

        Args:
            obs: Tensor holding all input observation layers.
            obs_mask: A binary mask identifying valid observation pixels.
            meta: The observation meta data.
            pos: The position encoding of the PrithviWxC model.
        """
        if obs.dim() < 8:
            obs = obs.unsqueeze(-3)
        B, T, GY, GX, O, _, H, W = obs.shape

        obs_enc = self.obs_encoder(obs.view(-1, 1, H, W))
        obs_enc = obs_enc.reshape((B, T, GY, GX, O, self.channels[-1], H // self.patch_height, W // self.patch_width))

        # Use MaxPooling with negative mask to perform MinPooling
        obs_mask_enc = -1.0 * self.mask_encoder(-1.0 * obs_mask.reshape(-1, 1, H, W))
        obs_mask_enc = obs_mask_enc.reshape((B, T, GY, GX, O, H // self.patch_height, W // self.patch_width))

        meta_enc = self.meta_encoder(meta.view(-1, self.n_meta_features, H, W))
        meta_enc = meta_enc.reshape((B, T, GY, GX, O, self.channels[-1], H // self.patch_height, W // self.patch_width))

        # pos_enc: [B x C_int x 180 x 288] -> [B x 1 x C_int x GY x LY x GX x LX]
        pos_enc = self.pos_encoder(pos).reshape(B, 1, pos.shape[1], GY, H // self.patch_height, GX, W // self.patch_width)
        # [B x 1 x C_int x GY x LY x GX x LX] -> [B x T x GY x GX x C_int x LY x LX]
        pos_enc = torch.permute(pos_enc, (0, 1, 3, 5, 2, 4, 6)).contiguous()

        subsampling = pos_enc.shape[-3] // self.channels[-1]
        pos_enc = pos_enc[..., ::subsampling, :, :].unsqueeze(-4)

        return obs_enc, obs_mask_enc, meta_enc, pos_enc


class MultiheadCrossAttention(nn.Module):
    """
    Multi-head cross attention layer for integrating observations into the PrithviWxC model.

    This layer allows a group of latent PrithviWxC pixels to attend to the spatially collocated observations.
    """

    def __init__(
            self,
            features_latent: int,
            features_obs: int,
            n_heads: int, dropout: float,
            obs_patch_size: Tuple[int, int] = (6, 4)
    ) -> None:
        """
        Args:
            features_latent: The number of features of the PrithviWxC encoding.
            features_obs: The number of features of the encoded observations.
            n_heads: Number of attention heads.
            dropout: Dropout.
        """
        super().__init__()

        self.features_latent = features_latent
        self.features_obs = features_obs
        self.n_heads = n_heads
        self.dropout = dropout

        self.q_layer = torch.nn.Linear(features_latent, self.n_heads * features_obs, bias=False)
        self.k_layer = torch.nn.Linear(features_obs, self.n_heads * features_obs, bias=False)
        self.v_layer = torch.nn.Linear(features_obs, self.n_heads * features_obs, bias=False)
        self.w_layer = torch.nn.Linear(self.n_heads * features_obs, features_latent, bias=False)
        self.patch_height = obs_patch_size[0] // 2
        self.patch_width = obs_patch_size[1] // 2

    def forward(self, args) -> torch.Tensor:
        """
        Args:
            args: A tuple ``(x, obs, obs_mask)`` containing the latent model state ``x``, the encoded observations
                ``obs``, and the ``obs_mask`` indicating which observation pixel contain valid observations.
        Returns:
            The result of the cross attention between the latent model state and the observations.
        """  # noqa: E501
        x, obs, obs_mask = args
        x = x.contiguous()
        obs_shape = obs.shape
        B, GL, O, CS = obs.shape

        x_shape = x.shape

        obs = obs.reshape((-1,) + obs_shape[-2:])

        # Target sequence: [B x G x L x C]
        B, G, L, C = x.shape
        # Target sequence: B x G x LY x PY x LX x PX x C]
        LY = 15 // self.patch_height
        LX = 16 // self.patch_width
        x_patched = x.view(B, G, LY, self.patch_height, LX, self.patch_width, C)
        # Target sequence: B x G x LY x LX x PY x PX x C]
        x_patched = torch.permute(x_patched, (0, 1, 2, 4, 3, 5, 6)).contiguous()
        _, _, LY, LX, PY, PX, _ = x_patched.shape

        x_patched = x_patched.reshape((-1, PY * PX, C))
        q = self.q_layer(x_patched)
        q = q.reshape(-1, PY * PX, self.n_heads, self.features_obs).transpose(1, 2).contiguous()
        k = self.k_layer(obs)
        k = k.reshape(-1, O, self.n_heads, self.features_obs).transpose(1, 2).contiguous()
        v = self.v_layer(obs)
        v = v.reshape(-1, O, self.n_heads, self.features_obs).transpose(1, 2).contiguous()

        obs_mask = obs_mask.flatten(0, 1).to(dtype=torch.bool)
        obs_mask = obs_mask[:, None].repeat_interleave(q.shape[-2], 1)
        obs_mask = obs_mask[:, None].repeat_interleave(q.shape[1], 1)

        # Let us enforce either flash (A100+) or memory efficient attention.
        if version("torch") > "2.3.0":
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                # x [B, H, S, C//H]
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=~obs_mask, dropout_p=self.dropout
                )
        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                # x [B, H, S, C//H]
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=~obs_mask, dropout_p=self.dropout
                )

        # x [B, L, C]
        x = x.transpose(1, 2).contiguous().reshape(B * G * LY * LX, PY * PX, self.n_heads * self.features_obs)

        # x [B, L, C]
        x = self.w_layer(x)

        x = x.view(B, G, LY, LX, PY, PX, self.features_latent)
        x = torch.permute(x, (0, 1, 2, 4, 3, 5, 6)).contiguous()
        x = x.reshape(B, G, L, self.features_latent)

        return x
