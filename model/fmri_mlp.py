# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from functools import partial

import pydantic
import torch
from torch import nn

try:
    from diffusers.models.vae import Decoder
except:
    from diffusers.models.autoencoders.vae import Decoder


logger = logging.getLogger(__name__)


class Mean(nn.Module):
    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=self.keepdim)


class SubjectLayers(nn.Module):
    """Per subject linear layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int,
        init_id: bool = False,
        mode: tp.Literal["gather", "for_loop"] = "gather",
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5
        self.mode = mode

    def forward(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        N, C, D = self.weights.shape

        if self.mode == "gather":
            weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
            out = torch.einsum("bct,bcd->bdt", x, weights)
        elif self.mode == "for_loop":
            B, _, T = x.shape
            out = torch.empty((B, D, T), device=x.device, dtype=x.dtype)
            for subject in subjects.unique():
                mask = subjects.reshape(-1) == subject
                id_weights = subject 
                out[mask] = torch.einsum("bct,cd->bdt", x[mask], self.weights[id_weights])
        else:
            raise NotImplementedError()

        return out

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class DeeperSubjectLayers(nn.Module):
    """Per subject linear layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int,
        init_id: bool = False,
        mode: tp.Literal["gather", "for_loop"] = "gather",
        mlp_n_blocks: int = 4,
    ):
        super().__init__()
        self.mlp_n_blocks = mlp_n_blocks
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5
        self.mode = mode

        norm_func = partial(nn.LayerNorm, normalized_shape=out_channels)
        act_fn = nn.GELU
        act_and_norm = (norm_func, act_fn)
        self.mlp = nn.ModuleList(
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(out_channels, out_channels),
                        *[item() for item in act_and_norm],
                        nn.Dropout(0.15),
                    )
                    for _ in range(mlp_n_blocks)
                ]
            )
            for _ in range(n_subjects)
        )

    def forward(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        N, C, D = self.weights.shape
        assert subjects.max() < N, (
            f"Subject index ({subjects.max()}) too high for number of subjects used to initialize"
            f" the weights ({N})."
        )

        if self.mode == "gather":
            weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
            out = torch.einsum("bct,bcd->bdt", x, weights)
        elif self.mode == "for_loop":
            B, _, T = x.shape
            out = torch.empty((B, D, T), device=x.device, dtype=x.dtype)
            for subject in subjects.unique():
                mask = subjects.reshape(-1) == subject
                out_int = torch.einsum("bct,cd->bdt", x[mask], self.weights[subject])
                out_int = out_int.permute(0, 2, 1)
                mlp = self.mlp[subject]
                residual = out_int
                for res_block in range(self.mlp_n_blocks):
                    out_int = mlp[res_block](out_int)
                    out_int += residual
                    residual = out_int
                out[mask] = out_int.permute(0, 2, 1)
        else:
            raise NotImplementedError()

        return out

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class FmriMLPConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["FmriMLP"] = "FmriMLP"  # type: ignore

    hidden: int = 4096
    n_blocks: int = 4
    norm_type: str = "ln"
    act_first: bool = False

    n_repetition_times: int = 1
    time_agg: tp.Literal["in_mean", "in_linear", "out_mean", "out_linear"] = "out_linear"

    # TR embeddings
    use_tr_embeds: bool = False
    tr_embed_dim: int = 16
    use_tr_layer: bool = False

    # Control output size explicitly
    out_dim: int | None = None

    # Subject-specific settings
    subject_layers: bool = False
    deep_subject_layers: bool = False
    n_subjects: int = 20
    subject_layers_dim: tp.Literal["input", "hidden"] = "hidden"
    subject_layers_id: bool = False

    # blurry recons
    blurry_recon: bool = False

    native_fmri_space: bool = False

    def build(self, n_in_channels: int, n_outputs: int | None) -> nn.Module:
        if n_outputs is None and self.out_dim is None:
            raise ValueError("One of n_outputs or config.out_dim must be set.")
        return FmriMLP(
            in_dim=n_in_channels,
            out_dim=self.out_dim if n_outputs is None else n_outputs,
            config=self,
        )


class FmriMLP(nn.Module):
    """Residual MLP adapted from [1].

    See https://github.com/MedARC-AI/fMRI-reconstruction-NSD/blob/main/src/models.py#L171

    References
    ----------
    [1] Scotti, Paul, et al. "Reconstructing the mind's eye: fMRI-to-image with contrastive
        learning and diffusion priors." Advances in Neural Information Processing Systems 36
        (2024).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: FmriMLPConfig | None = None,
    ):
        super().__init__()

        # Temporal aggregation
        self.in_time_agg, self.out_time_agg = None, None
        self.n_repetition_times = config.n_repetition_times
        self.blurry_recon = config.blurry_recon
        if config.time_agg == "in_mean":
            self.in_time_agg = Mean(dim=2, keepdim=True)
            self.n_repetition_times = 1
        elif config.time_agg == "in_linear":
            self.in_time_agg = nn.Linear(self.n_repetition_times, 1)

            self.n_repetition_times = 1
        elif config.time_agg == "out_mean":
            self.out_time_agg = Mean(dim=2)
        elif config.time_agg == "out_linear":
            self.out_time_agg = nn.Linear(self.n_repetition_times, 1)

        norm_func = (
            partial(nn.BatchNorm1d, num_features=config.hidden)
            if config.norm_type == "bn"
            else partial(nn.LayerNorm, normalized_shape=config.hidden)
        )
        act_fn = partial(nn.ReLU, inplace=True) if config.norm_type == "bn" else nn.GELU
        act_and_norm = (act_fn, norm_func) if config.act_first else (norm_func, act_fn)

        self.proj2flat = None
        if config.native_fmri_space:
            self.proj2flat = nn.Sequential(
                *[
                    nn.Conv3d(1, 8, 3, stride=2, padding=1),
                    nn.LayerNorm([39, 47, 40]),
                    nn.GELU(),
                    nn.Conv3d(8, 8, 3, stride=2, padding=1),
                    nn.LayerNorm([20, 24, 20]),
                    nn.GELU(),
                    nn.Conv3d(8, 4, 2, stride=1, padding=1),
                    nn.LayerNorm([21, 25, 21]),
                    nn.GELU(),
                    nn.Conv3d(4, 2, 2, stride=1, padding=1),
                ]
            )

        # Subject-specific linear layer
        self.subject_layers = None
        if config.subject_layers:
            dim = {"hidden": config.hidden, "input": in_dim}[config.subject_layers_dim]
            if not config.deep_subject_layers:
                self.subject_layers = SubjectLayers(
                    in_dim,
                    dim,
                    config.n_subjects,
                    config.subject_layers_id,
                    mode="for_loop",
                )
            else:
                self.subject_layers = DeeperSubjectLayers(
                    in_dim,
                    dim,
                    config.n_subjects,
                    config.subject_layers_id,
                    mode="for_loop",
                )

            in_dim = dim

        # TR embeddings
        self.tr_embeddings = None
        if config.use_tr_embeds:
            self.tr_embeddings = nn.Embedding(
                self.n_repetition_times, config.tr_embed_dim
            )
            in_dim += config.tr_embed_dim

        if config.use_tr_layer:
            # depthwise convolution
            # Each TR is passed to a (distinct) linear layer Linear(in_dim, config.hidden)
            self.lin0 = nn.Conv1d(
                in_channels=self.n_repetition_times,
                out_channels=self.n_repetition_times * config.hidden,
                kernel_size=in_dim,
                groups=self.n_repetition_times,
                bias=True,
            )
        else:
            self.lin0 = nn.Linear(in_dim, config.hidden)
        self.post_lin0 = nn.Sequential(
            *[item() for item in act_and_norm], nn.Dropout(0.5)
        )

        self.n_blocks = config.n_blocks
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden, config.hidden),
                    *[item() for item in act_and_norm],
                    nn.Dropout(0.15),
                )
                for _ in range(config.n_blocks)
            ]
        )
        if not self.blurry_recon:
            self.lin1 = nn.Linear(config.hidden, out_dim)
        else:
            self.blin1 = nn.Linear(config.hidden, 4 * 28 * 28)
            self.bdropout = nn.Dropout(0.3)
            self.bnorm = nn.GroupNorm(1, 64)
            self.bupsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=[
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                ],
                block_out_channels=[32, 64, 128],
                layers_per_block=1,
            )


    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        channel_positions: torch.Tensor | None = None,  # Unused
    ) -> torch.Tensor:

        if self.proj2flat is not None:
            bs = x.size(0)
            x = x.permute(0, 4, 1, 2, 3)  ## to have (B,T,D,H,W)
            x = x.reshape(
                x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]
            )  # (B*T,D,H,W)
            x = x[:, None]  # (B*T,1, D,H,W) to have C = 1 for 3d conv
            x = self.proj2flat(x)  # (B*T, 2, 22,26,22)
            x = x.reshape(bs, self.n_repetition_times, -1)  # (B, T, F)
            x = x.permute(0, 2, 1)  # (B, F, T)
        else:
            x = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, F, T)

        if self.in_time_agg is not None:
            x = self.in_time_agg(x)  # (B, F, 1)

        B, F, T = x.shape

        assert (
            T == self.n_repetition_times
        ), f"Mismatch between expected and provided number TRs: {T} != {self.n_repetition_times}"

        if self.subject_layers is not None:
            x = self.subject_layers(x, subject_ids)
        x = x.permute(0, 2, 1)  # (B, F, T) -> (B, T, F)

        if self.tr_embeddings is not None:
            embeds = self.tr_embeddings(torch.arange(T, device=x.device))
            embeds = torch.tile(embeds, dims=(B, 1, 1))
            x = torch.cat([x, embeds], dim=2)

        x = self.lin0(x).reshape(B, T, -1)  # (B, T, F) -> (B, T * F, 1) -> (B, T, F)
        x = self.post_lin0(x)

        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x

        x = x.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)
        if self.out_time_agg is not None:
            x = self.out_time_agg(x)  # (B, F, 1)
        x = x.flatten(1)  # Ensure 2D

        if self.blurry_recon:
            b = self.blin1(x)
            b = self.bdropout(b)
            b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
            b = self.bnorm(b)
            x_final = self.bupsampler(b)
        else:
            x_final = self.lin1(x)


        return {
            "MSELoss": x_final
        }
