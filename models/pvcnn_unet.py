import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from unet.unet_model import UNet
from .pvcnn2_ada import \
    create_pointnet2_sa_components, create_pointnet2_fp_modules, LinearAttention, create_mlp_components, SharedMLP
from render.gaussian_render import pts2render
from render.utils import preprocess_render

# the building block of encode and decoder for VAE

class PVCNN2Unet(nn.Module):
    """
        copied and modified from https://github.com/alexzhou907/PVD/blob/9747265a5f141e5546fd4f862bfa66aa59f1bd33/model/pvcnn_generation.py#L172
    """

    def __init__(self,
                 num_classes, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3,
                 input_dim=3,
                 width_multiplier=1,
                 voxel_resolution_multiplier=1,
                 time_emb_scales=1.0,
                 verbose=True,
                 condition_input=False,
                 point_as_feat=1, cfg={},
                 sa_blocks={}, fp_blocks={},
                 ):
        super().__init__()
        print('[Build Unet] extra_feature_channels={}, input_dim={}'.format(extra_feature_channels, input_dim))
        self.input_dim = input_dim

        self.sa_blocks = sa_blocks
        self.fp_blocks = fp_blocks
        self.point_as_feat = point_as_feat
        self.condition_input = condition_input
        assert extra_feature_channels >= 0
        self.time_emb_scales = time_emb_scales
        self.embed_dim = embed_dim
        if self.embed_dim > 0:  # has time embedding
            # for prior model, we have time embedding, for VAE model, no time embedding
            self.embedf = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )

        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = \
            create_pointnet2_sa_components(
                input_dim=input_dim,
                sa_blocks=self.sa_blocks,
                extra_feature_channels=extra_feature_channels,
                with_se=True,
                embed_dim=embed_dim,  # time embedding dim
                use_att=use_att, dropout=dropout,
                width_multiplier=width_multiplier,
                voxel_resolution_multiplier=voxel_resolution_multiplier,
                verbose=verbose, cfg=cfg
            )
        self.sa_layers = nn.ModuleList(sa_layers)

        self.global_att = None if not use_att else LinearAttention(channels_sa_features, 8, verbose=verbose)

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels + input_dim - 3
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            verbose=verbose, cfg=cfg
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, num_classes],  # was 0.5
            classifier=True, dim=2, width_multiplier=width_multiplier,
            cfg=cfg)
        self.classifier = nn.ModuleList(layers)

    def forward(self, inputs, **kwargs):
        # Input: coords: B3N
        coords = inputs[:, :self.input_dim, :].contiguous()
        features = inputs

        coords_list, in_features_list = [], []
        for i, sa_blocks in enumerate(self.sa_layers):

            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))

        in_features_list[0] = inputs[:, 3:, :].contiguous()
        if self.global_att is not None:
            features = self.global_att(features)

        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks((
                coords_list[-1 - fp_idx], coords,
                features,
                in_features_list[-1 - fp_idx]))


        for l in self.classifier:
            if isinstance(l, SharedMLP):
                features = l(features)
            else:
                features = l(features)
        return features


class Regressor(nn.Module):
    # encoder : B,N,3 -> B,N,2*D
    sa_blocks = [  # conv_configs, sa_configs
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (128, 128, 128))),
    ]
    fp_blocks = [
        ((128, 128), (128, 3, 8)),  # fp_configs, conv_configs
        ((128, 128), (128, 3, 8)),
        ((128, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, zdim, input_dim, scale_max, bg_color, args={}):
        super().__init__()
        self.scale_max = scale_max
        self.bg_color = bg_color
        self.zdim = zdim
        self.layers = PVCNN2Unet(zdim,
                                 embed_dim=0, use_att=1, extra_feature_channels=3,
                                 input_dim=3, cfg=args,
                                 sa_blocks=self.sa_blocks, fp_blocks=self.fp_blocks)

        self.input_dim = input_dim

        self.scale_head = nn.Sequential(
            nn.Conv1d(zdim, zdim // 2, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(zdim // 2, 3, 1, 1),
            nn.Softplus(beta=100))
        self.rot_head = nn.Sequential(
            nn.Conv1d(zdim, zdim // 2, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(zdim // 2, 4, 1, 1))
        self.opac_head = nn.Sequential(
            nn.Conv1d(zdim, zdim // 2, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(zdim // 2, 1, 1, 1),
            nn.Sigmoid())

    def forward(self, batch):
        x = batch['point_cloud']
        feat = self.layers(x).contiguous()  # B,32,N

        scale = self.scale_head(feat)
        scale = torch.clamp_max(scale, self.scale_max)
        rot = torch.nn.functional.normalize(self.rot_head(feat), dim=1)
        opacity = self.opac_head(feat)
        render_params = preprocess_render(batch)

        if self.bg_color == 1 :
            rgb_pred, feature_pred = pts2render(render_params, x, scale, opacity, rot, feat.clone(),
                                                bg_color=(torch.FloatTensor([1., 1., 1.])).to(x.device))
        else:
            rgb_pred, feature_pred = pts2render(render_params, x, scale, opacity, rot, feat.clone(),
                                                bg_color=(torch.FloatTensor([0., 0., 0.])).to(x.device))


        return rgb_pred, feature_pred


