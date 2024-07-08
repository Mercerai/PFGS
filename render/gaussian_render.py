import cv2
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from datasets import *
from utils import *
from torch.utils.data import DataLoader


def render(cam, idx, pts_xyz, pts_rgb, feature, rotations, scales, opacity, bg_color):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=pts_xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(cam['FovX'][idx] * 0.5)
    tanfovy = math.tan(cam['FovY'][idx] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam['H'][idx]),
        image_width=int(cam['W'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cam['world_view_transform'][idx].cuda(),
        projmatrix=cam['full_proj_transform'][idx].cuda(),
        sh_degree=3,
        campos=cam['camera_center'][idx].cuda(),
        prefiltered=False,
        debug=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, rendered_feature, _ = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=pts_rgb,
        semantic_feature=feature,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    return rendered_image, rendered_feature


def pts2render(cam, pcd, gs_scaling, gs_opacity, gs_rotation, gs_feature, bg_color):
    bs = pcd.shape[0]

    render_novel_list = []
    render_feature_list = []
    for i in range(bs):
        xyz_i = pcd[i, :3, :].permute(1, 0)
        rgb_i = pcd[i, 3:6, :].permute(1, 0)
        feature_i = gs_feature[i].permute(1, 0)
        scale_i = gs_scaling[i].permute(1, 0)
        opacity_i = gs_opacity[i].permute(1, 0)
        rot_i = gs_rotation[i].permute(1, 0)
        render_novel_i, rendered_feature_i = render(cam, i, xyz_i, rgb_i, feature_i, rot_i, scale_i, opacity_i, bg_color=bg_color)

        render_novel_list.append(render_novel_i)
        render_feature_list.append(rendered_feature_i)
    return torch.stack(render_novel_list, dim=0), torch.stack(render_feature_list, dim=0)

