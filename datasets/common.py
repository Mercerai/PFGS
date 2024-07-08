import os
import glob
import random
from PIL import Image
import numpy as np
import cv2
import open3d as o3d

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from kornia import create_meshgrid
import torchvision
import imageio
import lpips



def get_ray_directions_opencv(W, H, fx, fy, cx, cy):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape) # (H, W, 3)

    return rays_o, rays_d


def trivol_collate_fn(list_data):
    cam_extrinsics = torch.stack([d["cam_extrinsics"] for d in list_data])
    cam_intrinsics = torch.stack([d["cam_intrinsics"] for d in list_data])
    rgb_batch = torch.stack([d["rgbs"] for d in list_data])
    pointclouds_batch = torch.stack([d["point_cloud"] for d in list_data])
    H_batch = [d["H"] for d in list_data]
    W_batch = [d["W"] for d in list_data]
    ply_path = [d["ply_path"] for d in list_data]
    paths = [d["paths"] for d in list_data]
    filenames = [d["filename"] for d in list_data]
    znear = torch.stack([torch.tensor(d["znear"]) for d in list_data])
    zfar = torch.stack([torch.tensor(d["zfar"]) for d in list_data])

    return {
        "cam_extrinsics": cam_extrinsics,
        "cam_intrinsics": cam_intrinsics,
        "rgbs": rgb_batch,
        "H": H_batch,
        "W": W_batch,
        "point_cloud": pointclouds_batch,
        "ply_path": ply_path,
        "paths":paths,
        "filename": filenames,
        "znear":znear,
        "zfar": zfar,
    }
