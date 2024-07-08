import torch
import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def getWorld2View2(R, t, translate=torch.FloatTensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.inverse(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.inverse(C2W)
    return Rt


def getProjectionMatrix(znear, zfar, K, h, w):
    near_fx = znear / K[0, 0]
    near_fy = znear / K[1, 1]
    left = - (w - K[0, 2]) * near_fx
    right = K[0, 2] * near_fx
    bottom = (K[1, 2] - h) * near_fy
    top = K[1, 2] * near_fy

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def preprocess_render(batch):
    H, W = batch['H'][0], batch['W'][0]
    extrs = batch["cam_extrinsics"]
    intrs = batch["cam_intrinsics"]
    znear, zfar = 0.5, 10000
    B = extrs.shape[0]

    proj_mat = [getProjectionMatrix(znear, zfar, intrs[i], H, W).transpose(0, 1) for i in range(B)]
    world_view_transform = [
        getWorld2View2(extrs[i][:3, :3].reshape(3, 3).transpose(1, 0), extrs[i][:3, 3]).transpose(0, 1) for i in
        range(B)]
    proj_mat = torch.stack(proj_mat, dim=0)

    world_view_transform = torch.stack(world_view_transform, dim=0)

    full_proj_transform = (world_view_transform.bmm(proj_mat))
    camera_center = world_view_transform.inverse()[:, 3, :3]

    FovX = [torch.FloatTensor([focal2fov(intrs[i][0, 0], W)]) for i in range(B)]

    # print("111",FovX[0])
    FovY = [torch.FloatTensor([focal2fov(intrs[i][1, 1], H)]) for i in range(B)]

    return {"projection_matrix": proj_mat,
            "world_view_transform": world_view_transform,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "H": torch.ones(B) * H,
            "W": torch.ones(B) * W,
            "FovX": torch.stack(FovX, dim=0),
            "FovY": torch.stack(FovY, dim=0)
            }

