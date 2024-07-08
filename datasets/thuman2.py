import os
import random
from PIL import Image
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from .common import *
from .data_utils import *


class THuman2Dataset(Dataset):
    def __init__(self, phase, scene_dir, img_wh):
        self.phase = phase
        self.scene_dir = scene_dir
        if self.phase == 'train':
            scene_files = np.loadtxt(os.path.join(self.scene_dir, 'train.txt'), dtype=str).tolist()
            self.scene_files = scene_files * 2
            random.shuffle(self.scene_files)
        elif self.phase == 'val':
            self.scene_files = np.loadtxt(os.path.join(self.scene_dir, 'test.txt'), dtype=str).tolist()
            self.scene_files = self.scene_files
        else:
            scene_files = np.loadtxt(os.path.join(self.scene_dir, 'test.txt'), dtype=str).tolist()
            self.scene_files = scene_files
            self.len = len(scene_files)
            for i in range(0, 35):
                self.scene_files = [*self.scene_files, *scene_files]

        self.W = img_wh[0]
        self.H = img_wh[1]

    def select_view(self, filename, W, H, i):
        image_path = os.path.join(filename, 'images', '{:04d}.png'.format(i))

        # read imgs
        img_ori = Image.open(image_path).convert('RGB')
        W_ori, H_ori = img_ori.size

        # rgb
        rgb = img_ori.resize((W, H), Image.Resampling.LANCZOS)
        rgb = torchvision.transforms.ToTensor()(rgb)  # (3, H, W)
        rgb = rgb.permute(1, 2, 0)  # (H, W, 3) RGB

        mask_path = os.path.join(filename, 'mask', '{:04d}.png'.format(i))
        mask = Image.open(mask_path)
        mask = torch.FloatTensor(np.array(mask, dtype=np.float32)[..., np.newaxis] / 255.0)
        rgb = rgb * mask

        npz_path = os.path.join(filename, 'cams', '{:04d}.npz'.format(i))

        pose = find_pose(npz_path)
        extrinsic = torch.FloatTensor(np.linalg.inv(pose))
        W_scale = float(W) / float(W_ori)
        H_scale = float(H) / float(H_ori)

        intrinsic = torch.eye(4)
        intrinsic[0, 0] = 711.1111111111111 * W_scale
        intrinsic[1, 1] = 711.1111111111111 * H_scale
        intrinsic[0, 2] = 255.5 * W_scale
        intrinsic[1, 2] = 255.5 * H_scale

        # point color cloud
        ply_path = os.path.join(filename, 'color_pc_36views_12w.ply')
        pcd_color = o3d.io.read_point_cloud(ply_path)
        points_raw = np.array(pcd_color.points, dtype=np.float32)
        features = np.array(pcd_color.colors, dtype=np.float32)

        points_raw = torch.FloatTensor(points_raw)
        features = torch.FloatTensor(features)
        pcd = torch.cat((points_raw, features), dim=1).permute(1, 0)

        return ply_path, pcd, filename, image_path, rgb, extrinsic, intrinsic, mask

    def __len__(self):
        return len(self.scene_files)

    def __getitem__(self, i):
        # get scene
        filename = os.path.join(self.scene_dir, self.scene_files[i])

        if self.phase == 'test':
            idx = i // self.len
            ply_path, pcd, nerf_dir, image_path, rgb, extrinsic, intrinsic, mask = self.select_view(filename, self.W, self.H, idx) # 512,512
        else:
            # random idx
            idx = np.random.randint(low=0, high=36)
            ply_path, pcd, nerf_dir, image_path, rgb, extrinsic, intrinsic, mask = self.select_view(filename, self.W, self.H, idx)

        return {
            "cam_extrinsics": extrinsic,
            "cam_intrinsics": intrinsic,
            "H": self.H,
            "W": self.W,
            "rgbs": rgb,
            "point_cloud": pcd,
            "ply_path": ply_path,
            "paths": nerf_dir,  # scene di
            "filename": image_path,
            "znear": 0.5,
            "zfar": 100000
        }




