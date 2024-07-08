import os
import glob
import random
from PIL import Image
import numpy as np
import cv2
import imageio
import open3d as o3d

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from .common import *
from .data_utils import *


class ScanNetDataset(Dataset):
    def __init__(self, phase, scene_dir, img_wh):
        self.phase = phase
        self.scene_dir = scene_dir
        scene_files = glob.glob(os.path.join(scene_dir, '*/*vh_clean_2.ply'))

        if self.phase == 'train':
            scene_files = sorted(scene_files)
            del scene_files[1200:]
            scene_files = scene_files * 2
            random.shuffle(scene_files)
            self.scene_files = scene_files
        elif self.phase == 'val':
            scene_files = sorted(scene_files)
            self.scene_files = scene_files[1200:]
        else:
            scene_files = sorted(scene_files)
            self.scene_files = scene_files[1200:]

        self.W = img_wh[0]
        self.H = img_wh[1]

    def select_view(self, filename, W, H):
        nerf_dir = os.path.dirname(filename)

        image_paths = np.loadtxt(os.path.join(nerf_dir, 'images.txt'), dtype=str).tolist()

        image_path = random.choice(image_paths)

        # read imgs
        image_path = os.path.join(nerf_dir, 'color', image_path)
        img_ori = Image.open(image_path).convert('RGB')
        W_ori, H_ori = img_ori.size

        # rgb
        rgb = img_ori.resize((W, H), Image.LANCZOS)
        rgb = torchvision.transforms.ToTensor()(rgb)  # (3, H, W)
        rgb = rgb.permute(1, 2, 0)  # (H, W, 3) RGB

        # poses and extrinsic
        pose_path = image_path.replace('color', 'pose').replace('.jpg', '.txt')
        pose = torch.FloatTensor(np.loadtxt(pose_path))
        extrinsic = torch.FloatTensor(np.linalg.inv(pose))
        W_scale = float(W) / float(W_ori)
        H_scale = float(H) / float(H_ori)
        intrinsic_path = os.path.join(nerf_dir, 'intrinsic', 'intrinsic_color.txt')
        intrinsic = np.loadtxt(intrinsic_path)
        intrinsic[0, 0] = intrinsic[0, 0] * W_scale
        intrinsic[1, 1] = intrinsic[1, 1] * H_scale
        intrinsic[0, 2] = intrinsic[0, 2] * W_scale
        intrinsic[1, 2] = intrinsic[1, 2] * H_scale
        intrinsic = torch.FloatTensor(intrinsic)

        # point color cloud
        ply_path = filename
        pcd_color = o3d.io.read_point_cloud(filename)

        points_raw = np.array(pcd_color.points, dtype=np.float32)  # n,3
        features = np.array(pcd_color.colors, dtype=np.float32)
        points_raw, features = frustrum_clean(points_raw, features, intrinsic.numpy(), extrinsic.numpy(), W_ori)
        points_raw = torch.FloatTensor(points_raw)
        features = torch.FloatTensor(features)

        pcd = torch.cat((points_raw, features), dim=1).permute(1, 0)
        return ply_path, pcd, nerf_dir, image_path, rgb, extrinsic, intrinsic

    def __len__(self):
        return len(self.scene_files)

    def __getitem__(self, i):
        filename = self.scene_files[i]
        while 1:
            ply_path, pcd, nerf_dir, image_path, rgb, extrinsic, intrinsic = self.select_view(filename, self.W, self.H)
            if pcd.shape[1] > 5000 and pcd.shape[1] < 240000:
                break

        return {
            # "train_cameras": train_cam_infos,
            "cam_extrinsics": extrinsic,
            "cam_intrinsics": intrinsic,
            "H": self.H,
            "W": self.W,
            "rgbs": rgb, #rgb
            "point_cloud": pcd,
            "ply_path": ply_path,
            "paths": nerf_dir,  # scene di
            "filename": image_path,
            "znear": 0.5,
            "zfar": 100000
        }




