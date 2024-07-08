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

class DtuDataset(Dataset):
    def __init__(self, phase, scene_dir, img_wh=(640,512)):
        self.phase = phase
        self.scene_dir = scene_dir

        if self.phase == 'train':
            scene_files = np.loadtxt(os.path.join(self.scene_dir, 'train.txt'), dtype=str).tolist()
            self.scene_files = scene_files * 2
            random.shuffle(self.scene_files)
        elif self.phase == 'val':
            self.scene_files = np.loadtxt(os.path.join(self.scene_dir, 'test.txt'), dtype=str).tolist()
        else:
            self.scene_files = np.loadtxt(os.path.join(self.scene_dir, 'test.txt'), dtype=str).tolist()


        self.W = img_wh[0]
        self.H = img_wh[1]

    def select_view(self,filename, W, H, i):
        # mask
        mask_path = os.path.join(filename, 'masks/{:08d}.png'.format(i))
        the_mask = Image.open(mask_path)
        the_mask = np.array(the_mask, dtype=np.float32) / 255.0
        the_mask = torch.FloatTensor(the_mask[:, :, np.newaxis])

        # read imgs
        image_path = os.path.join(filename, 'images/3/{:08d}.jpg'.format(i))
        img_ori = Image.open(image_path).convert('RGB')
        W_ori, H_ori = img_ori.size
        # rgb
        rgb = img_ori.resize((W, H), Image.Resampling.LANCZOS)
        rgb = torchvision.transforms.ToTensor()(rgb)
        rgb = rgb.permute(1, 2, 0) * the_mask
        # poses and extrinsic
        cam_path = os.path.join(filename, 'cams/{:08d}_cam.txt'.format(i))
        ixt, ext, _ = read_cam_file(cam_path)
        extrinsic = torch.FloatTensor(ext)
        W_scale = float(W) / float(W_ori)
        H_scale = float(H) / float(H_ori)
        intrinsic = torch.eye(4)
        intrinsic[:3, :3] = torch.FloatTensor(ixt)
        intrinsic[0, 0] = intrinsic[0, 0] * W_scale
        intrinsic[1, 1] = intrinsic[1, 1] * H_scale
        intrinsic[0, 2] = intrinsic[0, 2] * W_scale
        intrinsic[1, 2] = intrinsic[1, 2] * H_scale

        # point color cloud
        ply_path = os.path.join(filename, 'points_voxel_downsampled.ply')
        pcd_color = o3d.io.read_point_cloud(ply_path)

        points_raw = torch.FloatTensor(np.array(pcd_color.points, dtype=np.float32))
        features = torch.FloatTensor(np.array(pcd_color.colors, dtype=np.float32))

        pcd = torch.cat((points_raw, features), dim=1).permute(1, 0)

        return ply_path, pcd, filename, image_path, rgb, extrinsic, intrinsic

    def __len__(self):
        return len(self.scene_files)

    def __getitem__(self, i):
        # get scene
        filename = os.path.join(self.scene_dir, self.scene_files[i])
        idx = np.random.randint(low=0, high=49)
        ply_path, pcd, nerf_dir, image_path, rgb, extrinsic, intrinsic = self.select_view(filename, self.W, self.H, idx)


        return {
            "cam_extrinsics": extrinsic,
            "cam_intrinsics": intrinsic,
            "H": self.H,
            "W": self.W,
            "rgbs": rgb,
            "point_cloud": pcd,
            "ply_path": ply_path,
            "paths": nerf_dir,
            "filename": image_path,
            "znear": 0.5,
            "zfar": 100000
        }









