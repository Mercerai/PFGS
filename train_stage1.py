# Copyright (c) NVIDIA Corporation.
# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import argparse
import numpy as np
import open3d as o3d

from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader

from pytorch_fid import fid_score, inception
import torchvision
import imageio
import cv2
from datasets import *
from utils import *

from models.pvcnn_unet import Regressor
####################################################################################
from skimage.metrics import structural_similarity as ssim_o
from skimage.metrics import peak_signal_noise_ratio as psnr_o
import lpips as lpips_o

import matplotlib
from render.gaussian_render import pts2render
from render.utils import preprocess_render

matplotlib.use('agg')
import matplotlib.pyplot as plt  # matplotlib.use('agg')必须在本句执行前运行
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

def convert_to(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1

class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips_o.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB, model=None):
        imgA = cv2.resize(imgA, (224, 224))
        imgB = cv2.resize(imgB, (224, 224))
        tA = convert_to(imgA).to(self.device)
        tB = convert_to(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB, gray_scale=True):
        if gray_scale:
            score, diff = ssim_o(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY),
                                 full=True, multichannel=True)
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        else:
            score, diff = ssim_o(imgA, imgB, full=True, multichannel=True)
        return score

    def psnr(self, imgA, imgB):
        ### print(imgA.shape, imgB.shape)
        psnr_val = psnr_o(imgA, imgB)
        return psnr_val

####################################################################################

class GSModule(LightningModule):
    def __init__(
            self,
            scene_dir,
            dataset,
            exp_name,
            train_mode,
            val_mode,
            img_wh,
            lr=1e-3,
            voxel_size=0.01,
            batch_size=1,
            val_batch_size=1,
            train_num_workers=4,
            val_num_workers=2,
            max_epochs=200,
            patch_size=64,
            scale_max=0.01,
    ):
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)

        self.pvcnn = Regressor(args.zdim, args.input_dim, args.scale_max, args)
        self.measure = Measure(use_gpu=False)
        self.validation_outputs = []
        ###################################################

    def train_dataloader(self):
        if self.dataset == 'dtu':
            self.train_dataset = DtuDataset(self.train_mode, scene_dir=self.scene_dir)
        elif self.dataset == 'thuman2':
            self.train_dataset = THuman2Dataset(self.train_mode, scene_dir=self.scene_dir, img_wh=self.img_wh)
        else:
            self.train_dataset = ScanNetDataset(self.train_mode, scene_dir=self.scene_dir, img_wh=self.img_wh)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=trivol_collate_fn,
            num_workers=self.train_num_workers,
            shuffle=True,
            pin_memory=False,
            drop_last=True
        )

    def val_dataloader(self):
        if self.dataset == 'dtu':
            self.val_dataset = DtuDataset(self.val_mode, scene_dir=self.scene_dir)
        elif self.dataset == 'thuman2':
            self.val_dataset = THuman2Dataset(self.val_mode, scene_dir=self.scene_dir, img_wh=self.img_wh)
        else:
            self.val_dataset = ScanNetDataset(self.val_mode, scene_dir=self.scene_dir, img_wh=self.img_wh)

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=trivol_collate_fn,
            num_workers=self.val_num_workers
        )

    def test_dataloader(self):
        if self.dataset == 'dtu':
            self.val_dataset = DtuDataset(self.val_mode, scene_dir=self.scene_dir)
        elif self.dataset == 'thuman2':
            self.val_dataset = THuman2Dataset(self.val_mode, scene_dir=self.scene_dir, img_wh=self.img_wh)
        else:
            self.val_dataset = ScanNetDataset(self.val_mode, scene_dir=self.scene_dir, img_wh=self.img_wh)

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=trivol_collate_fn,
            num_workers=self.val_num_workers
        )

    def forward(self, batch, is_training, batch_idx):
        rgbs_gt = batch['rgbs']  # (B,H,W,3)
        rgb_pred, _ = self.pvcnn(batch)

        return rgb_pred, rgbs_gt


    def training_step(self, batch, batch_idx):
        rgbs_prd, rgbs_gt = self(batch, is_training=True, batch_idx=batch_idx)
        rgbs_gt = rgbs_gt.permute(0,3,1,2)
        loss_l2 = torch.mean(((rgbs_prd - rgbs_gt) ** 2).mean(dim=1))
        loss = loss_l2

        if batch_idx % 100 == 0:
            psnr_nerf = psnr(rgbs_prd, rgbs_gt)
            self.log("train/loss_l2", loss_l2, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size,
                     sync_dist=True)
            self.log("train/loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size,
                     sync_dist=True)
            self.log("train/psnr_nerf", psnr_nerf, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.batch_size, sync_dist=True)
            self.log("train/lr", get_learning_rate(self.optimizer), on_epoch=True, logger=True,
                     batch_size=self.batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        rgbs_prd, rgbs_gt = self(batch, is_training=False, batch_idx=batch_idx)
        rgbs_prd = rgbs_prd.permute(0, 2, 3, 1)  # [1, 3, 256, 320] -> ([1, 256, 320, 3]

        # save image
        rgb_prd = rgbs_prd[0].cpu() # BHw3
        rgb_gt = rgbs_gt[0].cpu()

        img_path = batch['filename'][0]
        name = img_path.split('/')[-3] + img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        stack_nerf = torch.cat([rgb_gt, rgb_prd], dim=1)

        loss = torch.mean(((rgbs_prd - rgbs_gt) ** 2).mean(dim=1))

        psnr_nerf = psnr(rgbs_prd, rgbs_gt)

        rgbs_prd_numpy = (rgbs_prd.clone().detach().cpu().numpy()) * 255.0
        rgbs_gt_numpy = (rgbs_gt.clone().detach().cpu().numpy()) * 255.0
        batch_size = rgbs_prd_numpy.shape[0]
        ssim = 0
        lpips = 0
        for mm in range(batch_size):
            _, ssim_o, lpips_o = self.measure.measure(rgbs_prd_numpy[mm].astype(np.uint8),
                                                           rgbs_gt_numpy[mm].astype(np.uint8))
            ssim += ssim_o
            lpips += lpips_o
        ssim = ssim / batch_size
        lpips = lpips / batch_size
        ssim = torch.Tensor([ssim]).float()
        lpips = torch.Tensor([lpips]).float()

        img_path = os.path.join('logs', self.exp_name, 'fid', f"{self.current_epoch:04d}", "vis", f"{name}")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        stack_nerf = (stack_nerf * 255.0).cpu().numpy().astype(np.uint8)[..., [2, 1, 0]]
        cv2.imwrite(img_path, stack_nerf, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


        output = {"loss": loss,
                  "psnr_nerf": psnr_nerf,
                  "ssim_nerf": ssim,
                  "lpips_nerf": lpips
                  }
        self.validation_outputs.append(output)

    def on_validation_epoch_end(self):
        loss_val = torch.stack([out['loss'] for out in self.validation_outputs]).mean()
        psnr_nerf = torch.stack([out['psnr_nerf'] for out in self.validation_outputs]).mean()
        ssim_nerf = torch.stack([out['ssim_nerf'] for out in self.validation_outputs]).mean()
        lpips_nerf = torch.stack([out['lpips_nerf'] for out in self.validation_outputs]).mean()

        self.log("test/loss", loss_val, on_epoch=True, logger=True, batch_size=self.val_batch_size, sync_dist=True)
        self.log("test/psnr_nerf", psnr_nerf, on_epoch=True, prog_bar=True, logger=True, batch_size=self.val_batch_size,
                 sync_dist=True)
        self.log("test/ssim_nerf", ssim_nerf, on_epoch=True, logger=True, batch_size=self.val_batch_size,
                 sync_dist=True)
        self.log("test/lpips_nerf", lpips_nerf, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.val_batch_size, sync_dist=True)
        # torch empty cache
        # torch.cuda.empty_cache()

        self.validation_outputs.clear()

        paths = [os.path.join('logs', self.exp_name, 'fid', f"{self.current_epoch:04d}", 'real'),
                 os.path.join('logs', self.exp_name, 'fid', f"{self.current_epoch:04d}", 'fake')]
        if self.val_mode == "val" and os.path.exists(paths[0]):
            fid_value = fid_score.calculate_fid_given_paths(paths,
                                                            batch_size=50,
                                                            device='cuda:0',
                                                            dims=2048,
                                                            num_workers=0)
            self.log("test/fid", fid_value, on_epoch=True, prog_bar=True, logger=True, batch_size=self.val_batch_size,
                     sync_dist=True)
        if self.val_mode == "test":
            img_paths = glob.glob(os.path.join('logs', self.exp_name, 'vis', '*.jpg'))
            img_paths = sorted(img_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
            writer = imageio.get_writer(os.path.join('logs', self.exp_name, 'vis', 'demo.mp4'), fps=30)
            for im in img_paths:
                writer.append_data(imageio.imread(im))
            writer.close()
            assert True == False

    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=self.lr)
        return [self.optimizer], []  # [scheduler]


if __name__ == "__main__":
    pa = argparse.ArgumentParser()

    pa.add_argument("--scene_dir", type=str, default="./data/thuman2/", help="scene dir")
    pa.add_argument("--resume_path", type=str, help="resume ckpt path")
    pa.add_argument("--max_epochs", type=int, default=500, help="Max epochs")
    pa.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    pa.add_argument("--batch_size", type=int, default=1, help="batch size per GPU")
    pa.add_argument("--ngpus", type=int, default=1, help="num_gpus")
    pa.add_argument("--exp_name", type=str, default="22", help="num_gpus")
    pa.add_argument("--train_mode", type=str, default="train")
    pa.add_argument("--val_mode", type=str, default="val", help="test or val")
    pa.add_argument("--dataset", type=str, default='thuman2', help="dtu, scannet or thuman2")
    pa.add_argument('--img_wh', nargs="+", type=int, default=[512, 512],
                    help='resolution (img_w, img_h) of the image')
    pa.add_argument("--scale_max", default=0.01, type=float,help="gs scale")
    pa.add_argument("--bg_color", default=0)
    pa.add_argument("--zdim", default=16, type=int)
    pa.add_argument("--input_dim", default=3, type=int, help="to determine the coords")

    args = pa.parse_args()
    num_devices = min(args.ngpus, torch.cuda.device_count())
    print(f"Testing {num_devices} GPUs.")

    pl_module = GSModule(
        scene_dir=args.scene_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        exp_name=args.exp_name,
        train_mode=args.train_mode,
        val_mode=args.val_mode,
        img_wh=args.img_wh,
        scale_max=args.scale_max)

    tb_logger = pl_loggers.TensorBoardLogger("logs/%s" % args.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="train/psnr_nerf",
        save_top_k=5,
        save_last=True,
        mode="max"
    )

    trainer = Trainer(max_epochs=args.max_epochs,
                      devices=num_devices,  #num_devices
                      accelerator="gpu",
                      strategy="ddp",
                      num_nodes=1,
                      logger=tb_logger,
                      callbacks=[checkpoint_callback],
                      num_sanity_val_steps=1
                      )

    trainer.fit(pl_module, ckpt_path=args.resume_path)
