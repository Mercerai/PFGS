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

from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from pytorch_fid import fid_score
import torchvision
import imageio
import cv2
from datasets import *
from utils import *

from models.network import Net
####################################################################################
from skimage.metrics import structural_similarity as ssim_o
from skimage.metrics import peak_signal_noise_ratio as psnr_o
import lpips as lpips_o

import matplotlib
matplotlib.use('agg')
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
            batch_size=1,
            val_batch_size=1,
            train_num_workers=4,
            val_num_workers=2,
            max_epochs=200,
    ):
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)

        self.model = Net(args.zdim, args.input_dim, args.ckpt_stage1, args.scale_max, args.up_ratio, args.bg_color, args)
        self.measure = Measure(use_gpu=False)
        self.validation_outputs = []
        self.criterion = torch.nn.L1Loss()
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

    def compute_loss(self, rgbs_gt, out_list):
        rgbs_gt2 = torch.nn.functional.interpolate(rgbs_gt, scale_factor=0.5, mode='bilinear')
        rgbs_gt4 = torch.nn.functional.interpolate(rgbs_gt, scale_factor=0.25, mode='bilinear')

        pred0, pred1, pred2 = out_list[0], out_list[1], out_list[2]
        if pred0.shape[1] > 3:
            pred0, pred1, pred2 = pred0[:, :3], pred1[:, :3], pred2[:, :3]  # [1, 3, 128, 160]

        l1 = self.criterion(pred0, rgbs_gt4)
        l2 = self.criterion(pred1, rgbs_gt2)
        l3 = self.criterion(pred2, rgbs_gt)

        label_fft1 = torch.fft.fft2(rgbs_gt4, dim=[-1, -2])
        pred_fft1 = torch.fft.fft2(pred0, dim=[-1, -2])
        label_fft2 = torch.fft.fft2(rgbs_gt2, dim=[-1, -2])
        pred_fft2 = torch.fft.fft2(pred1, dim=[-1, -2])
        label_fft3 = torch.fft.fft2(rgbs_gt, dim=[-1, -2])
        pred_fft3 = torch.fft.fft2(pred2, dim=[-1, -2])

        f1 = self.criterion(pred_fft1, label_fft1)
        f2 = self.criterion(pred_fft2, label_fft2)
        f3 = self.criterion(pred_fft3, label_fft3)
        return (l1 + l2 + l3) + 0.15 * (f1 + f2 + f3)

    def forward(self, batch, is_training, batch_idx):
        rgbs_gt = batch['rgbs']  # (B,H,W,3)
        rgb_pred, rgb_feature_pred = self.model(batch)  # [b, c, h, w]
        return rgb_pred, rgbs_gt, rgb_feature_pred

    def training_step(self, batch, batch_idx):
        rgbs_prd, rgbs_gt, rgbs_feature_pred = self(batch, is_training=True, batch_idx=batch_idx)
        rgbs_gt = rgbs_gt.permute(0, 3, 1, 2)

        loss = self.criterion(rgbs_prd, rgbs_gt)

        for out_lis in rgbs_feature_pred:
            loss = loss + self.compute_loss(rgbs_gt, out_lis)

        if batch_idx % 100 == 0:
            psnr_nerf = psnr(rgbs_feature_pred[-1][-1], rgbs_gt)
            self.log("train/loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size,
                     sync_dist=True)
            self.log("train/loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size,
                     sync_dist=True)
            self.log("train/psnr_nerf", psnr_nerf, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.batch_size, sync_dist=True)
            self.log("train/lr", get_learning_rate(self.optimizer), on_epoch=True, logger=True,
                     batch_size=self.batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        rgbs_prd, rgbs_gt, rgbs_feature_pred = self(batch, is_training=False, batch_idx=batch_idx)

        rgbs_prd = rgbs_prd.permute(0, 2, 3, 1).cpu()  # [b, c, h, w] -> ([b, h, w, c]
        rgbs_gt = rgbs_gt.cpu()
        rgbs_feature_pred1 = torch.clamp(rgbs_feature_pred[-1][-1].permute(0, 2, 3, 1).cpu(), 0, 1)

        rgb_prd = rgbs_prd[0]  # BHw3
        rgb_gt = rgbs_gt[0]
        rgb_feature_pred1 = rgbs_feature_pred1[0]

        img_path = batch['filename'][0]
        name = img_path.split('/')[-3] + img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        stack_nerf = torch.cat([rgb_gt, rgb_prd, rgb_feature_pred1], dim=1)

        loss = 0.3 * ((rgbs_prd - rgbs_gt) ** 2).mean() + 0.7 * ((rgbs_feature_pred1 - rgbs_gt) ** 2).mean()

        psnr_nerf = psnr(rgbs_feature_pred1, rgbs_gt)

        rgbs_prd_numpy = np.clip((rgbs_feature_pred1.clone().detach().cpu().numpy()) * 255.0, 0, 255.0)
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
        stack_nerf = np.clip(stack_nerf, 0, 255)
        # cv2.putText(stack_nerf, "PSNR: %0.2f" % psnr_o, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(img_path, stack_nerf, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        output = {"loss": loss,
                  "psnr_nerf": psnr_nerf,
                  "ssim_nerf": ssim,
                  "lpips_nerf": lpips
                  }
        self.validation_outputs.append(output)

    def test_step(self, batch, batch_idx):
        rgbs_prd, rgbs_gt, rgbs_feature_pred = self(batch, is_training=False, batch_idx=batch_idx)

        rgbs_prd = rgbs_prd.permute(0, 2, 3, 1).cpu()  # [b, c, h, w] -> ([b, h, w, c]
        rgbs_gt = rgbs_gt.cpu()
        rgbs_feature_pred0 = torch.clamp(rgbs_feature_pred[-1][-1].permute(0, 2, 3, 1).cpu(), 0, 1)
        rgbs_feature_pred1 = torch.clamp(rgbs_feature_pred[-1][-1].permute(0, 2, 3, 1).cpu(), 0, 1)

        h, w, _ = rgbs_prd[0].shape
        rgb_prd = rgbs_prd[0]  # BHw3
        rgb_gt = rgbs_gt[0]
        rgb_feature_pred0 = rgbs_feature_pred0[0]
        rgb_feature_pred1 = rgbs_feature_pred1[0]

        img_path = batch['filename'][0]
        name = img_path.split('/')[-3] + '_' + img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        stack_nerf = torch.cat([rgb_gt, rgb_prd, rgb_feature_pred0, rgb_feature_pred1], dim=1)

        loss = 0.3 * ((rgbs_prd - rgbs_gt) ** 2).mean() + 0.7 * ((rgbs_feature_pred1 - rgbs_gt) ** 2).mean()

        psnr_nerf = psnr(rgbs_feature_pred1, rgbs_gt)

        rgbs_prd_numpy = np.clip((rgbs_feature_pred1.clone().detach().cpu().numpy()) * 255.0, 0, 255.0)
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

        scene = img_path.split('/')[-4]
        img_path = os.path.join('logs', self.exp_name, 'fid', f"{self.current_epoch:04d}", "vis", scene, f"{name}")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)

        stack_nerf = (stack_nerf * 255.0).cpu().numpy().astype(np.uint8)[..., [2, 1, 0]]
        stack_nerf = np.clip(stack_nerf, 0, 255)

        cv2.imwrite(img_path, stack_nerf, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        output = {"loss": loss,
                  "psnr_nerf": psnr_nerf,
                  "ssim_nerf": ssim,
                  "lpips_nerf": lpips
                  }
        self.validation_outputs.append(output)

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
        if os.path.exists(paths[0]):
            fid_value = fid_score.calculate_fid_given_paths(paths,
                                                            batch_size=50,
                                                            device='cuda:0',
                                                            dims=2048,
                                                            num_workers=0)
            self.log("test/fid", fid_value, on_epoch=True, prog_bar=True, logger=True, batch_size=self.val_batch_size,
                     sync_dist=True)


    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=args.lr)
        return [self.optimizer], []


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--scene_dir", type=str, default="./data/thuman2/", help="scene dir")
    pa.add_argument("--resume_path", type=str, help="resume ckpt path")
    pa.add_argument("--max_epochs", type=int, default=500, help="Max epochs")
    pa.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    pa.add_argument("--batch_size", type=int, default=1, help="batch size per GPU")
    pa.add_argument("--ngpus", type=int, default=1, help="num_gpus")
    pa.add_argument("--exp_name", type=str, default="t1", help="exp_name")
    pa.add_argument("--train_mode", type=str, default="train")
    pa.add_argument("--val_mode", type=str, default="val", help="test or val")
    pa.add_argument("--dataset", type=str, default='thuman2', help="dtu, scannet or thuman2")
    pa.add_argument('--img_wh', nargs="+", type=int, default=[640, 512], help='resolution (img_w, img_h) of the image')
    pa.add_argument("--feat_dim", type=int, default=16, help="the dimension of each feature")
    pa.add_argument("--bg_color", default=0)
    pa.add_argument("--zdim", default=16, type=int)
    pa.add_argument("--input_dim", default=3, type=int, help="to determine the coords")
    pa.add_argument("--up_ratio", default=2, type=int, help="up_ratio")
    pa.add_argument("--scale_max", default=0.01, type=float, help="up_ratio")
    pa.add_argument("--ckpt_stage1", default=None, type=str, help="stage1 checkpoint path")

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
        img_wh=args.img_wh)

    tb_logger = pl_loggers.TensorBoardLogger("logs/%s" % args.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="train/psnr_nerf",
        save_top_k=5,
        save_last=True,
        mode="max"
    )

    trainer = Trainer(max_epochs=args.max_epochs,
                      devices=num_devices,  # num_devices
                      accelerator="gpu",
                      strategy="ddp",
                      num_nodes=1,
                      logger=tb_logger,
                      callbacks=[checkpoint_callback],
                      num_sanity_val_steps=1
                      )

    if args.val_mode=='test':
        trainer.test(pl_module, ckpt_path=args.resume_path)
    else:
        trainer.fit(pl_module, ckpt_path=args.resume_path)
