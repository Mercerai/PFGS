import torch
import torchvision
import numpy as np
import cv2
from PIL import Image

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

class VoxTVLoss(torch.nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(VoxTVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        d_x = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]

        # normalize x
        x = (x-x.min())/(x.max()-x.min()+1e-8)

        count_d = self._tensor_size(x[:,:,1:,:,:])
        count_h = self._tensor_size(x[:,:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,:,1:])

        d_tv = torch.pow((x[:,:,1:,:,:]-x[:,:,:d_x-1,:,:]),2).sum()
        h_tv = torch.pow((x[:,:,:,1:,:]-x[:,:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,:,1:]-x[:,:,:,:,:w_x-1]),2).sum()

        return self.TVLoss_weight*(d_tv/count_d + h_tv/count_h + w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]*t.size()[4]


def visualize_depth(depth, mi=0.0, ma=5.0, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    depth = depth.clip(mi, ma)
    x = depth.detach().cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    # mi = np.min(x) # get minimum depth
    # ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = torchvision.transforms.ToTensor()(x_) # (3, H, W)
    return x_