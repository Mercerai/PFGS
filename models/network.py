import torch
import torch.nn as nn
from .pvcnn_unet import Regressor
from .mimo import build_net

class Net(nn.Module):
    # encoder : B,N,3 -> B,N,2*D
    def __init__(self, zdim, input_dim, ckpt_stage1, scale_max=0.01, recurrent=2, bg_color=0, args={}):
        super().__init__()
        self.pvcnn = Regressor(zdim, input_dim, scale_max, bg_color, args)
        if ckpt_stage1 != None :
            self.load_state_dict(torch.load(ckpt_stage1)["state_dict"])
        self.recurrent = recurrent
        self.render_networks = nn.ModuleList()
        self.render_networks.append(build_net("MIMO-Feature_out12"))
        for _ in range(recurrent-1):
            self.render_networks.append(build_net("MIMO-Feature"))
    def forward(self, batch):
        rgb_pred, feature_pred = self.pvcnn(batch)  # [1, 32, 256, 320]
        feature_pred = feature_pred[:, :9]
        feature_out = []
        render_inp = torch.cat([rgb_pred, feature_pred], dim=1)
        for i, l in enumerate(self.render_networks):
            render_out = l(render_inp)
            render_inp = render_out[-1]
            feature_out.append(render_out)
        return rgb_pred, feature_out







