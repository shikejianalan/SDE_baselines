import torch
import numpy as np
import functools
from torch import nn
from torch import Tensor
from samplers import samples_gen

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
    Returns:
      The standard deviation.
    """
    # t = torch.tensor(t, device=conf.device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.
    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
    Returns:
      The vector of diffusion coefficients.
    """
    return sigma ** t


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

"""
class AffordSDE(nn.Module):
    def __init__(self, conf) -> None:
        super(AffordSDE, self).__init__()
        self.margin_fn = functools.partial(marginal_prob_std, conf=self.conf)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, conf=self.conf)

    def forward(self, conf, x_p, f_p) -> Tensor:
        pass
"""

class CondResLayer(nn.Module):
    def __init__(self, feat_len=128, cond_len=128, time_embed_len=128) -> None:
        super(CondResLayer, self).__init__()
        self.layer1 = nn.Linear(feat_len + cond_len + time_embed_len, feat_len)
        self.layer2 = nn.Linear(feat_len, feat_len)
    def forward(self, x_first, f_p, embed_time):
        x = torch.cat([x_first, f_p, embed_time], dim=-1)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x += x_first
        return x

class SDE_Model(nn.Module):
    def __init__(self, conf, sigma, snr, num_steps, input_dim, cond_res_num=5, feat_len=128, cond_len=128, time_embed_len=128) -> None:
        super(SDE_Model, self).__init__()
        if cond_res_num < 1:
            print("num_layer cannot smaller than 1!")
            raise ValueError
        self.conf = conf
        self.sigma = sigma
        self.snr = snr
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.cond_res_num = cond_res_num
        self.feat_len = feat_len
        self.cond_len = cond_len
        self.margin_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

        """model structure"""
        self.t_embed = nn.Sequential(GaussianFourierProjection(embed_dim=time_embed_len),
                                   nn.Linear(time_embed_len, time_embed_len))
        # self.d1 = Dense(embed_dim, embed_dim)
        self.act = lambda x: x * torch.sigmoid(x)
        self.first_layer = nn.Linear(input_dim + cond_len + time_embed_len, feat_len)
        self.cond_res_layers = nn.ModuleList([CondResLayer(feat_len, cond_len, time_embed_len) for _ in range(self.cond_res_num)])
        self.last_layer = nn.Linear(feat_len + cond_len + time_embed_len, input_dim)


    def forward(self, x_p, f_p, t) -> Tensor:
        """input:
            x_p: B (BN) x input_dim
            f_p: B (BN) x feature_dim
            t: B (BN) x 1
        """
        embed_time = self.t_embed(t) # dim: B (BN) x time_embed_len
        x = torch.cat([x_p, f_p, embed_time], dim=-1)
        x = self.first_layer(x)
        x = torch.relu(x)
        for layer in self.cond_res_layers:
            x = layer(x, f_p, embed_time)
            x = torch.relu(x)
        x = torch.cat([x, f_p, embed_time], dim=-1)
        x = self.last_layer(x)
        x = x / (self.margin_fn(t)[:, None] + 1e-7)
        return x

    def train_one_batch_loss(self, x_p, f_p, eps=1e-5):
        batch_size = x_p.size(0)
        random_t = torch.rand(batch_size, device=self.conf.device) * (1. - eps) + eps
        z = torch.randn_like(x_p)
        std = self.margin_fn(random_t)[:, None]
        x_p += z * std

        # Get score
        score = self.forward(x_p=x_p, f_p=f_p, t=random_t)
        loss = torch.sum((score * std + z) ** 2, dim=-1)
        return loss

    def sample_one_batch(self, f_p, x_init=None, eps=1e-3):
        return samples_gen(self, f_p, x_init, eps)

class AssembleModel(nn.Module):
    def __init__(self, conf) -> None:
        super(AssembleModel, self).__init__()
        self.conf = conf
        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': conf.cond_len})
        self.affordSDE = SDE_Model(conf=conf, sigma=conf.affordSDE_sigma, snr=conf.affordSDE_snr, num_steps=conf.affordSDE_num_steps, input_dim=conf.affordSDE_input_dim, 
                        cond_res_num=conf.affordSDE_cond_res_num, feat_len=conf.affordSDE_feat_len, 
                        cond_len=conf.cond_len, time_embed_len=conf.affordSDE_time_embed_len)
        self.poseSDE = SDE_Model(conf=conf, sigma=conf.poseSDE_sigma, snr=conf.poseSDE_snr, num_steps=conf.poseSDE_num_steps, input_dim=conf.poseSDE_input_dim, 
                        cond_res_num=conf.poseSDE_cond_res_num, feat_len=conf.poseSDE_feat_len, 
                        cond_len=conf.cond_len, time_embed_len=conf.poseSDE_time_embed_len)
    
    def get_whole_feats(self, pcs):
        pcs = pcs.repeat(1, 1, 2)
        # push through PointNet++
        return self.pointnet2(pcs)
    
    def forward(self, pcs, dirs1, dirs2, gt_result):
        whole_feats = self.get_whole_feats(pcs)

        # feats for the interacting points
        f_p = whole_feats[:, :, 0]  # B x F

        input_s6d = torch.cat([dirs1, dirs2], dim=1)

        afford_loss = self.affordSDE.train_one_batch_loss(gt_result, f_p)
        pose_loss = self.poseSDE.train_one_batch_loss(input_s6d, f_p)

        return {"afford_loss": afford_loss, "pose_loss": pose_loss}
    
    def get_affordance(self, whole_feats):
        f_ps = whole_feats.permute(0, 2, 1)
        batch_size = f_ps.size(0)
        num_point = f_ps.size(1)
        f_ps = f_ps.reshape(batch_size*num_point, -1)

        # Sampling affordance
        return self.affordSDE.sample_one_batch(f_ps)


