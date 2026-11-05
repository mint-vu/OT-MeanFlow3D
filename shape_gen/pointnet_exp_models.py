import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

from pvcnn_completion import CompletionEnc

###################################
#ResidualMLP
###################################
class ResidualBlock(nn.Module):
    """Pre-activation 2-layer MLP block with a residual skip."""
    def __init__(self, d):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.fc1   = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.fc2   = nn.Linear(d, d)

    def forward(self, x):
        # Pre-activation residual block
        h = self.norm1(x)
        h = F.gelu(h)
        h = self.fc1(h)

        h = self.norm2(h)
        h = F.gelu(h)
        h = self.fc2(h)
        return x + h  # residual skip

class ResidualMLPAE(nn.Module):
    """
    MLP with residual skips in the hidden stack.

    Args:
        din (int):   input dimension
        dout (int):  output dimension
        dhidden (int): width of hidden layers
        depth (int): number of residual blocks
    """
    def __init__(self, din, dout, dhidden, depth, din_context, dhidden_context, dout_context):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"

        self.in_proj  = nn.Linear(din, dhidden)
        self.blocks   = nn.ModuleList([ResidualBlock(dhidden) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dhidden)
        self.out_proj = nn.Linear(dhidden, dout)
        self.din_context = din_context

        # Optional input→output skip if din==dout (helps when you want identity bias)
        self.enable_io_skip = (din == dout)
        if self.enable_io_skip:
            # If you prefer a learned projection even when dims match,
            # replace with nn.Linear(din, dout, bias=False)
            self.register_buffer("_io_identity", torch.eye(1))  # dummy to keep state_dict stable

    def forward(self, x, t, r, context, num_graphs=None, is_eval=False):
        """
        x: Tensor of shape (..., din)
        returns: Tensor of shape (..., dout)
        """
        N = x.shape[0] // num_graphs
        x = torch.cat([x, t[:, None], r[:, None], context], dim=-1)

        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        h = F.gelu(h)
        y = self.out_proj(h)

        if self.enable_io_skip:
            # pure identity skip when din==dout
            y = y + x

        return y


class ResidualMLP(nn.Module):
    """
    MLP with residual skips in the hidden stack.

    Args:
        din (int):   input dimension
        dout (int):  output dimension
        dhidden (int): width of hidden layers
        depth (int): number of residual blocks
    """
    def __init__(self, din, dout, dhidden, depth, din_context, dhidden_context, dout_context, uncond):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"

        self.in_proj  = nn.Linear(din, dhidden)
        self.blocks   = nn.ModuleList([ResidualBlock(dhidden) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dhidden)
        self.out_proj = nn.Linear(dhidden, dout)
        self.din_context = din_context

        self.context_encoder = CompletionEnc(num_classes=din_context, sv_points=600, embed_dim=1, use_att=True,
                dropout=0.1, extra_feature_channels=0,
                width_multiplier=1, voxel_resolution_multiplier=1)

        self.uncond = uncond

        # Optional input→output skip if din==dout (helps when you want identity bias)
        self.enable_io_skip = (din == dout)
        if self.enable_io_skip:
            # If you prefer a learned projection even when dims match,
            # replace with nn.Linear(din, dout, bias=False)
            self.register_buffer("_io_identity", torch.eye(1))  # dummy to keep state_dict stable

    def forward(self, x, t, r, context, num_graphs=None, is_eval=False):
        """
        x: Tensor of shape (..., din)
        returns: Tensor of shape (..., dout)
        """
        if is_eval and self.uncond:
            latent = torch.randn(num_graphs, self.din_context).to(x.device)
        else:
            latent = self.context_encoder(context)

        N = x.shape[0] // num_graphs
        context = latent[:, None, :].expand(-1, N, -1).reshape(-1, latent.shape[-1])
        x = torch.cat([x, t[:, None], r[:, None], context], dim=-1)

        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        h = F.gelu(h)
        y = self.out_proj(h)

        if self.enable_io_skip:
            # pure identity skip when din==dout
            y = y + x

        if self.uncond and not is_eval:
            return y, latent
        return y

class ResidualMLP2(nn.Module):
    """
    MLP with residual skips in the hidden stack.

    Args:
        din (int):   input dimension
        dout (int):  output dimension
        dhidden (int): width of hidden layers
        depth (int): number of residual blocks
    """
    def __init__(self, din, dout, dhidden, depth):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"

        self.in_proj  = nn.Linear(din, dhidden)
        self.blocks   = nn.ModuleList([ResidualBlock(dhidden) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dhidden)
        self.out_proj = nn.Linear(dhidden, dout)

        # Optional input→output skip if din==dout (helps when you want identity bias)
        self.enable_io_skip = (din == dout)
        if self.enable_io_skip:
            # If you prefer a learned projection even when dims match,
            # replace with nn.Linear(din, dout, bias=False)
            self.register_buffer("_io_identity", torch.eye(1))  # dummy to keep state_dict stable

    def forward(self, x, t, r, context=None, num_graphs=None, is_eval=None):
        """
        x: Tensor of shape (..., din)
        returns: Tensor of shape (..., dout)
        """
        x = torch.cat([x, t[:, None], r[:, None]], dim=-1)

        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        h = F.gelu(h)
        y = self.out_proj(h)

        if self.enable_io_skip:
            # pure identity skip when din==dout
            y = y + x
        return y

class PointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE, self).__init__()

        self.latent_size = latent_size
        self.point_size = point_size

        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = torch.nn.Conv1d(256, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, self.latent_size, 1)

        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(self.latent_size)

        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256,256)
        self.dec3 = nn.Linear(256,self.point_size*3)

    def encoder(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x))
        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, self.latent_size)
        return x

    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
