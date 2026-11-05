import torch
import torch.nn as nn
from .time_sampler import sample_two_timesteps
from .ema import init_ema, update_ema_net
from .optimal_transport import OTPlanSampler
import time

def reshape_time(tensor, target_shape):
    dims = len(target_shape)
    return tensor.view(-1, *([1] * (dims - 1)))

class MeanFlow(nn.Module):
    def __init__(self, arch, args, net_configs):
        super(MeanFlow, self).__init__()
        self.net = arch(**net_configs)
        self.args = args
        
        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))
        
        self.net_ema = init_ema(self.net, arch(**net_configs), args.ema_decay)
        
        # maintain extra ema nets
        self.ema_decays = args.ema_decays
        for i, ema_decay in enumerate(self.ema_decays):
            self.add_module(f"net_ema{i + 1}", init_ema(self.net, arch(**net_configs), ema_decay))
        
        # Initialize OT sampler if enabled
        self.use_ot = getattr(args, 'use_ot', False)

        if self.use_ot:
            self.ot_sampler = OTPlanSampler(
                method=getattr(args, 'ot_method', 'exact'),
                reg=getattr(args, 'ot_reg', 0.05),
                numItermax=getattr(args, 'ot_numItermax', 100),
                stopThr=getattr(args, 'ot_stopThr', 1e-9)
            )
    
    def update_ema(self):
        self.num_updates += 1
        num_updates = self.num_updates
        update_ema_net(self.net, self.net_ema, num_updates)
        # update extra ema
        for i in range(len(self.ema_decays)):
            update_ema_net(self.net, self._modules[f"net_ema{i + 1}"], num_updates)

    def forward_with_loss(self, x, aug_cond,e=None, context=None, num_graphs=None):

        device = x.device
        if e is None:
            e = torch.randn_like(x).to(device)
        else:
            e=e.to(device)
        
        
        # Apply OT if enabled
        s_time = time.time()
        if self.use_ot:
            if num_graphs is None:
                x_ot, e_ot = self.ot_sampler.sample_plan(x, e, replace=True)
            else:
                x_batch = x.view(num_graphs, -1, x.shape[-1])
                e_batch = e.view(num_graphs, -1, e.shape[-1])
                x_ot = []
                e_ot = []
                for g in range(num_graphs):
                    x_ot_batch, e_ot_batch = self.ot_sampler.sample_plan(x_batch[g], e_batch[g], replace=True)
                    x_ot.append(x_ot_batch)
                    e_ot.append(e_ot_batch)
                x_ot = torch.cat(x_ot, dim=0)
                e_ot = torch.cat(e_ot, dim=0)

        else:
            # Use original random pairing
            x_ot, e_ot = x, e
        e_time = time.time()
        ot_time = e_time - s_time

        # Sample time steps (after OT pairing)
        t, r = sample_two_timesteps(self.args, num_samples=x_ot.shape[0], device=device)
        t, r = reshape_time(t, x_ot.shape), reshape_time(r, x_ot.shape)

        # Interpolation using OT-paired samples
        z = (1 - t) * x_ot + t * e_ot

        dtdt = torch.ones_like(t, device=t.device)
        drdt = torch.zeros_like(r, device=t.device)
        
        with torch.amp.autocast("cuda", enabled=False):

            def u_func(z, t, r):
                h = t - r
                u = self.net(z, t.view(-1), h.view(-1), context=context, num_graphs=num_graphs)
                return u

            v = u_func(z, t, t)
            u_pred, dudt = torch.autograd.functional.jvp(u_func, (z, t, r), v=(v, dtdt, drdt), create_graph=True)

            V = u_pred + (t - r)  * dudt.detach()
            loss = (V - (e_ot - x_ot))**2

            sum_dims = tuple(range(1, loss.dim()))
            loss = loss.sum(dim=sum_dims)

            # adaptive weighting
            adp_wt = (loss.detach() + self.args.norm_eps) ** self.args.norm_p
            loss = loss / adp_wt

            loss = loss.mean()
        
        return loss, ot_time
    
    def forward(self, x, aug_cond,e=None, context=None, num_graphs=None):
        return self.forward_with_loss(x, aug_cond, e, context, num_graphs)
    
    def sample(self, samples_shape, net=None, device=None,e=None,T=1, context=None, num_graphs=None):
        
        net = net if net is not None else self.net_ema

        if e is None:
            e = torch.randn(samples_shape, dtype=torch.float32, device=device)
        else:
            e = e.to(device)

        batch_size = samples_shape[0]
        z = e
        for step in range(T,0,-1):
            t = torch.full((batch_size,), (step) / T, device=device)
            r = torch.full((batch_size,), (step-1) / T, device=device)
            t = reshape_time(t, z.shape)
            r = reshape_time(r, z.shape)
            h = t - r # delta t
            u = net(z, t.view(-1), h.view(-1), context=context, num_graphs=num_graphs)
            z = z - (u * reshape_time(h, u.shape))
        return z
