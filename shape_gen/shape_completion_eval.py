import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import ot
from tqdm import tqdm
import time
from datetime import timedelta

torch.manual_seed(42)
np.random.seed(42)

from argparse import ArgumentParser
from pointnet_exp_models import ResidualMLP

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.shapenet_data_sv import ShapeNet_Multiview_Points

import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, project_root+'/meanflow')

from meanflow.models.meanflow_mf import MeanFlow as VanillaMeanFlow
from meanflow.models.meanflow_imf import MeanFlow as ImprovedMeanFlow

MEANFLOW_VARIANTS = {"mf": VanillaMeanFlow, "imf": ImprovedMeanFlow}

device = 'cuda'

from pprint import pprint
from metrics.evaluation_metrics import compute_all_metrics, EMD_CD

class Args:
    def __init__(self, cmd_args):   
        self.ema_decay = 0.999
        self.ema_decays = [0.99, 0.999]
        self.norm_eps = 1e-6
        self.norm_p = 0.5
        self.use_ot = cmd_args.use_ot
        self.ot_method = cmd_args.ot_method # 'exact', 'sinkhorn', 'lot-ind', 'lot-group'
        self.tr_sampler = 'v1'
        self.svpoints = cmd_args.svpoints
        self.category = cmd_args.category
        self.mf_variant = cmd_args.mf_variant

        self.lr = 0.0006
        self.eval_frequency = 50
        self.epochs = 16000
        self.compute_fid = True
        self.log_per_step = 100

        self.P_mean_t = -0.6
        self.P_std_t = 1.6
        self.P_mean_r = -4.0
        self.P_std_r = 1.6
        self.warmup_epochs = 200
        self.norm_p = 0.75
        self.ratio = 0.75
        self.dropout = 0.2
        self.use_edm_aug = True



#####################
##load and test saved model
#####################
def load_saved_model(arch, args, net_configs, path, device):
    model = MEANFLOW_VARIANTS[args.mf_variant](arch, args, net_configs).to(device)
    state_dict = torch.load(path, map_location=device)
    clean_state_dict = state_dict
    model.load_state_dict(clean_state_dict)
    return model


def test(test_loader, model, is_train_data, epoch, dataset_name, plot_only=False):
    n_vis = 5
    vis_count = 0
    w2_distances = []
    ref = []
    samples = []
    masked = []
    k = 0
    with torch.no_grad():
        count = 0
        # print(model.net.in_proj.weight - state_dict['net.in_proj.weight'])
        for batch_idx, data in enumerate(test_loader):
            print(batch_idx)
            # data = data.to(device)

            gt_all = data['test_points'].to(device)
            x_all = data['sv_points'].to(device)


            B,V,N_context,C = x_all.shape
            gt_all = gt_all[:,None,:,:].expand(-1, V, -1,-1)
            _, _, N, _ = gt_all.shape

            x = x_all.reshape(B * V, N_context, C)
            gt_all = gt_all.reshape(B*V, N, C)

            m, s = data['mean'].float().to(device), data['std'].float().to(device)


            X1_reshaped = gt_all

            num_graphs = B
            
            subsampled_P = 256
            X1_subsampled = X1_reshaped
            X1_truncated = X1_subsampled[:, :-N_context, :]

            X0 = torch.randn_like(X1_truncated).to(device).view(-1, 3) # B x 512, 3

            # X0 = torch.randn_like(X1)
            samples_1step = model.sample(X0.shape, device=device, e=X0, T=1, context=x, num_graphs=B*V)

            samples_1step_reshaped = samples_1step.view(num_graphs*V, N - N_context, 3)

            recon = torch.cat([samples_1step_reshaped, x], dim=1)


            recon_adj = recon.reshape(B,V,N,C)* s.unsqueeze(1) + m.unsqueeze(1)
            gt_all = gt_all.reshape(B, V, N, C)

            ref.append( gt_all * s.unsqueeze(1) + m.unsqueeze(1))
            # masked.append(x_adj[:,:,:test_loader.dataset.sv_samples,:])
            samples.append(recon_adj)

    ref_pcs = torch.cat(ref, dim=0)
    sample_pcs = torch.cat(samples, dim=0)
    # masked = torch.cat(masked, dim=0)

    B, V, N, C = ref_pcs.shape

    results = EMD_CD(sample_pcs.reshape(B*V, N, C),
                    ref_pcs.reshape(B*V, N, C), 2, reduced=False)

    
    pprint({key: val.mean().item() for key, val in results.items()})


def get_dataset(dataroot_pc, dataroot_sv, npoints, svpoints, category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot_pc,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)
    tr_dataset = ShapeNet_Multiview_Points(root_pc=dataroot_pc, root_views=dataroot_sv,
                                            cache=os.path.join(dataroot_pc, '../cache'), split='train',
        categories=[category],
        npoints=npoints, sv_samples=svpoints,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return tr_dataset

def get_mvr_dataset(pc_dataroot, views_root, npoints,category,svpoints):
    tr_dataset = ShapeNet15kPointClouds(root_dir=pc_dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)
    te_dataset = ShapeNet_Multiview_Points(root_pc=pc_dataroot, root_views=views_root,
                                            cache=os.path.join(pc_dataroot, '../cache'), split='val',
        categories=[category],
        npoints=npoints, sv_samples=svpoints,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return te_dataset

def get_dataloader(opt, train_dataset, test_dataset=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,sampler=train_sampler,
                                                shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,sampler=test_sampler,
                                                shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler


if __name__ == "__main__":
    device = 'cuda'

    parser = ArgumentParser()

    parser.add_argument("--use_ot", action="store_true")
    parser.set_defaults(use_ot=False)


    parser.add_argument('--local_rank', '--local-rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))  # set by torchrun

    parser.add_argument("--dataset", type=str, choices=['shapenet'], default='shapenet')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument("--ot_method", type=str, choices=['exact', 'sinkhorn', 'lot-ind', 'lot-group'], default='exact')
    parser.add_argument("--mf_variant", type=str, choices=["mf", "imf"], default="imf",
                         help="Which MeanFlow formulation to use: vanilla MF or improved MF (iMF).")

    parser.add_argument('--dataset_name', type=str, default='ShapeNet')
    parser.add_argument('--npoints', default=2048)
    parser.add_argument('--svpoints', default=200)
    parser.add_argument('--dataroot_pc', default=os.environ.get('DATAROOT_PC', './data/ShapeNetCore.v2.PC15k/'))
    parser.add_argument('--dataroot_sv', default='./downloads/data/shapenet')
    parser.add_argument('--category', choices=['chair', 'car', 'airplane'], default='chair')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--distribution_type', default='single')
    parser.add_argument('--workers', type=int, default=4, help='workers')

    # Top-level hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent', type=int, default=1)
    parser.add_argument('--skip_weight', type=float, default=0.01)
    parser.add_argument('--sigma_offset', type=float, default=6.0)
    parser.add_argument('--loss', type=str, default='l1_sum')

    def namespace_to_dict(args):
        cfg = {}

        for k, v in vars(args).items():
            if '.' in k:
                parts = k.split('.')
                d = cfg
                for p in parts[:-1]:
                    d = d.setdefault(p, {})
                d[parts[-1]] = v
            else:
                cfg[k] = v

        return cfg

    from types import SimpleNamespace

    def dict_to_namespace(d):
        return SimpleNamespace(**{
            k: dict_to_namespace(v) if isinstance(v, dict) else v
            for k, v in d.items()
        })


    args = parser.parse_args()

    cfg = dict_to_namespace(namespace_to_dict(args))

    cmd_args = cfg

    local_rank = cmd_args.local_rank

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(hours=2))

    torch.manual_seed(42)
    np.random.seed(42)

    ##################################
    #### data loader
    ##################################

    if cmd_args.dataset == 'shapenet':
        train_dataset = get_dataset(cmd_args.dataroot_pc, cmd_args.dataroot_sv, cmd_args.npoints, cmd_args.svpoints, cmd_args.category)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cmd_args.batch_size,sampler=train_sampler,
                                                shuffle=train_sampler is None, num_workers=int(cmd_args.workers), drop_last=True)

        test_dataset = get_mvr_dataset(cmd_args.dataroot_pc, cmd_args.dataroot_sv,
                                            cmd_args.npoints, cmd_args.category, cmd_args.svpoints)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cmd_args.batch_size,
                                                        shuffle=False, num_workers=0, drop_last=False)
    
    else:
        raise NotImplementedError("dataset not implemented")




    torch.manual_seed(42)
    np.random.seed(42)

    arch = ResidualMLP
    net_configs = {"din": 5 + 256, "dout":3, "dhidden": 2048, "depth": 12, "din_context":256, "dhidden_context":256, "dout_context":256, "uncond":False}

    args = Args(cmd_args)

    print(args.use_ot)


    n_epochs = cmd_args.n_epochs # ShapeNet Chairs: 1000, ModelNet: 1500

    if args.category == "chair":
        model = load_saved_model(arch=ResidualMLP, args=args, net_configs=net_configs,
                                path=f"./pointnet_models/OT_shapenet_method_exact_AE_meanflow_{args.svpoints}_{args.mf_variant}.pth", device=device)
    else:
        model = load_saved_model(arch=ResidualMLP, args=args, net_configs=net_configs,
                            path=f"./pointnet_models/OT_shapenet_{args.category}_method_exact_AE_meanflow_{args.svpoints}_{args.mf_variant}.pth", device=device)
    model = model.eval()

    test(test_loader=test_loader, model=model, is_train_data=False, epoch=0, dataset_name="shapenet", plot_only=False)