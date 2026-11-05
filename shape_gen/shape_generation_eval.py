import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import ot
from tqdm import tqdm
import time
from datetime import timedelta

from argparse import ArgumentParser
from pointnet_exp_models import ResidualMLPAE, ResidualMLP2

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from datasets.shapenet_data_pc import ShapeNet15kPointClouds

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

class RandomContextDS(torch.utils.data.Dataset):
    def __init__(self, dataset, context_len=128):
        self.dataset = dataset
        self.contexts = torch.randn((len(dataset), context_len))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.contexts[index]

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
        self.context_num_steps = cmd_args.context_num_steps
        self.model_num_steps = cmd_args.model_num_steps
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


def test(test_loader, model, is_train_data, epoch, dataset_name, args, batch_size, context_generator):
    n_vis = 5
    vis_count = 0
    w2_distances = []
    with torch.no_grad():
        count = 0
        ref = []
        sample_pc = []
        for batch_idx, (data, context) in enumerate(test_loader):
            print(batch_idx)
            randind = np.random.choice(20) #20 views
            x = data['test_points'].to(device)

            m, s = data['mean'].float().to(device), data['std'].float().to(device)
            ref.append(x*s + m)

        
            X1_reshaped = x
            # X1_reshaped =  X1_reshaped_loss.permute(0, -1, 1)
            B, P, D = X1_reshaped.shape
            num_graphs = B
            
            subsampled_P = P

            context_X0 = torch.randn(B, 512).to(device) # B x 512
            with torch.no_grad():
                context = context_generator.sample(context_X0.shape, device=device, e=context_X0, T=args.context_num_steps, context=None, num_graphs=None)
            
            context = context.to(device)

            context_expanded = context.unsqueeze(1).expand(B, subsampled_P, -1).reshape(B * subsampled_P, -1)

            X0 = torch.randn_like(X1_reshaped).to(device).view(-1, 3) # B x 512, 3

            X0_reshaped = X0.view(num_graphs, -1, 3)

            samples_1step = model.sample(X0.shape, device=device, e=X0, T=args.model_num_steps, context=context_expanded, num_graphs=B)


            samples_1step_reshaped = samples_1step.view(num_graphs, -1, 3)

            sample_pc.append(samples_1step_reshaped * s + m)


            if batch_idx < 4:
                for i in range(B):
                    t1_points = (samples_1step_reshaped * s + m)[i]  # shape [1024, 3], a torch.FloatTensor
                    os.makedirs(f"ShapeNet_{args.category}_generaion_plot_points", exist_ok=True)
                    torch.save(t1_points,f"./ShapeNet_{args.category}_generaion_plot_points/Gen_points_count_{count}_{args.ot_method}.pt")

                    print("Saved one_step with shape:", t1_points.shape)
                    count += 1


        ref_pcs = torch.cat(ref, dim=0).contiguous()
        sample_pcs = torch.cat(sample_pc, dim=0).contiguous()
        results = compute_all_metrics(sample_pcs, ref_pcs, batch_size=batch_size)
        results = {k: (v.cpu().detach().item()
                    if not isinstance(v, float) else v) for k, v in results.items()}

    pprint(results)


def get_dataset(dataroot, npoints,category,use_mask=False):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True, use_mask = use_mask)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        use_mask=use_mask
    )
    return tr_dataset, te_dataset

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
    parser.add_argument('--svpoints', default=600)
    parser.add_argument('--dataroot_pc', default=os.environ.get('DATAROOT_PC', './data/ShapeNetCore.v2.PC15k/'))
    parser.add_argument('--dataroot_sv', default='./downloads/data/shapenet')
    parser.add_argument('--category', choices=['chair', 'car', 'airplane'], default='chair')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--distribution_type', default='single')
    parser.add_argument('--workers', type=int, default=4, help='workers')
    parser.add_argument('--context_num_steps', type=int, default=1,
                        help='Sampling steps for the context generator (1 = one-step, as in the paper).')
    parser.add_argument('--model_num_steps', type=int, default=1,
                        help='Sampling steps for the shape generator (1 = one-step, as in the paper).')
    parser.add_argument('--seed', type=int, default=42)

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

    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    ##################################
    #### data loader
    ##################################

    if cmd_args.dataset == 'shapenet':
        train_dataset, test_dataset = get_dataset(cmd_args.dataroot_pc,cmd_args.npoints, cmd_args.category, use_mask=False)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cmd_args.batch_size,sampler=train_sampler,
                                                shuffle=train_sampler is None, num_workers=int(cmd_args.workers), drop_last=True)

        test_dataset_context = RandomContextDS(test_dataset)

        test_loader = torch.utils.data.DataLoader(test_dataset_context, batch_size=cmd_args.batch_size,
                                                        shuffle=False, num_workers=0, drop_last=False)
    
    else:
        raise NotImplementedError("dataset not implemented")




    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)

    arch = ResidualMLPAE
    net_configs = {"din": 5 + 512, "dout":3, "dhidden": 2048, "depth": 12, "din_context":512, "dhidden_context":256, "dout_context":256}

    args = Args(cmd_args)

    print(args.use_ot)

    
    net_configs_context_gen = {"din": 512 + 2, "dout":512, "dhidden": 1024, "depth": 4}
    context_generator = load_saved_model(arch=ResidualMLP2, args=args, net_configs=net_configs_context_gen,
                            path=f"./latent_gen_models/OT_shapenet_{args.category}_method_exact_AE_meanflow_{args.mf_variant}.pth", device=device)
    context_generator = context_generator.eval()

    print("Loaded context generator")

    n_epochs = cmd_args.n_epochs # ShapeNet Chairs: 1000, ModelNet: 1500
    model = load_saved_model(arch=arch, args=args, net_configs=net_configs,
                            path=f"./pointnet_models/OT_shapenet_{args.category}_method_exact_AE_meanflow_generation_{args.mf_variant}.pth", device=device)
    
    model = model.eval()

    test(test_loader=test_loader, model=model,
         is_train_data=False, epoch=0, dataset_name="shapenet",
          batch_size=cmd_args.batch_size, args=args, context_generator = context_generator)
