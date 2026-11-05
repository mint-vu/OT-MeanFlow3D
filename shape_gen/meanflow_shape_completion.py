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


def train(arch, args, net_configs, device, n_epochs, train_loader, test_loader, local_rank, dataset):
    model = MEANFLOW_VARIANTS[args.mf_variant](arch, args, net_configs).to(device)
    print(f"n params: {sum([torch.numel(p) for p in model.net.parameters()])}")

    ### distributed
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer = optim.Adam(model.parameters(), lr=4e-5, betas=(0.5, 0.999))
    # optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9998)#, 0.998)
    

    # mse = torch.nn.MSELoss(reduction='none')
    losses = []
    loss_log = []

    s_time = time.time()  
    timings = []
    ot_times = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        # distributed
        train_sampler.set_epoch(epoch)
        s_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # data = data.to(device)

            randind = np.random.choice(20) #20 views
            sv_x = data['sv_points'][:,randind].to(device) # B, P, 3
            # sv_x = data['train_points'].to(device)
            x = data['train_points'] # ground truth 
            
            x = x.to(device)

            X1_reshaped = x
            # X1_reshaped =  X1_reshaped_loss.permute(0, -1, 1)
            B, P, D = X1_reshaped.shape
            num_graphs = B


            subsampled_P = 256

            X1_subsampled = []
            for index in range(X1_reshaped.shape[0]):
                X1_subsampled.append(X1_reshaped[index, torch.randperm(X1_reshaped.shape[1], device=device)[:subsampled_P]])
            X1_subsampled = torch.stack(X1_subsampled, dim=0).to(device)
            # X1_subsampled = X1_reshaped

            # X1_truncated = X1_subsampled[:, :-args.svpoints, :]
            # X0 = torch.randn_like(X1_subsampled).to(device).view(-1, 3) # B x 512, 3
            X0 = torch.randn_like(X1_subsampled).to(device).view(-1, 3) # B x 512, 3

            # X0_reshaped = X0.view(batch.num_graphs, -1, 3)

            # loss = model.forward_with_loss(x=X1_subsampled.view(-1, 3), aug_cond=None, e=X0, context=context_expanded, num_graphs = num_graphs)
            loss, ot_time = model(x=X1_subsampled.reshape(-1, 3), aug_cond=None, e=X0, context=sv_x, num_graphs = num_graphs)
            # loss, ot_time = model(x=X1_subsampled, aug_cond=None, e=X0, context=context_expanded, num_graphs = num_graphs)
            ot_times.append(ot_time)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            # print(f"Iteration: {epoch}, loss: {loss.item()}")
            optimizer.step()
            model.module.update_ema()

            epoch_loss += loss.detach().cpu().item() * B

        e_time = time.time()
        timings.append(e_time - s_time)
        scheduler.step()

        epoch_loss = torch.as_tensor(epoch_loss, device=torch.device("cuda"))
        torch.distributed.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.SUM)
        epoch_loss = epoch_loss.cpu().item()
        epoch_loss = epoch_loss / len(train_dataset)
        
        if (epoch % 10 == 0) or (epoch == (n_epochs - 1)):
            loss_log.append(torch.mean(torch.tensor(losses)))
            if dist.get_rank() == 0:
                e_time = time.time()
                print(f"Epoch {epoch}, Loss: {epoch_loss/len(train_loader):.4f}, time taken: {sum(timings) / len(timings)}, ot time: {sum(ot_times) / len(ot_times)}")
            torch.distributed.barrier()
            s_time = time.time()
            losses = []
            timings = []

        if (epoch % 20 == 0) or (epoch == (n_epochs - 1)):
            model.eval()
            if dist.get_rank() == 0:
                print("testing on test data")
                if ((epoch % 200 == 0) or (epoch == (n_epochs - 1))) and epoch != 0:
                    test(train_loader, model.module, is_train_data=False, epoch=epoch, dataset_name=dataset, plot_only=False)
                else:
                    test(train_loader, model.module, is_train_data=False, epoch=epoch, dataset_name=dataset, plot_only=True)
                # print("testing on train data")
                # test(train_loader_no_shuffle, model, is_train_data=True, epoch=epoch)
            torch.distributed.barrier()
    
        if (epoch % 100 == 0) or (epoch == (n_epochs - 1)):
            if dist.get_rank() == 0:
                print("saving model...")
                if args.use_ot:
                    torch.save(model.module.state_dict(), f"./pointnet_models/OT_{dataset}_{args.category}_method_{args.ot_method}_AE_meanflow_{args.svpoints}_{args.mf_variant}.pth")
                else:
                    torch.save(model.module.state_dict(), f"./pointnet_models/MeanFlow_{dataset}_{args.category}_AE_meanflow_{args.svpoints}_{args.mf_variant}.pth")
            torch.distributed.barrier()

#####################
##load and test saved model
#####################
def load_saved_model(arch, args, net_configs, path, device):
    model = MEANFLOW_VARIANTS[args.mf_variant](arch, args, net_configs).to(device)
    state_dict = torch.load(path, map_location=device)
    clean_state_dict = {}
    for k in state_dict.keys():
        new_k = k
        if k.startswith("module."):
            new_k = k.replace("module.", "")
        clean_state_dict[new_k] = state_dict[k]
    model.load_state_dict(clean_state_dict)
    return model


def test(test_loader, model, is_train_data, epoch, dataset_name, plot_only=False):
    n_vis = 5
    vis_count = 0
    w2_distances = []
    with torch.no_grad():
        count = 0
        # print(model.net.in_proj.weight - state_dict['net.in_proj.weight'])
        for batch_idx, data in enumerate(test_loader):
            print(batch_idx)
            # data = data.to(device)
                    
            randind = np.random.choice(20) #20 views
            sv_x = data['sv_points'][:,randind].to(device) # B, P, 3
            # sv_x = data['sv_points'].to(device)
            x = data['test_points'].to(device)
        

            X1_reshaped = x
            # X1_reshaped =  X1_reshaped_loss.permute(0, -1, 1)
            B, P, D = X1_reshaped.shape
            num_graphs = B
            
            subsampled_P = 256
            
            # with torch.no_grad():
            #     context = net.encode_context(sv_x) #bs * 512

            # context_expanded = context.unsqueeze(1).expand(B, subsampled_P, -1).reshape(B * subsampled_P, -1)
            X1_subsampled = X1_reshaped
            X1_truncated = X1_subsampled[:, :-args.svpoints, :]

            # X0 = torch.randn_like(X1_reshaped).to(device).view(-1, 3) # B x 512, 3
            X0 = torch.randn_like(X1_truncated).to(device).view(-1, 3) # B x 512, 3

            X0_reshaped = X0.view(num_graphs, -1, 3)

            # X0 = torch.randn_like(X1)
            samples_1step = model.sample(X0.shape, device=device, e=X0, T=1, context=sv_x, num_graphs=B)
            samples_5step = model.sample(X0.shape, device=device, e=X0, T=5, context=sv_x, num_graphs=B)
            samples_20step = model.sample(X0.shape, device=device, e=X0, T=20, context=sv_x, num_graphs=B)

            samples_1step_reshaped = samples_1step.view(num_graphs, -1, 3)
            samples_5step_reshaped = samples_5step.view(num_graphs, -1, 3)
            samples_20step_reshaped = samples_20step.view(num_graphs, -1, 3)

            for i in range(B):
                if count < 20:
                    fig = plt.figure(figsize=(18, 3))

                    # Ground truth
                    ax = fig.add_subplot(161, projection="3d")
                    if dataset_name == 'modelnet10':
                        ax.scatter(X1_reshaped[i, :, 0].cpu(), X1_reshaped[i, :, 1].cpu(), X1_reshaped[i, :, 2].cpu(), s=5, c="blue")
                    else:
                        ax.scatter(X1_reshaped[i, :, 0].cpu(), X1_reshaped[i, :, 2].cpu(), X1_reshaped[i, :, 1].cpu(), s=5, c="blue")
                    ax.set_title("Ground Truth")

                    # 1-step
                    ax = fig.add_subplot(162, projection="3d")
                    # samples_1step_reshaped = samples_1step.view(batch.num_graphs, -1, 3)
                    if dataset_name == 'modelnet10':
                        ax.scatter(samples_1step_reshaped[i, :, 0].cpu(), samples_1step_reshaped[i, :, 1].cpu(), samples_1step_reshaped[i, :, 2].cpu(), s=5, c="red")
                    else:
                        ax.scatter(samples_1step_reshaped[i, :, 0].cpu(), samples_1step_reshaped[i, :, 2].cpu(), samples_1step_reshaped[i, :, 1].cpu(), s=5, c="red")
                    ax.set_title("Generated (T=1)")

                    ax = fig.add_subplot(163, projection="3d")
                    # samples_5step_reshaped = samples_5step.view(batch.num_graphs, -1, 3)
                    if dataset_name == 'modelnet10':
                        ax.scatter(samples_5step_reshaped[i, :, 0].cpu(), samples_5step_reshaped[i, :, 1].cpu(), samples_5step_reshaped[i, :, 2].cpu(), s=5, c="red")
                    else:
                        ax.scatter(samples_5step_reshaped[i, :, 0].cpu(), samples_5step_reshaped[i, :, 2].cpu(), samples_5step_reshaped[i, :, 1].cpu(), s=5, c="red")
                    ax.set_title("Generated (T=5)")

                    ax = fig.add_subplot(164, projection="3d")
                    # samples_20step_reshaped = samples_20step.view(batch.num_graphs, -1, 3)
                    if dataset_name == 'modelnet10':
                        ax.scatter(samples_20step_reshaped[i, :, 0].cpu(), samples_20step_reshaped[i, :, 1].cpu(), samples_20step_reshaped[i, :, 2].cpu(), s=5, c="red")
                    else:
                        ax.scatter(samples_20step_reshaped[i, :, 0].cpu(), samples_20step_reshaped[i, :, 2].cpu(), samples_20step_reshaped[i, :, 1].cpu(), s=5, c="red")
                        
                    ax.set_title("Generated (T=20)")

                    ax = fig.add_subplot(165, projection="3d")
                    if dataset_name == 'modelnet10':
                        ax.scatter(sv_x[i, :, 0].cpu(), sv_x[i, :, 1].cpu(), sv_x[i, :, 2].cpu(), s=5, c="red")
                    else:
                        ax.scatter(sv_x[i, :, 0].cpu(), sv_x[i, :, 2].cpu(), sv_x[i, :, 1].cpu(), s=5, c="red")
                    ax.set_title("Partial input")

                    ax = fig.add_subplot(166, projection="3d")
                    inp = torch.cat([samples_5step_reshaped, sv_x], dim=1)
                    if dataset_name == 'modelnet10':
                        ax.scatter(inp[i, :, 0].cpu(), inp[i, :, 1].cpu(), inp[i, :, 2].cpu(), s=5, c="red")
                    else:
                        ax.scatter(inp[i, :, 0].cpu(), inp[i, :, 2].cpu(), inp[i, :, 1].cpu(), s=5, c="red")
                    ax.set_title("Mixed (Full)")

                    fig.suptitle(f'Point cloud (Use OT: {args.use_ot})')

                    plt.show()

                    if args.use_ot:
                        os.makedirs(f"OT_{args.ot_method}_{dataset_name}_{args.category}_test_{args.svpoints}_{args.mf_variant}", exist_ok=True)
                        plt.savefig(f"./OT_{args.ot_method}_{dataset_name}_{args.category}_test_{args.svpoints}_{args.mf_variant}/count_{count}.png")
                    else:
                        os.makedirs(f"MeanFlow_{dataset_name}_{args.category}_test_{args.svpoints}_{args.mf_variant}", exist_ok=True)
                        plt.savefig(f"./MeanFlow_{dataset_name}_{args.category}_test_{args.svpoints}_{args.mf_variant}/count_{count}.png")

                    plt.clf()
                    plt.close()
                else:
                    if plot_only:
                        return
                if not plot_only:
                    w2_dist = ot.emd2([],[],((X1_truncated[i].unsqueeze(0)-samples_20step_reshaped[i].to(device).unsqueeze(1))**2).sum(-1))

                    # print(w2_dist)
                    w2_distances.append(w2_dist)
                count += 1

            # if count > 20:
            #     break
    if not plot_only:
        print("Avg W2 distance: ", sum(w2_distances) / len(w2_distances))
    # torch.save(model.state_dict(), f"./pointnet_models/meanflow_{len(data)}_samples_OT_{args.use_ot}_epoch_{epoch}.pth")



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

def get_mvr_dataset(pc_dataroot, views_root, npoints,category, svpoints):
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
    parser.add_argument('--svpoints', default=200, type=int)
    parser.add_argument('--dataroot_pc', default=os.environ.get('DATAROOT_PC', './data/ShapeNetCore.v2.PC15k/'))
    parser.add_argument('--dataroot_sv', default='./downloads/data/shapenet')
    parser.add_argument('--category', choices=['chair', 'car', 'airplane'], default='chair')
    parser.add_argument('--batch_size', default=16)
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
    net_configs = {"din": 5 + 256, "dout":3, "dhidden": 2048, "depth": 12, "din_context":256, "dhidden_context":256, "dout_context":256, "uncond": False}

    args = Args(cmd_args)

    print(args.use_ot)


    n_epochs = cmd_args.n_epochs # ShapeNet Chairs: 1000, ModelNet: 1500
    train(arch=arch, args=args, net_configs=net_configs, device=device, n_epochs=n_epochs,
          train_loader=train_dataloader, test_loader=test_loader, local_rank=local_rank, dataset=cmd_args.dataset)
