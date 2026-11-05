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
from pointnet_exp_models import PointCloudAE, ResidualMLP2

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
        self.category = cmd_args.category
        self.mf_variant = cmd_args.mf_variant

        self.lr = cmd_args.lr
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


def train(arch, args, net_configs, device, n_epochs, train_loader, test_loader, local_rank, dataset, batch_size):
    model = MEANFLOW_VARIANTS[args.mf_variant](arch, args, net_configs).to(device)
    model.train()

    ### distributed
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # optimizer = optim.Adam(model.parameters(), lr=2e-5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    

    # mse = torch.nn.MSELoss(reduction='none')
    losses = []
    loss_log = []

    s_time = time.time()  
    timings = []
    ot_times = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0

        # distributed
        train_sampler.set_epoch(epoch)
        s_time = time.time()
        for batch_idx, (data, context) in enumerate(train_loader):
            # data = data.to(device)
            # sv_x = data['train_points'].to(device)
            x = data['train_points'] # ground truth 
            
            x = x.to(device)

            X1_reshaped = x
            # X1_reshaped =  X1_reshaped_loss.permute(0, -1, 1)
            B, P, D = X1_reshaped.shape
            num_graphs = B

            subsampled_P = 256
            # X1_reshaped B, P, d

            with torch.no_grad():
                context = net.encoder(x.permute(0, 2, 1)) #bs * 512

            X0 = torch.randn_like(context).to(device)

            loss, ot_time = model(x=context, aug_cond=None, e=X0, context=None, num_graphs = None)
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

        if (epoch % 100 == 0) or (epoch == (n_epochs - 1)):
            model.eval()
            if dist.get_rank() == 0:
                print("testing on test data")
                test(test_loader, model.module, is_train_data=False, epoch=epoch,
                      dataset_name=dataset, args = args, batch_size=batch_size)
                # print("testing on train data")
                # test(train_loader_no_shuffle, model, is_train_data=True, epoch=epoch)
            torch.distributed.barrier()
    
        if (epoch % 100 == 0) or (epoch == (n_epochs - 1)):
            if dist.get_rank() == 0:
                print("saving model...")
                if args.use_ot:
                    torch.save(model.module.state_dict(), f"./latent_gen_models/OT_{dataset}_{args.category}_method_{args.ot_method}_AE_meanflow_{args.mf_variant}.pth")
                else:
                    torch.save(model.module.state_dict(), f"./latent_gen_models/MeanFlow_{dataset}_{args.category}_AE_meanflow_{args.mf_variant}.pth")
            torch.distributed.barrier()

#####################
##load and test saved model
#####################
def load_saved_model(arch, args, net_configs, path, device):
    model = MEANFLOW_VARIANTS[args.mf_variant](arch, args, net_configs).to(device)
    # state_dict = torch.load("./pointnet_models/meanflow_400_samples_OT_False.pth", map_location=device)
    state_dict = torch.load(path, map_location=device)
    clean_state_dict = {}
    for k in state_dict.keys():
        new_k = k
        if k.startswith("module."):
            new_k = k.replace("module.", "")
        clean_state_dict[new_k] = state_dict[k]
    model.load_state_dict(clean_state_dict)
    print(model.module.net.in_proj.weight - clean_state_dict['net.in_proj.weight'])
    return model
# model = model.to(device)
# model.eval()


def test(test_loader, model, is_train_data, epoch, dataset_name, args, batch_size):
    n_vis = 5
    vis_count = 0
    w2_distances = []
    with torch.no_grad():
        count = 0
        ref = []
        sample_pc = []
        for batch_idx, (data, context) in enumerate(test_loader):
            print(batch_idx)
            # data = data.to(device)
                    
            randind = np.random.choice(20) #20 views
            # sv_x = data['train_points'].to(device)
            x = data['test_points'].to(device)

            m, s = data['mean'].float().to(device), data['std'].float().to(device)
            ref.append(x*s + m)

        
            X1_reshaped = x
            # X1_reshaped =  X1_reshaped_loss.permute(0, -1, 1)
            B, P, D = X1_reshaped.shape
            num_graphs = B
            
            subsampled_P = P
            
            with torch.no_grad():
                context = net.encoder(x.permute(0, 2, 1)) #bs * 512

            X0 = torch.randn_like(context).to(device) # B x 512

            # X0 = torch.randn_like(X1)
            samples_1step = model.sample(X0.shape, device=device, e=X0, T=1, context=None, num_graphs=None)
            samples_5step = model.sample(X0.shape, device=device, e=X0, T=5, context=None, num_graphs=None)
            samples_20step = model.sample(X0.shape, device=device, e=X0, T=20, context=None, num_graphs=None)


            # sample_pc.append(samples_1step_reshaped * s + m)

            decoded_samples_1step = net.decoder(samples_1step)
            decoded_samples_5step = net.decoder(samples_5step)
            decoded_samples_20step = net.decoder(samples_20step)

            for i in range(B):
                if count < 20:
                    fig = plt.figure(figsize=(12, 3))

                    # Ground truth
                    ax = fig.add_subplot(141, projection="3d")
                    if dataset_name == 'modelnet10':
                        ax.scatter(X1_reshaped[i, :, 0].cpu(), X1_reshaped[i, :, 1].cpu(), X1_reshaped[i, :, 2].cpu(), s=5, c="blue")
                    else:
                        ax.scatter(X1_reshaped[i, :, 0].cpu(), X1_reshaped[i, :, 2].cpu(), X1_reshaped[i, :, 1].cpu(), s=5, c="blue")
                    ax.set_title("Ground Truth")

                    # 1-step
                    ax = fig.add_subplot(142, projection="3d")
                    # samples_1step_reshaped = samples_1step.view(batch.num_graphs, -1, 3)
                    ax.scatter(decoded_samples_1step[i, :, 0].cpu(), decoded_samples_1step[i, :, 2].cpu(), decoded_samples_1step[i, :, 1].cpu(), s=5, c="red")
                    ax.set_title("Generated (T=1)")

                    ax = fig.add_subplot(143, projection="3d")
                    ax.scatter(decoded_samples_5step[i, :, 0].cpu(), decoded_samples_5step[i, :, 2].cpu(), decoded_samples_5step[i, :, 1].cpu(), s=5, c="red")
                    ax.set_title("Generated (T=5)")

                    ax = fig.add_subplot(144, projection="3d")
                    ax.scatter(decoded_samples_20step[i, :, 0].cpu(), decoded_samples_20step[i, :, 2].cpu(), decoded_samples_20step[i, :, 1].cpu(), s=5, c="red")
                    ax.set_title("Generated (T=20)")

                    fig.suptitle(f'Point cloud (Use OT: {args.use_ot})')

                    plt.show()

                    if args.use_ot:
                        os.makedirs(f"OT_{args.ot_method}_{dataset_name}_{args.category}_latent_gen_{args.mf_variant}", exist_ok=True)
                        plt.savefig(f"./OT_{args.ot_method}_{dataset_name}_{args.category}_latent_gen_{args.mf_variant}/count_{count}.png")
                    else:
                        os.makedirs(f"MeanFlow_{dataset_name}_{args.category}_latent_gen_{args.mf_variant}", exist_ok=True)
                        plt.savefig(f"./MeanFlow_{dataset_name}_{args.category}_latent_gen_{args.mf_variant}/count_{count}.png")

                    plt.clf()
                    plt.close()

                # w2_dist = ot.emd2([],[],((X1_reshaped[i].unsqueeze(0)-samples_1step_reshaped[i].to(device).unsqueeze(1))**2).sum(-1))

                # w2_distances.append(w2_dist)
                count += 1
            break


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
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument("--ot_method", type=str, choices=['exact', 'sinkhorn', 'lot-ind', 'lot-group'], default='exact')
    parser.add_argument("--mf_variant", type=str, choices=["mf", "imf"], default="imf",
                         help="Which MeanFlow formulation to use: vanilla MF or improved MF (iMF).")

    parser.add_argument('--dataset_name', type=str, default='ShapeNet')
    parser.add_argument('--npoints', default=2048)
    parser.add_argument('--svpoints', default=600)
    parser.add_argument('--dataroot_pc', default=os.environ.get('DATAROOT_PC', './data/ShapeNetCore.v2.PC15k/'))
    parser.add_argument('--dataroot_sv', default='./downloads/data/shapenet')
    parser.add_argument('--category', choices=['chair', 'car', 'airplane'], default='chair')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--distribution_type', default='single')
    parser.add_argument('--workers', type=int, default=4, help='workers')


    # Top-level hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent', type=int, default=1)
    parser.add_argument('--skip_weight', type=float, default=0.01)
    parser.add_argument('--sigma_offset', type=float, default=6.0)
    parser.add_argument('--loss', type=str, default='l1_sum')

    parser.add_argument('--ckpt_path', type=str, default=None)
    # # parser.add_argument('--data_recenter_per_shape', type=bool, default=False)
    # # parser.add_argument('--data_normalize_global', type=bool, default=True)

    from collections import defaultdict

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
    import os

    local_rank = cmd_args.local_rank

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(hours=2))

    torch.manual_seed(42)
    np.random.seed(42)

    ##################################
    #### data loader
    ##################################

    if cmd_args.dataset == 'shapenet':
        train_dataset, test_dataset = get_dataset(cmd_args.dataroot_pc,cmd_args.npoints, cmd_args.category, use_mask=False)

        train_dataset_context = RandomContextDS(train_dataset)
        train_sampler = DistributedSampler(train_dataset_context, shuffle=True)

        train_dataloader = torch.utils.data.DataLoader(train_dataset_context, batch_size=cmd_args.batch_size,sampler=train_sampler,
                                                shuffle=train_sampler is None, num_workers=int(cmd_args.workers), drop_last=True)


        # test_dataset = get_mvr_dataset(cmd_args.dataroot_pc, cmd_args.dataroot_sv,
        #                                     cmd_args.npoints, cmd_args.category)
        
        test_dataset_context = RandomContextDS(test_dataset)

        test_loader = torch.utils.data.DataLoader(test_dataset_context, batch_size=cmd_args.batch_size,
                                                        shuffle=False, num_workers=0, drop_last=False)
    
    else:
        raise NotImplementedError("dataset not implemented")



    ###########################
    #Load PointNetAE
    ###########################
    net = PointCloudAE(point_size=cmd_args.npoints,latent_size=512).to(device)
    state_dict = torch.load(f"./SN_{cmd_args.category}_pointnet_AE_output_deepset_new_no_swd/model_checkpoint.pth")
    net.load_state_dict(state_dict)
    net = net.cuda()
    net.eval()

    # print(net)


    torch.manual_seed(42)
    np.random.seed(42)

    arch = ResidualMLP2
    # net_configs = {"din": 512 + 2, "dout":512, "dhidden": 1024, "depth": 3}
    net_configs = {"din": 512 + 2, "dout":512, "dhidden": 1024, "depth": 4}

    args = Args(cmd_args)

    print(args.use_ot)


    n_epochs = cmd_args.n_epochs # ShapeNet Chairs: 1000, ModelNet: 1500
    if cmd_args.ckpt_path is None:
        train(arch=arch, args=args, net_configs=net_configs, device=device, n_epochs=n_epochs,
            train_loader=train_dataloader, test_loader=test_loader, local_rank=local_rank,
                dataset=cmd_args.dataset, batch_size = cmd_args.batch_size)
    else:
        model = MEANFLOW_VARIANTS[args.mf_variant](arch, args, net_configs).to(device)
        model.load_state_dict(torch.load(cmd_args.ckpt_path, map_location=device))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        model.eval()
        if dist.get_rank() == 0:
            test(test_loader, model.module, is_train_data=False, epoch=0,
                      dataset_name=cmd_args.dataset, args = args, batch_size=cmd_args.batch_size)
        torch.distributed.barrier()
        
