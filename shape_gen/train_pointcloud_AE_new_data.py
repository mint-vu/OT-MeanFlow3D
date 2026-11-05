## Code adapted from: https://github.com/cihanongun/Point-Cloud-Autoencoder/blob/master/PointCloudAE.ipynb

import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from torch.utils.data import DataLoader

parser = ArgumentParser()
parser.add_argument('--arch', type=str, choices=['deepset', 'transformer'], default="deepset")
parser.add_argument('--category', type=str, choices=['chair', 'car', 'airplane'], default="chair")
parser.add_argument('--dataroot_pc', type=str, default=os.environ.get('DATAROOT_PC', './data/ShapeNetCore.v2.PC15k/'))

args = parser.parse_args()


def slicedW2(x, L=64):
    ''' This function calculates the sliced Wasserstein
        distance between x and uniform distribution:
        Input
                x: batchsize*d
                L: number of slices

        Output
                dist: average distance
    '''
    theta = torch.randn((x.shape[1], L)).to(x.device)
    theta /= torch.norm(theta, dim=0).unsqueeze(0)
    # print((theta**2).sum(0))
    y = torch.randn_like(x)

    xslice, _ = torch.sort(x @ theta, 0)
    yslice, _ = torch.sort(y @ theta, 0)
    return ((xslice - yslice) ** 2).mean()


'''
PointNet AutoEncoder
Learning Representations and Generative Models For 3D Point Clouds
https://arxiv.org/abs/1707.02392
'''

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
    

# --- Transformer Encoder Block ---
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, d)
        attn_out, _ = self.attn(x, x, x)         # self-attention
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# --- Transformer AutoEncoder ---
class PointTransformerAE(nn.Module):
    def __init__(self, num_points=1024, dim=128, depth=5, num_heads=4, latent_dim=128):
        super().__init__()
        self.num_points = num_points

        # Input embedding
        self.input_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads=num_heads, ff_dim=dim*4)
            for _ in range(depth)
        ])

        # Latent representation
        self.to_latent = nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.ReLU()
        )

        # Decoder: simple MLP from latent → points
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3)
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, 3)
        Returns:
            recon: (B, N, 3)
            latent: (B, latent_dim)
        """
        B, N, _ = x.shape
        x = self.input_mlp(x)  # (B, N, dim)

        for layer in self.layers:
            x = layer(x)        # transformer layers

        # Global pooling to latent
        latent = torch.max(x, dim=1)[0]   # (B, dim)
        latent = self.to_latent(latent)   # (B, latent_dim)

        # Decode from latent
        recon = self.decoder(latent).view(B, self.num_points, 3)
        return recon#, latent

import shutil

def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def plotPCbatch(pcArray1, pcArray2, show = True, save = False, name=None, fig_count=8 , sizex = 12, sizey=3):
    
    # pc1 = pcArray1[0:fig_count]
    # pc2 = pcArray2[0:fig_count]

    pc1 = pcArray1
    pc2 = pcArray2
    
    fig=plt.figure(figsize=(sizex, sizey))
    
    for i in range(fig_count*2):

        ax = fig.add_subplot(2,fig_count,i+1, projection='3d')
        
        if(i<fig_count):
            ax.scatter(pc1[i,:,0], pc1[i,:,2], pc1[i,:,1], c='b', marker='.', alpha=0.8, s=8)
        else:
            ax.scatter(pc2[i-fig_count,:,0], pc2[i-fig_count,:,2], pc2[i-fig_count,:,1], c='b', marker='.', alpha=0.8, s=8)

    plt.subplots_adjust(wspace=0, hspace=0)
        
    if(save):
        fig.savefig(name + '.png')
        plt.close(fig)
    
    if(show):
        plt.show()
    else:
        return fig
    
def chamfer_distance(pcd1: torch.Tensor, pcd2: torch.Tensor) -> torch.Tensor:
    """
    Compute Chamfer distance between two batches of point clouds.

    Args:
        pcd1: (B, N, d) point cloud tensor
        pcd2: (B, M, d) point cloud tensor (M can differ from N)

    Returns:
        dist: (B,) Chamfer distance for each batch
    """
    B, N, d = pcd1.shape
    _, M, _ = pcd2.shape

    # Compute squared distances (B, N, M)
    diff = pcd1.unsqueeze(2) - pcd2.unsqueeze(1)   # (B, N, 1, d) - (B, 1, M, d)
    dist_sq = torch.sum(diff ** 2, dim=-1)         # (B, N, M)

    # For each point in pcd1, find nearest neighbor in pcd2
    dist1, _ = torch.min(dist_sq, dim=2)  # (B, N)

    # For each point in pcd2, find nearest neighbor in pcd1
    dist2, _ = torch.min(dist_sq, dim=1)  # (B, M)

    # Average over points
    cd = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)  # (B,)

    return torch.mean(cd)

batch_size = 32
output_folder = f"SN_{args.category}_pointnet_AE_output_{args.arch}_new_no_swd/" # folder path to save the results
save_results = True # save the results to output_folder
use_GPU = True # use GPU, False to use CPU
latent_size = 512 # bottleneck size of the Autoencoder model

torch.manual_seed(42)  # reproducibility


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


train_dataset, test_dataset = get_dataset(args.dataroot_pc, npoints=2048,
                                          category=args.category, use_mask = False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,)

point_size = 2048

if args.arch == 'deepset':
    net = PointCloudAE(point_size,latent_size)
else:
    net = PointTransformerAE(point_size, latent_size)
# net = PointCloudAENew(point_size, latent_size)


if(use_GPU):
    device = torch.device("cuda")
    # if torch.cuda.device_count() > 1: # if there are multiple GPUs use all
    #     net = torch.nn.DataParallel(net)
else:
    device = torch.device("cpu")

net = net.to(device)

# from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance

# optimizer = optim.Adam(net.parameters(), lr=0.0001)
optimizer = optim.Adam(net.parameters(), lr=(batch_size // 32) * 0.0005)

def train_epoch():
    epoch_loss = 0
    epoch_loss_recon = 0
    epoch_loss_swd = 0
    for i, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()

        X1 = data["train_points"] # B, P, 3
        X1 = X1.to(device)
        X1_reshaped_loss = X1 # B, P , 3
        X1_reshaped = X1_reshaped_loss.permute(0, -1, 1) # B, 3, P

        input = X1_reshaped_loss if args.arch == 'transformer' else X1_reshaped

        # output = net(input)
        latents = net.encoder(input)
        output = net.decoder(latents)

        # output = net(data.permute(0,2,1)) # transpose data for NumberxChannelxSize format
        # loss, _ = chamfer_distance(data, output)
        loss_recon = chamfer_distance(X1_reshaped_loss, output) 
        loss = loss_recon
        # loss_swd = slicedW2(latents, L=1024)
        # loss = loss_recon + 0.5 * loss_swd
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        # epoch_loss_swd += loss_swd.item()
        epoch_loss_recon += loss_recon.item()
    # print(f"recon loss: {epoch_loss_recon / i}, swd loss: {epoch_loss_swd / i}")
    print(f"recon loss: {epoch_loss_recon / i}")
    return epoch_loss/i

def test_batch(data): # test with a batch of inputs
    with torch.no_grad():
        X1 = data["test_points"] # B, P, 3
        X1 = X1.to(device)
        X1_reshaped_loss = X1 # B, P , 3
        # X1_reshaped = data.pos.view(data.num_graphs, 3, -1) #B, 3, P
        X1_reshaped = X1_reshaped_loss.permute(0, -1, 1) # B, 3, P

        input = X1_reshaped_loss if args.arch == 'transformer' else X1_reshaped

        output = net(input)

        # output = net(data.permute(0,2,1))
        # loss, _ = chamfer_distance(data, output)
        loss = chamfer_distance(X1_reshaped_loss, output)
        
    return loss.item(), output.cpu()

def test_epoch(): # test with all test set
    with torch.no_grad():
        epoch_loss = 0
        for i, data in enumerate(test_loader):
            loss, output = test_batch(data)
            epoch_loss += loss

    return epoch_loss/i

if(save_results):
    clear_folder(output_folder)

train_loss_list = []  
test_loss_list = []  

for i in range(2001) :

    startTime = time.time()
    
    net.train()
    train_loss = train_epoch() #train one epoch, get the average loss
    train_loss_list.append(train_loss)
    
    net.eval()
    test_loss = test_epoch() # test with test set
    test_loss_list.append(test_loss)
    
    epoch_time = time.time() - startTime
    
    writeString = "epoch " + str(i) + " train loss : " + str(train_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n"
    
    # plot train/test loss graph
    plt.plot(train_loss_list, label="Train")
    plt.plot(test_loss_list, label="Test")
    plt.legend()

    if(save_results): # save all outputs to the save folder

        # write the text output to file
        with open(output_folder + "prints.txt","a") as file: 
            file.write(writeString)

        # update the loss graph
        plt.savefig(output_folder + "loss.png")
        plt.close()

        # save input/output as image file
        if(i%20==0):
            test_samples = next(iter(test_loader))
            loss , test_output = test_batch(test_samples)
            print(test_output.shape)
            plotPCbatch(test_samples["test_points"].view(test_samples["test_points"].shape[0], -1, 3).cpu(), test_output.cpu(), show=False, save=True, name = (output_folder  + "epoch_" + str(i)))

            torch.save(net.state_dict(), os.path.join(output_folder, "model_checkpoint.pth"))

    else : # display all outputs
        
        test_samples = next(iter(test_loader))
        loss , test_output = test_batch(test_samples)
        plotPCbatch(test_samples,test_output)

        print(writeString)

        plt.show()