import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from Models.BaseModel import BaseModel
from Models.Layers.ChannelWiseAttention import ChannelWiseAttention
from Models.Layers.PE import PE
from Models.Layers.RevIN import RevIN
from Models.Layers.SAM import SAM
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow

from Utils.LogUtil import wirteLog
from torch.nn import functional as F

from Utils.ProtocolUtil import pa


class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm

    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x - x_hat).pow(2))
        print("z shape:",z.shape)
        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)

    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1)) * eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2 * np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0) * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean == True:
            E_z = torch.mean(E_z)
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        # phi = D
        phi = torch.sum(gamma, dim=0) / gamma.size(0)

        # mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l

    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


class DAGMM(BaseModel):
    def __init__(self,config):
        super(DAGMM, self).__init__()
        self.config = config
        self.epoch = config["epoch"]
        self.device = config["device"]

        self.learning_rate =config["learning_rate"]
        self.beta = 0.01
        self.n_feats = config["input_size"]
        self.n_hidden = config["hidden_size"]
        self.n_latent = config["latent_size"]
        self.n_window =  config["window_size"]  # DAGMM w_size = 5
        self.n = self.n_feats * self.n_window
        self.n_gmm = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.estimate = nn.Sequential(
            nn.Linear(self.n_latent + 2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
        )


    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x):
        ## Encode Decoder

        x = x.view(-1, self.n)

        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        ## Compute Reconstructoin
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        ## Estimate
        gamma = self.estimate(z)
        return z_c, x_hat.reshape(-1,self.n_window,self.n_feats), z, gamma.reshape(-1,self.n_window,self.n_feats)


    def fit(self, train_data, write_log=False):
        train_loader = self.processData(train_data)
        self.train()


        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-5)

        epoch_loss = []

        l = nn.MSELoss(reduction='none')
        compute = ComputeLoss(self, 0.1, 0.005, 'cpu', self.n_gmm)
        for ep in range(self.epoch):
            ep = ep + 1
            l1s = []
            running_loss = 0
            for d in train_loader:

                item = d[0].to(self.device)
                d = item
                _, x_hat, z, gamma = self.forward(d)
                # loss = compute.forward(item,x_hat,z,gamma)
                # print("loss:",loss)
                l1, l2 = l(x_hat, d), l(gamma, d)

                loss = torch.mean(l1)

                running_loss += loss.item()
                l1s.append(torch.mean(loss).item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



            # 计算当前epoch的平均损失
            epoch_loss.append(running_loss / len(train_loader))

            print(f'train epoch [{ep}/{self.epoch}],\t loss = {np.mean(l1s)}')

        identifier = self.config["identifier"]

        self.save()

        if write_log:
            wirteLog(self.config["base_path"] + "/Logs/" + identifier, "train_loss", {"epoch_loss": epoch_loss})

    def test(self, test_data):
        """
             在测试集上进行测试，输出的是归一到[0,1]的numpy数组类型的异常得分
             :param test_data: 测试数据

        """

        test_loader = self.processData(test_data)

        self.eval()

        l = nn.MSELoss(reduction='none')
        with torch.no_grad():
            score = []
            for d in test_loader:
                item = d[0].to(self.device)
                d = item
                _, x_hat, z, gamma = self.forward(d)

                l1, l2 = l(x_hat, d), l(gamma, d)

                loss = torch.sum(l1,dim=-1) 
                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss)

        score = torch.concatenate(score, dim=0).detach().cpu().numpy()
        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        return score

    def visualize(self,data_tensor):

        self.eval()
        score = []

        l = nn.MSELoss(reduction='none')
        attn_weight = None
        with torch.no_grad():

                item = data_tensor.to(self.device)
                output, attn_weight = self.forward(item)


        print("attn hape:",attn_weight.shape)
        # attn_weights_sample = attn_weight[-1]
        # # 可视化注意力权重
        # plt.figure(figsize=(12, 8))
        # sns.heatmap(attn_weights_sample.detach().numpy(), cmap='viridis')
        # plt.title("Attention Weights across Channels")
        # plt.xlabel("Channels")
        # plt.ylabel("Channels")
        # plt.show()






