import os

import torch
import torch.nn as nn

from torch.configim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from Models.BaseModel import BaseModel
from Models.Layers.AnomalyMultiHeadAttention import AnomalyBlock, AnomalyBlockList
from Models.Layers.PE import PE
from Models.Layers.RevIN import RevIN
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow

from Utils.LogUtil import wirteLog
from torch.nn import functional as F

from Utils.ProtocolUtil import pa
import matplotlib.pyplot as plt
import seaborn as sns













def normal(array,min_val,max_val):
    return (array-min_val)/(max_val-min_val)

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data)
        # nn.init.kaiming_uniform_(mod.weight.data)

    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') !=-1 :
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)

class Encoder(nn.Module):
    def __init__(self, config,out_z):
        super(Encoder, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 320
            nn.Conv1d(config.nc,config.ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 160
            nn.Conv1d(config.ndf, config.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(config.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 80
            nn.Conv1d(config.ndf * 2, config.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(config.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 40
            nn.Conv1d(config.ndf * 4, config.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(config.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 20
            nn.Conv1d(config.ndf * 8, config.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(config.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 10

            nn.Conv1d(config.ndf * 16, out_z, 10, 1, 0, bias=False),
            # state size. (nz) x 1
        )

    def forward(self, input):

        output = self.main(input)

        return output


##


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.main=nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(config.nz,config.ngf*16,10,1,0,bias=False),
            nn.BatchNorm1d(config.ngf*16),
            nn.ReLU(True),
            # state size. (ngf*16) x10
            nn.ConvTranspose1d(config.ngf * 16, config.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(config.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 20
            nn.ConvTranspose1d(config.ngf * 8, config.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(config.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 40
            nn.ConvTranspose1d(config.ngf * 4, config.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(config.ngf*2),
            nn.ReLU(True),
            # state size. (ngf) x 80
            nn.ConvTranspose1d(config.ngf * 2, config.ngf , 4, 2, 1, bias=False),
            nn.BatchNorm1d(config.ngf ),
            nn.ReLU(True),
            # state size. (ngf) x 160
            nn.ConvTranspose1d(config.ngf , config.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 320


        )

    def forward(self, input):

        output = self.main(input)
        return output



##
class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()
        model = Encoder(config,1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features




##
class Generator(nn.Module):

    def __init__(self, config):
        super(Generator, self).__init__()
        self.encoder1 = Encoder(config,config.nz)
        self.decoder = Decoder(config)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_x = self.decoder(latent_i)
        return gen_x, latent_i




class BeatGAN(BaseModel):


    def __init__(self, config):
        super(BeatGAN, self).__init__()

        self.device = config["device"]
        self.config = config

        self.batch_size = config["batch_size"]
        self.nz = config.nz
        self.epoch = config.epoch

        self.G = Generator(config).to(self.device)
        self.G.apply(weights_init)


        self.D = Discriminator(config).to(self.device)
        self.D.apply(weights_init)


        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()

        self.configimizerD = torch.optim.Adam(self.D.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
        self.configimizerG = torch.optim.Adam(self.G.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

        self.total_steps = 0
        self.cur_epoch = 0

        self.input = torch.empty(size=(self.config.batch_size, self.config.nc, self.config.isize), dtype=torch.float32,
                                 device=self.device)
        self.label = torch.empty(size=(self.config.batch_size,), dtype=torch.float32, device=self.device)

        self.gt = torch.empty(size=(config.batch_size,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.config.batch_size, self.config.nc, self.config.isize), dtype=torch.float32,
                                       device=self.device)
        self.real_label = 1
        self.fake_label = 0

        self.out_d_real = None
        self.feat_real = None

        self.fake = None
        self.latent_i = None
        self.out_d_fake = None
        self.feat_fake = None

        self.err_d_real = None
        self.err_d_fake = None
        self.err_d = None

        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None

    def fit(self,train_data):

        train_dataloader = self.processData(train_data)

        BCELoss = nn.BCELoss()
        MSELoss = nn.MSELoss()

        for epoch in range(self.epoch):
            self.cur_epoch += 1

            self.G.train()
            self.D.train()
            epoch_iter = 0
            for data in train_dataloader:
                data = data[0]
                self.y_real_, self.y_fake_ = torch.ones(self.batch_size).to(self.device), torch.zeros(self.batch_size).to(
                    self.device)
                epoch_iter += 1

                ##
                self.D.zero_grad()
                # --
                # Train with real


                out_d_real, _ = self.D(data)
                # Train with fake
                fake = self.G(data)
                out_d_fake, _ = self.D(fake)

                error_real = BCELoss(out_d_real, self.y_real_)
                error_fake = BCELoss(out_d_fake, self.y_fake_)

                error_discrimtor = error_real + error_fake
                error_discrimtor.backward(retain_graph=True)

                self.optimizerD.step()

                self.G.zero_grad()

                _, feat_fake = self.D(fake)
                _, feat_real = self.D(data)

                err_g_adv = MSELoss(feat_fake, feat_real)  # loss for feature matching
                err_g_rec = MSELoss(fake, data)  # constrain x' to look like x

                err_g = err_g_rec + 0.01 * err_g_adv
                err_g.backward()
                self.optimizerG.step()

                







    def set_input(self, input):
        self.input.data.resize_(input.size()).copy_(input[0])
        self.gt.data.resize_(input[1].size()).copy_(input[1])

        # fixed input for view
        if self.total_steps == self.config.batch_size:
            self.fixed_input.data.resize_(input[0].size()).copy_(input[0])

    ##
    def configimize(self):

        self.update_netd()
        self.update_netg()

        # If D loss too low, then re-initialize netD
        if self.err_d.item() < 5e-6:
            self.reinitialize_netd()

    def update_netd(self):
        ##
        self.D.zero_grad()
        # --
        # Train with real
        self.label.data.resize_(self.config.batch_size).fill_(self.real_label)
        self.out_d_real, self.feat_real = self.D(self.input)
        # --
        # Train with fake
        self.label.data.resize_(self.config.batch_size).fill_(self.fake_label)
        self.fake, self.latent_i = self.G(self.input)
        self.out_d_fake, self.feat_fake = self.D(self.fake)
        # --

        self.err_d_real = self.bce_criterion(self.out_d_real,
                                             torch.full((self.batch_size,), self.real_label, device=self.device))
        self.err_d_fake = self.bce_criterion(self.out_d_fake,
                                             torch.full((self.batch_size,), self.fake_label, device=self.device))

        self.err_d = self.err_d_real + self.err_d_fake
        self.err_d.backward()
        self.configimizerD.step()

    ##
    def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
        self.D.apply(weights_init)
        print('Reloading d net')

    ##
    def update_netg(self):
        self.G.zero_grad()
        self.label.data.resize_(self.config.batch_size).fill_(self.real_label)
        self.fake, self.latent_i = self.G(self.input)
        self.out_g, self.feat_fake = self.D(self.fake)
        _, self.feat_real = self.D(self.input)

        # self.err_g_adv = self.bce_criterion(self.out_g, self.label)   # loss for ce
        self.err_g_adv = self.mse_criterion(self.feat_fake, self.feat_real)  # loss for feature matching
        self.err_g_rec = self.mse_criterion(self.fake, self.input)  # constrain x' to look like x

        self.err_g = self.err_g_rec + self.err_g_adv * self.config.w_adv
        self.err_g.backward()
        self.configimizerG.step()

    ##
    def get_errors(self):

        errors = {'err_d': self.err_d.item(),
                  'err_g': self.err_g.item(),
                  'err_d_real': self.err_d_real.item(),
                  'err_d_fake': self.err_d_fake.item(),
                  'err_g_adv': self.err_g_adv.item(),
                  'err_g_rec': self.err_g_rec.item(),
                  }

        return errors

        ##

    def get_generated_x(self):
        fake = self.G(self.fixed_input)[0]
        return self.fixed_input.cpu().data.numpy(), fake.cpu().data.numpy()

    ##



    def test(self, test_data):
        
        dataloader_ = self.processData(test_data)
        self.eval()
        score = []
        with torch.no_grad():

            for i, data in enumerate(dataloader_, 0):
                test_x = data[0].to(self.device)


                fake = self.G(test_x)

                loss = torch.mean(
                    torch.pow((test_x.view(test_x.shape[0], -1) - fake.view(fake.shape[0], -1)), 2),
                    dim=1)

                score.append(loss.detach().numpy().cpu())

            score = torch.concatenate(score, dim=0).numpy()

            score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1,
                                      range_min=0)

        return score


def predict_for_right(self, dataloader_, min_score, max_score, threshold, save_dir):
        '''

        :param dataloader:
        :param min_score:
        :param max_score:
        :param threshold:
        :param save_dir:
        :return:
        '''
        assert save_dir is not None
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            # Create big error tensor for the test set.
            test_pair = []
            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)

            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake, latent_i = self.G(self.input)

                error = torch.mean(
                    torch.pow((self.input.view(self.input.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),
                    dim=1)

                self.an_scores[i * self.config.batch_size: i * self.config.batch_size + error.size(0)] = error.reshape(
                    error.size(0))

                # # Save test images.

                batch_input = self.input.cpu().numpy()
                batch_output = self.fake.cpu().numpy()
                ano_score = error.cpu().numpy()
                assert batch_output.shape[0] == batch_input.shape[0] == ano_score.shape[0]
                for idx in range(batch_input.shape[0]):
                    if len(test_pair) >= 100:
                        break
                    normal_score = (ano_score[idx] - min_score) / (max_score - min_score)

                    if normal_score >= threshold:
                        test_pair.append((batch_input[idx], batch_output[idx]))

            # print(len(test_pair))
            self.saveTestPair(test_pair, save_dir)






