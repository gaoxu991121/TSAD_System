import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from Models.BaseModel import BaseModel
from Models.Layers.PE import PE
from Models.Layers.RevIN import RevIN
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow

from Utils.LogUtil import wirteLog
from torch.nn import functional as F

from Utils.ProtocolUtil import pa



class TRANSFORMER(BaseModel):
    def __init__(self, config):
        super(TRANSFORMER, self).__init__()
        self.config = config

        self.epoch = self.config["epoch"]
        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.drop_out_rate = self.config["drop_out_rate"]
        self.window_size = self.config["window_size"]
        self.device = self.config["device"]
        self.num_heads = self.config["num_heads"]
        self.num_layers = self.config["num_layers"]

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.position_embedding = nn.Embedding(self.window_size, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_out_rate)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=4*self.hidden_size,  # Typically 4 times the hidden size
            dropout=self.drop_out_rate
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=4*self.hidden_size,
            dropout=self.drop_out_rate
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_layers)

        self.fc_out = nn.Linear(self.hidden_size, self.input_size)

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == 0
        return src_mask

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(trg.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(src.device)
        )
        trg_positions = (
            torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(trg.device)
        )

        embed_src = self.dropout(self.embedding(src) + self.position_embedding(src_positions))
        embed_trg = self.dropout(self.embedding(trg) + self.position_embedding(trg_positions))

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        memory = self.encoder(embed_src, src_key_padding_mask=src_padding_mask)
        out = self.decoder(embed_trg, memory, tgt_mask=trg_mask, memory_key_padding_mask=src_padding_mask)

        out = self.fc_out(out)

        return out


class TRANSFORMER(BaseModel):
    """

    """

    def __init__(self, config):
        super(TRANSFORMER, self).__init__()
        self.config = config

        self.epoch = self.config["epoch"]
        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.drop_out_rate = self.config["drop_out_rate"]

        self.window_size = self.config["window_size"]


        self.divice = self.config["device"]

        self.num_heads = self.config["num_heads"]
        self.num_layers = self.config["num_layers"]


        self.transformer = nn.Transformer(self.input_size,nhead=self.num_heads,batch_first=True,num_encoder_layers=self.num_layers,num_decoder_layers=self.num_layers,dim_feedforward=256)

        self.fc = nn.Linear(self.input_size,self.input_size)






    def forward(self,  x,target):

        x = self.transformer(x,target)

        x = self.fc(x)
        return x




    def fit(self, train_data, write_log=False):
        train_loader = self.processData(train_data)
        self.train()
        lr = self.config["learning_rate"]
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        # 设置余弦学习率衰减，这里的T_max是衰减周期
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)


        l = nn.MSELoss(reduction='sum')

        epoch_loss = []


        for ep in range(self.epoch):

            # l1s = []
            running_loss = 0
            for d in train_loader:
                optimizer.zero_grad()
                item = d[0].to(self.divice)



                output = self.forward(item,item[:,-1,:].unsqueeze(dim=1))

                loss = l(output, item[:,-1,:].unsqueeze(dim=1))



                # l1s.append(torch.mean(loss).item())

                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()  # 在每个epoch后更新学习率

            # 计算当前epoch的平均损失
            epoch_loss.append(running_loss / len(train_loader))

            print(f'train epoch [{ep+1}/{self.epoch}],\t loss = {epoch_loss[ep]}')


        identifier = self.config["identifier"]

        # self.save()

        if write_log:
            wirteLog(self.config["base_path"] + "/Logs/" + identifier, "train_loss", {"epoch_loss": epoch_loss})

    def test(self, test_data):
        """
             在测试集上进行测试，输出的是归一到[0,1]的numpy数组类型的异常得分
             :param test_data: 测试数据

        """
        test_dataloader = self.processData(test_data)
        self.eval()
        score = []

        l = nn.MSELoss(reduction='none')

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(self.divice)

                output = self.forward(item, item[:, -1, :].unsqueeze(dim=1))
                loss = l(output[:, -1, :], item[:, -1, :])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())


            score = torch.concatenate(score, dim=0).numpy()

            score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        return score








