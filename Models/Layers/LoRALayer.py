

import torch


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha,device):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev).to(device)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim)).to(device)
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha,input_size,output_size):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            input_size, output_size, rank, alpha
        )

    def forward(self, x):
        x = self.lora(x)
        return self.linear(x)