import torch

from Models.Layers.LoRALayer import LoRALayer


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha,device):
        super().__init__()
        self.linear = linear
        # print(linear.in_features)
        # print(linear.out_features)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha,device
        )


    def forward(self, x):
        return self.linear(x) + self.lora(x)


    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias