import torch
from torch import nn, Tensor
from torch.nn import functional as F
class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # B, C, H, W
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        # energy = torch.matmul(proj_query, proj_key.permute(0, 1, 3, 2))

        energy = torch.einsum('bctw,bchw->bthw', proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        # print("attention.shape is", attention.shape) #torch.Size([64, 5, 207, 207])
        # print("proj_value.shape is", proj_value.shape) #torch.Size([64, 40, 207, 12])

        # out = torch.matmul(attention, proj_value)
        out = torch.einsum('bthw,bchw->bctw', attention, proj_value)
        # out = out * mi_output

        out = self.gamma * out + x

        return out



