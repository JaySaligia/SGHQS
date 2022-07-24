import torch
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv
from config import *


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases=args.n_bases))
        for i in range(n_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=args.n_bases))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations, num_bases=args.n_bases))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x
