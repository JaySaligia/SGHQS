import torch
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv, HGTConv, Linear, FastRGCNConv, GCNConv, ChebConv, SAGEConv, GraphConv, \
    GravNetConv, ResGatedGraphConv, GATConv, GATv2Conv, TransformerConv, TAGConv, ARMAConv, SGConv, MFConv, EGConv
from config import *


class EG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(EGConv(in_channels=in_channels, out_channels=hidden_channels))
        for i in range(n_layers - 2):
            self.convs.append(EGConv(in_channels=hidden_channels, out_channels=hidden_channels))
        self.convs.append(EGConv(in_channels=hidden_channels, out_channels=out_channels, num_heads=1))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


class MF(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(MFConv(in_channels=in_channels, out_channels=hidden_channels))
        for i in range(n_layers - 2):
            self.convs.append(MFConv(in_channels=hidden_channels, out_channels=hidden_channels))
        self.convs.append(MFConv(in_channels=hidden_channels, out_channels=out_channels))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


class SG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(SGConv(in_channels=in_channels, out_channels=hidden_channels))
        for i in range(n_layers - 2):
            self.convs.append(SGConv(in_channels=hidden_channels, out_channels=hidden_channels))
        self.convs.append(SGConv(in_channels=hidden_channels, out_channels=out_channels))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


class ARMA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(ARMAConv(in_channels=in_channels, out_channels=hidden_channels))
        for i in range(n_layers - 2):
            self.convs.append(ARMAConv(in_channels=hidden_channels, out_channels=hidden_channels))
        self.convs.append(ARMAConv(in_channels=hidden_channels, out_channels=out_channels))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


class TAG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(TAGConv(in_channels=in_channels, out_channels=hidden_channels))
        for i in range(n_layers - 2):
            self.convs.append(TAGConv(in_channels=hidden_channels, out_channels=hidden_channels))
        self.convs.append(TAGConv(in_channels=hidden_channels, out_channels=out_channels))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


class Transformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels, heads=1))
        for i in range(n_layers - 2):
            self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=1))
        self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=out_channels, heads=1))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


class GATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index, edge_type):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index, edge_type):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GatedGraphGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(ResGatedGraphConv(in_channels=in_channels, out_channels=hidden_channels))
        for i in range(n_layers - 2):
            self.convs.append(ResGatedGraphConv(in_channels=hidden_channels, out_channels=hidden_channels))
        self.convs.append(ResGatedGraphConv(in_channels=hidden_channels, out_channels=out_channels))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(GraphConv(in_channels=in_channels, out_channels=hidden_channels))
        for i in range(n_layers - 2):
            self.convs.append(GraphConv(in_channels=hidden_channels, out_channels=hidden_channels))
        self.convs.append(GraphConv(in_channels=hidden_channels, out_channels=out_channels))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


class SAGEGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(SAGEConv(in_channels=in_channels, out_channels=hidden_channels))
        for i in range(n_layers - 2):
            self.convs.append(SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels))
        self.convs.append(SAGEConv(in_channels=hidden_channels, out_channels=out_channels))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


class ChebGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(ChebConv(in_channels=in_channels, out_channels=hidden_channels, K=2))
        for i in range(n_layers - 2):
            self.convs.append(ChebConv(in_channels=hidden_channels, out_channels=hidden_channels, K=2))
        self.convs.append(ChebConv(in_channels=hidden_channels, out_channels=out_channels, K=2))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(GCNConv(in_channels=in_channels, out_channels=hidden_channels))
        for i in range(n_layers - 2):
            self.convs.append(GCNConv(in_channels=hidden_channels, out_channels=hidden_channels))
        self.convs.append(GCNConv(in_channels=hidden_channels, out_channels=out_channels))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


class FASTRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(FastRGCNConv(in_channels, hidden_channels, num_relations, num_bases=args.n_bases))
        for i in range(n_layers - 2):
            self.convs.append(FastRGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=args.n_bases))
        self.convs.append(FastRGCNConv(hidden_channels, out_channels, num_relations, num_bases=args.n_bases))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=3):
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


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, hgraph, labeled_class):
        super().__init__()
        self.labeled_class = labeled_class
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in hgraph.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, hgraph.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict[self.labeled_class])  # ['item']
