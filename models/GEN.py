import torch
from torch_geometric.nn import GENConv, global_add_pool
import torch.nn.functional as F


class GEN(torch.nn.Module):
    '''
    DeeperGCN: All You Need to Train Deeper GCNs
    https://arxiv.org/pdf/2006.07739.pdf

    params

        in_channels [int]
            the dimension of vertex feature

        hidden_channels [int]
            dimensions of hidden layers
            also the embedding size

        num_layers [int]
            the number of GraphSAGE layers

        out_channels [int]
            the number of classes or regression targets

        dropout [float or None]
            the dropout rate, default is None

    '''
    def __init__(self, in_channels, edge_dim, hidden_channels, num_layers, out_channels,
                 dropout=None):
        super(GEN, self).__init__()

        #self.atom_encoder = torch.nn.Linear(in_channels, hidden_channels)
        #self.bond_encoder = torch.nn.Linear(edge_dim, hidden_channels)

        self.atom_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU()
        )

        self.bond_encoder = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU()
        )

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = GENConv(
                hidden_channels,
                hidden_channels,
                aggr='softmax',
                t=1.0,
                learn_t=True,
                learn_p=True,
                num_layers=2,
                norm='batch'
            )
            self.convs.append(conv)

        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_channels, out_channels)

        self.dropout = dropout

        #elf.activation = torch.nn.LeakyReLU(0.1)
        self.activation = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.atom_encoder(x)
        e = self.bond_encoder(edge_attr)

        for i in range(len(self.convs)-1):
            x = self.convs[i](x, edge_index, e)
            x = self.activation(x)
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # last layer, no relu and dropout
        x = self.convs[-1](x, edge_index, e)

        # global pooling
        x = global_add_pool(x, batch=batch)

        # linears
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x