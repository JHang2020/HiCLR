import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
import random

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, dropout_graph,add_graph, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        self.fc = nn.Linear(hidden_dim, num_class)

        self.dropout_graph = dropout_graph
        self.add_graph = add_graph
        # initialize parameters for edge importance weighting
        #if edge_importance_weighting:
        #    self.edge_importance = nn.ParameterList([
        #        nn.Parameter(torch.ones(self.A.size()))
        #        for i in self.st_gcn_networks
        #    ])
        if edge_importance_weighting:
            edge_importance1 = nn.Parameter(torch.ones(self.A.size()))
            edge_importance2 = nn.Parameter(torch.ones(self.A.size()))
            edge_importance3 = nn.Parameter(torch.ones(self.A.size()))
            edge_importance4 = nn.Parameter(torch.ones(self.A.size()))
            edge_importance5 = nn.Parameter(torch.ones(self.A.size()))
            edge_importance6 = nn.Parameter(torch.ones(self.A.size()))
            edge_importance7 = nn.Parameter(torch.ones(self.A.size()))
            edge_importance8 = nn.Parameter(torch.ones(self.A.size()))
            edge_importance9 = nn.Parameter(torch.ones(self.A.size()))
            edge_importance10 = nn.Parameter(torch.ones(self.A.size()))
            edge_importance_lis = [edge_importance1,edge_importance2,
                edge_importance3,edge_importance4,edge_importance5,
                edge_importance6,edge_importance7,
                edge_importance8,edge_importance9,edge_importance10,
                ]
            
            for i in range(1,len(self.st_gcn_networks)+1):
                self.register_parameter('edge_importance{}'.format(i), edge_importance_lis[i-1])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
    
    def drop_graph(self, A):
        r = random.uniform(0, self.dropout_graph)
        m = nn.Dropout(p=r)
        output = m(A)
        return output
    def drop_graph2(self, A):
        s1 = A.sum()
        r = random.uniform(0, self.dropout_graph)
        m = nn.Dropout(p=r)
        output = m(A)
        s2 = output.sum()
        output = output*s1/s2
        return output
    def gaussian_dropout(self, x):
        # r = random.uniform(0, 0.5)
        eps = torch.randn(x.size()).to(x.device) + 1
        A = x * eps
        return A

    def randadd(self,A):#keep the intesity same 
        r = random.uniform(0, self.add_graph)#add ratio
        soa = torch.sum(A)
        B = torch.ones_like(A)
        m = nn.Dropout(p=(1.0-r))
        ind = m(B)
        ind[ind!=0] = 1.0
        ind[A!=0] = 0.0
        sum_ratio = soa/(soa+torch.sum(ind))
        output = (A + ind) * sum_ratio
        return output

    def forward(self, x, drop_graph=False, return1d=False,KNN=False,return_feat=False):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        ori_x = x
        
        if drop_graph and return1d:#only return 1 dropout version
            x_d = ori_x
            for idx,gcn in enumerate(self.st_gcn_networks):
                importance = getattr(self,'edge_importance'+str(idx+1))
                A = self.drop_graph(self.A)
                #A = self.randadd(A)
                x_d, _ = gcn(x_d, A * importance) 
            x_d = F.avg_pool2d(x_d, x_d.size()[2:])
            x_d = x_d.view(N, M, -1).mean(dim=1)

            # prediction
            x_d = self.fc(x_d)
            x_d = x_d.view(x_d.size(0), -1)

            return x_d
        
        # forward
        for idx,gcn in enumerate(self.st_gcn_networks):
            importance = getattr(self,'edge_importance'+str(idx+1))
            x, _ = gcn(x, self.A * importance)
        
        if drop_graph:#return both drop and undrop version
            x_d = ori_x
            for idx,gcn in enumerate(self.st_gcn_networks):
                importance = getattr(self,'edge_importance'+str(idx+1))
                A = self.drop_graph(self.A)
                #A = self.randadd(A)
                x_d, _ = gcn(x_d, A * importance)  
            
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(N, M, -1).mean(dim=1)

            # prediction
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            x_d = F.avg_pool2d(x_d, x_d.size()[2:])
            x_d = x_d.view(N, M, -1).mean(dim=1)

            # prediction
            x_d = self.fc(x_d)
            x_d = x_d.view(x_d.size(0), -1)

            return x, x_d
        
        
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)
        feat = x
        # prediction
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        if return_feat:
            return feat,x

        return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A