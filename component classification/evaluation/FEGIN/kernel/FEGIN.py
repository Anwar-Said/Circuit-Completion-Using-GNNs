import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN,Conv1d, MaxPool1d
from torch_geometric.nn import GINConv, global_mean_pool, global_sort_pool,MLP
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class NestedGIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, use_z=False, use_rd=False):
        super(NestedGIN, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features
        if self.use_z or self.use_rd:
            input_dim += 8

        self.conv1 = GINConv(
            Sequential(
                Linear(input_dim, hidden),
                BN(hidden),
                ReLU(),
                Linear(hidden, hidden),
                BN(hidden),
                ReLU(),
            ),
            train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden), 
                        ReLU(),
                        Linear(hidden, hidden),
                        BN(hidden), 
                        ReLU(),
                    ),
                    train_eps=True))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        if self.use_rd:
            self.rd_projection.reset_parameters()
        if self.use_z:
            self.z_embedding.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # node label embedding
        z_emb = 0
        if self.use_z and 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj

        if self.use_rd or self.use_z:
            x = torch.cat([z_emb, x], -1)

        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]

        x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        x = global_mean_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__



class FEGIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden,emb_size, *args, **kwargs):
        super(FEGIN, self).__init__()
        # k = 0.6
        # if k < 1:  # Transform percentile to number.
        #     num_nodes = sorted([data.num_nodes for data in dataset])
        #     k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
        #     k = max(10, k)
        # self.k = int(k)
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ),
            train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ),
                    train_eps=True))
        # conv1d_channels = [16, 32]
        # total_latent_dim = hidden * num_layers+emb_size
        # conv1d_kws = [total_latent_dim, 5]
        # self.conv1d1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
        #                     conv1d_kws[0])
        # self.maxpool1d = MaxPool1d(2, 2)
        # self.conv1d2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
        #                     conv1d_kws[1], 1)
        # dense_dim = int((self.k - 2) / 2 + 1)
        # dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        # print("dense dim:",dense_dim)
        # hidden_dim = (self.k*hidden*(num_layers-1)) + emb_size
        # print(hidden_dim)
        # self.mlp = MLP([hidden_dim,64, dataset.num_classes], dropout=0.5, batch_norm=False)
        self.lin1 = Linear(num_layers * hidden+emb_size, hidden*2)
        # self.lin2 = Linear(hidden, hidden)
        # self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden*2, hidden)
        self.lin3 = Linear(hidden, hidden)
        self.lin4 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        # self.lin3.reset_parameters()

    def forward(self, data,des):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        emb = des.x
        # print(batch)
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = torch.cat(xs[1:], dim=-1)
        # print(x.shape)
        x = global_mean_pool(torch.cat(xs, dim=1), batch)
        # x = global_sort_pool(x, batch, self.k)
        # print(x.shape)
        emb = torch.reshape(emb,(x.shape[0],-1))
        x = torch.cat((x,emb),1)
        # x = F.normalize(x, 0.5, dim = 0)
        # x = x.unsqueeze(1)
        # x = torch.reshape(x,(x.shape[0],x.shape[2]))
        # print(x.shape)
        # x = self.conv1d1(x).relu()
        # x = self.maxpool1d(x)
        # x = self.conv1d2(x).relu()
        # print(x.shape)
        # x = self.mlp(x)  
        x = F.relu(self.lin1(x)).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x).relu()
        # x = self.lin3(x).relu()
        x = self.lin4(x)
        # x = self.lin3(x)
        return F.log_softmax(x, dim=1)


    def __repr__(self):
        return self.__class__.__name__
