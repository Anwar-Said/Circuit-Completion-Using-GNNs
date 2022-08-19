import os.path as osp
import sys, os
from shutil import rmtree
import torch
#from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree,from_networkx
import torch_geometric.transforms as T
sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
sys.path.append('%s/' % os.path.dirname(os.path.realpath(__file__)))
from utils import create_subgraphs, return_prob
from tu_dataset import TUDataset
from torch_geometric.data import Data, InMemoryDataset
import pdb
import pickle
import numpy as np
import networkx as nx

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, h, max_nodes_per_hop, node_label, use_rd,transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.h,self.max_nodes_per_hop, self.node_label, self.use_rd = h,max_nodes_per_hop, node_label, use_rd

#     @property
#     def raw_file_names(self):
#         return ['ltspice_examples_torch.pt']


    @property
    def processed_file_names(self):
        return ['ltspice_examples_dataset.pt']

#     def download(self):
#         # Download to `self.raw_dir`.
#         download_url(url, self.raw_dir)
#         ...

    def process(self):
        # Read data into huge `Data` list.
#         data_list = [...]
        print("processing data now!")
        # data_list = [torch.load('data/kicad_github_torch_LP.pt')]
        with open("data/ltspice_examples_GC.pkl",'rb') as f:
            data = pickle.load(f)
            f.close()
        with open('data/ltspice_examples_label_mapping.pkl', 'rb') as f:
            mapping = pickle.load(f)
            f.close()
        
        train_x = data['train_x']
        test_x = data['test_x'] 
        train_y = data['train_y'] 
        test_y = data['test_y']
        
        data_list = []
        for i, g in enumerate(train_x):
            X = []
            for j, n in enumerate(g.nodes()):
                feat = np.zeros((5,),dtype = np.float64)
                node_type = g.nodes[n]['type']
                index = mapping.get(node_type)
                feat[index] = 1.0
                X.append(feat)
            g_ = nx.Graph()
            g_.add_nodes_from(g.nodes())
            g_.add_edges_from(g.edges())
            d = from_networkx(g_)
            d.x = torch.tensor(np.array(X)).to(torch.float)
            d.y = torch.tensor(train_y[i])
            
            x = lambda d: create_subgraphs(d, 2, 1.0, None, 'spd', False)
            d.set = 'train'
            data_list.append(x(d))
        for i, g in enumerate(test_x):
            X = []
            for j, n in enumerate(g.nodes()):
                feat = np.zeros((5,),dtype = np.float64)
                node_type = g.nodes[n]['type']
                index = mapping.get(node_type)
                feat[index] = 1.0
                X.append(feat)
            g_ = nx.Graph()
            g_.add_nodes_from(g.nodes())
            g_.add_edges_from(g.edges())
            d = from_networkx(g_)
            d.x = torch.tensor(np.array(X)).to(torch.float)
            d.y = torch.tensor(test_y[i])
            
            x = lambda d: create_subgraphs(d, 2, 1.0, None, 'spd', False)
            d.set = 'test'
            data_list.append(x(d))


        print("length of datalist:", len(data_list))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])
        
class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, sparse=True, h=None, node_label='hop', use_rd=False, 
                use_rp=None, reprocess=False, clean=False, max_nodes_per_hop=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    pre_transform = None
    if h is not None:
        path += '/ngnn_h' + str(h)
        path += '_' + node_label
        if use_rd:
            path += '_rd'
        if max_nodes_per_hop is not None:
            path += '_mnph{}'.format(max_nodes_per_hop)

        pre_transform = lambda x: create_subgraphs(x, h, 1.0, max_nodes_per_hop, node_label, use_rd)

    if use_rp is not None:  # use RW return probability as additional features
        path += f'_rp{use_rp}'
        if pre_transform is None:
            pre_transform = return_prob(use_rp)
        else:
            pre_transform = T.Compose([return_prob(use_rp), pre_transform])

    if reprocess and os.path.isdir(path):
        rmtree(path)

    print(path)
    dataset = TUDataset(path, name, pre_transform=pre_transform, cleaned=clean)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset[torch.tensor(indices)]

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    return dataset
def get_circuit_dataset(name, sparse=True, h=None, node_label='hop', use_rd=False, 
                use_rp=None, reprocess=False, clean=False, max_nodes_per_hop=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    pre_transform = None
    if h is not None:
        path += '/ngnn_h' + str(h)
        path += '_' + node_label
        if use_rd:
            path += '_rd'
        if max_nodes_per_hop is not None:
            path += '_mnph{}'.format(max_nodes_per_hop)

        pre_transform = lambda x: create_subgraphs(x, h, 1.0, max_nodes_per_hop, node_label, use_rd)

    if use_rp is not None:  # use RW return probability as additional features
        path += f'_rp{use_rp}'
        if pre_transform is None:
            pre_transform = return_prob(use_rp)
        else:
            pre_transform = T.Compose([return_prob(use_rp), pre_transform])

    if reprocess and os.path.isdir(path):
        rmtree(path)

    print(path)
    
    # dataset = TUDataset(path, name, pre_transform=pre_transform, cleaned=clean)
    # dataset.data.edge_attr = None
    with open("data/"+name+"_GC.pkl",'rb') as f:
        data = pickle.load(f)
        f.close()
    with open('data/'+name+'_label_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
        f.close()
    
    train_x = data['train_x']
    test_x = data['test_x'] 
    train_y = data['train_y'] 
    test_y = data['test_y']
    
    dataset_train,dataset_test = [],[]
    for i, g in enumerate(train_x):
        X = []
        for j, n in enumerate(g.nodes()):
            feat = np.zeros((5,),dtype = np.float64)
            node_type = g.nodes[n]['type']
            index = mapping.get(node_type)
            feat[index] = 1.0
            X.append(feat)
        g_ = nx.Graph()
        g_.add_nodes_from(g.nodes())
        g_.add_edges_from(g.edges())
        d = from_networkx(g_)
        d.x = torch.tensor(np.array(X)).to(torch.float)
        d.y = torch.tensor(train_y[i])
        dataset_train.append(d)
    for i, g in enumerate(test_x):
        X = []
        for j, n in enumerate(g.nodes()):
            feat = np.zeros((5,),dtype = np.float64)
            node_type = g.nodes[n]['type']
            index = mapping.get(node_type)
            feat[index] = 1.0
            X.append(feat)
        g_ = nx.Graph()
        g_.add_nodes_from(g.nodes())
        g_.add_edges_from(g.edges())
        d = from_networkx(g_)
        d.x = torch.tensor(np.array(X)).to(torch.float)
        d.y = torch.tensor(test_y[i])
        dataset_test.append(d)
    
    return dataset_train, dataset_test

    # if not sparse:
    #     num_nodes = max_num_nodes = 0
    #     for data in dataset:
    #         num_nodes += data.num_nodes
    #         max_num_nodes = max(data.num_nodes, max_num_nodes)
    #     if name == 'REDDIT-BINARY':
    #         num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
    #     else:
    #         num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

    #     indices = []
    #     for i, data in enumerate(dataset):
    #         if data.num_nodes <= num_nodes:
    #             indices.append(i)
    #     dataset = dataset[torch.tensor(indices)]

    #     if dataset.transform is None:
    #         dataset.transform = T.ToDense(num_nodes)
    #     else:
    #         dataset.transform = T.Compose(
    #             [dataset.transform, T.ToDense(num_nodes)])

    return dataset
