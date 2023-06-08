#from ogb.graphproppred import PygGraphPropPredDataset
#from torch_geometric.data import DataLoader

# Download and process data at './dataset/ogbg_molhiv/'
#dataset = PygGraphPropPredDataset(name = "ogbl-ddi", root = 'dataset/')
#print(dataset)
#split_idx = dataset.get_idx_split()
#train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
#valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
#test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

#print(train_loader)
#print(dataset)

from ogb.linkproppred import PygLinkPropPredDataset
import argparse

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

#from logger import Logger
from torch_geometric.data import DataLoader

dataset = PygLinkPropPredDataset(name='ogbl-collab', transform=T.ToSparseTensor())
data = dataset[0]
print(data)

