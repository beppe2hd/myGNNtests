import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='PubMed',
                    transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}')

data = dataset[0]  # Get the first graph object.

import torch_sparse
from torch_geometric.loader import ClusterData, ClusterLoader

seed = 42
torch.manual_seed(seed)
cluster_data = ClusterData(data, num_parts=128)  # 1. Create subgraphs.
train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)  # 2. Stochastic partioning scheme.