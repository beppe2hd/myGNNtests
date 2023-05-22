import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='PubMed',
                    transform=NormalizeFeatures())

print(f'Dataset: {dataset}')

data = dataset[0]  # Get the first graph object.
print(f"Dataset: {data}")

from torch_geometric.loader import ClusterData, ClusterLoader

cluster_data = ClusterData(data, num_parts=128)  # 1. Create subgraphs.
train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)  # 2. Stochastic partioning scheme.

sumOfNodes = 0
for i, cluster in enumerate(train_loader):
    sumOfNodes += len(cluster.y)
    print(f"Cluster {i} is: {cluster}")

print(f"The total number of node is {sumOfNodes}")