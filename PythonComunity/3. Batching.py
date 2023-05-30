# Install required packages.
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('==================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===============================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.3f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Current device: {device}")
data.to(device)




from torch_geometric.loader import ClusterData, ClusterLoader

torch.manual_seed(12345)
cluster_data = ClusterData(data, num_parts=128)  # 1. Create subgraphs.
train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)  # 2. Stochastic partioning scheme.
print('===============================')
print('===============================')
print()
print(f'Original Dataset')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(data)
print()
print('===============================')
print()
total_num_nodes = 0
for step, sub_data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
    print(sub_data)
    total_num_nodes += sub_data.num_nodes
print()
print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')
print()
print('===============================')
print('===============================')


numOfEpoch = 30


def trainMiniBatch():
      model.train()

      for sub_data in train_loader:  # Iterate over each mini-batch.
          sub_data.to(device)
          out = model(sub_data.x, sub_data.edge_index)
          loss = criterion(out[sub_data.train_mask], sub_data.y[sub_data.train_mask])
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()


def trainFullBatch():
    model.train()
    # Work on the whole dataset
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)
      accs = []
      for mask in [data.train_mask, data.val_mask, data.test_mask]:
          correct = pred[mask] == data.y[mask]
          accs.append(int(correct.sum()) / int(mask.sum()))
      return accs

model = GCN(hidden_channels=16)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
model.to(device)
test_accuracy_miniBatch = []
for epoch in range(1, numOfEpoch):
    loss = trainMiniBatch()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    test_accuracy_miniBatch.append(test_acc)


del model
model = GCN(hidden_channels=16)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
model.to(device)

test_accuracy_fullBatch = []
for epoch in range(1, numOfEpoch):
    loss = trainFullBatch()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    test_accuracy_fullBatch.append(test_acc)


import matplotlib.pyplot as plt

# plot lines
plt.plot(test_accuracy_miniBatch, label="Accuracy Mini-Batch")
plt.plot(test_accuracy_fullBatch, label="Accuracy Full-Batch")
plt.legend()
plt.savefig("test.png")

print("Done")