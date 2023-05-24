import os
import torch
from tqdm import tqdm
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GATConv, Linear, HGTConv, HeteroConv, GCNConv, to_hetero

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--numofepoch", help="number of epoches", type=int,
                    default = 5)
args = parser.parse_args()
numofep =  args.numofepoch
print(f"Running on :{numofep}")


if torch.cuda.is_available():
    device = "cuda"
#elif torch.backends.mps.is_available():
#    device = "mps"
else:
    device = "cpu"

dataset = OGB_MAG(root='./data', preprocess='metapath2vec', transform=T.ToUndirected())
data = dataset[0]
num_of_class = dataset.num_classes


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('paper', 'cites', 'paper'): GCNConv(-1, hidden_channels),
                ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
                ('paper', 'rev_writes', 'author'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['paper'])

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['paper'])

def train():
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        batch.to(device)
        #model.to(device)
        optimizer.zero_grad()
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = criterion(out[:batch_size], batch['paper'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


def test():
    model.eval()
    total_examples = total_loss = 0
    for batch in tqdm(test_loader):
        batch.to(device)
        #model.to(device)
        optimizer.zero_grad()
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)

        pred = out.argmax(dim=1)
        correct = pred[:batch_size] == batch['paper'].y[:batch_size]
        accs = int(correct.sum()) / int(batch_size)
        return accs


    return total_loss / total_examples

train_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors=[15] * 2,
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=128,
    input_nodes=('paper', data['paper'].train_mask),
)

test_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors=[15] * 2,
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=128,
    input_nodes=('paper', data['paper'].test_mask),
)

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

'''
## model_HeteroGNN
loss_HeteroGNN = []
acc_HeteroGNN = []
model= HeteroGNN(hidden_channels=64, out_channels=num_of_class, num_layers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
model.to(device)
for i in range(1,numofep):
    print(f"Epoch {i}")
    loss = train()
    print(f"Current losso is: {loss}")
    loss_HeteroGNN.append(loss)
    acc_HeteroGNN.append(test())
'''
loss_HGT = []
acc_HGT = []
model= HGT(hidden_channels=64, out_channels=num_of_class, num_heads=4, num_layers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
model.to(device)
for i in range(1,numofep):
    print(f"Epoch {i}")
    loss = train()
    print(f"Current losso is: {loss}")
    loss_HGT.append(loss)
    acc_HGT.append(test())


import matplotlib.pyplot as plt
# plot lines

plt.plot(loss_HeteroGNN, label="loss_HeteroGNN")
plt.plot(loss_HGT, label="loss_HGT")
plt.legend()
plt.savefig("loss200.png")

plt.close()

plt.plot(acc_HeteroGNN, label="acc_HeteroGNN")
plt.plot(acc_HGT, label="acc_HGT")
plt.legend()
plt.savefig("acc200.png")

