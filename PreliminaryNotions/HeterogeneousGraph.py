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

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x['paper']

class HeteroGNN_changinglayer(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = HeteroConv({
            ('paper', 'cites', 'paper'): GCNConv(-1, hidden_channels),
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
            ('paper', 'rev_writes', 'author'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('paper', 'cites', 'paper'): SAGEConv((-1, -1), hidden_channels),
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
            ('paper', 'rev_writes', 'author'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        self.conv3 = HeteroConv({
            ('paper', 'cites', 'paper'): GCNConv(-1, hidden_channels),
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
            ('paper', 'rev_writes', 'author'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        return self.lin(x_dict['paper'])

class HeteroGNN_replicatedlayerConf(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = HeteroConv({
            ('paper', 'cites', 'paper'): GCNConv(-1, hidden_channels),
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
            ('paper', 'rev_writes', 'author'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('paper', 'cites', 'paper'): GCNConv(-1, hidden_channels),
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
            ('paper', 'rev_writes', 'author'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        self.conv3 = HeteroConv({
            ('paper', 'cites', 'paper'): GCNConv(-1, hidden_channels),
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
            ('paper', 'rev_writes', 'author'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        return self.lin(x_dict['paper'])

class HeteroGNN_iterativeLayer(torch.nn.Module):
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

class HGT_fixed(torch.nn.Module):
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

#data.node_types
#Out[1]: ['paper', 'author', 'institution', 'field_of_study']

class HGT_iterative(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channel2, hidden_channel3, out_channels, num_heads):
        super().__init__()

        self.lin_paper = Linear(128, hidden_channels)
        self.lin_author = Linear(128, hidden_channels)
        self.lin_institution = Linear(128, hidden_channels)
        self.lin_field_of_study = Linear(128, hidden_channels)

        self.conv1 = HGTConv(hidden_channels, hidden_channel2, data.metadata(),
                       num_heads, group='sum')
        self.conv2 = HGTConv(hidden_channel2, hidden_channel3, data.metadata(),
                       num_heads, group='sum')

        self.lin = Linear(hidden_channel3, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            if node_type == 'paper':
                x_dict[node_type] = self.lin_paper(x).relu_()
            if node_type == 'author':
                x_dict[node_type] = self.lin_author(x).relu_()
            if node_type == 'institution':
                x_dict[node_type] = self.lin_institution(x).relu_()
            if node_type == 'field_of_study':
                x_dict[node_type] = self.lin_field_of_study(x).relu_()

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)

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

train_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors=[15] * 2,
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=128,
    input_nodes=('paper', data['paper'].train_mask),
)

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

## model_GNN
#model = GNN(hidden_channels=64, out_channels=num_of_class)
#model = to_hetero(model, data.metadata(), aggr='sum')
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
#for i in range(1,5):
#    loss = train()
#    print(f"Current losso is: {loss}")



## model_HeteroGNN
loss_HeteroGNN_changinglayer=[]
model= HeteroGNN_changinglayer(hidden_channels=64, out_channels=num_of_class)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
model.to(device)
for i in range(1,numofep):
    print(f"Epoch {i}")
    loss = train()
    print(f"Current losso is: {loss}")
    loss_HeteroGNN_changinglayer.append(loss)

loss_HeteroGNN_replicatedlayerConf=[]
model= HeteroGNN_replicatedlayerConf(hidden_channels=64, out_channels=num_of_class)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
model.to(device)
for i in range(1,numofep):
    print(f"Epoch {i}")
    loss = train()
    print(f"Current losso is: {loss}")
    loss_HeteroGNN_replicatedlayerConf.append(loss)

loss_HeteroGNN_iterativeLayer=[]
model= HeteroGNN_iterativeLayer(hidden_channels=64, out_channels=num_of_class, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
model.to(device)
for i in range(1,numofep):
    print(f"Epoch {i}")
    loss = train()
    print(f"Current losso is: {loss}")
    loss_HeteroGNN_iterativeLayer.append(loss)

loss_HGT_iterative=[]
model = HGT_iterative(hidden_channels=64, out_channels=num_of_class, num_heads=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
model.to(device)
for i in range(1,numofep):
    print(f"Epoch {i}")
    loss = train()
    print(f"Current losso is: {loss}")
    loss_HGT_iterative.append(loss)

loss_HGT_fixed=[]
model = HGT_fixed(hidden_channels=64, out_channels=num_of_class, num_heads=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
model.to(device)
for i in range(1,numofep):
    print(f"Epoch {i}")
    loss = train()
    print(f"Current losso is: {loss}")
    loss_HGT_fixed.append(loss)

import matplotlib.pyplot as plt
# plot lines
plt.plot(loss_HeteroGNN_changinglayer, label="loss_HeteroGNN_changinglayer")
plt.plot(loss_HeteroGNN_replicatedlayerConf, label="loss_HeteroGNN_replicatedlayerConf")
plt.plot(loss_HeteroGNN_iterativeLayer, label="loss_HeteroGNN_iterativeLayer")
plt.plot(loss_HGT_iterative, label="loss_HGT_iterative")
plt.plot(loss_HGT_fixed, label="loss_HGT_fixed")
plt.legend()
plt.savefig("test.png")