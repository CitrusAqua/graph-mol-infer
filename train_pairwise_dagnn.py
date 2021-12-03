import numpy as np
import torch


if torch.cuda.is_available():
    print('Using CUDA')
    device = 'cuda'
else:
    print('Using CPU')
    device = 'cpu'


'''
================================================================================
Utils
I put utils here for convenience
================================================================================
'''
from itertools import combinations

def gen_pairs(batch):
    num_batches = batch.max()
    x_indices = np.array(list(range(len(batch))))
    pairs = []
    batch_np = batch.numpy()
    for i in range(num_batches):
        sub_indices = x_indices[batch_np == i]
        combs = combinations(sub_indices, 2)
        pairs.extend(combs)
    return pairs


'''
================================================================================
QM9 dataset
http://quantum-machine.org/datasets/
https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
================================================================================
'''
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

# Config
train_size = 100000
test_size = 1000
batch_size = 1280
random_seed = 42

print('-----------------')
print('Loading dataset.')

dataset = QM9(root='./datasets/data/QM9/')

unused_size = len(dataset) - train_size - test_size
train_set, test_set, unused_set = torch.utils.data.random_split(
    dataset=dataset,
    lengths=[train_size, test_size, unused_size],
    generator=torch.Generator().manual_seed(random_seed)
)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=test_size)

num_node_features = dataset.data.x.shape[1]
num_classes = 1

print('Done.')
print(dataset)
print('Train size:', train_size)


'''
================================================================================
Load your model here
Also, set proper parameters for your optimizer
================================================================================
'''
from models.deepergcn_dagnn import DeeperDAGNN_node_Virtualnode
from torch.optim import lr_scheduler

print('-----------------')
print('Building model.')

model = DeeperDAGNN_node_Virtualnode(
    atom_dim=num_node_features,
    edge_dim=4,
    num_layers=6,
    emb_dim=64
)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print('Done.')
print(model)


'''
================================================================================
Visualization
================================================================================
'''


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Config
point_size = 4
point_color = '#2A52BE'

plt.ion()
fig = plt.figure(figsize=(6,6))
plt.show(block=False)

def update_scatter(y_true, y_pred):
    plt.clf()
    plt.title('Pairwise Distance')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.xlim([0, 12])
    plt.ylim([0, 12])
    if len(y_true) != 0:
        plt.scatter(y_true, y_pred, s=point_size, c=point_color)
    plt.pause(0.0001)
    plt.draw()
    plt.pause(0.0001)

update_scatter([], [])


'''
================================================================================
Do training
================================================================================
'''
from sklearn.metrics import r2_score

# Config
num_epochs = 1000
display_every = 99999

print('-----------------')
print('Start training...')
print('-')
for epoch in range(num_epochs):

    model.train()
    sum_loss = 0
    for iter, batch in enumerate(train_loader):

        # Calculate pairs and ground truth distances
        # before transfering to GPU
        pairs = gen_pairs(batch.batch)
        coords = batch.pos[pairs]
        pdist_gt = (coords[:, 0, :]-coords[:, 1, :]).pow(2).sum(dim=1).sqrt().unsqueeze(-1)

        # forward
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch, pairs).cpu()

        loss = F.mse_loss(out, pdist_gt)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

        #print('Iteration', iter+1, '/', len(train_loader), 'loss', loss.item())

        if (iter+1) % display_every == 0:
            model.eval()
            for test_batch in test_loader:
                test_batch.to(device)
                out = model(test_batch)
                y_true = test_batch.y.cpu().detach().numpy()
                y_pred = out.cpu().detach().numpy()
            update_scatter(y_true, y_pred)
            model.train()

    avg_loss = sum_loss / len(train_loader) / num_classes
    scheduler.step()

    model.eval()
    for batch in test_loader:

        # Calculate pairs and ground truth distances
        # before transfering to GPU
        pairs = gen_pairs(batch.batch)
        coords = batch.pos[pairs]
        pdist_gt = (coords[:, 0, :]-coords[:, 1, :]).pow(2).sum(dim=1).sqrt().unsqueeze(-1)

        # forward
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch, pairs)
        pdist_pred = out.cpu().detach().numpy()
        pdist_gt = pdist_gt.numpy()

    update_scatter(pdist_gt, pdist_pred)

    r2 = r2_score(pdist_gt, pdist_pred)
    n = pdist_gt.shape[0]
    p = num_node_features
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    mae = np.mean(np.abs(pdist_pred - pdist_gt))

    print('Epoch {:03d} \t Avg. train loss: {:.4f} \t DISTANCE ADJ R2: {:.4f} \t MAE: {:.4f}'.format(epoch, avg_loss, adj_r2, mae))


'''
================================================================================
Save checkpoint
================================================================================
'''
# Not now!
#torch.save(model.state_dict(), 'checkpoints/weights.pt')