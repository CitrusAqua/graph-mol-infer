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
from torch_geometric.transforms import BaseTransform

class SelectTargets(BaseTransform):
    '''
    Filter out unwanted targets
    '''
    def __init__(self, targets):
        self.targets = targets

    def __call__(self, data):
        data.y = data.y[:,self.targets]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(targets={self.targets})'


'''
================================================================================
QM9 dataset
Target #1 (ALPHA) #3 (LUMO) #7 (U0) #11 (CV) are used.
Please find:
http://quantum-machine.org/datasets/
https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
https://arxiv.org/pdf/2107.02381.pdf
================================================================================
'''
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

# Config
targets = [1, 3, 7, 11]
test_size = 1000
batch_size = 2048
random_seed = 42

print('-----------------')
print('Loading dataset.')

dataset = QM9(root='./datasets/data/QM9/')

# Select targets
dataset.data.y = dataset.data.y[:, targets]

# Normalize target values
y_mean = dataset.data.y.mean(dim=0)
y_std = dataset.data.y.std(dim=0)
dataset.data.y -= y_mean
dataset.data.y /= y_std

train_size = len(dataset) - test_size
train_set, test_set = torch.utils.data.random_split(
    dataset=dataset,
    lengths=[train_size, test_size],
    generator=torch.Generator().manual_seed(random_seed)
)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=test_size)

num_atom_features = dataset.data.x.shape[1]
num_edge_features = dataset.data.edge_attr.shape[1]
num_classes = dataset.data.y.shape[1]

print('Done.')
print(dataset)
print('Train size:', train_size)


'''
================================================================================
Load your model here
Also, set proper parameters for your optimizer
================================================================================
'''
from models.GEN import GEN
from torch.optim import lr_scheduler

print('-----------------')
print('Building model.')

model = GEN(
    in_channels=num_atom_features,
    edge_dim=num_edge_features,
    hidden_channels=64,
    num_layers=4,
    out_channels=num_classes
)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

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
fig, axs = plt.subplots(1, 4, figsize=(28, 6))
plt.show(block=False)

def update_scatter(y_true, y_pred):
    axs[0].clear()
    axs[1].clear()
    axs[2].clear()
    axs[3].clear()
    axs[0].set_title('ALPHA')
    axs[0].set_xlabel('Ground Truth')
    axs[0].set_ylabel('Prediction')
    axs[1].set_title('LUMO')
    axs[1].set_xlabel('Ground Truth')
    axs[1].set_ylabel('Prediction')
    axs[2].set_title('U0')
    axs[2].set_xlabel('Ground Truth')
    axs[2].set_ylabel('Prediction')
    axs[3].set_title('CV')
    axs[3].set_xlabel('Ground Truth')
    axs[3].set_ylabel('Prediction')
    if len(y_true) != 0:
        axs[0].scatter(y_true[:, 0], y_pred[:, 0], s=point_size, c=point_color)
        axs[1].scatter(y_true[:, 1], y_pred[:, 1], s=point_size, c=point_color)
        axs[2].scatter(y_true[:, 2], y_pred[:, 2], s=point_size, c=point_color)
        axs[3].scatter(y_true[:, 3], y_pred[:, 3], s=point_size, c=point_color)
    axs[0].set_ylim(axs[0].get_xlim())
    axs[1].set_ylim(axs[1].get_xlim())
    axs[2].set_ylim(axs[2].get_xlim())
    axs[3].set_ylim(axs[3].get_xlim())
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
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

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
        batch.to(device)
        out = model(batch)
        y_true = batch.y.cpu().detach().numpy()
        y_pred = out.cpu().detach().numpy()

    update_scatter(y_true, y_pred)

    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    n = test_size
    p = num_atom_features + num_edge_features
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)

    mae = np.mean(np.abs(y_pred - y_true), axis=0)

    print('Epoch {:03d} \t Avg. train loss: {:.4f}'.format(epoch, avg_loss))
    print('-')
    print('\t\t ALPHA ADJ R2: {:.4f} \t MAE: {:.4f}'.format(adj_r2[0], mae[0]))
    print('\t\t LUMO  ADJ R2: {:.4f} \t MAE: {:.4f}'.format(adj_r2[1], mae[1]))
    print('\t\t U0    ADJ R2: {:.4f} \t MAE: {:.4f}'.format(adj_r2[2], mae[2]))
    print('\t\t CV    ADJ R2: {:.4f} \t MAE: {:.4f}'.format(adj_r2[3], mae[3]))
    print('-')


'''
================================================================================
Save checkpoint
================================================================================
'''
# Not now!
#torch.save(model.state_dict(), 'checkpoints/weights.pt')