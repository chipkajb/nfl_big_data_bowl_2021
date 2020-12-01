import pandas as pd
import numpy as np
from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

# Completion Probability Network (CPNet)
class CPNet(nn.Module):
    def __init__(self, B, M, H1, H2, H3, C, p_dropout):
        super(CPNet, self).__init__()
        self.linear1 = nn.Linear(M, H1)
        self.bn1 = nn.BatchNorm1d(H1)
        self.dropout1 = nn.Dropout(p_dropout)
        self.linear2 = nn.Linear(H1, H2)
        self.bn2 = nn.BatchNorm1d(H2)
        self.dropout2 = nn.Dropout(p_dropout/2.0)
        self.linear3 = nn.Linear(H2, H3)
        self.bn3 = nn.BatchNorm1d(H3)
        self.dropout3 = nn.Dropout(p_dropout/4.0)
        self.head = nn.Linear(H3, C)
        # self.activation = nn.ReLU()
        # self.activation = nn.LeakyReLU()
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.activation(self.dropout1(self.bn1(self.linear1(x))))
        x = self.activation(self.dropout2(self.bn2(self.linear2(x))))
        x = self.activation(self.dropout3(self.bn3(self.linear3(x))))
        x = self.head(x)
        return x

# Completion Probability dataset
class CPDataset(Dataset):
    def __init__(self, cp_df):
        self.cp_df = cp_df

    def __len__(self):
        return len(self.cp_df)

    def __getitem__(self, idx):
        data = self.cp_df.iloc[idx,3:-1].values
        labels = self.cp_df.iloc[idx,-1]
        sample = {'data': data, 'labels': labels}

        return sample

# load and clean data
def load_and_clean_data(keep_routes=False):
    # load data
    train_df = pd.read_csv('../input/nfl-bdb-data/training_data.csv')

    # clean up data
    train_df = train_df.query('not (ball_prox > 10 and result == 1)') # sometimes receiver is not tracked
    train_df.bl_prox[train_df.bl_prox.isnull()] = np.max(train_df.bl_prox) # set nan values to max value

    if not keep_routes:
        train_df = train_df.drop(columns=["route"])
    
    return train_df

# set up datasets
def setup_datasets(train_df):
    batch_size = 128
    dataset = CPDataset(train_df)
    train_indices = np.load("../input/nfl-bdb-data/train_idxs.npy").tolist()
    val_indices = np.load("../input/nfl-bdb-data/val_idxs.npy").tolist()
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    return train_loader, val_loader

# set up neural network
def setup_network(train_df):
    B = 128
    M = train_df.shape[1]-4
    H1 = 100
    H2 = 50
    H3 = 10
    C = 2
    p_dropout = 0.4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpnet = CPNet(B, M, H1, H2, H3, C, p_dropout).to(device)
    return cpnet

# train model
def train_model():
    train_df = load_and_clean_data()
    train_loader, val_loader = setup_datasets(train_df)
    cpnet = setup_network(train_df)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cpnet.parameters())

    # train
    n_epochs = 200
    min_avg_val_loss = 100
    for epoch_idx in range(n_epochs):
        running_loss = 0
        progress_bar = tqdm(train_loader, desc='epoch {:.0f}'.format(epoch_idx))
        cpnet.train()
        for idx, samples in enumerate(progress_bar):
            # load data
            data = samples['data']
            labels = samples['labels'].cuda(non_blocking=True).long()

            # zero out gradients
            optimizer.zero_grad()

            # make prediction
            x = data.cuda(non_blocking=True).float()
            y = cpnet(x)

            # update model
            loss = criterion(y, labels)
            loss.backward()
            for param in cpnet.parameters():
                param.grad.data.clamp_(-1,1)
            optimizer.step()

            # update progress bar
            running_loss += loss.item()
            avg_loss = running_loss/(idx+1)
            disp_dict = {"train loss": avg_loss}
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

        # run evaluation
        cpnet.eval()
        running_val_loss = 0
        for idx, samples in enumerate(val_loader):
            # load data
            data = samples['data']
            labels = samples['labels'].cuda(non_blocking=True).long()

            # make prediction
            x = data.cuda(non_blocking=True).float()
            y = cpnet(x)

            # get loss
            loss = criterion(y, labels)
            running_val_loss += loss.item()
            avg_val_loss = running_val_loss/(idx+1)
        
        progress_bar.close()
        print("Epoch {:.0f}, Val loss: {:.3f}\n".format(epoch_idx,avg_val_loss))
        
        # save model
        if avg_val_loss < min_avg_val_loss:
            min_avg_val_loss = avg_val_loss
            ckpt_path = "../input/nfl-bdb-data/best_{:04d}_{:.3f}.pt".format(epoch_idx, avg_val_loss)
            torch.save({'epoch': epoch_idx, \
                        'model_state_dict': cpnet.state_dict(), \
                        'optimizer_state_dict': optimizer.state_dict(), \
                        'loss': loss.item()}, \
                        ckpt_path)