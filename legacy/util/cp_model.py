from pdb import set_trace as bp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px

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

def test_model(net, cp_df, mode):
    # get metrics
    correct = 0
    incorrect = 0
    progress_bar = tqdm(cp_df.iterrows(), total=cp_df.shape[0], desc=mode)
    metrics = np.empty((0,2))
    for idx, cp_data in progress_bar:
        gt = cp_data.completion
        x = torch.from_numpy(cp_data.iloc[3:11].values).cuda(non_blocking=True).float().unsqueeze(0)
        logits = net(x)
        softmax = F.softmax(logits, dim=1)
        conf, pred = torch.max(softmax, 1)
        conf = conf.cpu().detach().numpy()[0]
        pred = pred.cpu().detach().numpy()[0]

        if pred == 1:
            new_row = np.array([[conf, gt]])
            metrics = np.append(metrics, new_row, axis=0)
        else:
            new_row = np.array([[1-conf, gt]])
            metrics = np.append(metrics, new_row, axis=0)

        if pred == gt:
            correct += 1
        else:
            incorrect += 1

        percentage = 100.0*correct/(incorrect + correct)
        disp_dict = {"percent_correct": percentage}
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()

    # generate histogram data
    n_bins = 10
    bounds = np.linspace(0,1,n_bins+1)
    actual_confs = np.empty((0,1))
    for i in range(len(bounds)-1):
        in_range_sel = (metrics[:,0] > bounds[i]) & (metrics[:,0] < bounds[i+1])
        metrics_in_range = metrics[in_range_sel]
        if len(metrics_in_range) > 0:
            actual_conf = np.sum(metrics_in_range[:,1]) / len(metrics_in_range)
            actual_confs = np.append(actual_confs, np.array([[actual_conf]]), axis=0)
        else:
            actual_conf = 0
            actual_confs = np.append(actual_confs, np.array([[actual_conf]]), axis=0)

    # plot histogram
    actual_confs = actual_confs.reshape(-1)
    width = bounds[1] - bounds[0]
    pred_confs = (bounds - 0.5*width)[1::]
    fig, ax = plt.subplots(1, 1)
    ax.bar(pred_confs, actual_confs, width=width, color=(0.1,0.1,0.7), edgecolor='black')
    ax.grid()
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')
    plt.xlabel('Predicted Completion Probability')
    plt.ylabel('Actual Completion Probability') 
    plt.title(mode + ' Completion Probability Model') 
    plt.xlim(0,1)
    plt.ylim(0,1)
    axes2 = plt.twinx()
    axes2.plot([0,1], [0,1], color='gray', linestyle='dashed')
    plt.ylim(0,1)
    axes2.tick_params(
        axis='y',
        which='both',
        right=False,
        left=False,
        labelright=False)
    plt.show()

def plot_input_data(cp_df):
    # plotting data
    fig = px.scatter(cp_df,
                    x=cp_df.index,
                    y="lat_dist",
                    color=cp_df["completion"].astype(str),
                    hover_data=["week", "game", "play"],
                    title="Lateral Pass Distance")
    fig.show()

    fig = px.scatter(cp_df,
                    x=cp_df.index,
                    y="lon_dist",
                    color=cp_df["completion"].astype(str),
                    hover_data=["week", "game", "play"],
                    title="Longitudinal Pass Distance")
    fig.show()

    fig = px.scatter(cp_df,
                    x=cp_df.index,
                    y="wr_prox",
                    color=cp_df["completion"].astype(str),
                    hover_data=["week", "game", "play"],
                    title="WR-Ball Proximity")
    fig.show()

    fig = px.scatter(cp_df,
                    x=cp_df.index,
                    y="db_prox",
                    color=cp_df["completion"].astype(str),
                    hover_data=["week", "game", "play"],
                    title="DB-WR Proximity")
    fig.show()

    fig = px.scatter(cp_df,
                    x=cp_df.index,
                    y="sl_prox",
                    color=cp_df["completion"].astype(str),
                    hover_data=["week", "game", "play"],
                    title="WR-Sideline Proximity")
    fig.show()

    fig = px.scatter(cp_df,
                    x=cp_df.index,
                    y="bl_prox",
                    color=cp_df["completion"].astype(str),
                    hover_data=["week", "game", "play"],
                    title="Blitzer-QB Proximity")
    fig.show()

    fig = px.scatter(cp_df,
                    x=cp_df.index,
                    y="qb_vel",
                    color=cp_df["completion"].astype(str),
                    hover_data=["week", "game", "play"],
                    title="QB Speed")
    fig.show()

    fig = px.scatter(cp_df,
                    x=cp_df.index,
                    y="t_throw",
                    color=cp_df["completion"].astype(str),
                    hover_data=["week", "game", "play"],
                    title="Time to Throw")
    fig.show()


def get_contribution(net, x, std, idx, factor):
    x1 = np.copy(x)
    x2 = np.copy(x)
    x1[idx] = x1[idx] - factor*std
    x2[idx] = x2[idx] + factor*std
    x1 = torch.from_numpy(x1).cuda(non_blocking=True).float().unsqueeze(0)
    x2 = torch.from_numpy(x2).cuda(non_blocking=True).float().unsqueeze(0)
    logits1 = net(x1)
    logits2 = net(x2)
    softmax1 = F.softmax(logits1, dim=1)
    softmax2 = F.softmax(logits2, dim=1)
    comp_prob1 = softmax1[0,1].item()
    comp_prob2 = softmax2[0,1].item()
    delta_comp_prob = np.abs(comp_prob2-comp_prob1)
    return delta_comp_prob
