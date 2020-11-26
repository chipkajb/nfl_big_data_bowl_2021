import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
from pdb import set_trace as bp


# append ../util directory to Python path
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'util'))

# import utility functions from util directory
from visualization import *
from cp_model import *
from temperature_scaling import *
from war import *

# load completion probability data
# [week, game, play, lat_dist, lon_dist, wr_prox, db_prox, sl_prox, bl_prox, qb_vel, t_throw, completion]
cp_data = np.load("../data/comp_prob_data.npy")

# convert data to pandas
cp_df = pd.DataFrame(data = cp_data,
                     columns = ["week", "game", "play", "lat_dist", "lon_dist", "wr_prox", "db_prox", 
                                "sl_prox", "bl_prox", "qb_vel", "t_throw", "completion"])

# clean up data
cp_df = cp_df.query('not (wr_prox > 10 and completion == 1)') # sometimes receiver is not tracked
cp_df.bl_prox[cp_df.bl_prox.isnull()] = np.max(cp_df.bl_prox) # set nan values to max value

# get data average
db_prox_mean = np.mean(cp_df.db_prox)

# set up NN
B = 64
M = cp_df.shape[1]-4
H1 = 100
H2 = 50
H3 = 10
C = 2
p_dropout = 0.4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpnet = CPNet(B, M, H1, H2, H3, C, p_dropout).to(device)
cpnet_calib = ModelWithTemperature(cpnet)

# load saved model
ckpt_file = "../models/best_0194_0.483_calib.pt"
cpnet_calib.load_state_dict(torch.load(ckpt_file)["model_state_dict"])
cpnet_calib.eval().cuda()

make_sense = 0
progress_bar = tqdm(cp_df.iterrows(), total=cp_df.shape[0])
for idx, cp_data in progress_bar:
    x = torch.from_numpy(cp_data.iloc[3:11].values).cuda(non_blocking=True).float().unsqueeze(0)
    logits = cpnet_calib(x)
    softmax = F.softmax(logits, dim=1)
    comp_prob1 = softmax[0,1].item()

    if x[0,3] < db_prox_mean:
        guess = 'up'
    else:
        guess = 'down'
    x[0,3] = db_prox_mean
    logits = cpnet_calib(x)
    softmax = F.softmax(logits, dim=1)
    comp_prob2 = softmax[0,1].item()

    if (comp_prob2 < comp_prob1) and (guess == 'down'):
        make_sense += 1
    elif (comp_prob2 > comp_prob1) and (guess == 'up'):
        make_sense += 1

    disp_dict = {"make_sense_percentage": 100.0*make_sense/(idx+1)}
    progress_bar.set_postfix(disp_dict)
    progress_bar.update()

progress_bar.close()