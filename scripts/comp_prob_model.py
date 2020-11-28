import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from pdb import set_trace as bp


# append ../util directory to Python path
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'util'))

# import utility functions from util directory
from visualization import *
from cp_model import *
from temperature_scaling import *

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

# # looking at data
# plot_input_data(cp_df)

# get data averages and standard deviations
lat_dist_mean = np.mean(cp_df.lat_dist)
lat_dist_std = np.std(cp_df.lat_dist)
lon_dist_mean = np.mean(cp_df.lon_dist)
lon_dist_std = np.std(cp_df.lon_dist)
wr_prox_mean = np.mean(cp_df.wr_prox)
wr_prox_std = np.std(cp_df.wr_prox)
db_prox_mean = np.mean(cp_df.db_prox)
db_prox_std = np.std(cp_df.db_prox)
sl_prox_mean = np.mean(cp_df.sl_prox)
sl_prox_std = np.std(cp_df.sl_prox)
bl_prox_mean = np.mean(cp_df.bl_prox)
bl_prox_std = np.std(cp_df.bl_prox)
qb_vel_mean = np.mean(cp_df.qb_vel)
qb_vel_std = np.std(cp_df.qb_vel)
t_throw_mean = np.mean(cp_df.t_throw)
t_throw_std = np.std(cp_df.t_throw)

# setting up datasets
batch_size = 128
# val_split = 0.2
# random_seed= 42
cp_dataset = CPDataset(cp_df)
# cp_dataloader = DataLoader(cp_dataset)
# dataset_size = len(cp_dataloader)
# indices = list(range(dataset_size))
# split = int(np.floor(val_split * dataset_size))
# np.random.seed(random_seed)
# np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]
# np.save("../data/train_idxs.npy", np.array(train_indices))
# np.save("../data/val_idxs.npy", np.array(val_indices))
train_indices = np.load("../data/train_idxs.npy").tolist()
val_indices = np.load("../data/val_idxs.npy").tolist()
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
train_loader = torch.utils.data.DataLoader(cp_dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(cp_dataset, batch_size=batch_size,
                                                sampler=val_sampler)

# set up NN
B = batch_size
M = cp_df.shape[1]-4
H1 = 100
H2 = 50
H3 = 10
C = 2
p_dropout = 0.4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpnet = CPNet(B, M, H1, H2, H3, C, p_dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(cpnet.parameters())

# #################
# ##### TRAIN #####
# #################

# # train
# n_epochs = 200
# min_avg_val_loss = 100
# for epoch_idx in range(n_epochs):
#     running_loss = 0
#     progress_bar = tqdm(train_loader, desc='epoch {:.0f}'.format(epoch_idx))
#     cpnet.train()
#     for idx, samples in enumerate(progress_bar):
#         # load data
#         data = samples['data']
#         labels = samples['labels'].cuda(non_blocking=True).long()

#         # zero out gradients
#         optimizer.zero_grad()

#         # make prediction
#         x = data.cuda(non_blocking=True).float()
#         y = cpnet(x)

#         # update model
#         loss = criterion(y, labels)
#         loss.backward()
#         for param in cpnet.parameters():
#             param.grad.data.clamp_(-1,1)
#         optimizer.step()

#         # update progress bar
#         running_loss += loss.item()
#         avg_loss = running_loss/(idx+1)
#         disp_dict = {"train loss": avg_loss}
#         progress_bar.set_postfix(disp_dict)
#         progress_bar.update()

#     # run evaluation
#     cpnet.eval()
#     running_val_loss = 0
#     for idx, samples in enumerate(val_loader):
#         # load data
#         data = samples['data']
#         labels = samples['labels'].cuda(non_blocking=True).long()

#         # make prediction
#         x = data.cuda(non_blocking=True).float()
#         y = cpnet(x)

#         # get loss
#         loss = criterion(y, labels)
#         running_val_loss += loss.item()
#         avg_val_loss = running_val_loss/(idx+1)
    
#     progress_bar.close()
#     print("Epoch {:.0f}, Val loss: {:.3f}\n".format(epoch_idx,avg_val_loss))
    
#     # save model
#     if avg_val_loss < min_avg_val_loss:
#         min_avg_val_loss = avg_val_loss
#         ckpt_path = "../models/best_{:04d}_{:.3f}.pt".format(epoch_idx, avg_val_loss)
#         torch.save({'epoch': epoch_idx, \
#                     'model_state_dict': cpnet.state_dict(), \
#                     'optimizer_state_dict': optimizer.state_dict(), \
#                     'loss': loss.item()}, \
#                     ckpt_path)


###############
#### TEST #####
###############

# # load saved model
# ckpt_file = "../models/best_0194_0.483.pt"
# cpnet.load_state_dict(torch.load(ckpt_file)["model_state_dict"])
# cpnet.eval()

# # test uncalibrated model
# test_model(cpnet, cp_df, "Uncalibrated")

# # calibrate model
cpnet_calib = ModelWithTemperature(cpnet)
# cpnet_calib.set_temperature(val_loader)
# # save model
# ckpt_path = "../models/best_0194_0.483_calib.pt"
# torch.save({'model_state_dict': cpnet_calib.state_dict()}, ckpt_path)

# load saved model
ckpt_file = "../models/best_0194_0.483_calib.pt"
cpnet_calib.load_state_dict(torch.load(ckpt_file)["model_state_dict"])
cpnet_calib.eval().cuda()

# test calibrated model
val_df = cp_df.ix[val_indices,:]
train_df = cp_df.ix[train_indices,:]
test_model(cpnet_calib, val_df, "Calibrated")
# test_model(cpnet_calib, train_df, "Calibrated")
# test_model(cpnet_calib, cp_df, "Calibrated")



# #######################
# #### CONTRIBUTION #####
# #######################

# # load saved model
# cpnet_calib = ModelWithTemperature(cpnet)
# ckpt_file = "../models/best_0194_0.483_calib.pt"
# cpnet_calib.load_state_dict(torch.load(ckpt_file)["model_state_dict"])
# cpnet_calib.eval().cuda()

# # get average data
# x_avg = np.array([lat_dist_mean, lon_dist_mean, wr_prox_mean, db_prox_mean, sl_prox_mean, bl_prox_mean, qb_vel_mean, t_throw_mean])

# # find lat_dist contribution
# factor = 1
# delta_prob_lat_dist = get_contribution(cpnet_calib, x_avg, lat_dist_std, 0, factor)
# delta_prob_lon_dist = get_contribution(cpnet_calib, x_avg, lon_dist_std, 1, factor)
# delta_prob_wr_prox = get_contribution(cpnet_calib, x_avg, wr_prox_std, 2, factor)
# delta_prob_db_prox = get_contribution(cpnet_calib, x_avg, db_prox_std, 3, factor)
# delta_prob_sl_prox = get_contribution(cpnet_calib, x_avg, sl_prox_std, 4, factor)
# delta_prob_bl_prox = get_contribution(cpnet_calib, x_avg, bl_prox_std, 5, factor)
# delta_prob_qb_vel = get_contribution(cpnet_calib, x_avg, qb_vel_std, 6, factor)
# delta_prob_t_throw = get_contribution(cpnet_calib, x_avg, t_throw_std, 7, factor)

# delta_sum = delta_prob_lat_dist + delta_prob_lon_dist + delta_prob_wr_prox + delta_prob_db_prox + delta_prob_sl_prox + delta_prob_bl_prox + delta_prob_qb_vel + delta_prob_t_throw
# lat_dist_contrib = delta_prob_lat_dist/delta_sum
# lon_dist_contrib = delta_prob_lon_dist/delta_sum
# wr_prox_contrib = delta_prob_wr_prox/delta_sum
# db_prox_contrib = delta_prob_db_prox/delta_sum
# sl_prox_contrib = delta_prob_sl_prox/delta_sum
# bl_prox_contrib = delta_prob_bl_prox/delta_sum
# qb_vel_contrib = delta_prob_qb_vel/delta_sum
# t_throw_contrib = delta_prob_t_throw/delta_sum

# print("Lat_dist ({:.1f} +/- {:.1f}): {:.1f}%".format(lat_dist_mean, lat_dist_std, 100*lat_dist_contrib))
# print("Lon_dist ({:.1f} +/- {:.1f}): {:.1f}%".format(lon_dist_mean, lon_dist_std, 100*lon_dist_contrib))
# print("WR_prox ({:.1f} +/- {:.1f}): {:.1f}%".format(wr_prox_mean, wr_prox_std, 100*wr_prox_contrib))
# print("DB_prox ({:.1f} +/- {:.1f}): {:.1f}%".format(db_prox_mean, db_prox_std, 100*db_prox_contrib))
# print("SL_prox ({:.1f} +/- {:.1f}): {:.1f}%".format(sl_prox_mean, sl_prox_std, 100*sl_prox_contrib))
# print("BL_prox ({:.1f} +/- {:.1f}): {:.1f}%".format(bl_prox_mean, bl_prox_std, 100*bl_prox_contrib))
# print("QB_speed ({:.1f} +/- {:.1f}): {:.1f}%".format(qb_vel_mean, qb_vel_std, 100*qb_vel_contrib))
# print("T_throw ({:.1f} +/- {:.1f}): {:.1f}%".format(t_throw_mean, t_throw_std, 100*t_throw_contrib))
