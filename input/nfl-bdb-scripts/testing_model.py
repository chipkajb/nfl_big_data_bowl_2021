from pdb import set_trace as bp
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from training_model import *
from temperature_scaling import *

# test model
def test_model(net, cp_df, mode):
    # get metrics
    correct = 0
    incorrect = 0
    progress_bar = tqdm(cp_df.iterrows(), total=cp_df.shape[0], desc=mode)
    metrics = np.empty((0,2))
    for idx, cp_data in progress_bar:
        gt = cp_data.result
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

# test uncalibrated model
def test_uncalibrated_model(model_path):
    # load data
    train_df = load_and_clean_data()

    # set up network
    cpnet = setup_network(train_df)

    # load saved model
    cpnet.load_state_dict(torch.load(model_path)["model_state_dict"])
    cpnet.eval()

    # test uncalibrated model
    test_model(cpnet, train_df, "Uncalibrated")

# calibrate model
def calibrate_model(model_input_path, model_output_path):
    # load data
    train_df = load_and_clean_data()

    # setup datasets
    _, val_loader = setup_datasets(train_df)

    # set up network
    cpnet = setup_network(train_df)

    # load saved model
    cpnet.load_state_dict(torch.load(model_input_path)["model_state_dict"])
    cpnet.eval()

    # set up model with temperature
    cpnet_calib = ModelWithTemperature(cpnet)
    cpnet_calib.set_temperature(val_loader)

    # save model
    torch.save({'model_state_dict': cpnet_calib.state_dict()}, model_output_path)

# test calibrated model
def test_calibrated_model(model_path):
    # load data
    train_df = load_and_clean_data()

    # set up network
    cpnet = setup_network(train_df)

    # set up model with temperature
    cpnet_calib = ModelWithTemperature(cpnet)

    # load saved model
    cpnet_calib.load_state_dict(torch.load(model_path)["model_state_dict"])
    cpnet_calib.eval().cuda()

    # test calibrated model
    val_indices = np.load("../input/nfl-bdb-data/val_idxs.npy").tolist()
    val_df = train_df.ix[val_indices,:]
    test_model(cpnet_calib, val_df, "Calibrated")

# get contributions helper functions
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

# check contribution amounts
def check_contributions(model_path):
    # load data
    train_df = load_and_clean_data()

    # set up network
    cpnet = setup_network(train_df)

    # load saved model
    cpnet_calib = ModelWithTemperature(cpnet)
    cpnet_calib.load_state_dict(torch.load(model_path)["model_state_dict"])
    cpnet_calib.eval().cuda()

    # get data metrics
    lat_dist_mean = np.mean(train_df.lat_dist)
    lon_dist_mean = np.mean(train_df.lon_dist)
    ball_prox_mean = np.mean(train_df.ball_prox)
    db_prox_mean = np.mean(train_df.db_prox)
    sl_prox_mean = np.mean(train_df.sl_prox)
    bl_prox_mean = np.mean(train_df.bl_prox)
    qb_speed_mean = np.mean(train_df.qb_speed)
    t_throw_mean = np.mean(train_df.t_throw)
    lat_dist_std = np.std(train_df.lat_dist)
    lon_dist_std = np.std(train_df.lon_dist)
    ball_prox_std = np.std(train_df.ball_prox)
    db_prox_std = np.std(train_df.db_prox)
    sl_prox_std = np.std(train_df.sl_prox)
    bl_prox_std = np.std(train_df.bl_prox)
    qb_speed_std = np.std(train_df.qb_speed)
    t_throw_std = np.std(train_df.t_throw)
    x_avg = np.array([lat_dist_mean, lon_dist_mean, ball_prox_mean, db_prox_mean, sl_prox_mean, \
                    bl_prox_mean, qb_speed_mean, t_throw_mean])

    # find lat_dist contribution
    factor = 1
    delta_prob_lat_dist = get_contribution(cpnet_calib, x_avg, lat_dist_std, 0, factor)
    delta_prob_lon_dist = get_contribution(cpnet_calib, x_avg, lon_dist_std, 1, factor)
    delta_prob_wr_prox = get_contribution(cpnet_calib, x_avg, ball_prox_std, 2, factor)
    delta_prob_db_prox = get_contribution(cpnet_calib, x_avg, db_prox_std, 3, factor)
    delta_prob_sl_prox = get_contribution(cpnet_calib, x_avg, sl_prox_std, 4, factor)
    delta_prob_bl_prox = get_contribution(cpnet_calib, x_avg, bl_prox_std, 5, factor)
    delta_prob_qb_vel = get_contribution(cpnet_calib, x_avg, qb_speed_std, 6, factor)
    delta_prob_t_throw = get_contribution(cpnet_calib, x_avg, t_throw_std, 7, factor)

    delta_sum = delta_prob_lat_dist + delta_prob_lon_dist + delta_prob_wr_prox + delta_prob_db_prox + delta_prob_sl_prox + delta_prob_bl_prox + delta_prob_qb_vel + delta_prob_t_throw
    lat_dist_contrib = delta_prob_lat_dist/delta_sum
    lon_dist_contrib = delta_prob_lon_dist/delta_sum
    wr_prox_contrib = delta_prob_wr_prox/delta_sum
    db_prox_contrib = delta_prob_db_prox/delta_sum
    sl_prox_contrib = delta_prob_sl_prox/delta_sum
    bl_prox_contrib = delta_prob_bl_prox/delta_sum
    qb_vel_contrib = delta_prob_qb_vel/delta_sum
    t_throw_contrib = delta_prob_t_throw/delta_sum

    print("Lat_dist ({:.1f} +/- {:.1f}): {:.1f}%".format(lat_dist_mean, lat_dist_std, 100*lat_dist_contrib))
    print("Lon_dist ({:.1f} +/- {:.1f}): {:.1f}%".format(lon_dist_mean, lon_dist_std, 100*lon_dist_contrib))
    print("WR_prox ({:.1f} +/- {:.1f}): {:.1f}%".format(ball_prox_mean, ball_prox_std, 100*wr_prox_contrib))
    print("DB_prox ({:.1f} +/- {:.1f}): {:.1f}%".format(db_prox_mean, db_prox_std, 100*db_prox_contrib))
    print("SL_prox ({:.1f} +/- {:.1f}): {:.1f}%".format(sl_prox_mean, sl_prox_std, 100*sl_prox_contrib))
    print("BL_prox ({:.1f} +/- {:.1f}): {:.1f}%".format(bl_prox_mean, bl_prox_std, 100*bl_prox_contrib))
    print("QB_speed ({:.1f} +/- {:.1f}): {:.1f}%".format(qb_speed_mean, qb_speed_std, 100*qb_vel_contrib))
    print("T_throw ({:.1f} +/- {:.1f}): {:.1f}%".format(t_throw_mean, t_throw_std, 100*t_throw_contrib))