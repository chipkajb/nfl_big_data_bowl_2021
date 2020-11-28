import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import trange
import torch
import torch.optim as optim
from pdb import set_trace as bp


# append ../util directory to Python path
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'util'))

# import utility functions from util directory
from cp_data import *
from cp_model import *
from temperature_scaling import *


# ####################
# ##### GET DATA #####
# ####################

# # load completion probability data
# # [week, game, play, lat_dist, lon_dist, wr_prox, db_prox, sl_prox, bl_prox, qb_vel, t_throw, completion]
# cp_data = np.load("../data/comp_prob_data.npy")

# # convert data to pandas
# cp_df = pd.DataFrame(data = cp_data,
#                      columns = ["week", "game", "play", "lat_dist", "lon_dist", "wr_prox", "db_prox", 
#                                 "sl_prox", "bl_prox", "qb_vel", "t_throw", "completion"])

# # clean up data
# cp_df = cp_df.query('not (wr_prox > 10 and completion == 1)') # sometimes receiver is not tracked
# cp_df.bl_prox[cp_df.bl_prox.isnull()] = np.max(cp_df.bl_prox) # set nan values to max value

# # get data average
# db_prox_mean = np.mean(cp_df.db_prox)

# # set up NN
# B = 64
# M = cp_df.shape[1]-4
# H1 = 100
# H2 = 50
# H3 = 10
# C = 2
# p_dropout = 0.4
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cpnet = CPNet(B, M, H1, H2, H3, C, p_dropout).to(device)
# cpnet_calib = ModelWithTemperature(cpnet)

# # load saved model
# ckpt_file = "../models/best_0194_0.483_calib.pt"
# cpnet_calib.load_state_dict(torch.load(ckpt_file)["model_state_dict"])
# cpnet_calib.eval().cuda()

# # load all data
# plays_df = pd.read_csv('../data/plays.csv')
# games_df = pd.read_csv('../data/games.csv')
# data_dict = {}
# for i in trange(1,18, desc='Loading data'):
#     data_dict[i] = pd.read_csv('../data/week{:.0f}.csv'.format(i))

# make_sense = 0
# progress_bar = tqdm(cp_df.iterrows(), total=cp_df.shape[0])
# column_names = ["def_name", "def_pos", "off_name", "off_pos", "route", "ipar", "comp_prob", "result", "epa"]
# metrics_df = pd.DataFrame(data = np.empty((0,len(column_names))),
#                           columns = column_names)
# for idx, cp_data in progress_bar:
#     # find original completion probability
#     x = torch.from_numpy(cp_data.iloc[3:11].values).cuda(non_blocking=True).float().unsqueeze(0)
#     logits1 = cpnet_calib(x)
#     softmax1 = F.softmax(logits1, dim=1)
#     comp_prob1 = softmax1[0,1].item()
#     db_prox_orig = x[0,3].item()

#     # find "replacement" completion probability
#     x[0,3] = db_prox_mean
#     logits2 = cpnet_calib(x)
#     softmax2 = F.softmax(logits2, dim=1)
#     comp_prob2 = softmax2[0,1].item()

#     # sanity check
#     ipar = 100.0*(comp_prob2 - comp_prob1)
#     if db_prox_orig < db_prox_mean:
#         guess = 'up'
#     else:
#         guess = 'down'
#     if (comp_prob2 < comp_prob1) and (guess == 'down'):
#         make_sense += 1
#     elif (comp_prob2 > comp_prob1) and (guess == 'up'):
#         make_sense += 1

#     # get play dataframe
#     week = cp_data.week
#     game_id = cp_data.game
#     play_id = cp_data.play
#     week_df = data_dict[week]
#     play_df = week_df.query('gameId == {:.0f} and playId == {:.0f}'.format(game_id, play_id))

#     # get possession info
#     possession_team = plays_df.query('gameId == {:.0f} and playId == {:.0f}'.format(game_id, play_id)).possessionTeam.iloc[0]
#     home_team = games_df.query('gameId == {:.0f}'.format(game_id)).homeTeamAbbr.iloc[0]
#     away_team = games_df.query('gameId == {:.0f}'.format(game_id)).visitorTeamAbbr.iloc[0]
#     if possession_team == home_team:
#         possession = 'home'
#     elif possession_team == away_team:
#         possession = 'away'
#     else:
#         print("Cannot determine who has ball")

#     # find defender/receiver info
#     ret, pass_arrived_df = find_pass_arrived_df(play_df, possession)
#     off_pass_arrived_df = pass_arrived_df.query('team == "' + possession + '"')
#     def_pass_arrived_df = pass_arrived_df.query('team != "' + possession + '" and displayName != "Football"')
#     off_players_xy = off_pass_arrived_df[["x","y"]].values
#     def_players_xy = def_pass_arrived_df[["x","y"]].values
#     football_xy = pass_arrived_df.query('displayName == "Football"')[["x","y"]].values
#     receiver_idx = np.argmin(np.linalg.norm(football_xy - off_players_xy, axis=1))
#     defender_idx = np.argmin(np.linalg.norm(football_xy - def_players_xy, axis=1))
#     receiver_df = off_pass_arrived_df.iloc[receiver_idx]
#     defender_df = def_pass_arrived_df.iloc[defender_idx]

#     # find whether the pass was complete
#     pass_result = plays_df.query('gameId == {:.0f} and playId == {:.0f}'.format(game_id, play_id)).passResult.item()

#     # get play's epa
#     epa = -plays_df.query('gameId == {:.0f} and playId == {:.0f}'.format(game_id, play_id)).epa.item()

#     # append results
#     new_row = pd.DataFrame([[defender_df.displayName, defender_df.position,receiver_df.displayName,
#                              receiver_df.position,receiver_df.route, ipar, comp_prob1,
#                              pass_result, epa]], columns=column_names)
#     metrics_df = metrics_df.append(new_row, ignore_index=True)

#     # update progress bar
#     disp_dict = {"make_sense_percentage": 100.0*make_sense/(idx+1)}
#     progress_bar.set_postfix(disp_dict)
#     progress_bar.update()

# progress_bar.close()

# metrics_df.to_csv("../data/ipar.csv", index=False)



########################
##### ANALYZE DATA #####
########################

metrics_df = pd.read_csv('../data/ipar.csv')

column_names = ["name", "position", "n_throws", "ipar", "incomp_per", \
                "int_per", "fool_incomp_to_comp", \
                "fool_comp_to_incomp", "avg_comp_prob", "epa"]
rankings_df = pd.DataFrame(data = np.empty((0,len(column_names))), columns = column_names)
defenders = np.unique(metrics_df.def_name)
routes = np.unique(metrics_df.route[metrics_df.route.notnull()])
for defender in tqdm(defenders, desc="Analyzing data"):
    defender_data = metrics_df.query('def_name == "' + defender + '"')
    def_name = defender_data.def_name.iloc[0]
    def_pos = defender_data.def_pos.iloc[0]
    ipar = np.mean(defender_data.ipar)
    n_throws = len(defender_data)
    n_incomp = len(defender_data.query('result == "I"'))
    n_comp = len(defender_data.query('result == "C"'))
    n_int = len(defender_data.query('result == "IN"'))
    per_incomp = 100.0*(n_incomp + n_int) / (n_incomp + n_int + n_comp)
    per_int = 100.0*(n_int) / (n_incomp + n_int + n_comp)
    n_fool_comp = len(defender_data.query('comp_prob < 0.5 and result == "C"'))
    n_fool_comp_total = len(defender_data.query('comp_prob < 0.5'))
    if n_fool_comp_total > 0:
        fool_comp = 100.0*n_fool_comp / n_fool_comp_total
    else:
        fool_comp = np.nan
    n_fool_incomp = len(defender_data.query('comp_prob > 0.5 and (result == "I" or result == "IN")'))
    n_fool_incomp_total = len(defender_data.query('comp_prob > 0.5'))
    if n_fool_incomp_total > 0:
        fool_incomp = 100.0*n_fool_incomp / n_fool_incomp_total
    else:
        fool_incomp = np.nan
    avg_cp = 100.0*np.mean(defender_data.comp_prob)
    epa = np.mean(defender_data.epa)
    new_row = pd.DataFrame([[def_name, def_pos, n_throws, ipar, per_incomp, per_int, fool_comp, \
                            fool_incomp, avg_cp, epa]], columns=column_names)
    rankings_df = rankings_df.append(new_row, ignore_index=True)

# Rankings
column_names = ["name", "position", "n_throws", "ipar", "incomp_per", \
                "int_per", "fool_incomp_to_comp", \
                "fool_comp_to_incomp", "avg_comp_prob", "epa", "overall_score"]
values_rank_df = pd.DataFrame(data = np.empty((0,len(column_names))), columns = column_names)
overall_rank_df = pd.DataFrame(data = np.empty((0,len(column_names))), columns = column_names)
min_throws = 45

# IPAR
rank_df = rankings_df.sort_values(by=['ipar'], ascending=False)
rank = rank_df.query('n_throws >= {:.0f} and position == "CB"'.format(min_throws)).reset_index().drop(columns=['index'])
values_rank_df.name = rank.sort_values(by=['name']).name
values_rank_df.position = rank.sort_values(by=['name']).position
values_rank_df.n_throws = rank.sort_values(by=['name']).n_throws
values_rank_df.ipar = rank.sort_values(by=['name']).ipar
overall_rank_df.name = rank.sort_values(by=['name']).name
overall_rank_df.position = rank.sort_values(by=['name']).position
overall_rank_df.n_throws = rank.sort_values(by=['name']).n_throws
overall_rank_df.ipar = rank.sort_values(by=['name']).index + 1
values_rank_df = values_rank_df.reset_index().drop(columns=['index'])
overall_rank_df = overall_rank_df.reset_index().drop(columns=['index'])

# Incompletion percentage
rank_df = rankings_df.sort_values(by=['incomp_per'], ascending=False)
rank = rank_df.query('n_throws >= {:.0f} and position == "CB"'.format(min_throws)).reset_index().drop(columns=['index'])
values_rank_df.incomp_per = rank.sort_values(by=['name']).incomp_per.reset_index().drop(columns=['index'])
overall_rank_df.incomp_per = rank.sort_values(by=['name']).index + 1

# Interception percentage
rank_df = rankings_df.sort_values(by=['int_per'], ascending=False)
rank = rank_df.query('n_throws >= {:.0f} and position == "CB"'.format(min_throws)).reset_index().drop(columns=['index'])
values_rank_df.int_per = rank.sort_values(by=['name']).int_per.reset_index().drop(columns=['index'])
overall_rank_df.int_per = rank.sort_values(by=['name']).index + 1

# Fooled (incomplete -> complete) percentage
rank_df = rankings_df.sort_values(by=['fool_incomp_to_comp'], ascending=True)
rank = rank_df.query('n_throws >= {:.0f} and position == "CB"'.format(min_throws)).reset_index().drop(columns=['index'])
values_rank_df.fool_incomp_to_comp = rank.sort_values(by=['name']).fool_incomp_to_comp.reset_index().drop(columns=['index'])
overall_rank_df.fool_incomp_to_comp = rank.sort_values(by=['name']).index + 1

# Fooled (complete -> incomplete) percentage
rank_df = rankings_df.sort_values(by=['fool_comp_to_incomp'], ascending=False)
rank = rank_df.query('n_throws >= {:.0f} and position == "CB"'.format(min_throws)).reset_index().drop(columns=['index'])
values_rank_df.fool_comp_to_incomp = rank.sort_values(by=['name']).fool_comp_to_incomp.reset_index().drop(columns=['index'])
overall_rank_df.fool_comp_to_incomp = rank.sort_values(by=['name']).index + 1

# Average completion probability 
rank_df = rankings_df.sort_values(by=['avg_comp_prob'], ascending=True)
rank = rank_df.query('n_throws >= {:.0f} and position == "CB"'.format(min_throws)).reset_index().drop(columns=['index'])
values_rank_df.avg_comp_prob = rank.sort_values(by=['name']).avg_comp_prob.reset_index().drop(columns=['index'])
overall_rank_df.avg_comp_prob = rank.sort_values(by=['name']).index + 1

# EPA
rank_df = rankings_df.sort_values(by=['epa'], ascending=False)
rank = rank_df.query('n_throws >= {:.0f} and position == "CB"'.format(min_throws)).reset_index().drop(columns=['index'])
values_rank_df.epa = rank.sort_values(by=['name']).epa.reset_index().drop(columns=['index'])
overall_rank_df.epa = rank.sort_values(by=['name']).index + 1

# Overall rank
overall_rank_df.overall_score = overall_rank_df.ipar + overall_rank_df.incomp_per + overall_rank_df.int_per + \
                                overall_rank_df.fool_incomp_to_comp + overall_rank_df.fool_comp_to_incomp + \
                                overall_rank_df.avg_comp_prob + overall_rank_df.epa
values_rank_df.overall_score = overall_rank_df.overall_score
overall_rank_df = overall_rank_df.sort_values(by=['overall_score'], ascending=True).reset_index().drop(columns=['index'])
values_rank_df = values_rank_df.sort_values(by=['overall_score'], ascending=True).reset_index().drop(columns=['index'])
overall_rank_df.index += 1 
values_rank_df.index += 1

print("")
print("")
print("")
print(values_rank_df)
print("")
print("")
print("")
print(overall_rank_df)
print("")
print("")
print("")

overall_rank_df.to_csv("~/Desktop/cb_rankings_1.csv", index=True)
values_rank_df.to_csv("~/Desktop/cb_rankings_2.csv", index=True)