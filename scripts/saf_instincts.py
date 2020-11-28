import numpy as np
import pandas as pd
from tqdm import tqdm
from pdb import set_trace as bp

# append ../util directory to Python path
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'util'))

# import utility functions from util directory
from util import *
from visualization import *
from cp_data import *



# ####################
# ##### GET DATA #####
# ####################

# # loop over weeks
# delta_t = 1 # in seconds
# n_good = 0
# n_bad = 0
# games_df = pd.read_csv('../data/games.csv')
# plays_df = pd.read_csv('../data/plays.csv')
# column_names = ["name", "pos", "t1_dist", "t2_dist", "dist_diff"]
# metrics_df = pd.DataFrame(data = np.empty((0,len(column_names))),
#                           columns = column_names)
# for week in range(1,18):
#     week_df = load_week_df(week)

#     # loop over games
#     game_id_list = np.unique(week_df.gameId)
#     game = 1
#     for game_id in game_id_list:
#         game_df = week_df.query('gameId == {:.0f}'.format(game_id))

#         # loop over plays
#         play_id_list = np.unique(game_df["playId"])
#         progress_bar = tqdm(play_id_list, desc='week {:.0f}, game {:.0f}'.format(week, game))
#         for play_id in progress_bar:
#             play_df = game_df.query('playId == {:.0f}'.format(play_id))

#             # determine who has ball (home/away)
#             possession_team = plays_df.query('gameId == {:.0f} and playId == {:.0f}'.format(game_id, play_id)).possessionTeam.iloc[0]
#             home_team = games_df.query('gameId == {:.0f}'.format(game_id)).homeTeamAbbr.iloc[0]
#             away_team = games_df.query('gameId == {:.0f}'.format(game_id)).visitorTeamAbbr.iloc[0]
#             if possession_team == home_team:
#                 possession = 'home'
#             elif possession_team == away_team:
#                 possession = 'away'
#             else:
#                 print("Cannot determine who has ball")
#                 n_bad += 1
#                 continue

#             # get pass thrown dataframe
#             ret, pass_thrown_df = find_pass_thrown_df(play_df)
#             if not ret:
#                 n_bad += 1
#                 continue
#             pass_thrown_frame = pass_thrown_df.frameId.iloc[0]

#             # get pass arrived dataframe
#             ret, pass_arrived_df = find_pass_arrived_df(play_df, possession)
#             if not ret:
#                 n_bad += 1
#                 continue

#             # check if data is good
#             ret = check_if_good(pass_thrown_df, pass_arrived_df)
#             if not ret:
#                 n_bad += 1
#                 continue

#             # get frame 1 dataframe
#             frame1 = np.max([pass_thrown_frame - 10*delta_t, 1])
#             frame1_df = play_df.query('frameId == {:.0f}'.format(frame1))

#             # get frame 2 dataframe
#             frame2 = np.min([pass_thrown_frame + 10*delta_t, play_df.frameId.iloc[-1]])
#             frame2_df = play_df.query('frameId == {:.0f}'.format(frame2))

#             # find target receiver
#             try:
#                 off_pass_arrived_df = pass_arrived_df.query('team == "' + possession + '"')
#                 off_players_xy = off_pass_arrived_df[["x","y"]].values
#                 football_xy = pass_arrived_df.query('displayName == "Football"')[["x","y"]].values
#                 receiver_idx = np.argmin(np.linalg.norm(football_xy - off_players_xy, axis=1))
#                 receiver = off_pass_arrived_df.iloc[receiver_idx].displayName
#             except:
#                 n_bad += 1
#                 continue

#             # find defender's position at frame 1 wrt receiver at frame 2
#             try:
#                 def_frame1_df = frame1_df.query('team != "' + possession + '" and displayName != "Football"')
#                 def_frame1_df = def_frame1_df.sort_values(by=['displayName']).reset_index().drop(columns=['index'])
#                 def_players_xy = def_frame1_df[["x","y"]].values
#                 receiver_xy = frame2_df.query('displayName == "' + receiver + '"')[["x","y"]].values
#                 def_dist1 = np.linalg.norm(receiver_xy - def_players_xy, axis=1)
#             except:
#                 n_bad += 1
#                 continue

#             # find defender's position at frame 2 wrt receiver at frame 2
#             try:
#                 def_frame2_df = frame2_df.query('team != "' + possession + '" and displayName != "Football"')
#                 def_frame2_df = def_frame2_df.sort_values(by=['displayName']).reset_index().drop(columns=['index'])
#                 def_players_xy = def_frame2_df[["x","y"]].values
#                 receiver_xy = frame2_df.query('displayName == "' + receiver + '"')[["x","y"]].values
#                 def_dist2 = np.linalg.norm(receiver_xy - def_players_xy, axis=1)
#             except:
#                 n_bad += 1
#                 continue

#             # check if data is good again
#             if len(def_frame1_df) != len(def_frame2_df):
#                 n_bad += 1
#                 continue

#             for i in range(len(def_frame1_df)):
#                 new_row = pd.DataFrame([[def_frame1_df.displayName.iloc[i], def_frame1_df.position.iloc[i],
#                                     def_dist1[i], def_dist2[i], def_dist1[i]-def_dist2[i]]],
#                                     columns=column_names)
#                 metrics_df = metrics_df.append(new_row, ignore_index=True)

#             # everything was successful
#             n_good += 1

#             ### animate play
#             #animate_play(play_df)
#             #plt.close()

#             # update progress bar
#             disp_dict = {"good": n_good, "bad": n_bad, "percent": 100.0*n_good/(n_good+n_bad)}
#             progress_bar.set_postfix(disp_dict)
#             progress_bar.update()

#         progress_bar.close()

#         game += 1


#     print("")

# metrics_df.to_csv("../data/saf_instincts.csv", index=False)



########################
##### ANALYZE DATA #####
########################

column_names = ["name", "position", "snaps", "avg_dist"]
new_metrics_df = pd.DataFrame(data = np.empty((0,len(column_names))),
                          columns = column_names)
metrics_df = pd.read_csv('../data/saf_instincts.csv')
for defender in tqdm(np.unique(metrics_df.name)):
    defender_df = metrics_df.query('name == "' + defender + '"')
    avg_dist_diff = np.mean(defender_df.dist_diff)
    new_row = pd.DataFrame([[defender, defender_df.pos.iloc[0], len(defender_df), avg_dist_diff]], columns=column_names)
    new_metrics_df = new_metrics_df.append(new_row, ignore_index=True)

rankings = new_metrics_df.query('position == "FS" and snaps > 45')
rankings = rankings.sort_values(by=['avg_dist'], ascending=False).reset_index().drop(columns=['index'])
rankings.index += 1
print(rankings)