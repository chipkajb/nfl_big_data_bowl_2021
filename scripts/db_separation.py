import numpy as np
import pandas as pd
from pdb import set_trace as bp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# append parent directory to Python path
import sys
sys.path.append("..")

# import utility functions from util directory
from util.util import *
from util.separation import *

# load week dataframe
week = 1
week_df = load_week_df(week)

# load targeted receiver data
target_rec_df = pd.read_csv("../data/targetedReceiver.csv")

# get play and game ID
play_idx = 0
play_id_list = np.unique(week_df["playId"])
play_df = get_play_df(week_df, play_id_list, play_idx)
play_id = play_id_list[play_idx]
game_id = play_df["gameId"].iloc[0]

# get targeted receiver
target_query = 'playId == {:.0f} and gameId == {:.0f}'.format(play_id, game_id)
target_rec_id = target_rec_df.query(target_query)["targetNflId"].iloc[0]
name_query = 'nflId == {:.0f}'.format(target_rec_id)
target_rec = play_df.query(name_query)["displayName"].iloc[0]

# find closest defender to target receiver when ball is thrown
defense_positions = np.array(['CB','DB','DE','DL','FS','ILB','LB','MLB','NT','OLB','S','SS'])
ball_thrown_query = 'event == "pass_forward"'
ball_thrown_df = play_df.query(ball_thrown_query)
def_sel = ball_thrown_df["position"].apply(lambda x : x in defense_positions)
defense_df = pass_arrival_df[def_sel]
#offense_df = pass_arrival_df[off_sel]
#def_players_xy = defense_df[["x","y"]].values
#off_players_xy = offense_df[["x","y"]].values
#defender_idx = np.argmin(np.linalg.norm(def_players_xy - football_xy,axis=1))
#defender_df = defense_df.iloc[defender_idx]
#receiver_idx = np.argmin(np.linalg.norm(off_players_xy - football_xy,axis=1))
#receiver_df = offense_df.iloc[receiver_idx]
bp()


