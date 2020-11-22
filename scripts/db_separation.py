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
from util.visualization import *


# load week dataframe
week = 1
week_df = load_week_df(week)

# load targeted receiver data
target_rec_df = pd.read_csv("../data/targetedReceiver.csv")

# get play and game ID
play_idx = 25
play_id_list = np.unique(week_df["playId"])
play_df = get_play_df(week_df, play_id_list, play_idx)
play_id = play_id_list[play_idx]


# get targeted receiver
receiver = get_targeted_receiver(target_rec_df, play_df, play_id)

# find closest defender to target receiver when ball arrives
defense_positions = np.array(['CB','DB','DE','DL','FS','ILB','LB','MLB','NT','OLB','S','SS'])
defender = get_defender(play_df, receiver, defense_positions)

# find defender's distance recovered while ball was in air
recovery_dist = find_defender_recovery(play_df, defender, receiver)
print("Distance recovered: {:.1f} yards".format(recovery_dist))

# plot play
offense_positions = np.array(['FB','HB','QB','RB','TE','WR'])
plot_play(play_df, offense_positions, defense_positions, receiver, defender)

# animate play
animate_play(play_df)