import numpy as np
from pdb import set_trace as bp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# append parent directory to Python path
import sys
sys.path.append("..")

# import utility functions from util directory
from util.util import *
from util.visualization import *


# load week dataframe
week = 1
week_df = load_week_df(week)

# get play and game IDs
play_idx = 25
play_id_list = np.unique(week_df["playId"])
play_df = get_play_df(week_df, play_id_list, play_idx)

# define offensive/defensive positions
defense_positions = np.array(['CB','DB','DE','DL','FS','ILB','LB','MLB','NT','OLB','S','SS'])
offense_positions = np.array(['FB','HB','QB','RB','TE','WR'])

# get intended receiver and defender
receiver_name, defender_name = get_intended_receiver_and_defender(play_df, offense_positions, defense_positions)
print("Receiver: " + receiver_name)
print("Defender: " + defender_name)

# plot play
plot_play(play_df, offense_positions, defense_positions, receiver_name, defender_name)

# animate play
animate_play(play_df)
