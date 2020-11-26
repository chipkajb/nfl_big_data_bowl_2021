import numpy as np
import pandas as pd
from tqdm import trange
from pdb import set_trace as bp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# append ../util directory to Python path
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'util'))

# import utility functions from util directory
from util import *
from separation import *
from visualization import *


# load week dataframe
for week in range(1,17):
    week_df = load_week_df(week)
    bp()

    # load targeted receiver data
    target_rec_df = pd.read_csv("../data/targetedReceiver.csv")

    # get play and game ID
    play_id_list = np.unique(week_df["playId"])
    for play_idx in trange(len(play_id_list), file=sys.stdout, desc='week {:.0f}'.format(week)):
        play_df = get_play_df(week_df, play_id_list, play_idx)
        play_id = play_id_list[play_idx]

        # get targeted receiver
        try:
            receiver = get_targeted_receiver(target_rec_df, play_df, play_id)
        except:
            #print("No intended receiver data found")
            continue

        # find closest defender to target receiver when ball arrives
        defense_positions = np.array(['CB','DB','DE','DL','FS','ILB','LB','MLB','NT','OLB','S','SS'])
        try:
            defender = get_defender(play_df, receiver, defense_positions)
        except:
            #print("No defender found")
            continue

        # find defender's distance recovered while ball was in air
        recovery_dist = find_defender_recovery(play_df, defender, receiver)
        #print("\nDistance recovered: {:.1f} yards".format(recovery_dist))

        ## plot play
        #offense_positions = np.array(['FB','HB','QB','RB','TE','WR'])
        #plot_play(play_df, offense_positions, defense_positions, receiver, defender)

        # animate play
        animate_play(play_df)
        plt.close()