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


# loop over weeks
n_good = 0
n_bad = 0
comp_prob_data = np.empty((0,12))
games_df = pd.read_csv('../data/games.csv')
plays_df = pd.read_csv('../data/plays.csv')
for week in range(1,18):
    week_df = load_week_df(week)

    # loop over games
    game_id_list = np.unique(week_df.gameId)
    game = 1
    for game_id in game_id_list:
        game_df = week_df.query('gameId == {:.0f}'.format(game_id))

        # loop over plays
        play_id_list = np.unique(game_df["playId"])
        progress_bar = tqdm(play_id_list, desc='week {:.0f}, game {:.0f}'.format(week, game))
        for play_id in progress_bar:
            play_df = game_df.query('playId == {:.0f}'.format(play_id))

            # get metrics
            ret, metrics = get_metrics(week, play_df, games_df, plays_df)

            # check if everything was successful
            if ret:
                n_good += 1
                # log data
                comp_prob_data = np.append(comp_prob_data, metrics, axis=0)
            else:
                n_bad += 1

            ### animate play
            #animate_play(play_df)
            #plt.close()

            # update progress bar
            disp_dict = {"good": n_good, "bad": n_bad, "percent": 100.0*n_good/(n_good+n_bad)}
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

        progress_bar.close()

        game += 1

    print("")

np.save("../data/comp_prob_data.npy", comp_prob_data)