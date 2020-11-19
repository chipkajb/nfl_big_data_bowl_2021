import numpy as np
from pdb import set_trace as bp
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)

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
play_idx = 0
play_id_list = np.unique(week_df["playId"])
play_df = get_play_df(week_df, play_id_list, play_idx)

## define offensive/defensive positions
#defense_positions = np.array(['CB','DB','DE','DL','FS','ILB','LB','MLB','NT','OLB','S','SS'])
#offense_positions = np.array(['FB','HB','QB','RB','TE','WR'])

## get intended receiver and defender
#receiver_name, defender_name = get_intended_receiver_and_defender(play_df, offense_positions, defense_positions)
#print(receiver_name)
#print(defender_name)

## plot play
#plot_play(play_df, offense_positions, defense_positions, receiver_name, defender_name)

home_query = 'playId == {:.0f} and team == "home"'.format(play_id_list[play_idx])
away_query = 'playId == {:.0f} and team == "away"'.format(play_id_list[play_idx])
football_query = 'playId == {:.0f} and displayName == "Football"'.format(play_id_list[play_idx])
play_home = week_df.query(home_query)
play_away = week_df.query(away_query)
play_football = week_df.query(football_query)
fig, ax = create_football_field(highlight_line=True, highlight_line_number=play_football.iloc[0]["x"]-10)
plt.ion()
plt.show()
for i in range(1,len(play_football)+1):
    frame_query = 'frameId == {:.0f}'.format(i)
    home_pts = play_home.query(frame_query).plot(x='x', y='y', kind='scatter', ax=ax, color='orange', s=30, legend='Away')
    away_pts = play_away.query(frame_query).plot(x='x', y='y', kind='scatter', ax=ax, color='blue', s=30, legend='Home')
    football_pt = play_football.query(frame_query).plot(x='x', y='y', kind='scatter', ax=ax, color='black', s=30, legend='Home')
    plt.pause(0.001)
