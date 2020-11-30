import os, sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()),'input/nfl-bdb-scripts'))
import pandas as pd
from visualization import *

animate_play(week=14, gameId=2018120911, playId=1584)

#scores_df = pd.read_csv('../input/nfl-bdb-data/cornerback_scores.csv')

#plot_playmaking_skills(scores_df)
#plot_coverage_skills(scores_df)
#plot_ball_skills(scores_df)
