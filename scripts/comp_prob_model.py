import numpy as np
import pandas as pd
from tqdm import tqdm
from pdb import set_trace as bp
import plotly.express as px

# append parent directory to Python path
import sys
sys.path.append("..")

# import utility functions from util directory
from util.visualization import *
from util.comp_prob_model import *

# load completion probability data
# [week, game, play, lat_dist, lon_dist, wr_prox, db_prox, sl_prox, bl_prox, qb_vel, t_throw, completion]
cp_data = np.load("../data/comp_prob_data.npy")

# convert data to pandas
cp_df = pd.DataFrame(data = cp_data,
                     columns = ["week", "game", "play", "lat_dist", "lon_dist", "wr_prox", "db_prox", 
                                "sl_prox", "bl_prox", "qb_vel", "t_throw", "completion"])

# clean up data
cp_df = cp_df.query('not (wr_prox > 10 and completion == 1)') # sometimes receiver is not tracked
cp_df.bl_prox[cp_df.bl_prox.isnull()] = np.max(cp_df.bl_prox) # set nan values to max value


























## normalize data
#lat_dist, min_lat_dist, max_lat_dist, avg_lat_dist, std_lat_dist = get_normalized_data(cp_df.lat_dist)
#cp.lat_dist = lat_dist
##cp_df.lat_dist = (cp_df.lat_dist - np.min(cp_df.lat_dist)) / (np.max(cp_df.lat_dist) - np.min(cp_df.lat_dist))
##cp_df.lon_dist = (cp_df.lon_dist - np.min(cp_df.lon_dist)) / (np.max(cp_df.lon_dist) - np.min(cp_df.lon_dist))
##cp_df.wr_prox = (cp_df.wr_prox - np.min(cp_df.wr_prox)) / (np.max(cp_df.wr_prox) - np.min(cp_df.wr_prox))
##cp_df.db_prox = (cp_df.db_prox - np.min(cp_df.db_prox)) / (np.max(cp_df.db_prox) - np.min(cp_df.db_prox))
##cp_df.sl_prox = (cp_df.sl_prox - np.min(cp_df.sl_prox)) / (np.max(cp_df.sl_prox) - np.min(cp_df.sl_prox))
##cp_df.bl_prox = (cp_df.bl_prox - np.min(cp_df.bl_prox)) / (np.max(cp_df.bl_prox) - np.min(cp_df.bl_prox))
##cp_df.qb_vel = (cp_df.qb_vel - np.min(cp_df.qb_vel)) / (np.max(cp_df.qb_vel) - np.min(cp_df.qb_vel))
##cp_df.t_throw = (cp_df.t_throw - np.min(cp_df.t_throw)) / (np.max(cp_df.t_throw) - np.min(cp_df.t_throw))

## plotting data
#fig = px.scatter(cp_df,
#                 x=cp_df.index,
#                 y="lat_dist",
#                 color=cp_df["completion"].astype(str),
#                 hover_data=["week", "game", "play"],
#                 title="Lateral Pass Distance")
#fig.show()

#fig = px.scatter(cp_df,
#                 x=cp_df.index,
#                 y="lon_dist",
#                 color=cp_df["completion"].astype(str),
#                 hover_data=["week", "game", "play"],
#                 title="Longitudinal Pass Distance")
#fig.show()

#fig = px.scatter(cp_df,
#                 x=cp_df.index,
#                 y="wr_prox",
#                 color=cp_df["completion"].astype(str),
#                 hover_data=["week", "game", "play"],
#                 title="WR-Ball Proximity")
#fig.show()

#fig = px.scatter(cp_df,
#                 x=cp_df.index,
#                 y="db_prox",
#                 color=cp_df["completion"].astype(str),
#                 hover_data=["week", "game", "play"],
#                 title="DB-WR Proximity")
#fig.show()

#fig = px.scatter(cp_df,
#                 x=cp_df.index,
#                 y="sl_prox",
#                 color=cp_df["completion"].astype(str),
#                 hover_data=["week", "game", "play"],
#                 title="WR-Sideline Proximity")
#fig.show()

#fig = px.scatter(cp_df,
#                 x=cp_df.index,
#                 y="bl_prox",
#                 color=cp_df["completion"].astype(str),
#                 hover_data=["week", "game", "play"],
#                 title="Blitzer-QB Proximity")
#fig.show()

#fig = px.scatter(cp_df,
#                 x=cp_df.index,
#                 y="qb_vel",
#                 color=cp_df["completion"].astype(str),
#                 hover_data=["week", "game", "play"],
#                 title="QB Speed")
#fig.show()

#fig = px.scatter(cp_df,
#                 x=cp_df.index,
#                 y="t_throw",
#                 color=cp_df["completion"].astype(str),
#                 hover_data=["week", "game", "play"],
#                 title="Time to Throw")
#fig.show()

## visualize data
#animate_play2(2, 2018091600, 3549)