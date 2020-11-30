import pandas as pd
import re
import numpy as np

# get week dataframe
def load_week_df(week):
    df = pd.read_csv('../data/week{:.0f}.csv'.format(week))

    return df

# get play dataframe
def get_play_df(week_df, play_id_list, play_idx):
    play_sel = (week_df["playId"] == play_id_list[play_idx]) # play selector
    play_df = week_df[play_sel]

    return play_df

# get frame at which football arrives at receiver
def get_pass_arrival_df(play_df):
    r = re.compile('pass_outcome*') # get all events that start with 'pass_outcome'
    vmatch = np.vectorize(lambda x:bool(r.match(x)))
    sel = vmatch(play_df["event"])
    pass_arrival_df = play_df[sel]

    return pass_arrival_df

# find intended receiver and defender
def get_intended_receiver_and_defender(play_df, offense_positions, defense_positions):
    pass_arrival_df = get_pass_arrival_df(play_df)
    football_df = pass_arrival_df[pass_arrival_df["displayName"] == 'Football']
    football_xy = football_df[["x","y"]].values
    def_sel = pass_arrival_df["position"].apply(lambda x : x in defense_positions)
    off_sel = pass_arrival_df["position"].apply(lambda x : x in offense_positions)
    defense_df = pass_arrival_df[def_sel]
    offense_df = pass_arrival_df[off_sel]
    def_players_xy = defense_df[["x","y"]].values
    off_players_xy = offense_df[["x","y"]].values
    defender_idx = np.argmin(np.linalg.norm(def_players_xy - football_xy,axis=1))
    defender_df = defense_df.iloc[defender_idx]
    receiver_idx = np.argmin(np.linalg.norm(off_players_xy - football_xy,axis=1))
    receiver_df = offense_df.iloc[receiver_idx]

    return receiver_df["displayName"], defender_df["displayName"]