import numpy as np
from pdb import set_trace as bp


# get targeted receiver
def get_targeted_receiver(target_rec_df, play_df, play_id):
    game_id = play_df["gameId"].iloc[0]
    target_query = 'playId == {:.0f} and gameId == {:.0f}'.format(play_id, game_id)
    target_rec_id = target_rec_df.query(target_query)["targetNflId"].iloc[0]
    name_query = 'nflId == {:.0f}'.format(target_rec_id)
    target_rec = play_df.query(name_query)["displayName"].iloc[0]
    return target_rec

# find closest defender to target receiver when ball arrives
def get_defender(play_df, target_rec, defense_positions):
    pass_arrive_query = 'event == "pass_arrived"'
    pass_arrive_df = play_df.query(pass_arrive_query)
    if len(pass_arrive_df) == 0:
        pass_arrive_df = play_df[play_df.event.str.startswith("pass_outcome")]
    target_query = 'displayName == "' + target_rec + '"'
    target_df = pass_arrive_df.query(target_query)
    target_xy = target_df[["x","y"]].values
    def_sel = pass_arrive_df["position"].apply(lambda x : x in defense_positions)
    defense_df = pass_arrive_df[def_sel]
    def_players_xy = defense_df[["x","y"]].values
    defender_idx = np.argmin(np.linalg.norm(def_players_xy - target_xy,axis=1))
    defender_df = defense_df.iloc[defender_idx]
    defender = defender_df["displayName"]
    return defender

# find defender's distance recovered while ball was in air
def find_defender_recovery(play_df, defender, receiver):
    ball_thrown_query = 'event == "pass_forward"'
    ball_thrown_df = play_df.query(ball_thrown_query)
    pass_arrived_query = 'event == "pass_arrived"'
    pass_arrived_df = play_df.query(pass_arrived_query)
    defender_query = 'displayName == "' + defender + '"'
    receiver_query = 'displayName == "' + receiver + '"'
    defender_ball_thrown_xy = ball_thrown_df.query(defender_query)[["x","y"]]
    defender_pass_arrived_xy = pass_arrived_df.query(defender_query)[["x","y"]]
    receiver_ball_thrown_xy = ball_thrown_df.query(receiver_query)[["x","y"]]
    receiver_pass_arrived_xy = pass_arrived_df.query(receiver_query)[["x","y"]]
    ball_thrown_dist = np.linalg.norm(np.array(receiver_ball_thrown_xy) - np.array(defender_ball_thrown_xy))
    pass_arrived_dist = np.linalg.norm(np.array(receiver_pass_arrived_xy) - np.array(defender_pass_arrived_xy))
    recovery_dist = ball_thrown_dist - pass_arrived_dist
    return recovery_dist