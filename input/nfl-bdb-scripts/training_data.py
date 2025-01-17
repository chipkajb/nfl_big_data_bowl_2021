import numpy as np
import pandas as pd
from tqdm import tqdm
from pdb import set_trace as bp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# find dataframe where ball is thrown
def find_pass_thrown_df(play_df):
    pass_thrown_df = play_df.query('event == "pass_forward"')

    # check if pass_forward event was found
    if len(pass_thrown_df) == 0:
        pass_thrown_df = play_df.query('event == "pass_shovel"')
        # check if pass_shovel not found
        if len(pass_thrown_df) == 0:
            return False, pass_thrown_df # if no "pass_forward" or "pass_shovel", then just ignore

    return True, pass_thrown_df


# find dataframe where pass arrives
def find_pass_arrived_df(play_df, possession):
    pass_arrived_df = play_df.query('event == "pass_arrived"')

    # check if pass_arrived event was found
    if len(pass_arrived_df) == 0:
        try:
            # find closest receiver when pass_outcome occurs
            pass_outcome_sel = play_df.event.str.contains("pass_outcome*")
            pass_outcome_df = play_df[pass_outcome_sel]
            off_pass_outcome_df = pass_outcome_df.query('team == "' + possession + '"')
            off_players_xy = off_pass_outcome_df[["x","y"]].values
            football_xy = pass_outcome_df.query('displayName == "Football"')[["x","y"]].values
            receiver_idx = np.argmin(np.linalg.norm(football_xy - off_players_xy, axis=1))
            receiver = off_pass_outcome_df.iloc[receiver_idx].displayName

            # find frame when ball was closest to receiver
            football_xy = play_df.query('displayName == "Football"')[["x","y"]].values
            receiver_xy = play_df.query('displayName == "' + receiver + '"')[["x","y"]].values
            idx_pass_arrived = np.argmin(np.linalg.norm(football_xy - receiver_xy, axis=1))
            frame_pass_arrived = play_df.query('displayName == "Football"').iloc[idx_pass_arrived].frameId
            q_pass_arrived = 'frameId == {:.0f}'.format(frame_pass_arrived)
            pass_arrived_df =  play_df.query(q_pass_arrived)
        except:
            return False, pass_arrived_df # if data is irregular in any way, then just ignore

        if len(pass_arrived_df) == 0:
            return False, pass_arrived_df # if still cannot be found, then just ignore

    return True, pass_arrived_df


# find WR proximity to ball
def find_wr_proximity(pass_arrived_df, possession):
    off_pass_arrived_df = pass_arrived_df.query('team == "' + possession + '"')
    off_players_xy = off_pass_arrived_df[["x","y"]].values
    football_xy = pass_arrived_df.query('displayName == "Football"')[["x","y"]].values
    wr_prox = np.min(np.linalg.norm(football_xy - off_players_xy, axis=1))
    return wr_prox


# find DB proximity to target receiver
def find_db_proximity(pass_arrived_df, possession):
    off_pass_arrived_df = pass_arrived_df.query('team == "' + possession + '"')
    def_pass_arrived_df = pass_arrived_df.query('team != "' + possession + '" and displayName != "Football"')
    off_players_xy = off_pass_arrived_df[["x","y"]].values
    def_players_xy = def_pass_arrived_df[["x","y"]].values
    football_xy = pass_arrived_df.query('displayName == "Football"')[["x","y"]].values
    receiver_idx = np.argmin(np.linalg.norm(football_xy - off_players_xy, axis=1))
    defender_idx = np.argmin(np.linalg.norm(football_xy - def_players_xy, axis=1))
    receiver_xy = off_players_xy[receiver_idx,:]
    defender_xy = def_players_xy[defender_idx,:]
    db_prox = np.linalg.norm(receiver_xy - defender_xy)
    return db_prox


# find sideline proximity to target receiver
def find_sideline_proximity(pass_arrived_df, possession):
    off_pass_arrived_df = pass_arrived_df.query('team == "' + possession + '"')
    off_players_xy = off_pass_arrived_df[["x","y"]].values
    football_xy = pass_arrived_df.query('displayName == "Football"')[["x","y"]].values
    receiver_idx = np.argmin(np.linalg.norm(football_xy - off_players_xy, axis=1))
    receiver_xy = off_players_xy[receiver_idx,:]
    sl_prox = np.min([np.abs(receiver_xy[1] - 0), np.abs(receiver_xy[1] - 53.333333)])
    return sl_prox


# find blitzer proximity to QB
def find_blitzer_proximity(play_df, pass_thrown_df, possession):
    los = play_df.query('event == "ball_snap" and displayName == "Football"').x.values[0]
    play_direction = play_df.query('event == "ball_snap" and displayName == "Football"').playDirection.values[0]
    def_pass_thrown_df = pass_thrown_df.query('team != "' + possession + '" and displayName != "Football"')
    def_players_xy = def_pass_thrown_df[["x","y"]].values
    if play_direction == 'left':
        blitzer_sel = def_players_xy[:,0] > los
    elif play_direction == 'right':
        blitzer_sel = def_players_xy[:,0] < los

    blitzers_xy = def_players_xy[blitzer_sel]
    if len(blitzers_xy) == 0:
        bl_prox = np.nan
    else:
        football_xy = pass_thrown_df.query('displayName == "Football"')[["x","y"]].values
        off_players_xy = pass_thrown_df.query('team == "' + possession + '"')[["x","y"]].values
        qb_idx = np.argmin(np.linalg.norm(off_players_xy - football_xy, axis=1))
        qb_xy = off_players_xy[qb_idx,:]
        blitzer_idx = np.argmin(np.linalg.norm(qb_xy - blitzers_xy, axis=1))
        blitzer_xy = blitzers_xy[blitzer_idx, :]
        bl_prox = np.linalg.norm(blitzer_xy - qb_xy)
        if bl_prox > 10: # set max distance
            bl_prox = np.nan

    return bl_prox


# find QB speed
def find_qb_speed(pass_thrown_df, possession):
    football_xy = pass_thrown_df.query('displayName == "Football"')[["x","y"]].values
    off_players_df = pass_thrown_df.query('team == "' + possession + '"')
    off_players_xy = off_players_df[["x","y"]].values
    dist = np.min(np.linalg.norm(off_players_xy - football_xy, axis=1))
    qb_idx = np.argmin(np.linalg.norm(off_players_xy - football_xy, axis=1))
    qb_df = off_players_df.iloc[qb_idx]
    qb_vel = qb_df.s
    return qb_vel


# find time to throw
def find_time_to_throw(play_df, pass_thrown_df):
    ball_snapped_df = play_df.query('event == "ball_snap"')
    colon_idx0 = ball_snapped_df.time.iloc[0].find(":")
    colon_idx1 = ball_snapped_df.time.iloc[0][colon_idx0+1::].find(":")
    t0 = float(ball_snapped_df.time.iloc[0][colon_idx0+colon_idx1+2:-1])
    colon_idx0 = pass_thrown_df.time.iloc[0].find(":")
    colon_idx1 = pass_thrown_df.time.iloc[0][colon_idx0+1::].find(":")
    t1 = float(pass_thrown_df.time.iloc[0][colon_idx0+colon_idx1+2:-1])
    t_throw = t1-t0
    if t_throw < 0:
        t_throw += 60 # fix time that wraps around the 60-sec mark
    return t_throw


# find completion
def find_completion(play_df):
    pass_outcome_sel = play_df.event.str.contains("pass_outcome*")
    pass_outcomes = np.unique(play_df[pass_outcome_sel].event)
    if len(pass_outcomes) == 0:
        return False, 0
    pass_outcome = pass_outcomes[-1]
    if (pass_outcome == "pass_outcome_caught") or (pass_outcome == "pass_outcome_touchdown"):
        completion = 1
    elif (pass_outcome == "pass_outcome_incomplete") or (pass_outcome == "pass_outcome_interception"):
        completion = 0
    else:
        return False, 0
    return True, completion


# check if the data is good
def check_if_good(pass_thrown_df, pass_arrived_df):
    good = True
    if len(pass_thrown_df.query('displayName == "Football"')) != 1:
        good = False
    elif len(pass_arrived_df.query('displayName == "Football"')) != 1:
        good = False
    return good


# find route
def find_route(pass_arrived_df, possession):
    off_pass_arrived_df = pass_arrived_df.query('team == "' + possession + '"')
    off_players_xy = off_pass_arrived_df[["x","y"]].values
    football_xy = pass_arrived_df.query('displayName == "Football"')[["x","y"]].values
    wr_idx = np.argmin(np.linalg.norm(football_xy - off_players_xy, axis=1))
    route = off_pass_arrived_df.iloc[wr_idx].route
    return route


# get metrics
def get_metrics(week, play_df, games_df, plays_df):
    column_names = ["week", "game", "play", "lat_dist", "lon_dist", \
                "ball_prox", "db_prox", "sl_prox", "bl_prox", "qb_speed", "t_throw", "result", "route"]
    metrics_df = pd.DataFrame(data = np.empty((0,len(column_names))), columns = column_names)
    game_id = play_df.gameId.iloc[0]
    play_id = play_df.playId.iloc[0]

    # determine who has ball (home/away)
    possession_team = plays_df.query('gameId == {:.0f} and playId == {:.0f}'.format(game_id, play_id)).possessionTeam.iloc[0]
    home_team = games_df.query('gameId == {:.0f}'.format(game_id)).homeTeamAbbr.iloc[0]
    away_team = games_df.query('gameId == {:.0f}'.format(game_id)).visitorTeamAbbr.iloc[0]
    if possession_team == home_team:
        possession = 'home'
    elif possession_team == away_team:
        possession = 'away'
    else:
        print("Cannot determine who has ball")
        return False, metrics_df

    # get dataframe when ball is thrown
    ret, pass_thrown_df = find_pass_thrown_df(play_df)
    if not ret:
        return False, metrics_df

    # get dataframe when ball arrives
    ret, pass_arrived_df = find_pass_arrived_df(play_df, possession)
    if not ret:
        return False, metrics_df

    ret = check_if_good(pass_thrown_df, pass_arrived_df)
    if not ret:
        return False, metrics_df

    # get lat/lon pass distance
    ball_0 = pass_thrown_df.query('displayName == "Football"')
    x_ball_0 = ball_0.x
    y_ball_0 = ball_0.y
    ball_1 = pass_arrived_df.query('displayName == "Football"')
    x_ball_1 = ball_1.x
    y_ball_1 = ball_1.y
    if play_df.playDirection.iloc[0] == 'right':
        lon_dist = x_ball_1.iloc[0] - x_ball_0.iloc[0]
    else:
        lon_dist = x_ball_0.iloc[0] - x_ball_1.iloc[0]
    lat_dist = np.abs(y_ball_1.iloc[0] - y_ball_0.iloc[0])

    # sometimes ball is out of bounds, ignore these plays
    if (y_ball_0.iloc[0] < 0) or (y_ball_0.iloc[0] > 53.3333):
        return False, metrics_df

    # get WR proximity to ball
    try:
        ball_prox = find_wr_proximity(pass_arrived_df, possession)
    except:
        return False, metrics_df # if data is irregular in any way, then just ignore

    # get DB proximity
    try:
        db_prox = find_db_proximity(pass_arrived_df, possession)
    except:
        return False, metrics_df # if data is irregular in any way, then just ignore

    # get sideline proximity
    sl_prox = find_sideline_proximity(pass_arrived_df, possession)

    # get blitzer proximity
    bl_prox = find_blitzer_proximity(play_df, pass_thrown_df, possession)

    # get qb speed
    qb_speed = find_qb_speed(pass_thrown_df, possession)

    # get time to throw
    t_throw = find_time_to_throw(play_df, pass_thrown_df)

    # get result
    ret, result = find_completion(play_df)
    if not ret:
        return False, metrics_df

    # get route
    try:
        route = find_route(pass_arrived_df, possession)
    except:
        return False, metrics_df # if data is irregular in any way, then just ignore

    new_row = pd.DataFrame([[week, game_id, play_id, lat_dist, lon_dist, ball_prox, db_prox, sl_prox, \
        bl_prox, qb_speed, t_throw, result, route]], columns=column_names)
    metrics_df = metrics_df.append(new_row, ignore_index=True)

    return True, metrics_df

# generate training data
def generate_training_data():
    # loop over weeks
    n_good = 0
    n_bad = 0
    column_names = ["week", "game", "play", "lat_dist", "lon_dist", \
                "ball_prox", "db_prox", "sl_prox", "bl_prox", "qb_speed", "t_throw", "result", "route"]
    train_df = pd.DataFrame(data = np.empty((0,len(column_names))), columns = column_names)
    games_df = pd.read_csv('../input/nfl-big-data-bowl-2021/games.csv')
    plays_df = pd.read_csv('../input/nfl-big-data-bowl-2021/plays.csv')
    for week in range(1,18):
        week_df = pd.read_csv('../input/nfl-big-data-bowl-2021/week{:.0f}.csv'.format(week))

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
                ret, metrics_df = get_metrics(week, play_df, games_df, plays_df)

                # check if everything was successful
                if ret:
                    n_good += 1
                    # log data
                    train_df = train_df.append(metrics_df, ignore_index=True)
                else:
                    n_bad += 1

                # update progress bar
                disp_dict = {"good": n_good, "bad": n_bad, "percent": 100.0*n_good/(n_good+n_bad)}
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()

            progress_bar.close()

            game += 1

        print("")

    # save training data
    train_df.to_csv("../input/nfl-bdb-data/training_data.csv", index=False)