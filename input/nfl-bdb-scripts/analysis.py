from pdb import set_trace as bp
from tqdm import tqdm
from tqdm import trange
import numpy as np
import torch
import pandas as pd
pd.options.mode.chained_assignment = None
from training_model import *
from training_data import *
from temperature_scaling import *

def get_advanced_metrics(model_path):
    train_df = load_and_clean_data(keep_routes=True)
    routes = train_df.route
    train_df = train_df.drop(columns=["route"])

    # get median DB proximity value for every route
    route_db_prox_dict = {}
    route_list = np.unique(routes[~routes.isnull()])
    for route in route_list:
        route_db_prox_dict[route] = np.median(train_df.db_prox[routes == route])
    route_db_prox_dict["NAN"] = np.median(train_df.db_prox)

    # set up NN
    cpnet = setup_network(train_df)
    cpnet_calib = ModelWithTemperature(cpnet)

    # load saved model
    cpnet_calib.load_state_dict(torch.load(model_path)["model_state_dict"])
    cpnet_calib.eval().cuda()

    # load all data
    plays_df = pd.read_csv('../input/nfl-big-data-bowl-2021/plays.csv')
    games_df = pd.read_csv('../input/nfl-big-data-bowl-2021/games.csv')
    data_dict = {}
    for i in trange(1,18, desc='Loading data'):
        data_dict[i] = pd.read_csv('../input/nfl-big-data-bowl-2021/week{:.0f}.csv'.format(i))

    make_sense = 0
    progress_bar = tqdm(train_df.iterrows(), total=train_df.shape[0], desc="Analyzing data")
    column_names = ["def_name", "def_pos", "off_name", "off_pos", "route", "ipa", \
                    "comp_prob", "result", "epa"]
    metrics_df = pd.DataFrame(data = np.empty((0,len(column_names))),
                              columns = column_names)
    for idx, data_row in progress_bar:
        # find original completion probability
        x = torch.from_numpy(data_row.iloc[3:11].values).cuda(non_blocking=True).float().unsqueeze(0)
        logits1 = cpnet_calib(x)
        softmax1 = F.softmax(logits1, dim=1)
        comp_prob1 = softmax1[0,1].item()
        db_prox_orig = x[0,3].item()

        # get play dataframe
        week = data_row.week
        game_id = data_row.game
        play_id = data_row.play
        week_df = data_dict[week]
        play_df = week_df.query('gameId == {:.0f} and playId == {:.0f}'.format(game_id, play_id))

        # get possession info
        possession_team = plays_df.query('gameId == {:.0f} and playId == {:.0f}'.format(game_id, play_id)).possessionTeam.iloc[0]
        home_team = games_df.query('gameId == {:.0f}'.format(game_id)).homeTeamAbbr.iloc[0]
        away_team = games_df.query('gameId == {:.0f}'.format(game_id)).visitorTeamAbbr.iloc[0]
        if possession_team == home_team:
            possession = 'home'
        elif possession_team == away_team:
            possession = 'away'
        else:
            print("Cannot determine who has ball")

        # find defender/receiver info
        _, pass_arrived_df = find_pass_arrived_df(play_df, possession)
        off_pass_arrived_df = pass_arrived_df.query('team == "' + possession + '"')
        def_pass_arrived_df = pass_arrived_df.query('team != "' + possession + '" and displayName != "Football"')
        off_players_xy = off_pass_arrived_df[["x","y"]].values
        def_players_xy = def_pass_arrived_df[["x","y"]].values
        football_xy = pass_arrived_df.query('displayName == "Football"')[["x","y"]].values
        receiver_idx = np.argmin(np.linalg.norm(football_xy - off_players_xy, axis=1))
        defender_idx = np.argmin(np.linalg.norm(football_xy - def_players_xy, axis=1))
        receiver_df = off_pass_arrived_df.iloc[receiver_idx]
        defender_df = def_pass_arrived_df.iloc[defender_idx]

        # find replacement completion probability
        route = receiver_df.route
        try:
            subs_val = route_db_prox_dict[route]
        except:
            subs_val = route_db_prox_dict["NAN"]
        x[0,3] = subs_val
        logits2 = cpnet_calib(x)
        softmax2 = F.softmax(logits2, dim=1)
        comp_prob2 = softmax2[0,1].item()

        # sanity check
        ipa = 100.0*(comp_prob2 - comp_prob1)
        if db_prox_orig < subs_val:
            guess = 'up'
        else:
            guess = 'down'
        if (comp_prob2 < comp_prob1) and (guess == 'down'):
            make_sense += 1
        elif (comp_prob2 > comp_prob1) and (guess == 'up'):
            make_sense += 1

        # find whether the pass was complete
        pass_result = plays_df.query('gameId == {:.0f} and playId == {:.0f}'.format(game_id, play_id)).passResult.item()

        # get play's epa
        epa = -plays_df.query('gameId == {:.0f} and playId == {:.0f}'.format(game_id, play_id)).epa.item()

        # append results
        new_row = pd.DataFrame([[defender_df.displayName, defender_df.position,receiver_df.displayName,
                                 receiver_df.position, receiver_df.route, ipa, comp_prob1,
                                 pass_result, epa]], columns=column_names)
        metrics_df = metrics_df.append(new_row, ignore_index=True)

        # update progress bar
        disp_dict = {"make_sense_percentage": 100.0*make_sense/(idx+1)}
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()

    progress_bar.close()

    metrics_df.to_csv("../input/nfl-bdb-data/advanced_metrics.csv", index=False)


# handle ties
def handle_ties(values_rank_df, overall_rank_df, c):
    value_counts = values_rank_df[c].value_counts()
    values = value_counts.keys()
    for value in values:
        count = value_counts[value]
        if count > 1:
            overall_rank_df.loc[values_rank_df[c] == value, c] = np.min(overall_rank_df[values_rank_df[c] == value][c])

    return overall_rank_df

# analyze advanced metrics
def analyze_advanced_metrics(pos="cb"):
    metrics_df = pd.read_csv('../input/nfl-bdb-data/advanced_metrics.csv')

    column_names = ["name", "position", "n_throws", "inc_rate", "int_rate", \
                    "epa", "eir", "irae", "ipa"]
    rankings_df = pd.DataFrame(data = np.empty((0,len(column_names))), columns = column_names)
    defenders = np.unique(metrics_df.def_name)
    for defender in tqdm(defenders, desc="Analyzing data"):
        defender_data = metrics_df.query('def_name == "' + defender + '"')
        def_name = defender_data.def_name.iloc[0]
        def_pos = defender_data.def_pos.iloc[0]
        ipa = np.mean(defender_data.ipa)
        n_throws = len(defender_data)
        n_incomp = len(defender_data.query('result == "I"'))
        n_comp = len(defender_data.query('result == "C"'))
        n_int = len(defender_data.query('result == "IN"'))
        inc_rate = 100.0*(n_incomp + n_int) / (n_incomp + n_int + n_comp)
        int_rate = 100.0*(n_int) / (n_incomp + n_int + n_comp)
        exp_inc_rate = 100.0 - 100.0*np.mean(defender_data.comp_prob)
        epa = np.mean(defender_data.epa)
        irae = inc_rate - exp_inc_rate
        new_row = pd.DataFrame([[def_name, def_pos, n_throws, inc_rate, int_rate, epa, exp_inc_rate, irae, ipa]], columns=column_names)
        rankings_df = rankings_df.append(new_row, ignore_index=True)

    # Rankings
    column_names = ["name", "position", "n_throws", "inc_rate", "int_rate", \
                    "epa", "eir", "irae", "ipa", "raw_score", "overall_score"]
    values_rank_df = pd.DataFrame(data = np.empty((0,len(column_names))), columns = column_names)
    overall_rank_df = pd.DataFrame(data = np.empty((0,len(column_names))), columns = column_names)
    if pos == "cb":
        query = 'n_throws >= 45 and (position == "CB")'
    elif pos == "saf":
        query = 'n_throws >= 25 and (position == "FS" or position == "SS" or position == "S" or position == "DB")'
    elif pos == "lb":
        query = 'n_throws >= 25 and (position == "ILB" or position == "LB" or position == "MLB" or position == "OLB")'

    # Incompletion probability added
    rank_df = rankings_df.sort_values(by=['ipa'], ascending=False)
    rank = rank_df.query(query).reset_index().drop(columns=['index'])
    values_rank_df.name = rank.sort_values(by=['name']).name
    values_rank_df.position = rank.sort_values(by=['name']).position
    values_rank_df.n_throws = rank.sort_values(by=['name']).n_throws
    values_rank_df.ipa = rank.sort_values(by=['name']).ipa
    overall_rank_df.name = rank.sort_values(by=['name']).name
    overall_rank_df.position = rank.sort_values(by=['name']).position
    overall_rank_df.n_throws = rank.sort_values(by=['name']).n_throws
    overall_rank_df.ipa = rank.sort_values(by=['name']).index + 1
    values_rank_df = values_rank_df.reset_index().drop(columns=['index'])
    overall_rank_df = overall_rank_df.reset_index().drop(columns=['index'])
    overall_rank_df = handle_ties(values_rank_df, overall_rank_df, 'ipa')

    # Incompletion rate
    rank_df = rankings_df.sort_values(by=['inc_rate'], ascending=False)
    rank = rank_df.query(query).reset_index().drop(columns=['index'])
    values_rank_df.inc_rate = rank.sort_values(by=['name']).inc_rate.reset_index().drop(columns=['index'])
    overall_rank_df.inc_rate = rank.sort_values(by=['name']).index + 1
    overall_rank_df = handle_ties(values_rank_df, overall_rank_df, 'inc_rate')

    # Interception rate
    rank_df = rankings_df.sort_values(by=['int_rate'], ascending=False)
    rank = rank_df.query(query).reset_index().drop(columns=['index'])
    values_rank_df.int_rate = rank.sort_values(by=['name']).int_rate.reset_index().drop(columns=['index'])
    overall_rank_df.int_rate = rank.sort_values(by=['name']).index + 1
    overall_rank_df = handle_ties(values_rank_df, overall_rank_df, 'int_rate')

    # EPA
    rank_df = rankings_df.sort_values(by=['epa'], ascending=False)
    rank = rank_df.query(query).reset_index().drop(columns=['index'])
    values_rank_df.epa = rank.sort_values(by=['name']).epa.reset_index().drop(columns=['index'])
    overall_rank_df.epa = rank.sort_values(by=['name']).index + 1
    overall_rank_df = handle_ties(values_rank_df, overall_rank_df, 'epa')

    # Expected incompletion rate
    rank_df = rankings_df.sort_values(by=['eir'], ascending=False)
    rank = rank_df.query(query).reset_index().drop(columns=['index'])
    values_rank_df.eir = rank.sort_values(by=['name']).eir.reset_index().drop(columns=['index'])
    overall_rank_df.eir = rank.sort_values(by=['name']).index + 1
    overall_rank_df = handle_ties(values_rank_df, overall_rank_df, 'eir')

    # Incompletion rate above expectation
    rank_df = rankings_df.sort_values(by=['irae'], ascending=False)
    rank = rank_df.query(query).reset_index().drop(columns=['index'])
    values_rank_df.irae = rank.sort_values(by=['name']).irae.reset_index().drop(columns=['index'])
    overall_rank_df.irae = rank.sort_values(by=['name']).index + 1
    overall_rank_df = handle_ties(values_rank_df, overall_rank_df, 'irae')

    # Overall rank
    overall_rank_df.raw_score = overall_rank_df.inc_rate + overall_rank_df.int_rate + \
                                    overall_rank_df.epa + overall_rank_df.eir + \
                                    overall_rank_df.irae + overall_rank_df.ipa
    values_rank_df.raw_score = overall_rank_df.raw_score
    # first sort by n_throws, so ties will go to player with higher n_throws
    overall_rank_df = overall_rank_df.sort_values(by=['n_throws'], ascending=False).reset_index().drop(columns=['index'])
    values_rank_df = values_rank_df.sort_values(by=['n_throws'], ascending=False).reset_index().drop(columns=['index'])
    overall_rank_df = overall_rank_df.sort_values(by=['raw_score'], ascending=True, kind='mergesort').reset_index().drop(columns=['index'])
    values_rank_df = values_rank_df.sort_values(by=['raw_score'], ascending=True, kind='mergesort').reset_index().drop(columns=['index'])
    overall_rank_df.index += 1 
    values_rank_df.index += 1

    overall_rank_df.overall_score = 100 - 100*(overall_rank_df.raw_score - overall_rank_df.raw_score.iloc[0]) / \
            (overall_rank_df.raw_score.iloc[-1] - overall_rank_df.raw_score.iloc[0])
    values_rank_df.overall_score = 100 - 100*(overall_rank_df.raw_score - overall_rank_df.raw_score.iloc[0]) / \
            (overall_rank_df.raw_score.iloc[-1] - overall_rank_df.raw_score.iloc[0])

    overall_rank_df = overall_rank_df.round(2)
    values_rank_df = values_rank_df.round(2)

    print("")
    print("")
    print("")
    print(values_rank_df)
    print("")
    print("")
    print("")
    print(overall_rank_df)
    print("")
    print("")
    print("")

    overall_rank_df.to_csv("../input/nfl-bdb-data/" + pos + "_rankings.csv", index=False)
    values_rank_df.to_csv("../input/nfl-bdb-data/" + pos + "_scores.csv", index=False)

# get specific play metrics
def get_specific_play_metrics(week, gameId, playId):
    train_df = pd.read_csv('../input/nfl-bdb-data/training_data.csv')
    slay_play = train_df.query('week == {:.0f} and game == {:.0f} and play == {:.0f}'.format(week, gameId, playId))
    slay_play.result = 'INCOMPLETE'
    slay_play = slay_play.rename(columns={"week": "Week", "game": "Game ID", "play": "Play ID", \
                                          "lat_dist": "Lateral pass distance", "lon_dist": "Longitudinal pass distance", \
                                          "ball_prox": "WR-ball proximity", "db_prox": "WR-DB proximity", \
                                          "sl_prox": "WR-sideline proximity", "bl_prox": "QB-blitzer proximity", \
                                          "qb_speed": "QB speed", "t_throw": "Time to throw", "result": "Result", "route": "Route"})
    print("Play metrics:")
    print(slay_play.iloc[0])

    print("")
    median_dist = np.median(train_df.query('route == "HITCH"').db_prox)
    print("Median WR-DB separation for HITCH routes: {:.2f} yards".format(median_dist))
    print("")
    replacement_slay_play = slay_play.copy()
    replacement_slay_play["WR-DB proximity"] = median_dist

    model_path = "../input/nfl-bdb-data/best_0194_0.483_calib.pt"
    train_df = load_and_clean_data()
    cpnet = setup_network(train_df)
    cpnet_calib = ModelWithTemperature(cpnet)
    cpnet_calib.load_state_dict(torch.load(model_path, map_location='cpu')["model_state_dict"])
    cpnet_calib.eval()
    x0 = torch.from_numpy(slay_play.iloc[0,3:11].values.astype(float)).float().unsqueeze(0)
    logits0 = cpnet_calib(x0)
    softmax0 = F.softmax(logits0, dim=1)
    inc_prob0 = 100*softmax0[0,0].item()
    print("Incompletion probability: {:.1f}%".format(inc_prob0))

    x1 = torch.from_numpy(replacement_slay_play.iloc[0,3:11].values.astype(float)).float().unsqueeze(0)
    logits1 = cpnet_calib(x1)
    softmax1 = F.softmax(logits1, dim=1)
    inc_prob1 = 100*softmax1[0,0].item()
    print("Incompletion probability with median WR-DB separation: {:.1f}%".format(inc_prob1))

    print("Incompletion probability added: {:.1f}%".format(inc_prob0-inc_prob1))