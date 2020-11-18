from matplotlib import pyplot as plt
import numpy as np

# plot play
def plot_play(play_df, offense_positions, defense_positions, receiver_name, defender_name):
    fig = plt.figure()
    ax = plt.gca()
    players = np.unique(play_df["displayName"])
    for player in players:
        player_df = play_df[play_df["displayName"] == player]
        x = player_df["x"]
        y = player_df["y"]
        name = player_df["displayName"].iloc[0]
        position = player_df["position"].iloc[0]
        if name == 'Football': # ball (black)
            ax.plot(x, y, color='k', linewidth=2)
        elif player == receiver_name: # intended receiver (cyan)
            ax.plot(x, y, color='c', linewidth=2)
        elif player == defender_name: # defender (magenta)
            ax.plot(x, y, color='m', linewidth=2)
        elif position in defense_positions: # defense (red)
            ax.plot(x, y, color='r', linewidth=2)
        else: # offense (blue)
            ax.plot(x, y, color='b', linewidth=2)
    print(np.unique(play_df["event"][play_df["event"] != "None"])) # print events of play (not in chronological order)
    plt.show()