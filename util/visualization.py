from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches
from util import *
from pdb import set_trace as bp


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
        elif player == receiver_name: # intended receiver (thicker blue)
            ax.plot(x, y, color='b', linewidth=3)
        elif player == defender_name: # defender (thicker red)
            ax.plot(x, y, color='r', linewidth=3)
        elif position in defense_positions: # defense (red)
            ax.plot(x, y, color='r', linewidth=1)
        else: # offense (blue)
            ax.plot(x, y, color='b', linewidth=1)
    plt.show()


# plot football field
def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='blue')
        #plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
        #         color='yellow')
    return fig, ax


# animate play
def animate_play(play_df):
    play_id = play_df.playId.iloc[0]
    home_query = 'playId == {:.0f} and team == "home"'.format(play_id)
    away_query = 'playId == {:.0f} and team == "away"'.format(play_id)
    football_query = 'playId == {:.0f} and displayName == "Football"'.format(play_id)
    play_home = play_df.query(home_query)
    play_away = play_df.query(away_query)
    play_football = play_df.query(football_query)
    fig, ax = create_football_field(highlight_line=True, highlight_line_number=play_football.iloc[0]["x"]-10)
    plt.ion()
    plt.show()
    for i in range(1,len(play_football)+1):
        frame_query = 'frameId == {:.0f}'.format(i)
        home_pts = plt.scatter(play_home.query(frame_query)["x"], play_home.query(frame_query)["y"], c='orange', marker='o')
        away_pts = plt.scatter(play_away.query(frame_query)["x"], play_away.query(frame_query)["y"], c='blue', marker='o')
        football_pt = plt.scatter(play_football.query(frame_query)["x"], play_football.query(frame_query)["y"], c='black', marker='o')
        event = play_df.query(frame_query).event.iloc[0]
        if event != 'None':
            plt.title(event)
        plt.pause(0.001)
        home_pts.remove()
        away_pts.remove()
        football_pt.remove()

# animate play 2
def animate_play2(week, game_id, play_id):
    week_df = load_week_df(week)
    game_df = week_df.query('gameId == {:.0f}'.format(game_id))
    play_df = game_df.query('playId == {:.0f}'.format(play_id))
    animate_play(play_df)