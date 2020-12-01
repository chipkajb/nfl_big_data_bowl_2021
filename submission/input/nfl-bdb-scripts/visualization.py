import plotly.express as px
import plotly.graph_objects as go
from pdb import set_trace as bp
import pandas as pd
import matplotlib.patches as patches
from matplotlib import pyplot as plt

def plot_playmaking_skills(scores_df):
    fig = px.scatter(scores_df,
                        x="eir",
                        y="epa",
                        color="overall_score",
                        color_continuous_scale="portland",
                        hover_data=["name"],
                        title="Playmaking Skills",
                        labels={
                        "eir": "EIR",
                        "epa": "EPA",
                        "overall_score": "Overall score",
                        "name": "Name"
                    })
    fig.update_traces(marker=dict(size=12,
                                line=dict(width=2,
                                            color='DarkSlateGrey')),
                    selector=dict(mode='markers'))
    fig.add_shape(type="rect",
        x0=20, y0=0, x1=60, y1=0.4,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_shape(type="rect",
        x0=48, y0=-0.8, x1=60, y1=0.4,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_trace(go.Scatter(
        x=[30, 55],
        y=[0.35, -0.5],
        text=["Top Playmaking Ability", "Top Tracking<br>Ability"],
        mode="text",
        textfont_size=16
    ))
    fig.update_layout(showlegend=False)
    fig.update_layout(
        xaxis_title="Expected Incompletion Rate (EIR) (%)",
        yaxis_title="Expected Points Added (EPA)",
        font=dict(
            size=14,
        )
    )
    fig.show()

def plot_ball_skills(scores_df):
    fig = px.scatter(scores_df,
                        x="irae",
                        y="int_rate",
                        color="overall_score",
                        color_continuous_scale="portland",
                        hover_data=["name"],
                        title="Ball Skills",
                        labels={
                         "irae": "IRAE",
                         "int_rate": "INT rate",
                         "overall_score": "Overall score",
                         "name": "Name"
                     })
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.add_shape(type="rect",
        x0=-15, y0=4, x1=17, y1=12,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_shape(type="rect",
        x0=5, y0=-1, x1=17, y1=12,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_trace(go.Scatter(
        x=[-8, 13],
        y=[10, 0],
        text=["Top Takeaway Ability", "Top Pass Breakup<br>Ability"],
        mode="text",
        textfont_size=16
    ))
    fig.update_layout(showlegend=False)
    fig.update_layout(
        xaxis_title="Incompletion Rate Above Expectation (IRAE) (%)",
        yaxis_title="Interception Rate (%)",
        font=dict(
            size=14,
        )
    )
    fig.show()

def plot_coverage_skills(scores_df):
    fig = px.scatter(scores_df,
                        x="ipa",
                        y="inc_rate",
                        color="overall_score",
                        color_continuous_scale="portland",
                        hover_data=["name"],
                        title="Coverage Skills",
                        labels={
                         "ipa": "IPA",
                         "inc_rate": "INC rate",
                         "overall_score": "Overall score",
                         "name": "Name"
                     })
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.add_shape(type="rect",
        x0=-2, y0=50, x1=18, y1=60,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_shape(type="rect",
        x0=12, y0=15, x1=18, y1=60,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_trace(go.Scatter(
        x=[2, 15],
        y=[57, 22],
        text=["Top Shutdown Ability", "Top True Coverage<br>Ability"],
        mode="text",
        textfont_size=16
    ))
    fig.update_layout(showlegend=False)
    fig.update_layout(
        xaxis_title="Incompletion Probability Added (IPA) (%)",
        yaxis_title="Incompletion Rate (%)",
        font=dict(
            size=14,
        )
    )
    fig.show()

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
def animate_play(week, gameId, playId):
    week_df = pd.read_csv('../input/nfl-big-data-bowl-2021/week{:.0f}.csv'.format(week))
    plays_df = pd.read_csv('../input/nfl-big-data-bowl-2021/plays.csv')
    game_df = week_df.query('gameId == {:.0f}'.format(gameId))
    play_df = game_df.query('playId == {:.0f}'.format(playId))
    playId = play_df.playId.iloc[0]
    home_query = 'playId == {:.0f} and team == "home"'.format(playId)
    away_query = 'playId == {:.0f} and team == "away"'.format(playId)
    football_query = 'playId == {:.0f} and displayName == "Football"'.format(playId)
    play_home = play_df.query(home_query)
    play_away = play_df.query(away_query)
    play_football = play_df.query(football_query)
    fig, ax = create_football_field(highlight_line=True, highlight_line_number=play_football.iloc[0]["x"]-10)
    plt.ion()
    plt.show()
    description = plays_df.query('gameId == {:.0f} and playId == {:.0f}'.format(gameId, playId)).playDescription.values[0]
    plt.title(description)
    for i in range(1,len(play_football)+1):
        frame_query = 'frameId == {:.0f}'.format(i)
        home_pts = plt.scatter(play_home.query(frame_query)["x"], play_home.query(frame_query)["y"], c='orange', marker='o')
        away_pts = plt.scatter(play_away.query(frame_query)["x"], play_away.query(frame_query)["y"], c='blue', marker='o')
        football_pt = plt.scatter(play_football.query(frame_query)["x"], play_football.query(frame_query)["y"], c='black', marker='o')
        event = play_df.query(frame_query).event.iloc[0]
        plt.pause(0.001)
        home_pts.remove()
        away_pts.remove()
        football_pt.remove()