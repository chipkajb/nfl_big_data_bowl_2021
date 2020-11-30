import plotly.express as px
import plotly.graph_objects as go

def plot_playmaking_skills():
    fig = px.scatter(values_rank_df,
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
        x0=50, y0=-0.8, x1=60, y1=0.4,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_trace(go.Scatter(
        x=[40, 55],
        y=[0.35, -0.5],
        text=["Top Playmaking Ability", "Top Tracking Ability"],
        mode="text",
        textfont_size=20
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

def plot_ball_skills():
    fig = px.scatter(values_rank_df,
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
        x0=-15, y0=5, x1=20, y1=12,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_shape(type="rect",
        x0=5, y0=-1, x1=20, y1=12,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_trace(go.Scatter(
        x=[-5, 16],
        y=[10, 0],
        text=["Top Takeaway Ability", "Top Pass Breakup Ability"],
        mode="text",
        textfont_size=20
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

def plot_coverage_skills():
    fig = px.scatter(values_rank_df,
                        x="irar",
                        y="inc_rate",
                        color="overall_score",
                        color_continuous_scale="portland",
                        hover_data=["name"],
                        title="Coverage Skills",
                        labels={
                         "irar": "IRAR",
                         "inc_rate": "INC rate",
                         "overall_score": "Overall score",
                         "name": "Name"
                     })
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.add_shape(type="rect",
        x0=5, y0=50, x1=26, y1=60,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_shape(type="rect",
        x0=20, y0=15, x1=26, y1=60,
        line=dict(
            color="rgba(255, 0, 0, 0.5)",
            width=2,
        ),
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
    fig.add_trace(go.Scatter(
        x=[10, 23],
        y=[57, 22],
        text=["Top Shutdown Ability", "Top True Coverage Ability"],
        mode="text",
        textfont_size=20
    ))
    fig.update_layout(showlegend=False)
    fig.update_layout(
        xaxis_title="Incompletion Rate Above Replacement (IRAR) (%)",
        yaxis_title="Incompletion Rate (%)",
        font=dict(
            size=14,
        )
    )
    fig.show()