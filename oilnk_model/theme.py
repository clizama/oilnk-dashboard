"""oilnk palette + Plotly layout, mirroring common/theme.css and figs_style.R."""

OILNK = {
    "ink": "#23303B",
    "slate": "#2E4756",
    "ad": "#3E6D8E",
    "sras": "#D08C34",
    "shock": "#A8322D",
    "green": "#5B8C6E",
    "paper": "#FBFAF7",
    "muted": "#8A97A0",
    "grid": "#E7E1D6",
}

# colours by policy regime / nested case
REGIME_COLORS = {
    "optimo": OILNK["green"],
    "core": OILNK["ad"],
    "headline": OILNK["shock"],
}


def plotly_layout(**kwargs):
    """Base Plotly layout in the oilnk style."""
    layout = dict(
        paper_bgcolor=OILNK["paper"],
        plot_bgcolor=OILNK["paper"],
        font=dict(color=OILNK["ink"], size=14),
        margin=dict(l=55, r=20, t=50, b=45),
        xaxis=dict(gridcolor=OILNK["grid"], zerolinecolor=OILNK["muted"],
                   linecolor=OILNK["ink"]),
        yaxis=dict(gridcolor=OILNK["grid"], zerolinecolor=OILNK["muted"],
                   linecolor=OILNK["ink"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    layout.update(kwargs)
    return layout
