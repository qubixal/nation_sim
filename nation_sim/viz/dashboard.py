"""Live Plotly/Dash dashboard — stock-exchange terminal theme.

Runs in a background thread alongside the sim and polls the Recorder's in-memory
ring buffers.

Layout (top → bottom = most → least important):
  Header   — world summary ticker strip
  Row 1    — Stock Index chart | Country Progress leaderboard
  Row 2    — Resource Price Board | Price Time-Series | Live Trade Feed
  Row 3    — Trade Volume | Geography Map
  Row 4    — Welfare | Pop | Reserves sparklines
"""
from __future__ import annotations

import datetime as _dt
import logging
import threading
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
import dash

from ..core.resources import RESOURCES
from ..recording.recorder import Recorder

logger = logging.getLogger(__name__)

# ── palette ──────────────────────────────────────────────────────────────────
_BG       = "#07090d"
_SURFACE  = "#0f141d"
_PANEL    = "#141a24"
_PANEL_2  = "#101722"
_BORDER   = "#273244"
_GRID     = "#202a3a"
_TEXT     = "#e7edf6"
_MUTED    = "#8492a8"
_GREEN    = "#2ec27e"
_RED      = "#ff5c5c"
_BLUE     = "#5aa7ff"
_YELLOW   = "#f4c542"
_ORANGE   = "#ff985c"
_CYAN     = "#42dbc8"
_PURPLE   = "#b892ff"
_PINK     = "#ff7aad"

_COUNTRY_COLORS = [
    "#4da3ff",  # blue
    "#e8873d",  # orange
    "#36d7b7",  # teal
    "#b07fd8",  # purple
    "#f0b429",  # gold
    "#e04848",  # red
    "#79c0ff",  # sky
    "#e06090",  # pink
    "#7ee787",  # green
    "#d4a0ff",  # violet
    "#ffa657",  # amber
    "#56d4dd",  # cyan
]

_RESOURCE_COLORS = [
    "#4da3ff",  # food     — blue
    "#e8873d",  # energy   — orange
    "#7ec8e3",  # metal    — light blue
    "#36d7b7",  # wood     — teal
    "#b07fd8",  # goods    — purple
    "#f0b429",  # luxury   — gold
    "#79c0ff",  # tech     — sky blue
    "#e04848",  # fuel     — red
    "#d4a0ff",  # services — violet
]

_CHART_LAYOUT = dict(
    paper_bgcolor=_PANEL,
    plot_bgcolor=_BG,
    font=dict(color=_TEXT, size=12, family="system-ui, 'Segoe UI', Roboto, sans-serif"),
    margin=dict(l=42, r=10, t=28, b=24),
    xaxis=dict(
        gridcolor=_GRID, zerolinecolor=_GRID,
        title_font=dict(color=_MUTED, size=11), tickfont=dict(color=_MUTED, size=10),
    ),
    yaxis=dict(
        gridcolor=_GRID, zerolinecolor=_GRID,
        title_font=dict(color=_MUTED, size=11), tickfont=dict(color=_MUTED, size=10),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", font=dict(color=_MUTED, size=10),
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    ),
    uirevision="persist",
)

_CELL = {
    "padding": "5px 8px",
    "borderBottom": f"1px solid {_BORDER}",
    "fontSize": "13px",
    "lineHeight": "1.4",
}

_STOCK_BADGE = {
    "display": "inline-block",
    "padding": "1px 6px",
    "borderRadius": "3px",
    "fontFamily": "'SF Mono', 'Cascadia Code', 'Consolas', monospace",
    "fontSize": "12px",
    "fontWeight": "600",
    "letterSpacing": "0.5px",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _welfare_bar(w: float) -> html.Span:
    filled = max(0, min(10, round(w * 10)))
    color = _GREEN if w >= 0.75 else (_YELLOW if w >= 0.50 else _RED)
    bar = "█" * filled + "░" * (10 - filled)
    return html.Span(bar, style={"color": color, "fontFamily": "monospace", "fontSize": "12px"})


def _class_bar(poor: float, middle: float, rich: float) -> html.Span:
    """10-cell stacked bar: poor(red) | middle(yellow) | rich(green)."""
    total = max(poor + middle + rich, 1e-9)
    n_poor = max(0, min(10, round(10 * poor / total)))
    n_mid  = max(0, min(10 - n_poor, round(10 * middle / total)))
    n_rich = max(0, 10 - n_poor - n_mid)
    return html.Span(
        [
            html.Span("█" * n_poor, style={"color": _RED}),
            html.Span("█" * n_mid,  style={"color": _YELLOW}),
            html.Span("█" * n_rich, style={"color": _GREEN}),
        ],
        style={"fontFamily": "monospace", "fontSize": "11px", "whiteSpace": "nowrap",
               "letterSpacing": "-0.5px", "lineHeight": "1"},
    )


def _price_change_color(pct: float) -> str:
    if pct > 0.5:
        return _GREEN
    if pct < -0.5:
        return _RED
    return _MUTED


def _stock_badge_html(perf: float) -> html.Span:
    if perf >= 120:
        bg, fg = "rgba(38,166,65,0.18)", _GREEN
    elif perf >= 100:
        bg, fg = "rgba(38,166,65,0.10)", _GREEN
    elif perf >= 80:
        bg, fg = "rgba(240,180,41,0.12)", _YELLOW
    else:
        bg, fg = "rgba(224,72,72,0.15)", _RED
    return html.Span(f"{perf:.1f}", style={**_STOCK_BADGE, "background": bg, "color": fg})


def _panel(**extra) -> dict:
    base = {
        "background": f"linear-gradient(180deg, {_PANEL} 0%, {_PANEL_2} 100%)",
        "border": f"1px solid {_BORDER}",
        "borderRadius": "8px",
        "boxShadow": "0 12px 30px rgba(0,0,0,0.24)",
        "padding": "13px 15px",
        "overflow": "hidden",
    }
    base.update(extra)
    return base


def _section_title(label: str, accent: str = _BLUE, sub: str | None = None) -> html.Div:
    children = [
        html.Span("", style={
            "display": "inline-block", "width": "7px", "height": "7px",
            "borderRadius": "50%", "background": accent, "marginRight": "8px",
            "boxShadow": f"0 0 14px {accent}",
        }),
        html.Span(label, style={
            "color": _TEXT, "fontSize": "12px", "fontWeight": "800",
            "letterSpacing": "1.2px", "textTransform": "uppercase",
        }),
    ]
    if sub:
        children.append(html.Span(sub, style={
            "color": _MUTED, "fontFamily": "monospace", "fontSize": "12px",
            "marginLeft": "10px",
        }))
    return html.Div(children, style={
        "display": "flex", "alignItems": "center", "minHeight": "20px",
        "marginBottom": "6px",
    })


def _delta_badge(value: float, suffix: str = "") -> html.Span:
    color = _GREEN if value > 0 else (_RED if value < 0 else _MUTED)
    arrow = "▲" if value > 0 else ("▼" if value < 0 else "■")
    return html.Span(
        f"{arrow} {value:+.2f}{suffix}",
        style={**_STOCK_BADGE, "background": f"{color}22", "color": color, "padding": "1px 5px"},
    )


# ── header ticker strip ─────────────────────────────────────────────────────

def _header_ticker(cdf: pd.DataFrame, now: str) -> html.Div:
    if cdf.empty:
        return html.Div(
            style={"display": "flex", "justifyContent": "space-between",
                   "alignItems": "center", "padding": "0 6px"},
            children=[
        html.Span("NATION SIMULATOR", style={"color": _BLUE, "fontWeight": "bold",
                                                        "fontSize": "18px", "letterSpacing": "2px"}),
                html.Span("Waiting for sim…", style={"color": _MUTED, "fontFamily": "monospace",
                                                        "fontSize": "12px"}),
            ],
        )

    tick = int(cdf["tick"].max())
    n_countries = cdf["country"].nunique()
    snap = cdf[cdf["tick"] == tick]
    avg_welfare = float(snap["welfare"].mean())
    total_pop = float(snap["population"].sum())
    avg_stock = float(snap["performance"].mean())

    def _stat(label: str, value: str, color: str = _TEXT) -> html.Span:
        return html.Span([
            html.Span(f"{label} ", style={"color": _MUTED, "fontSize": "11px",
                                           "letterSpacing": "0.8px", "textTransform": "uppercase"}),
            html.Span(value, style={"color": color, "fontFamily": "monospace",
                                     "fontSize": "13px", "fontWeight": "600"}),
        ], style={"marginRight": "24px"})

    w_color = _GREEN if avg_welfare >= 0.7 else (_YELLOW if avg_welfare >= 0.5 else _RED)
    s_color = _GREEN if avg_stock >= 100 else (_YELLOW if avg_stock >= 80 else _RED)

    return html.Div(
        style={"display": "flex", "justifyContent": "space-between",
               "alignItems": "center", "padding": "0 6px"},
        children=[
            html.Span("NATION SIMULATOR", style={"color": _BLUE, "fontWeight": "bold",
                                                   "fontSize": "18px", "letterSpacing": "2px"}),
            html.Div(style={"display": "flex", "alignItems": "center"},
                     children=[
                         _stat("tick", f"{tick:>5}", _CYAN),
                         _stat("nations", f"{n_countries}"),
                         _stat("welfare", f"{avg_welfare:.3f}", w_color),
                         _stat("pop", f"{total_pop:.0f}M"),
                         _stat("stock", f"{avg_stock:.1f}", s_color),
                         html.Span(f"  {now}", style={"color": _MUTED, "fontFamily": "monospace",
                                                       "fontSize": "11px"}),
                     ]),
        ],
    )


# ── resource price board ────────────────────────────────────────────────────

def _price_board(rdf: pd.DataFrame) -> html.Table:
    rows: list[html.Tr] = []
    if rdf.empty:
        return html.Table(style={"width": "100%", "borderCollapse": "collapse"})

    ref_country = rdf["country"].iloc[0]
    sub = rdf[rdf["country"] == ref_country].sort_values("tick")
    ticks = sub["tick"].unique()
    latest_tick = ticks[-1]
    prev_tick = ticks[-2] if len(ticks) >= 2 else latest_tick

    latest = sub[sub["tick"] == latest_tick].set_index("resource")["price"]
    prev   = sub[sub["tick"] == prev_tick].set_index("resource")["price"]

    _h = lambda l, a="left": html.Th(
        l, style={**_CELL, "color": _MUTED, "textAlign": a,
                  "borderBottom": f"2px solid {_BORDER}", "fontSize": "11px",
                  "letterSpacing": "0.8px", "textTransform": "uppercase"}
    )
    rows.append(html.Tr([_h("asset"), _h("price", "right"), _h("chg", "right")]))

    for r_idx, name in enumerate(RESOURCES):
        if name not in latest.index:
            continue
        p_now  = float(latest[name])
        p_prev = float(prev.get(name, p_now))
        pct = (p_now - p_prev) / max(p_prev, 1e-6) * 100
        chg_color = _price_change_color(pct)

        rows.append(html.Tr([
            html.Td(
                html.Span([
                    html.Span("●", style={"color": _RESOURCE_COLORS[r_idx % len(_RESOURCE_COLORS)],
                                           "marginRight": "8px", "fontSize": "10px"}),
                    name.upper(),
                ]),
                style={**_CELL, "color": _TEXT, "fontWeight": "600", "fontFamily": "monospace",
                       "letterSpacing": "0.5px"},
            ),
            html.Td(
                f"{p_now:.3f}",
                style={**_CELL, "color": _TEXT, "textAlign": "right", "fontFamily": "monospace"},
            ),
            html.Td(
                f"{pct:+.2f}%",
                style={**_CELL, "color": chg_color, "textAlign": "right", "fontFamily": "monospace",
                       "fontWeight": "600"},
            ),
        ]))

    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})


# ── country leaderboard ─────────────────────────────────────────────────────

_COUNTRY_SORT_COLUMNS: dict[str, tuple[str, bool]] = {
    "country":  ("country",     True),
    "welfare":  ("welfare",     False),
    "pop":      ("population",  False),
    "reserves": ("reserves",    False),
    "stock":    ("performance", False),
    "gdp":      ("gdp",         False),
    "debt":     ("debt",        True),
    "change":   ("stock_change", False),
}


def _country_leaderboard(
    cdf: pd.DataFrame,
    sort_by: str = "stock",
    ascending: bool = False,
) -> html.Table:
    """Country leaderboard with clickable column headers for sorting."""
    rows: list[html.Tr] = []
    if cdf.empty:
        return html.Table(style={"width": "100%", "borderCollapse": "collapse"})

    col, default_ascending = _COUNTRY_SORT_COLUMNS.get(sort_by, _COUNTRY_SORT_COLUMNS["stock"])
    latest_tick = cdf["tick"].max()
    snap = cdf[cdf["tick"] == latest_tick].copy()
    prev_ticks = sorted(t for t in cdf["tick"].unique() if t < latest_tick)
    prev = (
        cdf[cdf["tick"] == prev_ticks[-1]].drop_duplicates("country").set_index("country")
        if prev_ticks else pd.DataFrame()
    )
    snap["stock_change"] = snap.apply(
        lambda row: float(row.get("performance", 100.0) or 100.0)
        - float(prev.loc[row["country"], "performance"] if row["country"] in prev.index else row.get("performance", 100.0)),
        axis=1,
    )
    if col not in snap.columns:
        col = "performance"
    snap = snap.sort_values(col, ascending=ascending)

    _th_style = {
        **_CELL,
        "borderBottom": f"2px solid {_BORDER}",
        "fontSize": "12px",
        "letterSpacing": "0.8px",
        "textTransform": "uppercase",
        "cursor": "pointer",
        "userSelect": "none",
        "padding": "8px 12px",
    }

    def _th(label: str, key: str, align: str = "left") -> html.Th:
        is_active = key == sort_by
        arrow = ""
        if is_active:
            arrow = " ▲" if ascending else " ▼"
        return html.Th(
            html.Button(
                f"{label}{arrow}",
                id={"type": "sort-btn", "index": key},
                n_clicks=0,
                style={
                    "all": "unset", "cursor": "pointer", "display": "block",
                    "color": _BLUE if is_active else _MUTED,
                    "fontSize": "11px", "fontWeight": "800" if is_active else "600",
                    "letterSpacing": "0.8px", "textTransform": "uppercase",
                    "padding": "0", "fontFamily": "inherit",
                },
            ),
            style={**_th_style, "textAlign": align, "color": _BLUE if is_active else _MUTED,
                   "background": "rgba(90,167,255,0.06)" if is_active else "transparent"},
        )

    rows.append(html.Tr([
        _th("#",        "stock",   "center"),
        _th("country",  "country"),
        _th("stock",    "stock",   "right"),
        _th("chg",      "change",  "right"),
        _th("welfare",  "welfare"),
        html.Th("class", style={**_th_style, "cursor": "default"}),
        _th("gdp",      "gdp",     "right"),
        _th("pop M",    "pop",     "right"),
        _th("reserves", "reserves","right"),
        _th("debt",     "debt",    "right"),
    ]))

    for rank, (_, row) in enumerate(snap.iterrows(), 1):
        w = float(row["welfare"])
        debt_val = float(row.get("debt", 0.0) or 0.0)
        stock = float(row.get("performance", 100.0) or 100.0)
        change = float(row.get("stock_change", 0.0) or 0.0)

        rows.append(html.Tr([
            html.Td(
                html.Span(str(rank), style={**_STOCK_BADGE, "background": "rgba(77,163,255,0.10)",
                                             "color": _BLUE}) if rank <= 3
                else html.Span(str(rank), style={"color": _MUTED, "fontFamily": "monospace",
                                                   "fontSize": "13px"}),
                style={**_CELL, "textAlign": "center", "width": "44px"},
            ),
            html.Td(
                row["country"],
                style={**_CELL, "color": _BLUE, "fontWeight": "700",
                       "fontFamily": "monospace", "letterSpacing": "0.5px"},
            ),
            html.Td(_stock_badge_html(stock), style={**_CELL, "textAlign": "right"}),
            html.Td(_delta_badge(change), style={**_CELL, "textAlign": "right"}),
            html.Td(_welfare_bar(w), style={**_CELL, "width": "100px"}),
            html.Td(
                _class_bar(
                    float(row.get("frac_poor", 0.0) or 0.0),
                    float(row.get("frac_middle", 0.0) or 0.0),
                    float(row.get("frac_rich", 0.0) or 0.0),
                ),
                style={**_CELL, "width": "70px", "whiteSpace": "nowrap"},
            ),
            html.Td(
                f"{row.get('gdp', 0.0):.0f}",
                style={**_CELL, "color": _TEXT, "textAlign": "right", "fontFamily": "monospace"},
            ),
            html.Td(
                f"{row['population']:.1f}",
                style={**_CELL, "color": _TEXT, "textAlign": "right", "fontFamily": "monospace"},
            ),
            html.Td(
                f"{row['reserves']:.0f}",
                style={**_CELL, "color": _TEXT, "textAlign": "right", "fontFamily": "monospace"},
            ),
            html.Td(
                f"{debt_val:.0f}" if debt_val > 0 else "—",
                style={**_CELL, "color": _RED if debt_val > 0 else _MUTED,
                       "textAlign": "right", "fontFamily": "monospace"},
            ),
        ]))

    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse", "minWidth": "880px"})



# ── trade feed ──────────────────────────────────────────────────────────────

def _trade_feed(events: list[dict], crashes: list[dict] | None = None) -> list[html.Div]:
    lines: list[html.Div] = []

    if crashes:
        for crash in crashes[:3]:
            lines.append(html.Div(
                [
                    html.Span(f"[{crash['tick']:>4}] ", style={"color": _MUTED, "fontFamily": "monospace"}),
                    html.Span("⚠ ", style={"color": _RED}),
                    html.Span(crash["name"], style={"color": _ORANGE, "fontWeight": "bold",
                                                     "fontFamily": "monospace"}),
                    html.Span(f" — {crash['msg']}", style={"color": _TEXT, "fontSize": "13px"}),
                ],
                style={
                    "fontSize": "13px", "padding": "5px 10px", "lineHeight": "1.5",
                    "background": "rgba(224,72,72,0.08)",
                    "borderLeft": f"3px solid {_RED}",
                    "marginBottom": "3px", "borderRadius": "3px",
                },
            ))
        lines.append(html.Hr(style={"border": f"1px solid {_BORDER}", "margin": "4px 0"}))

    if not events:
        lines.append(html.Div("No trades yet…", style={"color": _MUTED, "padding": "8px",
                                                         "fontStyle": "italic", "fontSize": "13px"}))
        return lines

    r_color_map = {name: _RESOURCE_COLORS[i % len(_RESOURCE_COLORS)] for i, name in enumerate(RESOURCES)}
    for ev in reversed(events[-50:]):
        lines.append(html.Div(
            [
                html.Span(f"[{ev['tick']:>4}] ", style={"color": _MUTED}),
                html.Span(ev["seller"], style={"color": _RED, "fontWeight": "600"}),
                html.Span(" → ", style={"color": _MUTED}),
                html.Span(ev["buyer"], style={"color": _GREEN, "fontWeight": "600"}),
                html.Span(f"  {ev['qty']:>6.1f} ", style={"color": _TEXT}),
                html.Span(f"{ev['resource']:<8}", style={"color": r_color_map.get(ev["resource"], _TEXT)}),
                html.Span(f"@{ev['price']:.2f}", style={"color": _MUTED}),
                html.Span(f" ={ev['value']:.0f}", style={"color": _YELLOW}),
            ],
            style={"fontFamily": "monospace", "fontSize": "12px",
                   "padding": "2px 8px", "lineHeight": "1.5"},
        ))
    return lines


# ── chart builders ──────────────────────────────────────────────────────────

def _stock_index_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df.empty and "performance" in df.columns:
        for i, (name, grp) in enumerate(df.groupby("country")):
            grp_s = grp.sort_values("tick")
            fig.add_trace(go.Scatter(
                x=grp_s["tick"], y=grp_s["performance"],
                mode="lines", name=name,
                line=dict(width=1.8, color=_COUNTRY_COLORS[i % len(_COUNTRY_COLORS)]),
            ))
    fig.update_layout(
        title=dict(text="STOCK INDEX — National Performance",
                   font=dict(color=_TEXT, size=15)),
        xaxis_title="tick",
        yaxis_title="index (start=100)",
        **_CHART_LAYOUT,
    )
    return fig


def _price_time_series(rdf: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not rdf.empty:
        ref_country = rdf["country"].iloc[0]
        sub = rdf[rdf["country"] == ref_country]
        for r_idx, name in enumerate(RESOURCES):
            grp = sub[sub["resource"] == name].sort_values("tick")
            fig.add_trace(go.Scatter(
                x=grp["tick"], y=grp["price"],
                mode="lines", name=name.upper(),
                line=dict(color=_RESOURCE_COLORS[r_idx % len(_RESOURCE_COLORS)], width=1.5),
            ))
    fig.update_layout(
        title=dict(text="RESOURCE PRICES", font=dict(color=_TEXT, size=15)),
        xaxis_title="tick",
        yaxis_title="price (ref)",
        height=260,
        **_CHART_LAYOUT,
    )
    return fig


def _sparkline(df: pd.DataFrame, col: str, title: str, height: int = 110) -> go.Figure:
    """Sparkline with distinct color per country."""
    fig = go.Figure()
    if not df.empty:
        countries = sorted(df["country"].unique())
        for i, name in enumerate(countries):
            grp = df[df["country"] == name].sort_values("tick")
            color = _COUNTRY_COLORS[i % len(_COUNTRY_COLORS)]
            fig.add_trace(go.Scatter(
                x=grp["tick"], y=grp[col],
                mode="lines", name=name,
                line=dict(width=1.5, color=color),
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=_TEXT, size=13)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=height,
        margin=dict(l=6, r=6, t=26, b=6),
        paper_bgcolor=_PANEL,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, size=11),
        legend=dict(visible=False),
        uirevision="persist",
    )
    return fig


def _trade_volume_bar(events: list[dict]) -> go.Figure:
    fig = go.Figure()
    if events:
        from collections import defaultdict
        vol: dict[str, float] = defaultdict(float)
        for ev in events:
            vol[ev["resource"]] += ev["value"]
        names  = [r for r in RESOURCES if r in vol]
        values = [vol[r] for r in names]
        colors = [_RESOURCE_COLORS[RESOURCES.index(r) % len(_RESOURCE_COLORS)] for r in names]
        fig.add_trace(go.Bar(
            x=names, y=values,
            marker_color=colors, showlegend=False,
        ))
    fig.update_layout(
        title=dict(text="TRADE VOLUME", font=dict(color=_TEXT, size=15)),
        xaxis_title="resource",
        yaxis_title="ref value",
        height=240,
        **_CHART_LAYOUT,
    )
    return fig


def _flow_color(intensity: float) -> str:
    intensity = max(0.0, min(1.0, intensity))
    stops = [
        (0.00, (77, 163, 255)),
        (0.40, (54, 215, 183)),
        (0.70, (240, 180, 41)),
        (1.00, (224, 72, 72)),
    ]
    for k in range(len(stops) - 1):
        t0, c0 = stops[k]
        t1, c1 = stops[k + 1]
        if intensity <= t1:
            f = (intensity - t0) / max(t1 - t0, 1e-9)
            r = int(c0[0] + (c1[0] - c0[0]) * f)
            g = int(c0[1] + (c1[1] - c0[1]) * f)
            b = int(c0[2] + (c1[2] - c0[2]) * f)
            return f"rgb({r},{g},{b})"
    return _RED


def _geography_map(
    cdf: pd.DataFrame,
    events: list[dict],
    tariffs: np.ndarray | None = None,
    country_names: list[str] | None = None,
) -> go.Figure:
    fig = go.Figure()
    if cdf.empty or "pos_x" not in cdf.columns:
        fig.update_layout(
            title=dict(text="GEOGRAPHY", font=dict(color=_TEXT, size=15)),
            height=300,
            **_CHART_LAYOUT,
        )
        return fig

    latest = cdf["tick"].max()
    snap = cdf[cdf["tick"] == latest].drop_duplicates("country").set_index("country")

    from collections import defaultdict
    flows: dict[tuple[str, str], float] = defaultdict(float)
    for ev in events:
        flows[(ev["seller"], ev["buyer"])] += ev["value"]

    name_to_idx = {n: i for i, n in enumerate(country_names)} if country_names else {}

    annotations = []
    if flows:
        max_flow = max(flows.values())
        for (seller, buyer), v in flows.items():
            if seller not in snap.index or buyer not in snap.index:
                continue
            intensity = v / max(max_flow, 1e-9)
            x0, y0 = snap.loc[seller, "pos_x"], snap.loc[seller, "pos_y"]
            x1, y1 = snap.loc[buyer, "pos_x"], snap.loc[buyer, "pos_y"]
            color = _flow_color(intensity)
            width = 0.6 + 4.0 * intensity

            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(width=width + 3.0, color=color), opacity=0.12,
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(width=width, color=color), opacity=0.8,
                hoverinfo="text", text=f"{seller} → {buyer}: {v:,.0f} ref",
                showlegend=False,
            ))

            if tariffs is not None and seller in name_to_idx and buyer in name_to_idx:
                e_idx = name_to_idx[seller]
                i_idx = name_to_idx[buyer]
                avg_tariff = float(tariffs[i_idx, e_idx, :].mean())
                if avg_tariff > 0.001:
                    annotations.append(dict(
                        x=(x0 + x1) / 2, y=(y0 + y1) / 2,
                        text=f"{avg_tariff*100:.0f}%",
                        showarrow=False,
                        font=dict(color=_YELLOW, size=11, family="monospace"),
                        bgcolor="rgba(10,14,20,0.75)", borderpad=2,
                    ))

    perf = snap.get("performance", pd.Series([100.0] * len(snap), index=snap.index))
    perf_norm = (perf / max(perf.max(), 1e-6)).clip(0.05, 1.0)
    sizes = 10 + 24 * perf_norm

    fig.add_trace(go.Scatter(
        x=snap["pos_x"], y=snap["pos_y"],
        mode="markers+text", text=snap.index, textposition="top center",
        marker=dict(
            size=sizes, color=snap["welfare"],
            colorscale=[[0, _RED], [0.5, _YELLOW], [1.0, _GREEN]],
            cmin=0.0, cmax=1.0,
            line=dict(width=1.5, color=_BG), showscale=False,
        ),
        textfont=dict(color=_TEXT, size=12, family="monospace"),
        hovertemplate="<b>%{text}</b><br>welfare=%{marker.color:.2f}<br>stock=%{customdata:.1f}<extra></extra>",
        customdata=perf, showlegend=False,
    ))

    layout = {**_CHART_LAYOUT, "annotations": annotations}
    fig.update_layout(
        title=dict(text="GEOGRAPHY & TRADE FLOWS", font=dict(color=_TEXT, size=15)),
        height=300, **layout,
    )
    fig.update_xaxes(range=[-0.05, 1.05], showticklabels=False, title=None)
    fig.update_yaxes(range=[-0.05, 1.05], showticklabels=False, title=None,
                     scaleanchor="x", scaleratio=1)
    return fig


# ── app factory ──────────────────────────────────────────────────────────────

def make_app(recorder: Recorder, update_interval_ms: int = 500) -> Dash:
    app = Dash("nation_sim")
    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Nation Sim Exchange</title>
            {%favicon%}
            {%css%}
            <style>
                * { box-sizing: border-box; }
                body { margin: 0; background: #07090d; }
                ::selection { background: rgba(90, 167, 255, 0.35); }
                ::-webkit-scrollbar { width: 10px; height: 10px; }
                ::-webkit-scrollbar-track { background: #0f141d; }
                ::-webkit-scrollbar-thumb { background: #2d3a4f; border-radius: 8px; }
                ::-webkit-scrollbar-thumb:hover { background: #3d4d66; }
                button:hover { filter: brightness(1.12); }
                label:hover { border-color: #5aa7ff !important; }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """
    app.layout = html.Div(
        style={
            "background": f"radial-gradient(circle at 18% 0%, rgba(90,167,255,0.12), transparent 28%), "
                          f"linear-gradient(180deg, {_BG} 0%, #090d13 100%)",
            "minHeight": "100vh", "padding": "8px 12px 12px",
            "fontFamily": "system-ui, 'Segoe UI', Roboto, sans-serif",
            "color": _TEXT,
        },
        children=[
            # ── header ────────────────────────────────────────────────────
            html.Div(id="header-ticker", style=_panel(**{"marginBottom": "8px", "padding": "10px 12px"})),

            dcc.Interval(id="interval", interval=update_interval_ms, n_intervals=0),
            dcc.Store(id="sort-state", data={"key": "stock", "ascending": False}),
            dcc.Store(id="sort-click"),

            # ── row 1: market index | leaderboard ───────────────────────
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(420px, 1.35fr) minmax(440px, 1fr)",
                    "gridTemplateRows": "300px",
                    "gap": "8px", "marginBottom": "8px",
                },
                children=[
                    html.Div(
                        style={**_panel(), "display": "flex", "flexDirection": "column"},
                        children=[
                            _section_title("Market Index", _BLUE, "National performance"),
                            html.Div(style={"flex": "1", "minHeight": "0"},
                                     children=[dcc.Graph(id="performance", config={"displayModeBar": False},
                                                         style={"height": "100%"})]),
                        ],
                    ),
                    html.Div(
                        style={**_panel(), "display": "flex", "flexDirection": "column"},
                        children=[
                            _section_title("Country Progress", _CYAN, "Click column to sort"),
                            html.Div(id="country-table", style={"flex": "1", "minHeight": "0", "overflow": "auto"}),
                        ],
                    ),
                ],
            ),

            # ── row 2: resource curves | price board | trade tape ───────
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1.4fr 0.7fr 1fr",
                    "gridTemplateRows": "280px",
                    "gap": "8px", "marginBottom": "8px",
                },
                children=[
                    html.Div(
                        style={**_panel(), "display": "flex", "flexDirection": "column"},
                        children=[
                            _section_title("Resource Curves", _ORANGE),
                            html.Div(style={"flex": "1", "minHeight": "0"},
                                     children=[dcc.Graph(id="prices", config={"displayModeBar": False},
                                                         style={"height": "100%"})]),
                        ],
                    ),
                    html.Div(
                        style={**_panel(), "display": "flex", "flexDirection": "column"},
                        children=[
                            _section_title("Price Board", _YELLOW, "Reference market"),
                            html.Div(style={"flex": "1", "minHeight": "0", "overflowY": "auto"},
                                     children=[html.Div(id="price-ticker")]),
                        ],
                    ),
                    html.Div(
                        style={**_panel(**{"overflow": "visible"}), "display": "flex", "flexDirection": "column",
                               "maxHeight": "280px"},
                        children=[
                            html.Div(
                                _section_title("Trade Tape", _GREEN),
                                style={"position": "sticky", "top": "0", "background": _PANEL,
                                       "zIndex": 2, "flexShrink": "0"},
                            ),
                            html.Div(id="trade-feed", style={"flex": "1", "overflowY": "auto",
                                                              "minHeight": "0"}),
                        ],
                    ),
                ],
            ),

            # ── row 3: trade volume | geography ──────────────────────────
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(360px, 0.85fr) minmax(520px, 1.15fr)",
                    "gap": "8px", "marginBottom": "8px",
                },
                children=[
                    html.Div(
                        style={**_panel()},
                        children=[_section_title("Volume By Asset", _PURPLE), dcc.Graph(id="trade-volume", config={"displayModeBar": False})],
                    ),
                    html.Div(
                        style={**_panel()},
                        children=[_section_title("Trade Geography", _PINK), dcc.Graph(id="geography", config={"displayModeBar": False})],
                    ),
                ],
            ),

            # ── row 4: supporting sparklines ─────────────────────────────
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(240px, 1fr))", "gap": "8px"},
                children=[
                    html.Div(style={**_panel()}, children=[dcc.Graph(id="welfare-spark", config={"displayModeBar": False})]),
                    html.Div(style={**_panel()}, children=[dcc.Graph(id="pop-spark", config={"displayModeBar": False})]),
                    html.Div(style={**_panel()}, children=[dcc.Graph(id="reserves-spark", config={"displayModeBar": False})]),
                ],
            ),
        ],
    )

    @app.callback(
        Output("sort-click", "data"),
        Input({"type": "sort-btn", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def capture_click(_clicks):
        if ctx.triggered_id:
            return ctx.triggered_id["index"]
        return dash.no_update

    @app.callback(
        Output("sort-state", "data"),
        Input("sort-click", "data"),
        State("sort-state", "data"),
        prevent_initial_call=True,
    )
    def update_sort(clicked_key, state):
        if not clicked_key:
            return dash.no_update
        state = state or {"key": "stock", "ascending": False}
        previous_key = state.get("key", "stock")
        if clicked_key == previous_key:
            return {"key": clicked_key, "ascending": not state.get("ascending", False)}
        default_asc = _COUNTRY_SORT_COLUMNS.get(clicked_key, (None, False))[1]
        return {"key": clicked_key, "ascending": default_asc}

    @app.callback(
        [
            Output("header-ticker",  "children"),
            Output("country-table",  "children"),
            Output("price-ticker",   "children"),
            Output("performance",    "figure"),
            Output("prices",         "figure"),
            Output("welfare-spark",  "figure"),
            Output("pop-spark",      "figure"),
            Output("reserves-spark", "figure"),
            Output("trade-volume",   "figure"),
            Output("geography",      "figure"),
            Output("trade-feed",     "children"),
        ],
        [Input("interval", "n_intervals"),
         State("sort-state", "data")],
    )
    def update(_n, sort_state):
        cdf    = pd.DataFrame(recorder.live_country_rows)
        rdf    = pd.DataFrame(recorder.live_resource_rows)
        events = recorder.live_trade_events
        crashes = recorder.live_crash_events
        now    = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        sort_state = sort_state or {"key": "stock", "ascending": False}
        sort_by = sort_state.get("key", "stock")
        ascending = bool(sort_state.get("ascending", False))

        return (
            _header_ticker(cdf, now),
            _country_leaderboard(cdf, sort_by=sort_by or "stock", ascending=ascending),
            _price_board(rdf),
            _stock_index_chart(cdf),
            _price_time_series(rdf),
            _sparkline(cdf, "welfare",    "WELFARE"),
            _sparkline(cdf, "population", "POPULATION"),
            _sparkline(cdf, "reserves",   "RESERVES"),
            _trade_volume_bar(events),
            _geography_map(
                cdf, events,
                tariffs=recorder.tariffs,
                country_names=recorder.country_names,
            ),
            _trade_feed(events, crashes=crashes),
        )

    return app


def run_in_thread(app: Dash, host: str = "127.0.0.1", port: int = 8050) -> threading.Thread:
    def _run():
        try:
            logging.getLogger("werkzeug").setLevel(logging.ERROR)
            app.run(host=host, port=port, debug=False, use_reloader=False)
        except Exception:
            logger.exception("Dashboard thread crashed")

    t = threading.Thread(target=_run, daemon=True, name="dash-server")
    t.start()
    time.sleep(1.0)
    if not t.is_alive():
        logger.error("Dashboard thread failed to start")
    return t
