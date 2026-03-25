import numpy as np
import pandas as pd
import importlib


def _c_to_f(temp_c: float) -> float:
    return (temp_c * 9.0 / 5.0) + 32.0


def _format_mmss(seconds: float) -> str:
    total = int(round(seconds))
    m, s = divmod(max(0, total), 60)
    return f"{m}:{s:02d}"


def _smoothed_line(x_vals, y_vals) -> tuple[np.ndarray, np.ndarray]:
    """Return a dense, gently smoothed line passing through sampled points."""
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)

    if x.size <= 2:
        return x, y

    # x_dense: densely sampled x values
    # y_interp: linearly interpolated y values at x_dense
    # y_smooth: smoothed y values using rolling mean
    x_dense = np.linspace(float(x.min()), float(x.max()), max(200, x.size * 20))
    y_interp = np.interp(x_dense, x, y)
    y_smooth = pd.Series(y_interp).rolling(window=9, center=True, min_periods=1).mean().to_numpy()
    return x_dense, y_smooth


class RoastPlotter:
    def __init__(self, xmax_sec: int = 15 * 60):
        self.xmax_sec = xmax_sec

    def make_figure(self, df: pd.DataFrame, events: list[dict], ref_df: pd.DataFrame | None = None):
        try:
            go = importlib.import_module("plotly.graph_objects")
        except Exception as exc:
            raise RuntimeError(
                "Plotly is required for interactive hover/toggle chart. Install with: pip install plotly"
            ) from exc

        fig = go.Figure()

        max_temp_f = 480.0

        # Current temp points only where temp_current is not null
        if not df.empty and "temp_current" in df.columns:
            plot_df = df.dropna(subset=["temp_current"]).sort_values("t_sec")
            if not plot_df.empty:
                y_cur_f = plot_df["temp_current"].astype(float).apply(_c_to_f)
                x_smooth, y_smooth = _smoothed_line(plot_df["t_sec"].to_numpy(), y_cur_f.to_numpy())
                fig.add_trace(
                    go.Scatter(
                        x=x_smooth,
                        y=y_smooth,
                        mode="lines",
                        name="Current curve",
                        line={"color": "#1f77b4", "width": 3},
                        hoverinfo="skip",
                    )
                )

                hover_time = plot_df["t_sec"].apply(_format_mmss)
                hover_temp = y_cur_f.round(1)
                fig.add_trace(
                    go.Scatter(
                        x=plot_df["t_sec"],
                        y=y_cur_f,
                        mode="markers",
                        name="Current points",
                        marker={"color": "#1f77b4", "size": 8},
                        customdata=np.stack([hover_time, hover_temp], axis=-1),
                        hovertemplate="Time: %{customdata[0]}<br>Temperature: %{customdata[1]} F<extra></extra>",
                    )
                )
                max_temp_f = max(max_temp_f, float(y_cur_f.max()) + 10.0)

        # Reference overlay
        if ref_df is not None and not ref_df.empty and "temp_current" in ref_df.columns:
            r = ref_df.dropna(subset=["temp_current"]).sort_values("t_sec")
            if not r.empty:
                y_ref_f = r["temp_current"].astype(float).apply(_c_to_f)
                x_ref_smooth, y_ref_smooth = _smoothed_line(r["t_sec"].to_numpy(), y_ref_f.to_numpy())
                fig.add_trace(
                    go.Scatter(
                        x=x_ref_smooth,
                        y=y_ref_smooth,
                        mode="lines",
                        name="Reference curve",
                        line={"color": "#ff7f0e", "width": 2},
                        opacity=0.45,
                        hoverinfo="skip",
                    )
                )
                max_temp_f = max(max_temp_f, float(y_ref_f.max()) + 10.0)

        # Set-temp change vertical markers
        for e in events or []:
            if e.get("type") == "set_change":
                x = e.get("t_sec", 0)
                fig.add_vline(x=x, line_width=1, line_color="gray", opacity=0.35)

        # Roast stage markers
        yellow_t = None
        browning_t = None
        crack_t = None
        for e in events or []:
            et = e.get("type")
            if et == "yellowing_start" and yellow_t is None:
                yellow_t = float(e.get("t_sec", 0))
            elif et == "browning_start" and browning_t is None:
                browning_t = float(e.get("t_sec", 0))
            elif et == "first_crack" and crack_t is None:
                crack_t = float(e.get("t_sec", 0))

        if yellow_t is not None:
            fig.add_vline(x=yellow_t, line_width=2, line_color="#f2c300", opacity=0.9)
            # Drying phase: 0 -> yellowing start
            fig.add_vrect(x0=0, x1=max(0.0, yellow_t), fillcolor="#f6e6a8", opacity=0.22, line_width=0)

        if yellow_t is not None and browning_t is not None and browning_t >= yellow_t:
            fig.add_vline(x=browning_t, line_width=2, line_color="#b57b38", opacity=0.9)
            # Yellowing phase: yellowing start -> browning start
            fig.add_vrect(x0=yellow_t, x1=browning_t, fillcolor="#f2cf78", opacity=0.20, line_width=0)

        # Maillard phase: browning start -> first crack (or to end if first crack not marked yet)
        if browning_t is not None:
            maillard_end = crack_t if (crack_t is not None and crack_t >= browning_t) else self.xmax_sec
            if maillard_end > browning_t:
                fig.add_vrect(x0=browning_t, x1=maillard_end, fillcolor="#c08a52", opacity=0.18, line_width=0)

        if crack_t is not None:
            fig.add_vline(x=crack_t, line_width=2, line_color="#6b3f1d", opacity=0.95)
            # Development phase: first crack -> end
            fig.add_vrect(x0=crack_t, x1=self.xmax_sec, fillcolor="#7a4a2a", opacity=0.22, line_width=0)

        tick_vals = list(range(0, self.xmax_sec + 1, 30))
        tick_text = [_format_mmss(t) for t in tick_vals]

        fig.update_layout(
            title="Roast curve",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"color": "black"},
            title_font={"color": "black"},
            xaxis={
                "title": "Time (mm:ss)",
                "range": [0, self.xmax_sec],
                "tickmode": "array",
                "tickvals": tick_vals,
                "ticktext": tick_text,
                "tickangle": 45,
                #"titlefont": {"color": "black"},
                "tickfont": {"color": "black"},
                "gridcolor": "#e8e8e8",
                "zerolinecolor": "#e8e8e8",
            },
            yaxis={
                "title": "Temperature (F)",
                "range": [300, max_temp_f],
                #"titlefont": {"color": "black"},
                "tickfont": {"color": "black"},
                "gridcolor": "#e8e8e8",
                "zerolinecolor": "#e8e8e8",
            },
            height=520,
            legend={"orientation": "h", "y": 1.02, "x": 0.0, "font": {"color": "black"}},
            margin={"l": 70, "r": 20, "t": 70, "b": 80},
            hovermode="closest",
            hoverlabel={"bgcolor": "white", "font_color": "black", "bordercolor": "#cccccc"},
        )

        return fig