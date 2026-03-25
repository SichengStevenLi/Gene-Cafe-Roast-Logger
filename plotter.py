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
                        x=x_smoot