import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from storage import list_roasts, load_roast_meta, load_roast_curve, list_roasts_for_bean


st.set_page_config(page_title="Roast Library", layout="wide")
st.title("Roast Library")
st.caption("Browse all roast logs, filter by bean metadata, and view or compare roast curves.")


def _norm_title_key(title: str) -> str:
    return " ".join((title or "").strip().lower().split())


def _fmt_mmss(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"

# Safe title for filesystem paths, without batch number.
@st.cache_data(show_spinner=False)

def _load_library_rows() -> tuple[pd.DataFrame, dict[str, dict]]:
    # calls the list_roasts and load_roast_meta functions from storage.py to build a DataFrame of roast metadata for the library view, along with a map of roast_id to metadata dict for quick lookup when building the table and details view
    roast_ids = list_roasts()
    # meta_map: roast_id -> metadata dict for quick lookup when building the table and details view
    meta_map: dict[str, dict] = {}
    title_counts: dict[str, int] = {}

    # First pass: load metadata and count bean-title occurrences.
    # for each roast id, it calls load_roast_meta which reads meta.json.
    for rid in roast_ids:
        try:
            meta = load_roast_meta(rid)
        except Exception:
            continue
        meta_map[rid] = meta

        title = str(meta.get("bean_title", "") or "").strip()
        if title:
            key = _norm_title_key(title)
            title_counts[key] = title_counts.get(key, 0) + 1

    # Second pass: build rows with display names and parsed metadata.
    rows = []
    for rid in roast_ids:
        if rid not in meta_map:
            continue

        meta = meta_map[rid]
        title = str(meta.get("bean_title", "") or "").strip()
        if not title:
            title = rid

        key = _norm_title_key(title)
        batch = int(meta.get("batch_number", 1) or 1)
        show_batch = title_counts.get(key, 0) > 1
        display_name = f"{title} #{batch}" if show_batch else title

        saved_at_raw = str(meta.get("saved_at", "") or "")
        saved_date = None
        if saved_at_raw:
            try:
                saved_date = dt.datetime.fromisoformat(saved_at_raw).date()
            except Exception:
                saved_date = None

        rows.append(
            {
                "roast_id": rid,
                "display_name": display_name,
                "bean_title": title,
                "batch_number": batch,
                "origin": str(meta.get("origin", "") or ""),
                "process": str(meta.get("process", "") or ""),
                "variety": str(meta.get("variety", "") or ""),
                "appearance": str(meta.get("bean_appearance", "") or ""),
                "decaf": bool(meta.get("is_decaf", False)),
                "altitude_m": int(meta.get("altitude_m", 0) or 0),
                "raw_weight_g": float(meta.get("raw_weight_g", 0.0) or 0.0),
                "roasted_weight_g": float(meta.get("roasted_weight_g", 0.0) or 0.0),
                "weight_loss_pct": meta.get("weight_loss_pct", None),
                "saved_at": saved_at_raw,
                "saved_date": saved_date,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["saved_date", "batch_number", "display_name"], ascending=[False, False, True])
    return df, meta_map


@st.cache_data(show_spinner=False)
# calls load_roast_curve from storage.py to read the roast curve data for a given roast_id
def _load_curve(roast_id: str) -> pd.DataFrame:
    return load_roast_curve(roast_id)


def _build_compare_figure(selected_roasts: list[str], labels: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    max_temp = 482.0
    max_t = 0.0

    for rid in selected_roasts:
        try:
            curve = _load_curve(rid)
        except Exception:
            continue

        if curve.empty or "t_sec" not in curve.columns or "temp_current" not in curve.columns:
            continue

        c = curve.dropna(subset=["temp_current"]).sort_values("t_sec")
        if c.empty:
            continue

        x = c["t_sec"].astype(float)
        y = c["temp_current"].astype(float)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=labels.get(rid, rid),
                line={"width": 2.5},
                hovertemplate="Time: %{x:.1f}s<br>Temp: %{y:.1f} F<extra></extra>",
            )
        )

        max_temp = max(max_temp, float(y.max()) + 10.0)
        max_t = max(max_t, float(x.max()))

    tick_vals = list(range(0, int(max_t) + 31, 30))
    tick_text = [_fmt_mmss(t) for t in tick_vals]

    fig.update_layout(
        title="Roast Curves",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "black"},
        xaxis={
            "title": "Time (mm:ss)",
            "tickmode": "array",
            "tickvals": tick_vals,
            "ticktext": tick_text,
            "tickangle": 45,
            "gridcolor": "#e8e8e8",
        },
        yaxis={
            "title": "Temperature (F)",
            "range": [300, max_temp],
            "gridcolor": "#e8e8e8",
        },
        height=520,
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        margin={"l": 70, "r": 20, "t": 70, "b": 80},
        hovermode="closest",
    )
    return fig


library_df, meta_map = _load_library_rows()

if library_df.empty:
    st.info("No roast logs found yet.")
    st.stop()

st.sidebar.header("Library Filters")
search_text = st.sidebar.text_input("Search coffee name", value="")

origin_options = ["(all)"] + sorted([x for x in library_df["origin"].dropna().unique().tolist() if x])
process_options = ["(all)"] + sorted([x for x in library_df["process"].dropna().unique().tolist() if x])
variety_options = ["(all)"] + sorted([x for x in library_df["variety"].dropna().unique().tolist() if x])
bean_options = ["(all)"] + sorted([x for x in library_df["bean_title"].dropna().unique().tolist() if x])

filter_origin = st.sidebar.selectbox("Origin", origin_options, index=0)
filter_process = st.sidebar.selectbox("Process", process_options, index=0)
filter_variety = st.sidebar.selectbox("Variety", variety_options, index=0)
filter_bean = st.sidebar.selectbox("Coffee", bean_options, index=0)
filter_decaf = st.sidebar.selectbox("Decaf", ["(all)", "Non-decaf", "Decaf"], index=0)

# Optional date filter if dates exist.
dates = [d for d in library_df["saved_date"].dropna().tolist() if d is not None]
use_date_filter = st.sidebar.checkbox("Filter by date range", value=False)
start_date = None
end_date = None
if use_date_filter and dates:
    min_d, max_d = min(dates), max(dates)
    start_date, end_date = st.sidebar.date_input("Saved date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

# Apply filters.
filtered = library_df.copy()

if search_text.strip():
    q = search_text.strip().lower()
    filtered = filtered[filtered["display_name"].str.lower().str.contains(q, na=False)]
if filter_origin != "(all)":
    filtered = filtered[filtered["origin"] == filter_origin]
if filter_process != "(all)":
    filtered = filtered[filtered["process"] == filter_process]
if filter_variety != "(all)":
    filtered = filtered[filtered["variety"] == filter_variety]
if filter_bean != "(all)":
    filtered = filtered[filtered["bean_title"] == filter_bean]
if filter_decaf == "Decaf":
    filtered = filtered[filtered["decaf"] == True]
elif filter_decaf == "Non-decaf":
    filtered = filtered[filtered["decaf"] == False]
if use_date_filter and start_date and end_date:
    filtered = filtered[(filtered["saved_date"] >= start_date) & (filtered["saved_date"] <= end_date)]

if filtered.empty:
    st.warning("No roast logs match your filters.")
    st.stop()

st.subheader("Roast Logs")

visible_cols = [
    "display_name",
    "saved_at",
    "origin",
    "process",
    "variety",
    "altitude_m",
    "raw_weight_g",
    "roasted_weight_g",
    "weight_loss_pct",
]
show_df = filtered[visible_cols].copy()
show_df = show_df.rename(
    columns={
        "display_name": "Roast",
        "saved_at": "Saved at",
        "origin": "Origin",
        "process": "Process",
        "variety": "Variety",
        "altitude_m": "Altitude (m)",
        "raw_weight_g": "Raw (g)",
        "roasted_weight_g": "Roasted (g)",
        "weight_loss_pct": "Loss %",
    }
)
st.dataframe(show_df, use_container_width=True, hide_index=True)

# Same-bean version picker for quick navigation.
st.sidebar.divider()
st.sidebar.subheader("Same-Bean Versions")
bean_version_titles = [x for x in sorted(filtered["bean_title"].dropna().unique().tolist()) if x]
selected_version = None
if bean_version_titles:
    bean_for_versions = st.sidebar.selectbox("Bean title", bean_version_titles, index=0)
    version_options = [rid for rid, _meta in list_roasts_for_bean(bean_for_versions)]
    version_labels = {
        rid: next((row["display_name"] for _, row in filtered.iterrows() if row["roast_id"] == rid), rid)
        for rid in version_options
    }

    if version_options:
        selected_version = st.sidebar.selectbox(
            "Versions (newest to oldest)",
            version_options,
            index=0,
            format_func=lambda rid: version_labels.get(rid, rid),
        )
else:
    st.sidebar.caption("No named bean titles in current filter.")

label_map = {row["roast_id"]: row["display_name"] for _, row in filtered.iterrows()}
plot_choices = filtered["roast_id"].tolist()
default_selection = [selected_version] if selected_version else plot_choices[:1]

selected_roasts = st.multiselect(
    "Select roast(s) to view/compare",
    options=plot_choices,
    default=default_selection,
    format_func=lambda rid: label_map.get(rid, rid),
)

if not selected_roasts:
    st.info("Select at least one roast to plo