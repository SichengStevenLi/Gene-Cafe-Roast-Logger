import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from storage import list_roasts, load_roast_meta, load_roast_curve, update_roast_notes


st.set_page_config(page_title="Roast Library", layout="wide")
st.title("Roast Library")
st.caption("Browse all roast logs, filter by bean metadata, and view or compare roast curves.")


def _norm_title_key(title: str) -> str:
    return " ".join((title or "").strip().lower().split())


def _fmt_mmss(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"


def _fmt_saved_at(saved_at: str) -> str:
    raw = str(saved_at or "").strip()
    if not raw:
        return "Unknown date/time"
    try:
        return dt.datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return raw


def _parse_mmss_to_sec(mmss: str) -> float:
    s = str(mmss or "").strip()
    if ":" not in s:
        return 0.0
    try:
        m_str, s_str = s.split(":", 1)
        return float(int(m_str) * 60 + int(s_str))
    except Exception:
        return 0.0


def _compute_phase_stats(events: list[dict], elapsed_sec: float) -> dict[str, tuple[float, float]]:
    yellow_t = next((float(e.get("t_sec", 0)) for e in events if e.get("type") == "yellowing_start"), None)
    browning_t = next((float(e.get("t_sec", 0)) for e in events if e.get("type") == "browning_start"), None)
    crack_t = next((float(e.get("t_sec", 0)) for e in events if e.get("type") == "first_crack"), None)

    elapsed = max(0.0, float(elapsed_sec))
    drying = 0.0
    yellowing = 0.0
    maillard = 0.0
    development = 0.0

    if yellow_t is None:
        drying = elapsed
    elif browning_t is None:
        drying = min(elapsed, yellow_t)
        yellowing = max(0.0, elapsed - yellow_t)
    elif crack_t is None:
        drying = min(elapsed, yellow_t)
        yellowing = max(0.0, browning_t - yellow_t)
        maillard = max(0.0, elapsed - browning_t)
    else:
        drying = min(elapsed, yellow_t)
        yellowing = max(0.0, browning_t - yellow_t)
        maillard = max(0.0, crack_t - browning_t)
        development = max(0.0, elapsed - crack_t)

    if elapsed <= 0:
        return {
            "Drying": (0.0, 0.0),
            "Yellowing": (0.0, 0.0),
            "Maillard": (0.0, 0.0),
            "Development": (0.0, 0.0),
        }

    return {
        "Drying": ((drying / elapsed) * 100.0, drying),
        "Yellowing": ((yellowing / elapsed) * 100.0, yellowing),
        "Maillard": ((maillard / elapsed) * 100.0, maillard),
        "Development": ((development / elapsed) * 100.0, development),
    }

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


def _render_roast_panel(rid: str, label_map: dict[str, str], meta_map: dict[str, dict]) -> None:
    meta = meta_map.get(rid, {})
    fig = _build_compare_figure([rid], label_map)
    fig.update_layout(title=label_map.get(rid, rid), showlegend=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    start_temp = int(meta.get("preheat_temp", 0) or 0)
    st.markdown(f"**Starting set temp:** {start_temp} F")

    set_edits = [e for e in (meta.get("events", []) or []) if e.get("type") == "set_change"]
    st.markdown("**Set temp edits**")
    if set_edits:
        for e in set_edits:
            t_text = _fmt_mmss(float(e.get("t_sec", 0.0) or 0.0))
            from_v = e.get("from_value")
            to_v = e.get("value")
            if from_v is None:
                st.write(f"- {t_text}: set to {to_v} F")
            else:
                st.write(f"- {t_text}: {from_v} -> {to_v} F")
    else:
        st.caption("No set-temp edits recorded.")

    try:
        c = _load_curve(rid)
        elapsed = float(c["t_sec"].max()) if (not c.empty and "t_sec" in c.columns) else 0.0
    except Exception:
        elapsed = 0.0
    if elapsed <= 0:
        elapsed = _parse_mmss_to_sec(str(meta.get("total_roast_time", "") or ""))

    phase_stats = _compute_phase_stats(meta.get("events", []) or [], elapsed)
    st.markdown("**Phase ratios**")
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, (pct, secs)) in zip([c1, c2, c3, c4], phase_stats.items()):
        col.metric(label, f"{pct:.1f}%", _fmt_mmss(secs))

    st.markdown("**Weight loss**")
    raw_w = float(meta.get("raw_weight_g", 0.0) or 0.0)
    roasted_w = float(meta.get("roasted_weight_g", 0.0) or 0.0)
    loss_pct = meta.get("weight_loss_pct", None)
    if raw_w > 0 and roasted_w > 0 and loss_pct is not None:
        st.write(f"{raw_w - roasted_w:.1f} g ({float(loss_pct):.1f}%)")
    else:
        st.caption("Weight loss not available.")

    notes_key = f"library_notes_{rid}"
    notes_value = st.text_area(
        "Notes",
        value=str(meta.get("notes", "") or ""),
        key=notes_key,
        height=120,
    )
    if st.button("Save notes", key=f"save_notes_{rid}"):
        update_roast_notes(rid, notes_value)
        _load_library_rows.clear()
        st.success("Notes saved")
        st.rerun()


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

label_map = {row["roast_id"]: row["display_name"] for _, row in filtered.iterrows()}
all_plot_choices = library_df["roast_id"].tolist()
filtered_plot_choices = filtered["roast_id"].tolist()

# Primary roast selection (shown when compare is not explicitly applied)
st.sidebar.divider()
st.sidebar.subheader("Current Roast")
if "library_current_roast" not in st.session_state:
    st.session_state.library_current_roast = filtered_plot_choices[0] if filtered_plot_choices else "(none)"
if st.session_state.library_current_roast not in filtered_plot_choices:
    st.session_state.library_current_roast = filtered_plot_choices[0] if filtered_plot_choices else "(none)"
if "library_current_applied" not in st.session_state:
    st.session_state.library_current_applied = st.session_state.library_current_roast
if st.session_state.library_current_applied not in filtered_plot_choices:
    st.session_state.library_current_applied = st.session_state.library_current_roast

current_roast = st.sidebar.selectbox(
    "Current roast log",
    options=filtered_plot_choices,
    index=filtered_plot_choices.index(st.session_state.library_current_roast) if filtered_plot_choices else 0,
    format_func=lambda rid: label_map.get(rid, rid),
)
st.session_state.library_current_roast = current_roast
if st.sidebar.button("Apply and view selected curve"):
    st.session_state.library_current_applied = current_roast
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Compare Roasts")
compare_options = ["(none)"] + all_plot_choices
default_first = all_plot_choices[0] if all_plot_choices else "(none)"

if "library_compare_first" not in st.session_state:
    st.session_state.library_compare_first = default_first
if "library_compare_second" not in st.session_state:
    st.session_state.library_compare_second = "(none)"
if "library_compare_applied_first" not in st.session_state:
    st.session_state.library_compare_applied_first = "(none)"
if "library_compare_applied_second" not in st.session_state:
    st.session_state.library_compare_applied_second = "(none)"

if st.session_state.library_compare_first not in compare_options:
    st.session_state.library_compare_first = default_first
if st.session_state.library_compare_second not in compare_options:
    st.session_state.library_compare_second = "(none)"

first_roast = st.sidebar.selectbox(
    "First roast curve",
    options=compare_options,
    index=compare_options.index(st.session_state.library_compare_first) if st.session_state.library_compare_first in compare_options else 0,
    format_func=lambda rid: "(none)" if rid == "(none)" else label_map.get(rid, rid),
    key="library_compare_first",
)
second_roast = st.sidebar.selectbox(
    "Second roast curve",
    options=compare_options,
    index=compare_options.index(st.session_state.library_compare_second) if st.session_state.library_compare_second in compare_options else 0,
    format_func=lambda rid: "(none)" if rid == "(none)" else label_map.get(rid, rid),
    key="library_compare_second",
)

if st.sidebar.button("Apply compare"):
    st.session_state.library_compare_applied_first = first_roast
    st.session_state.library_compare_applied_second = second_roast
    st.rerun()

applied_first = st.session_state.library_compare_applied_first
applied_second = st.session_state.library_compare_applied_second

is_two_compare = (
    applied_first != "(none)"
    and applied_second != "(none)"
    and applied_first != applied_second
)

if is_two_compare:
    st.subheader("Compare Roasts")
    left_panel, right_panel = st.columns(2)
    left_meta = meta_map.get(applied_first, {})
    right_meta = meta_map.get(applied_second, {})

    with left_panel:
        st.markdown(f"### {label_map.get(applied_first, applied_first)}")
        st.caption(_fmt_saved_at(str(left_meta.get("saved_at", "") or "")))
        _render_roast_panel(applied_first, label_map, meta_map)

    with right_panel:
        st.markdown(f"### {label_map.get(applied_second, applied_second)}")
        st.caption(_fmt_saved_at(str(right_meta.get("saved_at", "") or "")))
        _render_roast_panel(applied_second, label_map, meta_map)
else:
    # Single-view mode: show current applied roast (or applied compare fallback if set).
    if applied_first != "(none)" and applied_second == "(none)":
        active_rid = applied_first
    elif applied_second != "(none)" and applied_first == "(none)":
        active_rid = applied_second
    else:
        active_rid = st.session_state.library_current_applied

    active_meta = meta_map.get(active_rid, {})
    st.markdown(f"## {label_map.get(active_rid, active_rid)}")
    st.caption(_fmt_saved_at(str(active_meta.get("saved_at", "") or "")))
    _render_roast_panel(active_rid, label_map, meta_map)
