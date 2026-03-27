"""
    Main Streamlit app for Gene Café Roast Logger MVP.
    Center of the app: UI, state management, and orchestration of components.
    Uses:
    - camera.py: Camera access and ROI cropping
    - ocr.py: OCR reading of temperature digits from ROI
    - classifier.py: Classifier logic to determine CURRENT_VIEW vs SET_VIEW
    - plotter.py: Plotting roast curves
    - storage.py: Saving/loading roast sessions and metadata
    - recommend.py: Simple roast recommendation based on metadata
"""

import time
from dataclasses import asdict
import re
import cv2
import streamlit as st
import pandas as pd


from camera import Camera
from ocr import read_temperature_from_frame, get_ocr_status
from classifier import TempClassifier, ClassifierResult
from plotter import RoastPlotter
from storage import (
    RoastMeta,
    ensure_data_dirs,
    save_roast_session,
    update_roasted_weight,
    list_roasts,
    load_roast_curve,
    load_roast_meta,
    load_camera_config,
    save_camera_config,
    save_bean_profile,
    list_roasts_for_bean,
    next_batch_number,
    make_roast_log_name,
)
from recommend import recommend_roasts, score_from_meta_cache


APP_TITLE = "Gene Café Roast Logger"
# customizable sampling interval (seconds)
SAMPLE_INTERVAL_SEC = 5.0
# Run OCR at a non-1s cadence so we don't phase-lock with a 1s display toggle.
OCR_READ_INTERVAL_SEC = 0.7
# Small preview tick for smoother camera updates without pegging CPU.
PREVIEW_REFRESH_SEC = 0.08
PLOT_WINDOW_SEC = 15 * 60
CAMERA_SCAN_MAX_INDEX = 4

ORIGIN_OPTIONS = [
    "(select)",
    "Ethiopia",
    "Colombia",
    "Rwanda",
    "Mexico",
    "Kenya",
    "Guatemala",
    "Costa Rica",
    "Brazil",
    "Panama",
    "El Salvador",
    "Honduras",
    "Costa Rica",
    "Nicaragua",
    "Hawaii",
    "Indonesia",
    "Vietnam",
    "India",
    "Peru",

    "Other",
]

BEAN_APPEARANCE_OPTIONS = [
    "(select)",
    "Small Beans",
    "Medium Beans",
    "Large Beans",
    "Peaberry",
    "Mixed Sizes",
    "Other",
]

TYPE_OPTIONS = [
    "(select)",
    "Arabica",
    "Robusta",
    "Liberica",
    "Excelsa",
    "Other",
]

VARIETY_OPTIONS = [
    "(select)",
    "Heirloom",
    "Gesha",
    "Caturra",
    "Catimor",
    "Bourbon",
    "Typica",
    "Laurina",
    "SL28",
    "SL34",
    "Castillo",
    "Pacamara",
    "Maragogype",
    "Other",
]

PROCESS_OPTIONS = [
    "(select)",
    "Washed",
    "Natural",
    "Honey",
    "Wet Hulled",
    "Anaerobic",
    "Anaerobic Honey",
    "Anaerobic Washed",
    "Anaerobic Natural",
    "Carbonic Maceration",
    "Thermal Shock",
    "Semi-Washed",
    "Cofermentation",
    "Experimental",
    "Other",
]

# Note: for simplicity, all state is in session_state. 
# In a more complex app, we might want to split into multiple pages or use a more robust state management approach.
def init_state():
    # Initialize all session state variables with defaults if they don't exist yet.
    # running state starts when roast starts, not on app load, so default is False
    if "running" not in st.session_state:
        st.session_state.running = False
    if "start_epoch" not in st.session_state:
        st.session_state.start_epoch = None
    if "last_sample_epoch" not in st.session_state:
        st.session_state.last_sample_epoch = 0.0
    if "last_ocr_epoch" not in st.session_state:
        st.session_state.last_ocr_epoch = 0.0
    if "last_capture_epoch" not in st.session_state:
        st.session_state.last_capture_epoch = 0.0
    if "live_frame" not in st.session_state:
        st.session_state.live_frame = None
    if "live_roi" not in st.session_state:
        st.session_state.live_roi = None
    if "capture_buffer" not in st.session_state:
        st.session_state.capture_buffer = []
    if "last_ocr_frame" not in st.session_state:
        st.session_state.last_ocr_frame = None
    if "last_ocr_roi" not in st.session_state:
        st.session_state.last_ocr_roi = None
    if "pending_ocr" not in st.session_state:
        # Holds the most recent confirmed CURRENT_VIEW read within the current 5s window
        st.session_state.pending_ocr = None  # dict with keys raw_read, result
    if "samples" not in st.session_state:
        st.session_state.samples = []  # list of dict rows
    if "events" not in st.session_state:
        st.session_state.events = []   # set temp change events
    if "set_temp" not in st.session_state:
        st.session_state.set_temp = None
    if "set_temp_is_applied" not in st.session_state:
        st.session_state.set_temp_is_applied = False
    if "current_set_temp_input" not in st.session_state:
        st.session_state.current_set_temp_input = 0
    if "classifier" not in st.session_state:
        st.session_state.classifier = TempClassifier()
    if "camera" not in st.session_state:
        st.session_state.camera = None
    if "roast_id" not in st.session_state:
        st.session_state.roast_id = None
    if "reference_roast_id" not in st.session_state:
        st.session_state.reference_roast_id = None
    if "_last_reference_prefill_roast_id" not in st.session_state:
        st.session_state._last_reference_prefill_roast_id = None
    if "preview_camera" not in st.session_state:
        st.session_state.preview_camera = None
    if "preview_live" not in st.session_state:
        st.session_state.preview_live = True
    if "preview_refresh_sec" not in st.session_state:
        st.session_state.preview_refresh_sec = PREVIEW_REFRESH_SEC
    if "active_monitor_live" not in st.session_state:
        st.session_state.active_monitor_live = True
    if "active_monitor_refresh_sec" not in st.session_state:
        st.session_state.active_monitor_refresh_sec = PREVIEW_REFRESH_SEC
    if "preview_cam_index" not in st.session_state:
        st.session_state.preview_cam_index = None
    if "preview_roi" not in st.session_state:
        st.session_state.preview_roi = None
    if "disabled_point_ids" not in st.session_state:
        st.session_state.disabled_point_ids = []
    if "roast_active" not in st.session_state:
        st.session_state.roast_active = False
    if "end_confirm_pending" not in st.session_state:
        st.session_state.end_confirm_pending = False
    if "final_roasted_weight" not in st.session_state:
        st.session_state.final_roasted_weight = 0.0
    if "final_total_roast_time" not in st.session_state:
        st.session_state.final_total_roast_time = ""
    if "end_confirm_prev_running" not in st.session_state:
        st.session_state.end_confirm_prev_running = False
    if "last_saved_roast_id" not in st.session_state:
        st.session_state.last_saved_roast_id = None
    if "suggested_roasts" not in st.session_state:
        st.session_state.suggested_roasts = []  # list[(roast_id, score)]
    if "suggested_roast_choice" not in st.session_state:
        st.session_state.suggested_roast_choice = None
    if "reference_mode" not in st.session_state:
        existing_source = st.session_state.get("reference_source")
        if st.session_state.get("reference_roast_id") and existing_source in {"manual", "suggested", "same_bean"}:
            st.session_state.reference_mode = existing_source
        else:
            st.session_state.reference_mode = "none"
    if "manual_ref_select" not in st.session_state:
        st.session_state.manual_ref_select = "(none)"
    if "_prev_manual_ref_select" not in st.session_state:
        st.session_state._prev_manual_ref_select = "(none)"
    if "reference_source" not in st.session_state:
        # Tracks which section last set the reference roast: "manual" | "suggested" | "same_bean"
        st.session_state.reference_source = None
    if "bean_title_input" not in st.session_state:
        st.session_state.bean_title_input = ""
    if "origin_choice_input" not in st.session_state:
        st.session_state.origin_choice_input = "(select)"
    if "origin_custom_input" not in st.session_state:
        st.session_state.origin_custom_input = ""
    if "type_choice_input" not in st.session_state:
        st.session_state.type_choice_input = "(select)"
    if "type_custom_input" not in st.session_state:
        st.session_state.type_custom_input = ""
    if "variety_choice_input" not in st.session_state:
        st.session_state.variety_choice_input = "(select)"
    if "variety_custom_input" not in st.session_state:
        st.session_state.variety_custom_input = ""
    if "appearance_choice_input" not in st.session_state:
        st.session_state.appearance_choice_input = "(select)"
    if "appearance_custom_input" not in st.session_state:
        st.session_state.appearance_custom_input = ""
    if "process_choice_input" not in st.session_state:
        st.session_state.process_choice_input = "(select)"
    if "process_custom_input" not in st.session_state:
        st.session_state.process_custom_input = ""
    if "altitude_input" not in st.session_state:
        st.session_state.altitude_input = 0
    if "is_decaf_input" not in st.session_state:
        st.session_state.is_decaf_input = False
    if "raw_weight_input" not in st.session_state:
        st.session_state.raw_weight_input = 0.0
    if "roast_notes_input" not in st.session_state:
        st.session_state.roast_notes_input = ""
    if "_clear_roast_notes_on_rerun" not in st.session_state:
        st.session_state._clear_roast_notes_on_rerun = False
    if "cam_index_input" not in st.session_state:
        _cam_cfg = load_camera_config()
        st.session_state.cam_index_input = int(_cam_cfg["cam_index"])
        st.session_state.roi_x_input = int(_cam_cfg["roi_x"])
        st.session_state.roi_y_input = int(_cam_cfg["roi_y"])
        st.session_state.roi_w_input = int(_cam_cfg["roi_w"])
        st.session_state.roi_h_input = int(_cam_cfg["roi_h"])
    if "camera_ready_cache_key" not in st.session_state:
        st.session_state.camera_ready_cache_key = None
    if "camera_ready_cache_ts" not in st.session_state:
        st.session_state.camera_ready_cache_ts = 0.0
    if "camera_ready_cache_ok" not in st.session_state:
        st.session_state.camera_ready_cache_ok = False
    if "camera_ready_cache_msg" not in st.session_state:
        st.session_state.camera_ready_cache_msg = "Camera check pending"
    if "detected_cameras" not in st.session_state:
        st.session_state.detected_cameras = []
    if "camera_scan_done" not in st.session_state:
        st.session_state.camera_scan_done = False
    if "camera_select_input" not in st.session_state:
        st.session_state.camera_select_input = int(st.session_state.cam_index_input)

# Get elapsed seconds since roast start, or 0 if not started
def current_elapsed_sec() -> float:
    if st.session_state.start_epoch is None:
        return 0.0
    return time.time() - st.session_state.start_epoch

# Add a new sample to session state with classifier result
def add_sample(elapsed_sec: float, raw_read: int | None, result: ClassifierResult):
    st.session_state.samples.append(
        {
            "t_sec": round(elapsed_sec, 3), # keep 3 decimals for better x-axis plotting
            "raw_read": raw_read, # for debugging; not necessarily the same as temp_current
            "view_mode": result.view_mode, # "CURRENT_VIEW" | "SET_VIEW" | "UNKNOWN"
            "temp_current": result.current_temp, # this is what we plot as the "current temp" curve (can be None if uncertain)
            "set_temp": result.set_temp, # for reference; should match the latest set temp input
            "confidence": result.confidence, # for debugging; not currently used in logic, but could be used to filter out low-confidence OCR reads in the future
        }
    )


def _camera_ready(index: int, roi: tuple[int, int, int, int]) -> tuple[bool, str]:
    cam = Camera(index=index, roi=roi)
    frame, _ = cam.read()
    cam.close()
    if frame is None:
        return False, "Camera is not readable with current index/ROI."
    return True, "OK"


def _probe_camera_index(idx: int, warmup_reads: int = 8, open_attempts: int = 2) -> bool:
    # On macOS, some USB cameras only respond on specific backends or after a longer warmup.
    backends = [cv2.CAP_ANY]
    avf = getattr(cv2, "CAP_AVFOUNDATION", None)
    if avf is not None:
        backends = [avf, cv2.CAP_ANY]

    for backend in backends:
        for _ in range(max(1, int(open_attempts))):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                time.sleep(0.05)
                continue

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Give USB devices a moment to settle after opening.
            time.sleep(0.10)

            ok = False
            frame = None
            for _ in range(max(3, int(warmup_reads))):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    break
                time.sleep(0.04)

            cap.release()

            if ok and frame is not None and frame.size > 0:
                return True

    return False


def _detect_camera_indices(max_index: int = 6, passes: int = 2) -> list[int]:
    found: set[int] = set()
    for _ in range(max(1, int(passes))):
        for idx in range(max(1, int(max_index))):
            if idx in found:
                continue
            if _probe_camera_index(idx):
                found.add(idx)
        if len(found) == max(1, int(max_index)):
            break
        time.sleep(0.10)
    return sorted(found)


def _camera_ready_cached(index: int, roi: tuple[int, int, int, int], ttl_sec: float = 1.0) -> tuple[bool, str]:
    key = (int(index), int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    now = time.time()
    cache_key = st.session_state.get("camera_ready_cache_key")
    cache_ts = float(st.session_state.get("camera_ready_cache_ts", 0.0))
    if cache_key == key and (now - cache_ts) < float(ttl_sec):
        return bool(st.session_state.get("camera_ready_cache_ok", False)), str(
            st.session_state.get("camera_ready_cache_msg", "Camera check pending")
        )

    ok, msg = _camera_ready(index=index, roi=roi)
    st.session_state.camera_ready_cache_key = key
    st.session_state.camera_ready_cache_ts = now
    st.session_state.camera_ready_cache_ok = ok
    st.session_state.camera_ready_cache_msg = msg
    return ok, msg


def _valid_mmss(s: str) -> bool:
    if not s:
        return False
    if not re.fullmatch(r"\d{1,2}:\d{2}", s.strip()):
        return False
    m_str, s_str = s.strip().split(":")
    return 0 <= int(s_str) < 60 and int(m_str) >= 0


def _format_mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


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


def main():
    # Initialize state and ensure data directories exist
    ensure_data_dirs()
    init_state()

    # Streamlit page setup
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        #MainMenu,
        footer {
            visibility: hidden;
        }
        .stApp .main .block-container,
        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }
        .stApp h1,
        [data-testid="stAppViewContainer"] h1 {
            margin-top: 0rem !important;
            padding-top: 0rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(APP_TITLE)

    ocr_ready, ocr_message = get_ocr_status()
    if not ocr_ready:
        st.warning(f"{ocr_message} Temperature reads will be unavailable until OCR is configured.")

    # ---- Sidebar: Metadata + Controls ----
    st.sidebar.header("Roast Setup")

    # Bean metadata inputs (before roast)
    # De-duplicate by bean title: show only the most recent roast per unique bean name.
    # list_roasts() returns newest-first so the first seen per title is the latest batch.
    _all_profile_ids = list_roasts()
    roast_profile_meta: dict[str, dict] = {}
    roast_profile_labels: dict[str, str] = {}
    _seen_profile_titles: set[str] = set()
    roast_profile_ids: list[str] = []

    def _norm_profile_title(title: str) -> str:
        return " ".join((title or "").strip().lower().split())

    for rid in _all_profile_ids:
        try:
            meta = load_roast_meta(rid)
        except Exception:
            meta = {}
        roast_profile_meta[rid] = meta
        title = str(meta.get("bean_title", "") or meta.get("title", "") or "").strip()
        key = _norm_profile_title(title) if title else rid
        if key in _seen_profile_titles:
            continue
        _seen_profile_titles.add(key)
        roast_profile_ids.append(rid)
        roast_profile_labels[rid] = title if title else rid

    bean_profile_choices = ["(none)"] + roast_profile_ids
    selected_profile = st.sidebar.selectbox(
        "Saved bean profile",
        bean_profile_choices,
        index=0,
        format_func=lambda rid: "(none)" if rid == "(none)" else roast_profile_labels.get(rid, rid),
        disabled=st.session_state.roast_active,
    )
    st.sidebar.caption("Type in the dropdown to narrow the list.")
    st.sidebar.caption("Press Load Bean Profile to automatically fill in the fields.")
    load_profile_btn = st.sidebar.button("Load bean profile", disabled=st.session_state.roast_active)

    if st.session_state.roast_active:
        st.sidebar.caption("Bean profile is locked while roast is active.")

    if load_profile_btn and selected_profile != "(none)":
        p = load_roast_meta(selected_profile) or {}

        def _set_choice_and_custom(value: str, options: list[str], choice_key: str, custom_key: str):
            clean = (value or "").strip()
            if clean and clean in options and clean != "Other":
                st.session_state[choice_key] = clean
                st.session_state[custom_key] = ""
            elif clean:
                st.session_state[choice_key] = "Other"
                st.session_state[custom_key] = clean
            else:
                st.session_state[choice_key] = "(select)"
                st.session_state[custom_key] = ""

        st.session_state.bean_title_input = str(p.get("bean_title", "") or p.get("title", "") or selected_profile)
        _set_choice_and_custom(str(p.get("origin", "") or ""), ORIGIN_OPTIONS, "origin_choice_input", "origin_custom_input")
        _set_choice_and_custom(str(p.get("bean_category", "") or ""), TYPE_OPTIONS, "type_choice_input", "type_custom_input")
        _set_choice_and_custom(str(p.get("variety", "") or ""), VARIETY_OPTIONS, "variety_choice_input", "variety_custom_input")
        _set_choice_and_custom(str(p.get("bean_appearance", "") or ""), BEAN_APPEARANCE_OPTIONS, "appearance_choice_input", "appearance_custom_input")
        _set_choice_and_custom(str(p.get("process", "") or ""), PROCESS_OPTIONS, "process_choice_input", "process_custom_input")
        st.session_state.altitude_input = int(p.get("altitude_m", 0) or 0)
        st.session_state.is_decaf_input = bool(p.get("is_decaf", False))
        st.session_state.raw_weight_input = float(p.get("raw_weight_g", 0.0) or 0.0)

        # Loading a profile seeds the input, but the user must still explicitly apply it.
        profile_set_temp = int(p.get("preheat_temp", 0) or 0)
        st.session_state.current_set_temp_input = profile_set_temp
        st.session_state.set_temp = None
        st.session_state.set_temp_is_applied = False
        st.rerun()

    bean_title = st.sidebar.text_input("Coffee name", value=st.session_state.get("bean_title_input", ""), key="bean_title_input")

    origin_choice = st.sidebar.selectbox(
        "Origin",
        ORIGIN_OPTIONS,
        index=ORIGIN_OPTIONS.index(st.session_state.get("origin_choice_input", "(select)")) if st.session_state.get("origin_choice_input", "(select)") in ORIGIN_OPTIONS else 0,
        key="origin_choice_input",
    )
    origin = origin_choice
    if origin_choice == "Other":
        origin = st.sidebar.text_input("Origin (custom)", value=st.session_state.get("origin_custom_input", ""), key="origin_custom_input")

    type_choice = st.sidebar.selectbox(
        "Type",
        TYPE_OPTIONS,
        index=TYPE_OPTIONS.index(st.session_state.get("type_choice_input", "(select)")) if st.session_state.get("type_choice_input", "(select)") in TYPE_OPTIONS else 0,
        key="type_choice_input",
    )
    bean_type_value = type_choice
    if type_choice == "Other":
        bean_type_value = st.sidebar.text_input("Type (custom)", value=st.session_state.get("type_custom_input", ""), key="type_custom_input")

    variety_choice = st.sidebar.selectbox(
        "Variety",
        VARIETY_OPTIONS,
        index=VARIETY_OPTIONS.index(st.session_state.get("variety_choice_input", "(select)")) if st.session_state.get("variety_choice_input", "(select)") in VARIETY_OPTIONS else 0,
        key="variety_choice_input",
    )
    variety_value = variety_choice
    if variety_choice == "Other":
        variety_value = st.sidebar.text_input("Variety (custom)", value=st.session_state.get("variety_custom_input", ""), key="variety_custom_input")

    size_choice = st.sidebar.selectbox(
        "Appearance",
        BEAN_APPEARANCE_OPTIONS,
        index=BEAN_APPEARANCE_OPTIONS.index(st.session_state.get("appearance_choice_input", "(select)")) if st.session_state.get("appearance_choice_input", "(select)") in BEAN_APPEARANCE_OPTIONS else 0,
        key="appearance_choice_input",
    )
    appearance = size_choice
    if size_choice == "Other":
        appearance = st.sidebar.text_input("Appearance (custom)", value=st.session_state.get("appearance_custom_input", ""), key="appearance_custom_input")

    # Keep compatibility with existing storage/recommendation schema.
    bean_type = ""
    if bean_type_value.strip() and variety_value.strip() and bean_type_value != "(select)" and variety_value != "(select)":
        bean_type = f"{bean_type_value} / {variety_value}"
    elif bean_type_value.strip() and bean_type_value != "(select)":
        bean_type = bean_type_value
    elif variety_value.strip() and variety_value != "(select)":
        bean_type = variety_value

    altitude = st.sidebar.number_input(
        "Altitude (m)",
        min_value=0,
        max_value=10000,
        value=int(st.session_state.get("altitude_input", 0)),
        step=10,
        key="altitude_input",
    )
    process_choice = st.sidebar.selectbox(
        "Processing method",
        PROCESS_OPTIONS,
        index=PROCESS_OPTIONS.index(st.session_state.get("process_choice_input", "(select)")) if st.session_state.get("process_choice_input", "(select)") in PROCESS_OPTIONS else 0,
        key="process_choice_input",
    )
    process = process_choice
    if process_choice == "Other":
        process = st.sidebar.text_input("Processing method (custom)", value=st.session_state.get("process_custom_input", ""), key="process_custom_input")

    is_decaf = st.sidebar.checkbox("Decaf", value=bool(st.session_state.get("is_decaf_input", False)), key="is_decaf_input")

    raw_weight = st.sidebar.number_input(
        "Raw weight (g)",
        min_value=0.0,
        value=float(st.session_state.get("raw_weight_input", 0.0)),
        step=0.1,
        key="raw_weight_input",
    )

    if st.sidebar.button("Save bean profile"):
        if not bean_title.strip():
            st.sidebar.warning("Coffee name is required to save a bean profile.")
        else:
            save_bean_profile(
                bean_title,
                {
                    "origin": origin if origin != "(select)" else "",
                    "bean_category": bean_type_value if bean_type_value != "(select)" else "",
                    "variety": variety_value if variety_value != "(select)" else "",
                    "bean_appearance": appearance if appearance != "(select)" else "",
                    "altitude_m": int(altitude),
                    "process": process if process != "(select)" else "",
                    "is_decaf": bool(is_decaf),
                    "raw_weight_g": float(raw_weight),
                },
            )
            st.sidebar.success(f"Saved bean profile: {bean_title}")

    st.sidebar.divider()

    # Camera detection and selection
    if not st.session_state.camera_scan_done:
        st.session_state.detected_cameras = _detect_camera_indices(max_index=CAMERA_SCAN_MAX_INDEX)
        st.session_state.camera_scan_done = True

    detect_cols = st.sidebar.columns(2)
    detect_cols[0].caption("Camera")
    if detect_cols[1].button("Detect", disabled=st.session_state.roast_active):
        st.session_state.detected_cameras = _detect_camera_indices(max_index=CAMERA_SCAN_MAX_INDEX)
    st.sidebar.caption(f"Detect scans camera indexes 0-{CAMERA_SCAN_MAX_INDEX - 1} using macOS-friendly probing.")

    detected_cams = sorted(set(int(x) for x in st.session_state.detected_cameras))
    current_cam = int(st.session_state.cam_index_input)

    if detected_cams:
        cam_options = sorted(set(detected_cams + [current_cam]))
        selected_cam = int(st.session_state.get("camera_select_input", current_cam))
        if selected_cam not in cam_options:
            selected_cam = current_cam if current_cam in cam_options else cam_options[0]
            st.session_state.camera_select_input = int(selected_cam)

        cam_index = st.sidebar.selectbox(
            "Detected camera",
            options=cam_options,
            index=cam_options.index(int(st.session_state.camera_select_input)),
            format_func=lambda x: f"Camera {x}",
            key="camera_select_input",
            disabled=st.session_state.roast_active,
        )
        st.session_state.cam_index_input = int(cam_index)
        st.sidebar.caption("Detected: " + ", ".join(str(x) for x in detected_cams))
    else:
        st.sidebar.caption("No camera auto-detected. Use manual index.")
        cam_index = st.sidebar.number_input(
            "Camera index (manual)",
            min_value=0,
            step=1,
            key="cam_index_input",
            disabled=st.session_state.roast_active,
        )
        st.session_state.camera_select_input = int(cam_index)

    # ROI calibration
    st.sidebar.caption("ROI (pixels): x, y, w, h")
    roi_x = st.sidebar.number_input("ROI x", min_value=0, step=1, key="roi_x_input", disabled=st.session_state.roast_active)
    roi_y = st.sidebar.number_input("ROI y", min_value=0, step=1, key="roi_y_input", disabled=st.session_state.roast_active)
    roi_w = st.sidebar.number_input("ROI w", min_value=10, step=10, key="roi_w_input", disabled=st.session_state.roast_active)
    roi_h = st.sidebar.number_input("ROI h", min_value=10, step=10, key="roi_h_input", disabled=st.session_state.roast_active)
    if st.sidebar.button("Save camera settings", help="Remember these ROI and camera values for next time"):
        save_camera_config(int(cam_index), int(roi_x), int(roi_y), int(roi_w), int(roi_h))
        st.sidebar.success("Camera settings saved.")

    st.sidebar.divider()

    if st.session_state.roast_active and st.session_state.camera is not None:
        camera_ok, camera_msg = True, "OK"
    elif (
        st.session_state.preview_camera is not None
        and int(st.session_state.preview_cam_index or -1) == int(cam_index)
    ):
        camera_ok, camera_msg = True, "OK"
    else:
        camera_ok, camera_msg = _camera_ready_cached(
            index=int(cam_index),
            roi=(int(roi_x), int(roi_y), int(roi_w), int(roi_h)),
        )

    set_temp_ready = (
        bool(st.session_state.get("set_temp_is_applied", False))
        and isinstance(st.session_state.set_temp, int)
        and st.session_state.set_temp > 0
        and int(st.session_state.current_set_temp_input) == int(st.session_state.set_temp)
    )

    requirement_checks = [
        ("Coffee name", bool(bean_title.strip()), "Coffee name is required"),
        ("Origin", bool(origin.strip()) and origin != "(select)", "Origin is required"),
        ("Type", bool(bean_type_value.strip()) and bean_type_value != "(select)", "Type is required"),
        ("Variety", bool(variety_value.strip()) and variety_value != "(select)", "Variety is required"),
        ("Appearance", bool(appearance.strip()) and appearance != "(select)", "Appearance is required"),
        ("Processing method", bool(process.strip()) and process != "(select)", "Processing method is required"),
        ("Raw weight", float(raw_weight) > 0, "Raw weight must be greater than 0"),
        (
            "Current set temp",
            set_temp_ready,
            "Click Apply set temp(F) before starting",
        ),
        ("Camera", camera_ok, camera_msg),
        ("OCR", ocr_ready, "OCR must be configured before start"),
    ]
    unmet_requirements = [message for _, ok, message in requirement_checks if not ok]

    if not st.session_state.roast_active:
        st.sidebar.caption("Start requirements:")
        for label, ok, _ in requirement_checks:
            status = "PASS" if ok else "MISSING"
            bg = "rgba(20, 163, 74, 0.32)" if ok else "rgba(248, 81, 73, 0.08)"
            border = "rgba(21, 128, 61, 0.75)" if ok else "rgba(248, 81, 73, 0.25)"
            st.sidebar.markdown(
                (
                    f"<div style='background:{bg}; border:1px solid {border}; "
                    "padding:6px 8px; border-radius:6px; margin-bottom:6px;'>"
                    f"<strong>{label}</strong>: {status}</div>"
                ),
                unsafe_allow_html=True,
            )

        if unmet_requirements:
            st.sidebar.warning("Fix missing requirements before starting.")
        else:
            st.sidebar.success("Ready to start")

    st.sidebar.divider()
    st.sidebar.header("Reference curve (optional)")

    # Recommend similar roasts based on metadata if available
    roasts = list_roasts()

    roast_meta_cache: dict[str, dict] = {}
    title_counts: dict[str, int] = {}

    def _norm_title_key(title: str) -> str:
        return " ".join((title or "").strip().lower().split())

    for rid in roasts:
        try:
            m = load_roast_meta(rid)
            roast_meta_cache[rid] = m
            t = str(m.get("bean_title", "") or "").strip()
            if t:
                key = _norm_title_key(t)
                title_counts[key] = title_counts.get(key, 0) + 1
        except Exception:
            continue

    def _roast_label(rid: str) -> str:
        m = roast_meta_cache.get(rid, {})
        t = str(m.get("bean_title", "") or "").strip()
        if not t:
            return rid
        count = title_counts.get(_norm_title_key(t), 0)
        batch_no = int(m.get("batch_number", 1) or 1)
        if count <= 1:
            return t
        return f"{t} #{batch_no}"

    def _stage_caption(rid: str) -> str:
        """Return a short stage-times string like 'Y: 3:45  M: 6:10  1C: 8:30' from cached meta events."""
        m = roast_meta_cache.get(rid, {})
        evts = m.get("events") or []
        yellow_t = next((float(e["t_sec"]) for e in evts if e.get("type") == "yellowing_start"), None)
        browning_t = next((float(e["t_sec"]) for e in evts if e.get("type") == "browning_start"), None)
        crack_t = next((float(e["t_sec"]) for e in evts if e.get("type") == "first_crack"), None)
        parts = []
        if yellow_t is not None:
            parts.append(f"Y: {_format_mmss(yellow_t)}")
        if browning_t is not None:
            parts.append(f"M: {_format_mmss(browning_t)}")
        if crack_t is not None:
            parts.append(f"1C: {_format_mmss(crack_t)}")
        return "  ".join(parts) if parts else ""

    # Precompute per-roast similarity so each selector can show a dynamic score text below it.
    _all_curve_scores: dict[str, float | None] = {}
    for rid in roasts:
        _all_curve_scores[rid] = score_from_meta_cache(
            rid=rid,
            meta_cache=roast_meta_cache,
            origin=origin,
            altitude=int(altitude),
            process=process,
            appearance=appearance,
            raw_weight_g=float(raw_weight),
            is_decaf=bool(is_decaf),
            bean_category=bean_type_value if bean_type_value != "(select)" else "",
            variety=variety_value if variety_value != "(select)" else "",
        )

    bean_title_clean = bean_title.strip()
    same_bean_versions = list_roasts_for_bean(bean_title_clean) if bean_title_clean else []
    same_bean_ids = [rid for rid, _ in same_bean_versions]

    suggested_pairs = [
        (rid, score)
        for rid, score in _all_curve_scores.items()
        if score is not None and rid not in same_bean_ids
    ]
    suggested_pairs.sort(key=lambda item: item[1], reverse=True)
    suggested_pairs = suggested_pairs[:5]
    suggestion_scores = {rid: score for rid, score in suggested_pairs}
    suggestion_options = [rid for rid, _ in suggested_pairs]
    st.session_state.suggested_roasts = suggested_pairs

    mode_labels = {
        "none": "No reference",
        "same_bean": "Use same-bean roast",
        "suggested": "Use suggested match",
        "manual": "Browse all saved roasts",
    }
    mode_options = ["none", "same_bean", "suggested", "manual"]

    if st.session_state.reference_mode not in mode_options:
        st.session_state.reference_mode = "none"

    selected_mode = st.sidebar.radio(
        "Reference mode",
        options=mode_options,
        index=mode_options.index(st.session_state.reference_mode),
        format_func=lambda mode: mode_labels[mode],
        key="reference_mode",
        help="Choose one source for the active reference curve, or disable reference entirely.",
    )

    active_reference_id = st.session_state.reference_roast_id
    pending_reference_id: str | None = None
    pending_reference_source: str | None = None

    if selected_mode == "none":
        st.session_state.reference_source = None
        st.session_state.reference_roast_id = None
        active_reference_id = None
        st.sidebar.caption("Active reference: None")

    elif selected_mode == "same_bean":
        if same_bean_ids:
            if st.session_state.get("same_bean_versions_select") not in same_bean_ids:
                st.session_state.same_bean_versions_select = same_bean_ids[0]
            selected_same_bean_version = st.sidebar.selectbox(
                "Same-bean roast version",
                same_bean_ids,
                format_func=lambda rid: _roast_label(rid),
                key="same_bean_versions_select",
                help="Previous batches of this same coffee, newest first.",
            )
            pending_reference_id = selected_same_bean_version
            pending_reference_source = "same_bean"
        else:
            st.sidebar.info("No saved roast versions found for this coffee yet.")

    elif selected_mode == "suggested":
        if suggestion_options:
            if st.session_state.suggested_roast_choice not in suggestion_options:
                st.session_state.suggested_roast_choice = suggestion_options[0]
            selected_suggestion = st.sidebar.selectbox(
                "Suggested reference",
                suggestion_options,
                index=suggestion_options.index(st.session_state.suggested_roast_choice),
                format_func=lambda rid: _roast_label(rid),
                help="Top compatibility matches from your roast history.",
            )
            st.session_state.suggested_roast_choice = selected_suggestion
            pending_reference_id = selected_suggestion
            pending_reference_source = "suggested"
        else:
            st.sidebar.info("No compatible saved roasts available yet.")

    else:
        manual_options = ["(none)"] + roasts
        if st.session_state.manual_ref_select not in manual_options:
            st.session_state.manual_ref_select = "(none)"
        selected_manual = st.sidebar.selectbox(
            "Select reference roast",
            manual_options,
            format_func=lambda rid: "(none)" if rid == "(none)" else _roast_label(rid),
            key="manual_ref_select",
            help="Browse all roast logs manually.",
        )
        pending_reference_id = None if selected_manual == "(none)" else selected_manual
        pending_reference_source = "manual"

    if selected_mode != "none" and pending_reference_source is not None:
        apply_reference = st.sidebar.button("Use selected reference", key="apply_selected_reference")
        if apply_reference:
            if pending_reference_id is None:
                st.session_state.reference_source = None
                st.session_state.reference_roast_id = None
                active_reference_id = None
            else:
                st.session_state.reference_source = pending_reference_source
                st.session_state.reference_roast_id = pending_reference_id
                active_reference_id = pending_reference_id
            st.rerun()

    if active_reference_id is None:
        st.sidebar.caption("Active reference: None")

    if active_reference_id:
        # st.sidebar.caption(f"Active reference: {_roast_label(active_reference_id)}")
        _active_score = _all_curve_scores.get(active_reference_id)
        if _active_score is not None:
            st.sidebar.caption(f"Match score: **{_active_score:.1f}%**")
        else:
            st.sidebar.caption("Match score: N/A (decaf mismatch)")

    # When the user chooses a different reference before starting, prefill the set-temp input
    # with that roast's initial starting temperature. The user still needs to click Apply.
    if not st.session_state.roast_active:
        last_prefill_ref = st.session_state.get("_last_reference_prefill_roast_id")
        current_prefill_ref = active_reference_id
        if current_prefill_ref != last_prefill_ref:
            st.session_state._last_reference_prefill_roast_id = current_prefill_ref
            if current_prefill_ref:
                ref_meta = roast_meta_cache.get(current_prefill_ref, {})
                ref_start_temp = int(ref_meta.get("preheat_temp", 0) or 0)
                if ref_start_temp > 0:
                    st.session_state.current_set_temp_input = ref_start_temp
                    st.session_state.set_temp_is_applied = False
                    st.rerun()


    start_btn = st.sidebar.button(
        "Start logging",
        type="primary",
        disabled=st.session_state.roast_active or len(unmet_requirements) > 0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Roast Controls")
    set_temp_input = st.sidebar.number_input(
        "Current set temp(F)",
        min_value=0,
        value=int(st.session_state.current_set_temp_input),
        step=1,
        key="current_set_temp_input",
    )
    new_set_temp = int(set_temp_input)
    if int(st.session_state.set_temp or 0) != new_set_temp:
        st.session_state.set_temp_is_applied = False
    apply_set = st.sidebar.button("Apply set temp(F)")

    if st.session_state.get("_clear_roast_notes_on_rerun", False):
        st.session_state.roast_notes_input = ""
        st.session_state._clear_roast_notes_on_rerun = False

    st.sidebar.text_area(
        "Roast notes",
        key="roast_notes_input",
        height=100,
        help="You can write notes before, during, or after logging. Notes are saved with the roast.",
    )

    pause_resume_label = "Pause roast" if st.session_state.running else "Resume roast"
    pause_resume_btn = st.sidebar.button(
        pause_resume_label,
        type="secondary",
        disabled=not st.session_state.roast_active,
    )

    end_roast_btn = st.sidebar.button(
        "End roast",
        type="primary",
        disabled=not st.session_state.roast_active,
    )
    reset_session = st.sidebar.button("Reset session", type="secondary")

    # ---- Control actions ----
    if reset_session:
        st.session_state.running = False
        st.session_state.roast_active = False
        st.session_state.end_confirm_pending = False
        st.session_state.start_epoch = None
        st.session_state.last_sample_epoch = 0.0
        st.session_state.last_ocr_epoch = 0.0
        st.session_state.last_capture_epoch = 0.0
        st.session_state.live_frame = None
        st.session_state.live_roi = None
        st.session_state.capture_buffer = []
        st.session_state.last_ocr_frame = None
        st.session_state.last_ocr_roi = None
        st.session_state.pending_ocr = None
        st.session_state.samples = []
        st.session_state.events = []
        st.session_state.set_temp = None
        st.session_state.set_temp_is_applied = False
        st.session_state.classifier = TempClassifier()
        if st.session_state.camera:
            try:
                st.session_state.camera.close()
            except Exception:
                pass
        if st.session_state.preview_camera:
            try:
                st.session_state.preview_camera.close()
            except Exception:
                pass
        st.session_state.camera = None
        st.session_state.preview_camera = None
        st.session_state.preview_cam_index = None
        st.session_state.preview_roi = None
        st.session_state.disabled_point_ids = []
        st.session_state.final_roasted_weight = 0.0
        st.session_state.final_total_roast_time = ""
        st.session_state.end_confirm_prev_running = False
        st.session_state.roast_id = None
        st.session_state.last_saved_roast_id = None
        st.session_state._clear_roast_notes_on_rerun = True
        st.rerun()

    # Apply set temp immediately when "Apply" button is pressed, even if not currently running. This allows the classifier to use the updated set temp for its logic right away.
    if apply_set:
        prev_set_temp = st.session_state.set_temp
        st.session_state.set_temp = new_set_temp
        st.session_state.set_temp_is_applied = new_set_temp > 0
        # Event marker
        if st.session_state.roast_active and st.session_state.start_epoch is not None:
            st.session_state.events.append(
                {
                    "t_sec": current_elapsed_sec(),
                    "type": "set_change",
                    "value": new_set_temp,
                    "from_value": prev_set_temp,
                }
            )

    # Start logging: initialize state, open camera, and add initial sample with set temp as current temp (x=0 anchor)
    if start_btn:
        requested_roi = (int(roi_x), int(roi_y), int(roi_w), int(roi_h))
        if st.session_state.preview_camera:
            st.session_state.preview_camera.roi = requested_roi

        st.session_state.roast_id = bean_title.strip() or None
        st.session_state.roast_active = True
        st.session_state.running = True
        st.session_state.end_confirm_pending = False
        st.session_state.start_epoch = time.time()
        st.session_state.last_sample_epoch = 0.0
        st.session_state.last_ocr_epoch = 0.0
        st.session_state.last_capture_epoch = 0.0
        st.session_state.live_frame = None
        st.session_state.live_roi = None
        st.session_state.capture_buffer = []
        st.session_state.last_ocr_frame = None
        st.session_state.last_ocr_roi = None
        st.session_state.pending_ocr = None
        st.session_state.samples = []
        st.session_state.events = []
        st.session_state.classifier = TempClassifier()
        st.session_state.set_temp = int(st.session_state.set_temp)
        st.session_state.set_temp_is_applied = True
        st.session_state.disabled_point_ids = []
        st.session_state.final_roasted_weight = 0.0
        st.session_state.final_total_roast_time = ""
        st.session_state.end_confirm_prev_running = False

        # Add initial point at x=0 with set temp (as you wanted)
        init_result = st.session_state.classifier.force_initial(set_temp=new_set_temp)
        add_sample(0.0, raw_read=new_set_temp, result=init_result)

        if (
            st.session_state.preview_camera is not None
            and int(st.session_state.preview_cam_index or -1) == int(cam_index)
        ):
            st.session_state.camera = st.session_state.preview_camera
        else:
            st.session_state.camera = Camera(index=int(cam_index), roi=requested_roi)

        st.session_state.preview_camera = None
        st.session_state.preview_cam_index = None
        st.session_state.preview_roi = None
        st.rerun()

    if pause_resume_btn:
        st.session_state.running = not st.session_state.running
        st.rerun()

    if end_roast_btn:
        st.session_state.end_confirm_prev_running = bool(st.session_state.running)
        st.session_state.running = False
        st.session_state.final_total_roast_time = _format_mmss(current_elapsed_sec())
        st.session_state.end_confirm_pending = True
        st.rerun()

    if st.session_state.end_confirm_pending:
        st.sidebar.error("Confirm end roast? This will autosave and close camera.")
        st.sidebar.caption(f"Total roast time (captured now): {st.session_state.final_total_roast_time}")
        st.session_state.final_roasted_weight = st.sidebar.number_input(
            "Roasted weight (g)",
            min_value=0.0,
            value=float(st.session_state.final_roasted_weight),
            step=0.1,
            key="final_roasted_weight_input",
        )
        raw_w = float(raw_weight)
        roasted_w = float(st.session_state.final_roasted_weight)
       