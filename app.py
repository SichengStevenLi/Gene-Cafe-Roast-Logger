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
from recommend import recommend_roasts


APP_TITLE = "Gene Café Roast Logger"
# customizable sampling interval (seconds)
SAMPLE_INTERVAL_SEC = 5.0
# Run OCR at a non-1s cadence so we don't phase-lock with a 1s display toggle.
OCR_READ_INTERVAL_SEC = 0.7
# Small preview tick for smoother camera updates without pegging CPU.
PREVIEW_REFRESH_SEC = 0.05
PLOT_WINDOW_SEC = 15 * 60

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
    "Semi-Washed",
    "Cofermentation",
    "Experimental",
    "Other",
]

# Note: for simplicity, all state is in session_state. 
# In a more complex app, we might want to split into multiple pages or use a more robust state management approach.
def init_state():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "start_epoch" not in st.session_state:
        st.session_state.start_epoch = None
    if "last_sample_epoch" not in st.session_state:
        st.session_state.last_sample_epoch = 0.0
    if "last_ocr_epoch" not in st.session_state:
        st.session_state.last_ocr_epoch = 0.0
    if "pending_ocr" not in st.session_state:
        # Holds the most recent confirmed CURRENT_VIEW read within the current 5s window
        st.session_state.pending_ocr = None  # dict with keys raw_read, result
    if "samples" not in st.session_state:
        st.session_state.samples = []  # list of dict rows
    if "events" not in st.session_state:
        st.session_state.events = []   # set temp change events
    if "set_temp" not in st.session_state:
        st.session_state.set_temp = None
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
    if "preview_camera" not in st.session_state:
        st.session_state.preview_camera = None
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


def _probe_camera_index(idx: int) -> bool:
    # On macOS, some USB cameras only respond on specific backends or after a short warmup.
    backends = [cv2.CAP_ANY]
    avf = getattr(cv2, "CAP_AVFOUNDATION", None)
    if avf is not None:
        backends = [avf, cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        ok = False
        frame = None
        # Warmup reads improve detection for some USB capture devices.
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.03)
        cap.release()

        if ok and frame is not None:
            return True

    return False


def _detect_camera_indices(max_index: int = 6) -> list[int]:
    found: list[int] = []
    for idx in range(max(1, int(max_index))):
        if _probe_camera_index(idx):
            found.append(idx)
    return found


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
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    ocr_ready, ocr_message = get_ocr_status()
    if not ocr_ready:
        st.warning(f"{ocr_message} Temperature reads will be unavailable until OCR is configured.")

    # ---- Sidebar: Metadata + Controls ----
    st.sidebar.header("Roast Setup")

    # Bean metadata inputs (before roast)
    roast_profile_ids = list_roasts()
    roast_profile_labels: dict[str, str] = {}
    for rid in roast_profile_ids:
        try:
            meta = load_roast_meta(rid)
            roast_profile_labels[rid] = str(meta.get("bean_title", "") or meta.get("title", "") or rid)
        except Exception:
            roast_profile_labels[rid] = rid

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

        # Apply initial set temp from selected roast profile metadata.
        profile_set_temp = int(p.get("preheat_temp", 0) or 0)
        st.session_state.current_set_temp_input = profile_set_temp
        st.session_state.set_temp = profile_set_temp if profile_set_temp > 0 else None
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
        st.session_state.detected_cameras = _detect_camera_indices(max_index=16)
        st.session_state.camera_scan_done = True

    detect_cols = st.sidebar.columns(2)
    detect_cols[0].caption("Camera")
    if detect_cols[1].button("Detect", disabled=st.session_state.roast_active):
        st.session_state.detected_cameras = _detect_camera_indices(max_index=16)
    st.sidebar.caption("Detect scans camera indexes 0-15 using macOS-friendly probing.")

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
            isinstance(st.session_state.set_temp, int) and st.session_state.set_temp > 0,
            "Set and apply current set temp before starting",
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

    roast_choices = ["(none)"] + roasts
    ref_default_index = roast_choices.index(st.session_state.reference_roast_id) if st.session_state.reference_roast_id in roast_choices else 0
    ref_choice = st.sidebar.selectbox(
        "Select reference roast",
        roast_choices,
        index=ref_default_index,
        format_func=lambda rid: "(none)" if rid == "(none)" else _roast_label(rid),
    )

    # If a reference roast is selected, store it in session state to overlay in the plot
    if ref_choice != "(none)":
        st.session_state.reference_roast_id = ref_choice
    else:
        st.session_state.reference_roast_id = None

    # Detect exact same bean by title and offer quick-reference action.
    bean_title_clean = bean_title.strip()
    if bean_title_clean:
        same_bean_versions = list_roasts_for_bean(bean_title_clean)
        if same_bean_versions:
            same_bean_ids = [rid for rid, _ in same_bean_versions]
            default_same_idx = 0
            if st.session_state.reference_roast_id in same_bean_ids:
                default_same_idx = same_bean_ids.index(st.session_state.reference_roast_id)

            st.sidebar.info(
                f"Detected the same bean in your history: {bean_title_clean}. "
                "Do you want to use one of its previous roast versions as reference?"
            )

            selected_same_bean_version = st.sidebar.selectbox(
                "Same-bean roast versions (newest to oldest)",
                same_bean_ids,
                index=default_same_idx,
                format_func=lambda rid: _roast_label(rid),
                key="same_bean_versions_select",
            )

            if st.sidebar.button("Use selected same-bean version", key="use_same_bean_reference"):
                st.session_state.reference_roast_id = selected_same_bean_version
                st.session_state.suggested_roast_choice = selected_same_bean_version
                st.rerun()

    # Suggest reference roasts if not in log (simple recommender)
    if st.sidebar.button("Suggest similar roasts"):
        suggestions = recommend_roasts(
            origin=origin,
            altitude=int(altitude),
            process=process,
            appearance=appearance,
            raw_weight_g=float(raw_weight),
            is_decaf=bool(is_decaf),
            bean_category=bean_type_value if bean_type_value != "(select)" else "",
            variety=variety_value if variety_value != "(select)" else "",
            limit=5,
        )
        if suggestions:
            st.session_state.suggested_roasts = suggestions
            current_options = [rid for rid, _ in suggestions]
            if st.session_state.suggested_roast_choice not in current_options:
                st.session_state.suggested_roast_choice = current_options[0]
            st.session_state.reference_roast_id = st.session_state.suggested_roast_choice
        else:
            st.session_state.suggested_roasts = []
            st.session_state.suggested_roast_choice = None
            st.sidebar.info("No saved roasts yet to compare.")

    if st.session_state.suggested_roasts:
        st.sidebar.caption("Suggested compatible curves")
        suggestion_scores = {rid: score for rid, score in st.session_state.suggested_roasts}
        suggestion_options = [rid for rid, _ in st.session_state.suggested_roasts]

        if st.session_state.suggested_roast_choice not in suggestion_options:
            st.session_state.suggested_roast_choice = suggestion_options[0]

        selected_suggestion = st.sidebar.radio(
            "Suggested options",
            options=suggestion_options,
            index=suggestion_options.index(st.session_state.suggested_roast_choice),
            format_func=lambda rid: f"{_roast_label(rid)} (score {suggestion_scores[rid]:.1f})",
        )
        st.session_state.suggested_roast_choice = selected_suggestion
        st.session_state.reference_roast_id = selected_suggestion


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
    apply_set = st.sidebar.button("Apply set temp(F)")

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
        st.session_state.pending_ocr = None
        st.session_state.samples = []
        st.session_state.events = []
        st.session_state.set_temp = None
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
        st.session_state.roast_notes_input = ""
        st.rerun()

    # Apply set temp immediately when "Apply" button is pressed, even if not currently running. This allows the classifier to use the updated set temp for its logic right away.
    if apply_set:
        prev_set_temp = st.session_state.set_temp
        st.session_state.set_temp = new_set_temp
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
        if st.session_state.preview_camera:
            try:
                st.session_state.preview_camera.close()
            except Exception:
                pass
            st.session_state.preview_camera = None
            st.session_state.preview_cam_index = None
            st.session_state.preview_roi = None

        st.session_state.roast_id = bean_title.strip() or None
        st.session_state.roast_active = True
        st.session_state.running = True
        st.session_state.end_confirm_pending = False
        st.session_state.start_epoch = time.time()
        st.session_state.last_sample_epoch = 0.0
        st.session_state.last_ocr_epoch = 0.0
        st.session_state.pending_ocr = None
        st.session_state.samples = []
        st.session_state.events = []
        st.session_state.classifier = TempClassifier()
        st.session_state.set_temp = new_set_temp  # initial set temp at start
        st.session_state.disabled_point_ids = []
        st.session_state.final_roasted_weight = 0.0
        st.session_state.final_total_roast_time = ""
        st.session_state.end_confirm_prev_running = False

        # Add initial point at x=0 with set temp (as you wanted)
        init_result = st.session_state.classifier.force_initial(set_temp=new_set_temp)
        add_sample(0.0, raw_read=new_set_temp, result=init_result)

        st.session_state.camera = Camera(index=int(cam_index), roi=(roi_x, roi_y, roi_w, roi_h))
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
        if raw_w > 0 and roasted_w > 0:
            loss_g = raw_w - roasted_w
            loss_pct = (loss_g / raw_w) * 100.0
            if loss_g >= 0:
                st.sidebar.info(f"Weight loss: {loss_g:.1f} g ({loss_pct:.1f}%)")
            else:
                st.sidebar.warning(
                    f"Roasted weight is {abs(loss_g):.1f} g above raw ({abs(loss_pct):.1f}% gain). Check entries."
                )
        st.sidebar.caption("Roasted weight can be recorded later; 0 means not recorded yet.")

        end_col_a, end_col_b = st.sidebar.columns(2)
        confirm_end_yes = end_col_a.button("Yes, end", type="primary")
        confirm_end_no = end_col_b.button("No, continue")

        if confirm_end_no:
            st.session_state.end_confirm_pending = False
            st.session_state.running = bool(st.session_state.end_confirm_prev_running and st.session_state.roast_active)
            st.session_state.end_confirm_prev_running = False
            st.rerun()

        if confirm_end_yes:
            final_errors = []
            if not st.session_state.final_total_roast_time:
                final_errors.append("Total roast time could not be captured")

            if final_errors:
                st.sidebar.error("Cannot end roast yet:")
                for e in final_errors:
                    st.sidebar.write(f"- {e}")
            else:
                clean_bean_title = bean_title.strip()
                if not clean_bean_title:
                    clean_bean_title = "Untitled Coffee"
                batch_number = next_batch_number(clean_bean_title)
                roast_log_name = make_roast_log_name(clean_bean_title, batch_number)

                meta = RoastMeta(
                    roast_id=roast_log_name,
                    bean_title=clean_bean_title,
                    origin=origin,
                    bean_type=bean_type,
                    altitude_m=int(altitude),
                    process=process,
                    raw_weight_g=float(raw_weight),
                    roasted_weight_g=float(st.session_state.final_roasted_weight),
                    total_roast_time=st.session_state.final_total_roast_time,
                    preheat_temp=int(st.session_state.set_temp or 0),
                    is_decaf=bool(is_decaf),
                    bean_category=bean_type_value if bean_type_value != "(select)" else "",
                    variety=variety_value if variety_value != "(select)" else "",
                    bean_appearance=appearance if appearance != "(select)" else "",
                    batch_number=batch_number,
                )

                end_df = pd.DataFrame(st.session_state.samples)
                disabled_ids = set(int(x) for x in st.session_state.disabled_point_ids)
                if not end_df.empty and disabled_ids:
                    end_df = end_df.loc[~end_df.index.isin(disabled_ids)].reset_index(drop=True)
                save_roast_session(
                    meta=meta,
                    curve_df=end_df,
                    events=st.session_state.events,
                    notes=st.session_state.roast_notes_input,
                )
                st.session_state.last_saved_roast_id = meta.roast_id

                st.session_state.running = False
                st.session_state.roast_active = False
                st.session_state.end_confirm_pending = False
                st.session_state.end_confirm_prev_running = False
                if st.session_state.camera:
                    st.session_state.camera.close()
                    st.session_state.camera = None
                if st.session_state.preview_camera:
                    st.session_state.preview_camera.close()
                    st.session_state.preview_camera = None
                    st.session_state.preview_cam_index = None
                    st.session_state.preview_roi = None
                if raw_w > 0 and roasted_w > 0:
                    loss_g = raw_w - roasted_w
                    loss_pct = (loss_g / raw_w) * 100.0
                    st.success(
                        f"Saved roast: {meta.roast_id} | Weight loss: {loss_g:.1f} g ({loss_pct:.1f}%)"
                    )
                else:
                    st.success(f"Saved roast: {meta.roast_id}")
                st.rerun()

    if (not st.session_state.roast_active) and st.session_state.last_saved_roast_id:
        st.sidebar.divider()
        st.sidebar.subheader("Post-roast weight")
        try:
            latest_meta = load_roast_meta(st.session_state.last_saved_roast_id)
            raw_saved = float(latest_meta.get("raw_weight_g", 0.0) or 0.0)
            roasted_saved = float(latest_meta.get("roasted_weight_g", 0.0) or 0.0)

            post_weight = st.sidebar.number_input(
                "Enter roasted weight after cooling (g)",
                min_value=0.0,
                value=roasted_saved,
                step=0.1,
                key="post_roasted_weight_input",
            )

            if raw_saved > 0 and post_weight > 0:
                loss_g = raw_saved - post_weight
                loss_pct = (loss_g / raw_saved) * 100.0
                if loss_g >= 0:
                    st.sidebar.info(f"Weight loss: {loss_g:.1f} g ({loss_pct:.1f}%)")
                else:
                    st.sidebar.warning(
                        f"Roasted weight is {abs(loss_g):.1f} g above raw ({abs(loss_pct):.1f}% gain). Check entries."
                    )

            if st.sidebar.button("Save post-roast weight", key="save_post_roast_weight"):
                update_roasted_weight(st.session_state.last_saved_roast_id, float(post_weight))
                st.sidebar.success("Post-roast weight saved")
                st.rerun()
        except Exception as e:
            st.sidebar.warning(f"Could not load latest roast meta for weight update: {e}")

    # ---- Main layout ----
    left, right = st.columns([3.2, 1.0])

    #-- Live plot on the left, camera feed and debug info on the right --
    with left:
        st.subheader("Live plot")
        plotter = RoastPlotter(xmax_sec=PLOT_WINDOW_SEC)

        df = pd.DataFrame(st.session_state.samples) if st.session_state.samples else pd.DataFrame(
            columns=["t_sec", "temp_current", "set_temp", "view_mode", "raw_read", "confidence"]
        )

        disabled_ids = set(int(x) for x in st.session_state.disabled_point_ids)
        if not df.empty:
            df = df.copy()
            df["point_id"] = df.index.astype(int)
            plot_df = df.loc[~df["point_id"].isin(disabled_ids)].drop(columns=["point_id"])
        else:
            plot_df = df

        # Reference curve underlay — only shown while roast is active (acts as a guide)
        ref_df = None
        if st.session_state.roast_active and st.session_state.reference_roast_id:
            try:
                ref_df = load_roast_curve(st.session_state.reference_roast_id)
            except Exception as e:
                st.warning(f"Could not load reference curve: {e}")

        fig = plotter.make_figure(df=plot_df, events=st.session_state.events, ref_df=ref_df)
        live_plot_slot = st.empty()
        live_plot_slot.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
            key="live_plot_chart",
        )

        if st.session_state.roast_active:
            phase_elapsed = current_elapsed_sec()
        elif not df.empty and "t_sec" in df.columns:
            phase_elapsed = float(df["t_sec"].max())
        else:
            phase_elapsed = 0.0

        phase_stats = _compute_phase_stats(st.session_state.events, phase_elapsed)
        p1, p2, p3, p4 = st.columns(4)
        for column, (label, (pct, secs)) in zip(
            [p1, p2, p3, p4],
            phase_stats.items(),
        ):
            column.metric(label, f"{pct:.1f}%", _format_mmss(secs))

        # Stage markers: unlock sequentially and allow one click each.
        yellow_evt = next((e for e in st.session_state.events if e.get("type") == "yellowing_start"), None)
        browning_evt = next((e for e in st.session_state.events if e.get("type") == "browning_start"), None)
        crack_evt = next((e for e in st.session_state.events if e.get("type") == "first_crack"), None)

        st.caption("Roast stages")
        s1, s2, s3 = st.columns(3)

        yellow_btn = s1.button(
            "Yellowing starts",
            disabled=(not st.session_state.roast_active) or (not st.session_state.running) or (yellow_evt is not None),
            key="btn_yellowing_start",
        )
        browning_btn = s2.button(
            "Browning starts",
            disabled=(
                (not st.session_state.roast_active)
                or (not st.session_state.running)
                or (yellow_evt is None)
                or (browning_evt is not None)
            ),
            key="btn_browning_start",
        )
        crack_btn = s3.button(
            "1st crack",
            disabled=(
                (not st.session_state.roast_active)
                or (not st.session_state.running)
                or (browning_evt is None)
                or (crack_evt is not None)
            ),
            key="btn_first_crack",
        )

        if yellow_btn and yellow_evt is None:
            st.session_state.events.append({"t_sec": current_elapsed_sec(), "type":