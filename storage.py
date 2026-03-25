from __future__ import annotations
import os
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd


DATA_DIR = os.path.join("data", "roasts")
CAMERA_CONFIG_PATH = os.path.join("data", "camera_config.json")
BEAN_PROFILES_PATH = os.path.join("data", "bean_profiles.json")

_CAMERA_CONFIG_DEFAULTS = {
    "cam_index": 0,
    "roi_x": 0,
    "roi_y": 0,
    "roi_w": 320,
    "roi_h": 180,
}


def load_camera_config() -> dict:
    """Load persisted camera/ROI settings, or return defaults."""
    if os.path.exists(CAMERA_CONFIG_PATH):
        try:
            with open(CAMERA_CONFIG_PATH, "r") as f:
                data = json.load(f)
            # Fill in any missing keys with defaults
            return {**_CAMERA_CONFIG_DEFAULTS, **data}
        except Exception:
            pass
    return dict(_CAMERA_CONFIG_DEFAULTS)


def save_camera_config(cam_index: int, roi_x: int, roi_y: int, roi_w: int, roi_h: int) -> None:
    """Persist camera/ROI settings to disk."""
    os.makedirs("data", exist_ok=True)
    with open(CAMERA_CONFIG_PATH, "w") as f:
        json.dump(
            {"cam_index": cam_index, "roi_x": roi_x, "roi_y": roi_y, "roi_w": roi_w, "roi_h": roi_h},
            f,
            indent=2,
        )


def load_bean_profiles() -> dict[str, dict]:
    """Load saved bean profiles keyed by title."""
    if os.path.exists(BEAN_PROFILES_PATH):
        try:
            with open(BEAN_PROFILES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def list_bean_profile_titles() -> list[str]:
    return sorted(load_bean_profiles().keys(), key=lambda s: s.lower())


def get_bean_profile(title: str) -> dict | None:
    profiles = load_bean_profiles()
    clean_title = (title or "").strip()
    if not clean_title:
        return None
    return profiles.get(clean_title)


def save_bean_profile(title: str, profile: dict) -> None:
    """Create or update a named bean profile."""
    clean_title = (title or "").strip()
    if not clean_title:
        raise ValueError("Bean title is required")

    profiles = load_bean_profiles()
    profiles[clean_title] = {
        "bean_title": clean_title,
        "origin": str(profile.get("origin", "") or ""),
        "bean_category": str(profile.get("bean_category", "") or ""),
        "variety": str(profile.get("variety", "") or ""),
        "bean_appearance": str(profile.get("bean_appearance", "") or ""),
        "altitude_m": int(profile.get("altitude_m", 0) or 0),
        "process": str(profile.get("process", "") or ""),
        "is_decaf": bool(profile.get("is_decaf", False)),
        "raw_weight_g": float(profile.get("raw_weight_g", 0.0) or 0.0),
        "updated_at": datetime.now().isoformat(),
    }

    os.makedirs("data", exist_ok=True)
    with open(BEAN_PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


@dataclass
class RoastMeta:
    roast_id: str
    bean_title: str
    origin: str
    bean_type: str
    altitude_m: int
    process: str
    raw_weight_g: float
    roasted_weight_g: float
    total_roast_time: str
    preheat_temp: int
    is_decaf: bool = False
    bean_category: str = ""
    variety: str = ""
    bean_appearance: str = ""
    batch_number: int = 1


def ensure_data_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unknown"


def make_roast_id(origin: str, process: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{ts}_{slugify(origin)}_{slugify(process)}"


def _normalize_bean_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def _safe_bean_title_for_path(title: str) -> str:
    clean = re.sub(r"\s+", " ", (title or "").strip())
    clean = re.sub(r"[\\/:*?\"<>|]", "-", clean)
    clean = clean.strip(" .")
    return clean or "Untitled Coffee"


def make_roast_log_name(bean_title: str, batch_number: int) -> str:
    base = _safe_bean_title_for_path(bean_title)
    return base if int(batch_number) <= 1 else f"{base} #{int(batch_number)}"


def _extract_batch_number(roast_id: str) -> int | None:
    m = re.search(r"#(\d+)\s*$", str(roast_id or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def list_roasts_for_bean(bean_title: str) -> list[tuple[str, dict]]:
    """Return roasts for an exact bean title (case-insensitive), newest batch first."""
    target = _normalize_bean_title(bean_title)
    if not target:
        return []

    matches: list[tuple[str, dict]] = []
    for rid in list_roasts():
        try:
            meta = load_roast_meta(rid)
        except Exception:
            continue

        meta_title = _normalize_bean_title(str(meta.get("bean_title", "") or ""))
        if meta_title != target:
            continue

        batch_no = int(meta.get("batch_number", 0) or 0)
        if batch_no <= 0:
            inferred = _extract_batch_number(rid)
            batch_no = inferred if inferred is not None else 1
        meta["batch_number"] = batch_no
        matches.append((rid, meta))

    matches.sort(key=lambda item: int(item[1].get("batch_number", 1) or 1), reverse=True)
    return matches


def next_batch_number(bean_title: str) -> int:
    roasts = list_roasts_for_bean(bean_title)
    if not roasts:
        return 1
    highest = max(int(meta.get("batch_number", 1) or 1) for _, meta in roasts)
    return highest + 1


def roast_path(roast_id: str) -> str:
    return os.path.join(DATA_DIR, roast_id)


def save_roast_session(meta: RoastMeta, curve_df: pd.DataFrame, events: list[dict]):
    rp = roast_path(meta.roast_id)
    os.makedirs(rp, exist_ok=True)

    # compute weight loss
    weight_loss_pct = None
    if meta.raw_weight_g > 0 and meta.roasted_weight_g > 0:
        weight_loss_pct = 100.0 * (meta.raw_weight_g - meta.roasted_weight_g) / meta.raw_weight_g

    meta_dict = asdict(meta)
    meta_dict["weight_loss_pct"] = weight_loss_pct
    meta_dict["saved_at"] = datetime.now().isoformat()
    meta_dict["events"] = events or []

    curve_df.to_csv(os.path.join(rp, "curve.csv"), index=False)
    with open(os.path.join(rp, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)


def list_roasts() -> list[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))], reverse=True)


def load_roast_curve(roast_id: str) -> pd.DataFrame:
    path = os.path.join(roast_path(roast_id), "curve.csv")
    return pd.read_csv(path)


def load_roast_meta(roast_id: str) -> dict:
    path = os.path.join(roast_path(roast_id), "meta.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_roasted_weight(roast_id: str, roasted_weight_g: float) -> dict:
    path = os.path.join(roast_path(roast_id), "meta.json")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["roasted_weight_g"] = float(roasted_weight_g)
    raw_weight = float(meta.get("raw_weight_g", 0.0) or 0.0)
    if raw_weight > 0 and float(roasted_weight_g) > 0:
        meta["weight_loss_pct"] = 100.0 * (raw_weight - float(roasted_weight_g)) / raw_weight
    else:
        meta["weight_loss_pct"] = None

    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta