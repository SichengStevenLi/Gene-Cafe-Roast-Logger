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
    profiles[c