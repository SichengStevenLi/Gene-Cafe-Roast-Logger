from __future__ import annotations
import os
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd


DATA_DIR = os.path.join("data", "roasts")


@dataclass
class RoastMeta:
    roast_id: str
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