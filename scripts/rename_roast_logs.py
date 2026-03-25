""" Remaming script to rename roast log folders to a naming standard and update metadata.
- It reads all roast records' titles, normalizes them for comparison, and groups them by
  bean title.
- It assigns batch numbers and target names based on the bean title and existing batches,
  following a naming strategy Title + batch number
  (e.g., "Ethiopia" for single batch, "Ethiopia #2" for the second batch).

  - If no bean title is present, it derives one from origin + process + variety.

"""


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "roasts"
META_FILENAME = "meta.json"


@dataclass
class RoastRecord:
    old_name: str
    old_path: Path
    meta_path: Path
    meta: dict
    bean_title: str
    saved_at: Optional[datetime]
    batch_number: int = 1
    total_for_bean: int = 1
    new_name: str = ""

# reads all roast records' titles
# make the title to lowe case and remove extra spaces for comparison
def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())

# removing and replacing special characters for safe file paths
def safe_title_for_path(title: str) -> str:
    clean = re.sub(r"\s+", " ", (title or "").strip())
    clean = re.sub(r"[\\/:*?\"<>|]", "-", clean)
    clean = clean.strip(" .")
    return clean or "Untitled Coffee"

# Parse the saved_at timestamp from the metadata, returning a datetime object or None if parsing fails.
# example: if saved_at is "2024-06-01T12:34:56", it will return a datetime object representing that timestamp. 
# If saved_at is missing or not in a valid format, it will return None.
def parse_saved_at(meta: dict) -> Optional[datetime]:
    raw = str(meta.get("saved_at", "") or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None

# Generate a target roast name based on the bean title, batch number, and total batches for that bean.
# If there is only one batch for the bean, it returns the cleaned title.(no #)
# If there are multiple batches, it appends the batch number to the title (e.g., "Ethiopia #2").
def bean_title_for_record(meta: dict, fallback_name: str) -> str:
    title = str(meta.get("bean_title", "") or "").strip()
    if title:
        return title

    # Fallback when bean_title is missing: origin process variety.
    origin = str(meta.get("origin", "") or "").strip() or "Unknown Origin"
    process = str(meta.get("process", "") or "").strip() or "Unknown Process"
    variety = str(meta.get("variety", "") or "").strip() or "Unknown Variety"
    derived = " ".join([origin, process, variety]).strip()

    # Last-resort fallback for malformed metadata.
    return derived or fallback_name

# Generate the target roast name based on the bean title and batch number, following the naming strategy:
# - If there is only one batch for the bean, return the cleaned title (e.g., "Ethiopia").
# - If there are multiple batches, return the cleaned title with the batch number appended (e.g., "Ethiopia #2").
def target_roast_name(bean_title: str, batch_number: int, total_for_bean: int) -> str:
    """
    Naming strategy (edit this function for future schema changes):
    - if only one batch for this bean: "<title>"
    - if multiple batches: "<title> #<batch_number>"
    """
    base = safe_title_for_path(bean_title)
    if total_for_bean <= 1:
        return base
    return f"{base} #{int(batch_number)}"

# Load roast records from the given root directory, returning a list of RoastRecord objects containing metadata and paths for each roast entry found in the directory.
def load_records(root: Path) -> list[RoastRecord]:
    records: list[RoastRecord] = []
    if not root.exists() or not root.is_dir():
        return records

    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        meta_path = p / META_FILENAME
        if not meta_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        records.append(
            RoastRecord(
                old_name=p.name,
                old_path=p,
                meta_path=meta_path,
                meta=meta,
                bean_title=bean_title_for_record(meta, p.name),
                saved_at=parse_saved_at(meta),
            )
        )

    return records


def assign_batch_and_names(records: list[RoastRecord]) -> None:
    groups: dict[str, list[RoastRecord]] = {}
    for r in records:
        key = normalize_title(r.bean_title)
        groups.setdefault(key, []).append(r)

    for _, group in groups.items():
        # Oldest -> newest so newest gets highest batch number.
        group.sort(key=lambda r: (r.saved_at or datetime.min, r.old_name))
        total = len(group)
        for idx, rec in enumerate(group, start=1):
            rec.batch_number = idx
            rec.total_for_bean = total
            rec.new_name = target_roast_name(rec.bean_title, idx, total)


def plan_renames(records: list[RoastRecord]) -> list[tuple[RoastRecord, Path]]:
    plans: list[tuple[RoastRecord, Path]] = []
    used_targets: set[str] = set()

    for rec in records:
        target = rec.old_path.parent / rec.new_name
        if rec.new_name in used_targets:
            raise RuntimeError(f"Naming collision detected for target: {rec.new_name}")
        used_targets.add(rec.new_name)
        plans.append((rec, target))

    return plans


def apply_renames(plans: list[tuple[RoastRecord, Path]], apply_changes: bool) -> None:
    # Two-phase rename prevents collisions when swapping names.
    rename_ops = [(rec, target) for rec, target in plans if rec.old_path != target]

    if not rename_ops and not apply_changes:
        return

    if not apply_changes:
        return

    temp_ops: list[tuple[Path, Path, Path]] = []
    for i, (rec, _target) in enumerate(rename_ops, start=1):
        temp = rec.old_path.parent / f".__rename_tmp__{i}__{rec.old_path.name}"
        rec.old_path.rename(temp)
        temp_ops.append((temp, rec.old_path, rec.old_path.parent / rec.new_name))

    for temp, _old, final in temp_ops:
        temp.rename(final)


def update_metadata(plans: list[tuple[RoastRecord, Path]], apply_changes: bool) -> None:
    for rec, target_path in plans:
        new_meta = dict(rec.meta)
        new_meta["roast_id"] = rec.new_name
        new_meta["bean_title"] = rec.bean_title
        new_meta["batch_number"] = int(rec.batch_number)

        if not apply_changes:
            continue

        meta_path = target_path / META_FILENAME
        meta_path.write_text(json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8")

# Summarize planned changes, printing out the renames and metadata updates that will be applied.
# If apply_changes is False, it only prints the planned changes without modifying any files. 
# If apply_changes is True, it executes the renames and metadata updates.
def summarize(plans: list[tuple[RoastRecord, Path]], apply_changes: bool) -> None:
    mode = "APPLY" if apply_changes else "DRY-RUN"
    print(f"[{mode}] Found {len(plans)} roast entries.")

    rename_count = 0
    for rec, target in plans:
        changed_name = rec.old_path != target
        changed_batch = int(rec.meta.get("batch_number", 0) or 0) != rec.batch_number
        changed_meta_id = str(rec.meta.get("roast_id", "") or "") != rec.new_name

        if changed_name:
            rename_count += 1
            print(f"- RENAME: '{rec.old_name}' -> '{target.name}'")
        elif changed_batch or changed_meta_id:
            print(f"- META UPDATE: '{rec.old_name}' (batch={rec.batch_number}, roast_id='{rec.new_name}')")

    print(f"[{mode}] Planned folder renames: {rename_count}")
    if not apply_changes:
        print("[DRY-RUN] No files were modified. Re-run with --apply to execute.")


def main() -> None:
    # Argument parsing for root directory and apply flag. By default, it runs in dry-run mode, only printing planned changes without modifying any files. 
    parser = argparse.ArgumentParser(description="Rename roast log folders to a naming standard and update metadata.")
    # The --root argument allows specifying the root directory containing roast logs (default: data/roasts).
    parser.add_argument("--root", default=str(DATA_ROOT), help="Roasts root directory (default: data/roasts)")
    # The --apply flag allows executing the planned changes. Without this flag, the script runs in dry-run mode.
    parser.add_argument("--apply", action="store_true", help="Apply changes. Without this flag, runs as dry-run.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    print(f"Scanning roast root: {root}")
    records = load_records(root)
    # if records is empty
    if not records:
        print("No roast entries found.")
        return

    # Assign batch numbers and target names based on bean titles and existing batches
    # following the naming strategy defined in target_roast_name.
    assign_batch_and_names(records)
    plans = plan_renames(records)

    # Summarize planned changes. If --apply is not set, it only prints the planned renames and metadata updates without modifying any files. If --apply is set, it executes the renames and metadata updates.
    summarize(plans, apply_changes=args.apply)
    if not args.apply:
        return

    # renames the folders according to the planned changes
    # using a two-phase rename to avoid collisions. 
    # It first renames all folders to temporary names,
    #  then renames them to their final target names.
    apply_renames(plans, apply_changes=True)
    update_metadata(plans, apply_changes=True)
    print("[APPLY] Rename and metadata update complete.")


if __name__ == "__main__":
    main()
