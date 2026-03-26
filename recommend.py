from __future__ import annotations
from storage import list_roasts, load_roast_meta


# Parse a process label into a comparable profile.
def _process_profile(label: str) -> tuple[str | None, str | None]:
    text = (label or "").strip().lower()
    if not text:
        return None, None

    modifier = None
    if "anaerobic" in text:
        modifier = "anaerobic"
    elif "carbonic maceration" in text:
        modifier = "carbonic maceration"
    elif "coferment" in text or "cofermentation" in text:
        modifier = "cofermentation"
    elif "experimental" in text:
        modifier = "experimental"

    # Specific families first to avoid semi-washed collapsing into washed.
    if text.endswith("honey") or text.endswith("semi-washed"):
        return "honey", modifier
    if "wet hulled" in text:
        return "wet hulled", modifier
    if text.endswith("washed") and not text.endswith("semi-washed"):
        return "washed", modifier
    if text.endswith("natural"):
        return "natural", modifier

    # If the label is experimental/modifier-heavy, infer hidden base process if present.
    if any(token in text for token in ["anaerobic", "coferment", "cofermentation", "experimental", "carbonic maceration", "other"]):
        if "honey" in text or "semi-washed" in text:
            return "honey", modifier
        if "wet hulled" in text:
            return "wet hulled", modifier
        if "washed" in text and "semi-washed" not in text:
            return "washed", modifier
        if "natural" in text:
            return "natural", modifier

    return None, modifier


def _normalize_process_family(label: str) -> str | None:
    family, _modifier = _process_profile(label)
    return family


def recommend_roasts(
    origin: str,
    altitude: int,
    process: str,
    appearance: str,
    raw_weight_g: float,
    is_decaf: bool,
    bean_category: str,
    variety: str,
    limit: int = 5,
    bean_type: str = "",
    **_unused,
):
    """
    Default base weights (out of 100):
    - Processing: 40
    - Altitude: 25
    - Bean appearance/size: 20
    - Variety: 10
    - Origin: 5

    Additional priority boosts:
    - Gesha-first behavior (conditional)
    - Decaf matching
    """
    roast_entries = [(rid, load_roast_meta(rid)) for rid in list_roasts()]
    results = []

    current_process_label = (process or "").strip().lower()
    current_process_family, current_process_modifier = _process_profile(process)
    available_process_labels = {
        str(meta.get("process", "")).strip().lower()
        for _, meta in roast_entries
        if str(meta.get("process", "")).strip()
    }

    available_process_families = {
        family
        for _, meta in roast_entries
        for family in [_normalize_process_family(str(meta.get("process", "")))]
        if family is not None
    }

    available_modifier_base_pairs = {
        _process_profile(str(meta.get("process", "")))
        for _, meta in roast_entries
    }

    has_exact_process_history = bool(current_process_label) and current_process_label in available_process_labels

    # Honey can fall back to Washed when there are no Honey references.
    effective_process_family = current_process_family
    effective_process_modifier = current_process_modifier

    if (not has_exact_process_history) and current_process_family == "honey" and "honey" not in available_process_families and "washed" in available_process_families:
        effective_process_family = "washed"

    # Bare anaerobic falls back to other anaerobic-labelled roasts first.
    if (not has_exact_process_history) and current_process_modifier == "anaerobic" and current_process_family is None:
        effective_process_modifier = "anaerobic"

    # If an anaerobic+base combo has no history, fall back to its base family.
    if (
        not has_exact_process_history
        and current_process_modifier == "anaerobic"
        and current_process_family is not None
        and (current_process_family, current_process_modifier) not in available_modifier_base_pairs
    ):
        effective_process_modifier = None

    current_variety = (variety or "").strip().lower()
    current_is_gesha = current_variety in {"gesha"}

    # Hard pre-filter: only consider roasts with matching decaf status.
    roast_entries = [
        (rid, meta) for rid, meta in roast_entries
        if bool(meta.get("is_decaf", False)) == bool(is_decaf)
    ]

    for rid, meta in roast_entries:
        score = 0.0

        # Variety from newer schema; fallback to legacy bean_type split.
        meta_variety = str(meta.get("variety", "")).strip().lower()
        if not meta_variety:
            legacy = str(meta.get("bean_type", "")).strip()
            if "/" in legacy:
                meta_variety = legacy.split("/", 1)[1].strip().lower()

        # If current roast is Gesha, prioritize historical Gesha roasts first.
        if current_is_gesha:
            if meta_variety in {"gesha"}:
                score += 420
            else:
                score -= 200

        meta_process_label = str(meta.get("process", "")).strip().lower()
        meta_process_family, meta_process_modifier = _process_profile(str(meta.get("process", "")))

        # If this exact entered process exists in history, prefer exact label matches.
        if has_exact_process_history:
            if current_process_label and meta_process_label == current_process_label:
                score += 40
        else:
            # Otherwise use smart fallback:
            # 1) modifier+base when available
            # 2) bare modifier-only match for labels like 'Anaerobic'
            # 3) base family fallback
            if (
                effective_process_modifier is not None
                and effective_process_family is not None
                and meta_process_modifier == effective_process_modifier
                and meta_process_family == effective_process_family
            ):
                score += 40
            elif (
                effective_process_modifier is not None
                and effective_process_family is None
                and meta_process_modifier == effective_process_modifier
            ):
                score += 40
            elif effective_process_family and meta_process_family == effective_process_family:
                score += 40

        # Altitude closeness contributes up to 25 points via piecewise tiers.
        # Gaps above 1200m earn nothing; each 200m band reduces the score.
        #   0–200m  → 25 pts   (essentially same density)
        #   200–400m → 20 pts
        #   400–600m → 14 pts
        #   600–800m → 8 pts
        #   800–1000m → 3 pts  (noticeable roast difference)
        #   1000–1200m → 1 pt  (borderline)
        #   >1200m  →  0 pts
        try:
            alt2 = int(meta.get("altitude_m", 0))
            if altitude > 0 and alt2 > 0:
                _gap = abs(alt2 - altitude)
                if _gap < 200:
                    score += 25.0
                elif _gap < 400:
                    score += 20.0
                elif _gap < 600:
                    score += 14.0
                elif _gap < 800:
                    score += 8.0
                elif _gap < 1000:
                    score += 3.0
                elif _gap < 1200:
                    score += 1.0
                # else: 0 — too different to be useful
        except Exception:
            pass

        # Bean appearance / size
        meta_appearance = str(meta.get("bean_appearance", "")).strip().lower()
        if appearance and meta_appearance == appearance.strip().lower():
            score += 20

        # Variety
        if current_variety and meta_variety == current_variety:
            score += 10

        # Origin
        if origin and meta.get("origin", "").strip().lower() == origin.strip().lower():
            score += 5

        # Weight closeness ratio is a tie-break signal.
        try:
            rw2 = float(meta.get("raw_weight_g", 0.0))
            if raw_weight_g > 0 and rw2 > 0:
                ratio_closeness = max(0.0, 1.0 - abs(rw2 - raw_weight_g) / raw_weight_g)
                score += ratio_closeness * 8.0
        except Exception:
            pass

        # Bean type last priority.
        meta_type = str(meta.get("bean_category", "")).strip()
        if not meta_type:
            # Backward compatibility with older saved roasts.
            legacy = str(meta.get("bean_type", "")).strip()
            meta_type = legacy.split("/")[0].strip() if legacy else ""

        if bean_category and meta_type.lower() == bean_category.strip().lower():
            score += 4

        if score > 0:
            results.append((rid, score))

    results.sort(key=lambda x: x[1], reverse=True)

    # Convert raw scores to 0-100 % similarity against the theoretical maximum.
    # Decaf is a hard pre-filter (not scored), so the max reflects roast compatibility only.
    # Max breakdown:
    #   Process          40
    #   Altitude         25   (continuous, 0-25)
    #   Appearance       20
    #   Variety          10
    #   Origin           5
    #   Weight           8    (continuous, 0-8)
    #   Bean category    4
    #   Gesha bonus      420  (only when current bean is Gesha)
    _THEORETICAL_MAX = 40 + 25 + 20 + 10 + 5 + 8 + 4 + (420 if current_is_gesha else 0)

    pct_results = [
        (rid, round(max(0.0, min(100.0, score / _THEORETICAL_MAX * 100)), 1))
        for rid, score in results[:limit]
    ]
    return pct_results