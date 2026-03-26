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
    if any(token in text for token in ["anaerobic", "c