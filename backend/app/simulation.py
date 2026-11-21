"""Parser for the X-Simulate-Failure test header.

Format: comma-separated ``<key_id>=<status>`` pairs, ``*`` meaning every key.
Allowed statuses: 429, 500, 502, 503. Example: ``key_1=429,key_2=500``.
"""

from __future__ import annotations

ALLOWED_STATUSES = {429, 500, 502, 503}


class SimulationHeaderError(ValueError):
    pass


def parse_simulation_header(value: str | None) -> dict[str, int]:
    """Return a mapping key_id -> status. ``*`` is kept as a wildcard entry."""
    if value is None or not value.strip():
        return {}
    plan: dict[str, int] = {}
    for raw in value.split(","):
        part = raw.strip()
        if not part:
            continue
        if "=" not in part:
            raise SimulationHeaderError(f"invalid X-Simulate-Failure entry: {part!r}")
        key_id, _, status_text = part.partition("=")
        key_id, status_text = key_id.strip(), status_text.strip()
        if not key_id:
            raise SimulationHeaderError(f"invalid X-Simulate-Failure entry: {part!r}")
        try:
            status = int(status_text)
        except ValueError as exc:
            raise SimulationHeaderError(f"invalid status in X-Simulate-Failure: {status_text!r}") from exc
        if status not in ALLOWED_STATUSES:
            raise SimulationHeaderError(
                f"unsupported status {status} in X-Simulate-Failure (allowed: 429, 500, 502, 503)"
            )
        plan[key_id] = status
    return plan


def simulated_status_for(plan: dict[str, int], key_id: str) -> int | None:
    if key_id in plan:
        return plan[key_id]
    return plan.get("*")
