"""Single source of truth for the ACT-based KeyCollect runtime contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config" / "system_contract.yaml"


class ContractError(ValueError):
    """Raised when a runtime artifact violates the system contract."""


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ContractError(f"Unsupported or malformed contract: {path}")
    return value


CONTRACT = load_contract()


def control_period_s(fps: float) -> float:
    if fps <= 0:
        raise ContractError("control frequency must be positive")
    return 1.0 / float(fps)


def per_cycle_limits(fps: float) -> dict[str, float]:
    """Convert the contract's physical rates into per-control-cycle limits."""
    period = control_period_s(fps)
    rates = CONTRACT["motion_limits"]
    return {
        "translation_m": float(rates["translation_m_s"]) * period,
        "rotation_rad": float(rates["rotation_rad_s"]) * period,
        "finger_rad": float(rates["finger_rad_s"]) * period,
    }


def state_names() -> list[str]:
    return list(CONTRACT["act"]["state_names"])


def action_names() -> list[str]:
    return list(CONTRACT["act"]["action_names"])


def tactile_state_names() -> list[str]:
    return [
        f"tactile.finger_{finger}.{feature}"
        for finger in range(1, 6)
        for feature in CONTRACT["tactile"]["per_finger_features"]
    ]
