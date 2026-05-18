from __future__ import annotations

from typing import Any, Dict, List, Tuple

CONTROL_FRAME_ID = 0x168

ACTION_TO_FRAME_CONTENT = {
    # Command byte mapping (8-bit binary strings):
    # START=0000 0110, BRAKE=0000 1010, FORWARD=0001 0010,
    # REVERSE=0010 0010, ACCELERATE=0100 0010, DECELERATE=1000 0010
    'START': ['0000 0110', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000'],
    'BRAKE': ['0000 1010', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000'],
    'FORWARD': ['0001 0010', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000'],
    'REVERSE': ['0010 0010', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000'],
    'ACCELERATE': ['0100 0010', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000'],
    'DECELERATE': ['1000 0010', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000', '0000 0000'],
    'HOLD': None,
}


def decision_to_control_frame(decision: Dict[str, Any]) -> Tuple[int, List[int]] | None:
    action = str((decision or {}).get('action', '') or '').upper()
    payload = ACTION_TO_FRAME_CONTENT.get(action)
    if payload is None:
        return None
    return CONTROL_FRAME_ID, list(payload)
