from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from oellm_agent.can.can_decoder import ID_181, get_supported_ids
from oellm_agent.can.closed_loop_sim_generator import ClosedLoopSim
from oellm_agent.can.simulate_sensor_pipeline import encode_frames

from .config import DT
from .websocket_server_stream import WebSocketServerFrameStream
from .websocket_stream import WebSocketFrameStream


FrameBatch = Tuple[int, float, List[Tuple[int, List[int]]]]


def stream_steps_from_csv(csv_path: Path) -> Iterator[FrameBatch]:
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cur_step = None
        cur_t = 0.0
        frames: List[Tuple[int, List[int]]] = []

        for row in reader:
            step = int(row['step'])
            t_sec = float(row['t_sec'])

            fid_text = str(row['frame_id_hex']).strip()
            try:
                frame_id = int(fid_text, 16)
            except ValueError:
                frame_id = int(fid_text[-8:], 16)

            payload = [int(row[f'byte{i}']) for i in range(8)]
            if frame_id == 0x18F181A0 and 'can_heartbeat_ok' in row and row['can_heartbeat_ok'] != '':
                payload[7] = int(row['can_heartbeat_ok']) & 0xFF

            if cur_step is None:
                cur_step = step
                cur_t = t_sec

            if step != cur_step:
                yield cur_step, cur_t, frames
                cur_step = step
                cur_t = t_sec
                frames = []

            frames.append((frame_id, payload))

        if cur_step is not None:
            yield cur_step, cur_t, frames


def stream_steps_from_socketcan(channel: str, bitrate: int = 250000, model: str = '50') -> Iterator[FrameBatch]:
    try:
        import can  # type: ignore
    except Exception as e:
        raise RuntimeError('实时CAN模式需要安装 python-can') from e

    bus = can.interface.Bus(channel=channel, bustype='socketcan', bitrate=bitrate)
    supported_ids = get_supported_ids(model)
    group_start_id = ID_181
    step = 0
    start = time.time()
    current_group: List[Tuple[int, List[int]]] = []
    seen_ids = set()
    current_group_started_at: Optional[float] = None

    while True:
        msg = bus.recv(timeout=1.0)
        if msg is None:
            continue

        frame_id = int(msg.arbitration_id)
        if frame_id not in supported_ids:
            continue

        payload = list(msg.data[:8])
        now = time.time()

        if frame_id == group_start_id and current_group:
            t_sec = round(current_group_started_at - start, 3) if current_group_started_at is not None else round(step * DT, 3)
            yield step, t_sec, current_group
            step += 1
            current_group = []
            seen_ids = set()
            current_group_started_at = None

        if current_group_started_at is None:
            current_group_started_at = now

        if frame_id in seen_ids and current_group:
            t_sec = round(current_group_started_at - start, 3)
            yield step, t_sec, current_group
            step += 1
            current_group = []
            seen_ids = set()
            current_group_started_at = now

        current_group.append((frame_id, payload))
        seen_ids.add(frame_id)

        if supported_ids.issubset(seen_ids):
            t_sec = round(current_group_started_at - start, 3)
            yield step, t_sec, current_group
            step += 1
            current_group = []
            seen_ids = set()
            current_group_started_at = None


class ClosedLoopStepStream:
    def __init__(self, duration_sec: float = 120.0, default_action: str = 'FORWARD', realtime: bool = False, model: str = '50'):
        self.sim = ClosedLoopSim()
        self.model = str(model or '190')
        self.total_steps = int(max(1.0, duration_sec) / DT)
        self.current_action = default_action.upper()
        self.realtime = realtime
        self.wall_start = time.time()
        self.step = 0

    def set_action(self, action: str) -> None:
        a = str(action or '').upper()
        if a in {'EMERGENCY_STOP', 'FORWARD', 'REVERSE', 'ACCELERATE', 'DECELERATE', 'BRAKE'}:
            self.current_action = a

    def __iter__(self):
        return self

    def __next__(self) -> FrameBatch:
        if self.step >= self.total_steps:
            raise StopIteration
        t_sec = self.step * DT
        if self.realtime:
            target = self.wall_start + t_sec
            now = time.time()
            if target > now:
                time.sleep(target - now)

        state = self.sim.step_once(self.current_action)
        self.last_state = state
        self.last_speed_mps = float(getattr(state, 'speed_mps', 0.0) or 0.0)
        self.last_action_applied = self.current_action
        frames = encode_frames(state, model=self.model)
        out = (self.step, t_sec, frames)
        self.step += 1
        return out


def stream_steps_from_closed_loop(
    duration_sec: float = 120.0,
    default_action: str = 'FORWARD',
    realtime: bool = False,
    model: str = '50',
) -> ClosedLoopStepStream:
    return ClosedLoopStepStream(duration_sec=duration_sec, default_action=default_action, realtime=realtime, model=model)


def stream_steps_from_websocket(ws_url: str, realtime: bool = True, heartbeat_timeout_sec: float = 10.0) -> WebSocketFrameStream:
    stream = WebSocketFrameStream(ws_url=ws_url, realtime=realtime, heartbeat_timeout_sec=heartbeat_timeout_sec)
    stream.start()
    return stream


def stream_steps_from_websocket_server(host: str = '0.0.0.0', port: int = 9001, realtime: bool = True, heartbeat_timeout_sec: float = 10.0, csv_path: str | None = None) -> WebSocketServerFrameStream:
    stream = WebSocketServerFrameStream(host=host, port=port, realtime=realtime, heartbeat_timeout_sec=heartbeat_timeout_sec, csv_path=csv_path)
    stream.start()
    return stream
