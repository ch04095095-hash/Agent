from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from .config import SHORT_WINDOW_SEC
from .runner import run_online
from .streams import stream_steps_from_closed_loop, stream_steps_from_csv, stream_steps_from_socketcan, stream_steps_from_websocket, stream_steps_from_websocket_server
from ..can.control_frame_encoder import decision_to_control_frame
from .control_websocket_sender import ControlWebSocketSender


def _init_run_log(base_dir: Path, run_id: str) -> Path:
    logs_dir = (base_dir / 'logs').resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f'run_history_{run_id}.jsonl'
    path.write_text('', encoding='utf-8')
    return path


def _append_run_event(log_path: Path, event: Dict[str, Any]) -> Path:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')
    return log_path


def _save_run_record(log_path: Path, summary: Dict[str, Any], args: argparse.Namespace) -> Path:
    rec = {
        'event': 'run_completed',
        'ts': time.time(),
        'date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'source': args.source,
        'agent_url': args.agent_url,
        'l2_every_sec': args.l2_every_sec,
        'window_sec': SHORT_WINDOW_SEC,
        'run': summary,
    }
    return _append_run_event(log_path, rec)


def main() -> None:
    parser = argparse.ArgumentParser(description='Online 3-layer pipeline (single chain for csv/socketcan/closed_loop).')
    parser.add_argument('--source', choices=['csv', 'socketcan', 'closed_loop', 'websocket', 'websocket-server'], default='closed_loop')
    parser.add_argument('--source-csv', default='/mnt/ssd/Agent/oellm_agent/sim_data/sim_can_frames_10min_10hz.csv')
    parser.add_argument('--can-channel', default='can0')
    parser.add_argument('--can-bitrate', type=int, default=250000)
    parser.add_argument('--ws-url', default='ws://127.0.0.1:9001', help='websocket source url')
    parser.add_argument('--ws-host', default='0.0.0.0', help='websocket server host')
    parser.add_argument('--ws-port', type=int, default=9001, help='websocket server port')
    parser.add_argument('--ws-heartbeat-timeout-sec', type=float, default=10.0, help='websocket source flush timeout')
    parser.add_argument('--ws-save-csv', default='', help='save received websocket frames to csv')
    parser.add_argument('--sim-duration-sec', type=float, default=1800, help='closed_loop source duration (sec)')
    parser.add_argument('--sim-default-action', default='FORWARD', help='closed_loop default action when agent call fails')
    parser.add_argument('--realtime', action='store_true', help='CSV/closed_loop回放按组周期真实节奏跑')
    parser.add_argument('--agent-url', default='http://127.0.0.1:18080', help='agent HTTP base url')
    parser.add_argument('--model', choices=['190', '105', '50'], default='50', help='车型协议与阈值选择')
    parser.add_argument('--l2-every-sec', type=float, default=1.0, help='L2调用agent周期(秒)，默认1.0，建议按组周期的整数倍设置')
    parser.add_argument('--agent-timeout-sec', type=float, default=5.0, help='agent HTTP调用超时(秒)')
    parser.add_argument('--ws-control-forward-url', default='', help='control websocket url for sending decision frames')
    args = parser.parse_args()

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{args.source}"

    summary: Dict[str, Any]

    control_sender = ControlWebSocketSender(args.ws_control_forward_url) if args.ws_control_forward_url else None
    if control_sender is not None:
        control_sender.start()
        print(json.dumps({'layer': 'PIPELINE', 'event': 'control_forward_enabled', 'ws_control_forward_url': args.ws_control_forward_url}, ensure_ascii=False), flush=True)
    try:
        if args.source == 'csv':
            stream = stream_steps_from_csv(Path(args.source_csv))
            summary = run_online(
                stream,
                agent_url=args.agent_url,
                model=args.model,
                realtime=args.realtime,
                l2_every_sec=args.l2_every_sec,
                agent_timeout_sec=args.agent_timeout_sec,
                run_id=run_id,
                run_event_cb=None,
                control_sender=control_sender,
            )
        elif args.source == 'socketcan':
            stream = stream_steps_from_socketcan(args.can_channel, args.can_bitrate, model=args.model)
            summary = run_online(
                stream,
                agent_url=args.agent_url,
                model=args.model,
                realtime=False,
                l2_every_sec=args.l2_every_sec,
                agent_timeout_sec=args.agent_timeout_sec,
                run_id=run_id,
                run_event_cb=None,
                control_sender=control_sender,
            )
        elif args.source == 'websocket':
            stream = stream_steps_from_websocket(
                ws_url=args.ws_url,
                realtime=args.realtime,
                heartbeat_timeout_sec=args.ws_heartbeat_timeout_sec,
            )
            summary = run_online(
                stream,
                agent_url=args.agent_url,
                model=args.model,
                realtime=args.realtime,
                l2_every_sec=args.l2_every_sec,
                agent_timeout_sec=args.agent_timeout_sec,
                run_id=run_id,
                run_event_cb=None,
                control_sender=control_sender,
            )
        elif args.source == 'websocket-server':
            stream = stream_steps_from_websocket_server(
                host=args.ws_host,
                port=args.ws_port,
                realtime=args.realtime,
                heartbeat_timeout_sec=args.ws_heartbeat_timeout_sec,
                csv_path=args.ws_save_csv or None,
            )
            summary = run_online(
                stream,
                agent_url=args.agent_url,
                model=args.model,
                realtime=args.realtime,
                l2_every_sec=args.l2_every_sec,
                agent_timeout_sec=args.agent_timeout_sec,
                run_id=run_id,
                run_event_cb=None,
                control_sender=control_sender,
            )
        else:
            stream = stream_steps_from_closed_loop(
                duration_sec=args.sim_duration_sec,
                default_action=args.sim_default_action,
                realtime=args.realtime,
                model=args.model,
            )
            summary = run_online(
                stream,
                agent_url=args.agent_url,
                model=args.model,
                realtime=args.realtime,
                l2_every_sec=args.l2_every_sec,
                agent_timeout_sec=args.agent_timeout_sec,
                run_id=run_id,
                run_event_cb=None,
            )

        print(json.dumps({'layer': 'RUN_COMPLETED', 'run_id': summary.get('run_id')}, ensure_ascii=False), flush=True)
    except Exception:
        raise
    finally:
        if control_sender is not None:
            control_sender.stop()


if __name__ == '__main__':
    main()
