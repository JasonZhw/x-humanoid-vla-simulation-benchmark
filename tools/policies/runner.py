"""
ZMQ inference loop — receives obs from simulation, sends back actions.
Policy implementation is in act_policy.py.
"""
import os
import sys
import argparse
import time
import cv2
import numpy as np

# Ensure project root is on sys.path regardless of how this module is imported
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from common.utils.zmq_utils import ZmqPublisher, ZmqReceiver
from tools.policies import POLICY_MAP


def parse_args():
    parser = argparse.ArgumentParser(description="Policy inference server")
    parser.add_argument(
        "--policy", choices=list(POLICY_MAP.keys()), required=True,
        help="Policy type: act | act_v1"
    )
    parser.add_argument(
        "--model-path", required=False, default=None,
        help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Inference device: cuda | cpu  (default: cuda)"
    )
    parser.add_argument("--zmq-recv-port", type=int, default=5556)
    parser.add_argument("--zmq-send-port", type=int, default=5557)
    parser.add_argument("--sim-host", default="127.0.0.1",
                        help="Organizer sim host to receive obs from "
                             "(cross-machine eval; default same-machine 127.0.0.1)")
    parser.add_argument(
        "--task", default="",
        help="Language task description (published via ZMQ topic 'task' before inference)"
    )
    # ACT 参数
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Action chunk size (ACT default: 50)"
    )
    parser.add_argument(
        "--temporal-agg", action="store_true", default=None,
        help="Enable temporal aggregation (ACT default: True)"
    )
    parser.add_argument(
        "--no-temporal-agg", dest="temporal_agg", action="store_false",
        help="Disable temporal aggregation"
    )
    # ACT image resize override (defaults set per policy name: act=320x240, act_v1=640x480)
    parser.add_argument("--img-w", type=int, default=None,
                        help="ACT inference image resize width (override per-policy default)")
    parser.add_argument("--img-h", type=int, default=None,
                        help="ACT inference image resize height (override per-policy default)")
    return parser.parse_args()


def build_action_dict(np_action: np.ndarray, robot_type: str) -> dict:
    """
    将 action array 按末端执行器类型拆分为仿真侧期望的 dict。

    robot_type="brainco2": [left_arm(7), left_hand(6), right_arm(7), right_hand(6)]
    robot_type="gripper":  [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
    """
    if robot_type == "brainco2":
        return {
            "left_arm":  np_action[:7].tolist(),
            "left_hand": np_action[7:13].tolist(),
            "right_arm": np_action[13:20].tolist(),
            "right_hand": np_action[20:26].tolist(),
        }
    elif robot_type == "gripper":
        return {
            "left_arm":  np_action[:7].tolist(),
            "left_hand": np_action[7:8].tolist(),
            "right_arm": np_action[8:15].tolist(),
            "right_hand": np_action[15:16].tolist(),
        }
    else:
        raise ValueError(f"Unknown robot_type: {robot_type}")


def run_inference_server():
    args = parse_args()

    print(f"[runner] Loading {args.policy} policy from {args.model_path} on {args.device}")
    extra_kwargs = {}
    if args.policy in ("act", "act_v1"):
        # Single merged ACT file; resolution differs only by policy name / flags.
        from tools.policies.act_policy import ACTPolicy as _PolicyCls
        if args.policy == "act":
            _PolicyCls.IMG_W, _PolicyCls.IMG_H = 320, 240
        else:  # act_v1
            _PolicyCls.IMG_W, _PolicyCls.IMG_H = 640, 480
        if args.img_w is not None:
            _PolicyCls.IMG_W = args.img_w
        if args.img_h is not None:
            _PolicyCls.IMG_H = args.img_h
        if args.chunk_size is not None:
            _PolicyCls.CHUNK_SIZE = args.chunk_size
        if args.temporal_agg is not None:
            _PolicyCls.TEMPORAL_AGG = args.temporal_agg
    else:
        raise ValueError(f"Unknown policy: {args.policy}")
    policy = _PolicyCls(args.model_path, args.device, **extra_kwargs)

    time.sleep(2)  # 等待仿真侧 ZMQ bind 完成

    zmq_receiver  = ZmqReceiver(port=args.zmq_recv_port, host=args.sim_host)
    zmq_publisher = ZmqPublisher(port=args.zmq_send_port)
    print(f"[runner] Ready — recv:{args.sim_host}:{args.zmq_recv_port}  send:*:{args.zmq_send_port}")

    simulation_running = False
    robot_type = "brainco2"

    # Publish task name to sim side, wait for task_cbd ack
    print(f"[runner] Publishing task {args.task!r}, waiting for task_cbd...")
    while True:
        zmq_publisher.send_msg(args.task, topic=b"task")
        result = zmq_receiver.receive_envelope(timeout=500)
        if result is not None and str(result.get("topic", "")) == "task_cbd":
            print("[runner] task_cbd received, starting inference loop")
            break

    try:
        while True:
            result = zmq_receiver.receive_envelope()
            if result is None:
                continue

            topic      = str(result.get("topic", "")).encode("utf-8")
            data       = result.get("payload")
            episode_id = int(result.get("episode_id", -1))
            step_id    = int(result.get("step_id", -1))

            if topic == b"start":
                print(f"[runner] Episode {episode_id} started, robot={data}")
                simulation_running = True
                if isinstance(data, dict):
                    robot_type = data.get("end_effector", "brainco2")
                policy.reset()

            elif topic == b"obs":
                if not simulation_running:
                    print(f"[runner] Auto-start on first obs (episode {episode_id}) — start msg was lost")
                    simulation_running = True
                    policy.reset()
                if data is None:
                    print("[runner] Error: obs payload is None, skipping")
                    continue

                np_action = policy.infer(obs=data)
                action_dict = build_action_dict(np_action, robot_type)
                zmq_publisher.send_msg(
                    action_dict, topic=b"action",
                    episode_id=episode_id, step_id=step_id
                )
                # DEBUG: 每10步记录一次 action
                if step_id % 10 == 0:
                    r_arm = np_action[13:20]
                    r_hand = np_action[20:26]
                    has_nan = np.any(np.isnan(np_action))
                    msg = f"[runner|step={step_id}] r_arm={np.round(r_arm,4).tolist()}  r_hand={np.round(r_hand,3).tolist()}  nan={has_nan}"
                    print(msg)
                    if hasattr(policy, '_dbg'):
                        policy._dbg(msg)

            elif topic == b"reset":
                print(f"[runner] Episode {episode_id} ended, resetting")
                simulation_running = False
                policy.reset()

            elif topic == b"test":
                zmq_publisher.send_msg(
                    data=None, topic=b"test",
                    episode_id=episode_id, step_id=step_id
                )

            else:
                print(f"[runner] Unknown topic: {topic}")

    except KeyboardInterrupt:
        print("\n[runner] Shutting down...")
    except Exception as e:
        print(f"[runner] Fatal error: {e}")
        raise
    finally:
        zmq_receiver.close()
        cv2.destroyAllWindows()
        print("[runner] Done")


if __name__ == "__main__":
    run_inference_server()
