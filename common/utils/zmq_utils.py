import pickle
import time
from typing import Any, Optional
import zmq
from common.logger_loader import logger

class ZmqPublisher(object):
    def __init__(self, port, NUM_SNDHWM=1, host="0.0.0.0"):
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, NUM_SNDHWM)
        self.socket.bind(f"tcp://{host}:{port}")
        print(f"ZmqPublisher bind to {host}:{port}")

    def send_msg(self, data: Any, topic: bytes, episode_id: int = 0, step_id: int = 0, timestamp: Optional[float] = None):
        topic_value = topic.decode("utf-8") if isinstance(topic, (bytes, bytearray)) else str(topic)
        envelope = {
            "schema_version": 1,
            "topic": topic_value,
            "episode_id": int(episode_id),
            "step_id": int(step_id),
            "timestamp": float(time.time() if timestamp is None else timestamp),
            "payload": data,
        }
        message = pickle.dumps(envelope)
        self.socket.send(message)
        print(f"send msg from zmq with topic: {topic}")

class ZmqPublisher2Xrocs(object):
    def __init__(self, port):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://127.0.0.1:{port}")
    
    def send_msg_to_xrocs(self, data):
        serialized_static_data = pickle.dumps(data)
        self.socket.send(serialized_static_data)
        reply = self.socket.recv()
        logger.info(f"Received reply: {reply.decode()}")

class ZmqReceiver(object):
    def __init__(self, port, NUM_RCVHWM=1, host="127.0.0.1"):
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, NUM_RCVHWM)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"ZmqReceiver connected to {host}:{port}, subscribing to all topics")
        self._latest_step_by_episode: dict[int, int] = {}
        self.discarded_old_action_count = 0

    def _recv_raw(self, timeout=None):
        if timeout is not None:
            if self.socket.poll(timeout) & zmq.POLLIN:
                return self.socket.recv(zmq.NOBLOCK)
            return None
        return self.socket.recv()

    def _parse_legacy_message(self, message: bytes):
        known_topics = [b"action", b"obs", b"start", b"reset", b"test"]
        for test_topic in known_topics:
            if message.startswith(test_topic):
                try:
                    remaining_data = message[len(test_topic):]
                    data = pickle.loads(remaining_data)
                    return {
                        "schema_version": 0,
                        "topic": test_topic.decode("utf-8"),
                        "episode_id": -1,
                        "step_id": -1,
                        "timestamp": time.time(),
                        "payload": data,
                    }
                except Exception:
                    continue
        pickle_headers = [b'\x80\x04', b'\x80\x03', b'\x80\x02']
        for header in pickle_headers:
            header_pos = message.find(header)
            if header_pos > 0:
                topic = message[:header_pos]
                remaining_data = message[header_pos:]
                try:
                    data = pickle.loads(remaining_data)
                    return {
                        "schema_version": 0,
                        "topic": topic.decode("utf-8", errors="ignore"),
                        "episode_id": -1,
                        "step_id": -1,
                        "timestamp": time.time(),
                        "payload": data,
                    }
                except Exception:
                    continue
        return None

    def _to_envelope(self, message: bytes):
        try:
            data = pickle.loads(message)
            if isinstance(data, dict) and "topic" in data and "payload" in data:
                return {
                    "schema_version": int(data.get("schema_version", 1)),
                    "topic": str(data.get("topic")),
                    "episode_id": int(data.get("episode_id", -1)),
                    "step_id": int(data.get("step_id", -1)),
                    "timestamp": float(data.get("timestamp", time.time())),
                    "payload": data.get("payload"),
                }
        except Exception:
            pass
        return self._parse_legacy_message(message)

    def receive_envelope(self, timeout=None):
        try:
            message = self._recv_raw(timeout=timeout)
            if message is None:
                return None
            return self._to_envelope(message)
        except zmq.Again:
            return None
        except Exception as e:
            logger.warning(f"Error receiving message: {e}")
            return None

    def receive_msg(self, timeout=None):
        envelope = self.receive_envelope(timeout=timeout)
        if envelope is None:
            return None
        topic = str(envelope.get("topic", "")).encode("utf-8")
        return topic, envelope.get("payload")

    def is_old_action(self, envelope: dict) -> bool:
        if str(envelope.get("topic")) != "action":
            return False
        episode_id = int(envelope.get("episode_id", -1))
        step_id = int(envelope.get("step_id", -1))
        if episode_id < 0 or step_id < 0:
            return False
        latest_step = self._latest_step_by_episode.get(episode_id, -1)
        if step_id <= latest_step:
            self.discarded_old_action_count += 1
            return True
        self._latest_step_by_episode[episode_id] = step_id
        return False

    def close(self):
        self.socket.close()
        print("ZmqReceiver socket closed")


def eval_endpoints_from_env():
    """组织方 sim 侧跨机评测端点配置(从环境变量读,带同机默认)。
    返回 (obs_port, action_port, infer_host)。
    EVAL_OBS_PORT / EVAL_ACTION_PORT / EVAL_INFER_HOST 未设时 = (5556, 5557, '127.0.0.1')。"""
    import os
    obs_port = int(os.environ.get("EVAL_OBS_PORT", 5556))
    action_port = int(os.environ.get("EVAL_ACTION_PORT", 5557))
    infer_host = os.environ.get("EVAL_INFER_HOST", "127.0.0.1")
    return obs_port, action_port, infer_host
