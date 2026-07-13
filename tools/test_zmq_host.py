"""zmq_utils host 参数 + eval_endpoints_from_env 单元测试(不需 Isaac)。
从仓库根运行:  python3 tools/test_zmq_host.py"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.utils.zmq_utils import ZmqPublisher, ZmqReceiver, eval_endpoints_from_env

fails = []
def ck(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {name}")
    if not cond:
        fails.append(name)

def roundtrip(pub_host, recv_host, port):
    """PUB/SUB 慢加入者:重复发直到收到一帧 obs;收到返回信封,否则 None。"""
    pub = ZmqPublisher(port=port, host=pub_host)
    recv = ZmqReceiver(port=port, host=recv_host)
    try:
        for _ in range(50):
            pub.send_msg({"v": 1}, topic=b"obs", episode_id=1, step_id=0)
            env = recv.receive_envelope(timeout=100)
            if env and env.get("topic") == "obs":
                return env
        return None
    finally:
        recv.close()

# 1) 显式 host 同机往返(证明 host 参数贯通 bind+connect)
ck("显式 host=127.0.0.1 往返", roundtrip("127.0.0.1", "127.0.0.1", 5770) is not None)
# 2) 跨机语义在 loopback 上验证:Publisher bind 0.0.0.0 + Receiver connect 127.0.0.1
ck("跨机语义 bind 0.0.0.0 + connect 127.0.0.1 往返", roundtrip("0.0.0.0", "127.0.0.1", 5772) is not None)
# 3) 默认值往返(Publisher 默认 0.0.0.0 / Receiver 默认 127.0.0.1)—— 同机零配置
ck("默认值往返(同机零配置)", roundtrip("0.0.0.0", "127.0.0.1", 5771) is not None)
# 4) env helper 默认
for k in ("EVAL_OBS_PORT", "EVAL_ACTION_PORT", "EVAL_INFER_HOST"):
    os.environ.pop(k, None)
ck("env helper 默认 (5556,5557,127.0.0.1)", eval_endpoints_from_env() == (5556, 5557, "127.0.0.1"))
# 5) env helper 覆盖
os.environ["EVAL_OBS_PORT"] = "6000"
os.environ["EVAL_ACTION_PORT"] = "6001"
os.environ["EVAL_INFER_HOST"] = "10.0.0.9"
ck("env helper 覆盖", eval_endpoints_from_env() == (6000, 6001, "10.0.0.9"))

print()
if fails:
    print("[FAIL] " + "; ".join(fails)); sys.exit(1)
print("[PASS] zmq_utils host 参数 + env helper"); sys.exit(0)
