"""runner --sim-host 参数解析单元测试(不需 Isaac/ckpt)。
从仓库根运行:  python3 tools/test_sim_host.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.policies import runner

fails = []
def ck(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {name}")
    if not cond:
        fails.append(name)

sys.argv = ["policy_infer.py", "--policy", "act", "--sim-host", "10.0.0.5"]
ck("--sim-host 解析为传入值", runner.parse_args().sim_host == "10.0.0.5")

sys.argv = ["policy_infer.py", "--policy", "act"]
ck("--sim-host 默认 127.0.0.1", runner.parse_args().sim_host == "127.0.0.1")

print()
if fails:
    print("[FAIL] " + "; ".join(fails)); sys.exit(1)
print("[PASS] runner --sim-host"); sys.exit(0)
