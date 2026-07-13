#!/usr/bin/env bash
# run_sim_local.sh — 一键本地仿真可视化：起 Isaac 仿真 + 你的推理服务，跑完自动收尾
#
# 用法:
#   bash tools/run_sim_local.sh --task ind_task_01 --ckpt checkpoints/ind_task_01_act/policy_last.ckpt
# 可选:
#   --headless   无窗口跑（服务器/无显示器机器），事后看 logs/task_videos/ 的每集 mp4
#   --loop N     跑几集 (默认 3)
#   --seed S     场景随机种子 (默认 0)
#   --timeout T  单集时长秒 (默认 300)。本地没有任务判定，每集固定跑满该时长后自动进下一集
#   --policy P   策略名 (默认 act)
#   --no-video   不录每集 mp4（默认录，存 logs/task_videos/）
#
# 前置（见 README §7）:
#   1) 安装 Isaac Sim，并把 common/isaac_config.toml 的 python_path 改成你的安装路径
#   2) 仿真依赖装进 Isaac 的 python:  <IsaacSim>/python.sh -m pip install -r requirements.sim.txt
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TASK="" CKPT="" HEADLESS="" LOOP=3 SEED=0 TIMEOUT=300 POLICY="act" NO_VIDEO=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)     TASK="$2"; shift 2 ;;
        --ckpt)     CKPT="$2"; shift 2 ;;
        --headless) HEADLESS="--headless"; shift ;;
        --loop)     LOOP="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --timeout)  TIMEOUT="$2"; shift 2 ;;
        --policy)   POLICY="$2"; shift 2 ;;
        --no-video) NO_VIDEO=1; shift ;;
        *) echo "[ERR] 未知参数: $1（支持 --task --ckpt --headless --loop --seed --timeout --policy --no-video）"; exit 1 ;;
    esac
done
[[ -n "$NO_VIDEO" ]] && export RECORD_VIDEO=0

# ---------- 预检 ----------
case "$TASK" in
    ind_task_01|ind_task_02|ind_task_03|lab_task_01|lab_task_03) ;;
    *) echo "[ERR] --task 必须是: ind_task_01 ind_task_02 ind_task_03 lab_task_01 lab_task_03（当前: '${TASK}'）"; exit 1 ;;
esac
[[ -f "$CKPT" ]] || { echo "[ERR] checkpoint 不存在: $CKPT"; exit 1; }
[[ -f "$(dirname "$CKPT")/dataset_stats.pkl" ]] || {
    echo "[ERR] $(dirname "$CKPT")/dataset_stats.pkl 不存在（必须与 ckpt 同目录，训练脚本会自动生成）"; exit 1; }

ISAAC_PYTHON="$(grep -oP 'python_path\s*=\s*"\K[^"]+' common/isaac_config.toml | head -1)"
[[ -n "$ISAAC_PYTHON" && -x "$ISAAC_PYTHON" ]] || {
    echo "[ERR] Isaac Sim python 不可用: '${ISAAC_PYTHON}'"
    echo "      请把 common/isaac_config.toml 的 python_path 改成你的 Isaac Sim 安装路径（形如 /home/you/isaacsim/python.sh）"; exit 1; }

if [[ -z "$HEADLESS" && -z "${DISPLAY:-}" ]]; then
    echo "[ERR] 当前机器没有图形显示（DISPLAY 为空），看不了 GUI 窗口。"
    echo "      请加 --headless 无窗口运行，跑完看 logs/task_videos/ 里每集的 mp4。"; exit 1
fi

# 机器人模型以 .gz 入库（GitHub 单文件 100MB 限制），首次运行解压一次
ROBOT_USD="usds/tianyi_2_6d_force_brainco2_hand_v1/tianyi_2_6d_force_brainco2_hand_v1_pitch_17.usd"
if [[ ! -f "$ROBOT_USD" ]]; then
    if [[ -f "$ROBOT_USD.gz" ]]; then
        echo "[INFO] 首次运行：解压机器人模型（~120MB，仅此一次）..."
        gunzip -kf "$ROBOT_USD.gz"
    else
        echo "[ERR] 机器人模型不存在: $ROBOT_USD(.gz)（仓库不完整？请重新 clone）"; exit 1
    fi
fi

for PORT in 5556 5557; do
    if timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/$PORT" 2>/dev/null; then
        echo "[ERR] 端口 $PORT 已被占用（可能有残留的 benchmark.py / policy_infer.py，先 pkill 再重试）"; exit 1
    fi
done

RUN_DIR="logs/sim_local/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

# ---------- 进程管理 ----------
BENCH_PID="" INFER_PID=""
cleanup() {
    [[ -n "$INFER_PID" ]] && kill "$INFER_PID" 2>/dev/null || true
    # benchmark 会 fork Isaac 子进程；setsid 使其自成进程组，负号杀整组
    [[ -n "$BENCH_PID" ]] && { kill -- -"$BENCH_PID" 2>/dev/null || kill "$BENCH_PID" 2>/dev/null || true; }
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------- 1/3 起仿真（常驻，等推理端发任务名） ----------
echo "[1/3] 启动 Isaac 仿真${HEADLESS:+（headless）} ... 日志: $RUN_DIR/benchmark.log"
setsid "$ISAAC_PYTHON" benchmark.py --loop "$LOOP" --seed "$SEED" --timeout "$TIMEOUT" $HEADLESS \
    > "$RUN_DIR/benchmark.log" 2>&1 &
BENCH_PID=$!

# ---------- 2/3 起推理（发布任务名，握手后进入推理循环） ----------
echo "[2/3] 启动推理服务（policy=$POLICY）... 日志: $RUN_DIR/infer.log"
PYTHONUNBUFFERED=1 python3 tools/policy_infer.py \
    --policy "$POLICY" --model-path "$CKPT" --task "$TASK" \
    > "$RUN_DIR/infer.log" 2>&1 &
INFER_PID=$!

# ---------- 3/3 等待完成 ----------
echo "[3/3] 运行中 — $TASK × ${LOOP} 集（seed=$SEED，每集 ${TIMEOUT}s）。首集需等 Isaac 加载（1-3 分钟）；Ctrl-C 可随时中止。"
while true; do
    sleep 5
    if ! kill -0 "$BENCH_PID" 2>/dev/null; then
        echo "[ERR] 仿真进程提前退出，benchmark.log 最后 30 行："; tail -30 "$RUN_DIR/benchmark.log"; exit 1
    fi
    if ! kill -0 "$INFER_PID" 2>/dev/null; then
        echo "[ERR] 推理进程提前退出，infer.log 最后 30 行："; tail -30 "$RUN_DIR/infer.log"; exit 1
    fi
    grep -q "Practice run finished" "$RUN_DIR/benchmark.log" && break
done

echo
echo "=== 跑完 ${LOOP} 集 ==="
[[ -z "$NO_VIDEO" ]] && echo "每集视频: logs/task_videos/（--headless 时在这里看仿真过程）"
echo "提示: 本地只提供运行画面，无成绩输出；正式评测由组织方进行。"
