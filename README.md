# ACT Training / Inference — Contestant Guide

This repository provides complete ACT (Action Chunking Transformer) training + inference code; the training data is open-sourced on ModelScope (see §2). You can train the ACT baseline directly, or replace the inference logic with your own algorithm — just follow the ZMQ contract in §5. Evaluation runs in the organizer's simulation and talks to your inference server over ZMQ.

## 1. Environment

Python 3.10 + one CUDA GPU (training/inference do **not** need Isaac Sim). Install (the combination verified by the organizer; you may use your own versions):

```bash
# torch must use --index-url (otherwise a CPU build is installed); match cuXXX to your CUDA: 12.x→cu128 / 13.x→cu130 / older→cu121
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.lock.txt
```

## 2. Data

The training data is open-sourced on **ModelScope**, ~200 episodes per task (one episode = one `.hdf5`), all used for training:

> https://modelscope.cn/datasets/X-Humanoid/VLA-Challenge-Dataset

**Download & place**: after downloading you get `<task>/h5_data/*.hdf5`; the training code reads `data/<task>/train/`, so put the hdf5 files there (**do not rename the task folders**; replace `<download>` with your download path):

```bash
for t in ind_task_01 ind_task_02 ind_task_03 lab_task_01 lab_task_03; do
  mkdir -p data/$t/train && cp <download>/$t/h5_data/*.hdf5 data/$t/train/
done
```

**Validation set (optional)**: move some files from `train/` to `val/` to enable periodic validation; if `val/` is empty or absent, all data is used for training.

```bash
cd data/ind_task_01 && mkdir -p val && ls train/*.hdf5 | shuf | head -20 | xargs -I{} mv {} val/
```

**Data fields** (each `.hdf5` = one episode, T frames):

```
camera_observations/color_images/camera_head   (T,) object   # JPEG; decoded to (720,1280,3) uint8 RGB
camera_observations/depth_images/camera_head   (T,) object   # PNG; decoded to (720,1280) uint16 mm (baseline unused)
puppet/arm_left_position_align/data            (T, 7)         # left arm joint angles
puppet/end_effector_left_position_align/data   (T, 6)         # left hand (fingers)
puppet/arm_right_position_align/data           (T, 7)         # right arm joint angles
puppet/end_effector_right_position_align/data  (T, 6)         # right hand (fingers)
```

Training state/action are both **26-dim**: `[left_arm 7, left_hand 6, right_arm 7, right_hand 6]` (baseline uses color + joints only).

## 3. Training

One command per task (run from the repo root; tmux background + auto-resume):

```bash
bash tools/train/run_train_act_5tasks.sh ind_task_01   # replace with one of the 5 task names
```

Defaults: 320×240 / batch 8 / chunk 50 / 100000 steps. Or call it directly:

```bash
python3 tools/train/train_act.py --task-dir data/ind_task_01 --ckpt-dir checkpoints/ind_task_01_act \
    --img-w 320 --img-h 240 --num-steps 100000 --batch-size 8 --chunk-size 50 --use-aug
```

Outputs in `checkpoints/<task>_act/`: `policy_last.ckpt` (used by inference by default) + `dataset_stats.pkl` (**must sit next to the ckpt**, auto-loaded) + `train.log`; `agent_best.ckpt` is produced only if you set aside a non-empty `val/`.

## 4. Inference server

```bash
python3 tools/policy_infer.py --policy act \
    --model-path checkpoints/ind_task_01_act/policy_last.ckpt \
    --zmq-recv-port 5556 --zmq-send-port 5557
```

When ready it prints `[runner] Ready — recv:5556  send:5557`. ZMQ binds `127.0.0.1` (the inference server and the simulation must be mutually reachable on the same host; the exact deployment follows the organizer's instructions). Training and inference resolution must match (default 320×240); the task is resolved automatically from `dataset_stats.pkl`, so no task name is needed.

## 5. ZMQ contract (read this for custom algorithms)

The simulation side (organizer) sends `obs` and receives `action`. If you use your own algorithm, just make your inference server follow the formats below.

**Obs (simulation → you, port 5556)**:

```python
obs = {
    "puppet": {
        "arm_left_position_raw":  {"data": np.ndarray(7,)},
        "arm_right_position_raw": {"data": np.ndarray(7,)},
        "end_effector_left_position_raw":  {"data": np.ndarray(6,)},
        "end_effector_right_position_raw": {"data": np.ndarray(6,)},
        "end_effector_left_pose_raw":  {"data": np.ndarray(7,)},   # left end-effector cartesian pose xyz+quat; optional, ACT ignores it
        "end_effector_right_pose_raw": {"data": np.ndarray(7,)},   # right end-effector cartesian pose; optional
    },
    "camera_observations": {
        "color_images": {
            "camera_head": np.ndarray(H, W, 3)   # RGB, uint8, native 1280x720
        },
        "depth_images": {
            "camera_head": np.ndarray(H, W)      # float32, unit mm (depth). Read it for an RGBD model; ignore if RGB-only
        }
    }
}
```

**Action (you → simulation, port 5557)**:

```python
action = {
    "left_arm":   list[float] * 7,   # left arm target joint angles
    "left_hand":  list[float] * 6,   # left hand target positions
    "right_arm":  list[float] * 7,   # right arm target joint angles
    "right_hand": list[float] * 6,   # right hand target positions
}
```

### Using your own algorithm (two options)

**Option A — reuse this repo's ZMQ loop (recommended, least effort)**: write one policy class, no ZMQ/eval to touch.

1. Create `tools/policies/my_policy.py`, subclass `BasePolicy` (`tools/policies/base_policy.py`), and implement 3 methods:

```python
import numpy as np
from tools.policies.base_policy import BasePolicy

class MyPolicy(BasePolicy):
    def _load_model(self):
        # load and return your model (self.model_path = the --model-path value, self.device = --device)
        return load_my_model(self.model_path)

    def reset(self):
        # clear internal state at the start of each episode (action buffer / history, etc.)
        ...

    def infer(self, obs: dict) -> np.ndarray:
        # obs structure is shown above; return a 26-dim action np.ndarray
        l_arm = np.array(obs["puppet"]["arm_left_position_raw"]["data"])     # (7,)
        r_arm = np.array(obs["puppet"]["arm_right_position_raw"]["data"])    # (7,)
        img   = obs["camera_observations"]["color_images"]["camera_head"]   # (H,W,3) RGB uint8
        action = ...                       # your inference logic
        return action.astype(np.float32)   # shape (26,) = [left_arm 7, left_hand 6, right_arm 7, right_hand 6]
```

> `BasePolicy.__init__(model_path, device)` automatically calls your `_load_model()` and stores it in `self.model`.

2. Add one line to `POLICY_MAP` in `tools/policies/__init__.py`: `"my_policy": None,`
3. In `tools/policies/runner.py`, at the policy-dispatch section, add an `elif` **before the `else:` fallback** (putting it after `else` raises `SyntaxError`):

```python
elif args.policy == "my_policy":
    from tools.policies.my_policy import MyPolicy as _PolicyCls
```

Launch: `python3 tools/policy_infer.py --policy my_policy --model-path <your model>`.
The runner automatically splits the 26-dim vector returned by your `infer` into 4 action fields via `[:7]/[7:13]/[13:20]/[20:26]` — you **never touch ZMQ**.

**Option B — write your own inference server**: independent of this repo's code; just follow this section's ZMQ contract — receive `obs` on port 5556, send `action` on 5557, handle the handshake (the simulation first sends `test` to warm up → `start` → the `obs`/`action` loop → `reset`), send actions back as `{left_arm[7], left_hand[6], right_arm[7], right_hand[6]}`, with a monotonically increasing `step_id`.

## 6. The 5 tasks

> Each task provides 50 scenarios; evaluation picks them by a random seed (reproducible, identical for all contestants).

| Task | Task description | Example scenario |
|---|---|---|
| `ind_task_01` | Use the left hand to place the gear into the red tray. | ![ind_task_01](assets/ind_106.png) |
| `ind_task_02` | Use both hands to place two industrial switches / red buttons into the pink storage basket. | ![ind_task_02](assets/ind_107.png) |
| `ind_task_03` | Pick up the switch with the left hand, transfer it to the right hand, and place it into the red tray. | ![ind_task_03](assets/ind_108.png) |
| `lab_task_01` | Use the right arm to open the door of the electronic balance. | ![lab_task_01](assets/lab_101.png) |
| `lab_task_03` | Use the right arm to close the door of the electronic balance. | ![lab_task_03](assets/lab_103.png) |
