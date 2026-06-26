"""
Self-contained ACT inference policy for the benchmark.

Uses the custom DETR-VAE implementation in tools/policies/detr/
(no lerobot dependency).

Obs format received from benchmark ZMQ (same as HDF5 dataset structure):
  obs['puppet']['arm_left_position_raw']['data']    shape (7,)
  obs['puppet']['end_effector_left_position_raw']['data']  shape (6,)  (推理不读, 用上一步命令)
  obs['puppet']['arm_right_position_raw']['data']   shape (7,)
  obs['puppet']['end_effector_right_position_raw']['data'] shape (6,)  (推理不读)
  obs['camera_observations']['color_images']['camera_head']  ndarray (H, W, 3) RGB

Action output: np.ndarray shape (26,)
  [left_arm(7), left_hand(6), right_arm(7), right_hand(6)]
"""
from __future__ import annotations
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

# Ensure project root is on path so tools.policies.detr is importable
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from tools.policies.detr.main import build_ACT_model_and_optimizer
from tools.policies.base_policy import BasePolicy

# Inference image resize is a per-policy setting: see ACTPolicy.IMG_W / IMG_H
# class attributes (runner.py sets them from the policy name + optional
# --img-w/--img-h flags), so this single file serves both 320x240 and 640x480.

# Normalization used by ACT backbone (ImageNet stats)
_IMAGENET_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Keys used to read state (qpos) from obs — matches eval_policy.py
QPOS_KEYS = [
    "arm_left_position_raw",
    "end_effector_left_position_raw",
    "arm_right_position_raw",
    "end_effector_right_position_raw",
]


def build_act_model(
    chunk_size: int = 50,
    camera_names: list[str] | None = None,
    backbone: str = "resnet18",
    hidden_dim: int = 512,
    dim_feedforward: int = 3200,
    enc_layers: int = 4,
    dec_layers: int = 7,
    nheads: int = 8,
    action_dim: int = 26,
    state_dim: int = 26,
    kl_weight: int = 10,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
) -> "ACTModelWrapper":
    """Build and return an ACTModelWrapper (nn.Module with configure_optimizers)."""
    if camera_names is None:
        camera_names = ["camera_head"]

    args = {
        "lr": lr,
        "lr_backbone": lr_backbone,
        "backbone": backbone,
        "hidden_dim": hidden_dim,
        "dim_feedforward": dim_feedforward,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "nheads": nheads,
        "num_queries": chunk_size,
        "chunk_size": chunk_size,
        "camera_names": camera_names,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "kl_weight": kl_weight,
        # unused in ACT (set defaults)
        "use_vq": False,
        "vq_class": None,
        "vq_dim": None,
        "no_encoder": False,
        "use_depth_image": False,
        "no_sepe_backbone": False,
        "use_lang": False,
        "weight_decay": 1e-4,
        "position_embedding": "sine",
        "masks": False,
        "dilation": False,
        "dropout": 0.1,
        "pre_norm": False,
    }
    return ACTModelWrapper(args)


class ACTModelWrapper(nn.Module):
    """Thin wrapper around the DETR-VAE model matching the training interface."""

    def __init__(self, args: dict) -> None:
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args["kl_weight"]
        self.num_queries = args["num_queries"]

    def __call__(self, qpos, image, depth_image=None, actions=None, is_pad=None,
                 vq_sample=None, language_distilbert=None, logger=None):
        env_state = None
        image = _IMAGENET_NORMALIZE(image)

        if actions is not None:   # training
            actions  = actions[:, :self.model.num_queries]
            is_pad   = is_pad[:, :self.model.num_queries]
            a_hat, _, (mu, logvar), probs, binaries = self.model(
                qpos, image, depth_image, env_state, actions, is_pad, vq_sample,
                lang_embed=language_distilbert)
            total_kld, _, _ = _kl_divergence(mu, logvar)
            all_l1 = torch.nn.functional.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            return {"l1": l1, "kl": total_kld[0], "loss": l1 + total_kld[0] * self.kl_weight}
        else:   # inference
            a_hat, _, (_, _), _, _ = self.model(
                qpos, image, depth_image, env_state, vq_sample=vq_sample,
                lang_embed=language_distilbert)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


def _kl_divergence(mu, logvar):
    if mu.data.ndimension() == 4:
        mu     = mu.view(mu.size(0), mu.size(1))
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total   = klds.sum(1).mean(0, True)
    dim_wise = klds.mean(0)
    mean    = klds.mean(1).mean(0, True)
    return total, dim_wise, mean


# ---------------------------------------------------------------------------
# Inference policy — used by tools/policies/runner.py
# ---------------------------------------------------------------------------

class ACTPolicy(BasePolicy):
    """
    ACT inference policy.

    Loads checkpoint + dataset_stats.pkl from the checkpoint directory.
    Implements temporal aggregation (exp-weighted average over chunk predictions).

    Args (passed via model_path):
      model_path: path to .ckpt file  (dataset_stats.pkl must be in the same dir)
    """

    # Configurable at load time via extra kwargs passed from runner
    CHUNK_SIZE: int = 50
    TEMPORAL_AGG: bool = True
    EPISODE_LEN: int = 2000
    CAM_NAME: str = "camera_head"
    # Inference image resize (width, height). Default 320x240. Overridden by
    # runner.py per policy name (act → 320x240, act_v1 → 640x480) and/or the
    # --img-w/--img-h flags. Must match the resolution the checkpoint was trained at.
    IMG_W: int = 320
    IMG_H: int = 240

    # Debug log file (written alongside the checkpoint)
    _dbg_file = None

    def _open_debug_log(self):
        log_path = os.path.join(os.path.dirname(self.model_path), "act_debug.log")
        self._dbg_file = open(log_path, "w", buffering=1)
        print(f"[ACTPolicy] Debug log → {log_path}")

    def _dbg(self, msg: str):
        print(msg)
        if self._dbg_file:
            self._dbg_file.write(msg + "\n")

    def _load_model(self):
        ckpt_dir = os.path.dirname(self.model_path)
        self._open_debug_log()

        # --- load norm stats ---
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"dataset_stats.pkl not found in {ckpt_dir}")
        with open(stats_path, "rb") as f:
            self.norm_stats = pickle.load(f)

        # --- build model ---
        self.chunk_size = self.CHUNK_SIZE
        policy = build_act_model(chunk_size=self.chunk_size)
        ckpt = torch.load(self.model_path, map_location="cpu")
        policy.deserialize(ckpt["nets"])
        policy.eval()
        policy.to(self.device)
        print(f"[ACTPolicy] Loaded from {self.model_path} (step {ckpt.get('step', '?')})")

        # temporal aggregation buffer (reset on each episode)
        self._init_buffers()
        return policy

    # --- Hand "home" pose: initial finger-joint targets at episode start ---
    #
    # FIXED VALUE SHARED BY ALL 5 TASKS (lab_task_01/03, ind_task_01/02/03), used
    # for BOTH left and right hand.  This is intentional, not yet per-task — change
    # here if you want per-task values or are debugging a bad first step.
    #
    # _get_qpos() does not read the measured finger obs; it feeds the last
    # *commanded* finger position instead, and this home seeds that buffer.  It
    # sets the qpos finger dims at t=0; from t=1 on the buffer is overwritten by
    # the policy's own commanded action (see infer(): _last_*_hand).
    #
    # Options if tuning later: a per-task lookup; or this task's qpos_mean (hand
    # dims) from dataset_stats.pkl; or the per-task mean of frame0 finger angles
    # over the training HDF5.
    _L_HAND_HOME = np.array([1.316, 0.204, 0.209, 0.261, 0.320, 0.312], dtype=np.float32)
    _R_HAND_HOME = np.array([1.316, 0.204, 0.209, 0.261, 0.320, 0.312], dtype=np.float32)

    def _init_buffers(self):
        self.t = -1
        # Last EE positions commanded by the policy (used as obs instead of measured)
        self._last_l_hand = self._L_HAND_HOME.copy()
        self._last_r_hand = self._R_HAND_HOME.copy()
        if self.TEMPORAL_AGG:
            from collections import deque
            # Rolling buffer: stores (base_t, chunk_pred) for last chunk_size predictions
            # No fixed episode length limit — works for any task duration
            self._pred_history: deque = deque()
        else:
            self._chunk_actions = None

    def reset(self):
        self._init_buffers()

    # --- obs pre-processing ---

    def _get_qpos(self, obs: dict) -> np.ndarray:
        l_arm = np.array(obs["puppet"]["arm_left_position_raw"]["data"]).ravel().astype(np.float32)
        r_arm = np.array(obs["puppet"]["arm_right_position_raw"]["data"]).ravel().astype(np.float32)
        l_arm = np.nan_to_num(l_arm, nan=0.0)
        r_arm = np.nan_to_num(r_arm, nan=0.0)
        # Use last commanded EE positions as obs (physical joints have no drives,
        # so measured positions are unreliable / stuck at 0).
        return np.concatenate([l_arm, self._last_l_hand, r_arm, self._last_r_hand])  # (26,)

    def _get_image(self, obs: dict) -> torch.Tensor:
        img_rgb = obs["camera_observations"]["color_images"][self.CAM_NAME]  # (H, W, 3) RGB
        img_rgb = cv2.resize(img_rgb, (self.IMG_W, self.IMG_H))
        img_bgr = img_rgb[:, :, ::-1].copy()                                 # RGB → BGR，与训练侧 dataset.py 一致
        img_f = (img_bgr / 255.0).astype(np.float32)
        t = torch.from_numpy(img_f).permute(2, 0, 1)                         # (3, H, W)
        return t.unsqueeze(0).unsqueeze(0)                                    # (1, 1, 3, H, W)

    def _normalize_qpos(self, qpos: np.ndarray) -> torch.Tensor:
        q = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        return torch.from_numpy(q).float().unsqueeze(0)                       # (1, 26)

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.norm_stats["action_std"] + self.norm_stats["action_mean"]

    # --- main inference ---

    def infer(self, obs: dict) -> np.ndarray:
        self.t += 1
        t = self.t

        # DEBUG: 第一步打印 obs 结构，确认 key 和维度
        if t == 0:
            try:
                self._dbg("[ACT DEBUG] obs puppet keys:")
                for k in obs['puppet']:
                    d = np.array(obs['puppet'][k]['data'])
                    self._dbg(f"  puppet[{k}]: shape={d.shape}, val={np.round(d.ravel()[:7],3).tolist()}")
                qpos_raw = self._get_qpos(obs)
                self._dbg(f"[ACT DEBUG] qpos(26): {np.round(qpos_raw,3).tolist()}")
            except Exception as e:
                self._dbg(f"[ACT DEBUG] obs inspect error: {e}")

        query_frequency = 1 if self.TEMPORAL_AGG else self.chunk_size

        with torch.inference_mode():
            if t % query_frequency == 0:
                qpos = self._get_qpos(obs)
                qpos_t  = self._normalize_qpos(qpos).to(self.device)     # (1, 26)
                image_t = self._get_image(obs).to(self.device)            # (1, 1, 3, H, W)

                # model expects image shape (B, num_cams, C, H, W)
                all_actions = self.model(qpos_t, image_t)                 # (1, chunk, 26)
                all_actions = all_actions.cpu().numpy()
                self._chunk_actions = all_actions                          # cache for non-query steps
                if self.TEMPORAL_AGG:
                    self._pred_history.append((t, all_actions[0]))
                    if len(self._pred_history) > self.chunk_size:
                        self._pred_history.popleft()
                # DEBUG: 每10步打印一次原始 chunk 首尾，看模型有没有预测运动
                if t % 10 == 0:
                    mid  = self.chunk_size // 2
                    r0   = np.round(all_actions[0,   0, 13:20], 3).tolist()
                    r_end= np.round(all_actions[0,  -1, 13:20], 3).tolist()
                    h0   = np.round(all_actions[0,   0, 20:26], 3).tolist()
                    h_mid= np.round(all_actions[0, mid, 20:26], 3).tolist()
                    h_end= np.round(all_actions[0,  -1, 20:26], 3).tolist()
                    self._dbg(f"[CHUNK t={t}] r_arm@0={r0} @{self.chunk_size-1}={r_end}")
                    self._dbg(f"[CHUNK t={t}] r_hand@0={h0} @{mid}={h_mid} @{self.chunk_size-1}={h_end}")

            if self.TEMPORAL_AGG:
                chunk = self.chunk_size
                # Collect all historical predictions that cover step t
                actions_for_t = []
                for (base_t, chunk_pred) in self._pred_history:
                    offset = t - base_t
                    if 0 <= offset < chunk:
                        actions_for_t.append(chunk_pred[offset])
                # DEBUG: 每50步打印一次 populated 数量
                if t % 50 == 0:
                    self._dbg(f"[ACT DEBUG t={t}] populated={len(actions_for_t)}/{min(t+1, chunk)}")
                if len(actions_for_t) == 0:
                    raw = all_actions[0, 0]
                else:
                    actions_for_t = np.array(actions_for_t)
                    k = 0.01
                    weights = np.exp(-k * np.arange(len(actions_for_t)))
                    weights /= weights.sum()
                    raw = (actions_for_t * weights[:, np.newaxis]).sum(0)  # (26,)
            else:
                raw = self._chunk_actions[0, t % query_frequency]         # (26,)

        action = self._denormalize_action(raw)
        # Track commanded EE so next step's qpos obs stays self-consistent
        self._last_l_hand = action[7:13].copy()
        self._last_r_hand = action[20:26].copy()
        return action
