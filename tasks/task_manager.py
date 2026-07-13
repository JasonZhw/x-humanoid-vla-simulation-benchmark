#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Management Module
Responsible for task configuration reading and execution control (process spawn/timeout).
"""
import os
import sys
import time
import random
import subprocess
import signal
from datetime import datetime
from typing import Optional

# Define project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root directory to Python path
sys.path.append(PROJECT_ROOT)
from common.logger_loader import logger
from common.config_loader import config_loader
from common.utils.usd_selector import select_random_usd_path


class BenchmarkTaskManager:
    """Task Manager, responsible for task configuration and execution"""

    def __init__(self, task_name: str, condition: str = 'standard', baseline: str = '', seed: int = 0):
        """
        Initialize task manager

        Args:
            task_name: Task name (e.g.: TienKung_task_03)
        """
        self.task_name = task_name
        # Process management
        self.task_process = None
        self.is_task_running = False
        self.log_file = None
        # Keep the path of the Isaac Sim log file so we can tail it on errors
        self.log_file_path = None
        # Batch timestamp for grouping logs from same batch
        self.batch_timestamp = None
        self.current_usd_path = None
        self.current_episode_id: int = 0
        self.condition: str = condition
        self.baseline: str = baseline
        self.seed: int = seed
        self._episode_usd_list: list = []  # pre-built per-run scene list

    def init_batch_timestamp(self):
        """Initialize batch timestamp for grouping logs from same batch"""
        if self.batch_timestamp is None:
            self.batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_all_usds_from_dir(self, dir_path: str) -> list:
        """从目录返回所有 .usd 文件的绝对路径列表（排序后稳定）。"""
        abs_dir = dir_path if os.path.isabs(dir_path) else os.path.join(PROJECT_ROOT, dir_path)
        usd_files = sorted(f for f in os.listdir(abs_dir) if f.endswith('.usd'))
        if not usd_files:
            logger.warning(f"No .usd files found in {abs_dir}")
            return []
        return [os.path.join(abs_dir, f) for f in usd_files]

    def _get_all_usds_for_condition(self) -> list:
        """返回该任务 USD 目录下的全部场景。评测统一使用 usd_path_spatial 目录，
        不再区分 condition（standard/lighting/texture 已停用）。需先调用 config_loader.load_task_toml。"""
        env_cfg = config_loader.task_config.get('environment', {})
        path_cfg = env_cfg.get('usd_path_spatial') or env_cfg.get('usd_path')
        if not path_cfg:
            return []
        return self._get_all_usds_from_dir(path_cfg)

    def prepare_episode_usds(self, n: int) -> None:
        """运行前预建 N 个 episode 的 USD 列表（seed 控制采样顺序，可复现）。

        - n <= 可用场景数：无放回采样（每个场景最多出现一次）
        - n >  可用场景数：循环补足（多轮 shuffle 拼接）
        需先调用 config_loader.load_task_toml(self.task_name)。
        """
        config_loader.load_task_toml(self.task_name)
        all_usds = self._get_all_usds_for_condition()
        if not all_usds:
            logger.warning(f"No USDs found for condition '{self.condition}'")
            self._episode_usd_list = [None] * n
            return

        rng = random.Random(self.seed)
        if n <= len(all_usds):
            self._episode_usd_list = rng.sample(all_usds, n)
        else:
            result = []
            pool = all_usds[:]
            while len(result) < n:
                rng.shuffle(pool)
                result.extend(pool)
            self._episode_usd_list = result[:n]

        logger.info(f"[{self.condition}] Prepared {len(self._episode_usd_list)} episode USDs "
                    f"from {len(all_usds)} available (seed={self.seed})")

    def _resolve_usd_for_condition(self) -> Optional[str]:
        """单次随机选取 USD（fallback，prepare_episode_usds 未调用时使用）。"""
        all_usds = self._get_all_usds_for_condition()
        if not all_usds:
            return None
        return random.choice(all_usds)

    def start_task_process(self, num_steps: int = 60000, headless: bool = False, episode_id: int = 0) -> bool:
        """Start task process"""
        try:
            if self.is_task_running:
                logger.warning("Task process is already running")
                return True

            logger.info(f"Starting task process: {self.task_name}")

            # Build task startup script path
            run_script_path = os.path.join(PROJECT_ROOT, "tasks", "run_task.py")

            if not os.path.exists(run_script_path):
                logger.error(f"Task startup script does not exist: {run_script_path}")
                return False

            # Get Isaac Sim Python path
            isaac_python_path = config_loader.get_isaac_python_path()
            # Start subprocess
            cmd = [isaac_python_path, run_script_path, "--task", self.task_name,
                   "--steps", str(num_steps), "--episode-id", str(episode_id),
                   "--condition", self.condition,
                   "--baseline", self.baseline,
                   "--seed", str(self.seed)]
            if headless:
                cmd.append("--headless")
            if self.current_usd_path:
                cmd.extend(["--usd-path", self.current_usd_path])

            # Create timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Ensure logs/isaac_sim directory exists
            logs_dir = os.path.join(PROJECT_ROOT, "logs", "isaac_sim")
            os.makedirs(logs_dir, exist_ok=True)

            log_file_path = os.path.join(logs_dir, f"isaac_sim_{self.task_name}_{timestamp}.log")
            log_file = open(log_file_path, "w")

            # 评测只跑 spatial：不做环境随机化，确保子进程不继承父进程可能残留的 ENV_RAND
            proc_env = {**os.environ}
            proc_env.pop('ENV_RAND', None)
            existing_pythonpath = proc_env.get("PYTHONPATH", "")
            proc_env["PYTHONPATH"] = (
                PROJECT_ROOT if not existing_pythonpath
                else f"{PROJECT_ROOT}:{existing_pythonpath}"
            )

            self.task_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                env=proc_env,
                preexec_fn=os.setsid  # Create new process group
            )

            # Save log file reference for later closure
            self.log_file = log_file
            self.log_file_path = log_file_path
            logger.info(f"Isaac Sim log output to: {log_file_path}")

            self.is_task_running = True

            logger.success(f"Task process started successfully, PID: {self.task_process.pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start task process: {e}")
            return False

    def _close_log_file(self):
        if self.log_file is None:
            return
        try:
            self.log_file.flush()
            self.log_file.close()
        except Exception as e:
            logger.warning(f"Failed to close log file: {e}")
        self.log_file = None

    def _release_task_video_recorder(self):
        """Ask task subprocess to release video writer before termination."""
        if self.task_process is None or self.task_process.poll() is not None:
            return
        try:
            logger.info("Requesting video recorder release before stopping task process...")
            os.killpg(os.getpgid(self.task_process.pid), signal.SIGUSR1)
            time.sleep(2)
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.warning(f"Failed to release video recorder: {e}")

    def stop_task_process(self) -> bool:
        """Stop task process"""
        try:
            if self.task_process is None:
                return True

            if self.task_process.poll() is None:
                logger.info("Stopping task process...")
                try:
                    self._release_task_video_recorder()
                    os.killpg(os.getpgid(self.task_process.pid), signal.SIGTERM)
                    try:
                        self.task_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning("Process did not end within 10 seconds, force terminating")
                        os.killpg(os.getpgid(self.task_process.pid), signal.SIGKILL)
                        self.task_process.wait(timeout=5)
                except ProcessLookupError:
                    pass
            else:
                try:
                    self.task_process.wait(timeout=1)
                except Exception:
                    pass

            self.is_task_running = False
            self.task_process = None
            self._close_log_file()

            logger.success("Task process has been stopped")
            time.sleep(1)  # allow OS to release ZMQ ports
            return True

        except Exception as e:
            logger.error(f"Failed to stop task process: {e}")
            return False

    def run_task(self, loop_idx: int = 0, num_steps: int = 60000, headless: bool = False, timeout: int = 300) -> bool:
        """Run one practice episode: select USD, spawn sim process, enforce timeout."""
        self.current_episode_id = loop_idx
        episode_id = loop_idx

        # Select USD for this episode: use pre-built list if available, else fallback to random
        if self._episode_usd_list and loop_idx < len(self._episode_usd_list):
            self.current_usd_path = self._episode_usd_list[loop_idx]
            logger.info(f"[{self.condition}] Episode {loop_idx}: {self.current_usd_path}")
        else:
            config_loader.load_task_toml(self.task_name)
            self.current_usd_path = self._resolve_usd_for_condition()
            if self.current_usd_path:
                logger.info(f"[{self.condition}] Selected USD: {self.current_usd_path}")
            else:
                logger.warning(f"No USD path configured for task: {self.task_name}")

        logger.info(f"Starting episode: {self.task_name} (iteration {loop_idx + 1})")
        if not self.start_task_process(num_steps, headless, episode_id):
            logger.error("Failed to start task process")
            return False
        try:
            self.task_process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.info(f"Episode reached time limit ({timeout}s), moving on")
        finally:
            # Guarantee the process/log-file are released before the next episode starts,
            # whether this episode finished on its own or hit the timeout above.
            self.stop_task_process()
        return True

    def is_process_running(self) -> bool:
        """Check if task process is running"""
        return self.is_task_running and self.task_process is not None

    def cleanup(self) -> bool:
        """Clean up task manager"""
        try:
            logger.info("Cleaning up task manager...")

            # Stop task process
            if self.is_task_running:
                self.stop_task_process()

            self._close_log_file()

            # Reset state
            self.task_process = None
            self.is_task_running = False
            self.log_file = None
            self.log_file_path = None

            logger.success("Task manager cleanup completed")
            return True

        except Exception as e:
            logger.error(f"Failed to clean up task manager: {e}")
            return False


TaskManager = BenchmarkTaskManager
