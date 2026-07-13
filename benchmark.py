#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark main entry file
Robot evaluation system based on Isaac Sim
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent))

from common.logger_loader import logger
from common.config_loader import config_loader
from tasks.task_manager import TaskManager
from common.utils.zmq_utils import ZmqPublisher, ZmqReceiver, eval_endpoints_from_env

TaskNameMapping = {
    "ind_task_01":"TianYi2_Brainco2_ind_task_01",
    "ind_task_02":"TianYi2_Brainco2_ind_task_02",
    "ind_task_03":"TianYi2_Brainco2_ind_task_03",
    "lab_task_01":"TianYi2_Brainco2_lab_task_01",
    "lab_task_03":"TianYi2_Brainco2_lab_task_03",
}

class BenchmarkRunner:
    """Benchmark evaluation system main controller"""

    def __init__(self):
        self.task_manager = None

    def parse_arguments(self, task_override: Optional[str] = None) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description='Isaac Sim robot evaluation system',
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog="""
            Example usage:
            python benchmark.py --task tasks/pick_apple.toml --model model1.pth,model2.pth --loop 10
            python benchmark.py --task tasks/pick_apple.toml --model model1.pth --loop 5 --headless
            """)

        parser.add_argument('--task', type=str, required=(task_override is None), help='task name')

        parser.add_argument('--loop', type=int, default=1, help='Number of loop tests (default: 1)')

        parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI)')

        parser.add_argument('--output', type=str, default='./logs', help='Result output directory (default: ./logs)')

        parser.add_argument('--zmq-port', type=int, default=5555, help='ZMQ communication port (default: 5555)')

        parser.add_argument('--verbose', action='store_true', help='Verbose log output')

        parser.add_argument('--timeout', type=int, default=300, help='Task execution timeout in seconds (default: 300)')

        parser.add_argument(
            '--condition',
            type=str,
            default='spatial',
            choices=['spatial'],
            help='评测统一使用 spatial：直接读任务 USD 目录、无环境随机化（默认，可不填）'
        )
        parser.add_argument(
            '--baseline',
            type=str,
            default='',
            help='Baseline method name (e.g. ACT)'
        )
        parser.add_argument(
            '--seed',
            type=int,
            default=0,
            help='Random seed for this run (default: 0)'
        )

        args = parser.parse_args()
        if task_override is not None:
            args.task = task_override
        return args

    def validate_args(self, args: argparse.Namespace) -> bool:
        """Validate command line arguments"""
        # Validate task file exists
        if not config_loader.check_task_toml(args.task):
            logger.error(f"Task configuration file does not exist: {args.task}")
            return False

        # Validate loop count
        if args.loop <= 0:
            logger.error(f"Loop count must be greater than 0: {args.loop}")
            return False

        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        return True

    def initialize_components(self, args: argparse.Namespace) -> bool:
        """Initialize components"""
        try:
            # Initialize task manager (pass task name)
            # Extract task name from config file path
            self.task_manager = TaskManager(
                args.task,
                condition=args.condition,
                baseline=args.baseline,
                seed=args.seed,
            )
            logger.info("Component initialization completed")
            return True

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False

    def run_benchmark(self, args: argparse.Namespace) -> bool:
        """Run practice episodes (no scoring — official evaluation is organizer-side)."""
        logger.info(f"Starting practice run - Task: {args.task}, Episodes: {args.loop}")
        self.task_manager.prepare_episode_usds(args.loop)

        for loop_idx in range(args.loop):
            logger.info(f"\n--- Episode {loop_idx + 1}/{args.loop} ---")
            try:
                self.task_manager.run_task(loop_idx=loop_idx, headless=args.headless,
                                           timeout=args.timeout, num_steps=60000)
            except Exception as e:
                logger.error(f"Episode {loop_idx + 1} failed to run: {e}")
            logger.info(f"Episode {loop_idx + 1}/{args.loop} finished")

        logger.info(f"Practice run finished ({args.loop} episodes)")
        logger.info("Videos: logs/task_videos/  (one mp4 per episode)")
        return True

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.task_manager:
                self.task_manager.cleanup()
            logger.info("Resource cleanup completed")
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")

    def run(self, task_override: Optional[str] = None) -> int:
        """Main run function"""
        try:
            # Parse arguments
            args = self.parse_arguments(task_override=task_override)
            logger.debug("parse_arguments done")
            # Validate arguments
            if not self.validate_args(args):
                return 1
            logger.debug("validate_arguments done")
            # Initialize components
            if not self.initialize_components(args):
                return 1
            logger.debug("initialize_components done")
            # Run benchmark
            if not self.run_benchmark(args):
                return 1
            logger.debug("run_benchmark done")
            return 0

        except KeyboardInterrupt:
            logger.info("\nUser interrupted benchmark")
            return 1
        except Exception as e:
            logger.error(f"Benchmark run failed: {e}")
            return 1
        finally:
            self.cleanup()


def _task_name_from_payload(payload) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8")
    if isinstance(payload, str):
        return payload
    return str(payload)


def _wait_for_task_assignment(zmq_receiver: ZmqReceiver) -> str:
    """Wait until a ZMQ message with topic 'task' is received."""
    logger.info("Waiting for task assignment via ZMQ (topic=task)...")
    while True:
        envelope = zmq_receiver.receive_envelope(timeout=1000)
        if envelope is None:
            continue
        if str(envelope.get("topic", "")) == "task":
            task_name = _task_name_from_payload(envelope.get("payload"))
            logger.info(f"Received task: {task_name}")
            return task_name


def main(task_override: Optional[str] = None) -> int:
    """Main function"""
    runner = BenchmarkRunner()
    return runner.run(task_override=task_override)


if __name__ == '__main__':
    while True:
        obs_port, action_port, infer_host = eval_endpoints_from_env()
        logger.info(
            f"ZMQ task handshake — recv:{infer_host}:{action_port}  send:*:{obs_port}"
        )

        zmq_publisher = ZmqPublisher(port=obs_port)
        zmq_receiver = ZmqReceiver(port=action_port, host=infer_host)

        try:
            task_name = _wait_for_task_assignment(zmq_receiver)
            if task_name in TaskNameMapping:
                task_all_name = TaskNameMapping[task_name]
                zmq_publisher.send_msg(data=None, topic=b"task_cbd")
                logger.info("Sent task_cbd acknowledgment")
            else:
                logger.warning(f"Task name {task_name} not found in TaskNameMapping")
                continue
        finally:
            zmq_receiver.close()
            zmq_publisher.socket.close()

        exit_code = main(task_override=task_all_name)
        if exit_code != 0:
            logger.warning(f"Benchmark finished with exit code {exit_code}")
        logger.info("Waiting for next task assignment...")
