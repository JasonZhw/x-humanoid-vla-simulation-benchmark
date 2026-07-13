#!/usr/bin/env python3
"""
Task startup script
Used to start specified tasks through the Registry system
"""

import sys
import os
import argparse

# Add project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _ensure_project_on_path() -> None:
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


_ensure_project_on_path()

from isaacsim.simulation_app import SimulationApp


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run specified task')
    parser.add_argument('--task', type=str, required=True, help='Task name')
    parser.add_argument('--steps', type=int, default=60000, help='Simulation steps')
    parser.add_argument('--headless', default=False, action='store_true', help='Run in headless mode')
    parser.add_argument('--usd-path', type=str, help='USD environment path to use for the task')
    parser.add_argument('--episode-id', type=int, default=0, help='Episode id for communication tracking')
    parser.add_argument('--condition', type=str, default='standard',
                        help='Experimental condition')
    parser.add_argument('--baseline', type=str, default='',
                        help='Baseline method name')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    return parser.parse_args()


def main():
    """Main function"""
    try:
        args = parse_arguments()

        # Configure Isaac Sim
        config = {
            "width": 1280,
            "height": 720,
            "sync_loads": False,
            "headless": bool(args.headless),
            "renderer": "RayTracedLighting",
            "anti_aliasing": 3,
        }
        # Start SimulationApp
        simulation_app = SimulationApp(config)
        # SimulationApp may reorder sys.path; re-insert project root before imports.
        _ensure_project_on_path()
        from common.logger_loader import logger
        from common.config_loader import config_loader
        from common.x_registry import Registry

        # Import task modules (must be after SimulationApp startup)
        import robots
        import tasks.TianYi2_Brainco2_tasks

        config_loader.load_task_toml(args.task)
        task_config = config_loader.task_config

        # Prepare task parameters
        # Use command line USD path if provided, otherwise auto-select from task config
        environment_path = getattr(args, 'usd_path', None)
        if environment_path is None:
            from common.utils.usd_selector import select_random_usd_path
            environment_path = select_random_usd_path()
            logger.info(f"Auto-selected USD environment: {environment_path}")
        task_params = {
            'simulation_app': simulation_app,
            'environment_path': environment_path,
            'robot_init_position': task_config.get('robot', {}).get('init_position'),
            'robot_init_orientation': task_config.get('robot', {}).get('init_orientation'),
            'episode_id': getattr(args, 'episode_id', 0),
            'task_name': args.task,
            'condition': getattr(args, 'condition', 'standard'),
            'baseline': getattr(args, 'baseline', ''),
            'seed': getattr(args, 'seed', 0),
        }

        task_instance = Registry.create(args.task, **task_params)
        logger.info(f"Task instance created successfully: {type(task_instance).__name__}")
        if task_instance is None:
            logger.error(f"Unable to create task instance: {args.task}")
            return 1

        logger.info(f"Task instance created successfully: {type(task_instance).__name__}")

        # Start task
        task_instance.start()

        # # Run simulation
        task_instance.play(num_steps=args.steps)

        logger.success(f"Task {args.task} completed")
        return 0

    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Ensure SimulationApp is properly closed
        try:
            simulation_app.close()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
