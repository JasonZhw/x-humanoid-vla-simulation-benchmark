# TianYi2_Brainco2 任务基类

import sys
import os
# append to sys path to allow absolute project imports
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)
# isaac sim
from isaacsim.simulation_app import SimulationApp
import omni
# project
from common.logger_loader import logger
from common.config_loader import config_loader
from common.utils.p2p_traj import P2P_Trajectory
from common.utils.zmq_utils import ZmqPublisher, ZmqReceiver, eval_endpoints_from_env
from common.utils.video_recorder import EpisodeVideoRecorder
from tasks.task_base import TaskRunnerBase
# others
import numpy as np
import time
import signal
import tqdm


class TianYi2_Brainco2_Task_Base(TaskRunnerBase):
    """TianYi2_Brainco2 task base class providing common task behaviors for 01/02/03/04.

    Subclasses should:
    - Create and assign self.robot in their __init__ using Registry.create("TianYi2_Brainco2", ...)
    - Optionally override init_play if they need custom warm-up sequences
    """

    TIER: int = 1  # 子类重写：Tier 2 任务设为 2

    def __init__(self,
                 simulation_app: SimulationApp,
                 physics_dt: float = 1.0 / 120,
                 render_dt: float = 1.0 / 120,
                 stage_units_in_meters: float = 1.00,
                 environment_path: str = None,
                 episode_id: int = 0,
                 enable_gpu: bool = False,
                 task_name: str = '',
                 condition: str = 'standard',
                 baseline: str = '',
                 seed: int = 0):
        logger.debug("TianYi2_Brainco2_Task_Base init")
        super().__init__(simulation_app=simulation_app,
                         physics_dt=physics_dt,
                         render_dt=render_dt,
                         stage_units_in_meters=stage_units_in_meters,
                         environment_path=environment_path,
                         enable_gpu=enable_gpu)

        # Basic runtime states
        self.start_flag = False
        self.episode_id = episode_id
        self.task_name = task_name
        self.condition = condition
        self.baseline = baseline
        self.seed = seed
        self._episode_start_time = time.time()
        self.current_step = 0
        # ZMQ client
        _obs_port, _action_port, _infer_host = eval_endpoints_from_env()
        self.zmq_publisher = ZmqPublisher(port=_obs_port)                       # obs PUB, bind 0.0.0.0
        self.zmq_receiver  = ZmqReceiver(port=_action_port, host=_infer_host)   # action SUB, connect 选手机
        self.l_arm_init_pose = config_loader.task_config["robot"]["l_arm_init_pose"]
        self.r_arm_init_pose = config_loader.task_config["robot"]["r_arm_init_pose"]
        self.l_arm_home_pose = config_loader.task_config["robot"]["l_arm_home_pose"]
        self.r_arm_home_pose = config_loader.task_config["robot"]["r_arm_home_pose"]
        video_path = os.path.join(
            "logs", "task_videos",
            f"{self.task_name}_{time.strftime('%Y%m%d_%H%M%S')}",
            f"loop{self.episode_id}.mp4")
        self.video_recorder = EpisodeVideoRecorder(video_path)
        self.record_video = os.environ.get("RECORD_VIDEO", "1") != "0"
        signal.signal(signal.SIGUSR1, self._release_video_on_signal)
        logger.debug(f"l_arm_home_pose: {self.l_arm_home_pose}")
        logger.debug(f"r_arm_home_pose: {self.r_arm_home_pose}")
        logger.success('TianYi2_Brainco2_Task_Base initialized')

    def init_play(self, step_num: int):
        """Common warm-up and callback registration, then go to home pose."""
        for _ in tqdm.tqdm(range(step_num)):
            self.one_step()
        # Activate joints
        self.robot.active_art()
        logger.info("art activated")
        # Add callbacks
        self.physics_callback_dict = {
            'execute_joint': self.robot.joint_callback,
            'pub_joint': self.collect_data_callback,
            'update_joint': self.update_joint_callback,
        }
        for physics_callback_name, physics_callback_fn in self.physics_callback_dict.items():
            self.simulation_context.add_physics_callback(physics_callback_name, callback_fn=physics_callback_fn)
        
        self.to_arm_init_pose()
        for _ in tqdm.tqdm(range(5)):
            self.one_step()
        self.to_home_pose()
        logger.success("init play done")

    def update_joint_callback(self, step_size) -> None:
        if self.start_flag == True:
            envelope = self.zmq_receiver.receive_envelope(timeout=10)
            if envelope is None:
                return
            if self.zmq_receiver.is_old_action(envelope):
                return
            topic = str(envelope.get("topic", "")).encode("utf-8")
            data = envelope.get("payload")
            if topic == b"action" and data is not None:
                self.robot.update_command(data)

    def collect_data_callback(self, step_size) -> None:
        self.sim_step += 1
        if self.sim_step % 4 == 0:
            curr_time = int(omni.timeline.get_timeline_interface().get_current_time() * 1000)
            self.buffer_pool_align["puppet"] = self.robot.pub_l_r_joints(curr_time)
            self.buffer_pool_align["camera_observations"]["timestamp"] = curr_time
            for cam_name in self.robot.cam_dict:
                self.robot.cam_dict[cam_name].get_rgb(
                    out_buffer=self.buffer_pool_align["camera_observations"]["color_images"][cam_name])
                # 填充深度（供选手部署 RGBD 模型）：distance_to_image_plane，单位 mm，shape (H,W) float32。
                # 不填则 depth_images 恒为预分配的全 0（buffer 见 tasks/task_base.py）。
                self.robot.cam_dict[cam_name].get_depth(
                    out_buffer=self.buffer_pool_align["camera_observations"]["depth_images"][cam_name])
            if self.start_flag == True:
                self.zmq_publisher.send_msg(data=self.buffer_pool_align,
                                            topic=b"obs",
                                            episode_id=self.episode_id,
                                            step_id=self.current_step)
                if self.record_video:
                    self.video_recorder.add_frame(self.buffer_pool_align)

    def to_home_pose(self):
        # pass
        left_arm_motion = []
        right_arm_motion = []
        time_duration = 2
        steps = int(time_duration / self.physics_dt)
        for idx in range(self.robot.num_arm_dof):
            left_arm_motion.append(
                P2P_Trajectory(start_p=self.l_arm_init_pose[idx],
                               end_p=self.l_arm_home_pose[idx],
                               start_v=0.0,
                               end_v=0.0,
                               start_a=0.0,
                               end_a=0.0,
                               start_t=0,
                               end_t=time_duration))
            right_arm_motion.append(
                P2P_Trajectory(start_p=self.r_arm_init_pose[idx],
                               end_p=self.r_arm_home_pose[idx],
                               start_v=0.0,
                               end_v=0.0,
                               start_a=0.0,
                               end_a=0.0,
                               start_t=0,
                               end_t=time_duration))

        # Hand home pose: matches training data initial ee positions (EE_DIMS-sliced from 12-dim)
        l_hand_home = np.array([1.571, -0.093, -0.172, -0.172, -0.173, -0.173])
        r_hand_home = np.array([1.571, 0.079, -0.179, -0.175, -0.178, -0.177])
        for step in range(steps):
            cmd_left_positions = np.zeros(self.robot.num_arm_dof)
            cmd_right_positions = np.zeros(self.robot.num_arm_dof)
            for idx in range(self.robot.num_arm_dof):
                cmd_left_positions[idx] = left_arm_motion[idx].get_point(step * self.physics_dt)[0]
                cmd_right_positions[idx] = right_arm_motion[idx].get_point(step * self.physics_dt)[0]
            self.robot.update_command({
                "left_arm": cmd_left_positions,
                "right_arm": cmd_right_positions,
                "left_hand": l_hand_home,
                "right_hand": r_hand_home,
            })
            self.one_step()

    def play(self, num_steps: int = 60000) -> None:
        super().play()
        self.init_play(step_num=50)
        self.start_flag = True
        logger.debug("start sending obs")
        for i in range(num_steps):
            self.current_step = i
            self.one_step()
        self.stop()

    def start(self):
        logger.success("Attempt to start sim")
        for _ in range(10):
            self.zmq_publisher.send_msg(data=None, topic=b"test", episode_id=self.episode_id, step_id=self.current_step)
            time.sleep(0.1)
            recv_msg = self.zmq_receiver.receive_msg(timeout=10)
            if recv_msg is not None and recv_msg[0] == b"test":
                logger.info("Sim recv func warmed up")
        # add depth to annotator registry
        for cam_name in self.robot.cam_dict:
            self.robot.cam_dict[cam_name].init_depth()
        omni.timeline.get_timeline_interface().set_current_time(0)
        self.sim_step = 0
        self.zmq_publisher.send_msg(data=b"TienKung", topic=b"start", episode_id=self.episode_id, step_id=self.current_step)
        if self.record_video:
            self.video_recorder.start_recording()

    def _release_video_on_signal(self, signum, frame):
        logger.info("Received SIGUSR1, releasing video writer")
        self.video_recorder.stop_recording()

    def stop(self):
        self.video_recorder.stop_recording()
        self.zmq_publisher.send_msg(data=None, topic=b"reset", episode_id=self.episode_id, step_id=self.current_step)
        self.shut_down()

    def to_arm_init_pose(self):
        left_arm_motion = []
        right_arm_motion = []
        time_duration = 2
        steps = int(time_duration / self.physics_dt)
        for idx in range(self.robot.num_arm_dof):
            left_arm_motion.append(
                P2P_Trajectory(start_p=0.0,
                               end_p=self.l_arm_init_pose[idx],
                               start_v=0.0,
                               end_v=0.0,
                               start_a=0.0,
                               end_a=0.0,
                               start_t=0,
                               end_t=time_duration))
            right_arm_motion.append(
                P2P_Trajectory(start_p=0.0,
                               end_p=self.r_arm_init_pose[idx],
                               start_v=0.0,
                               end_v=0.0,
                               start_a=0.0,
                               end_a=0.0,
                               start_t=0,
                               end_t=time_duration))

        l_hand_home = np.array([1.571, -0.093, -0.172, -0.172, -0.173, -0.173])
        r_hand_home = np.array([-1.571, 0.079, -0.179, -0.175, -0.178, -0.177])
        for step in range(steps):
            cmd_left_positions = np.zeros(self.robot.num_arm_dof)
            cmd_right_positions = np.zeros(self.robot.num_arm_dof)
            for idx in range(self.robot.num_arm_dof):
                cmd_left_positions[idx] = left_arm_motion[idx].get_point(step * self.physics_dt)[0]
                cmd_right_positions[idx] = right_arm_motion[idx].get_point(step * self.physics_dt)[0]
            self.robot.update_command({
                "left_arm": cmd_left_positions,
                "right_arm": cmd_right_positions,
                "left_hand": l_hand_home,
                "right_hand": r_hand_home,
            })
            self.one_step()
