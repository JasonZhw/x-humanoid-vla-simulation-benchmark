# project
from common.logger_loader import logger
from common.config_loader import config_loader
from common.x_registry import Registry
from robots.arm_base import ArmBase
from robots.cam.sim_cam import SimCam
from common.utils.transform_utils import StandaloneUtils
# isaac sim
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationActions
from isaacsim.core.prims import SingleXFormPrim
import isaacsim.core.utils.rotations as rotations_utils
from pxr import Gf, UsdPhysics
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import get_prim_at_path
# others
from typing import Optional
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

logger.debug("TianYi2_Brainco2 import passed")


@Registry.register("TianYi2_Brainco2")
class TianYi2_Brainco2(ArmBase):

    def __init__(
        self,
        init_position: Optional[npt.NDArray[np.float64]] = None,
        init_orientation: Optional[npt.NDArray[np.float64]] = None,
        l_arm_home_pose: Optional[npt.NDArray[np.float64]] = None,
        r_arm_home_pose: Optional[npt.NDArray[np.float64]] = None,
    ):
        logger.debug("in TianYi2_Brainco2 class")
        config_loader.load_robot_toml("TianYi2_Brainco2")
        super().__init__()
        self.left_recv_ee_positions = np.zeros(6)
        self.right_recv_ee_positions = np.zeros(6)
        logger.debug("humanoid base class passed")
        self.init_position = init_position if init_position is not None else config_loader.robot_config["robot"][
            "config"]["init_position"]
        self.init_orientation = init_orientation if init_orientation is not None else config_loader.robot_config[
            "robot"]["config"]["init_orientation"]
        self.l_arm_home_pose = l_arm_home_pose if l_arm_home_pose is not None else config_loader.robot_config["robot"][
            "config"]["l_arm_home_pose"]
        self.r_arm_home_pose = r_arm_home_pose if r_arm_home_pose is not None else config_loader.robot_config["robot"][
            "config"]["r_arm_home_pose"]
        logger.debug(self.init_position)
        logger.debug(self.init_orientation)
        logger.debug("humanoid base class passed")
        # Insert robot
        add_reference_to_stage(self.robot_usd_path, self.NAMESPACE)
        logger.debug("robot already in stage")
        # Update robot world pose.
        self.change_pose(self.init_position, self.init_orientation, True)
        logger.debug("change pose")
        self.robot_prim = SingleXFormPrim(prim_path=self.NAMESPACE,
                                          position=self.init_position,
                                          orientation=rotations_utils.euler_angles_to_quat(self.init_orientation, True))
        self.left_ee_prim = SingleXFormPrim(self.NAMESPACE + "/revo2_left_hand_v1/left_base_link")
        self.right_ee_prim = SingleXFormPrim(self.NAMESPACE + "/revo2_right_hand_v1/right_base_link")
        self.base_prim = SingleXFormPrim(self.NAMESPACE + "/body_yaw_link")
        self.init_cam()
        logger.success("init TianYi2_Brainco2")

    def change_pose(self, pos, rot, degrees=True) -> None:
        #pelvis
        pelvis_joint_prim = get_prim_at_path(self.NAMESPACE + "/base/FixedJoint")
        pelvis_joint = UsdPhysics.Joint(pelvis_joint_prim)
        pelvis_joint.GetLocalRot0Attr().Set(
            Gf.Quatf(*rotations_utils.euler_angles_to_quat(rot, degrees=degrees).astype(float)))
        pelvis_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*np.array(pos, dtype=float)))

    def init_cam(self):
        self.cam_dict = {}
        for cam_name in config_loader.robot_config['cam']:
            self.cam_dict[cam_name] = SimCam(
                resolution=tuple(config_loader.robot_config['cam'][cam_name]['cam_resolution']),
                frequency=int(config_loader.robot_config['cam'][cam_name]['frequency']),
                focal_length=float(config_loader.robot_config['cam'][cam_name]['focal_length']),
                clip_range=(0.01, 10.0),
                cam_name=self.NAMESPACE + config_loader.robot_config['cam'][cam_name]['base_link'],
                translation=config_loader.robot_config['cam'][cam_name]['position'],
                orientation=config_loader.robot_config['cam'][cam_name]['orientation'],
                camera_matrix=config_loader.robot_config['cam'][cam_name]['camera_matrix'],
                fake_resolution=[640,480],
                match_cfg_quat_to_isaac_ui=True)
        logger.debug("camera inited")

    def active_art(self):
        self.art = Articulation(self.NAMESPACE + config_loader.robot_config["robot"]["config"]["art_prim"])
        if self.art.is_non_root_articulation_link == True:
            logger.error("robot is not an articulation")
        self.art.initialize()
        self.left_arm_joint_handles = [
            self.art.get_dof_index(name) for name in config_loader.robot_config["robot"]["joint"]["l_arm_joint_name"]
        ]
        self.right_arm_joint_handles = [
            self.art.get_dof_index(name) for name in config_loader.robot_config["robot"]["joint"]["r_arm_joint_name"]
        ]
        self.left_ee_joint_handles = [
            self.art.get_dof_index(name) for name in config_loader.robot_config["robot"]["joint"]["l_ee_joint_name"]
        ]
        self.right_ee_joint_handles = [
            self.art.get_dof_index(name) for name in config_loader.robot_config["robot"]["joint"]["r_ee_joint_name"]
        ]
        logger.debug("joints activated")

    def update_command(self, command_dict):
        self.left_recv_arm_positions = command_dict["left_arm"]
        self.right_recv_arm_positions = command_dict["right_arm"]
        self.left_recv_ee_positions = command_dict["left_hand"]
        self.right_recv_ee_positions = command_dict["right_hand"]

    def pub_master_l_r_joints(self, curr_time):
        return {
            "arm_left_position_raw": {
                "timestamp": curr_time,
                "is_intervene": False,
                "data": self.left_recv_arm_positions
            },
            "arm_right_position_raw": {
                "timestamp": curr_time,
                "is_intervene": False,
                "data": self.right_recv_arm_positions
            },
            "end_effector_left_position_raw": {
                "timestamp": curr_time,
                "is_intervene": False,
                "data": self.left_recv_ee_positions
            },
            "end_effector_right_position_raw": {
                "timestamp": curr_time,
                "is_intervene": False,
                "data": self.right_recv_ee_positions
            },
        }

    def pub_l_r_joints(self, curr_time):
        positions = self.art.get_joint_positions().squeeze()
        return {
            "arm_left_position_raw": {
                "timestamp":
                    curr_time,
                "is_intervene":
                    False,
                "data": [
                    positions[14], positions[17], positions[20], positions[23], positions[25], positions[27],
                    positions[29]
                ]
            },
            "arm_right_position_raw": {
                "timestamp":
                    curr_time,
                "is_intervene":
                    False,
                "data": [
                    positions[15], positions[18], positions[21], positions[24], positions[26], positions[28],
                    positions[30]
                ]
            },
            "end_effector_left_position_raw": {
                "timestamp":
                    curr_time,
                "is_intervene":
                    False,
                "data": [
                    positions[35],
                    positions[45],
                    positions[31],
                    positions[32],
                    positions[34],
                    positions[33],
                ]
            },
            "end_effector_right_position_raw": {
                "timestamp":
                    curr_time,
                "is_intervene":
                    False,
                "data": [
                    positions[40],
                    positions[50],
                    positions[36],
                    positions[37],
                    positions[39],
                    positions[38],
                ]
            },
            "end_effector_left_pose_raw": {
                "timestamp":
                    curr_time,
                "is_intervene":
                    False,
                "data":
                    StandaloneUtils.affine_to_xyz_quaternion(
                        StandaloneUtils.transform_xfroms_pose(self.left_ee_prim, self.base_prim))
            },
            "end_effector_right_pose_raw": {
                "timestamp":
                    curr_time,
                "is_intervene":
                    False,
                "data":
                    StandaloneUtils.affine_to_xyz_quaternion(
                        StandaloneUtils.transform_xfroms_pose(self.right_ee_prim, self.base_prim))
            },
        }

    def joint_callback(self, step_size) -> None:
        # Arm control in base class
        for idx in range(self.num_arm_dof):
            self.art.apply_action(
                ArticulationActions(joint_positions=self.left_recv_arm_positions[idx],
                                    joint_indices=self.left_arm_joint_handles[idx]))
            self.art.apply_action(
                ArticulationActions(joint_positions=self.right_recv_arm_positions[idx],
                                    joint_indices=self.right_arm_joint_handles[idx]))

        for i in range(6):
                    self.art.apply_action(
                        ArticulationActions(joint_positions=self.left_recv_ee_positions[i], joint_indices=self.left_ee_joint_handles[i]))
                    self.art.apply_action(
                        ArticulationActions(joint_positions=self.right_recv_ee_positions[i], joint_indices=self.right_ee_joint_handles[i]))
