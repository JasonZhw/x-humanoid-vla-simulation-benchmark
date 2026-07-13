import sys
import os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)
from isaacsim.simulation_app import SimulationApp
import carb
import omni.kit.commands
from isaacsim.core.utils.stage import is_stage_loading
from isaacsim.core.prims import SingleXFormPrim
from common.logger_loader import logger
from common.config_loader import config_loader
from common.x_registry import Registry
from tasks.TianYi2_Brainco2_tasks.TianYi2_Brainco2_task_base import TianYi2_Brainco2_Task_Base
# others
import numpy as np
from pxr import Sdf
# settings
np.set_printoptions(precision=4)
carb.settings.get_settings().set("persistent/app/omniverse/gamepadCameraControl", False)
logger.info("TianYi2_Brainco2_lab_task_03 imports all deps")


@Registry.register("TianYi2_Brainco2_lab_task_03")
class TianYi2_Brainco2_lab_task_03(TianYi2_Brainco2_Task_Base):

    TIER = 1

    def __init__(
        self,
        simulation_app: SimulationApp,
        physics_dt=1.0 / 120,
        render_dt=1.0 / 120,
        stage_units_in_meters=1.00,
        environment_path=None,
        robot_init_position=None,
        robot_init_orientation=None,
        episode_id=0,
        task_name='',
        condition='standard',
        baseline='',
        seed=0,
    ):
        logger.debug("TianYi2_Brainco2 runner in")
        super().__init__(simulation_app=simulation_app,
                         physics_dt=physics_dt,
                         render_dt=render_dt,
                         stage_units_in_meters=stage_units_in_meters,
                         environment_path=environment_path,
                         episode_id=episode_id,
                         task_name=task_name,
                         condition=condition,
                         baseline=baseline,
                         seed=seed)
        # Set rendering
        # Smaller number reduces material resolution to avoid out-of-memory, max is 15
        carb.settings.get_settings().set("/rtx-transient/resourcemanager/maxMipCount", 15)
        # Enable DLSS
        carb.settings.get_settings().set("/rtx-transient/dlssg/enabled", True)
        # "Auto": 3, "Performance": 0, "Balanced": 1, "Quality": 2
        carb.settings.get_settings().set("/rtx/post/dlss/execMode", 0)
        logger.info("rendering mode all set")
        # Tiangong
        self.robot = Registry.create(
            "TianYi2_Brainco2",
            init_position=robot_init_position,
            init_orientation=robot_init_orientation,
            l_arm_home_pose=self.l_arm_home_pose,
            r_arm_home_pose=self.r_arm_home_pose,
        )
        logger.info('TianYi2_Brainco2 initialized.')
        while is_stage_loading():
            simulation_app.update()
        self.init_states()
        logger.success('TianYi2_Brainco2 runner initialized')
