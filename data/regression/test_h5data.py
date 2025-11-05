import os

from isaaclab.app import AppLauncher
import argparse
os.environ["ENABLE_CAMERAS"] = "1"

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str,
                    default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments to spawn.")
# parser.add_argument("--enable_cameras", type=bool, default=True)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from IPython import embed
from omegaconf import DictConfig
import hydra
import os
import os.path as osp
import omni.replicator.core as rep
import isaacsim.core.utils.prims as prim_utils
from isaaclab.utils import convert_dict_to_backend
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
import numpy as np
from copy import deepcopy
import h5py
import json
import torch
import random
from matplotlib import pyplot as plt
import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils import quat_to_rotmat, transform_points_by_pose

def print_h5_tree(h5_path):
    def visitor(name, obj):
        # 计算缩进层级
        depth = name.count('/')
        indent = "  " * depth

        if isinstance(obj, h5py.Group):
            print(f"{indent}+ [G] {name}/")
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f"{indent}- [D] {name}    shape={shape} dtype={dtype}")

    with h5py.File(h5_path, 'r') as f:
        print(f"H5 file: {h5_path}")
        f.visititems(visitor)
        for img in f["rgb"]:
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.show()

TARGET_SCALE = 0.2
TARGET_INIT_POS = (0.5, 0.0, 0.2)
TARGET_INIT_ROT = (0.7, 0.0, 0.7, 0.0)

MY_NEW_OBJECT_CFG = ArticulationCfg(
    # 1. 指向你转换后的 USD 文件
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/hhhar/liuliu/vld/assets/articulation/100392/mobility_annotation_gapartnet/mobility_annotation_gapartnet.usd",  # <-- 替换成你的文件路径
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False
        ),
        scale=[TARGET_SCALE for _ in range(3)],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=TARGET_INIT_POS,
        rot=TARGET_INIT_ROT,),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],  # <-- 使用正则表达式匹配你的关节
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        )
    })

def _print(msg):
    print('----------------------------------------------------------------------')
    print(f"msg:{msg}")

bbox_pts_cfg = VisualizationMarkersCfg(
    prim_path="/World/Visuals/testMarkers",
    markers={
        "marker1": sim_utils.SphereCfg(
            radius=0.005,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)),
        ), })

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(
            f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

    my_new_articulation = MY_NEW_OBJECT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/MyNewObject"  # 给它一个唯一的场景路径
    )

class PlanningDemo:
    def __init__(self, data_prefix="/home/hhhar/liuliu/vld/data/regression/data_res"):
        # self.target_id = -1
        self.camera_id = -1
        self.bbox = []
        self.scaled_bbox = []
        self.poses = []
        self.actions = []
        self.rgbs = []
        self.depthes = []
        self.cam_mat = []
        self.action_type = []
        self.part_num = -1
        # self.part_id = -1
        self.data_prefix = data_prefix
    
    def load_h5data(self, h5_path):
        def visitor(name, obj):
            # 计算缩进层级
            depth = name.count('/')
            indent = "  " * depth

            if isinstance(obj, h5py.Group):
                print(f"{indent}+ [G] {name}/")
            elif isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype
                print(f"{indent}- [D] {name}    shape={shape} dtype={dtype}")

        with h5py.File(h5_path, 'r') as f:
            print(f"H5 file: {h5_path}")
            f.visititems(visitor)
            self.rgbs = f["rgb"][:]
            self.poses = f["poses"][:]
            self.scaled_bbox = f["scaled_bbox"][:]
    

    def run_simulator(self, sim: sim_utils.SimulationContext, scene: InteractiveScene):
        """Runs the simulation loop."""
        # Extract scene entities
        # note: we only do this here for readability.
        robot = scene["robot"]
        # self.camera = scene["camera"]
        # Create controller
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
        
        #load_h5
        h5_path = osp.join(self.data_prefix, f"data_{1:04d}.h5")
        self.load_h5data(h5_path)

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        goal_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        pt_marker = VisualizationMarkers(
            bbox_pts_cfg.replace(prim_path="/Visuals/bbox_points"))


        # Track the given command
        current_goal_idx = 0
        # Create buffers to store actions
        ik_commands = torch.zeros(
            scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
        ik_commands[:] = torch.from_numpy(self.poses[current_goal_idx])

        # Specify robot-specific parameters
        if args_cli.robot == "franka_panda":
            robot_entity_cfg = SceneEntityCfg(
                "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        elif args_cli.robot == "ur10":
            robot_entity_cfg = SceneEntityCfg(
                "robot", joint_names=[".*"], body_names=["ee_link"])
        else:
            raise ValueError(
                f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
        # Resolving the scene entities
        robot_entity_cfg.resolve(scene)
        # Obtain the frame index of the end-effector
        # For a fixed base robot, the frame index is one less than the body index. This is because
        # the root body is not included in the returned Jacobians.
        if robot.is_fixed_base:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0]

        # Define simulation stepping
        sim_dt = sim.get_physics_dt()
        count = 0
        # Simulation loop
        turn = 0
        while simulation_app.is_running():
            if count % 150 == 0:
                count = 0
                # reset joint state
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()
                # reset actions
            if count % 5 == 0:
                # _print(count//5)
                ik_commands = torch.tensor(self.poses[count//5]).to(robot.device)

                joint_pos_des = joint_pos[:,
                                          robot_entity_cfg.joint_ids].clone()
                # reset controller
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)

                turn += 1

            else:
                # obtain quantities from simulation
                jacobian = robot.root_physx_view.get_jacobians(
                )[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
                ee_pose_w = robot.data.body_pose_w[:,
                                                   robot_entity_cfg.body_ids[0]]
                root_pose_w = robot.data.root_pose_w
                joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                # compute frame in root frame
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:,
                                                     3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                # compute the joint commands
                joint_pos_des = diff_ik_controller.compute(
                    ee_pos_b, ee_quat_b, jacobian, joint_pos)

            # apply actions
            robot.set_joint_position_target(
                joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            scene.write_data_to_sim()
            # perform step
            sim.step()
            # _print(self.camera.data)
            # update sim-time
            count += 1
            # update buffers
            scene.update(sim_dt)

            # obtain quantities from simulation
            ee_pose_w = robot.data.body_state_w[:,
                                                robot_entity_cfg.body_ids[0], 0:7]
            # update marker positions
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(
                ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])
            pt_marker.visualize(self.scaled_bbox[0])

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    demo = PlanningDemo()
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    demo.run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
