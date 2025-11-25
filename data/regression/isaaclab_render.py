# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""


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
"""Rest everything follows."""

##
# Pre-defined configs
##

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


def _print(msg: str):
    print('----------------------------------------------------------------------')
    print(msg)


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


def quat_to_rotmat(q):
    """
    q: array-like [w, x, y, z]
    returns 3x3 rotation matrix
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        q = q.ravel()[:4]
    # normalize to be safe
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    # 3x3 rotation matrix (Hamilton convention)
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R


def transform_points_by_pose(points, p, q, scale=1.0):
    """
    points: (N,3) np.array in local coordinates
    p: (3,) translation in world coords
    q: [w,x,y,z] quaternion (world orientation of the local frame)
    scale: scalar or (3,) array, applied in local frame before rotation/translation
    returns transformed_points: (N,3)
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    # apply local scale (support scalar or per-axis)
    scale = np.asarray(scale, dtype=float)
    if scale.size == 1:
        pts = pts * float(scale)
    else:
        pts = pts * scale.reshape(1, 3)
    # rotate
    R = quat_to_rotmat(q)
    pts_rot = pts.dot(R.T)   # (N,3)
    # translate
    return pts_rot + np.asarray(p).reshape(1, 3)


class PlanningDemo:
    def __init__(self, target_id, camera_id, save_data=False, output_prefix="./output/camera"):
        self.target_id = target_id
        self.save_data = save_data
        self.camera_id = camera_id
        self.camera = []
        self.bbox = []
        self.scaled_bbox = []
        self.poses = []
        self.actions = []
        self.rgbs = []
        self.depthes = []
        self.cam_mat = []
        self.action_type = []
        self.part_num = -1
        self.output_prefix = output_prefix
        self.data_dict = {}
        self.define_sensor()
        
    def define_sensor(self):
        """Defines the camera sensor to add to the scene."""
        # Setup camera sensor
        # In contrast to the ray-cast camera, we spawn the prim at these locations.
        # This means the camera sensor will be attached to these prims.
        prim_utils.create_prim("/World/Origin_00", "Xform")
        prim_utils.create_prim("/World/Origin_01", "Xform")
        camera_cfg = CameraCfg(
            prim_path="/World/Origin_.*/CameraSensor",
            update_period=0,
            height=480,
            width=640,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "normals",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
            ],
            colorize_semantic_segmentation=True,
            colorize_instance_id_segmentation=True,
            colorize_instance_segmentation=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
        )
        # Create camera
        self.camera = Camera(cfg=camera_cfg)


    def data2h5(self, h5_path):
        with h5py.File(h5_path, "w") as f:
            for key in self.data_dict.keys():
                f.create_dataset(key, data=self.data_dict[key])
            f.flush()
        print('this turn save ok')

    def get_bbox(self, pos=TARGET_INIT_POS, rot=TARGET_INIT_ROT, scale=TARGET_SCALE):
        anno_path = f"/home/hhhar/liuliu/vld/assets/articulation/{self.target_id}/link_annotation_gapartnet.json"
        with open(anno_path, "r") as f:
            anno = json.load(f)
            self.part_num = len(anno)
        target_id = random.randint(0, self.part_num - 1)
        target_anno = anno[target_id]
        while target_anno["is_gapart"] == "false":
            target_id = random.randint(0, self.part_num - 1)
            target_anno = anno[target_id]
        self.bbox = target_anno['bbox']
        if target_anno['category'] == "slider_button":
            self.action_type = 'press'
        else:
            self.action_type = "rot"
        self.data_dict["action_type"] = self.action_type

        self.scaled_bbox = transform_points_by_pose(
            self.bbox, pos, rot, scale)
        self.scaled_bbox = np.array(self.scaled_bbox).reshape(-1, 1, 3)
        self.data_dict["scaled_bbox"] = self.scaled_bbox


    def collect_data(self, turn, frame_id, robot, robot_entity_cfg):
        output_dir = osp.join(self.output_prefix, f"data_{turn:04d}", f"frame_{frame_id:05d}")
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        rep_writer = rep.BasicWriter(
            output_dir=output_dir,
            frame_padding=0,
            colorize_instance_id_segmentation=self.camera.cfg.colorize_instance_id_segmentation,
            colorize_instance_segmentation=self.camera.cfg.colorize_instance_segmentation,
            colorize_semantic_segmentation=self.camera.cfg.colorize_semantic_segmentation,
        )

        single_cam_data = convert_dict_to_backend(
            {k: v[self.camera_id] for k, v in self.camera.data.output.items()}, backend="numpy"
        )
        # Extract the other information
        single_cam_info = self.camera.data.info[self.camera_id]
        # Pack data back into replicator format to save them using its writer
        rep_output = {"annotators": {}}
        for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
            if info is not None:
                rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
            else:
                rep_output["annotators"][key] = {"render_product": {"data": data}}
        # Save images
        # Note: We need to provide On-time data for Replicator to save the images.
        rep_output["trigger_outputs"] = {"on_time": self.camera.frame[self.camera_id]}
        rep_writer.write(rep_output)


        pose = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        # embed()
        self.data_dict["poses"].append(pose.cpu().numpy())

    def cal_cammat(self, pos, tar):
        # 计算朝向向量
        forward = tar - pos
        forward /= np.linalg.norm(forward)

        # 定义上方向
        up = np.array([0, 0, 1])

        # 计算左方向
        left = np.cross(up, forward)
        left /= np.linalg.norm(left)

        # 重新计算正交的上方向
        up = np.cross(forward, left)

        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = pos
        self.cam_mat.append(mat44)

    def _reset(self):
        self.data_dict = {
            "poses": [], "scaled_bbox": [], "cam_mat": [], "action_type": "", "camera_id": self.camera_id,}
        self.get_bbox()
        self.data_dict["cam_mat"]= self.cam_mat[self.camera_id]

    def run_simulator(self, sim: sim_utils.SimulationContext, scene: InteractiveScene):
        """Runs the simulation loop."""
        # Extract scene entities
        # note: we only do this here for readability.
        robot = scene["robot"]
        # self.camera = scene["camera"]

        # Camera positions, targets, orientations
        camera_positions = torch.tensor(
            [[2.5, 2.5, 2.5], [-2.5, -2.5, 2.5]], device=sim.device)
        camera_targets = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)

        # Set pose
        self.camera.set_world_poses_from_view(camera_positions, camera_targets)
        for i in range(camera_positions.shape[0]):
            self.cal_cammat(camera_positions[i].cpu().numpy(),
                           camera_targets[i].cpu().numpy())
        # Create controller
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        goal_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        pt_marker = VisualizationMarkers(
            bbox_pts_cfg.replace(prim_path="/Visuals/bbox_points"))

        # Define goals for the arm
        ee_goals = [
            [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
            [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
            [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ]
        ee_goals = torch.tensor(ee_goals, device=sim.device)
        # Track the given command
        current_goal_idx = 0
        # Create buffers to store actions
        ik_commands = torch.zeros(
            scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
        ik_commands[:] = ee_goals[current_goal_idx]

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
        self._reset()
        while simulation_app.is_running():
            # reset
            if count % 150 == 0:
                h5_path = osp.join(self.output_prefix, f"data_{turn:04d}/status.h5")
                # reset time
                count = 0
                # reset joint state
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()
                # reset actions
                # ik_commands[:] = ee_goals[current_goal_idx]
                bbox_pts = deepcopy(self.scaled_bbox).reshape(-1, 3)
                center = bbox_pts.mean(0).tolist()
                center[2] += 0.1
                ik_commands[0, :3] = torch.tensor(center)
                ik_commands[0, 3:] = torch.tensor([0, 1, 0, 0])

                joint_pos_des = joint_pos[:,
                                          robot_entity_cfg.joint_ids].clone()
                # reset controller
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)

                turn += 1
                if turn > 1 and self.save_data:
                    print(f"HDF5 writer initialized. Saving data to: {h5_path}")
                    self.data2h5(h5_path)
                    self._reset()   

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

            if count % 5 == 0:
                if self.save_data:
                    self.collect_data(turn, count, robot, robot_entity_cfg)

            # apply actions
            robot.set_joint_position_target(
                joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            scene.write_data_to_sim()
            # perform step
            sim.step()

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
            # pt_marker.visualize(self.scaled_bbox[0])


@hydra.main(config_path="config", config_name="data")
def main(cfg: DictConfig):
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([1.5, 0.0, 1.5], [0.0, 0.0, 0.0])
    demo = PlanningDemo(**cfg.PlanningDemo)
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
