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
parser.add_argument("--gpu_id", type=int, default=2,
                    help="GPU device index to use.")

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
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat
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

item_id = 100392

MY_NEW_OBJECT_CFG = ArticulationCfg(
    # 1. 指向你转换后的 USD 文件
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/hhhar/liuliu/vld/assets/articulation/{item_id}/mobility_annotation_gapartnet/mobility_annotation_gapartnet.usd",  # <-- 替换成你的文件路径
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False
        ),
        scale=[TARGET_SCALE for _ in range(3)],
        semantic_tags=[("class", "target")],
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
    robot.spawn.semantic_tags = [("class", "robot")]
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
        self.scaled_bboxs = []
        self.scaled_bbox = []
        self.poses = []
        self.actions = []
        self.rgbs = []
        self.depthes = []
        self.cam_mat = []
        self.action_types = []
        self.part_num = -1
        self.output_prefix = output_prefix
        self.data_dict = {}
        self.define_sensor()
        self.part_id = -1
        self.over = False

        if osp.exists(self.output_prefix) is False and self.save_data:
            os.makedirs(self.output_prefix)

        self.get_bboxs()
        
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
                "semantic_segmentation",
                "instance_segmentation_fast",
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

    def get_bboxs(self, pos=TARGET_INIT_POS, rot=TARGET_INIT_ROT, scale=TARGET_SCALE):
        anno_path = f"/home/hhhar/liuliu/vld/assets/articulation/{self.target_id}/link_annotation_gapartnet.json"
        with open(anno_path, "r") as f:
            annos = json.load(f)
            self.part_num = len(annos)

        for anno in annos:
            if anno["is_gapart"] == False:
                continue
            bbox = anno['bbox']
            self.scaled_bboxs.append(transform_points_by_pose(bbox, pos, rot, scale))
            if anno['category'] == "slider_button":
                self.action_types.append('press')
            else:
                self.action_types.append("rot")

    def choose_part(self, turn):
        print(f"Choosing part for turn {turn}...")
        if turn == -1:
            turn = 0
        target_id = turn // 20
        if target_id >= len(self.scaled_bboxs):
            self.over = True
            return 0

        self.scaled_bbox = np.array(self.scaled_bboxs[target_id]).reshape(-1, 1, 3)
        self.data_dict["scaled_bbox"] = self.scaled_bbox

        self.part_id = target_id
        self.data_dict["part_id"] = self.part_id

        self.data_dict["action_type"] = self.action_types[target_id]

    def collect_data(self, frame_id, robot, robot_entity_cfg):
        single_cam_data = convert_dict_to_backend(
            {k: v[self.camera_id] for k, v in self.camera.data.output.items()}, backend="numpy"
        )

        # # Extract the other information
        # single_cam_info = self.camera.data.info[self.camera_id]
        # group_name = f"frame_{frame_id}"

        # 2. 将数据保存为 dataset
        for key, data in single_cam_data.items():
            if data is not None:
                img = np.array(data)
                if img.max() > 1.0000002:
                    img = img.astype(np.int16)
                else:
                    img = img.astype(np.float32)  # already 0..1
                if self.data_dict.get(key) is None:
                    self.data_dict[key] = []
                self.data_dict[key].append(data)

                # # 3. 将元数据 (info) 保存为 dataset 的 attributes
                # info = single_cam_info.get(key)
                # if info is not None:
                #     for info_key, info_val in info.items():
                #         # 确保值是 h5py 支持的类型 (str, int, float, numpy scalar)
                #         if info_val is None:
                #             info_val = 'None' # H5 不支持 None
                #             if isinstance(info_val, (list, tuple)):
                #                 info_val = np.array(info_val)
                #             elif isinstance(info_val, (int, float, str, np.number)):
                #                 info_val = info_val
                #             else:
                #                 dset.attrs[info_key] = json.dumps(info_val, ensure_ascii=False)

        pose = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        # embed()
        self.data_dict["poses"].append(pose.cpu().numpy())

    def cal_cammat(self, pos, tar):
        """
        计算 Isaac Sim (USD) 标准的相机位姿矩阵。
        约定：-Z 为前方, +Y 为上方, +X 为右方
        """
        # 1. 计算 Z 轴 (Camera Z)
        # 因为 Isaac 相机看的是 -Z 方向，所以 +Z 应该指向“相机后方” (从目标指向相机)
        z_axis = pos - tar
        z_axis /= np.linalg.norm(z_axis)

        # 2. 定义世界 上方向 (World Up)
        world_up = np.array([0, 0, 1])

        # 3. 计算 X 轴 (Camera Right)
        # 右 = WorldUp x Z_axis
        x_axis = np.cross(world_up, z_axis)
        # 极点保护：如果相机垂直向下看，x_axis 模长会接近0，需要特殊处理
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1, 0, 0]) # 默认值
        else:
            x_axis /= np.linalg.norm(x_axis)

        # 4. 计算 Y 轴 (Camera Up)
        # 上 = Z_axis x X_axis (严格正交)
        y_axis = np.cross(z_axis, x_axis)
        
        # 5. 组装矩阵
        # 旋转矩阵的列向量分别是 X, Y, Z 轴
        mat44 = np.eye(4)
        mat44[:3, 0] = x_axis
        mat44[:3, 1] = y_axis
        mat44[:3, 2] = z_axis
        mat44[:3, 3] = pos
        self.cam_mat.append(mat44)
    
    # def cal_cammat(self):
    #     pos = self.camera.data.pos_w[self.camera_id]      # Shape: (3,) -> [x, y, z]
    #     quat = self.camera.data.quat_w_world[self.camera_id]  # Shape: (4,) -> [w, x, y, z]
    #     # 2. 将四元数转换为 3x3 旋转矩阵
    #     # Isaac Lab 提供了基于 torch 的转换工具
    #     embed()
    #     _print(pos)
    #     _print(quat)
    #     rot_matrix = matrix_from_quat(quat) # Shape: (3, 3)
    #     _print(rot_matrix)
    #     # 3. 组装 4x4 矩阵
    #     # 创建一个 4x4 的单位阵
    #     pose_matrix = torch.eye(4, device=self.camera.device)

    #     # 填入旋转部分 [0:3, 0:3]
    #     pose_matrix[:3, :3] = rot_matrix

    #     # 填入平移部分 [0:3, 3]
    #     pose_matrix[:3, 3] = pos
    #     pose_matrix = pose_matrix.cpu().numpy()
    #     self.cam_mat.append(pose_matrix)
    #     self.data_dict["cam_mat"]= self.cam_mat[self.camera_id]


    def _reset(self, turn):
        self.data_dict = {
            "poses": [], "scaled_bbox": [], "cam_mat": [], "action_type": "", "camera_id": self.camera_id,}
        self.choose_part(turn)
        self.data_dict["target_id"] = self.target_id
        self.data_dict["cam_mat"]= self.cam_mat[self.camera_id]

    def initialize_robot_with_ik(self, sim, robot, robot_entity_cfg, ik_controller, target_pos_w, target_quat_w, scene, max_iters=20):
        """
        使用 IK 求解器在这一帧"瞬间"算出目标位置的关节角，并将其设为机器人的默认初始姿态。
        
        原理：在内部运行一个小的收敛循环，利用 IK 迭代逼近目标，避免直接 Set 导致的瞬移误差或抽搐。

        Args:
            robot: Articulation 对象
            robot_entity_cfg: 包含 joint_ids, body_ids 的配置对象
            ik_controller: DifferentialIKController 实例
            target_pos_w: 世界坐标系下的目标位置 (num_envs, 3)
            target_quat_w: 世界坐标系下的目标姿态 (num_envs, 4)
            scene: InteractiveScene 对象 (用于在循环内刷新数据)
            max_iters: 内部收敛循环的次数，通常 20 次足够收敛
            
        Returns:
            final_joint_pos: 收敛后的关节角度 (num_envs, num_joints)
        """
        
        print(f"[IK Init] Starting IK convergence loop ({max_iters} iters)...")
        
        # 1. 准备数据
        # 获取机械臂末端的 body index (通常配置里只有一个 body_id 对应 ee)
        ee_body_idx = robot_entity_cfg.body_ids[0]
        # 获取受控关节的 indices
        arm_joint_indices = robot_entity_cfg.joint_ids
        
        # 临时保存物理步长，循环里用
        sim_dt = scene.physics_dt
        
        # 2. 坐标系预处理：将目标从 World Frame 转为 Base Frame (假设 Controller 需要 Base Frame)
        # 这是一个初始转换，但由于 Robot Base 可能在移动（虽然初始化时一般不动），
        # 我们最好在循环里动态计算，但为了性能，初始化时算一次通常也可以。
        # 这里我们在循环外先准备好目标的 Tensor
        
        # --- A. 获取当前物理状态 ---
        # 必须读 robot.data，它是缓存的数据
        root_pose_w = robot.data.root_pose_w
        ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]
        jacobian = robot.root_physx_view.get_jacobians(
            )[:, ee_body_idx, :, arm_joint_indices]
        current_arm_joint_pos = robot.data.joint_pos[:, arm_joint_indices]
        
        # 计算目标在基座坐标系下的位置 (每一帧都算一下，防止基座有微小滑动)
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            target_pos_w.unsqueeze(0).float(), target_quat_w.unsqueeze(0).float()
        )

        # --- C. 计算 IK ---
        # 拼接目标指令
        ik_commands = torch.cat([target_pos_b, target_quat_b], dim=-1)
        ik_controller.reset()
        ik_controller.set_command(ik_commands)

        for i in range(max_iters):
        
            # 计算当前末端在基座坐标系下的位置
            jacobian = robot.root_physx_view.get_jacobians(
                )[:, ee_body_idx, :, arm_joint_indices]
            current_arm_joint_pos = robot.data.joint_pos[:, arm_joint_indices]
            ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            
            # 计算下一步的关节位置
            desired_arm_pos = ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, current_arm_joint_pos
            )

            # --- D. 拼接完整关节状态 (Arm + Gripper) ---
            # 我们不能只改 Arm，必须保留 Gripper 状态，或者设为默认
            # 最好是拿当前的 default_joint_pos 作为底板
            full_joint_pos = robot.data.default_joint_pos.clone()
            
            # 覆盖手臂部分的关节角度
            # 注意：这里假设 robot_entity_cfg.joint_ids 对应的就是 arm 的索引
            full_joint_pos[:, arm_joint_indices] = desired_arm_pos

            # --- E. 强制瞬移 (Teleport) ---
            # 将速度设为 0，防止物理引擎计算出巨大的惯性# apply actions
            robot.set_joint_position_target(
                desired_arm_pos, joint_ids=robot_entity_cfg.joint_ids)
            # scene.write_data_to_sim()
            # sim.step()
            # robot.write_joint_state_to_sim(full_joint_pos, torch.zeros_like(full_joint_pos))
            scene.write_data_to_sim()
            sim.step(render=False)
            
            # --- F. 步进物理 (Step Physics) ---
            # 这一步至关重要！它让 PhysX 更新 Jacobian 和刚体位置
            # render=False 保证速度极快
            scene.update(sim_dt)

        # 3. 循环结束，收尾工作
        print("[IK Init] Convergence finished.")
        
        # 获取最终算出来的这个比较完美的关节角度
        final_joint_pos = robot.data.joint_pos.clone()
        
        # 关键：更新默认位置。这样下次你调用 robot.reset() 时，它就会回到这里，而不是回到配置文件里的默认姿态
        robot.data.default_joint_pos = final_joint_pos
        robot.data.default_joint_vel = torch.zeros_like(final_joint_pos)

        # 关键：重置控制器内部状态 (积分项等)，防止之前的累积误差影响接下来的正式控制
        ik_controller.reset()
        
        # 最后做一次正式的 Reset，确保一切同步
        robot.reset()
        scene.update(dt=0.0) # 刷新 Buffer
        
        return final_joint_pos
    
    def run_simulator(self, sim: sim_utils.SimulationContext, scene: InteractiveScene):
        """Runs the simulation loop."""
        # Extract scene entities
        # note: we only do this here for readability.
        robot = scene["robot"]
        # self.camera = scene["camera"]

        # Camera positions, targets, orientations
        camera_positions = torch.tensor(
            [[0.9, 0.4, 1.0], [-0.9, -0.4, 1.0]], device=sim.device)
        camera_targets = torch.tensor(
            [TARGET_INIT_POS, TARGET_INIT_POS], device=sim.device)

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
        pt_marker_1 = VisualizationMarkers(
            bbox_pts_cfg.replace(prim_path="/Visuals/bbox_points_1"))
        pt_marker_2 = VisualizationMarkers(
            bbox_pts_cfg.replace(prim_path="/Visuals/bbox_points_2"))

        # Define goals for the arm
        ee_goals = [
            [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
            [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
            [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ]
        ee_goals = torch.tensor(ee_goals, device=sim.device)
        # Create buffers to store actions
        ik_commands = torch.zeros(
            scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
        ik_commands[:] = ee_goals[self.part_id]

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
        turn = -1
        self._reset(turn)
        while simulation_app.is_running() and self.over==False:
            if count % 100 == 0:
                turn += 1
                h5_path = osp.join(self.output_prefix, f"data_{turn:04d}.h5")
                # reset time
                if turn > 0 :
                    if self.save_data:
                        print(f"HDF5 writer initialized. Saving data to: {h5_path}")
                        self.data2h5(h5_path)
                self._reset(turn)  

                count = 0
                # reset joint state
                init_pos_id = np.random.randint(0, self.part_num)
                center = -1
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)

                if init_pos_id < len(self.scaled_bboxs):
                    chosen_center = self.scaled_bboxs[init_pos_id].mean(0)
                    chosen_center[2] += 0.1
                    # embed()

                    noise_pos = torch.randn(3, device=sim.device) * 0.02
                    init_pos = torch.from_numpy(chosen_center).to(sim.device) + noise_pos
                    init_pos[2] += 0.04

                    noise_ori = torch.randn(4, device=sim.device) * 0.01
                    init_orientation = torch.tensor([0., 1., 0., 0.], device=sim.device)
                    init_orientation = init_orientation + noise_ori
                    init_orientation = init_orientation / init_orientation.norm()
                    joint_pos = self.initialize_robot_with_ik(sim, robot, robot_entity_cfg, diff_ik_controller, init_pos, init_orientation, scene)
                else:
                    self.robot.reset()

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
                # pt_marker_1.visualize(self.scaled_bbox[0])
                # pt_marker_2.visualize(self.scaled_bbox[-1])

                
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
            self.camera.update(dt=sim.get_physics_dt())

            if turn == 0 and count == 0:
                intrinsics = self.camera.data.intrinsic_matrices
                self.data_dict["intrinsics"] = intrinsics[self.camera_id].cpu().numpy()

            # _print(self.camera.data)
            # update sim-time
            count += 1
            # update buffers
            scene.update(sim_dt)
            self.camera.update(0.0)

            # obtain quantities from simulation
            ee_pose_w = robot.data.body_state_w[:,
                                                robot_entity_cfg.body_ids[0], 0:7]
            # # update marker positions
            # ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            # goal_marker.visualize(
            #     ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

            if count % 5 == 0:
                if self.save_data:
                    self.collect_data(count, robot, robot_entity_cfg)


@hydra.main(config_path="config", config_name="data")
def main(cfg: DictConfig):
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([0.9, 0.4, 1.5], TARGET_INIT_POS)
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
