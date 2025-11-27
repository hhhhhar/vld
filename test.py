import os
import argparse


from isaaclab.app import AppLauncher
os.environ["ENABLE_CAMERAS"] = "1"

# 1. 启动 Isaac Sim (必须在其他 import 之前)
parser = argparse.ArgumentParser(description="Evaluate VLD Policy in Isaac Sim.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Standard Imports
# ---------------------------------------------------------------------------
import torch
import numpy as np
import hydra
import json
import math
from omegaconf import DictConfig
from copy import deepcopy
from IPython import embed
from scipy.spatial.transform import Rotation
import random

# Isaac Lab Imports
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import convert_dict_to_backend
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import subtract_frame_transforms
import isaacsim.core.utils.prims as prim_utils

# VLD Model Imports (假设这些文件在当前目录下)
from model.main import VLDDiffusionSystem
from transformers import BertTokenizer

# ---------------------------------------------------------------------------
# Scene Configuration (复用 isaac_render.py 的配置)
# ---------------------------------------------------------------------------
TARGET_SCALE = 0.2
TARGET_INIT_POS = (0.5, 0.0, 0.2)
TARGET_INIT_ROT = (0.7, 0.0, 0.7, 0.0)

# 这里你需要根据实际情况修改 USD 路径
MY_NEW_OBJECT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/hhhar/liuliu/vld/assets/articulation/100392/mobility_annotation_gapartnet/mobility_annotation_gapartnet.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        scale=[TARGET_SCALE for _ in range(3)],
        semantic_tags=[("class", "target")],
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=TARGET_INIT_POS, rot=TARGET_INIT_ROT),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        )
    }
)

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
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )
    
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    robot.spawn.semantic_tags = [("class", "robot")]
    my_new_articulation = MY_NEW_OBJECT_CFG.replace(prim_path="{ENV_REGEX_NS}/MyNewObject")

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



# ---------------------------------------------------------------------------
# Inference Agent (Core Logic)
# ---------------------------------------------------------------------------
class VLDInferenceAgent:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.scaled_bbox = []
        self.action_type = []
        self.bbox_info = []
        self.action_queue = [] 
        
        # 1. 加载统计量 (Min/Max) 用于反归一化
        print(f"Loading stats from {cfg.path.action_stats_path}")
        with open(cfg.path.action_stats_path, 'r') as f:
            stats = json.load(f)
        self.min_val = torch.tensor(stats['min'], device=device)
        self.max_val = torch.tensor(stats['max'], device=device)
        self.bins = torch.linspace(-1, 1, cfg.model.num_action_bins, device=device)

        # 2. 初始化模型
        print("Initializing Model...")
        self.model = VLDDiffusionSystem(**cfg.model)
        self.tokenizer = BertTokenizer.from_pretrained(cfg.model.text_model_name)
        
        # 3. 加载权重
        print(f"Loading checkpoint from {cfg.path.checkpoint_path}")
        checkpoint = torch.load(cfg.path.checkpoint_path, map_location=device)
        # 处理可能存在的 'model_state_dict' 键
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        self.step_counter = 0
        self.action_queue = [] # 用于 Receding Horizon Control

    def get_txts_bbox_at(self, pos=TARGET_INIT_POS, rot=TARGET_INIT_ROT, scale=TARGET_SCALE):
        anno_path = f"/home/hhhar/liuliu/vld/assets/articulation/{self.cfg.sim.item_id}/link_annotation_gapartnet.json"
        with open(anno_path, "r") as f:
            anno = json.load(f)
            part_num = len(anno)
        target_id = random.randint(0, part_num - 1)
        target_anno = anno[target_id]
        while target_anno["is_gapart"] == False:
            target_id = random.randint(0, part_num - 1)
            target_anno = anno[target_id]
        bbox = target_anno['bbox']
        if target_anno['category'] == "slider_button":
            action_type = 'press'
        else:
            action_type = "rot"

        scaled_bbox = transform_points_by_pose(
            bbox, pos, rot, scale)
        scaled_bbox = np.array(scaled_bbox).reshape(-1, 3)
        self.bbox_info = self.convert_bbox_8points_to_pose(scaled_bbox)
        bbox_info = deepcopy(self.bbox_info)
        del bbox_info["matrix"]  # 删除矩阵，保留中心、尺寸、四元数
        return f"action: {action_type}, target_area: {bbox_info}"
    
    def reset(self):
        """重置 Agent 内部状态 (清空动作队列)"""
        self.action_queue = []
    
    def convert_bbox_8points_to_pose(self, bbox):
        """
        将 8 个角点的 BBox 转换为中心、尺寸和四元数。

        Args:
            points (np.ndarray): 形状为 (8, 3) 的 numpy 数组，表示 8 个角点的坐标。

        Returns:
            dict: 包含以下键的字典:
                - "center": (3,) np.array, [x, y, z]
                - "extent": (3,) np.array, [length, width, height] (也称为 size)
                - "quat":   (4,) np.array, [w, x, y, z] (Isaac Lab 标准)
                - "matrix": (3, 3) np.array, 旋转矩阵
        """
        points = np.array(bbox)

        if points.shape != (8, 3):
            raise ValueError(
                f"Input points must be (8, 3), got {points.shape}")

        # 1. 计算中心点 (Center)
        # 几何中心即所有点的平均值
        center = np.mean(points, axis=0)

        # 2. 计算旋转矩阵 (Rotation)
        # 使用 PCA/SVD 方法找到物体的主轴。
        # 将点去中心化
        centered_points = points - center

        # 计算协方差矩阵并进行特征分解，或者直接使用 SVD
        # U, S, Vh = SVD(X) -> Vh 的行是特征向量（主轴方向）
        # 注意：PCA 对轴的方向（正负）有模糊性，且如果物体是正方体，轴向可能不稳定，
        # 但对于定义 BBox 依然有效。
        u, s, vh = np.linalg.svd(centered_points)

        # 旋转矩阵的列应该是主轴方向，vh 的行是特征向量
        rotation_matrix = vh.T

        # 【关键修正】SVD 出来的矩阵可能带有反射（行列式为 -1），即左手坐标系
        # 我们需要强制转换为右手坐标系
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 2] *= -1  # 反转 Z 轴方向

        # 3. 计算尺寸 (Dimensions/Extent)
        # 将原始点投影到新的主轴坐标系下，计算各轴的跨度
        # projected shape: (8, 3)
        projected_points = centered_points @ rotation_matrix

        min_xyz = np.min(projected_points, axis=0)
        max_xyz = np.max(projected_points, axis=0)

        # 尺寸 = 最大值 - 最小值
        extent = max_xyz - min_xyz

        # 4. 计算四元数 (Quaternion)
        # Isaac Sim/Lab 使用 [w, x, y, z]
        # Scipy 默认输出 [x, y, z, w]，需要调整顺序
        r = Rotation.from_matrix(rotation_matrix)
        quat_scipy = r.as_quat()  # [x, y, z, w]
        quat_isaac = np.array(
            [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])

        return {
            "center": center.tolist(),
            "extent": extent.tolist(),
            "quat": quat_isaac.tolist(),
            "matrix": rotation_matrix
        }


    def _depth_to_world_pointcloud(self, depth, intrinsic_matrix, camera_pose_4x4, num_points=1024):
        """
        在推理时实时生成点云。逻辑需与训练 Dataset 保持严格一致。
        """
        # 注意：这里输入已经是 GPU Tensor
        H, W = depth.shape
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

        # 1. 生成网格
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )

        # 2. 反投影 (Image Plane -> Camera Frame)
        # 过滤无效深度 (Isaac Sim 有时返回极大值或 0 表示无穷远)
        valid_mask = (depth > 0) & (depth < 10.0) 
        z = depth
        x = (u_grid - cx) * z / fx
        y = (v_grid - cy) * z / fy
        
        points_cam = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        valid_mask_flat = valid_mask.reshape(-1)
        points_cam = points_cam[valid_mask_flat]

        # 3. 坐标系修正 (CV Frame -> Isaac USD Frame)
        # CV: +Z 前, +Y 下, +X 右
        # Isaac Camera Prim: -Z 前, +Y 上, +X 右
        # 修正向量: [1, -1, -1]
        correction = torch.tensor([1.0, -1.0, -1.0], device=self.device)
        points_cam = points_cam * correction

        # 4. 转世界坐标 (Camera Frame -> World Frame)
        # P_world = P_cam @ R.T + T
        R = camera_pose_4x4[:3, :3]
        T = camera_pose_4x4[:3, 3]
        points_world = points_cam @ R.T + T

        # 5. 采样 (Sampling)
        num_curr = points_world.shape[0]
        if num_curr == 0:
            return torch.zeros((num_points, 3), device=self.device)
            
        if num_curr >= num_points:
            choice = torch.randperm(num_curr, device=self.device)[:num_points]
        else:
            choice = torch.randint(0, num_curr, (num_points,), device=self.device)
        
        return points_world[choice]

    def _dequantize_action(self, action_tokens):
        """
        离散 Token (0-255) -> 归一化值 (-1, 1) -> 物理值 (Pose)
        """
        # 1. Token -> Normalized Value
        # 找到 bin 的中心值或者下界
        # self.bins 是 linspace(-1, 1, 256)
        # action_tokens shape: (B, Chunk, 7)
        
        # 使用 embedding lookup 的思想或者直接索引
        # 为了简单，直接用索引映射回 bin 的值
        # 注意: action_tokens 是 LongTensor
        norm_actions = self.bins[action_tokens] # (B, Chunk, 7)

        # 2. Denormalize
        # x_norm = 2 * (x - min) / (max - min) - 1
        # => x = (x_norm + 1) * (max - min) / 2 + min
        
        min_v = self.min_val.view(1, 1, -1)
        max_v = self.max_val.view(1, 1, -1)
        
        phys_actions = (norm_actions + 1) * (max_v - min_v) / 2 + min_v
        
        # 强制归一化四元数
        pos = phys_actions[..., :3]
        quat = phys_actions[..., 3:]
        quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
        
        phys_actions = torch.cat([pos, quat], dim=-1)
        return phys_actions

    def get_action(self, obs_dict, vis=False):
        """
        主推理函数
        Args:
            obs_dict: 包含 'rgb', 'depth', 'camera_pose', 'intrinsic', 'instruction'
        Returns:
            next_action: (7,) Tensor, 下一步要执行的 EE Pose
        """
        # 1. 处理输入数据
        img = obs_dict['rgb'].permute(0, 3, 1, 2).float() / 255.0 # (B, 3, H, W)
        
        # 生成点云 (假设 batch=1，简单处理 loop)
        pcs = []
        for i in range(img.shape[0]):
            pc = self._depth_to_world_pointcloud(
                obs_dict['depth'][i].squeeze(-1), 
                obs_dict['intrinsic'][i], 
                obs_dict['camera_pose'][i]
            )
            pcs.append(pc)
        pcs = torch.stack(pcs) # (B, 1024, 3)

        raw_texts = obs_dict['instruction'] # List[str]
        text_inputs = self.tokenizer(
            raw_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        for keys in text_inputs:
            text_inputs[keys] = text_inputs[keys].to(self.cfg.sim.device)

        if vis:
            try:
                import open3d as o3d
                print("\n🎨 正在启动 Open3D 可视化窗口...")
                print("   (按 'Q' 键退出窗口)")
                
                obb = o3d.geometry.OrientedBoundingBox()
                obb.center = np.array(self.bbox_info["center"])
                obb.extent = np.array(self.bbox_info["extent"])
                obb.R = self.bbox_info["matrix"]
                # 设置框的颜色 (例如红色线条)
                obb.color = (1, 0, 0)

                # 转为 Numpy CPU
                pcd_np = pcs[0].cpu().numpy()

                # 创建 Open3D 对象
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)

                # 添加一个坐标轴 (红X, 绿Y, 蓝Z) 用作世界原点参考
                origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=1.0, origin=[0, 0, 0])

                # 为了展示相机位置，我们在相机位置也画个小坐标轴
                cam_pos_np = obs_dict['camera_pose'][0][:3, 3].cpu().numpy()
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.5)
                cam_frame.translate(cam_pos_np)

                o3d.visualization.draw_geometries([pcd_o3d, origin_frame, cam_frame, obb],
                                                    window_name="Isaac Lab World Point Cloud")
            except ImportError:
                print("⚠️ 未安装 Open3D，无法进行点云可视化。请安装 open3d 库后重试。")


        # 2. Receding Horizon Control 逻辑
        # 如果队列空了，或者非空但还没执行完 horizon，这里简化逻辑：
        # 每次都清空队列重新预测 (Closed Loop)，或者用队列缓存
        
        execution_horizon = self.cfg.inference.execution_horizon
        # 只要队列里的动作少于 horizon，就重新推理
        if len(self.action_queue) == 0:
            with torch.no_grad():
                # Diffusion 推理
                # action_tokens: (B, Chunk, 7)
                action_tokens = self.model.predict(img, pcs, text_inputs, num_steps=self.cfg.inference.num_steps)
                
                # 反量化
                pred_actions = self._dequantize_action(action_tokens) # (B, Chunk, 7)
                
                # 将预测的未来动作存入队列
                # 假设 Batch=0 (单环境测试)
                chunk_len = pred_actions.shape[1]
                # 只取前 execution_horizon 步，或者全部取完
                steps_to_take = min(execution_horizon, chunk_len)
                
                for t in range(steps_to_take):
                    self.action_queue.append(pred_actions[0, t])
        
        # 3. 弹出一个动作执行
        return self.action_queue.pop(0)

def cal_cammat(pos, tar):
    # 确保输入是 Tensor (如果传入的是 list 或 numpy，这里兼容处理一下，实际使用建议直接传 Tensor)
    if not isinstance(pos, torch.Tensor):
        pos = torch.tensor(pos)
    if not isinstance(tar, torch.Tensor):
        tar = torch.tensor(tar)
    
    # 获取输入的设备和数据类型，确保后续新建的 tensor 保持一致
    device = pos.device
    dtype = pos.dtype

    # 1. 计算 Z 轴 (Camera Z)
    # 也就是从目标指向相机 (+Z 指向相机后方)
    z_axis = pos - tar
    z_axis = z_axis / torch.linalg.norm(z_axis)

    # 2. 定义世界 上方向 (World Up)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    # 3. 计算 X 轴 (Camera Right)
    # 右 = WorldUp x Z_axis
    x_axis = torch.linalg.cross(world_up, z_axis)

    # 极点保护：如果 x_axis 模长接近 0 (说明相机视线与 WorldUp 平行)
    if torch.linalg.norm(x_axis) < 1e-6:
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    else:
        x_axis = x_axis / torch.linalg.norm(x_axis)

    # 4. 计算 Y 轴 (Camera Up)
    # 上 = Z_axis x X_axis (严格正交)
    y_axis = torch.linalg.cross(z_axis, x_axis)

    # 5. 组装矩阵
    # 创建单位矩阵
    mat44 = torch.eye(4, device=device, dtype=dtype)
    
    # 填充旋转部分 (列向量)
    mat44[:3, 0] = x_axis
    mat44[:3, 1] = y_axis
    mat44[:3, 2] = z_axis
    
    # 填充位移部分
    mat44[:3, 3] = pos

    return mat44


class EvaluationRunner:
    def __init__(self, cfg, sim, scene, agent, diff_ik_controller, camera):
        self.cfg = cfg
        self.sim = sim
        self.scene = scene
        self.agent = agent
        self.controller = diff_ik_controller
        self.camera = camera
        self.max_steps_per_round = cfg.runner.max_steps_per_round  # 每一轮测试最大步数 (例如 200步 ≈ 4秒操作)
        self.total_rounds = self.cfg.runner.total_rounds          # 总共测试多少轮

        
        # 获取 Robot Entity 配置用于重置
        self.robot = scene["robot"]
        if args_cli.robot == "franka_panda":
            self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["panda_hand"])
        else:
             self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
        self.robot_entity_cfg.resolve(scene)
        
        # 计数器
        self.step_count = 0
        self.round_count = 0
        self.txts = []
        self.goal_marker = None
        self.pt_marker = None

    def reset_env(self):
        """重置环境、机器人、控制器和 Agent"""
        print(f"[INFO] Resetting environment... (Round {self.round_count})")
        
        # 1. 重置 Robot 关节状态
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()
        
        # 2. 重置目标物体 (MyNewObject)
        # 如果需要随机位置，可以在这里生成新的 Pose 并 write_root_pose_to_sim
        if "my_new_articulation" in self.scene.keys():
            target_obj = self.scene["my_new_articulation"]
            root_state = target_obj.data.default_root_state.clone()
            # 示例：随机加点噪声
            # root_state[:, :3] += torch.randn_like(root_state[:, :3]) * 0.05 
            target_obj.write_root_state_to_sim(root_state)
            target_obj.reset()

        # 3. 重置控制器
        self.controller.reset()
        
        # 4. 重置 Agent (清空动作队列)
        self.agent.reset()
        
        # 5. 重置计数器
        self.step_count = 0
        self.round_count += 1

        
        # 6. 刷新 Simulator
        self.scene.write_data_to_sim()
        self.sim.step()

        self.txts = [self.agent.get_txts_bbox_at()] * self.cfg.sim.num_envs

    def run(self):
        """主循环"""
        self.reset_env() # 初始重置

        # 设置相机的固定位姿 (Eye-to-hand)
        # 位置: [0.9, 0.4, 1.0], 看向目标中心
        camera_positions = torch.tensor([[0.9, 0.4, 1.0]] * self.cfg.sim.num_envs, device=self.cfg.sim.device)
        camera_targets = torch.tensor([TARGET_INIT_POS] * self.cfg.sim.num_envs, device=self.cfg.sim.device)
        self.camera.set_world_poses_from_view(camera_positions, camera_targets)
        cam_mats = [cal_cammat(camera_positions[0], camera_targets[0])] * self.cfg.sim.num_envs

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.goal_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        self.pt_marker = VisualizationMarkers(
            bbox_pts_cfg.replace(prim_path="/Visuals/bbox_points"))

        while simulation_app.is_running():
            # 判断是否需要开启新一轮
            if self.step_count >= self.max_steps_per_round:
                if self.round_count >= self.total_rounds:
                    print("[INFO] All rounds completed.")
                    break
                self.reset_env()

            # -----------------------------------------------------------
            # 1. 获取观测
            # -----------------------------------------------------------
            
            obs = {
                'rgb': self.camera.data.output['rgb'],
                'depth': self.camera.data.output['distance_to_image_plane'],
                'camera_pose': cam_mats, # (B, 4, 4)
                'intrinsic': self.camera.data.intrinsic_matrices,
                'instruction': self.txts
            }


            # -----------------------------------------------------------
            # 2. Agent 推理
            # -----------------------------------------------------------
            # Target Pose: (7,) [x, y, z, qx, qy, qz, qw]
            target_pose = self.agent.get_action(obs)

            # -----------------------------------------------------------
            # 3. IK 控制
            # -----------------------------------------------------------
            # 构造 IK command (Batch=1)
            ik_commands = target_pose.unsqueeze(0) 
            self.controller.set_command(ik_commands)

            self.goal_marker.visualize(
                ik_commands[:, 0:3] + self.scene.env_origins, ik_commands[:, 3:7])
            self.pt_marker.visualize(np.array(self.agent.bbox_info["center"]).reshape(-1, 3))

            # 计算 Jacobian 和当前状态
            ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]
            jacobian = self.robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
            ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
            root_pose_w = self.robot.data.root_pose_w
            joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

            # 相对 Pose 计算
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            
            # 计算关节位置目标
            joint_pos_des = self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            
            # -----------------------------------------------------------
            # 4. 执行与更新
            # -----------------------------------------------------------
            self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
            self.scene.write_data_to_sim()
            
            self.sim.step()
            self.scene.update(self.cfg.sim.dt)
            self.camera.update(self.cfg.sim.dt)
            
            self.step_count += 1


# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------
@hydra.main(config_path="conf", config_name="eval_config", version_base=None)
def main(cfg: DictConfig):
    # 1. Init Sim
    sim_cfg = sim_utils.SimulationCfg(dt=cfg.sim.dt, device=cfg.sim.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 1.2], TARGET_INIT_POS)

    # 2. Scene & Sensor & Controller
    scene_cfg = TableTopSceneCfg(num_envs=cfg.sim.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Camera setup (ensure prim paths match your scene logic)
    for i in range(cfg.sim.num_envs):
        prim_utils.create_prim(f"/World/Origin_{i:02d}", "Xform")

    camera_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0, height=480, width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, 
            horizontal_aperture=20.955, clipping_range=(0.1, 100.0)
        ),
    )
    camera = Camera(cfg=camera_cfg)
    
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # 3. Agent
    agent = VLDInferenceAgent(cfg, device=sim.device)

    # 4. Set Fixed Camera Pose (Eye-to-hand)
    sim.reset()
    camera_positions = torch.tensor([[0.9, 0.4, 1.0]] * cfg.sim.num_envs, device=sim.device)
    camera_targets = torch.tensor([TARGET_INIT_POS] * cfg.sim.num_envs, device=sim.device)
    camera.set_world_poses_from_view(camera_positions, camera_targets)

    # 5. Run Evaluation Loop
    runner = EvaluationRunner(cfg, sim, scene, agent, diff_ik_controller, camera)
    runner.run()

if __name__ == "__main__":
    main()
    simulation_app.close()