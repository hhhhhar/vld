from transformers import BertTokenizer
from copy import deepcopy
from IPython import embed
import matplotlib.pyplot as plt
import h5py
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import unittest
import os
import shutil
import matplotlib
matplotlib.use("Qt5Agg")
import glob
import bisect
import tempfile

intrinsic = np.array([[732.9993,   0.0000, 320.0000],
                      [0.0000, 732.9993, 240.0000],
                      [0.0000,   0.0000,   1.0000]])


class H5RobotDataset(Dataset):
    def __init__(self, h5_dir, action_stats_path, chunk_size=10, num_bins=256, 
                 num_points=1024, vis=False, to_world=False, device='cpu'):
        """
        Args:
            h5_file_path (str): 单个 H5 文件的路径 (或者你可以改为接收文件列表)
            action_stats_path (str): 任务一生成的 json 路径
            num_bins (int): 动作离散化的桶数
            num_points (int): 点云采样的点数 (PointNet++ 需要固定点数)
            text_instruction (str): 既然H5里没有文本，我们暂时用统一的指令
        """
        self.h5_dir = h5_dir
        self.chunk_size = chunk_size
        self.num_bins = num_bins
        self.num_points = num_points
        self.vis = vis
        self.bbox = []
        self.action_type = []
        self.to_world = to_world
        self.device = device

        # 获取所有 H5 文件并排序 (确保跨平台顺序一致)
        self.h5_files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
        if len(self.h5_files) == 0:
            raise ValueError(f"No .h5 files found in {h5_dir}")

        # 加载统计量
        with open(action_stats_path, 'r') as f:
            stats = json.load(f)
        self.min_val = np.array(stats['min'])
        self.max_val = np.array(stats['max'])
        self.bins = np.linspace(-1, 1, num_bins)

        # 建立索引映射 (Global Index -> File + Local Index)
        # 我们需要知道每个文件的长度，以便在 __getitem__ 中定位
        self.cumulative_sizes = []
        self.total_len = 0
        
        print(f"Indexing {len(self.h5_files)} H5 files...")
        for h5_path in self.h5_files:
            # 只需要快速读取一次长度，不需要读取具体数据
            with h5py.File(h5_path, 'r') as f:
                seq_len = f['rgb'].shape[0]
                self.total_len += seq_len
                self.cumulative_sizes.append(self.total_len)
        
        print(f"Total frames: {self.total_len}")

    def __len__(self):
        return self.total_len

    def _locate_file(self, idx):
        """
        根据全局 index 找到 (文件路径, 文件内局部 index)
        使用二分查找提高效率
        """
        # 找到 idx 属于哪个区间
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[file_idx - 1]
            
        return self.h5_files[file_idx], local_idx

    def convert_bbox_8points_to_pose(self):
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
        points = np.array(self.bbox)

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

    def depth_to_world_pointcloud(self, depth, mask, intrinsic_matrix, camera_pose_4x4):
        """
        将 Isaac Lab 的 depth (distance_to_image_plane) 转换为世界坐标系点云。

        参数:
            depth_tensor (torch.Tensor): (H, W) 深度图
            intrinsic_matrix (torch.Tensor): (3, 3) 相机内参 K
            camera_pose_4x4 (torch.Tensor): (4, 4) 相机到世界的变换矩阵 (Camera-to-World Pose)
            device (str): 运行设备
            to_world (bool): 是否转换到世界坐标系 (默认 False，只返回相机坐标系下的点云)

        返回:
            torch.Tensor: (N, 3) 世界坐标系下的点云
        """
        # 1. 数据准备与设备移动
        cam_mat_np = deepcopy(camera_pose_4x4)
        depth_tensor = torch.from_numpy(depth).float()
        mask_tensor = torch.from_numpy(mask).float()
        intrinsic_matrix = torch.from_numpy(intrinsic_matrix).float()
        camera_pose_4x4 = torch.from_numpy(camera_pose_4x4).float()

        is_valid_mask = torch.any(mask_tensor != 0, dim=-1)
        depth_tensor = depth_tensor * is_valid_mask.float()

        H, W = depth_tensor.shape

        # 2. 提取内参
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

        # 3. 生成像素网格 (u, v)
        # indexing='ij' -> v(行/高), u(列/宽)
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )

        # 4. 反投影 (Back-projection) 到相机坐标系
        # 这里使用的是标准针孔相机模型 (OpenCV Convention)
        # 坐标系定义: +Z 前, +X 右, +Y 下
        z = depth_tensor
        x = (u_grid - cx) * z / fx
        y = (v_grid - cy) * z / fy

        # 堆叠为 (N, 3) 矩阵
        points_cam_cv = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        # 5. 【关键步骤】坐标系修正
        # Isaac Sim 的 Camera Prim (USD) 坐标系定义为: -Z 前, +Y 上, +X 右
        # 而上面的计算结果是: +Z 前, +Y 下, +X 右
        # 必须将点从 "CV Frame" 旋转到 "USD Camera Frame" 才能应用 pose 矩阵
        # 变换逻辑: x -> x, y -> -y, z -> -z

        correction_vector = torch.tensor([1.0, -1.0, -1.0])
        points = points_cam_cv * correction_vector

        if self.to_world:
            # 6. 转换到世界坐标系
            # 提取旋转 R (3x3) 和 平移 T (3)
            R = camera_pose_4x4[:3, :3]
            T = camera_pose_4x4[:3, 3]

            # 应用公式: P_world = R * P_local + T
            # 矩阵乘法注意: points 是 (N, 3), R 是 (3, 3)
            # 线性代数写法应为 (R @ points.T).T + T
            # 简化代码写法为 points @ R.T + T
            points = points @ R.T + T
        else:
            view_matrix = np.linalg.inv(cam_mat_np)
            R = view_matrix[:3, :3]
            T = view_matrix[:3, 3]
            self.bbox = self.bbox @ R.T + T

        self.bbox = self.convert_bbox_8points_to_pose()

        num_curr = points.shape[0]

        # 降采样或补全到固定点数 (num_points)
        if num_curr >= self.num_points:
            # 降采样: 无放回采样 (replace=False)
            # torch.randperm 生成 0 到 N-1 的随机排列，取前 num_points 个
            choice = torch.randperm(num_curr, device=points.device)[
                :self.num_points]
            points = points[choice, :]
        else:
            # 如果点不够，随机重复: 有放回采样 (replace=True)
            if num_curr > 0:
                # torch.randint 生成 (0, num_curr) 范围内的随机整数索引
                choice = torch.randint(
                    0, num_curr, (self.num_points,), device=points.device)
                points = points[choice, :]
            else:
                # 极端情况：全黑深度图，创建全零 Tensor
                points = torch.zeros((self.num_points, 3),
                                     dtype=points.dtype, device=points.device)

        if self.vis:
            try:
                import open3d as o3d
                print("\n🎨 正在启动 Open3D 可视化窗口...")
                print("   (按 'Q' 键退出窗口)")
                
                obb = o3d.geometry.OrientedBoundingBox()
                obb.center = self.bbox["center"]
                obb.extent = self.bbox["extent"]
                obb.R = self.bbox["matrix"]
                
                # 设置框的颜色 (例如红色线条)
                obb.color = (1, 0, 0)
                # 转为 Numpy CPU
                pcd_np = points.cpu().numpy()

                # 创建 Open3D 对象
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)

                # 添加一个坐标轴 (红X, 绿Y, 蓝Z) 用作世界原点参考
                origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=1.0, origin=[0, 0, 0])

                # 为了展示相机位置，我们在相机位置也画个小坐标轴
                cam_pos_np = camera_pose_4x4[:3, 3].cpu().numpy()
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.5)
                cam_frame.translate(cam_pos_np)

                o3d.visualization.draw_geometries([pcd_o3d, origin_frame, cam_frame, obb],
                                                  window_name="Isaac Lab World Point Cloud")
            except ImportError:
                print("⚠️ 未安装 Open3D，无法进行点云可视化。请安装 open3d 库后重试。")

        return points

    def _discretize_action(self, action):
        """连续动作 -> 离散 Token"""
        # 1. Normalize [-1, 1]
        norm_action = 2 * (action - self.min_val) / \
            (self.max_val - self.min_val + 1e-8) - 1
        norm_action = np.clip(norm_action, -1, 1)
        # 2. Binning
        idxs = np.digitize(norm_action, self.bins) - 1
        return np.clip(idxs, 0, self.num_bins - 1)

    def __getitem__(self, idx):
        # 在这里打开文件以支持多线程 DataLoader
        file_path, local_idx = self._locate_file(idx)

        with h5py.File(file_path, 'r') as f:
            self.bbox = np.array(f["scaled_bbox"][()]).reshape(8, 3)
            self.seq_len = f['rgb'].shape[0]  # 通常是 30
            self.action_type = f["action_type"][()].decode('utf-8')

            
            # 1. 读取 RGB
            # shape: (480, 640, 3) -> 转为 (3, 480, 640) 并归一化
            rgb = f['rgb'][local_idx]  # uint8
            if self.vis:
                plt.figure(figsize=(10, 6))
                plt.imshow(rgb)
                plt.show()

            image_tensor = torch.from_numpy(
                rgb).permute(2, 0, 1).float() / 255.0

            # 2. 读取深度并生成点云
            # H5 shape: (30, 480, 640, 1) -> 取 [local_idx, :, :, 0]
            depth = f['distance_to_image_plane'][local_idx, :, :, 0]
            mask = f["semantic_segmentation"][local_idx, :, :, :3]
            cam_mat = f['cam_mat'][:]
            pc_tensor = self.depth_to_world_pointcloud(
                depth, mask, intrinsic, cam_mat)  # (N, 3)

            # 3. 读取动作
             # 我们需要从 local_idx 开始，往后读取 chunk_size 帧
            end_idx = local_idx + self.chunk_size
            
            # f['poses'] shape: (Seq_Len, 1, 7)
            
            if end_idx <= self.seq_len:
                # 情况 A: 剩余长度足够，直接切片
                # 取出 (Chunk, 1, 7) -> Squeeze -> (Chunk, 7)
                raw_actions_chunk = f['poses'][local_idx : end_idx, 0, :]
            else:
                # 情况 B: 接近视频末尾，长度不够 -> 需要 Padding
                # 策略: 复制最后一帧的动作来填充
                
                # 1. 先读出所有剩下的有效帧
                valid_len = self.seq_len - local_idx
                if valid_len > 0:
                    valid_actions = f['poses'][local_idx : self.seq_len, 0, :] # (Valid, 7)
                    last_action = valid_actions[-1]
                else:
                    # 极端情况 (虽然理论上不会进这里)，全部用 0 填充
                    valid_actions = np.zeros((0, 7))
                    last_action = np.zeros(7) # 或者上一帧动作

                # 2. 构造 Padding
                pad_len = self.chunk_size - valid_len
                pad_block = np.tile(last_action, (pad_len, 1)) # 重复最后一帧
                
                # 3. 拼接
                raw_actions_chunk = np.concatenate([valid_actions, pad_block], axis=0)

            # 离散化
            action_tokens = self._discretize_action(raw_actions_chunk)
            action_tensor = torch.LongTensor(action_tokens) # (Chunk, 7)

        # 4. 文本 (构造 BERT 需要的输入，这里只是返回 raw text，需要在 collate_fn 里 tokenize)
        bbox_info = deepcopy(self.bbox)
        del bbox_info["matrix"]  # 删除矩阵，保留中心、尺寸、四元数
        text = f"action: {self.action_type}, target_area: {bbox_info}"

        return {
            "image": image_tensor,
            "point_cloud": pc_tensor,
            "text": text,  # 原始文本，稍后由 Tokenizer 处理
            "action_tokens": action_tensor
        }


# ================================
# DataLoader 的 Collate Function
# ================================
# 因为 BERT Tokenizer 不能在 Dataset 里直接做（效率低），通常在 Batch 组装时做


class VLDCollator:
    def __init__(self, tokenizer_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, batch):
        images = torch.stack([item['image'] for item in batch])
        point_clouds = torch.stack([item['point_cloud'] for item in batch])
        action_tokens = torch.stack([item['action_tokens'] for item in batch])

        # 处理文本 Batch Tokenization
        raw_texts = [item['text'] for item in batch]
        text_inputs = self.tokenizer(
            raw_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return images, point_clouds, text_inputs, action_tokens


class TestH5RobotDatasetChunked(unittest.TestCase):
    def setUp(self):
        """
        测试前的准备工作：
        1. 创建临时文件夹
        2. 生成假的 action_stats.json
        3. 生成假的 .h5 数据文件 (包含 RGB, Depth, Pose 等)
        """
        # 1. 创建临时目录
        self.test_dir = tempfile.mkdtemp()
        self.h5_dir = os.path.join(self.test_dir, "data")
        os.makedirs(self.h5_dir, exist_ok=True)
        self.stats_path = os.path.join(self.test_dir, "action_stats.json")

        # 2. 创建 Dummy action_stats.json
        stats = {
            "min": [-1.0] * 7,
            "max": [1.0] * 7
        }
        with open(self.stats_path, 'w') as f:
            json.dump(stats, f)

        # 3. 创建 Dummy H5 文件
        # 模拟一个长度为 30 帧的视频序列
        self.seq_len = 30
        self.chunk_size = 10
        self.n_joints = 7
        self.h5_path = os.path.join(self.h5_dir, "test_01.h5")
        
        with h5py.File(self.h5_path, 'w') as f:
            # RGB: (Seq, H, W, 3) - 纯黑图片
            f.create_dataset("rgb", data=np.zeros((self.seq_len, 480, 640, 3), dtype=np.uint8))
            
            # Depth: (Seq, H, W, 1) - 全为 1.0 的深度
            f.create_dataset("distance_to_image_plane", data=np.ones((self.seq_len, 480, 640, 1), dtype=np.float32))
            
            # Seg Mask: (Seq, H, W, 3) - 全 1 表示全是有效像素
            f.create_dataset("semantic_segmentation", data=np.ones((self.seq_len, 480, 640, 3), dtype=np.uint8))
            
            # Cam Mat (相机内参矩阵占位)
            f.create_dataset("cam_mat", data=np.eye(4, dtype=np.float32))
            
            # Poses (Actions): (Seq, 1, 7) 
            # 构造有规律的数据：第 i 帧的所有关节值都是 i/100
            # 这样我们在测试时可以通过数值判断它是哪一帧
            poses = np.zeros((self.seq_len, 1, self.n_joints), dtype=np.float32)
            for i in range(self.seq_len):
                poses[i, :, :] = float(i) / 100.0 
            f.create_dataset("poses", data=poses)
            
            # Metadata
            f.create_dataset("action_type", data=np.string_("pick_test"))
            # BBox (1, 8, 3)
            f.create_dataset("scaled_bbox", data=np.zeros((1, 8, 3), dtype=np.float32))

        # 4. 初始化 Dataset 实例
        print(f"\n[SetUp] Initializing Dataset with chunk_size={self.chunk_size}...")
        self.dataset = H5RobotDataset(
            h5_dir=self.h5_dir,
            action_stats_path=self.stats_path,
            chunk_size=self.chunk_size,
            num_bins=256,
            num_points=1024,
            vis=False,
            to_world=False,
            device='cpu' # 显式测试 CPU 模式
        )

    def tearDown(self):
        """测试结束后的清理：删除临时文件夹"""
        shutil.rmtree(self.test_dir)

    def test_01_len(self):
        """测试 1: 数据集总长度是否正确"""
        self.assertEqual(len(self.dataset), self.seq_len, 
                         f"Dataset length should be {self.seq_len}")

    def test_02_getitem_shapes(self):
        """测试 2: 返回数据的张量形状是否符合 Diffusion 要求"""
        idx = 0
        data = self.dataset[idx]
        
        # Image
        self.assertEqual(data['image'].shape, (3, 480, 640))
        self.assertTrue(isinstance(data['image'], torch.Tensor))
        
        # Point Cloud
        self.assertEqual(data['point_cloud'].shape, (1024, 3))
        
        # Action (关键检查点)
        # 期望形状: (Chunk_Size, Joints) 即 (10, 7)
        self.assertEqual(data['action_tokens'].shape, (self.chunk_size, self.n_joints),
                         f"Action shape mismatch! Expected ({self.chunk_size}, 7)")
        self.assertTrue(isinstance(data['action_tokens'], torch.Tensor))
        
        # Text
        self.assertTrue(isinstance(data['text'], str))

    def test_03_padding_logic(self):
        """测试 3: 边界情况 (Padding) 是否正确"""
        # 选取最后一个索引 (idx = 29)
        # 此时只能读取 1 帧有效数据，剩下 9 帧应该由 Padding 填充
        idx = self.seq_len - 1
        data = self.dataset[idx]
        
        actions = data['action_tokens'] # shape (10, 7)
        
        # 验证形状
        self.assertEqual(actions.shape, (self.chunk_size, self.n_joints))
        
        # 验证 Padding 值
        # 第 0 个元素是真实的最后一帧动作
        # 第 1 到 9 个元素应该是复制的第 0 个元素
        first_token = actions[0]
        last_token = actions[-1]
        
        # 检查是否所有行的值都相等 (这意味着 Padding 成功复制了最后一帧)
        are_equal = torch.all(actions == first_token)
        self.assertTrue(are_equal, "Padding logic failed: padded frames do not match the last valid frame.")
        print(f"[Test Padding] Index {idx}: Action Chunk returned successfully with padding.")

    def test_04_sliding_window(self):
        """测试 4: 正常情况下的滑动窗口读取"""
        idx = 0
        data = self.dataset[idx]
        actions = data['action_tokens']
        
        # 由于我们造的数据是 poses[i] = i/100
        # 第 0 个 chunk 应该包含 pose[0] 到 pose[9]
        # 对应的 token 值应该是不一样的 (除非 num_bins 太小导致分辨率不够，但这里假设足够)
        
        # 简单的检查：确保 chunk 里的数据不是全部一样的 (证明读到了序列)
        # 注意：如果 bin 只有 1 个，那确实会一样，所以这里只是个弱检查
        if self.dataset.num_bins > 10:
            is_all_same = torch.all(actions == actions[0])
            self.assertFalse(is_all_same, "Normal chunking failed: all actions in chunk are identical (expected variation).")

    def test_05_cpu_device_check(self):
        """测试 5: 验证 Dataset 返回的数据确实在 CPU 上 (防止多进程报错)"""
        idx = 5
        data = self.dataset[idx]
        self.assertEqual(data['image'].device.type, 'cpu')
        self.assertEqual(data['point_cloud'].device.type, 'cpu')
        self.assertEqual(data['action_tokens'].device.type, 'cpu')

    def test_06_dataloader_multiprocessing(self):
        """测试 6: 集成测试 - 放入 DataLoader 并开启多进程"""
        print("\n[Test DataLoader] Testing with num_workers=2...")
        
        # collate_fn 简单 mock 一下，因为 text 是 string，默认 collate 可能会报错或者变成 list
        def simple_collate(batch):
            return {
                'image': torch.stack([b['image'] for b in batch]),
                'point_cloud': torch.stack([b['point_cloud'] for b in batch]),
                'action_tokens': torch.stack([b['action_tokens'] for b in batch]),
                'text': [b['text'] for b in batch]
            }

        loader = DataLoader(
            self.dataset, 
            batch_size=4, 
            shuffle=True, 
            num_workers=2, # 开启多进程！这是最容易报错的地方
            collate_fn=simple_collate
        )
        
        # 尝试读取一个 batch
        try:
            for batch in loader:
                # 检查 Batch 维度
                expected_shape = (4, self.chunk_size, self.n_joints)
                self.assertEqual(batch['action_tokens'].shape, expected_shape)
                print("[Test DataLoader] Successfully loaded a batch with multiprocessing.")
                break 
        except RuntimeError as e:
            self.fail(f"DataLoader failed with multiprocessing: {e}")

if __name__ == '__main__':
    # unittest.main()
    h5_path = "/home/hhhar/liuliu/vld/data/regression/data_res/"
    stats_path = "/home/hhhar/liuliu/vld/dataset/action_stats.json"
    dataset = H5RobotDataset(h5_path, stats_path, num_bins=256, vis=True,to_world=True)
    for i in range(len(dataset)):
        item = dataset[i]
        print(
            f"Item {i}: Image {item['image'].shape}, PC {item['point_cloud'].shape}, Action {item['action_tokens']}")
