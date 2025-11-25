import h5py
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

intrinsic = np.array([[732.9993,   0.0000, 320.0000],
         [  0.0000, 732.9993, 240.0000],
         [  0.0000,   0.0000,   1.0000]])

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
        # for img in f["rgb"]:
        #     plt.figure(figsize=(10, 6))
        #     plt.imshow(img)
        #     plt.show()
        cam_mat = f["cam_mat"][:]
        print(cam_mat)
        for depth_isaac in f["distance_to_image_plane"]:
            bbox = f["scaled_bbox"][()].squeeze(1)

            depth_isaac = depth_isaac.squeeze(-1)
            plt.figure(figsize=(10, 6))
            plt.imshow(depth_isaac)
            plt.show()
            pcd_cam = depth_to_world_pointcloud(depth_isaac, intrinsic, cam_mat, to_world=True)
            try:
                import open3d as o3d
                print("\n🎨 正在启动 Open3D 可视化窗口...")
                print("   (按 'Q' 键退出窗口)")

                spheres_list = []
    
                for point in bbox:
                    # 1. 创建一个球体网格 (代替 PointCloud)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    
                    # 2. 将球体移动到对应的角点位置
                    sphere.translate(point)
                    
                    # 3. 设置颜色
                    sphere.paint_uniform_color([1, 0, 0])
                    
                    # 4. 为了平滑显示，计算一下法线
                    sphere.compute_vertex_normals()
                    
                    spheres_list.append(sphere)
                
                # 5. 将这 8 个球体合并成一个 Mesh 对象，方便后续调用 draw_geometries
                #    (Open3D 支持用 += 运算符合并 Mesh)
                combined_mesh = spheres_list[0]
                for i in range(1, len(spheres_list)):
                    combined_mesh += spheres_list[i]
                
                # 转为 Numpy CPU
                pcd_np = pcd_cam.cpu().numpy()  

                # 创建 Open3D 对象
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
                
                # 添加一个坐标轴 (红X, 绿Y, 蓝Z) 用作世界原点参考
                origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                
                # 为了展示相机位置，我们在相机位置也画个小坐标轴
                cam_pos_np = cam_mat[:3, 3]
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                cam_frame.translate(cam_pos_np)
                
                o3d.visualization.draw_geometries([pcd_o3d, origin_frame, cam_frame, combined_mesh], 
                                                window_name="Isaac Lab World Point Cloud")
            except ImportError:
                print("⚠️ 未安装 Open3D，无法进行点云可视化。请安装 open3d 库后重试。")


        for seg in f["semantic_segmentation"]:
            seg = torch.from_numpy(seg)
            rgb_channels = seg[..., :3]

            # 步骤 B: 创建掩码 (Mask)
            # 逻辑：在最后一个维度上(dim=-1)，只要有任何一个值(any)不等于0，就为 True
            # mask shape: (H, W)
            is_valid_mask = torch.any(rgb_channels != 0, dim=-1)

            # 也可以用 sum 方法 (对于 uint8 类型，sum > 0 等价于任何一个不为0)
            # is_valid_mask = rgb_channels.sum(dim=-1) > 0

            valid_coords = torch.nonzero(is_valid_mask, as_tuple=False)

            print(f"\n--- 2. 找到 {len(valid_coords)} 个有效像素的坐标 ---")
            plt.figure(figsize=(10, 6))
            plt.imshow(seg, cmap='jet', interpolation='nearest')
            plt.show()

def depth_to_world_pointcloud(depth_tensor, intrinsic_matrix, camera_pose_4x4, device="cuda", to_world=False):
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
    depth_tensor = torch.from_numpy(depth_tensor).float()
    intrinsic_matrix = torch.from_numpy(intrinsic_matrix).float()
    camera_pose_4x4 = torch.from_numpy(camera_pose_4x4).float()

    if depth_tensor.device.type != device:
        depth_tensor = depth_tensor.to(device)
    if intrinsic_matrix.device.type != device:
        intrinsic_matrix = intrinsic_matrix.to(device)
    if camera_pose_4x4.device.type != device:
        camera_pose_4x4 = camera_pose_4x4.to(device)
    H, W = depth_tensor.shape
    
    # 2. 提取内参
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # 3. 生成像素网格 (u, v)
    # indexing='ij' -> v(行/高), u(列/宽)
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
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
    
    correction_vector = torch.tensor([1.0, -1.0, -1.0], device=device)
    points = points_cam_cv * correction_vector
    
    if to_world:
        # 6. 转换到世界坐标系
        # 提取旋转 R (3x3) 和 平移 T (3)
        R = camera_pose_4x4[:3, :3]
        T = camera_pose_4x4[:3, 3]
        
        # 应用公式: P_world = R * P_local + T
        # 矩阵乘法注意: points 是 (N, 3), R 是 (3, 3)
        # 线性代数写法应为 (R @ points.T).T + T
        # 简化代码写法为 points @ R.T + T
        points = points @ R.T + T
    
    # 7. 过滤无效点 (可选)
    # 过滤掉深度为无穷大(天空)或 0 的点
    valid_mask = (z.reshape(-1) > 0.0) & (z.reshape(-1) < 1000.0)
    points = points[valid_mask]
    
    return points


print_h5_tree("/home/hhhar/liuliu/vld/data/regression/data_res/data_0001.h5")