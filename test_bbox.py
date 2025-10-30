import os.path as osp
import re
import sapien.core as sapien
from transforms3d.quaternions import mat2quat
import imageio
import numpy as np
import os
import json
from typing import Dict, Tuple, List
from scipy.spatial.transform import Rotation as R_obj
import cv2
import base64


class BBOXExtractor:
    def __init__(
            self,
            urdf_path: str,
            annotation_path: str,
            use_ai: bool = False,
            ai_api_key: str = None,
            target_pos: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ):
        self.urdf_path = urdf_path
        self.annotation_path = annotation_path
        self.use_ai = use_ai
        self.ai_api_key = ai_api_key or os.getenv("DASHSCOPE_API_KEY")
        self.target_pos = target_pos

        # 加载 bbox 注释
        self.bboxes_dict = self._load_bbox_annotations()

        # 初始化 SAPIEN（只初始化一次）
        self._setup_sapien()

        # 设置保存目录
        base_dir = os.path.dirname(urdf_path)
        self.rgb_dir = os.path.join(base_dir, "bbox", "rgb")
        self.rgbd_dir = os.path.join(base_dir, "bbox", "rgbd")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.rgbd_dir, exist_ok=True)

        # 使用Qwen标注细节
        if self.use_ai:
            if not self.ai_api_key:
                raise ValueError(
                    "use_ai=True 但未提供 ai_api_key 或 DASHSCOPE_API_KEY 环境变量")
            from openai import OpenAI
            self.ai_client = OpenAI(
                api_key=self.ai_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.summary_path = os.path.join(
                self.rgb_dir, "..", "usage_summary.json")
            # 初始化汇总文件（如果不存在）
            if not os.path.exists(self.summary_path):
                with open(self.summary_path, 'w') as f:
                    json.dump([], f)

        # 处理每个 link
        for link_name, (bbox_points, category) in self.bboxes_dict.items():
            # 创建 link_name 子目录
            link_rgb_dir = os.path.join(self.rgb_dir, link_name)
            link_rgbd_dir = os.path.join(self.rgbd_dir, link_name)
            os.makedirs(link_rgb_dir, exist_ok=True)
            os.makedirs(link_rgbd_dir, exist_ok=True)

            # 渲染并保存bbox图像
            bbox_actors = self._create_bounding_box(
                bbox_points, name=link_name)
            image_path = self._render_and_save_image(link_name, link_rgb_dir)

            # 调用Qwen对单个link图片用途进行分析
            if self.use_ai:
                ai_result = self._query_ai_for_usage(
                    image_path, link_name, category)

                # 保存 usage.json
                usage_json_path = os.path.join(link_rgb_dir, "usage.json")
                with open(usage_json_path, 'w', encoding='utf-8') as f:
                    json.dump(ai_result, f, ensure_ascii=False, indent=2)
                print(f"✅ AI 回复已保存到 {usage_json_path}")

                # 更新汇总文件：使用 ai_result 中的 function 和 label
                with open(self.summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)

                summary.append({
                    "link_name": link_name,
                    "category": category,
                    "function": ai_result["function"],
                    "label": ai_result["label"]
                })

                with open(self.summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                print(f"✅ 汇总信息已更新到 {self.summary_path}")

            for actor in bbox_actors:
                self.scene.remove_actor(actor)

            # # 计算 3D 和 2D bbox
            corners_3d = np.array(bbox_points)
            bbox_3d = self.calculate_3d_bbox_parameters(corners_3d)
            cx, cy, w, h, angle, contour_2d, rect_2d = self.calculate_2d_bbox_parameters(
                corners_3d)

            # 保存原始 3D bbox points 及 3D JSON文件
            self._save_3d_bbox_points(bbox_points, link_name, link_rgbd_dir)
            self._save_3d_json(bbox_3d, link_name, category, link_rgbd_dir)

            # 保存 2D bbox 轮廓点 ,外接矩形角点及 2D JSON文件
            self._save_2d_contour_points(contour_2d, link_name, link_rgb_dir)
            bbox_2d = (cx, cy, w, h, angle)
            self._save_2d_json(bbox_2d, link_name, link_rgb_dir)
            self._save_2d_rect_points(rect_2d, link_name, link_rgb_dir)

    def _load_bbox_annotations(self) -> Dict[str, Tuple[list, str]]:
        """读取json里的bbox，返回字典 {link_name: (bbox_points, category)}"""
        with open(self.annotation_path, 'r') as f:
            annotations = json.load(f)

        bboxes = {}
        for annotation in annotations:
            if annotation.get("is_gapart", False) and annotation.get("bbox"):
                link_name = annotation["link_name"]
                # list of 8 points, each [x,y,z]
                bbox_points = annotation["bbox"]
                category = annotation["category"]
                bboxes[link_name] = (bbox_points, category)
        return bboxes

    def _setup_sapien(self):
        """设置 SAPIEN 引擎、场景、相机"""

        # sapien引擎配置
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)

        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(1 / 240.0)

        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        # 相机配置
        camera_pos = self.target_pos + np.array([-2.0, 0.0, 0.0])
        forward = self.target_pos - camera_pos
        forward /= np.linalg.norm(forward)
        up = np.array([0, 0, 1])
        left = np.cross(up, forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        rotation_matrix = np.stack([forward, left, up], axis=1)

        # 创建渲染相机
        self.render_camera = self.scene.add_camera(
            name="render_camera",
            width=1920,
            height=1080,
            fovy=np.deg2rad(60),
            near=0.1,
            far=100.0
        )
        quat = mat2quat(rotation_matrix)  # [w, x, y, z]
        self.render_camera.set_pose(sapien.Pose(p=camera_pos, q=quat))

        # 加载 URDF
        loader = self.scene.create_urdf_loader()
        self.robot = loader.load(self.urdf_path)
        self.robot.set_pose(sapien.Pose(p=self.target_pos))

    def _create_bounding_box(self, bbox_points, name="") -> List:
        """创建 bbox 可视化（8 个小方块）"""
        cube_size = 0.03
        half_size = cube_size / 2
        colors = [[1, 0, 0, 0.8]]  # 红色半透明

        actors = []
        for i, point in enumerate(bbox_points):
            point = np.array(point)
            color = colors[i % len(colors)]

            builder = self.scene.create_actor_builder()
            builder.add_box_visual(
                pose=sapien.Pose(p=point),
                half_size=[half_size, half_size, half_size],
                name=f"{name}_vertex_{i}"
            )
            actor = builder.build_static(name=f"{name}_vertex_{i}")
            actors.append(actor)
        return actors

    def _render_and_save_image(self, link_name: str, link_rgb_dir: str) -> str:
        """渲染并保存第25帧图像（只保存当前 link_name）"""
        for frame_id in range(26):
            self.scene.step()
            self.scene.update_render()
            if frame_id == 25:
                self.render_camera.take_picture()
                albedo = self.render_camera.get_picture('Albedo')
                albedo_img = (albedo[..., :3] * 255).astype(np.uint8)

                # 只保存当前 link_name 的图像
                out_path = os.path.join(link_rgb_dir, f"{link_name}.png")
                imageio.imwrite(out_path, albedo_img)
                print(f"✅ 保存 {link_name} 的第25帧图像到 {out_path}")
                return out_path

    def _image_to_base64_url(self, image_path: str) -> dict:
        """图像转为Base64 url"""
        ext = os.path.splitext(image_path)[-1].lower()
        mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return f"data:{mime};base64,{encoded}"

    def _query_ai_for_usage(self, image_path: str, link_name: str, category: str) -> dict:
        """调用 AI 模型，强制返回 {'function': str, 'label': str} 格式的字典（英文）"""
        try:
            base64_url = self._image_to_base64_url(image_path)
            prompt = (
                "请问图片中bbox（4个红色方块）圈出来的部件的功能是什么？\n"
                "我将要将你回复的数据用于对这个按键的功能数据标注，请注意你回复内容的格式，\n"
                "应该是这样：{'function':'xxxxx','label':'xxxxx'}。\n"
                "请用英文回复，并且注意精细一点。不要包含任何其他文字、解释或 Markdown。"
            )
            completion = self.ai_client.chat.completions.create(
                model="qwen3-vl-plus",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": base64_url}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
            raw_text = completion.choices[0].message.content.strip()

            # 清理可能的 Markdown 代码块（如 ```json 或 ```）
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[-1]  # 去掉第一行 ```json 或 ```
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3].strip()

            # 尝试直接解析 JSON
            try:
                result = json.loads(raw_text)
            except json.JSONDecodeError:
                # 如果失败，尝试修复单引号 → 双引号（因为 prompt 用了单引号示例）
                fixed_text = raw_text.replace("'", '"')
                try:
                    result = json.loads(fixed_text)
                except json.JSONDecodeError:
                    # 再尝试用正则提取
                    func_match = re.search(
                        r'"?function"?\s*:\s*"([^"]*)"', fixed_text)
                    label_match = re.search(
                        r'"?label"?\s*:\s*"([^"]*)"', fixed_text)
                    if func_match and label_match:
                        result = {
                            "function": func_match.group(1),
                            "label": label_match.group(1)
                        }
                    else:
                        raise ValueError(f"无法解析 AI 输出: {raw_text}")

            # 验证字段
            if not isinstance(result.get("function"), str) or not isinstance(result.get("label"), str):
                raise ValueError("function 或 label 不是字符串")

            return {
                "function": result["function"].strip(),
                "label": result["label"].strip()
            }

        except Exception as e:
            print(f"⚠️ AI 查询失败 ({link_name}): {e}")
            return {
                "function": "Failed to analyze functionality",
                "label": "UNKNOWN"
            }

    def _save_3d_bbox_points(self, bbox_points: list, link_name: str, link_rgbd_dir: str):
        """保存原始的8个角点数据"""
        data = {
            "bbox_points": bbox_points,
            "num_points": len(bbox_points)
        }
        json_path = os.path.join(link_rgbd_dir, f"{link_name}_points.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ 保存 3d bbox points 数据到 {json_path}")

    def _save_2d_contour_points(self, contour_points: np.ndarray, link_name: str, link_rgb_dir: str):
        """保存2D凸包轮廓点数据"""
        # 将numpy数组转换为列表，便于JSON序列化
        contour_list = contour_points.tolist()

        data = {
            "contour_points": contour_list,
            "num_points": len(contour_list)
        }
        json_path = os.path.join(link_rgb_dir, f"{link_name}_2d_contour.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ 保存 2D contour points 数据到 {json_path}")

    def _save_2d_rect_points(self, rect_points: np.ndarray, link_name: str, link_rgb_dir: str):
        """保存2D外接矩形角点数据"""
        # 将numpy数组转换为列表，便于JSON序列化
        rect_list = rect_points.tolist()

        data = {
            "rect_points": rect_list,
            "num_points": len(rect_list)
        }
        json_path = os.path.join(link_rgb_dir, f"{link_name}_2d_rect.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ 保存 2D rectangle points 数据到 {json_path}")

    def calculate_3d_bbox_parameters(self, corners_3d: np.ndarray) -> Tuple[float, ...]:
        """计算3Dbbox需要的数据内容"""

        # 1. 根据传入的8个角点获得中心x,y,z的位置
        center = np.mean(corners_3d, axis=0)
        x, y, z = center[0], center[1], center[2]

        # 2.根据8个角点计算出整个bbox的dx,dy,dz
        min_coords = np.min(corners_3d, axis=0)
        max_coords = np.max(corners_3d, axis=0)
        dx = max_coords[0] - min_coords[0]  # x方向尺寸
        dy = max_coords[1] - min_coords[1]  # y方向尺寸
        dz = max_coords[2] - min_coords[2]  # z方向尺寸

        # 3.根据目前的8个角点的位置算出整个bbox相对，z,x,y三个方向的角度
        # 计算的角点相对于中心的位置
        half_dx, half_dy, half_dz = dx / 2, dy / 2, dz / 2
        standard_vertices = np.array([
            [-half_dx, -half_dy, -half_dz],
            [half_dx, -half_dy, -half_dz],
            [half_dx, half_dy, -half_dz],
            [-half_dx, half_dy, -half_dz],
            [-half_dx, -half_dy, half_dz],
            [half_dx, -half_dy, half_dz],
            [half_dx, half_dy, half_dz],
            [-half_dx, half_dy, half_dz]
        ])

        # 实际的角点相对与中心的位置
        actual_vertices_centered = corners_3d - center

        # 计算协方差矩阵
        H = standard_vertices.T @ actual_vertices_centered

        # SVD分解
        U, S, Vt = np.linalg.svd(H)

        # 计算旋转矩阵
        R_matrix = Vt.T @ U.T

        # 处理反射情况（确保是纯旋转，不是镜像）
        if np.linalg.det(R_matrix) < 0:
            Vt[2, :] *= -1
            R_matrix = Vt.T @ U.T

        # 从旋转矩阵提取欧拉角 (Z(yaw)-Y(picth)-X(roll) 顺序)
        rotation = R_obj.from_matrix(R_matrix)
        yaw, pitch, roll = rotation.as_euler('zyx', degrees=False)

        return (
            float(x), float(y), float(z),  # center
            float(dx), float(dy), float(dz),  # dimensions
            float(yaw), float(pitch), float(roll),  # euler angles
            R_matrix.copy()  # rotation matrix (3x3)
        )

    def _project_3d_to_2d(self, corners_3d: np.ndarray) -> np.ndarray:
        """将3D点投影到2D像素坐标"""
        intrinsic = self.render_camera.get_intrinsic_matrix()
        extrinsic = self.render_camera.get_extrinsic_matrix()

        # 转换为齐次坐标
        corners_h = np.hstack([corners_3d, np.ones((corners_3d.shape[0], 1))])

        # 应用外参：世界坐标 → 相机坐标
        corners_cam = (extrinsic @ corners_h.T).T
        corners_cam = corners_cam[:, :3] / corners_cam[:, 2:3]  # 归一化

        # 应用内参：相机坐标 → 像素坐标
        corners_pix = (intrinsic @ corners_cam.T).T
        return corners_pix[:, :2]  # (N, 2)

    def calculate_2d_bbox_parameters(self, corners_3d: np.ndarray) -> Tuple[
            float, float, float, float, float, np.ndarray, np.ndarray]:
        """从3D边界框的8个角点计算对应的2D最小外接矩形，并返回轮廓点和外接矩形角点。"""

        if corners_3d.shape != (8, 3):
            raise ValueError(
                f"Expected corners_3d shape (8, 3), got {corners_3d.shape}")

        # 1. 投影3D角点到2D图像平面
        points_2d = self._project_3d_to_2d(corners_3d)  # shape: (8, 2)

        # 2. 计算凸包（通常为4个点，但保留通用性）
        hull = cv2.convexHull(points_2d.astype(np.float32))
        silhouette_2d = hull.squeeze(axis=1)  # shape: (K, 2), K >= 3

        if len(silhouette_2d) < 3:
            # 退化情况：所有点重合或共线，回退到轴对齐包围盒
            x_min, y_min = points_2d.min(axis=0)
            x_max, y_max = points_2d.max(axis=0)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min
            angle = 0.0
            # 在退化情况下，使用所有投影点作为轮廓
            final_contour = points_2d
            # 轴对齐矩形的4个角点
            rect_points = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ])
        else:
            # 3. 计算最小外接矩形
            rect = cv2.minAreaRect(silhouette_2d.astype(np.float32))
            (cx, cy), (w, h), angle_deg = rect

            # 4. 转换角度为弧度
            angle = np.deg2rad(angle_deg)

            # 5. 确保 w >= h，并相应调整角度
            if w < h:
                w, h = h, w
                angle += np.pi / 2

            # 6. 规范化角度到 [-π/2, π/2)
            angle = (angle + np.pi / 2) % np.pi - np.pi / 2

            # 使用凸包点作为最终轮廓
            final_contour = silhouette_2d

            # 获取外接矩形的4个角点
            rect_points = cv2.boxPoints(rect)  # shape: (4, 2)

        # 返回bbox参数、轮廓点和外接矩形角点
        return float(cx), float(cy), float(w), float(h), float(angle), final_contour, rect_points

    def _save_2d_json(self, bbox_2d: Tuple, link_name: str, link_rgb_dir: str):
        """保存2D bbox数据"""
        cx, cy, w, h, angle = bbox_2d
        data = {
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "theta": angle
        }
        json_path = os.path.join(link_rgb_dir, f"{link_name}_2d.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ 保存 2D bbox 数据到 {json_path}")

    def _save_3d_json(self, bbox_3d: Tuple, link_name: str, category: str, link_rgbd_dir: str):
        """保存3D bbox数据（包含欧拉角和旋转矩阵）"""
        # 解包10个返回值
        x, y, z, dx, dy, dz, yaw, pitch, roll, R_matrix = bbox_3d

        # 将旋转矩阵转换为列表格式（JSON可序列化）
        R_list = R_matrix.tolist()  # 3x3 嵌套列表

        data = {
            "center": {
                "x": float(x),
                "y": float(y),
                "z": float(z)
            },
            "dimensions": {
                "dx": float(dx),
                "dy": float(dy),
                "dz": float(dz)
            },
            "euler_angles": {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll)
            },
            "rotation_matrix": R_list,  # 3x3 矩阵
            "category": category
        }

        # 保存到 link_rgbd_dir 目录下
        json_path = os.path.join(link_rgbd_dir, f"{link_name}_3d.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ 保存 3D bbox 数据到 {json_path}")

# ================================== 批处理函数 ==========================================


def process_all_scenes(data_root: str, use_ai=False, ai_api_key=None):
    for name in os.listdir(data_root):
        scene_dir = os.path.join(data_root, name)
        if not os.path.isdir(scene_dir):
            continue

        ann_file = os.path.join(scene_dir, "link_annotation_gapartnet.json")
        urdf_file = os.path.join(
            scene_dir, "mobility_annotation_gapartnet.urdf")

        if not (os.path.exists(ann_file) and os.path.exists(urdf_file)):
            continue

        print(f"🚀 处理: {name}")
        try:
            BBOXExtractor(
                annotation_path=ann_file,
                urdf_path=urdf_file,
                use_ai=use_ai,
                ai_api_key=ai_api_key
            )
            print(f"✅ {name} 完成")
        except Exception as e:
            print(f"❌ {name} 失败: {e}")


# ====================== 单个运行示例 =================================
prefix = '/mnt/4dba1798-fc0d-4700-a472-04acb2f7b630/hhhar/partnet'
target_id = 12727
extractor = BBOXExtractor(
    annotation_path=osp.join(
        prefix, f'{target_id}', "link_annotation_gapartnet.json"),
    urdf_path=osp.join(
        prefix, f'{target_id}', "mobility_annotation_gapartnet.urdf"),
    use_ai=False,
    ai_api_key=None,
)

# ====================== 批量运行示例 ======================================
# process_all_scenes(
#     data_root="/home/yilin-510/PycharmProjects/TargetProject/data",
#     use_ai=True,
#     ai_api_key = "sk-3365395806bf407cab0bba4064dd9546",
# )
