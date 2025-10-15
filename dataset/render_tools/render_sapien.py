import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
from transforms3d.euler import mat2euler
import keyboard


# 创建引擎和渲染器
engine = sapien.Engine()
renderer = sapien.VulkanRenderer()
engine.set_renderer(renderer)

# 创建场景
scene = engine.create_scene()
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([1, -1, -1], [0.5, 0.5, 0.5])

# 创建 Viewer 窗口
viewer = Viewer(renderer)
viewer.set_scene(scene)

current_index = -1  # 当前显示的包围盒索引


# 主循环
def main_cycle():
    while not viewer.closed:
        scene.step()  # 可选：推进物理模拟
        scene.update_render()
        viewer.render()


def load_urdf_in_sapien(
    urdf_path: str,
    target_pos: np.ndarray = np.array([0.0, 0.0, 0.0]),
):

    # 设置摄像机位置
    camera_pos = target_pos + np.array([-2.0, 0.0, 0.0])
    viewer.set_camera_xyz(*camera_pos)  # 设置摄像机位置

    # 计算朝向向量
    forward = target_pos - camera_pos
    forward /= np.linalg.norm(forward)

    # 定义上方向
    up = np.array([0, 0, 1])

    # 计算左方向
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)

    # 重新计算正交的上方向
    up = np.cross(forward, left)

    # 构建旋转矩阵
    rotation_matrix = np.stack([forward, left, up], axis=1)

    # 将旋转矩阵转换为欧拉角
    rpy = mat2euler(rotation_matrix) * np.array([1, -1, -1])  # SAPIEN 的坐标系调整

    viewer.set_camera_rpy(*rpy)  # 设置摄像机角度

    # 创建 URDF Loader
    loader = sapien.URDFLoader(scene)
    robot = loader.load(urdf_path)

    # 设置初始姿态
    robot.set_pose(sapien.Pose(p=target_pos))

    # bounding_boxes = create_box(robot)
    main_cycle()


def compute_aabb(link: sapien.LinkBase):
    global_vertices = []
    for visual_body in link.get_visual_bodies():
        shapes = visual_body.get_render_shapes()
        for shape in shapes:
            # 将顶点转换到全局坐标系
            mesh = shape.mesh
            vertices = mesh.vertices.tolist()
            # global_vertices += [link_pose.transform_point(v) for v in vertices]
            global_vertices += vertices

    min_coord = [float("inf")] * 3
    max_coord = [float("-inf")] * 3
    for v in global_vertices:
        for i in range(3):
            min_coord[i] = min(min_coord[i], v[i])
            max_coord[i] = max(max_coord[i], v[i])
    return np.array(min_coord), np.array(max_coord)


def create_box(urdf_model: sapien.Articulation) -> list:
    links = urdf_model.get_links()
    # 创建材质
    material = renderer.create_material()
    material.base_color = [1, 0, 0, 0.3]  # 红色，透明度为0.3

    # 存储包围盒
    bounding_boxes = []
    for link in links:
        # 获取链接的 AABB
        lower, upper = compute_aabb(link)
        center = (lower + upper) / 2
        size = upper - lower

        # 创建包围盒 actor
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=size / 2, material=material)
        box_actor = builder.build_static(name=f"bbox_{link.name}")
        # box_actor.set_pose(sapien.Pose(p=center, q=np.array([0.5, 0.5, -0.5, -0.5])))
        box_actor.set_pose(sapien.Pose(p=center))
        # box_actor.hide_visual()
        # box_actor.set_visibility(False)  # 初始隐藏
        bounding_boxes.append(box_actor)

    return bounding_boxes


def setup_sapien_window():

    # 创建一个 SAPIEN 引擎（RendererType 默认为 PXR）
    engine = sapien.Engine()  # 或者使用: sapien.Engine(renderer_type='client')

    # 创建一个渲染器
    renderer = sapien.VulkanRenderer()
    engine.set_renderer(renderer)

    # 创建一个场景
    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)  # 设置仿真时间步长

    # 创建一个窗口（viewer）
    viewer = Viewer(renderer)  # 创建 Vulkan 渲染的 Viewer 窗口
    viewer.set_scene(scene)  # 将场景绑定到 Viewer
    viewer.set_camera_xyz(0, 0, 0)  # 设置摄像机位置
    viewer.set_camera_rpy(0, 0, 0)  # 设置摄像机角度

    # 添加一些物体（示例）
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.1, 0.1, 0.1])
    builder.add_box_visual(half_size=[0.1, 0.1, 0.1])
    box = builder.build(name="box")
    box.set_pose(sapien.Pose([0, 0, 0.1]))

    main_cycle()


if __name__ == "__main__":
    # setup_sapien_window()
    urdf_path = "/home/hhhar/liuliu/gpt_test/dataset/partnet/100392/mobility_annotation_gapartnet.urdf"
    load_urdf_in_sapien(urdf_path)
