import open3d as o3d
import numpy as np
import xml.etree.ElementTree as ET
from os.path import join


class URDFRenderer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.geometries = []

    def parse_urdf(self, urdf_path, obj_dir):
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # 首先构建link和joint的映射
        self.links = {link.get("name"): link for link in root.findall("link")}
        self.joints = {joint.get("name"): joint for joint in root.findall("joint")}

        # 找到根link（没有父joint的link）
        root_links = set(self.links.keys())
        for joint in self.joints.values():
            child_link = joint.find("child").get("link")
            if child_link in root_links:
                root_links.remove(child_link)

        # 从根link开始递归构建场景图
        for link_name in root_links:
            self.process_link(link_name, obj_dir, np.eye(4))

    def process_link(self, link_name, obj_dir, parent_transform):
        link = self.links[link_name]

        # 处理当前link的visual部分
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is not None:
                mesh_elem = geometry.find("mesh")
                if mesh_elem is not None and mesh_elem.get("filename").endswith(".obj"):
                    mesh_path = join(obj_dir, mesh_elem.get("filename"))
                    mesh = o3d.io.read_triangle_mesh(
                        mesh_path, enable_post_processing=True
                    )

                    # 获取visual的局部变换
                    origin = visual.find("origin")
                    local_transform = np.eye(4)
                    if origin is not None:
                        xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
                        rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]

                        translation = np.eye(4)
                        translation[:3, 3] = xyz

                        rotation = o3d.geometry.get_rotation_matrix_from_xyz(rpy)
                        rot_matrix = np.eye(4)
                        rot_matrix[:3, :3] = rotation

                        local_transform = np.dot(translation, rot_matrix)

                    # 应用父变换和局部变换
                    mesh.transform(np.dot(parent_transform, local_transform))

                    # 设置颜色
                    material = visual.find("material")
                    if material is not None:
                        color = material.find("color")
                        if color is not None:
                            rgba = [float(x) for x in color.get("rgba").split()]
                            mesh.paint_uniform_color(rgba[:3])

                    self.geometries.append(mesh)
                    self.vis.add_geometry(mesh)

        # 处理子joint
        for joint in self.joints.values():
            if joint.find("parent").get("link") == link_name:
                child_link = joint.find("child").get("link")

                # 获取joint的变换
                origin = joint.find("origin")
                joint_transform = np.eye(4)
                if origin is not None:
                    xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
                    rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]

                    translation = np.eye(4)
                    translation[:3, 3] = xyz

                    rotation = o3d.geometry.get_rotation_matrix_from_xyz(rpy)
                    rot_matrix = np.eye(4)
                    rot_matrix[:3, :3] = rotation

                    joint_transform = np.dot(translation, rot_matrix)

                # 递归处理子link
                self.process_link(
                    child_link, obj_dir, np.dot(parent_transform, joint_transform)
                )

    def render(self):
        self.vis.run()
        self.vis.destroy_window()


# 使用示例
urdf_path = "/home/hhhar/liuliu/gpt_test/dataset/partnet/100392/mobility.urdf"
obj_files = "/home/hhhar/liuliu/gpt_test/dataset/partnet/100392"
renderer = URDFRenderer()
renderer.parse_urdf(urdf_path, obj_files)
renderer.render()
