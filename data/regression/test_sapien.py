import os
import h5py
import sapien.core as sapien
# import sapien
import mplib
import numpy as np
from sapien.utils.viewer import Viewer
from transforms3d.euler import mat2euler
from sapien.core import PinocchioModel
from PIL import Image
import json
from IPython import embed
import open3d as o3d

import sys
sys.path.append('..')
from utils import pose_relative, recover_pose, read_h5_file

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

# ---------------- 方法 A：给 (p, q, scale) 直接变换点 ----------------


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
    def __init__(self, robot_prefix, target_id):
        self.robot_prefix = robot_prefix
        self.target_id = target_id
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)

        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(1 / 240.0)
        self.scene.add_ground(-0.8)
        physical_material = self.scene.create_physical_material(1, 1, 0.0)
        self.scene.default_physical_material = physical_material

        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light(
            [0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        self.viewer = Viewer(self.renderer)  # , resolutions=(1920, 1080))
        self.viewer.set_scene(self.scene)
        # self.viewer.set_camera_xyz(x=1.2, y=0.25, z=0.4)
        # self.viewer.set_camera_rpy(r=0, p=-0.4, y=2.7)

        # Robot
        # Load URDF
        robot_loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        robot_loader.fix_root_link = True
        robot_loader.scale = 1
        self.robot: sapien.Articulation = robot_loader.load(
            self.robot_prefix+'.urdf')
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        self.hand_obj = self.robot.find_link_by_name("panda_hand")
        self.camera = []
        self.bbox = []
        self.scaled_bbox = []
        self.poses = []
        self.actions = []
        self.rgbs = []
        self.depthes = []
        self.cam_mat = []
        self.action_type = []
        

        # Set initial joint positions
        init_qpos = [
            0,
            0.19634954084936207,
            0.0,
            -2.617993877991494,
            0.0,
            2.941592653589793,
            0.7853981633974483,
            0,
            0,
        ]
        self.robot.set_qpos(init_qpos)

        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=1000, damping=200)

        # table top
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.4, 0.4, 0.025])
        builder.add_box_visual(half_size=[0.4, 0.4, 0.025])
        self.table = builder.build_kinematic(name="table")
        self.table.set_pose(sapien.Pose([0.56, 0, -0.025]))

        # builder.add_box_visual(
        #     half_size=[100, 0.4, 50])  # 可调半径与颜色
        # # 你也可以选择不添加 collision
        # box_actor = builder.build_kinematic(name="sky_1")
        # # 将球体放置在摄像机中心：
        # box_actor.set_pose(sapien.Pose(p=[0, 10, 0]))

        # builder.add_box_visual(
        #     half_size=[0.4, 100, 50])  # 可调半径与颜色
        # # 你也可以选择不添加 collision
        # box_actor = builder.build_kinematic(name="sky_2")
        # # 将球体放置在摄像机中心：
        # box_actor.set_pose(sapien.Pose(p=[10, 0, 0]))

        # # boxes
        # builder = self.scene.create_actor_builder()
        # builder.add_box_collision(half_size=[0.02, 0.02, 0.06])
        # builder.add_box_visual(half_size=[0.02, 0.02, 0.06], color=[1, 0, 0])
        # self.red_cube = builder.build(name='red_cube')
        # self.red_cube.set_pose(sapien.Pose([0.4, 0.3, 0.06]))

        # builder = self.scene.create_actor_builder()
        # builder.add_box_collision(half_size=[0.02, 0.02, 0.04])
        # builder.add_box_visual(half_size=[0.02, 0.02, 0.04], color=[0, 1, 0])
        # self.green_cube = builder.build(name='green_cube')
        # self.green_cube.set_pose(sapien.Pose([0.2, -0.3, 0.04]))

        # builder = self.scene.create_actor_builder()
        # builder.add_box_collision(half_size=[0.02, 0.02, 0.07])
        # builder.add_box_visual(half_size=[0.02, 0.02, 0.07], color=[0, 0, 1])
        # self.blue_cube = builder.build(name='blue_cube')
        # self.blue_cube.set_pose(sapien.Pose([0.6, 0.1, 0.07]))

        # drawer
        self.get_bbox()
        target_path = f"/mnt/4dba1798-fc0d-4700-a472-04acb2f7b630/hhhar/partnet/{self.target_id}/mobility_annotation_gapartnet.urdf"
        self.load_urdf_in_sapien(target_path, 0.2, pos=np.array(
            [0.3, 0.0, 0.03]), q=np.array([0.7, 0.0, 0.7, 0.0]))

        # # print(self.part.get_links())
        # for i, joint in enumerate(self.part.get_joints()):
        #     print(joint.get_name())

        # pin = self.part.create_pinocchio_model()
        # pose = pin.get_link_pose(7)
        # print(pose)

        self.setup_planner(robot_prefix)

    def get_bbox(self):
        anno_path = f"/mnt/4dba1798-fc0d-4700-a472-04acb2f7b630/hhhar/partnet/{self.target_id}/link_annotation_gapartnet.json"
        with open(anno_path, "r") as f:
            anno = json.load(f)
        target_anno = anno[1]
        self.bbox = target_anno['bbox']
        if target_anno['category'] == "slider_button":
            self.action_type.append('press')
        else:
            self.action_type.append("rot")

    def load_urdf_in_sapien(
        self,
        urdf_path: str,
        scale: float,
        pos: np.ndarray = np.array([0.0, 0.0, 0.0]),
        q: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0]),
    ):

        near, far = 0.1, 100
        width, height = 640, 480
        # 设置摄像机位置
        camera_pos = pos + np.array([0.1, 0.35, 0.25])

        # 计算朝向向量
        forward = pos - camera_pos
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
        # rpy = mat2euler(rotation_matrix) * \
        #     np.array([1, -1, -1])  # SAPIEN 的坐标系调整
        rpy = mat2euler(rotation_matrix)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = camera_pos
        self.cam_mat.append(mat44)
        
        # self.camera = self.scene.add_camera(
        #     name="camera",
        #     width=width,
        #     height=height,
        #     fovy=np.deg2rad(35),
        #     near=near,
        #     far=far,
        # )
        
        # cam_pose = sapien.Pose(p=camera_pos)
        # cam_pose.set_rpy(rpy)
        # self.camera.set_entity_pose(cam_pose)
        self.viewer.set_camera_pose(sapien.Pose(mat44))
    
        
        # 创建 URDF Loader
        loader = self.scene.create_urdf_loader()
        loader.scale = scale
        self.part = loader.load(urdf_path)

        # print(self.part.get_joints())

        # for i, joint in enumerate(self.part.get_joints()):
        #     if i == 2:
        #         joint.set_limits([[0.0, 0.01]])

        # 设置初始姿态
        self.part.set_pose(sapien.Pose(p=pos, q=q))

        self.scaled_bbox = transform_points_by_pose(self.bbox, pos, q, scale)
        cube_size = 0.01
        half_size = cube_size / 2
        for i, point in enumerate(self.scaled_bbox):
            point = np.array(point)

            builder = self.scene.create_actor_builder()
            builder.add_box_visual(
                pose=sapien.Pose(p=point),
                half_size=[half_size, half_size, half_size],
            )

            actor = builder.build_static(name=f"vertex_{i}")

    def setup_planner(self, urdf_prefix: str):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name()
                       for joint in self.robot.get_active_joints()]
        self.planner = mplib.Planner(
            urdf=urdf_prefix + ".urdf",
            srdf=urdf_prefix + ".srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7),
        )


    def data2h5(self, h5_path):
        with h5py.File(h5_path,"w") as f:
            f.create_dataset("scaled_bbox", data=np.array(self.scaled_bbox))
            f.create_dataset("poses", data=self.poses)
            f.create_dataset("actions", data=self.actions)
            f.create_dataset("rgbs", data=np.array(self.rgbs))
            f.create_dataset("depthes", data=np.array(self.depthes))
            f.create_dataset("cam_mat", data=np.array(self.cam_mat))
            f.create_dataset("action_type", data=np.array(self.action_type, dtype=object))
        print('save ok')
    
    def pose2np(self, pose):
        q = np.array(pose.q)
        p = np.array(pose.p)
        return np.concatenate((p, q))
    
    def append_data(self, former_pos):
        cur_pos = self.hand_obj.get_entity_pose()
        self.poses.append(self.pose2np(cur_pos))
        
        # 获取 RGBA 浮点图（H, W, 4），每通道在 [0,1]
        rgba = self.viewer.window.get_picture('Color')
        # 将 RGBA 转换成 uint8 RGB
        rgba8 = (rgba * 255).clip(0, 255).astype(np.uint8)
        rgb_img = rgba8[..., :3]
        img = Image.fromarray(rgb_img)

        # 获取 position / 深度 / segmentation
        pos = self.viewer.window.get_picture('Position')  # shape (H, W, 4)，前三通道为 3D 点，第四通道为 z-buffer 值
        depth = -pos[..., 2]  # 根据文档，z 是负方向的深度值 :contentReference[oaicite:6]{index=6}

        self.rgbs.append(rgb_img)
        self.depthes.append(depth)
        self.actions.append(pose_relative(self.pose2np(former_pos), self.pose2np(cur_pos)))
        

    def follow_path(self, result, save_data):
        n_step = result["position"].shape[0]
        for i in range(n_step):
            former_pos = self.hand_obj.get_entity_pose()
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            for j in range(7):
                self.active_joints[j].set_drive_target(
                    result["position"][i][j])
                self.active_joints[j].set_drive_velocity_target(
                    result["velocity"][i][j]
                )
            self.scene.step()
            if i % 4 == 3:
                self.scene.update_render()
                self.viewer.render()
                self.scene.update_render()
                if save_data:
                    self.append_data(former_pos)

    def open_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.4)
        for i in range(100):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def close_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0)
        for i in range(100):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()
        print("close over")
            

    def move_to_pose_with_RRTConnect(self, pose, save_data):
        result = self.planner.plan(
            pose, self.robot.get_qpos(), time_step=1 / 250)
        if result["status"] != "Success":
            print(result["status"])
            return -1
        self.follow_path(result, save_data)
        return 0

    def move_to_pose_with_screw(self, pose, save_data):
        result = self.planner.plan_screw(
            pose, self.robot.get_qpos(), qpos_step=1 / 400, time_step=1 / 400)
        if result["status"] != "Success":
            result = self.planner.plan(
                pose, self.robot.get_qpos(), qpos_step=1 / 400, time_step=1 / 400)
            if result["status"] != "Success":
                print(result["status"])
                return -1
        self.follow_path(result, save_data)
        return 0

    def move_to_pose(self, pose, with_screw, save_data=False):
        res = -1
        root = self.robot.get_root_pose()
        pose[:3] -= root.p
        self.hand_obj.get_entity_pose()
        if with_screw:
            res = self.move_to_pose_with_screw(pose, save_data)
        else:
            res = self.move_to_pose_with_RRTConnect(pose, save_data)
        
        if res == -1:
            print("move fail")
        else:
            print(f'target_pose:{pose}')
            input(f'hand_pose:{self.hand_obj.get_entity_pose()}')
            print("move success")

    # def get_pic(self, dir_name: str, name: str):
    #     if os.path.exists(f"{dir_name}"):
    #         os.mkdir(f"{dir_name}")
    #     obs = self.viewer.window.get_float_texture("Color")
    #     obs_img = (obs * 255).clip(0, 255).astype("uint8")
    #     Image.fromarray(obs_img).save(f"{dir_name}_{name}.png")

    def demo(self, with_screw=True):
        # pose = [0.2, -0.27, 0.85, 0.5, 0.5, 0.5, 0.5] #水平往前
        # pose = [0.35, 0, 0.6, 0, 1, 0, 0] # 竖直
        # pose = [0.3, 0, 0.57, 0.5, 0.5, 0.5, 0.5]
        # pose = [0.3, 0.0, 0.57, 0, 0.7, 0, 0.7] #水平往前

        # pose = [0.5, 0.0, 0.5, 0, 1, 0, 0]
        # res = self.move_to_pose(pose, with_screw)

        
        center = self.scaled_bbox.mean(0).tolist()
        center[2] += 0.2
        # hand = self.robot.find_link_by_name("panda_hand").get_pose()
        root = self.robot.get_root_pose()
        target = center - root.p
        # print(center)
        # print(root)
        # input(target)
        pose = [target[0], target[1], target[2], 0, 1, 0, 0]  # 竖直
        
        
        # pose = [0.5, 0.0, 0.9, 0, 1, 0, 0]
        self.move_to_pose(pose, with_screw)

        # self.open_gripper()

        # self.get_pic("washer_1", "obs")
        # pose[0] += 0.03
        # # self.move_to_pose(pose, with_screw)
        # self.planner.IK(pose, self.robot.get_qpos())

        # pose[2] -= 0.1
        # self.move_to_pose(pose, with_screw)
        # for i, joint in enumerate(self.part.get_joints()):
        #     if i == 2:
        #         joint.set_limits([[0.0, 10.0]])
        # self.move_to_pose(pose, with_screw)
        self.close_gripper()
        print(self.hand_obj.get_entity_pose())
        
        pose[2] -= 0.08
        print(pose)
        self.move_to_pose(pose, with_screw)
        print(self.hand_obj.get_entity_pose())

        # pose[2] -= 0.03
        # pose[0] -= 0.1
        # self.move_to_pose(pose, with_screw)
        # self.contacts = self.scene.get_contacts()
        # print(self.contacts)
        # pose[0] -= 0.21
        # self.move_to_pose(pose, with_screw)
        # self.get_pic("washer_1", "over")

        # poses = [[0.4, 0.3, 0.12, 0, 1, 0, 0],
        #         [0.2, -0.3, 0.08, 0, 1, 0, 0],
        #         [0.6, 0.1, 0.14, 0, 1, 0, 0]]
        # for i in range(3):
        #     pose = poses[i]
        #     pose[2] += 0.2
        #     self.move_to_pose(pose, with_screw)
        #     self.open_gripper()
        #     pose[2] -= 0.12
        #     self.move_to_pose(pose, with_screw)
        #     self.close_gripper()
        #     pose[2] += 0.12
        #     self.move_to_pose(pose, with_screw)
        #     pose[0] += 0.1
        #     self.move_to_pose(pose, with_screw)
        #     pose[2] -= 0.12
        #     self.move_to_pose(pose, with_screw)
        #     self.open_gripper()
        #     pose[2] += 0.12
        #     self.move_to_pose(pose, with_screw)
        print(self.action_type)
        self.data2h5("test.h5")
        while not self.viewer.closed:
            for _ in range(4):  # render every 4 steps
                qf = self.robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                self.robot.set_qf(qf)
                self.scene.step()
            self.scene.update_render()
            self.viewer.render()

    def test_h5(self, h5_path):
        data = read_h5_file(h5_path)
        self.test_action(data)
    
    def test_action(self, data):
        scaled_bbox = data.get("scaled_bbox")   # 可能是 numpy array 或 list
        poses = data.get("poses")               # 可能是 (N,7) ndarray 或 list of arrays
        actions = data.get("actions")
        rgbs = data.get("rgbs")                 # 可能是 (N,H,W,3) ndarray
        depthes = data.get("depthes")
        cam_mat = data.get("cam_mat")
        action_type = data.get("action_type")
        
        # in_pose = poses[0]
        init_pose = [0.45, 0.0, 0.2, 0, 1, 0, 0]
        cur_pose = self.pose2np(self.hand_obj.get_entity_pose())
        action = pose_relative(cur_pose, init_pose)
        tar_pose = recover_pose(cur_pose, action)
        self.move_to_pose(tar_pose, True)
        
        for action in actions:
            cur_pose = self.hand_obj.get_entity_pose()
            cur_pose = self.pose2np(cur_pose)
            target_pose = recover_pose(cur_pose, action)
            self.move_to_pose(target_pose, True)
            # print(center)
            # print(root)
            # input(target)
        
        while not self.viewer.closed:
            for _ in range(4):  # render every 4 steps
                qf = self.robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                self.robot.set_qf(qf)
                self.scene.step()
            self.scene.update_render()
            self.viewer.render()


if __name__ == "__main__":
    robot_prefix = "/home/huanganran/field/vld/assets/robot/panda/panda"
    target_id = 100392
    demo = PlanningDemo(robot_prefix, target_id)
    # demo.demo()
    demo.test_h5("test.h5")
