import pickle
import h5py
import open3d as o3d
import numpy as np
   
def pos2pcd(self,pos, rgba):
    mask = pos[..., 3] < 1.0   # bool mask (H, W)
    # 展平
    pts_cam = pos[..., :3].reshape(-1, 3)
    valid = mask.reshape(-1)
    pts_cam = pts_cam[valid]

    # 对应的颜色
    colors = rgba[..., :3].reshape(-1, 3)[valid]

    # 4. 构建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_cam)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    
def normalize_quat(q):
    """Normalize quaternion given as [w,x,y,z]."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Quaternion norm too small to normalize.")
    return q / n

def quat_conjugate(q):
    """Quaternion conjugate for [w,x,y,z]."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)

def quat_inverse(q):
    """Quaternion inverse (assuming non-zero norm). For unit quaternions it's conjugate."""
    q = np.asarray(q, dtype=float)
    n2 = np.dot(q, q)
    if n2 < 1e-12:
        raise ValueError("Quaternion norm too small to invert.")
    return quat_conjugate(q) / n2

def quat_mul(q1, q2):
    """Quaternion multiplication (Hamilton product).
    Input/output quaternions are [w,x,y,z].
    Returns q = q1 * q2 (apply q2, then q1 if used for rotations in some conventions).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=float)

def quat_rotate_vector(q, v):
    """Rotate vector v (3,) by unit quaternion q (w,x,y,z)."""
    q = normalize_quat(q)
    v = np.asarray(v, dtype=float)
    # v' = q * (0, v) * q^{-1}
    qv = np.concatenate(([0.0], v))
    qr = quat_mul(quat_mul(q, qv), quat_inverse(q))
    return qr[1:]

def pose_inverse(pose):
    """Inverse of pose given as [t(3), q(4:w,x,y,z)].
    Returns [t_inv(3), q_inv(4)] such that T_inv * T = I.
    """
    t = np.asarray(pose[0:3], dtype=float)
    q = normalize_quat(pose[3:7])
    q_inv = quat_inverse(q)  # for unit quaternion this is conjugate
    t_inv = -quat_rotate_vector(q_inv, t)
    return np.concatenate((t_inv, q_inv))

def pose_mul(pose_a, pose_b):
    """Compose two poses: T = A * B.
    pose format: [t(3), q(4:w,x,y,z)]
    returns pose T as same format.
    """
    ta = np.asarray(pose_a[0:3], dtype=float)
    qa = normalize_quat(pose_a[3:7])
    tb = np.asarray(pose_b[0:3], dtype=float)
    qb = normalize_quat(pose_b[3:7])
    # rotation composition: q = qa * qb
    q = quat_mul(qa, qb)
    # translation: t = ta + R(qa) * tb
    t = ta + quat_rotate_vector(qa, tb)
    return np.concatenate((t, q))

def pose_relative(pose1, pose2):
    """Compute relative pose from pose1 to pose2: T_rel = inv(T1) * T2.
    Both poses format: [tx,ty,tz, qw,qx,qy,qz]
    Returns relative pose in same format.
    """
    inv1 = pose_inverse(pose1)
    rel = pose_mul(inv1, pose2)
    # Normalize quaternion part before returning
    rel_t = rel[0:3]
    rel_q = normalize_quat(rel[3:7])
    return np.concatenate((rel_t, rel_q))

def recover_pose(pose1, rel):
    """ 给定 pose1 和相对变换 rel (inv(pose1)*pose2)，还原 pose2 """
    return pose_mul(pose1, rel)


def rotation_matrix_to_quat_wxyz(R):
    """
    将 3x3 旋转矩阵 R 转为四元数 (w, x, y, z)。
    使用数值稳定的分支方法，并归一化结果。
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        S = np.sqrt(trace + 1.0) * 2.0  # S = 4*w
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        # 找到最大的对角元对应的分支
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # S = 4*x
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # S = 4*y
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # S = 4*z
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S

    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    return q

def pose2mat44(pose):
    mat44 = np.eye(4)
    mat33 = quat_wxyz_to_rotmat(pose[3:])
    mat44[0:3, 0:3] = mat33
    mat44[0:3, 3] = pose[0:3]
    return mat44

def try_deserialize_bytes(x):
    """尝试把 bytes 反序列化成原始对象（优先 utf-8 字符串，然后 pickle）。"""
    if not isinstance(x, (bytes, bytearray)):
        return x
    # 尝试 utf-8 decode
    try:
        return x.decode("utf-8")
    except Exception:
        pass
    # 尝试 pickle
    try:
        return pickle.loads(x)
    except Exception:
        pass
    # 返回原始 bytes
    return x

def read_dataset_safely(ds):
    """
    安全读取 dataset ds（h5py.Dataset）。
    - 如果 dtype 不是 object，直接返回 ds[:]
    - 如果 dtype 是 object：对每个元素进行处理（bytes->decode/pickle，numpy->保持，list->np.array）
    最后尝试把元素堆叠为一个普通 numpy array（如果 shape 可对齐），否则返回 Python list。
    """
    data = ds[:]
    # 普通数值/字符串类型，直接返回
    if not (isinstance(data, np.ndarray) and data.dtype == object):
        # 对 bytes 字符串数组（S dtype）做解码（兼容不同 h5py/numpy 版本）
        if isinstance(data, np.ndarray) and data.dtype.kind == 'S':
            # bytes -> str
            return np.array([x.decode('utf-8') for x in data])
        return data

    # data 是 object 数组：逐元素处理
    processed = []
    for i, v in enumerate(data):
        if isinstance(v, (bytes, bytearray)):
            processed.append(try_deserialize_bytes(v))
        elif isinstance(v, np.ndarray):
            processed.append(v)
        elif isinstance(v, (list, tuple)):
            processed.append(np.array(v))
        else:
            # 其他 Python 对象（int/float/str等）
            processed.append(v)

    # 尝试把 processed 转为统一的 numpy array（如果每个元素 shape 相同）
    try:
        stacked = np.stack([np.asarray(x) for x in processed])
        return stacked
    except Exception:
        # 无法堆叠，就返回原始 python list
        return processed

def read_h5_file(path):
    result = {}
    with h5py.File(path, "r") as f:
        print("Datasets in file:", list(f.keys()))
        for name, ds in f.items():
            try:
                val = read_dataset_safely(ds)
            except Exception as e:
                print(f"Failed to read dataset {name}: {e}")
                val = None
            # 打印一些调试信息
            if isinstance(val, np.ndarray):
                print(f"  {name}: numpy array, dtype={val.dtype}, shape={val.shape}")
            elif isinstance(val, list):
                print(f"  {name}: python list, len={len(val)}; element types: {[type(x) for x in val[:3]]}")
            else:
                print(f"  {name}: type={type(val)}; value sample={val if hasattr(val,'shape') or isinstance(val,(int,float,str)) else str(val)[:100]}")
            result[name] = val
    return result

def quat_wxyz_to_rotmat(q):
    """
    将四元数 (w,x,y,z) 转为 3x3 旋转矩阵
    自动归一化四元数，并保证旋转矩阵正交性
    """
    q = normalize_quat(q)
    w, x, y, z = q

    # 构建旋转矩阵公式
    R = np.array([
        [1-2*(y**2 + z**2),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1-2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1-2*(x**2 + y**2)]
    ])

    # 用 SVD 做正交修正（确保 det(R)=1）
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        # 避免反射矩阵
        U[:, -1] *= -1
        R = U @ Vt
    return R

def snap_quat(q, tol=1e-2):
    q = np.asarray(q, dtype=float).copy()
    # 小于 tol 的置零
    q[np.abs(q) < tol] = 0.0
    # 接近 ±1 的分量钳到 ±1
    q[np.isclose(np.abs(q), 1.0, atol=tol)] = np.sign(q[np.isclose(np.abs(q), 1.0, atol=tol)]) * 1.0
    return normalize_quat(q)


# --- 简短示例 / 测试 ---
if __name__ == "__main__":
    # pose1: at origin, identity rotation
    pose1 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # t=(0,0,0), q=(1,0,0,0)

    # pose2: translated by (1,0,0), rotated 90 deg around z (w,x,y,z)
    angle = np.pi/2
    qw = np.cos(angle/2)
    qz = np.sin(angle/2)
    pose2 = np.array([1.0, 0.0, 0.0, qw, 0.0, 0.0, qz])

    rel = pose_relative(pose1, pose2)
    print("Relative pose (pose1 -> pose2):")
    print("t_rel =", rel[0:3])
    print("q_rel (w,x,y,z) =", rel[3:7])

    # 用 pose1 和 rel 还原 pose2
    recovered = recover_pose(pose1, rel)

    print("原始 pose2:", pose2)
    print("还原 pose2:", recovered)

    # 检查误差（平移和四元数）
    print("平移误差:", np.linalg.norm(recovered[0:3] - pose2[0:3]))
    print("四元数误差 (按向量差):", np.linalg.norm(recovered[3:7] - pose2[3:7]))