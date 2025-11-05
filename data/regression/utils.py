import numpy as np

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