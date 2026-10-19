"""Small drawing helpers shared by the scene scripts."""
import numpy as np


def segment_pose(p0, p1):
    """4x4 pose of a cylinder whose z-axis runs from p0 to p1 (centred between them)."""
    v = p1 - p0
    z = v / np.linalg.norm(v)
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack((x, y, z))
    T[:3, 3] = (p0 + p1) / 2
    return T
