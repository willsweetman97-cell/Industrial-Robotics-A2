"""PartMover: original 6-DOF arm with a spherical wrist and a two-finger gripper.
Standard DH, reach about 0.88 m, built to hand small tools between robots.

    link   theta   d    a    alpha
     1      q1     D1   0    +90
     2      q2     0    A2   0
     3      q3     0    A3   +90
     4      q4     D4   0    -90
     5      q5     0    0    +90
     6      q6     D6   0    0
"""

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
from spatialmath import SE3

# Compatibility: older Swift versions call shape._update_pyb(), which newer spatialgeometry removed.
if not hasattr(sg.Shape, "_update_pyb"):
    sg.Shape._update_pyb = lambda self: None

D1, A2, A3, D4, D6 = 0.27, 0.36, 0.04, 0.34, 0.10     # metres
GRIP_CENTRE = 0.075                                    # flange to the point between the fingertips
OPEN_WIDTH = 0.08                                      # gap between fingers when open
QLIM = np.radians([[-170, 170], [-110, 110], [-140, 140], [-180, 180], [-115, 115], [-360, 360]])
BODY, HUB = (0.18, 0.20, 0.24, 1), (0.0, 0.62, 0.65, 1)


def _make(cls, pose, colour, **kw):
    """Newer spatialgeometry uses `pose`, older versions use `base`."""
    try:
        return cls(pose=pose, color=colour, **kw)
    except TypeError:
        return cls(base=pose, color=colour, **kw)


def _cylinder(T_link, p0, p1, radius, colour):
    """Cylinder from point p0 to p1 (robot frame at q=0), attached to a link frame."""
    a, b = (T_link.inv() * SE3(*p0)).t, (T_link.inv() * SE3(*p1)).t
    length = np.linalg.norm(b - a)
    if length < 1e-6:                         # joints at the same point: nothing to draw
        return None
    z = (b - a) / length
    axis = np.cross([0, 0, 1], z)
    if np.linalg.norm(axis) < 1e-9:           # already along +z or -z
        rot = SE3() if z[2] > 0 else SE3.Rx(np.pi)
    else:
        rot = SE3.AngleAxis(np.arctan2(np.linalg.norm(axis), z[2]), axis / np.linalg.norm(axis))
    return _make(sg.Cylinder, SE3(*((a + b) / 2)) * rot, colour, radius=radius, length=length)


class PartMover(rtb.DHRobot):
    Q_READY = np.array([0.0, -0.6, 1.6, 0.0, 1.0, 0.0])

    def __init__(self, base=None):
        links = [rtb.RevoluteDH(d=d, a=a, alpha=al, qlim=QLIM[i]) for i, (d, a, al) in enumerate(
            [(D1, 0, np.pi / 2), (0, A2, 0), (0, A3, np.pi / 2), (D4, 0, -np.pi / 2), (0, 0, np.pi / 2), (D6, 0, 0)])]
        super().__init__(links, name="PartMover")
        self._add_visuals()
        self.base = base if base is not None else SE3()
        self.q = self.Q_READY

    def _add_visuals(self):
        """A tube along each link, a coloured hub at each joint, a base plinth and a palm plate."""
        frames = list(self.fkine_all(np.zeros(6)))
        if len(frames) == 6:                      # older toolbox omits the base frame
            frames = [SE3()] + frames
        for i, link in enumerate(self.links, start=1):
            p, z = frames[i - 1].t, frames[i - 1].R[:, 2]
            for shape in (_cylinder(frames[i], p, frames[i].t, 0.045, BODY),
                          _cylinder(frames[i], p - 0.04 * z, p + 0.04 * z, 0.06, HUB)):
                if shape is not None:
                    link.geometry.append(shape)
        self.links[0].geometry.append(_cylinder(frames[1], (0, 0, 0), (0, 0, 0.03), 0.12, BODY))
        palm = (frames[6].inv() * SE3(*(frames[6].t + frames[6].R[:, 2] * 0.015))).A
        self.links[5].geometry.append(_make(sg.Cuboid, SE3(palm), (0.75, 0.77, 0.8, 1), scale=[0.11, 0.08, 0.016]))

    def grip_pose(self, q=None):
        """Point between the fingertips, in the world."""
        return self.fkine(self.q if q is None else q) * SE3(0, 0, GRIP_CENTRE)


class Gripper:
    """Two fingers that follow the tool flange and can open or close."""

    def __init__(self):
        self.left = _make(sg.Cuboid, SE3(), (0.1, 0.1, 0.1, 1), scale=[0.012, 0.05, 0.075])
        self.right = _make(sg.Cuboid, SE3(), (0.1, 0.1, 0.1, 1), scale=[0.012, 0.05, 0.075])
        self.width = OPEN_WIDTH

    def shapes(self):
        return [self.left, self.right]

    def update(self, T_flange, width=None):
        self.width = self.width if width is None else float(np.clip(width, 0, OPEN_WIDTH))
        off = self.width / 2 + 0.006
        self.left.T = (T_flange * SE3(-off, 0, 0.054)).A
        self.right.T = (T_flange * SE3(off, 0, 0.054)).A
