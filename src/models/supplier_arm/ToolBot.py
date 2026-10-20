"""
Supplier arm model (ToolBot, role: selector / part mover).
Original 6-DoF design, dimensions justified by the task geometry in Garage.py:
 - base at x=-1.40, y=0.22, z=0.75 (table height), near the pegboard
 - must reach pegboard tools up to ~1.11 m away, at z ~1.38-1.40 m
 - must reach handoff zone at approx (-0.925, -0.08, 0.75), ~0.56 m away
 - shoulder (joint 2) limited to +/-150 deg so the arm can reach the saw at the far end of the tool wall
Task: pick the requested tool from the pegboard and deliver it to the handoff zone.
Author: Will Sweetman
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import spatialgeometry as sg
from Garage import build_garage

# Compatibility shim: swift-sim 1.1.0 calls shape._update_pyb(), which newer
# spatialgeometry versions renamed. The method only syncs the optional pybullet
# collision backend, so a no-op is safe for visualisation.
if not hasattr(sg.Shape, "_update_pyb"):
    sg.Shape._update_pyb = lambda self: None

LINK_COLOUR = (0.15, 0.45, 0.9, 1.0)    # matches the selector base marker
JOINT_COLOUR = (0.1, 0.1, 0.1, 1.0)
TOOL_COLOUR = (0.95, 0.5, 0.1, 1.0)
HANDOFF_ZONE = SE3(-0.925, -0.08, 0.75)


class SupplierArm(DHRobot):
    def __init__(self):
        links = [
            RevoluteDH(d=0.15, a=0.00, alpha=-np.pi / 2, qlim=[-np.deg2rad(170), np.deg2rad(170)]),
            RevoluteDH(d=0.00, a=0.50, alpha=0,          qlim=[-np.deg2rad(150), np.deg2rad(150)]),
            RevoluteDH(d=0.00, a=0.45, alpha=0,          qlim=[-np.deg2rad(150), np.deg2rad(150)]),
            RevoluteDH(d=0.15, a=0.00, alpha=-np.pi / 2, qlim=[-np.pi, np.pi]),
            RevoluteDH(d=0.00, a=0.00, alpha=np.pi / 2,  qlim=[-np.deg2rad(120), np.deg2rad(120)]),
            RevoluteDH(d=0.10, a=0.00, alpha=0,          qlim=[-2 * np.pi, 2 * np.pi]),
        ]
        super().__init__(links, name="SupplierArm")
        self.qz = np.zeros(6)
        self.q_home = np.deg2rad([0, -30, 60, 0, 60, 0])


def _link_pose(p0, p1):
    """Pose at the midpoint of p0->p1 with local z along the link, plus the link length."""
    v = p1 - p0
    length = np.linalg.norm(v)
    if length < 1e-6:
        return None, 0.0
    z = v / length
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, (p0 + p1) / 2
    return SE3(T, check=False), length


class SupplierArmVisual:
    """Cylinders for links and spheres for joints, redrawn from forward kinematics each step."""

    def __init__(self, robot, env):
        self.robot = robot
        pts = [f.t for f in robot.fkine_all(robot.q)]
        self.links = []
        for i in range(len(pts) - 1):
            pose, length = _link_pose(pts[i], pts[i + 1])
            if pose is None:
                self.links.append(None)
                continue
            c = sg.Cylinder(radius=0.04, length=length, color=LINK_COLOUR)
            c.T = pose.A
            env.add(c)
            self.links.append(c)
        self.joints = []
        for p in pts:
            s = sg.Sphere(radius=0.055, color=JOINT_COLOUR)
            s.T = SE3(*p).A
            env.add(s)
            self.joints.append(s)

    def update(self, q):
        pts = [f.t for f in self.robot.fkine_all(q)]
        for i, c in enumerate(self.links):
            if c is not None:
                pose, _ = _link_pose(pts[i], pts[i + 1])
                c.T = pose.A
        for s, p in zip(self.joints, pts):
            s.T = SE3(*p).A


def solve_ik(robot, target, q0):
    """Position-only IK (orientation free). Returns joint angles or None if unreachable."""
    sol = robot.ikine_LM(target, q0=q0, mask=[1, 1, 1, 0, 0, 0])
    return sol.q if sol.success else None


def move(robot, vis, env, q_target, steps=60, carried=None):
    """Smooth joint-space (quintic) trajectory from the current pose to q_target."""
    for q in rtb.jtraj(robot.q, q_target, steps).q:
        robot.q = q
        vis.update(q)
        if carried is not None:
            carried.T = robot.fkine(q).A
        env.step(0.03)


if __name__ == "__main__":
    robot = SupplierArm()
    print(robot)

    env, poses = build_garage(launch=True, realtime=True)
    robot.base = poses["selector"]
    robot.q = robot.q_home
    vis = SupplierArmVisual(robot, env)
    env.step()

    print("\nReachability check, position-only:")
    for name, pose in poses["tools"].items():
        ok = solve_ik(robot, SE3(pose.t), robot.q_home) is not None
        print(f"  {name:12s}: {'OK' if ok else 'out of reach'}")

    tool_name = "drill"
    tool_pos = poses["tools"][tool_name].t
    waypoints = [
        ("approach tool", SE3(tool_pos[0], tool_pos[1] - 0.12, tool_pos[2])),
        ("grasp tool",    SE3(*tool_pos)),
        ("retract",       SE3(tool_pos[0], tool_pos[1] - 0.15, tool_pos[2])),
        ("above handoff", SE3(HANDOFF_ZONE.t[0], HANDOFF_ZONE.t[1], HANDOFF_ZONE.t[2] + 0.20)),
        ("place tool",    SE3(HANDOFF_ZONE.t[0], HANDOFF_ZONE.t[1], HANDOFF_ZONE.t[2] + 0.04)),
        ("clear handoff", SE3(HANDOFF_ZONE.t[0], HANDOFF_ZONE.t[1], HANDOFF_ZONE.t[2] + 0.20)),
    ]

    carried = None
    q_prev = robot.q
    print(f"\nDelivering {tool_name} to the handoff zone:")
    for label, target in waypoints:
        q_goal = solve_ik(robot, target, q_prev)
        if q_goal is None:
            print(f"  {label}: unreachable, stopping")
            break
        print(f"  {label}")
        move(robot, vis, env, q_goal, carried=carried)
        q_prev = q_goal

        if label == "grasp tool":
            carried = sg.Cuboid(scale=[0.06, 0.06, 0.12], color=TOOL_COLOUR)
            carried.T = robot.fkine(robot.q).A
            env.add(carried)
        elif label == "place tool":
            carried = None

    print("  returning home")
    move(robot, vis, env, robot.q_home)

    print("\nDone. Close with Ctrl+C in the terminal.")
    env.hold()
