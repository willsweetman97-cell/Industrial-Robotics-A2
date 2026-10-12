"""
Supplier arm model (role: selector / part mover).
Original 6-DoF design, dimensions justified by the task geometry in Garage.py:
 - base at x=-1.40, y=0.22, z=0.75 (table height), near the pegboard
 - must reach pegboard tools up to ~1.11 m away, at z ~1.38-1.40 m
 - must reach handoff zone at approx (-0.925, -0.08, 0.75), ~0.56 m away
Author: Will Sweetman
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import numpy as np
from Garage import build_garage


class SupplierArm(DHRobot):
    def __init__(self):
        links = [
            RevoluteDH(d=0.15, a=0.00, alpha=-np.pi/2, qlim=[-np.deg2rad(170), np.deg2rad(170)]),
            RevoluteDH(d=0.00, a=0.50, alpha=0,         qlim=[-np.deg2rad(90),  np.deg2rad(90)]),
            RevoluteDH(d=0.00, a=0.45, alpha=0,         qlim=[-np.deg2rad(150), np.deg2rad(150)]),
            RevoluteDH(d=0.15, a=0.00, alpha=-np.pi/2, qlim=[-np.pi, np.pi]),
            RevoluteDH(d=0.00, a=0.00, alpha=np.pi/2,   qlim=[-np.deg2rad(120), np.deg2rad(120)]),
            RevoluteDH(d=0.10, a=0.00, alpha=0,         qlim=[-2*np.pi, 2*np.pi]),
        ]
        super().__init__(links, name="SupplierArm")
        self.qz = np.zeros(6)


if __name__ == "__main__":
    robot = SupplierArm()
    print(robot)

    env, poses = build_garage(launch=False)
    robot.base = poses["selector"]
    print("\nSupplier arm base set to:", robot.base)

    print("\nTool positions available:")
    for name, pose in poses["tools"].items():
        print(f"  {name}: {pose.t}")

    q_guess = robot.qz

    print("\nReachability check, full orientation constraint (gripper pointing down):")
    for name, pose in poses["tools"].items():
        target = SE3(pose.t) * SE3.Rx(np.pi)
        sol = robot.ikine_LM(target, q0=q_guess)
        status = "OK" if sol.success else "FAILED"
        print(f"  {name:12s}: {status}  (iterations: {sol.iterations})")

    print("\nReachability check, position-only (orientation free):")
    for name, pose in poses["tools"].items():
        target = SE3(pose.t)
        sol = robot.ikine_LM(target, q0=q_guess, mask=[1, 1, 1, 0, 0, 0])
        status = "OK" if sol.success else "FAILED"
        print(f"  {name:12s}: {status}  (iterations: {sol.iterations})")

    print("\nReachability check, handoff zone (gripper pointing down):")
    handoff_zone = SE3(-0.925, -0.08, 0.75) * SE3.Rx(np.pi)
    sol = robot.ikine_LM(handoff_zone, q0=q_guess)
    status = "OK" if sol.success else "FAILED"
    print(f"  handoff_zone: {status}  (iterations: {sol.iterations})")