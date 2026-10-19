"""Robot 2 - kinematic model (tool put-away robot).

Author: [Your name]

Parameter-based standard-DH model of a 6-DoF industrial arm. Every value in
DH_TABLE below is a PLACEHOLDER: replace it with the dimensions from your
chosen robot's datasheet or kinematic drawing, and cite the source in
SOURCES. Units: metres and radians (limits are written in degrees for readability).

This file builds the kinematics only. For the realistic Swift appearance
(meshes, colours, proportions) follow the 41013 lab examples for ir_support's
DHRobot3D and reuse build_links() below, so the kinematics stay in one place.
"""
from __future__ import annotations

import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3

ROBOT_NAME = "Robot2"
MANUFACTURER = "[Manufacturer]"
SOURCES = "[Datasheet / manual / drawing the DH values came from]"

# Standard DH rows: (d [m], a [m], alpha [rad], offset [rad], q_min [deg], q_max [deg])
DH_TABLE = [
    (0.350, 0.050, -np.pi / 2, 0.0, -170, 170),   # joint 1
    (0.000, 0.400,  0.0,       0.0, -135, 135),   # joint 2
    (0.000, 0.030, -np.pi / 2, 0.0, -150, 150),   # joint 3
    (0.380, 0.000,  np.pi / 2, 0.0, -180, 180),   # joint 4
    (0.000, 0.000, -np.pi / 2, 0.0, -120, 120),   # joint 5
    (0.080, 0.000,  0.0,       0.0, -360, 360),   # joint 6
]

# Ready pose: arm raised, gripper out in front. Must sit inside the joint limits and
# must not fold the gripper into the base (the old [0,-30,60,0,30,0] did).
Q_HOME = np.deg2rad([0, -70, 30, 0, 60, 0])
GRIPPER_LENGTH = 0.15                        # flange to tool centre point (m)


def build_links() -> list[RevoluteDH]:
    """Return the DH links built from DH_TABLE (reuse this for a DHRobot3D version)."""
    links = []
    for d, a, alpha, offset, q_min, q_max in DH_TABLE:
        links.append(
            RevoluteDH(d=d, a=a, alpha=alpha, offset=offset,
                       qlim=np.deg2rad([q_min, q_max]))
        )
    return links


class Robot2(DHRobot):
    """6-DoF DH model. Pass base=SE3(...) to place it in the shared workcell frame."""

    def __init__(self, base: SE3 | None = None):
        super().__init__(build_links(), name=ROBOT_NAME, manufacturer=MANUFACTURER)
        if base is not None:
            self.base = base
        self.tool = SE3(0, 0, GRIPPER_LENGTH)
        self.q = Q_HOME
        self.addconfiguration("home", Q_HOME)
