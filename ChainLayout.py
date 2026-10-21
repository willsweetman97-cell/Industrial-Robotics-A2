"""Layout constants for the ToolBot -> TransferBot hand-over chain, built on Garage.py.

Author: [Your name]

ToolBot (blue square) takes the requested tool off the pegboard and puts it on the first
hand-over square (the "supply" mark). TransferBot (yellow square) moves it to the second
hand-over square, where the real robot uses it, then puts the used tool on the extra green
"used tools" mark between the two robots. ToolBot takes it from there back to the pegboard.
Everything is in the Garage.py world frame (metres).
"""
import numpy as np
from spatialmath import SE3

import Garage
import supply_layout as S

TABLE_H = Garage.TABLE_H
TOOL_Y = Garage.TOOL_FACE_Y - 0.02            # y of the tool bodies hanging on the pegboard
HAND_Y = -0.08                                # the two green hand-over squares
HAND1_X, HAND2_X = -0.925, 0.025              # ToolBot -> TransferBot, TransferBot -> real robot
USED_X, USED_Y = -0.70, -0.12                 # extra green "used tools" mark (drawn by big_sim, not in Garage.py)
USED_SIZE = 0.12                              # side of that mark (m), same as the other green squares
TOOL_NAMES = list(Garage.tool_poses())        # saw, hammer, drill, screwdriver, wrench

# What the real robot can be asked to do, and the tools each job uses (in order)
JOBS = {"sawing": ["saw"],
        "drilling and screwing": ["drill", "screwdriver"],
        "both": ["saw", "drill", "screwdriver"]}

# Half the tool's thickness (its y size), so it rests on the table when lying flat
TOOL_REST_HALF = {"saw": 0.015, "hammer": 0.025, "drill": 0.035, "screwdriver": 0.02, "wrench": 0.005}

# Tool frame: x = width, y = thickness (toward the wall when hanging), z = long axis (up).
LYING = SE3.Rx(-np.pi / 2)    # lying on the table: long axis along +y, thickness pointing down
# Gripper orientation relative to the tool frame at the grasp: TCP = tool * GRASP_ROT.
# On the wall this points the gripper at the wall; on the table it points straight down.
GRASP_ROT = SE3.Rx(-np.pi / 2)

TOOLBOT_BASE = Garage.base_poses()["selector"] * SE3.Rz(np.pi / 4)     # faces the wall/handoff
TRANSFER_BASE = Garage.base_poses()["delivery"] * SE3.Rz(-np.pi / 2)   # faces the front edge
UR3E_BASE = Garage.base_poses()["working"] * SE3.Rz(-np.pi / 2)        # the real robot: green square, faces the table front
UR_TILT_DEG = 20     # the UR3e grips the tool at this angle: the hand-over square is just beyond its straight-down reach
UR_GRASP = SE3.Rz(np.deg2rad(UR_TILT_DEG)) * GRASP_ROT
TOOLBOT_GRIP = 0.12            # ToolBot's gripper length (flange to tool centre point)


def tool_start(name):
    """Where the tool hangs on the pegboard (tool frame = world axes)."""
    p = Garage.tool_poses()[name].t
    return SE3(p[0], TOOL_Y, p[2])


def mark_xy(which):
    """(x, y) of hand-over square 1 (ToolBot -> TransferBot), 2 (real robot) or "used"."""
    return {1: (HAND1_X, HAND_Y), 2: (HAND2_X, HAND_Y), "used": (USED_X, USED_Y)}[which]


def tool_on_square(name, which):
    """Where the tool lies on hand-over square 1 or 2, or on the "used" mark."""
    x, y = mark_xy(which)
    return SE3(x, y, TABLE_H + TOOL_REST_HALF[name]) * LYING


def _pose(columns, p):
    """SE3 from the three axis directions (columns of the rotation) and a position."""
    M = np.eye(4)
    M[:3, :3] = np.column_stack(columns)
    M[:3, 3] = p
    return SE3(M)


SCREW_TWIST_DEG = (20, -20, 20, -20)          # the screwdriver turns back and forth; the screw turns with it
TIP_GAP = 0.002                               # the screwdriver tip stops this far above the screw head to start


def screw_motion(size):
    """[(screw pose before, screw pose after)] for each screwing stroke: it turns with the driver and sinks in."""
    n = len(SCREW_TWIST_DEG)
    depths = np.linspace(S.SEAT_DEPTH, S.screw_driven_depth(size), n + 1)
    motion, prev = [], 0.0
    for k, deg in enumerate(SCREW_TWIST_DEG):
        angle = np.deg2rad(deg)
        motion.append((S.screw_pose(size, depths[k], prev), S.screw_pose(size, depths[k + 1], angle)))
        prev = angle
    return motion


def real_robot_path(tool, screw=None):
    """Tool-origin poses for the real robot's job on the timber.

    [0] is a pose clear above the start, [1] the start of the work, the rest are the strokes.
    Tool frames: saw blade edge = -x, handle along +z; drill chuck = +x, handle -z;
    screwdriver tip = -z (see tools.py). screw = size of the screw standing in the drilled hole (screwdriver
    only): the screwdriver then goes down onto its head and drives it in, otherwise it just twists on the mark.
    """
    cx, cy, cz = Garage.WOOD_CENTRE
    top = cz + Garage.WOOD_SIZE[2] / 2
    if tool == "saw":                     # blade in the cut plane x = cx, edge down, strokes along y
        R = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
        y0, amp = cy + 0.10, 0.04         # the cut is 0.10 m in front of the grip
        legs = [(y0 + amp, top + 0.16), (y0 + amp, top + 0.06)]
        for k, z in enumerate((0.05, 0.04, 0.035, 0.03, 0.02)):
            legs.append((y0 - amp if k % 2 == 0 else y0 + amp, top + z))
        return [_pose(R, (cx, y, z)) for y, z in legs]
    px, py = Garage.drill_point_pose().t[:2]
    if tool == "drill":                   # chuck pointing down at the drill mark, two pecks
        R = [(0, 0, -1), (0, -1, 0), (-1, 0, 0)]
        return [_pose(R, (px + 0.05, py, top + z)) for z in (0.22, 0.12, 0.085, 0.12, 0.085)]
    if tool == "screwdriver":             # tip down on the mark, twisted back and forth
        base = np.array([(-1, 0, 0), (0, -1, 0), (0, 0, 1)], float).T      # columns = the tool's x, y, z axes
        if screw:                         # tip (0.12 m below the tool origin) on the head, following it down
            def origin_z(depth):
                return S.screw_head_top(screw, depth) + 0.12 + TIP_GAP
            z0 = origin_z(S.SEAT_DEPTH)
            path = [_pose(list(base.T), (px, py, z)) for z in (z0 + 0.10, z0 + 0.03, z0)]
            depths = np.linspace(S.SEAT_DEPTH, S.screw_driven_depth(screw), len(SCREW_TWIST_DEG) + 1)
            for k, angle in enumerate(SCREW_TWIST_DEG):
                c, s_ = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
                R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ base
                path.append(_pose(list(R.T), (px, py, origin_z(depths[k + 1]))))
            return path
        path = [_pose(list(base.T), (px, py, top + z)) for z in (0.24, 0.14, 0.12)]
        for k, angle in enumerate((20, -20, 20, -20)):
            c, s_ = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
            Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            R = Rz @ base
            path.append(_pose(list(R.T), (px, py, top + 0.12 - 0.005 * (k + 1))))
        return path
    raise ValueError(f"the real robot has no job for the {tool}")
