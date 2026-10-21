"""Layout constants for Supply Bot, built on the garage table layout (Garage.py).

Author: [Your name]

Supply Bot stands on the RED square (the one Garage.py calls "ur3e"), at the
right-hand end of the 4.4 m table, facing -x (back along the table). It does two jobs:
  1. if the real robot needs hardware: carry a nut and a bolt of the requested sizes from
     their red boxes (four boxes: small/large nuts, small/large bolts) to the timber
  2. once the real robot has finished: carry the timber to the output area
The four red supply boxes and the output box are switched ON (see the SHOW_* flags below):
timber still move between the same spots, but no boxes are drawn.

World frame is the garage frame (metres): x along the table, y toward the pegboard wall,
z up, table top at TABLE_H. All sizes and positions are ESTIMATES - edit them here.
"""
import numpy as np
from spatialmath import SE3

from Garage import BASE_Y, BASES, TABLE_H, TOOL_FACE_Y, WOOD_CENTRE, WOOD_SIZE, drill_point_pose

SHOW_SUPPLY_BOXES = True       # the four red nut/bolt boxes
SHOW_OUTPUT_BOX = True         # the output box beside the red square
SHOW_SHELVES = True            # two wall shelves with nut and bolt stock
SIDE = +1                      # +1 = boxes on the far (right-hand) side of the red square, -1 = table-centre side

RED_MARK = (BASES["ur3e"], BASE_Y)             # the red square
BASE_Z = TABLE_H + 0.006                       # arm base sits on the 6 mm square
BASE_POSE = SE3(RED_MARK[0], RED_MARK[1], BASE_Z) * SE3.Rz(np.pi)  # arm "forward" = world -x

# What the arm must keep clear of (used by the collision check)
FLOOR_LIMIT_Z = TABLE_H + 0.03                 # joints stay above the table top
WALL_LIMIT_Y = TOOL_FACE_Y - 0.12              # joints stay in front of the wall shelves

# Six red supply boxes, one per part type and size: small/large nuts, bolts and screws.
# Nuts are in the far column, bolts in the middle column, screws in the near column; small
# in the back row, large in the front row. half = half the object's height, so the object's centre
# sits on the box floor.
SUPPLY_BOX_SIZE = (0.09, 0.12)         # x, y
SUPPLY_BOX_FLOOR = TABLE_H + (0.01 if SHOW_SUPPLY_BOXES else 0.0)   # top of the box floor
SUPPLY_BOX_WALL_H = 0.03
SUPPLY_BOXES = {                       # (kind, size) -> box centre (x, y)
    ("nut", "small"): (RED_MARK[0] + SIDE * 0.45, RED_MARK[1] + 0.18),
    ("bolt", "small"): (RED_MARK[0] + SIDE * 0.35, RED_MARK[1] + 0.18),
    ("nut", "large"): (RED_MARK[0] + SIDE * 0.45, RED_MARK[1] - 0.16),
    ("bolt", "large"): (RED_MARK[0] + SIDE * 0.35, RED_MARK[1] - 0.16),
    ("screw", "small"): (RED_MARK[0] + SIDE * 0.25, RED_MARK[1] + 0.18),
    ("screw", "large"): (RED_MARK[0] + SIDE * 0.25, RED_MARK[1] - 0.16),
}
SIZES = {
    "small": {"nut_half": 0.004, "bolt_half": 0.005, "screw_half": 0.003},
    "large": {"nut_half": 0.007, "bolt_half": 0.008, "screw_half": 0.004},
}

# The timber (same block as in Garage.py) and where the nut and bolt go on it
TIMBER_TOP = WOOD_CENTRE[2] + WOOD_SIZE[2] / 2
# The nut and bolt lie on the left half of the timber, clear of the cut line (x = centre) and of the
# drilled hole (0.10 m right of the centre), where the screw goes.
NUT_DX, BOLT_DX = -0.10, -0.04                       # offsets along the timber
HARDWARE_ORDER = ("nut", "bolt", "screw")            # the order Supply Bot brings them
_DX = {"nut": NUT_DX, "bolt": BOLT_DX}

# Screws stand upright (head up) in their boxes and go head-up into the hole the real robot drills.
# length = shank length, head_x = head centre along the screw axis, head_len = head thickness (see tools.py)
SCREW_SPECS = {"small": {"length": 0.025, "head_x": 0.015, "head_len": 0.005},
               "large": {"length": 0.040, "head_x": 0.024, "head_len": 0.008}}
SCREW_UP = SE3.Ry(-np.pi / 2)                        # the screw model's x axis (towards the head) turned to point up
SEAT_DEPTH = 0.010                                   # how far the tip is in the hole when Supply Bot lets go (m)
HOLE_XY = tuple(float(v) for v in drill_point_pose().t[:2])    # the drill mark on the timber
HOLE_HIDDEN_Z = -5.0                                 # the hole disc waits far below the scene until it is drilled

# Output area (and box, when switched on) beside the red square
BOX_CENTRE = (RED_MARK[0] + SIDE * 0.35, RED_MARK[1] - 0.42)
BOX_INNER = (0.38, 0.18)
BOX_WALL = 0.01
BOX_HEIGHT = 0.10
BOX_FLOOR = TABLE_H + (0.01 if SHOW_OUTPUT_BOX else 0.0)

# Two wall shelves on the pegboard holding nut and bolt stock: (name, height of shelf board)
SHELF_X = RED_MARK[0]
SHELVES = [("nuts", 1.15), ("bolts", 1.45)]
SHELF_SIZE = (0.34, 0.12, 0.02)        # x, depth, thickness


def start_pose(kind, size):
    """Pose of the part ('nut', 'bolt' or 'screw') waiting in its red box."""
    cx, cy = SUPPLY_BOXES[(kind, size)]
    if kind == "screw":                    # stands on its tip, head up, so the gripper takes it from above
        return SE3(cx, cy, SUPPLY_BOX_FLOOR + SCREW_SPECS[size]["length"] / 2 + 0.002) * SCREW_UP
    return SE3(cx, cy, SUPPLY_BOX_FLOOR + SIZES[size][f"{kind}_half"])


def nut_start(size):
    """Pose of the nut waiting in its red box."""
    return start_pose("nut", size)


def bolt_start(size):
    """Pose of the bolt waiting in its red box."""
    return start_pose("bolt", size)


def screw_start(size):
    """Pose of the screw waiting in its red box."""
    return start_pose("screw", size)


def screw_pose(size, depth, angle=0.0):
    """Screw standing in the hole, tip `depth` m below the timber top, turned `angle` rad about the vertical."""
    return SE3(HOLE_XY[0], HOLE_XY[1], TIMBER_TOP - depth + SCREW_SPECS[size]["length"] / 2) * SE3.Rz(angle) * SCREW_UP


def screw_driven_depth(size):
    """Depth at which the screw is fully in (head flush with the timber)."""
    return SCREW_SPECS[size]["length"]


def screw_head_top(size, depth):
    """Height of the top of the screw head when the tip is `depth` below the timber top."""
    spec = SCREW_SPECS[size]
    return TIMBER_TOP - depth + spec["length"] / 2 + spec["head_x"] + spec["head_len"] / 2


def hole_pose():
    """The drilled hole: a dark disc on the timber top (appears when the drill has gone in)."""
    return SE3(HOLE_XY[0], HOLE_XY[1], TIMBER_TOP + 0.001)


def hole_hidden_pose():
    return SE3(HOLE_XY[0], HOLE_XY[1], HOLE_HIDDEN_Z)


def on_wood(kind, size):
    """Where a nut or bolt sits on top of the timber, or where the screw is set down in the drilled hole."""
    if kind == "screw":
        return screw_pose(size, SEAT_DEPTH)
    return SE3(WOOD_CENTRE[0] + _DX[kind], WOOD_CENTRE[1], TIMBER_TOP + SIZES[size][f"{kind}_half"])


def screw_on_wood(size):
    """Where the screw is set down: standing in the drilled hole."""
    return on_wood("screw", size)


def grasp_rot(kind, size):
    """Gripper pose in the object's frame: from above for a nut or bolt lying flat, from above on the head for the screw."""
    if kind == "screw":
        return SE3(SCREW_SPECS[size]["head_x"], 0, 0) * SE3.Ry(-np.pi / 2)
    return GRIP_TO_OBJ


def loose_hardware(sizes):
    """The nut and bolt (they just lie on the timber), without the screw."""
    return [(k, s) for k, s in hardware(sizes) if k != "screw"]


def hardware(sizes):
    """[(kind, size)] for the hardware asked for, in delivery order. sizes: {'nut':..,'bolt':..,'screw':..} or None."""
    return [(k, sizes[k]) for k in HARDWARE_ORDER if sizes and k in sizes]


def describe(sizes):
    """'small nut, large bolt and small screw'"""
    parts = [f"{size} {kind}" for kind, size in hardware(sizes)]
    return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] if parts else "nothing"


def nut_on_wood(size):
    """Where the nut sits on top of the timber."""
    return SE3(WOOD_CENTRE[0] + NUT_DX, WOOD_CENTRE[1], TIMBER_TOP + SIZES[size]["nut_half"])


def bolt_on_wood(size):
    """Where the bolt sits on top of the timber."""
    return SE3(WOOD_CENTRE[0] + BOLT_DX, WOOD_CENTRE[1], TIMBER_TOP + SIZES[size]["bolt_half"])


# Timber poses (object-frame origin at the centre of the object, axes aligned with world)
WOOD_START = SE3(*WOOD_CENTRE)
WOOD_IN_BOX = SE3(BOX_CENTRE[0], BOX_CENTRE[1], BOX_FLOOR + WOOD_SIZE[2] / 2)

# Gripper pointing straight down at an object lying flat: object frame = gripper frame
# turned 180 degrees about x (its own inverse, so it works both ways).
GRIP_TO_OBJ = SE3.Rx(np.pi)
