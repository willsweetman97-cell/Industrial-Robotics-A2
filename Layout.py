"""Layout constants for the Robot 2 bench-and-board patch.

Author: Jake Salamon

World frame (metres): origin at the centre of the BLUE mark on the bench top,
x to the robot's right, y toward the board, z up. Robot 2 is mounted on the blue
mark; the tool starts on the green mark to its right and is returned to the board.

All sizes are ESTIMATES from a photo - measure the real bench and board and edit here.
Both the scene and the motion code read these values, so they stay consistent.
"""
import numpy as np
from spatialmath import SE3

# Bench patch: a square that just fits the two marks (not the full bench)
BENCH_CENTRE = (0.25, 0.0)
BENCH_SIZE = (0.8, 0.8, 0.04)  # x, y, thickness (top surface at z = 0)

BLUE_MARK = (0.0, 0.0)
GREEN_MARK = (0.32, 0.0)       # to Robot 2's right
MARK_SIZES = {"blue": 0.10, "green": 0.05}

# Board patch on the wall behind the bench (only the part holding our 3 tools)
WALL_Y = 0.40                  # board face, measured from the blue mark
BOARD_CENTRE_X = 0.25
BOARD_SIZE = (0.8, 0.02, 0.55)  # x, thickness, height
BOARD_CENTRE_Z = 0.525          # board spans z = 0.25 .. 0.80
HANG_Z = 0.60                   # height of each tool's grip point when hung
SLOT_X = {"hammer": 0.05, "screwdriver": 0.25, "saw": 0.45}
RETURN_TOOL = "saw"             # starts on the green mark, goes to the empty slot

GRASP_Z = 0.008                 # tool frame height when lying on the bench

# Tool frame vs gripper frame: the tool lies flat, top face up, while the gripper
# points down, so the tool is the gripper frame turned 180 degrees about x.
TCP_TO_TOOL = SE3.Rx(np.pi)
# Gripper orientation at the board: approach axis toward the wall, tool hanging
# vertically with its grip end up and its top face toward the robot.
HANG_ORIENT = SE3.Rx(-np.pi / 2) * SE3.Rz(-np.pi / 2)
TOOL_THICKNESS = 0.006          # tool back sits this far in front of the board face


def hung_pose(kind):
    """Pose of a tool's grip-point frame when it hangs in its slot on the board."""
    return SE3(SLOT_X[kind], WALL_Y - TOOL_THICKNESS, HANG_Z) * HANG_ORIENT * TCP_TO_TOOL
