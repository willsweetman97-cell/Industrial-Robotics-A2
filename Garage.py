from __future__ import annotations

from dataclasses import dataclass

import spatialgeometry as sg
from spatialmath import SE3

GARAGE_W = 6.0          # x extent
GARAGE_D = 5.0          # y extent
GARAGE_H = 2.8          # wall height
WALL_T = 0.1

BACK_WALL_Y = 0.95      # table sits close to the back wall so the tools are in reach

TABLE_W = 4.4           # x length
TABLE_D = 1.0           # y depth
TABLE_H = 0.75          # table-top height above the floor
TOP_T = 0.06            # table-top thickness
LEG = 0.08

BASE_SQ = 0.30          # square marker for each robot base
BASE_Y = 0.22           # base centre, y on the table

Colour = tuple

# Colours (r, g, b, a)
CONCRETE = (0.55, 0.55, 0.57, 1.0)
WALL = (0.85, 0.85, 0.82, 1.0)
TABLE_TOP = (0.45, 0.45, 0.50, 1.0)
TABLE_LEG = (0.2, 0.2, 0.22, 1.0)
PEGBOARD = (0.78, 0.62, 0.42, 1.0)
STEEL = (0.7, 0.7, 0.72, 1.0)
BLACK = (0.1, 0.1, 0.1, 1.0)
RED = (0.75, 0.1, 0.1, 1.0)
ORANGE = (0.95, 0.5, 0.1, 1.0)
WOOD = (0.76, 0.6, 0.38, 1.0)
HANDLE = (0.55, 0.3, 0.12, 1.0)
GREEN = (0.1, 0.65, 0.25, 1.0)

# Base marker colours so each robot spot is easy to tell apart
BASE_COLOURS = {
    "selector": (0.15, 0.45, 0.9, 1.0),
    "delivery": (0.95, 0.75, 0.1, 1.0),
    "working": (0.15, 0.7, 0.3, 1.0),
    "ur3e": (0.85, 0.2, 0.2, 1.0),
}


@dataclass(frozen=True)
class BaseSpec:
    name: str
    x: float


# Robot order along the table. Selector is under the tool wall, working robot
# sits next to the timber, UR3e is at the far end.
BASES = [
    BaseSpec("selector", -1.40),
    BaseSpec("delivery", -0.45),
    BaseSpec("working", 0.50),
    BaseSpec("ur3e", 1.45),
]




def _make(cls, pose, colour, **kw):
    """Newer spatialgeometry uses `pose`, older versions use `base`."""
    try:
        return cls(pose=pose, color=colour, **kw)
    except TypeError:
        return cls(base=pose, color=colour, **kw)


def box(size, centre, colour, rpy=(0, 0, 0)):
    """Cuboid of size (x, y, z) centred at `centre`, optional roll-pitch-yaw."""
    pose = SE3(*centre) * SE3.RPY(rpy, order="xyz")
    return _make(sg.Cuboid, pose, colour, scale=list(size))


def cyl(radius, length, centre, colour, rpy=(0, 0, 0)):
    """Cylinder whose axis is local z, centred at `centre`."""
    pose = SE3(*centre) * SE3.RPY(rpy, order="xyz")
    return _make(sg.Cylinder, pose, colour, radius=radius, length=length)


# --------------------------------------------------------------------------
# Garage shell
# --------------------------------------------------------------------------


def make_garage_shell():
    half_w, half_d = GARAGE_W / 2, GARAGE_D / 2
    # Floor top surface is z = 0
    objs = [box((GARAGE_W, GARAGE_D, 0.05), (0, 0, -0.025), CONCRETE)]
    # Back wall (behind the pegboard)
    objs.append(box((GARAGE_W, WALL_T, GARAGE_H),
                    (0, BACK_WALL_Y + WALL_T / 2, GARAGE_H / 2), WALL))
    # Side walls run from the back wall towards the open front
    side_len = (BACK_WALL_Y + WALL_T) + half_d
    side_cy = (BACK_WALL_Y + WALL_T - half_d) / 2
    for sx in (-1, 1):
        objs.append(box((WALL_T, side_len, GARAGE_H),
                        (sx * (half_w + WALL_T / 2), side_cy, GARAGE_H / 2), WALL))
    return objs


# --------------------------------------------------------------------------
# Table and robot base markers
# --------------------------------------------------------------------------


def make_table():
    objs = [box((TABLE_W, TABLE_D, TOP_T), (0, 0, TABLE_H - TOP_T / 2), TABLE_TOP)]
    leg_h = TABLE_H - TOP_T
    for lx in (-TABLE_W / 2 + LEG, TABLE_W / 2 - LEG):
        for ly in (-TABLE_D / 2 + LEG, TABLE_D / 2 - LEG):
            objs.append(box((LEG, LEG, leg_h), (lx, ly, leg_h / 2), TABLE_LEG))
    # Cross bars for a sturdier look
    for ly in (-TABLE_D / 2 + LEG, TABLE_D / 2 - LEG):
        objs.append(box((TABLE_W - 2 * LEG, 0.04, 0.06), (0, ly, 0.2), TABLE_LEG))
    return objs


def base_poses():
    """SE3 pose of each robot base on the table top (z = table surface)."""
    return {b.name: SE3(b.x, BASE_Y, TABLE_H) for b in BASES}


def make_base_markers():
    """Flat coloured squares showing where each robot base will be bolted."""
    objs = []
    for b in BASES:
        objs.append(box((BASE_SQ, BASE_SQ, 0.006),
                        (b.x, BASE_Y, TABLE_H + 0.003), BASE_COLOURS[b.name]))
        # Thin dark frame so the square reads as a mounting plate
        t = 0.012
        for dx in (-1, 1):
            objs.append(box((t, BASE_SQ + t, 0.008),
                            (b.x + dx * BASE_SQ / 2, BASE_Y, TABLE_H + 0.004), BLACK))
        for dy in (-1, 1):
            objs.append(box((BASE_SQ + t, t, 0.008),
                            (b.x, BASE_Y + dy * BASE_SQ / 2, TABLE_H + 0.004), BLACK))
    return objs


def make_handoff_zones():
    """Small green squares where tools are passed between robots."""
    objs = []
    for x in (-0.925, 0.025):
        objs.append(box((0.12, 0.12, 0.004), (x, -0.08, TABLE_H + 0.002), GREEN))
    return objs


# --------------------------------------------------------------------------
# Pegboard and wall-mounted tools
# --------------------------------------------------------------------------

PEG_Z0, PEG_Z1 = 1.00, 1.75     # pegboard bottom / top height
PEG_W = TABLE_W
PEG_T = 0.02
FACE_Y = BACK_WALL_Y - PEG_T    # y of the pegboard front face


def make_pegboard():
    objs = [box((PEG_W, PEG_T, PEG_Z1 - PEG_Z0),
                (0, BACK_WALL_Y - PEG_T / 2, (PEG_Z0 + PEG_Z1) / 2), PEGBOARD)]
    # Shelf under the tools
    objs.append(box((PEG_W, 0.10, 0.02), (0, BACK_WALL_Y - 0.05, PEG_Z0 - 0.01), TABLE_LEG))
    return objs


def tool_poses():
    """Where each tool hangs. Handy as grasp targets: x, y(face), z."""
    return {
        "saw": SE3(-1.90, FACE_Y, 1.38),
        "hammer": SE3(-1.55, FACE_Y, 1.40),
        "drill": SE3(-1.15, FACE_Y, 1.38),
        "screwdriver": SE3(-0.85, FACE_Y, 1.40),
        "wrench": SE3(-0.55, FACE_Y, 1.40),
        "clamp": SE3(1.20, FACE_Y, 1.38),
    }


def make_tools():
    objs = []
    p = tool_poses()
    d = 0.02  # offset off the board face

    # Saw: blade hangs from handle at top
    x, y, z = p["saw"].t
    objs.append(box((0.08, 0.008, 0.42), (x, y - d, z - 0.08), STEEL))
    objs.append(box((0.10, 0.03, 0.16), (x, y - d, z + 0.21), HANDLE))
    for i in range(10):  # teeth
        objs.append(box((0.012, 0.01, 0.012), (x + 0.04, y - d, z - 0.27 + i * 0.04), STEEL))

    # Hammer: vertical handle, head at the top
    x, y, z = p["hammer"].t
    objs.append(box((0.03, 0.03, 0.34), (x, y - d, z), HANDLE))
    objs.append(box((0.16, 0.05, 0.05), (x, y - d, z + 0.19), STEEL))

    # Drill: body, handle, chuck
    x, y, z = p["drill"].t
    objs.append(box((0.20, 0.07, 0.07), (x, y - d - 0.01, z + 0.05), ORANGE))
    objs.append(box((0.045, 0.05, 0.14), (x - 0.04, y - d - 0.01, z - 0.04), BLACK))
    objs.append(cyl(0.015, 0.07, (x + 0.135, y - d - 0.01, z + 0.05), STEEL, rpy=(0, 1.5708, 0)))
    objs.append(cyl(0.004, 0.08, (x + 0.205, y - d - 0.01, z + 0.05), STEEL, rpy=(0, 1.5708, 0)))

    # Screwdriver
    x, y, z = p["screwdriver"].t
    objs.append(cyl(0.02, 0.12, (x, y - d, z + 0.08), RED))
    objs.append(cyl(0.005, 0.14, (x, y - d, z - 0.05), STEEL))

    # Wrench
    x, y, z = p["wrench"].t
    objs.append(box((0.03, 0.01, 0.30), (x, y - d, z), STEEL))
    objs.append(box((0.07, 0.01, 0.06), (x, y - d, z + 0.16), STEEL))

    # Clamp (near the UR3e end)
    x, y, z = p["clamp"].t
    objs.append(box((0.03, 0.02, 0.22), (x - 0.06, y - d, z), STEEL))
    objs.append(box((0.14, 0.02, 0.03), (x, y - d, z + 0.095), STEEL))
    objs.append(box((0.14, 0.02, 0.03), (x, y - d, z - 0.095), STEEL))
    return objs


# --------------------------------------------------------------------------
# Timber block + jig
# --------------------------------------------------------------------------

WOOD_SIZE = (0.30, 0.10, 0.05)          # x, y, z
WOOD_CENTRE = (0.50, -0.22, TABLE_H + 0.03 + WOOD_SIZE[2] / 2)  # on a 3 cm jig


def wood_top_z():
    return WOOD_CENTRE[2] + WOOD_SIZE[2] / 2


def cut_line_pose():
    """Start of the cut line on the top face (cut runs along +y)."""
    return SE3(WOOD_CENTRE[0], WOOD_CENTRE[1] - WOOD_SIZE[1] / 2, wood_top_z())


def drill_point_pose():
    return SE3(WOOD_CENTRE[0] + 0.10, WOOD_CENTRE[1], wood_top_z())


def make_wood_block():
    cx, cy, cz = WOOD_CENTRE
    objs = [
        # Jig / vice base plate
        box((0.36, 0.16, 0.03), (cx, cy, TABLE_H + 0.015), TABLE_LEG),
        # Two clamp stops
        box((0.02, 0.16, 0.06), (cx - 0.17, cy, TABLE_H + 0.06), STEEL),
        box((0.02, 0.16, 0.06), (cx + 0.17, cy, TABLE_H + 0.06), STEEL),
        # The timber
        box(WOOD_SIZE, WOOD_CENTRE, WOOD),
    ]
    top = wood_top_z()
    # Cut line: thin dark strip across the block (runs along y)
    objs.append(box((0.004, WOOD_SIZE[1], 0.002), (cx, cy, top + 0.001), BLACK))
    # Drill mark: small dark disc
    objs.append(cyl(0.008, 0.002, (cx + 0.10, cy, top + 0.001), BLACK))
    return objs


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def make_scene_objects():
    """All static scene geometry as a flat list (no Swift needed)."""
    return (make_garage_shell() + make_table() + make_base_markers()
            + make_handoff_zones() + make_pegboard() + make_tools()
            + make_wood_block())


def build_garage(launch: bool = True, realtime: bool = True):
    """
    Create the garage scene. Returns (env, poses).

    poses has the robot base SE3s plus tool, cut-line and drill-point targets.
    Pass launch=False to just get the geometry list without opening Swift.
    """
    objs = make_scene_objects()
    poses = {
        **base_poses(),
        "tools": tool_poses(),
        "cut_line": cut_line_pose(),
        "drill_point": drill_point_pose(),
    }
    if not launch:
        return objs, poses

    import swift  # imported late so the geometry can be tested without it

    env = swift.Swift()
    try:
        # Hide Swift's default ground plane so it doesn't clash with our floor
        env.launch(realtime=realtime, ground_opacity=0.0)
    except TypeError:  # older Swift versions don't have this option
        env.launch(realtime=realtime)
    for o in objs:
        env.add(o)
    return env, poses


if __name__ == "__main__":
    env, poses = build_garage()
    print("Robot base poses:")
    for name in ("selector", "delivery", "working", "ur3e"):
        print(f"  {name:9s} {poses[name].t}")
    env.hold()
