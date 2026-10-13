import spatialgeometry as sg
from spatialmath import SE3

TABLE_H = 0.75                 # table-top height
BACK_Y = 0.95                  # back wall position (table sits close to the tool wall)
BASE_Y = 0.22                  # robot base squares, y on the table
BASES = {"selector": -1.40, "delivery": -0.45, "working": 0.50, "ur3e": 1.45}   # x positions
COLOURS = {"selector": (0.15, 0.45, 0.9, 1), "delivery": (0.95, 0.75, 0.1, 1),
           "working": (0.15, 0.7, 0.3, 1), "ur3e": (0.85, 0.2, 0.2, 1)}
GREY, WALL, DARK = (0.55, 0.55, 0.57, 1), (0.85, 0.85, 0.82, 1), (0.2, 0.2, 0.22, 1)
STEEL, WOOD, BLACK, ORANGE = (0.7, 0.7, 0.72, 1), (0.76, 0.6, 0.38, 1), (0.1, 0.1, 0.1, 1), (0.95, 0.5, 0.1, 1)
WOOD_SIZE = (0.30, 0.10, 0.05)
WOOD_CENTRE = (0.50, -0.22, TABLE_H + 0.03 + WOOD_SIZE[2] / 2)   # sits on a 3 cm jig
TOOL_FACE_Y = BACK_Y - 0.02    # y of the pegboard face


def _make(cls, pose, colour, **kw):
    """Newer spatialgeometry uses `pose`, older versions use `base`."""
    try:
        return cls(pose=pose, color=colour, **kw)
    except TypeError:
        return cls(base=pose, color=colour, **kw)


def box(size, centre, colour):
    return _make(sg.Cuboid, SE3(*centre), colour, scale=list(size))


def cyl(radius, length, centre, colour, rpy=(0, 0, 0)):
    return _make(sg.Cylinder, SE3(*centre) * SE3.RPY(rpy), colour, radius=radius, length=length)


def base_poses():
    """SE3 of each robot base on the table top."""
    return {name: SE3(x, BASE_Y, TABLE_H) for name, x in BASES.items()}


def tool_poses():
    """Where each tool hangs on the pegboard (grasp targets)."""
    xs = {"saw": -1.90, "hammer": -1.55, "drill": -1.15, "screwdriver": -0.85, "wrench": -0.55}
    return {name: SE3(x, TOOL_FACE_Y, 1.38) for name, x in xs.items()}


def cut_line_pose():
    """Start of the cut line on the timber (cut runs along +y)."""
    return SE3(WOOD_CENTRE[0], WOOD_CENTRE[1] - WOOD_SIZE[1] / 2, WOOD_CENTRE[2] + WOOD_SIZE[2] / 2)


def drill_point_pose():
    return SE3(WOOD_CENTRE[0] + 0.10, WOOD_CENTRE[1], WOOD_CENTRE[2] + WOOD_SIZE[2] / 2)


def make_scene_objects():
    objs = [box((6, 5, 0.05), (0, 0, -0.025), GREY),                       # floor
            box((6, 0.1, 2.8), (0, BACK_Y + 0.05, 1.4), WALL)]             # back wall
    objs += [box((0.1, 3.0, 2.8), (sx * 3.05, -0.5, 1.4), WALL) for sx in (-1, 1)]   # side walls

    # Table: top plus four legs
    objs.append(box((4.4, 1.0, 0.06), (0, 0, TABLE_H - 0.03), GREY))
    objs += [box((0.08, 0.08, 0.69), (sx * 2.16, sy * 0.46, 0.345), DARK) for sx in (-1, 1) for sy in (-1, 1)]

    # Robot base squares and hand-over squares (flat, coloured)
    objs += [box((0.3, 0.3, 0.006), (x, BASE_Y, TABLE_H + 0.003), COLOURS[n]) for n, x in BASES.items()]
    objs += [box((0.12, 0.12, 0.004), (x, -0.08, TABLE_H + 0.002), (0.1, 0.65, 0.25, 1)) for x in (-0.925, 0.025)]

    # Tool wall: pegboard + simple tools (a body and a head/handle each)
    objs.append(box((4.4, 0.02, 0.75), (0, BACK_Y - 0.01, 1.375), (0.78, 0.62, 0.42, 1)))
    y = TOOL_FACE_Y - 0.02
    for name, pose in tool_poses().items():
        x, _, z = pose.t
        if name == "saw":
            objs += [box((0.08, 0.01, 0.40), (x, y, z - 0.08), STEEL), box((0.10, 0.03, 0.16), (x, y, z + 0.20), DARK)]
        elif name == "hammer":
            objs += [box((0.03, 0.03, 0.34), (x, y, z), (0.55, 0.3, 0.12, 1)), box((0.16, 0.05, 0.05), (x, y, z + 0.19), STEEL)]
        elif name == "drill":
            objs += [box((0.20, 0.07, 0.07), (x, y, z + 0.05), ORANGE), box((0.045, 0.05, 0.14), (x - 0.04, y, z - 0.04), BLACK)]
        elif name == "screwdriver":
            objs += [cyl(0.02, 0.12, (x, y, z + 0.08), (0.75, 0.1, 0.1, 1)), cyl(0.005, 0.14, (x, y, z - 0.05), STEEL)]
        else:   # wrench
            objs += [box((0.03, 0.01, 0.30), (x, y, z), STEEL), box((0.07, 0.01, 0.06), (x, y, z + 0.16), STEEL)]

    # Timber block in a jig, with a cut line and a drill mark on top
    cx, cy, cz = WOOD_CENTRE
    top = cz + WOOD_SIZE[2] / 2
    objs += [box((0.36, 0.16, 0.03), (cx, cy, TABLE_H + 0.015), DARK), box(WOOD_SIZE, WOOD_CENTRE, WOOD),
             box((0.004, WOOD_SIZE[1], 0.002), (cx, cy, top + 0.001), BLACK),
             cyl(0.008, 0.002, (cx + 0.10, cy, top + 0.001), BLACK)]
    return objs


def build_garage(launch=True, realtime=True, headless=False):
    """Returns (env, poses). With launch=False returns (geometry list, poses) and doesn't open Swift."""
    objs = make_scene_objects()
    poses = {**base_poses(), "tools": tool_poses(), "cut_line": cut_line_pose(), "drill_point": drill_point_pose()}
    if not launch:
        return objs, poses
    import swift
    env = swift.Swift()
    extra = {"headless": True} if headless else {}
    try:
        env.launch(realtime=realtime, ground_opacity=0.0, **extra)   # hide Swift's own ground plane
    except TypeError:   # older Swift versions lack these options
        env.launch(realtime=realtime)
    for o in objs:
        env.add(o)
    return env, poses


if __name__ == "__main__":
    env, _ = build_garage()
    env.hold()
