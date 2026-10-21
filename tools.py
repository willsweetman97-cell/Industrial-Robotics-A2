"""Placeholder tool models built from boxes (hammer, screwdriver, saw).

Author: [Your name]

Each tool's frame origin is its GRIP POINT. The tool extends along -x of its own
frame, so it hangs downward when its x-axis points up. Replace with real meshes later.
"""
from spatialgeometry import Cuboid, Cylinder
from spatialmath import SE3

# (box size x,y,z), (centre offset x,y,z), colour)
TOOL_PARTS = {
    "hammer": [
        ((0.20, 0.02, 0.010), (-0.100, 0, 0), (0.9, 0.9, 0.9)),
        ((0.05, 0.10, 0.025), (-0.205, 0, 0), (0.75, 0.1, 0.1)),
    ],
    "screwdriver": [
        ((0.08, 0.03, 0.030), (-0.040, 0, 0), (0.95, 0.5, 0.1)),
        ((0.12, 0.008, 0.008), (-0.140, 0, 0), (0.6, 0.6, 0.65)),
    ],
    # small hardware and the wood piece: frame origin at the centre of the object
    "nut": [
        ((0.022, 0.022, 0.010), (0, 0, 0), (0.6, 0.6, 0.65)),
        ((0.010, 0.010, 0.0104), (0, 0, 0), (0.12, 0.12, 0.15)),
    ],
    "bolt": [
        ((0.045, 0.008, 0.008), (0, 0, 0), (0.75, 0.75, 0.8)),
        ((0.012, 0.018, 0.012), (0.0285, 0, 0), (0.3, 0.3, 0.35)),
    ],
    # two sizes each, for the red supply boxes (frame origin at the object's centre)
    "nut_small": [
        ((0.018, 0.018, 0.008), (0, 0, 0), (0.6, 0.6, 0.65)),
        ((0.008, 0.008, 0.0084), (0, 0, 0), (0.12, 0.12, 0.15)),
    ],
    "nut_large": [
        ((0.030, 0.030, 0.014), (0, 0, 0), (0.6, 0.6, 0.65)),
        ((0.014, 0.014, 0.0144), (0, 0, 0), (0.12, 0.12, 0.15)),
    ],
    "bolt_small": [
        ((0.035, 0.006, 0.006), (0, 0, 0), (0.75, 0.75, 0.8)),
        ((0.010, 0.014, 0.010), (0.0225, 0, 0), (0.3, 0.3, 0.35)),
    ],
    "bolt_large": [
        ((0.055, 0.010, 0.010), (0, 0, 0), (0.75, 0.75, 0.8)),
        ((0.016, 0.024, 0.016), (0.0355, 0, 0), (0.3, 0.3, 0.35)),
    ],
    "screw_small": [
        ((0.025, 0.004, 0.004), (0, 0, 0), (0.85, 0.7, 0.3)),
        ((0.005, 0.010, 0.006), (0.015, 0, 0), (0.7, 0.55, 0.2)),
    ],
    "screw_large": [
        ((0.040, 0.006, 0.006), (0, 0, 0), (0.85, 0.7, 0.3)),
        ((0.008, 0.014, 0.008), (0.024, 0, 0), (0.7, 0.55, 0.2)),
    ],
    "hole": [   # the hole the real robot drills: a dark disc, frame origin at its centre
        ((0.0055, 0.002), (0, 0, 0), (0.05, 0.03, 0.02, 1), "cyl"),
    ],
    "wood": [   # the timber block, same size and colour as in Garage.py
        ((0.30, 0.10, 0.05), (0, 0, 0), (0.76, 0.6, 0.38)),
    ],
    # The pegboard tools from Garage.py. Frame origin = the grasp point Garage.py hangs the
    # tool by (x, y of the tool body, z = 1.38); x = width, y = thickness, z = long axis.
    "tool_saw": [
        ((0.08, 0.01, 0.40), (0, 0, -0.08), (0.7, 0.7, 0.72, 1)),
        ((0.10, 0.03, 0.16), (0, 0, 0.20), (0.2, 0.2, 0.22, 1)),
    ],
    "tool_hammer": [
        ((0.03, 0.03, 0.34), (0, 0, 0), (0.55, 0.3, 0.12, 1)),
        ((0.16, 0.05, 0.05), (0, 0, 0.19), (0.7, 0.7, 0.72, 1)),
    ],
    "tool_drill": [
        ((0.20, 0.07, 0.07), (0, 0, 0.05), (0.95, 0.5, 0.1, 1)),
        ((0.045, 0.05, 0.14), (-0.04, 0, -0.04), (0.1, 0.1, 0.1, 1)),
    ],
    "tool_screwdriver": [
        ((0.02, 0.12), (0, 0, 0.08), (0.75, 0.1, 0.1, 1), "cyl"),
        ((0.005, 0.14), (0, 0, -0.05), (0.7, 0.7, 0.72, 1), "cyl"),
    ],
    "tool_wrench": [
        ((0.03, 0.01, 0.30), (0, 0, 0), (0.7, 0.7, 0.72, 1)),
        ((0.07, 0.01, 0.06), (0, 0, 0.16), (0.7, 0.7, 0.72, 1)),
    ],
    "saw": [
        ((0.08, 0.03, 0.015), (-0.040, 0, 0), (0.15, 0.15, 0.2)),
        ((0.16, 0.05, 0.004), (-0.160, 0, 0), (0.7, 0.7, 0.75)),
    ],
}


class ToolModel:
    """A tool made of boxes that moves as one rigid body."""

    def __init__(self, kind):
        self.parts = []
        for part in TOOL_PARTS[kind]:
            size, offset, color = part[:3]
            if len(part) > 3 and part[3] == "cyl":          # size = (radius, length), axis = z
                shape = Cylinder(radius=size[0], length=size[1], color=color)
            else:                                           # size = (x, y, z) box
                shape = Cuboid(scale=list(size), color=color)
            self.parts.append((shape, SE3(*offset)))

    def add_to(self, env):
        for shape, _ in self.parts:
            env.add(shape)

    def set_pose(self, T):
        """Place the tool with its grip-point frame at pose T (an SE3)."""
        for shape, offset in self.parts:
            shape.T = (T * offset).A
