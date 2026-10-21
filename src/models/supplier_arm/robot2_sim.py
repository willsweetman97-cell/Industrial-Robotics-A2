"""Swift simulation: Robot 2 returns a tool from the bench to the board.

Author: Jake Salamon

Run:   python robot2_sim.py   (all project files sit in the same folder).

Shows only a patch of the real setup: Robot 2 on the blue mark, a tool on the green
mark to its right, and a section of the board with two hung tools and one empty slot.
Robot 2 picks the tool and hangs it in the empty slot (the 'return' feature).

The arm is drawn as a simple capsule skeleton computed from the DH model, so it
shows up in Swift without meshes. Replace it with the realistic DHRobot3D version
once your meshes are ready.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import swift
from spatialgeometry import Cuboid, Cylinder, Sphere
from spatialmath import SE3

from robot2 import DH_TABLE, GRIPPER_LENGTH, Q_HOME, Robot2
import robot2_task as _task
from robot2_task import (DT, RACK_PLACE, TABLE_PICK, approach, hold, joint_move,
                         plan_joint_move, quintic, rmrc_line)

SIM_VERSION = "v5 - three parts, Enter between parts"
PLAYBACK_SPEED = 1.0  # 1.0 = planned speed; 0.7 = slower; 1.5 = faster. Total time stays honest.
RENDER_FPS = 25       # frames sent to Swift per second (lower this if you see connection errors)
import layout as L
from draw_utils import segment_pose
from tools import ToolModel


class ArmDrawing:
    """Capsule-skeleton drawing of a DH robot, updated from joint angles."""

    def __init__(self, env, robot, dh_table=None, gripper_length=None, color=(0.9, 0.5, 0.1), radius=0.045):
        self.robot = robot
        dh_table = DH_TABLE if dh_table is None else dh_table
        gripper_length = GRIPPER_LENGTH if gripper_length is None else gripper_length
        # link length between consecutive frame origins is sqrt(d^2 + a^2) (constant)
        self.lengths = [float(np.hypot(row[0], row[1])) for row in dh_table]
        self.links = [Cylinder(radius=radius, length=max(length, 1e-3), color=color)
                      if length > 1e-6 else None for length in self.lengths]
        self.tcp_link = Cylinder(radius=0.02, length=gripper_length, color=(0.2, 0.2, 0.2))
        self.joints = [Sphere(radius=radius * 1.2, color=(0.25, 0.25, 0.3))
                       for _ in range(len(dh_table) + 1)]
        for shape in [*self.links, self.tcp_link, *self.joints]:
            if shape is not None:
                env.add(shape)

    def update(self, q):
        T = self.robot.base.A
        points = [T[:3, 3].copy()]
        for i, link in enumerate(self.robot.links):
            T = T @ link.A(q[i]).A
            points.append(T[:3, 3].copy())
        for i, shape in enumerate(self.links):
            if shape is not None:
                shape.T = segment_pose(points[i], points[i + 1])
        for p, joint in zip(points, self.joints):
            joint.T = SE3(*p).A
        flange = points[-1]
        tcp = self.robot.fkine(q).t
        if np.linalg.norm(tcp - flange) > 1e-6:
            self.tcp_link.T = segment_pose(flange, tcp)


def build_scene(env):
    """Bench patch, marks, board patch and three tools. Returns the tool Robot 2 returns."""
    def box(size, pos, color):
        shape = Cuboid(scale=list(size), color=color)
        shape.T = SE3(*pos).A
        env.add(shape)
        return shape

    # Bench patch (top surface at z = 0) and the two marks
    box(L.BENCH_SIZE, (L.BENCH_CENTRE[0], L.BENCH_CENTRE[1], -L.BENCH_SIZE[2] / 2),
        (0.72, 0.72, 0.8))
    box((L.MARK_SIZES["blue"], L.MARK_SIZES["blue"], 0.002), (*L.BLUE_MARK, 0.001), (0.1, 0.3, 0.9))
    box((L.MARK_SIZES["green"], L.MARK_SIZES["green"], 0.002), (*L.GREEN_MARK, 0.001), (0.1, 0.8, 0.2))
    Cyl = Cylinder(radius=0.06, length=0.01, color=(0.3, 0.3, 0.35))  # Robot 2 base plate
    Cyl.T = SE3(*L.BLUE_MARK, 0.005).A
    env.add(Cyl)

    # Board patch on the wall, with an outline marking the empty return slot
    box((L.BOARD_SIZE[0], L.BOARD_SIZE[1], L.BOARD_SIZE[2]),
        (L.BOARD_CENTRE_X, L.WALL_Y + L.BOARD_SIZE[1] / 2, L.BOARD_CENTRE_Z), (0.93, 0.7, 0.35))
    box((0.08, 0.002, 0.26), (L.SLOT_X[L.RETURN_TOOL], L.WALL_Y - 0.001, L.HANG_Z - 0.12),
        (0.6, 0.45, 0.2))

    # Tools: two already hung on the board, one lying on the green mark
    for kind in ("hammer", "screwdriver"):
        tool = ToolModel(kind)
        tool.add_to(env)
        tool.set_pose(L.hung_pose(kind))
    returning = ToolModel(L.RETURN_TOOL)
    returning.add_to(env)
    returning.set_pose(SE3(*L.GREEN_MARK, L.GRASP_Z))
    return returning


def plan_sequence(robot):
    """Plan the task as three parts. Returns [(part name, plan)] or None if unreachable.

    Each plan is a list of (label, joint trajectory, tool_state) where tool_state is
    'free' (tool stays put), 'held' (tool follows the gripper), 'release' (tool eases
    into its slot) or 'placed' (tool stays in its slot).
    """
    q = Q_HOME.copy()
    parts = []

    def new_part(name):
        parts.append((name, []))

    def add(label, traj, tool_state):
        nonlocal q
        parts[-1][1].append((label, traj, tool_state))
        q = traj[-1]

    def go_joint(label, T, tool_state):
        traj, _ = plan_joint_move(robot, T, q)
        if traj is None:
            print(f"Pose '{label}' is NOT reachable - move the base/marks/board in "
                  "layout.py or fix the DH values, then retry.")
            return False
        add(label, traj, tool_state)
        return True

    new_part("Part 1: pick up the tool")
    if not go_joint("above tool", approach(TABLE_PICK), "free"):
        return None
    add("down to tool", rmrc_line(robot, TABLE_PICK, q), "free")
    add("grip", hold(q, 0.4), "free")
    add("lift tool", rmrc_line(robot, approach(TABLE_PICK), q), "held")
    add("lifted", hold(q, 0.3), "held")

    new_part("Part 2: place the tool on the shelf")
    if not go_joint("above empty slot", approach(RACK_PLACE), "held"):
        return None
    add("hang on board", rmrc_line(robot, RACK_PLACE, q), "held")
    placed = robot.fkine(q) * L.TCP_TO_TOOL
    error_mm = np.linalg.norm(placed.t - L.hung_pose(L.RETURN_TOOL).t) * 1000
    print(f"Tool placement error before final settle: {error_mm:.1f} mm")
    add("release", hold(q, 0.5), "release")

    new_part("Part 3: return to the original position")
    add("retreat", rmrc_line(robot, approach(RACK_PLACE), q), "placed")
    add("home", joint_move(q, Q_HOME), "placed")
    return parts


def flatten(plan):
    """One list of frames (q, tool_state, ease, label) at the planning rate DT."""
    frames = []
    for label, traj, tool_state in plan:
        ease = quintic(len(traj))
        frames += [(q, tool_state, float(ease[i]), label) for i, q in enumerate(traj)]
    return frames


def play(env, robot, arm, tool, frames, speed=PLAYBACK_SPEED):
    """Play frames from flatten() using the wall clock, skipping frames if Swift is slow.

    Returns False if the Swift connection failed part-way through.
    """
    slot_pose = L.hung_pose(L.RETURN_TOOL)
    stride = max(1, round(1 / (RENDER_FPS * DT)))  # only send every stride-th planned frame
    t0 = time.perf_counter()
    last, last_label = -1, None
    while last < len(frames) - 1:
        k = min(int((time.perf_counter() - t0) * speed / DT), len(frames) - 1)
        if k < len(frames) - 1:
            k -= k % stride
        if k == last:
            time.sleep(0.002)
            continue
        q, tool_state, ease, label = frames[k]
        if label != last_label:
            print("Stage:", label)
            last_label = label
        arm.update(q)
        if tool_state == "held":
            tool.set_pose(robot.fkine(q) * L.TCP_TO_TOOL)
        elif tool_state == "release":  # settle the tool exactly into its slot
            tool.set_pose((robot.fkine(q) * L.TCP_TO_TOOL).interp(slot_pose, ease))
        elif tool_state == "placed":
            tool.set_pose(slot_pose)
        try:
            env.step(DT)
        except Exception as exc:  # e.g. the browser tab was closed or reloaded
            print(f"Swift connection problem ({exc}). Keep the Swift tab open and visible, "
                  "then run again.")
            return False
        last = k
    return True


def main():
    print(f"robot2_sim {SIM_VERSION}")
    print(f"robot2_task {getattr(_task, 'VERSION', 'OLD VERSION - replace robot2_task.py')}")
    print(f"  loaded from {_task.__file__}")
    robot = Robot2(base=SE3(*L.BLUE_MARK, 0.01))  # standing on the blue mark
    parts = plan_sequence(robot)
    if parts is None:
        return
    total = sum(len(t) for _, plan in parts for _, t, _ in plan) * DT
    print(f"Planned duration: {total:.1f} s in {len(parts)} parts (playback speed x{PLAYBACK_SPEED})")

    env = swift.Swift()
    env.launch(realtime=False)  # we pace playback ourselves from the wall clock
    tool = build_scene(env)
    arm = ArmDrawing(env, robot)
    arm.update(Q_HOME)

    for name, plan in parts:
        frames = flatten(plan)
        input(f"\nPress Enter to start {name} ({len(frames) * DT / PLAYBACK_SPEED:.1f} s)... ")
        print(name)
        if not play(env, robot, arm, tool, frames):
            break
    else:
        print("\nAll three parts done. Close the Swift window or press Ctrl+C to exit.")
    env.hold()


if __name__ == "__main__":
    main()
