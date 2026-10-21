"""Swift simulation: Supply Bot (red square) feeds the timber and clears the output.

Author: Jake Salamon

Run:   python supply_bot_sim.py   (all project files sit in the same folder)

Uses the garage table layout from Garage.py (unchanged). On start it asks two questions:
  1. Does the real robot need a nut and bolt?  (if yes: which size nut, which size bolt)
  2. Does it need a screw?                     (if yes: which size)
Then it supplies the chosen parts to the timber, waits for the real robot to finish and carries the finished timber to the output area.

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import spatialgeometry as sg
import swift
from spatialgeometry import Cuboid
from spatialmath import SE3

import Garage

import robot2_task as _task
import supply_layout as S
import tools as _tools
from robot2_sim import ArmDrawing
from robot2_task import (DT, hold, joint_move, plan_joint_move, quintic, rmrc_line,
                         set_collision_limits)
from supply_bot import DH_TABLE, GRIPPER_LENGTH, Q_HOME, SupplyBot
from tools import ToolModel

SIM_VERSION = "v6 - boxes on"
PLAYBACK_SPEED = 1.0   # 1.0 = planned speed; 0.7 = slower; 1.5 = faster
RENDER_FPS = 25        # frames sent to Swift per second (lower if you see connection errors)
JOINT_SPEED = 0.9      # average joint speed for joint moves (rad/s)
CART_SPEED = 0.12      # average tool speed for straight RMRC moves (m/s)
APPROACH_H = 0.08      # hover height above an object (m)
BOX_APPROACH_H = 0.20  # hover height above the output box, clear of its walls (m)
ARM_COLOR = (0.15, 0.55, 0.75)


# ---------------------------------------------------------------- scene
def _timber_group_is_last(objs):
    """True if the last three objects Garage.py adds are the timber, cut line and drill mark."""
    try:
        return (isinstance(objs[-1], sg.Cylinder) and isinstance(objs[-2], sg.Cuboid)
                and isinstance(objs[-3], sg.Cuboid)
                and np.allclose(np.asarray(objs[-3].scale, dtype=float), Garage.WOOD_SIZE))
    except (IndexError, AttributeError, ValueError):
        return False


def launch_garage():
    """Open Swift with the Garage.py scene, minus the timber (Supply Bot moves its own).

    Garage.py is used exactly as it is: its scene objects are fetched with
    build_garage(launch=False), the timber, cut line and drill mark are left out, and the
    rest is added to a Swift window opened here.
    """
    objs, _ = Garage.build_garage(launch=False)
    if _timber_group_is_last(objs):
        objs = objs[:-3]
    else:
        print("Warning: could not find the timber in Garage.py's scene, so it was not removed "
              "- you will see the timber twice.")
    env = swift.Swift()
    try:
        env.launch(realtime=False, ground_opacity=0.0)   # we pace playback ourselves
    except TypeError:   # older Swift versions lack this option
        env.launch(realtime=False)
    for o in objs:
        env.add(o)
    return env


def build_scene(env, sizes, hole_pose=None):
    """Add what Supply Bot needs to the garage scene already in env.

    sizes is {'nut': 'small'/'large', 'bolt': 'small'/'large'} or None when no hardware is
    needed. Returns the movers: the objects Supply Bot carries, keyed 'nut', 'bolt', 'wood'.
    """
    def box(shape_size, pos, color):
        shape = Cuboid(scale=list(shape_size), color=color)
        shape.T = SE3(*pos).A
        env.add(shape)

    def prop(kind, pose):
        model = ToolModel(kind)
        model.add_to(env)
        model.set_pose(pose)
        return model

    movers = {}

    # Six red supply boxes (small/large nuts, bolts and screws)
    if S.SHOW_SUPPLY_BOXES:
        sx, sy = S.SUPPLY_BOX_SIZE
        floor, wall_h, wall_t = S.SUPPLY_BOX_FLOOR, S.SUPPLY_BOX_WALL_H, 0.006
        base_z = S.TABLE_H
        zc = floor + wall_h / 2
        red = (0.85, 0.12, 0.12)
        for (kind, size_name), (cx, cy) in S.SUPPLY_BOXES.items():
            half = S.SIZES[size_name][f"{kind}_half"]
            box((sx, sy, floor - base_z), (cx, cy, base_z + (floor - base_z) / 2), red)
            box((sx, wall_t, wall_h), (cx, cy + sy / 2, zc), red)
            box((sx, wall_t, wall_h), (cx, cy - sy / 2, zc), red)
            box((wall_t, sy, wall_h), (cx - sx / 2, cy, zc), red)
            box((wall_t, sy, wall_h), (cx + sx / 2, cy, zc), red)
            for dy in (0.035, -0.035):                       # spare hardware for show
                prop(f"{kind}_{size_name}", SE3(0, dy, 0) * S.start_pose(kind, size_name))

    # The chosen nut and bolt, waiting where their boxes would be
    if sizes is not None:
        for kind, size in S.hardware(sizes):
            movers[kind] = prop(f"{kind}_{size}", S.start_pose(kind, size))

    # The hole for the screw: already drilled in the stand-alone run, hidden until drilled in big_sim
    if hole_pose is not None:
        movers["hole"] = prop("hole", hole_pose)

    # Output box beside the red square - off for now
    if S.SHOW_OUTPUT_BOX:
        ox, oy = S.BOX_CENTRE
        ix, iy = S.BOX_INNER
        w, h = S.BOX_WALL, S.BOX_HEIGHT
        zc = S.BOX_FLOOR + h / 2
        box_color = (0.55, 0.4, 0.25)
        box((ix + 2 * w, iy + 2 * w, S.BOX_FLOOR - S.TABLE_H), (ox, oy, S.TABLE_H + (S.BOX_FLOOR - S.TABLE_H) / 2), box_color)
        box((ix + 2 * w, w, h), (ox, oy + iy / 2 + w / 2, zc), box_color)
        box((ix + 2 * w, w, h), (ox, oy - iy / 2 - w / 2, zc), box_color)
        box((w, iy, h), (ox - ix / 2 - w / 2, oy, zc), box_color)
        box((w, iy, h), (ox + ix / 2 + w / 2, oy, zc), box_color)

    # Two shelves on the pegboard with nut and bolt stock
    if S.SHOW_SHELVES:
        for name, z in S.SHELVES:
            shelf_y = S.TOOL_FACE_Y - S.SHELF_SIZE[1] / 2
            box(S.SHELF_SIZE, (S.SHELF_X, shelf_y, z), (0.4, 0.4, 0.45))
            bin_top = z + S.SHELF_SIZE[2] / 2 + 0.05
            box((0.16, 0.09, 0.05), (S.SHELF_X, shelf_y, z + S.SHELF_SIZE[2] / 2 + 0.025),
                (0.25, 0.45, 0.7) if name == "nuts" else (0.7, 0.45, 0.25))
            kind = "nut" if name == "nuts" else "bolt"
            for dx in (-0.04, 0.0, 0.04):
                prop(kind, SE3(S.SHELF_X + dx, shelf_y, bin_top + 0.006))

    # The timber: the garage scene is built without it, so Supply Bot can move it
    movers["wood"] = prop("wood", S.WOOD_START)
    return movers


# ---------------------------------------------------------------- planning
def tcp_for(obj_pose):
    """Gripper pose (pointing down) that holds an object lying flat at obj_pose."""
    return obj_pose * S.GRIP_TO_OBJ


def above(T, h):
    """Pose h metres before T along the gripper's approach axis."""
    return T * SE3(0, 0, -h)


class Planner:
    """Builds the task as parts, each a list of stage dicts."""

    def __init__(self, robot):
        self.robot = robot
        self.q = Q_HOME.copy()
        self.parts = []

    def new_part(self, name, wait_msg=None):
        self.parts.append((name, [], wait_msg))

    def add(self, label, traj, carry=None, attached=(), ease_to=None):
        self.parts[-1][1].append({"label": label, "traj": traj, "carry": carry,
                                  "attached": tuple(attached), "ease_to": ease_to})
        self.q = traj[-1]

    def joint_to(self, label, T, carry=None, attached=()):
        traj, _ = plan_joint_move(self.robot, T, self.q, speed=JOINT_SPEED)
        if traj is None:
            print(f"Pose '{label}' is NOT reachable - move things in supply_layout.py "
                  "or fix the DH values, then retry.")
            return False
        self.add(label, traj, carry, attached)
        return True

    def line_to(self, label, T, carry=None, attached=()):
        self.add(label, rmrc_line(self.robot, T, self.q, speed=CART_SPEED), carry, attached)

    def pause(self, label, seconds, carry=None, attached=(), ease_to=None):
        self.add(label, hold(self.q, seconds), carry, attached, ease_to)

    def move_object(self, name, start_pose, end_pose, attached=(), h_dst=APPROACH_H, grasp_rot=None):
        """Pick the object at start_pose, carry it and set it down at end_pose."""
        grasp_rot = S.GRIP_TO_OBJ if grasp_rot is None else grasp_rot
        grasp, drop = start_pose * grasp_rot, end_pose * grasp_rot
        if not self.joint_to(f"above {name}", above(grasp, APPROACH_H)):
            return False
        self.line_to(f"down to {name}", grasp)
        self.pause(f"grip {name}", 0.3)
        self.line_to(f"lift {name}", above(grasp, APPROACH_H), name, attached)
        if not self.joint_to(f"carry {name}", above(drop, h_dst), name, attached):
            return False
        self.line_to(f"lower {name}", drop, name, attached)
        placed = self.robot.fkine(self.q) * grasp_rot.inv()
        error_mm = np.linalg.norm(placed.t - end_pose.t) * 1000
        print(f"  {name}: placement error before final settle {error_mm:.1f} mm")
        self.pause(f"release {name}", 0.3, name, attached, ease_to=end_pose)
        self.line_to(f"clear of {name}", above(drop, h_dst))
        return True


def plan_task(robot, sizes):
    """Plan the task. sizes = {'nut': size, 'bolt': size} to supply hardware first, or None.

    Returns [(part name, stages, wait message)] or None if something is unreachable.
    """
    p = Planner(robot)
    attached = ()
    if sizes is not None:
        p.new_part(f"Supply the {S.describe(sizes)}")
        for kind, size in S.hardware(sizes):
            if not p.move_object(kind, S.start_pose(kind, size), S.on_wood(kind, size), grasp_rot=S.grasp_rot(kind, size)):
                return None
        attached = tuple(kind for kind, _ in S.hardware(sizes)) + (("hole",) if "screw" in sizes else ())
    p.new_part("Collect the finished piece",
               "Waiting for the real robot to finish. Press Enter when it is done")
    if not p.move_object("wood", S.WOOD_START, S.WOOD_IN_BOX, attached, BOX_APPROACH_H):
        return None
    p.add("home", joint_move(p.q, Q_HOME, speed=JOINT_SPEED))
    return p.parts


def initial_state(sizes):
    """Where the movable objects start."""
    state = {"wood": S.WOOD_START}
    if sizes is not None:
        for kind, size in S.hardware(sizes):
            state[kind] = S.start_pose(kind, size)
        if "screw" in sizes:
            state["hole"] = S.hole_pose()
    return state


def flatten_parts(parts, robot, state):
    """Turn stages into frames [(q, {object: 4x4 pose}, label)] per part, tracking objects.

    Returns [(part name, frames, wait message)].
    """
    result = []
    for part_name, stages, wait_msg in parts:
        frames = []
        for st in stages:
            traj, carry = st["traj"], st["carry"]
            ease = quintic(len(traj))
            rel = {}
            if carry:
                rel = {a: state[carry].inv() * state[a] for a in st["attached"]}
            for i, q in enumerate(traj):
                if carry:
                    held = robot.fkine(q) * S.GRIP_TO_OBJ
                    obj = held.interp(st["ease_to"], float(ease[i])) if st["ease_to"] is not None else held
                    state[carry] = obj
                    for a, offset in rel.items():
                        state[a] = obj * offset
                frames.append((q, {k: v.A.copy() for k, v in state.items()}, st["label"]))
        result.append((part_name, frames, wait_msg))
    return result


# ---------------------------------------------------------------- playback
def play(env, arm, movers, frames, rendered, speed=PLAYBACK_SPEED):
    """Play frames using the wall clock, skipping frames if Swift is slow.

    rendered remembers each object's last drawn pose. Returns False if Swift disconnects.
    """
    stride = max(1, round(1 / (RENDER_FPS * DT)))
    t0 = time.perf_counter()
    last, last_label = -1, None
    while last < len(frames) - 1:
        k = min(int((time.perf_counter() - t0) * speed / DT), len(frames) - 1)
        if k < len(frames) - 1:
            k -= k % stride
        if k == last:
            time.sleep(0.002)
            continue
        q, poses, label = frames[k]
        if label != last_label:
            print("  -", label)
            last_label = label
        arm.update(q)
        for name, M in poses.items():
            prev = rendered.get(name)
            if prev is None or not np.array_equal(prev, M):
                movers[name].set_pose(SE3(M, check=False))
                rendered[name] = M
        try:
            env.step(DT)
        except Exception as exc:  # e.g. the browser tab was closed or reloaded
            print(f"Swift connection problem ({exc}). Keep the Swift tab open and visible, "
                  "then run again.")
            return False
        last = k
    return True


def check_files():
    """Make sure the shared files are the latest versions; name any that are not."""
    import inspect
    stale = []
    if not getattr(_task, "VERSION", "v0").startswith(("v7", "v8", "v9")):
        stale.append("robot2_task.py")
    if "screw_large" not in _tools.TOOL_PARTS or not hasattr(S, "hardware"):
        stale.append("tools.py")
    if "dh_table" not in inspect.signature(ArmDrawing.__init__).parameters:
        stale.append("robot2_sim.py")
    if stale:
        print("Out-of-date files in this folder - download the latest and replace: "
              + ", ".join(stale))
    return not stale


def ask_choice(prompt, options, default=None):
    """Keep asking until the answer is one of options (case-insensitive).

    If default is given, pressing Enter on its own picks it.
    """
    while True:
        answer = input(prompt).strip().lower()
        if answer == "" and default is not None:
            return default
        if answer in options:
            return answer
        print(f"  Please answer one of: {', '.join(options)}")


def ask_hardware():
    """Nut / bolt and screw questions shared with big_sim. Returns a sizes dict or None."""
    sizes = {}
    options = list(S.SIZES)
    if ask_choice("1) Does the real robot need a nut and bolt? (y/n): ", ["y", "yes", "n", "no"]) in ("y", "yes"):
        nut = ask_choice(f"   Which size NUT? ({' / '.join(options)}): ", options)
        bolt = ask_choice(f"   Which size BOLT? ({' / '.join(options)}, Enter = same as nut): ", options, default=nut)
        sizes.update(nut=nut, bolt=bolt)
    if ask_choice("2) Does the real robot need a screw? (y/n): ", ["y", "yes", "n", "no"]) in ("y", "yes"):
        sizes["screw"] = ask_choice(f"   Which size SCREW? ({' / '.join(options)}): ", options)
    if not sizes:
        print("  No hardware needed. Supply Bot will wait for the real robot, then take the piece.")
        return None
    print(f"  Supplying the {S.describe(sizes)}.")
    return sizes


def ask_job():
    """The start-up questions. Returns {'nut':.., 'bolt':.., 'screw':..} (only what is needed), or None."""
    print("\n--- Supply Bot: job setup ---")
    return ask_hardware()


def main():
    print(f"supply_bot_sim {SIM_VERSION}")
    print(f"robot2_task {getattr(_task, 'VERSION', 'OLD VERSION - replace robot2_task.py')}")
    if not check_files():
        return
    try:
        sizes = ask_job()
    except EOFError:
        return
    robot = SupplyBot(base=S.BASE_POSE)
    set_collision_limits(floor_z=S.FLOOR_LIMIT_Z, wall_y=S.WALL_LIMIT_Y)
    parts = plan_task(robot, sizes)
    if parts is None:
        return
    parts = flatten_parts(parts, robot, initial_state(sizes))
    total = sum(len(frames) for _, frames, _ in parts) * DT
    print(f"Planned duration: {total:.1f} s in {len(parts)} part(s) (playback speed x{PLAYBACK_SPEED})")

    env = launch_garage()  # Garage.py scene without the timber
    movers = build_scene(env, sizes, S.hole_pose() if sizes and 'screw' in sizes else None)
    arm = ArmDrawing(env, robot, DH_TABLE, GRIPPER_LENGTH, ARM_COLOR)
    arm.update(Q_HOME)

    rendered = {}
    for name, frames, wait_msg in parts:
        duration = len(frames) * DT / PLAYBACK_SPEED
        input(f"\n{wait_msg or 'Press Enter to start'}: {name} ({duration:.1f} s)... ")
        print(name)
        if not play(env, arm, movers, frames, rendered):
            break
    else:
        print("\nAll done. Close the Swift window or press Ctrl+C to exit.")
    env.hold()


if __name__ == "__main__":
    main()
