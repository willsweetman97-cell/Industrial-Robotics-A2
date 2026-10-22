"""One big simulation: ToolBot, TransferBot, the UR3e and SupplyBot working a tool pipeline.

Author: [Your name]

Run:   python big_sim.py   (all project files sit in the same folder)

Uses the garage table from Garage.py (unchanged) plus one extra green "used tools" mark
(drawn here) between ToolBot and TransferBot. It asks what the real robot should do
(sawing / drilling and screwing / both) and whether it needs a nut and bolt (and a screw), then runs the
robots, and the whole job then runs in one go (no Enter presses). For a job with tools [A, B]:
  1. ToolBot (blue)      takes A off the pegboard and puts it on the first green mark
  2. TransferBot (yellow) carries A to the second green mark, where the real robot uses it
  3. The UR3e (green square) takes A to the timber and saws / drills / screws, then puts it
     back on the second green mark -> meanwhile ToolBot gets B ready on the first green mark
  4. TransferBot puts the used A on the "used tools" mark
  5. TransferBot carries B to the real robot, while ToolBot hangs A back on the pegboard
     (the two run together only if their arms stay clear of each other)
  ...and so on for the next tool. SupplyBot (red) supplies the nut and bolt (if needed) while the first
  tool is being fetched. The drill leaves a hole in the timber; once it is drilled SupplyBot stands the
  screw in the hole, and the UR3e's screwdriver turns it in. At the end SupplyBot carries the finished
  piece away.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import spatialgeometry as sg
import swift
from spatialmath import SE3

import Garage
import ChainLayout as C
import robot2_task as _task
import supply_layout as S
import tools as _tools
import visual_servo as VS
from robot2_sim import ArmDrawing
from robot2_task import (DT, _ik_candidates, _segment_distance, arm_collision, arm_points, hold,
                         joint_move, path_collision_free, plan_joint_move, quintic, rmrc_line,
                         set_collision_limits)
from Ur3e import DH_TABLE as UR_DH, GRIPPER_LENGTH as UR_GRIP, LINK_RADIUS as UR_RADIUS, UR3e
from supply_bot import DH_TABLE as SUPPLY_DH, GRIPPER_LENGTH as SUPPLY_GRIP, Q_HOME as SUPPLY_HOME, SupplyBot
from supply_bot_sim import _timber_group_is_last, build_scene as build_supply_scene
from ToolBot import SupplierArm
from tools import ToolModel
from TransferBot import GRIP_CENTRE, OPEN_WIDTH, Gripper, PartMover

SIM_VERSION = "v7 - drilled hole; SupplyBot stands the screw in it and the UR3e drives it in"
PLAYBACK_SPEED = 1.0   # 1.0 = planned speed; 0.7 = slower; 1.5 = faster
RENDER_FPS = 25        # frames sent to Swift per second (lower if you see connection errors)
JOINT_SPEED = 0.9      # average joint speed for joint moves (rad/s)
CART_SPEED = 0.12      # average tool speed for straight RMRC moves (m/s)
HOVER = 0.10           # ToolBot hover height above a tool pick/place (m)
UR_HOVER = 0.12        # UR3e hover height above the tool on the hand-over square (m)
TRANSFER_HOVER = 0.15  # TransferBot hovers higher so a carried tool clears one lying on the used mark
SMALL_HOVER = 0.08     # hover height above a nut, bolt or the timber (m)
BOX_HOVER = 0.20       # hover height above the output spot (m)
CLOSED_WIDTH = 0.03    # TransferBot's finger gap when holding a tool (m)
POSITION_TOL = 0.005   # a straight-line move must end within 5 mm of its target (m)
IDLE_MARGIN = 0.09     # SupplyBot keeps this far from the UR3e standing at its ready pose (m)
PARALLEL_MARGIN = 0.14 # two arms running together must stay at least this far apart (m)
# The saw hangs at the far end of the pegboard. With tool_bot.py's shoulder (joint 2) limited to +-90
# degrees ToolBot misses it by ~4 cm; +-150 degrees reaches it. This widens the limit in the simulation
# only (tool_bot.py is untouched). Set to None to use tool_bot.py exactly as written.
TOOLBOT_SHOULDER_LIMIT_DEG = 150
TASK_VERB = {"saw": "sawing", "drill": "drilling", "screwdriver": "screwing",
             "hammer": "hammering", "wrench": "tightening"}

# Ready poses (deg); the first collision-free candidate is used
TOOLBOT_HOMES = [[0, -60, 90, 0, -30, 0], [0, -60, 30, 0, -30, 0], [0, -45, 60, 0, 0, 0]]
UR3E_HOMES = [[0, -45, 100, -145, -90, 0], [0, -90, 90, -90, -90, 0], [0, -110, 110, -90, -90, 0], [0, -70, 70, -90, -90, 0]]
# SupplyBot's ready pose is swung 40 degrees toward the table front so its raised gripper is not above the UR3e
SUPPLY_HOMES = [[40, -80, 10, 0, 30, 0]]
TRANSFER_HOMES = [[0, 60, 0, 0, 0, 0], [0, 45, 30, 0, 0, 0], [0, 30, 60, 0, 30, 0]]
TOOLBOT_COLOR = (0.15, 0.45, 0.9)
SUPPLY_COLOR = (0.9, 0.35, 0.3)
UR3E_COLOR = (0.55, 0.68, 0.82)       # UR blue-grey
TRANSFER_COLOR = (0.95, 0.75, 0.1)   # yellow, to match its square
TRANSFER_NATIVE_VISUALS = False      # True = use PartMover's own link visuals instead of the skeleton


# ---------------------------------------------------------------- planning
def above(T, h):
    """Pose h metres before T along the gripper's approach axis."""
    return T * SE3(0, 0, -h)


class Planner:
    """Builds one robot's task as a list of stage dicts."""

    def __init__(self, robot, q_home, gripper=False, avoid=None):
        """avoid: skeletons (lists of points, see arm_points) of arms standing still that this arm must keep clear of."""
        self.robot, self.gripper = robot, gripper
        self.avoid = avoid or []
        self.servo_logs = []
        self.q = np.array(q_home, dtype=float)
        self.home = self.q.copy()
        self.stages = []

    def add(self, label, traj, **extra):
        stage = {"label": label, "traj": traj, "carry": None, "attached": (), "ease_to": None,
                 "tcp_obj": None, "grip": None}
        stage.update(extra)
        self.stages.append(stage)
        self.q = traj[-1]

    def joint_to(self, label, T, **extra):
        traj, _ = plan_joint_move(self.robot, T, self.q, speed=JOINT_SPEED)
        if traj is None:
            print(f"    '{label}' is NOT reachable")
            return False
        self.add(label, traj, **extra)
        return True

    def line_to(self, label, T, **extra):
        traj = rmrc_line(self.robot, T, self.q, speed=CART_SPEED)
        miss = float(np.linalg.norm(T.t - self.robot.fkine(traj[-1]).t))
        if miss > POSITION_TOL:
            print(f"    '{label}': the straight-line move stops {miss * 1000:.0f} mm short of its target")
            return False
        self.add(label, traj, **extra)
        return True

    def _clear_of_idle(self, traj, stride=4):
        """True if the arm keeps IDLE_MARGIN away from every arm it has to avoid along the whole trajectory."""
        for q in traj[::stride]:
            mine = arm_points(self.robot, q)
            for idle in self.avoid:
                for i in range(len(mine) - 1):
                    for j in range(len(idle) - 1):
                        if _segment_distance(mine[i], mine[i + 1], idle[j], idle[j + 1]) < IDLE_MARGIN:
                            return False
        return True

    def descend(self, joint_label, T_hover, line_label, T_end, joint_extra=None, line_extra=None, tries=8):
        """Joint move to T_hover, then a straight line to T_end.

        Tries the arm's different ways of reaching T_hover (elbow up/down, wrist flipped) and keeps the
        first one from which the straight line works, because a bad choice can run out of joint range.
        If the direct joint move would sweep the arm into the wall or itself, it first turns the base
        with the arm tucked, or goes to a waypoint further back along the approach direction.
        """
        joint_extra, line_extra = joint_extra or {}, line_extra or {}
        robot = self.robot
        tried = 0
        for q_goal in _ik_candidates(robot, T_hover, self.q):
            if arm_collision(robot, q_goal):
                continue
            line = rmrc_line(robot, T_end, q_goal, speed=CART_SPEED, quiet=True)      # the final straight move
            miss = float(np.linalg.norm(T_end.t - robot.fkine(line[-1]).t))
            if miss > POSITION_TOL or not path_collision_free(robot, line) or not self._clear_of_idle(line):
                continue                                                          # this arm configuration cannot finish it
            tried += 1
            if tried > tries:
                break
            for route in self._routes(T_hover, q_goal):
                trajs, q_from = [], self.q
                for q_to in route:
                    trajs.append(joint_move(q_from, q_to, speed=JOINT_SPEED))
                    q_from = q_to
                if all(path_collision_free(robot, t) and self._clear_of_idle(t) for t in trajs):
                    for n, traj in enumerate(trajs):
                        self.add(joint_label if n == len(trajs) - 1 else f"{joint_label} (first leg)",
                                 traj, **joint_extra)
                    self.add(line_label, line, **line_extra)
                    return True
        print(f"    '{joint_label}' / '{line_label}': no collision-free way to reach it with a straight final move")
        return False

    def _routes(self, T_hover, q_goal):
        """Ways to get from the current pose to q_goal, simplest first (each a list of waypoints)."""
        robot = self.robot
        yield [q_goal]
        turn_first = self.q.copy()
        turn_first[0] = q_goal[0]                                  # swing the base round with the arm tucked
        yield [turn_first, q_goal]
        for back in (0.15, 0.30):                                  # a waypoint further back along the approach
            vias = [v for v in _ik_candidates(robot, T_hover * SE3(0, 0, -back), q_goal, tries=10)
                    if not arm_collision(robot, v)]
            if vias:
                yield [min(vias, key=lambda v: float(np.linalg.norm(v - q_goal))), q_goal]
        lo, hi = robot.qlim                                        # last resort: random waypoints
        rng = np.random.default_rng(7)
        samples = []
        for _ in range(400):
            v = rng.uniform(0.7 * lo, 0.7 * hi)
            if not arm_collision(robot, v):
                samples.append(v)
        samples.sort(key=lambda v: float(np.linalg.norm(v - self.q) + np.linalg.norm(v - q_goal)))
        for v in samples[:60]:
            yield [v, q_goal]

    def pause(self, label, seconds, **extra):
        self.add(label, hold(self.q, seconds), **extra)

    def go_home(self):
        self.add("home", joint_move(self.q, self.home, speed=JOINT_SPEED))

    def _servo_to(self, hover_dst, points, carry):
        """Visual servoing above the target: start from a deliberately imperfect hover, then let the camera fix it.

        The arm is at the ideal hover pose; what the camera sees from there is the goal image. The arm is then
        moved off by a few millimetres and a few degrees (a simulated positioning error) and IBVS steers it back
        until the goal image is reached.
        """
        goal_xy, _ = VS.project(self.robot.fkine(self.q), points)
        wrong = VS.perturb(self.robot.fkine(self.q))
        if not self.joint_to("approach the hole (simulated 1 cm positioning error)", wrong, **carry):
            return False
        traj, log = VS.ibvs(self.robot, self.q, points, goal_xy)
        miss = float(log["error_px"][-1])
        print(f"    visual servoing: image error {log['error_px'][0]:.1f} px -> {miss:.2f} px "
              f"in {len(traj) * DT:.1f} s")
        if miss > 2.0:
            print("    visual servoing did not converge")
            return False
        self.servo_logs.append(log)
        self.add("visual servoing: centre the screw over the hole", traj, **carry)
        return True

    def move_object(self, name, start, end, grasp_rot, hover_src, hover_dst, attached=(), servo_points=None):
        """Pick the object at pose `start`, carry it, set it down at pose `end`.

        grasp_rot: gripper orientation relative to the object frame (TCP = object * grasp_rot).
        servo_points: world points the gripper camera sees at the drop-off (e.g. a hole's rim); if given, the
        last part of the approach is done by visual servoing before the object is lowered.
        """
        grasp, drop, tcp_obj = start * grasp_rot, end * grasp_rot, grasp_rot.inv()
        closed, opened = (CLOSED_WIDTH, OPEN_WIDTH) if self.gripper else (None, None)
        carry = {"carry": name, "attached": tuple(attached), "tcp_obj": tcp_obj}
        if not self.descend(f"above {name}", above(grasp, hover_src), f"down to {name}", grasp,
                            joint_extra={"grip": opened}):
            return False
        self.pause(f"grip {name}", 0.4, grip=closed)
        if not self.line_to(f"lift {name}", above(grasp, hover_src), **carry):
            return False
        if not self.descend(f"carry {name}", above(drop, hover_dst), f"lower {name}", drop,
                            joint_extra=carry, line_extra=carry):
            return False
        if servo_points is not None:
            self.stages.pop()                          # take the open-loop "lower" back out ...
            self.q = self.stages[-1]["traj"][-1]
            if not self._servo_to(hover_dst, servo_points, carry):           # ... servo over the target ...
                return False
            if not self.line_to(f"lower {name}", drop, **carry):             # ... then lower from the servoed pose
                return False
        placed = self.robot.fkine(self.q) * tcp_obj
        print(f"    {name}: placement error before final settle "
              f"{np.linalg.norm(placed.t - end.t) * 1000:.1f} mm")
        self.pause(f"release {name}", 0.4, ease_to=end, grip=opened, **carry)
        return self.line_to(f"clear of {name}", above(drop, hover_dst))


# Tool actions: (robot, where the tool starts, where it ends, description)
TOOL_ACTIONS = {
    "fetch": ("ToolBot", lambda t: (C.tool_start(t), C.tool_on_square(t, 1)),
              "take the {t} off the pegboard"),
    "deliver": ("TransferBot", lambda t: (C.tool_on_square(t, 1), C.tool_on_square(t, 2)),
                "carry the {t} to the real robot"),
    "retire": ("TransferBot", lambda t: (C.tool_on_square(t, 2), C.tool_on_square(t, "used")),
               "put the used {t} on the used-tools mark"),
    "return": ("ToolBot", lambda t: (C.tool_on_square(t, "used"), C.tool_start(t)),
               "hang the used {t} back on the pegboard"),
    "use": ("UR3e", None, "use the {t} on the timber"),
}


def plan_real_robot(robot, tool, screw=None):
    """UR3e: take the tool from the second hand-over square, saw / drill / screw the timber, put it back."""
    p = Planner(robot, robot.home_q)
    verb = TASK_VERB.get(tool, "working")
    pick = C.tool_on_square(tool, 2)
    grasp = pick * C.UR_GRASP
    carry = {"carry": tool, "attached": (), "tcp_obj": C.UR_GRASP.inv()}
    path = [T * C.UR_GRASP for T in C.real_robot_path(tool, screw)]
    motion = C.screw_motion(screw) if (tool == "screwdriver" and screw) else None
    if not p.descend(f"above {tool}", above(grasp, UR_HOVER), f"down to {tool}", grasp):
        return None
    p.pause(f"grip {tool}", 0.4)
    if not p.line_to(f"lift {tool}", above(grasp, UR_HOVER), **carry):
        return None
    if not p.descend(f"take the {tool} to the timber", path[0], f"start {verb}", path[1],
                     joint_extra=carry, line_extra=carry):
        return None
    for n, T in enumerate(path[2:], 1):
        extra = dict(carry)
        if tool == "drill" and n == 2:             # the bit comes back up out of its first hole: show the hole
            extra["drive"] = ("hole", S.hole_pose(), S.hole_pose())
        if motion and n >= 2:                      # each screwing stroke turns the screw and sinks it in
            extra["drive"] = ("screw",) + motion[n - 2]
        if not p.line_to(f"{verb} {n}", T, **extra):
            return None
    if not p.line_to("lift off the timber", path[0], **carry):
        return None
    if not p.descend(f"bring the {tool} back", above(grasp, UR_HOVER), f"lower {tool}", grasp,
                     joint_extra=carry, line_extra=carry):
        return None
    p.pause(f"release {tool}", 0.4, ease_to=pick, **carry)
    if not p.line_to(f"clear of {tool}", above(grasp, UR_HOVER)):
        return None
    p.go_home()
    return p.stages


def plan_tool_action(robot, kind, tool, screw=None):
    """Stages for one tool action. The robot starts and ends at home, so actions chain."""
    who, poses, text = TOOL_ACTIONS[kind]
    if who == "UR3e":
        print(f"  {who}: {text.format(t=tool)}")
        set_collision_limits(floor_z=C.TABLE_H + 0.03, wall_y=10.0)
        return plan_real_robot(robot, tool, screw)
    if who == "ToolBot":
        set_collision_limits(floor_z=C.TABLE_H + 0.03, wall_y=Garage.TOOL_FACE_Y - 0.06)
        hover = HOVER
    else:
        set_collision_limits(floor_z=C.TABLE_H + 0.03, wall_y=10.0)
        hover = TRANSFER_HOVER
    start, end = poses(tool)
    p = Planner(robot, robot.home_q, gripper=(who == "TransferBot"))
    print(f"  {who}: {text.format(t=tool)}")
    if not p.move_object(tool, start, end, C.GRASP_ROT, hover, hover):
        return None
    p.go_home()
    return p.stages


def plan_job_tools(robots, tools, screw=None):
    """Plan every tool action the job needs. Returns ({(kind, tool): stages}, None) or (None, error text)."""
    plans = {}
    for kind in ("fetch", "return", "use", "deliver", "retire"):  # ToolBot first: it is the fussiest
        for tool in tools:
            stages = plan_tool_action(robots[TOOL_ACTIONS[kind][0]], kind, tool, screw)
            if stages is None:
                return None, f"{TOOL_ACTIONS[kind][0]} cannot {TOOL_ACTIONS[kind][2].format(t=tool)}"
            plans[(kind, tool)] = stages
    return plans, None


def plan_supplybot(robot, sizes, idle_arms=()):
    """SupplyBot's three jobs, each starting and ending at its ready pose.

    "hardware": the nut and bolt to the timber (during the first tool fetch);
    "screw":    the screw from its box into the hole the real robot has drilled;
    "collect":  the finished timber (with everything on or in it) to the output box.
    If the job drills a hole it travels with the timber too.
    idle_arms: robots standing at their ready pose while SupplyBot works (the UR3e); SupplyBot keeps clear of
    them if it can find a way to.
    Returns {name: (step title, stages, wait message) or None}, or None if something is unreachable.
    """
    avoid = [arm_points(r, r.home_q) for r in idle_arms]
    steps = _plan_supplybot(robot, sizes, avoid)
    if steps is None and avoid:
        print("  (SupplyBot cannot keep clear of the UR3e on every move - planning again without that check.)")
        steps = _plan_supplybot(robot, sizes, [])
    return steps


def _plan_supplybot(robot, sizes, avoid):
    set_collision_limits(floor_z=S.FLOOR_LIMIT_Z, wall_y=S.WALL_LIMIT_Y)
    steps = {"hardware": None, "screw": None, "collect": None}
    attached = ()
    if sizes is not None:
        loose = S.loose_hardware(sizes)
        if loose:
            p = Planner(robot, robot.home_q, avoid=avoid)
            for kind, size in loose:
                if not p.move_object(kind, S.start_pose(kind, size), S.on_wood(kind, size),
                                     S.grasp_rot(kind, size), SMALL_HOVER, SMALL_HOVER):
                    return None
            p.go_home()
            steps["hardware"] = (f"supply the {S.describe({k: sz for k, sz in loose})}", p.stages, None)
        if "screw" in sizes:
            size = sizes["screw"]
            p = Planner(robot, robot.home_q, avoid=avoid)
            if not p.move_object("screw", S.start_pose("screw", size), S.on_wood("screw", size),
                                 S.grasp_rot("screw", size), SMALL_HOVER, SMALL_HOVER,
                                 servo_points=VS.rim_points(S.hole_pose().t)):
                return None
            if p.servo_logs and VS.save_plot(p.servo_logs, str(Path(__file__).resolve().parent / "visual_servoing.png")):
                print("    (feature paths and error saved to visual_servoing.png)")
            p.go_home()
            steps["screw"] = (f"put the {size} screw in the drilled hole", p.stages, None)
        attached = tuple(kind for kind, _ in S.hardware(sizes))
    attached += ("hole",)                      # only moved if the job has drilled it (see flatten)
    p = Planner(robot, robot.home_q, avoid=avoid)
    if not p.move_object("wood", S.WOOD_START, S.WOOD_IN_BOX, S.GRIP_TO_OBJ, SMALL_HOVER, BOX_HOVER, attached):
        return None
    p.go_home()
    steps["collect"] = ("collect the finished piece", p.stages, None)
    return steps


def flatten(stages, robot, state, width):
    """Turn stages into frames [(q, {object: 4x4 pose}, label, finger width)].

    state tracks every movable object's pose. A frame only lists the objects the robot is
    carrying at that moment (so frames of two robots can be played together); the last frame
    lists every object the task touched, so nothing is left short of its final pose.
    width is the gripper's current finger gap (None for robots without a gripper).
    Returns (frames, final finger width).
    """
    frames, touched = [], []
    for st in stages:
        traj, carry = st["traj"], st["carry"]
        ease = quintic(len(traj))
        rel = {}
        if carry:
            rel = {a: state[carry].inv() * state[a] for a in st["attached"] if a in state}
            touched += [n for n in (carry, *rel) if n not in touched]
        start_width, target = width, st["grip"]
        drive = st.get("drive")                    # (object, pose before, pose after): an object moved by the tool, not carried
        if drive and drive[0] not in touched:
            touched.append(drive[0])
        for i, q in enumerate(traj):
            poses = {}
            if drive:
                state[drive[0]] = drive[1].interp(drive[2], float(ease[i]))
                poses[drive[0]] = state[drive[0]].A.copy()
            if carry:
                held = robot.fkine(q) * st["tcp_obj"]
                obj = held.interp(st["ease_to"], float(ease[i])) if st["ease_to"] is not None else held
                state[carry] = obj
                poses[carry] = obj.A.copy()
                for name, offset in rel.items():
                    state[name] = obj * offset
                    poses[name] = state[name].A.copy()
            w = None
            if start_width is not None:
                w = start_width if target is None else start_width + (target - start_width) * float(ease[i])
            frames.append((q, poses, st["label"], w))
        if target is not None and width is not None:
            width = target
    if frames:
        q, _, label, w = frames[-1]
        frames[-1] = (q, {n: state[n].A.copy() for n in touched}, label, w)
    return frames, width


def find_stagger(robots, first, second, stride=4, max_delay=20.0):
    """Shortest delay (s) before `second` can start so that it and `first` never come within margin.

    first and second are (robot name, frames). Returns None if no delay up to max_delay works.
    """
    (name_a, frames_a), (name_b, frames_b) = first, second
    pts_a = [arm_points(robots[name_a], f[0]) for f in frames_a[::stride]]
    pts_b = [arm_points(robots[name_b], f[0]) for f in frames_b[::stride]]

    def clear(pa, pb):
        for i in range(len(pa) - 1):
            for j in range(len(pb) - 1):
                if _segment_distance(pa[i], pa[i + 1], pb[j], pb[j + 1]) < PARALLEL_MARGIN:
                    return False
        return True

    for shift in range(0, int(max_delay / (stride * DT)) + 1):          # second starts `shift` samples late
        ok = True
        for k in range(len(pts_a) + 0):
            kb = k - shift
            if kb < 0:
                kb = 0                                                   # second still waiting at its home pose
            if not clear(pts_a[k], pts_b[min(kb, len(pts_b) - 1)]):
                ok = False
                break
        if ok:
            return shift * stride * DT
    return None


def delayed(frames, seconds):
    """Frames with the robot standing still for `seconds` first."""
    n = int(round(seconds / DT))
    if n <= 0:
        return frames
    q, _, _, w = frames[0]
    return [(q, {}, "waiting for the other arm to clear", w)] * n + frames


# ---------------------------------------------------------------- scene and actors
def launch_garage(tools):
    """Open Swift with the Garage.py scene minus the timber and the job's tools, plus the used-tools mark.

    Garage.py is used as it is: its objects come from build_garage(launch=False). The timber
    (cut line and drill mark) and the pegboard shapes of the job's tools are left out because the
    robots move their own copies. The extra green "used tools" mark is added here.
    """
    objs, _ = Garage.build_garage(launch=False)
    if _timber_group_is_last(objs):
        objs = objs[:-3]
    else:
        print("Warning: could not find the timber in Garage.py's scene - you will see it twice.")
    kinds = [sg.Cuboid] * 6 + [sg.Cylinder] * 2 + [sg.Cuboid] * 2     # saw, hammer, drill, screwdriver, wrench
    block = objs[-11:-1]                                              # the tools sit just before the jig
    if len(block) == 10 and all(isinstance(o, k) for o, k in zip(block, kinds)):
        base = len(objs) - 11
        for index in sorted((C.TOOL_NAMES.index(t) for t in tools), reverse=True):
            del objs[base + 2 * index:base + 2 * index + 2]
    else:
        print("Warning: could not find the tools in Garage.py's scene - you will see them twice.")
    objs.append(Garage.box((C.USED_SIZE, C.USED_SIZE, 0.004), (C.USED_X, C.USED_Y, C.TABLE_H + 0.002),
                           (0.1, 0.65, 0.25, 1)))                     # same green as the other marks
    env = swift.Swift()
    try:
        env.launch(realtime=False, ground_opacity=0.0)   # we pace playback ourselves
    except TypeError:   # older Swift versions lack this option
        env.launch(realtime=False)
    for o in objs:
        env.add(o)
    return env


class SkeletonActor:
    """A DH arm drawn as a capsule skeleton (ToolBot and SupplyBot have no meshes yet)."""

    def __init__(self, env, robot, dh_rows, grip_length, color, radius=0.045):
        self.arm = ArmDrawing(env, robot, dh_rows, grip_length, color, radius)

    def show(self, q, width=None):
        self.arm.update(q)


class PartMoverActor:
    """TransferBot: a skeleton drawing (always visible) or its own link visuals, plus the gripper."""

    def __init__(self, env, robot):
        self.robot, self.gripper = robot, Gripper()
        self.skeleton = None
        if TRANSFER_NATIVE_VISUALS:
            env.add(robot)
        else:
            rows = [(link.d, link.a) for link in robot.links]
            self.skeleton = ArmDrawing(env, robot, rows, GRIP_CENTRE, TRANSFER_COLOR)
        for shape in self.gripper.shapes():
            env.add(shape)

    def show(self, q, width=None):
        self.robot.q = q
        if self.skeleton is not None:
            self.skeleton.update(q)
        self.gripper.update(self.robot.fkine(q) * SE3(0, 0, -GRIP_CENTRE), width)


def play(env, actors, movers, tracks, rendered, speed=PLAYBACK_SPEED):
    """Play one or more robots' frames together using the wall clock, skipping frames if Swift is slow.

    tracks is a list of (robot name, frames). rendered remembers each object's last drawn pose.
    Returns False if Swift disconnects.
    """
    stride = max(1, round(1 / (RENDER_FPS * DT)))
    total = max(len(frames) for _, frames in tracks)
    t0 = time.perf_counter()
    last, last_label = -1, {}
    while last < total - 1:
        k = min(int((time.perf_counter() - t0) * speed / DT), total - 1)
        if k < total - 1:
            k -= k % stride
        if k == last:
            time.sleep(0.002)
            continue
        for name, frames in tracks:
            q, poses, label, width = frames[min(k, len(frames) - 1)]
            if label != last_label.get(name):
                print(f"  - {name}: {label}" if len(tracks) > 1 else f"  - {label}")
                last_label[name] = label
            actors[name].show(q, width)
            for obj, M in poses.items():
                prev = rendered.get(obj)
                if prev is None or not np.array_equal(prev, M):
                    movers[obj].set_pose(SE3(M, check=False))
                    rendered[obj] = M
        try:
            env.step(DT)
        except Exception as exc:  # e.g. the browser tab was closed or reloaded
            print(f"Swift connection problem ({exc}). Keep the Swift tab open and visible, "
                  "then run again.")
            return False
        last = k
    return True


# ---------------------------------------------------------------- questions and set-up
def ask_choice(prompt, options, default=None):
    """Keep asking until the answer is one of options (case-insensitive); Enter picks default."""
    while True:
        answer = input(prompt).strip().lower()
        if answer == "" and default is not None:
            return default
        if answer in options:
            return answer
        print(f"  Please answer one of: {', '.join(options)}")


def ask_job_type(options):
    """What should the real robot do? Returns a key of options (name -> tool list)."""
    names = list(options)
    print("What should the real robot do?")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}   (tools: {', '.join(options[name])})")
    answer = ask_choice(f"Choose 1-{len(names)} (Enter = 1): ", names + [str(i) for i in range(1, len(names) + 1)],
                        default="1")
    return names[int(answer) - 1] if answer.isdigit() else answer


def screw_possible(tools):
    """A screw needs a drilled hole and the screwdriver to turn it in afterwards."""
    return "drill" in tools and "screwdriver" in tools and tools.index("drill") < tools.index("screwdriver")


def ask_job(tools):
    """Nut / bolt and screw questions. Returns {'nut':.., 'bolt':.., 'screw':..} (only what is needed), or None."""
    sizes = {}
    options = list(S.SIZES)
    if ask_choice("Does the real robot need a nut and bolt? (y/n): ", ["y", "yes", "n", "no"]) in ("y", "yes"):
        nut = ask_choice(f"  Which size NUT? ({' / '.join(options)}): ", options)
        bolt = ask_choice(f"  Which size BOLT? ({' / '.join(options)}, Enter = same as nut): ", options, default=nut)
        sizes.update(nut=nut, bolt=bolt)
    if not screw_possible(tools):
        print("  (A screw goes into a drilled hole, so it is only offered for the drilling and screwing job.)")
    elif ask_choice("Does the real robot need a screw in the drilled hole? (y/n): ", ["y", "yes", "n", "no"]) in ("y", "yes"):
        sizes["screw"] = ask_choice(f"  Which size SCREW? ({' / '.join(options)}): ", options)
    if not sizes:
        print("  No hardware needed. SupplyBot will just take the finished piece away.")
        return None
    print(f"  Supplying the {S.describe(sizes)}.")
    return sizes


def pick_home(robot, candidates_deg, label):
    """First ready pose in the list that does not hit the arm itself, the table or the wall."""
    for deg in candidates_deg:
        q = np.deg2rad(deg)
        if not arm_collision(robot, q):
            return q
    print(f"Warning: none of {label}'s ready poses is collision-free - using the first one.")
    return np.deg2rad(candidates_deg[0])


def check_files():
    """Make sure the shared files are the latest versions; name any that are not."""
    stale = []
    if "v9" not in getattr(_task, "VERSION", ""):
        stale.append("robot2_task.py")
    if "tool_saw" not in _tools.TOOL_PARTS or "screw_large" not in _tools.TOOL_PARTS:
        stale.append("tools.py")
    if not hasattr(S, "FLOOR_LIMIT_Z") or not hasattr(S, "hardware"):
        stale.append("supply_layout.py")
    if not hasattr(C, "UR_GRASP") or not hasattr(C, "real_robot_path"):
        stale.append("chain_layout.py")
    if "radius" not in ArmDrawing.__init__.__code__.co_varnames:
        stale.append("robot2_sim.py")
    if stale:
        print("Out-of-date files in this folder - download the latest and replace: " + ", ".join(stale))
    return not stale


def build_robots():
    """The four robots, placed on their squares, each with its gripper point as the tool."""
    toolbot = SupplierArm()
    if TOOLBOT_SHOULDER_LIMIT_DEG is not None:
        lim = np.deg2rad(TOOLBOT_SHOULDER_LIMIT_DEG)
        toolbot.links[1].qlim = np.array([-lim, lim])
        if abs(float(toolbot.qlim[1][1]) - lim) > 1e-6:          # qlim cached by this toolbox version
            try:
                toolbot.qlim = np.array([[l.qlim[0] for l in toolbot.links], [l.qlim[1] for l in toolbot.links]])
            except AttributeError:
                print("Warning: could not widen ToolBot's shoulder limit in this toolbox version.")
        print(f"ToolBot's shoulder limit is widened to +-{TOOLBOT_SHOULDER_LIMIT_DEG} degrees in the sim "
              "(so it can reach the saw); tool_bot.py is unchanged.")
    toolbot.base = C.TOOLBOT_BASE
    toolbot.tool = SE3(0, 0, C.TOOLBOT_GRIP)
    transfer = PartMover(base=C.TRANSFER_BASE)
    transfer.tool = SE3(0, 0, GRIP_CENTRE)          # so fkine() gives the point between the fingertips
    supply = SupplyBot(base=S.BASE_POSE)
    ur3e = UR3e(base=C.UR3E_BASE)
    # ready poses, checked against each robot's own collision limits
    set_collision_limits(floor_z=C.TABLE_H + 0.03, wall_y=Garage.TOOL_FACE_Y - 0.06)
    toolbot.home_q = pick_home(toolbot, TOOLBOT_HOMES, "ToolBot")
    set_collision_limits(floor_z=C.TABLE_H + 0.03, wall_y=10.0)
    transfer.home_q = pick_home(transfer, TRANSFER_HOMES, "TransferBot")
    set_collision_limits(floor_z=S.FLOOR_LIMIT_Z, wall_y=S.WALL_LIMIT_Y)
    supply.home_q = pick_home(supply, SUPPLY_HOMES + [list(np.rad2deg(SUPPLY_HOME))], "SupplyBot")
    set_collision_limits(floor_z=C.TABLE_H + 0.03, wall_y=10.0)
    ur3e.home_q = pick_home(ur3e, UR3E_HOMES, "UR3e")
    return toolbot, transfer, supply, ur3e


def make_step(title, tracks, wait_msg=None):
    return {"title": title, "tracks": tracks, "wait": wait_msg}


def build_schedule(tools, plans, robots, state, widths, supply_steps):
    """Order the robots' actions into steps. Frames are made in playback order."""
    def track(kind, tool):
        who = TOOL_ACTIONS[kind][0]
        frames, widths[who] = flatten(plans[(kind, tool)], robots[who], state, widths[who])
        return (who, frames)

    def supply_track(key):
        name, stages, _ = supply_steps[key]
        frames, _ = flatten(stages, robots["SupplyBot"], state, None)
        return name, ("SupplyBot", frames)

    first = [track("fetch", tools[0])]
    title = f"ToolBot takes the {tools[0]} off the pegboard"
    if supply_steps["hardware"]:                                 # nut / bolt needed: SupplyBot works alongside ToolBot
        name, supply_frames = supply_track("hardware")
        first.append(supply_frames)
        title += f", while SupplyBot does this: {name}"
    steps = [make_step(title, first),
             make_step(f"TransferBot carries the {tools[0]} to the real robot", [track("deliver", tools[0])])]
    for i, tool in enumerate(tools):
        nxt = tools[i + 1] if i + 1 < len(tools) else None
        use = [track("use", tool)] + ([track("fetch", nxt)] if nxt else [])
        steps.append(make_step(f"UR3e (real robot) {TASK_VERB.get(tool, 'works')} with the {tool}"
                               + (f", while ToolBot gets the {nxt} ready" if nxt else ""), use))
        retire = track("retire", tool)
        retire_title = f"TransferBot puts the used {tool} on the used-tools mark"
        if tool == "drill" and supply_steps["screw"]:            # the hole now exists: SupplyBot stands the screw in it
            screw_name, screw_frames = supply_track("screw")
            wait = find_stagger(robots, retire, screw_frames)
            if wait is None:
                print(f"  (TransferBot and SupplyBot would collide, so the screw goes in the hole after the "
                      f"used {tool} is put away.)")
                steps.append(make_step(retire_title, [retire]))
                steps.append(make_step(f"SupplyBot: {screw_name}", [screw_frames]))
            else:
                if wait > 0:
                    print(f"  (SupplyBot waits {wait:.1f} s before the screw so the two arms stay apart.)")
                steps.append(make_step(f"{retire_title}, while SupplyBot does this: {screw_name}",
                                       [retire, (screw_frames[0], delayed(screw_frames[1], wait))]))
        else:
            steps.append(make_step(retire_title, [retire]))
        back = track("return", tool)
        if nxt is None:
            # The UR3e has nothing left to do and the hand-over square is empty, so SupplyBot can take the
            # finished piece away while ToolBot hangs the last tool back (unless the arms would get in each other's way).
            name, collect = supply_track("collect")
            wait = find_stagger(robots, back, collect)
            hang = f"ToolBot hangs the used {tool} back on the pegboard"
            if wait is None:
                print("  (ToolBot and SupplyBot would collide, so the finished piece is collected after the "
                      f"{tool} is back on the pegboard.)")
                steps.append(make_step(hang, [back]))
                steps.append(make_step(f"SupplyBot: {name}", [collect], wait_msg=supply_steps["collect"][2]))
            else:
                if wait > 0:
                    print(f"  (SupplyBot waits {wait:.1f} s before collecting so the two arms stay apart.)")
                steps.append(make_step(f"{hang}, while SupplyBot does this: {name}",
                                       [back, (collect[0], delayed(collect[1], wait))],
                                       wait_msg=supply_steps["collect"][2]))
            continue
        deliver = track("deliver", nxt)
        wait = find_stagger(robots, deliver, back)           # TransferBot goes straight away, ToolBot may wait
        if wait is None:
            print(f"  (ToolBot and TransferBot would collide, so the {tool} return and the {nxt} delivery "
                  "run one after the other.)")
            steps.append(make_step(f"TransferBot carries the {nxt} to the real robot", [deliver]))
            steps.append(make_step(f"ToolBot hangs the used {tool} back on the pegboard", [back]))
        else:
            if wait > 0:
                print(f"  (ToolBot waits {wait:.1f} s before hanging the {tool} back so the two arms stay apart.)")
            steps.append(make_step(f"TransferBot carries the {nxt} to the real robot, while ToolBot hangs "
                                   f"the used {tool} back", [deliver, (back[0], delayed(back[1], wait))]))
    warn_idle_conflicts(steps, robots)
    return steps


def warn_idle_conflicts(steps, robots):
    """Warn if a working arm comes close to an arm that is standing at its ready pose."""
    warned = set()
    for step in steps:
        working = {name for name, _ in step["tracks"]}
        for name, frames in step["tracks"]:
            for other, robot in robots.items():
                if other in working or (name, other) in warned:
                    continue
                idle = arm_points(robot, robot.home_q)
                for f in frames[::6]:
                    pts = arm_points(robots[name], f[0])
                    if any(_segment_distance(pts[i], pts[i + 1], idle[j], idle[j + 1]) < PARALLEL_MARGIN * 0.6
                           for i in range(len(pts) - 1) for j in range(len(idle) - 1)):
                        print(f"  Warning: {name} passes very close to {other}'s ready pose ({step['title']}).")
                        warned.add((name, other))
                        break


WAIT_LABEL = "waiting for the other arm to clear"


def run_step_showcase(env, actors, movers, rendered, step):
    """Showcase mode: play each robot's task on its own, pausing for Enter before each one.

    A step that normally has several robots working together is split up, so the group can explain one
    robot at a time. Waiting frames added to keep two arms apart are skipped, since the arms no longer overlap.
    """
    for name, frames in step["tracks"]:
        frames = [f for f in frames if f[2] != WAIT_LABEL]
        if not frames:
            continue
        input(f"  Press Enter to show {name}... ")
        if not play(env, actors, movers, [(name, frames)], rendered):
            return False
    return True


def run_step(env, actors, movers, rendered, step):
    """Play one step (one or more robots together). Returns False if Swift disconnects."""
    return play(env, actors, movers, step["tracks"], rendered) if step["tracks"] else True


def main():
    print(f"big_sim {SIM_VERSION}")
    print(f"robot2_task {getattr(_task, 'VERSION', 'OLD VERSION - replace robot2_task.py')}")
    if not check_files():
        return
    toolbot, transfer, supply, ur3e = build_robots()
    robots = {"ToolBot": toolbot, "TransferBot": transfer, "UR3e": ur3e, "SupplyBot": supply}

    try:
        print("\n--- Job setup ---")
        showcase = ask_choice("Showcase each robot one at a time (pause for Enter before each task)? "
                              "(y = showcase each robot, n = full system as normal): ",
                              ["y", "yes", "n", "no"]) in ("y", "yes")
        options = dict(C.JOBS)
        while True:
            if not options:
                print("No job can be planned with the current robot dimensions - see the messages above.")
                return
            job = ask_job_type(options)
            tools = options[job]
            sizes = ask_job(tools)
            print(f"\nPlanning {job} (this can take a minute)...")
            plans, error = plan_job_tools(robots, tools, (sizes or {}).get("screw"))
            if plans is not None:
                break
            print(f"  Cannot do '{job}': {error}.")
            del options[job]
    except EOFError:
        return

    print("\nPlanning SupplyBot...")
    supply_steps = plan_supplybot(supply, sizes, idle_arms=[ur3e])
    if supply_steps is None:
        print("SupplyBot could not complete its task - see the messages above.")
        return

    # Track every movable object through the whole run, in playback order
    state = {tool: C.tool_start(tool) for tool in tools}
    state["wood"] = S.WOOD_START
    if "drill" in tools:
        state["hole"] = S.hole_hidden_pose()
    if sizes is not None:
        for kind, size in S.hardware(sizes):
            state[kind] = S.start_pose(kind, size)
    print("\nChecking that the arms stay clear of each other...")
    steps = build_schedule(tools, plans, robots, state, {"ToolBot": None, "TransferBot": OPEN_WIDTH,
                                                         "UR3e": None, "SupplyBot": None}, supply_steps)
    total = sum(max((len(f) for _, f in s["tracks"]), default=0) for s in steps) * DT
    print(f"Planned robot motion: {total:.1f} s (playback speed x{PLAYBACK_SPEED})")

    env = launch_garage(tools)
    movers = build_supply_scene(env, sizes, S.hole_hidden_pose() if "drill" in tools else None)   # + the hidden hole
    for tool in tools:
        movers[tool] = ToolModel(f"tool_{tool}")
        movers[tool].add_to(env)
        movers[tool].set_pose(C.tool_start(tool))
    toolbot_rows = [(link.d, link.a) for link in toolbot.links]
    actors = {"ToolBot": SkeletonActor(env, toolbot, toolbot_rows, C.TOOLBOT_GRIP, TOOLBOT_COLOR),
              "TransferBot": PartMoverActor(env, transfer),
              "UR3e": SkeletonActor(env, ur3e, [(row[0], row[1]) for row in UR_DH], UR_GRIP, UR3E_COLOR, UR_RADIUS),
              "SupplyBot": SkeletonActor(env, supply, SUPPLY_DH, SUPPLY_GRIP, SUPPLY_COLOR)}
    for name, robot in robots.items():
        actors[name].show(robot.home_q, OPEN_WIDTH if name == "TransferBot" else None)
    env.step(DT)

    rendered = {}
    if showcase:
        print("\nShowcase mode: keep the Swift tab open and visible. Each robot's task waits for you to press Enter.")
    else:
        print("\nStarting in 3 s - keep the Swift tab open and visible. The whole job now runs in one go.")
        time.sleep(3)
    play_step = run_step_showcase if showcase else run_step
    for step in steps:
        print(f"\n{step['title']}")
        if not play_step(env, actors, movers, rendered, step):
            env.hold()
            return
    print("\nAll done. Close the Swift window or press Ctrl+C to exit.")
    env.hold()


if __name__ == "__main__":
    main()
