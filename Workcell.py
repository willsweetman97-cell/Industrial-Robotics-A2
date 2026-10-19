"""The workcell controller: plays the job, jogs the robots, and enforces the safety rules.

Author: [Your name]

No GUI code and no blocking loops live here. The GUI (gui.py) calls tick() from a timer about 25 times
a second; everything else is an event handler. Each tick:
  1. read the e-stop inputs and any queued sensor events           -> SafetyController
  2. predict whether a planned path hits a placed object            -> SafetyController
  3. if the controller says RUNNING, advance the job clock (at safe speed if somebody is in the
     warning zone) and draw the frames; otherwise everything stands still
  4. update the safety-equipment models (beacon, light curtain, e-stop button, operator figure)

Educational demo only - not a certified safety system.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np
import spatialgeometry as sg
from spatialmath import SE3

import Garage
import big_sim as B
import chain_layout as C
import supply_layout as S
from estop_input import NoPlcEStop, SimulatedEStop
from robot2_task import (DT, _segment_distance, arm_collision, arm_points, joint_move, numeric_jacobian,
                         set_collision_limits)
from safety import SafetyController, State

TOOLS_ON_WALL = ["saw", "drill", "screwdriver"]       # tools that get movers in the scene
OBSTACLE_SIZE = 0.12                                  # side of the placed obstacle (m)
ARM_RADIUS = 0.05                                     # capsule radius used for collision prediction
PREDICT_MARGIN = 0.05                                 # extra gap before we call it a collision (m)
LOOKAHEAD = 1.5                                       # how far ahead (real seconds) planned paths are checked
HOMING_SPEED = 0.5                                    # rad/s when returning to the ready pose
JOG_STEP_MAX = 0.12                                   # biggest joint change per Cartesian jog step (rad)
HIDDEN = SE3(0, 0, -5).A

# Operator zones (in front of the UR3e's side of the table). The light curtain is the line y = CURTAIN_Y.
CURTAIN_Y = -0.56
CURTAIN_X = (-0.10, 1.10)
WARN_Y = (-1.11, CURTAIN_Y)
OPERATOR_AT = {1: (0.5, -0.90), 2: (0.5, CURTAIN_Y)}

LIMITS = {   # per robot: collision-check limits, same as used when planning
    "ToolBot": dict(floor_z=C.TABLE_H + 0.03, wall_y=Garage.TOOL_FACE_Y - 0.06),
    "TransferBot": dict(floor_z=C.TABLE_H + 0.03, wall_y=10.0),
    "UR3e": dict(floor_z=C.TABLE_H + 0.03, wall_y=10.0),
    "SupplyBot": dict(floor_z=S.FLOOR_LIMIT_Z, wall_y=S.WALL_LIMIT_Y),
}


# ---------------------------------------------------------------- safety equipment models
class Toggle:
    """A shape that is shown at its pose or hidden under the floor (colours are not changed at run time)."""

    def __init__(self, shape, pose):
        self.shape, self.pose, self.on = shape, pose, None

    def show(self, on):
        if on != self.on:
            self.shape.T = self.pose if on else HIDDEN
            self.on = on
            return True
        return False


class Equipment:
    """Light curtain, barriers, warning strip, beacon, e-stop station and the operator figure."""

    def __init__(self, env):
        self.env = env
        self.toggles = {}
        box, cyl = Garage.box, Garage.cyl
        yellow, dark = (0.95, 0.8, 0.1, 1), (0.15, 0.15, 0.17, 1)

        def add(shape):
            self.env.add(shape)
            return shape

        def fixed(shape):
            add(shape)

        def toggle(key, shape, on=False):
            add(shape)
            self.toggles[key] = Toggle(shape, shape.T.copy() if hasattr(shape.T, "copy") else shape.T)
            return self.toggles[key]

        # warning strip on the floor in front of the table
        fixed(box((2.0, WARN_Y[1] - WARN_Y[0], 0.004), (0.5, (WARN_Y[0] + WARN_Y[1]) / 2, 0.002), (0.95, 0.8, 0.1, 0.8)))
        # light curtain: two posts, a status strip on each, and the beam plane (green = clear, red = broken)
        for key, x in (("post_l", CURTAIN_X[0]), ("post_r", CURTAIN_X[1])):
            fixed(box((0.06, 0.06, 1.5), (x, CURTAIN_Y, 0.75), dark))
            toggle(f"{key}_ok", box((0.02, 0.02, 1.3), (x, CURTAIN_Y + 0.035, 0.75), (0.1, 0.8, 0.2, 1)))
            toggle(f"{key}_bad", box((0.02, 0.02, 1.3), (x, CURTAIN_Y + 0.035, 0.75), (0.9, 0.1, 0.1, 1)))
        width = CURTAIN_X[1] - CURTAIN_X[0]
        mid = (CURTAIN_X[0] + CURTAIN_X[1]) / 2
        toggle("beam_ok", box((width, 0.006, 0.6), (mid, CURTAIN_Y, 1.1), (0.1, 0.9, 0.3, 0.30)))
        toggle("beam_bad", box((width, 0.006, 0.6), (mid, CURTAIN_Y, 1.1), (0.95, 0.1, 0.1, 0.45)))
        # fixed barriers either side of the curtain gap
        for x0, x1 in ((-0.70, CURTAIN_X[0]), (CURTAIN_X[1], 1.90)):
            for x in (x0, x1):
                fixed(box((0.05, 0.05, 1.05), (x, CURTAIN_Y, 0.525), yellow))
            for z in (0.55, 1.0):
                fixed(box((abs(x1 - x0), 0.03, 0.04), ((x0 + x1) / 2, CURTAIN_Y, z), yellow))
        # stack light on the table
        bx, by = 1.2, -0.40
        fixed(box((0.04, 0.04, 0.5), (bx, by, C.TABLE_H + 0.25), dark))
        for name, z, lit, dim in (("red", 0.64, (1, 0.1, 0.1, 1), (0.35, 0.05, 0.05, 1)),
                                  ("amber", 0.58, (1, 0.75, 0.0, 1), (0.35, 0.28, 0.0, 1)),
                                  ("green", 0.52, (0.1, 1, 0.2, 1), (0.05, 0.3, 0.08, 1))):
            toggle(f"lamp_{name}_on", cyl(0.04, 0.05, (bx, by, C.TABLE_H + z), lit))
            toggle(f"lamp_{name}_off", cyl(0.04, 0.05, (bx, by, C.TABLE_H + z), dim))
        # e-stop station (mushroom button pops up when released, sits low when pressed)
        ex, ey = 1.3, -0.44
        fixed(box((0.14, 0.10, 0.14), (ex, ey, C.TABLE_H + 0.07), yellow))
        toggle("estop_up", cyl(0.035, 0.05, (ex, ey, C.TABLE_H + 0.17), (0.85, 0.05, 0.05, 1)))
        toggle("estop_down", cyl(0.035, 0.05, (ex, ey, C.TABLE_H + 0.15), (0.45, 0.02, 0.02, 1)))
        # the operator: body and head
        self.body = Garage.cyl(0.17, 1.3, (0, 0, 0.65), (0.95, 0.5, 0.1, 1))
        self.head = sg.Sphere(radius=0.11, color=(0.9, 0.75, 0.6, 1))
        self.head.T = SE3(0, 0, 1.5).A
        add(self.body)
        add(self.head)
        self.operator_pose = {}
        self.show_operator(0)
        self.set_state(State.READY, 0, False)

    def show_operator(self, level):
        if level in OPERATOR_AT:
            x, y = OPERATOR_AT[level]
            self.body.T = SE3(x, y, 0.65).A
            self.head.T = SE3(x, y, 1.5).A
        else:
            self.body.T = HIDDEN
            self.head.T = HIDDEN

    def set_state(self, state, zone_level, estop_active):
        """Update the lamps, curtain colour and e-stop button. Returns True if anything changed."""
        red = state in (State.ESTOP, State.PROTECTIVE_STOP)
        amber = state in (State.PAUSED, State.STOPPED) or (state is State.RUNNING and zone_level == 1)
        green = state in (State.RUNNING, State.READY) and not amber
        changed = False
        for name, on in (("red", red), ("amber", amber), ("green", green)):
            changed |= self.toggles[f"lamp_{name}_on"].show(on)
            changed |= self.toggles[f"lamp_{name}_off"].show(not on)
        broken = zone_level >= 2
        for key in ("post_l", "post_r"):
            changed |= self.toggles[f"{key}_ok"].show(not broken)
            changed |= self.toggles[f"{key}_bad"].show(broken)
        changed |= self.toggles["beam_ok"].show(not broken)
        changed |= self.toggles["beam_bad"].show(broken)
        changed |= self.toggles["estop_up"].show(not estop_active)
        changed |= self.toggles["estop_down"].show(estop_active)
        return changed


# ---------------------------------------------------------------- geometry helpers
def segment_hits_box(p, q, centre, half):
    """True if segment p-q passes within the axis-aligned box (centre +- half)."""
    n = max(2, int(np.linalg.norm(q - p) / 0.02) + 1)
    pts = p + np.outer(np.linspace(0, 1, n), q - p)
    return bool(np.any(np.all(np.abs(pts - centre) <= half, axis=1)))


# ---------------------------------------------------------------- the workcell
class Workcell:
    def __init__(self):
        self.controller = SafetyController()
        self.controller.on_stop = self._on_stop
        self.sim_estop = SimulatedEStop()
        self.plc_estop = NoPlcEStop()
        self.events: queue.Queue = queue.Queue()      # (name, value) posted from any thread
        self.faults: list[str] = []                   # short fault messages for the GUI
        self.toolbot, self.transfer, self.supply, self.ur3e = B.build_robots()
        self.robots = {"ToolBot": self.toolbot, "TransferBot": self.transfer, "UR3e": self.ur3e,
                       "SupplyBot": self.supply}
        self.q = {n: np.array(r.home_q, float) for n, r in self.robots.items()}
        self.width = {"TransferBot": B.OPEN_WIDTH}
        self.plans, self.supply_steps, self.sizes = {}, None, None
        self.steps, self.prepared = [], {}            # steps being played / prepared per job name
        self.job_name, self.si, self.t = None, 0, 0.0
        self.recovering = False
        self.env = None
        self.movers, self.actors, self.rendered = {}, {}, {}
        self.equipment = None
        self.obstacle = None                          # (shape, centre array) while an obstacle is placed
        self._last_tick = None
        self._dirty, self._since_step = True, 0
        self._random_stop = threading.Event()
        self.swift_ok = True

    # ------------------------------------------------------------ set-up (planning is slow: call from a worker thread)
    def plan_hardware(self, sizes, progress=print):
        """SupplyBot's motions for the chosen nut / bolt sizes (sizes None = no hardware)."""
        self.sizes = sizes
        progress("Planning SupplyBot...")
        self.supply_steps = B.plan_supplybot(self.supply, sizes)
        return self.supply_steps is not None

    def plan_job(self, job, progress=print):
        """Plan every tool action the job needs and build its schedule. Returns an error text or None."""
        tools = C.JOBS[job]
        for kind in ("fetch", "return", "use", "deliver", "retire"):
            for tool in tools:
                if (kind, tool) in self.plans:
                    continue
                progress(f"Planning {B.TOOL_ACTIONS[kind][0]}: {kind} {tool}...")
                stages = B.plan_tool_action(self.robots[B.TOOL_ACTIONS[kind][0]], kind, tool)
                if stages is None:
                    return f"{B.TOOL_ACTIONS[kind][0]} cannot {kind} the {tool}"
                self.plans[(kind, tool)] = stages
        progress("Checking that the arms stay clear of each other...")
        state = {tool: C.tool_start(tool) for tool in tools}
        state["wood"] = S.WOOD_START
        if self.sizes is not None:
            state["nut"], state["bolt"] = S.nut_start(self.sizes["nut"]), S.bolt_start(self.sizes["bolt"])
        self.prepared[job] = B.build_schedule(
            tools, self.plans, self.robots, state,
            {"ToolBot": None, "TransferBot": B.OPEN_WIDTH, "UR3e": None, "SupplyBot": None}, self.supply_steps)
        progress(f"'{job}' is ready to start.")
        return None

    # ------------------------------------------------------------ scene (call from the main thread)
    def launch(self):
        self.env = B.launch_garage(TOOLS_ON_WALL)
        self.movers = B.build_supply_scene(self.env, self.sizes)
        for tool in TOOLS_ON_WALL:
            self.movers[tool] = B.ToolModel(f"tool_{tool}")
            self.movers[tool].add_to(self.env)
        rows = lambda r: [(link.d, link.a) for link in r.links]
        self.actors = {
            "ToolBot": B.SkeletonActor(self.env, self.toolbot, rows(self.toolbot), C.TOOLBOT_GRIP, B.TOOLBOT_COLOR),
            "TransferBot": B.PartMoverActor(self.env, self.transfer),
            "UR3e": B.SkeletonActor(self.env, self.ur3e, rows(self.ur3e), B.UR_GRIP, B.UR3E_COLOR, B.UR_RADIUS),
            "SupplyBot": B.SkeletonActor(self.env, self.supply, B.SUPPLY_DH, B.SUPPLY_GRIP, B.SUPPLY_COLOR)}
        self.equipment = Equipment(self.env)
        self.reset_scene()
        for name in self.robots:
            self._show(name, self.q[name])
        self._step_swift(force=True)

    def reset_scene(self):
        """Put every movable object back where it starts."""
        for tool in TOOLS_ON_WALL:
            self.movers[tool].set_pose(C.tool_start(tool))
        self.movers["wood"].set_pose(S.WOOD_START)
        if self.sizes is not None:
            self.movers["nut"].set_pose(S.nut_start(self.sizes["nut"]))
            self.movers["bolt"].set_pose(S.bolt_start(self.sizes["bolt"]))
        self.rendered.clear()
        self._dirty = True

    # ------------------------------------------------------------ event inputs (safe to call from any thread)
    def post(self, name, value=None):
        self.events.put((name, value))

    def press_sim_estop(self):
        self.sim_estop.press()
        self.controller.set_estop("GUI", True, "Emergency stop pressed (on-screen button)")   # immediate, no waiting for the tick

    def release_sim_estop(self):
        self.sim_estop.release()
        self.controller.set_estop("GUI", False)

    def start_random_intrusions(self, mean_period=20.0):
        """Simulated asynchronous sensor: now and then an operator walks up to the cell on their own."""
        self._random_stop.clear()

        def run():
            rng = np.random.default_rng()
            while not self._random_stop.wait(float(rng.uniform(0.5, 1.5) * mean_period)):
                self.post("zone", 2 if rng.random() < 0.5 else 1)
                if self._random_stop.wait(float(rng.uniform(4, 8))):
                    break
                self.post("zone", 0)

        threading.Thread(target=run, name="random-intrusions", daemon=True).start()

    def stop_random_intrusions(self):
        self._random_stop.set()

    # ------------------------------------------------------------ operator actions
    def start_job(self, job):
        if job not in self.prepared:
            return False, "Plan the job first"
        if not self.controller.jog_allowed:
            return False, f"Cannot start while {self.controller.state.value}"
        ok, msg = self.controller.begin_job()
        if not ok:
            return ok, msg
        self.reset_scene()
        self.job_name, self.recovering = job, False
        self.steps = list(self.prepared[job])
        homing = self._homing_step("Moving the robots to their ready poses")
        if homing["tracks"]:
            self.steps.insert(0, homing)
        self.si, self.t = 0, 0.0
        return True, msg

    def reset(self):
        ok, msg = self.controller.reset()
        if ok:
            self.faults.clear()
        else:
            self.faults.append(msg)
        return ok, msg

    def start_resume(self):
        return self.controller.start_resume()

    def pause(self):
        self.controller.pause()

    def abort_job(self):
        """Stop the job and drive every robot back to its ready pose (needs Reset first, then this explicit action)."""
        if not self.controller.job_active:
            return False, "No job running"
        if self.controller.state not in (State.STOPPED, State.PAUSED):
            return False, "Pause or reset the stop first"
        self.steps = [self._homing_step("Recovering: all robots return to their ready poses")]
        self.si, self.t, self.recovering = 0, 0.0, True
        self.controller.state = State.RUNNING
        self.controller._note("Recovery: returning robots to ready poses")
        return True, "Recovering"

    def _homing_step(self, title):
        tracks = []
        for name, robot in self.robots.items():
            if np.max(np.abs(self.q[name] - robot.home_q)) > 1e-3:
                traj = joint_move(self.q[name], robot.home_q, speed=HOMING_SPEED)
                w = self.width.get(name)
                tracks.append((name, [(q, {}, "to ready pose", w if name == "TransferBot" else None) for q in traj]))
        return {"title": title, "tracks": tracks, "wait": None}

    # ------------------------------------------------------------ the tick
    def tick(self):
        """Call about 25 times a second from the GUI's timer. Never blocks."""
        now = time.monotonic()
        dt = 0.0 if self._last_tick is None else min(now - self._last_tick, 0.1)
        self._last_tick = now
        self._poll_inputs()
        if self.steps and self.controller.state is State.RUNNING:
            self._predict_collisions()
        if self.steps and self.controller.motion_enabled:
            self._advance(dt * self.controller.speed_scale)
        if self.equipment is not None:
            if self.equipment.set_state(self.controller.state, self.controller.zone_level,
                                        any(self.controller.estop_inputs.values())):
                self._dirty = True
        self._step_swift()

    def _poll_inputs(self):
        for src in (self.sim_estop, self.plc_estop):
            reading = src.poll()
            name = "GUI" if src is self.sim_estop else "PLC"
            if reading.stop:
                self.controller.set_estop(name, True, f"Emergency stop ({src.name}): {reading.text}")
            else:
                self.controller.set_estop(name, False)
        while True:
            try:
                name, value = self.events.get_nowait()
            except queue.Empty:
                break
            if name == "zone":
                self.controller.set_zone(int(value))
                if self.equipment is not None:
                    self.equipment.show_operator(int(value))
                self._dirty = True
            elif name == "obstacle_path":
                self.place_obstacle_in_path()
            elif name == "obstacle_at":
                self.place_obstacle_at(np.array(value, float))
            elif name == "obstacle_remove":
                self.remove_obstacle()

    # --- job playback
    def _advance(self, dt):
        step = self.steps[self.si]
        duration = max((len(f) for _, f in step["tracks"]), default=0) * DT
        self.t = min(self.t + dt, duration)
        k = int(self.t / DT)
        for name, frames in step["tracks"]:
            self._show(name, *self._frame(frames, k))
        if self.t >= duration:
            self.si += 1
            self.t = 0.0
            if self.si >= len(self.steps):
                finished_recovery = self.recovering
                self.steps, self.si = [], 0
                self.controller.job_finished()
                if finished_recovery:
                    self.reset_scene()
                    self.controller._note("Recovered to a known safe state")
                self.recovering = False
                self.job_name = None
            self._dirty = True

    def _frame(self, frames, k):
        f = frames[min(k, len(frames) - 1)]
        return f[0], f[1], f[3]

    def _show(self, name, q, poses=None, width=None):
        self.q[name] = np.asarray(q, float)
        if width is not None:
            self.width[name] = width
        self.actors[name].show(q, width if width is not None else self.width.get(name))
        for obj, M in (poses or {}).items():
            prev = self.rendered.get(obj)
            if prev is None or not np.array_equal(prev, M):
                self.movers[obj].set_pose(SE3(M, check=False))
                self.rendered[obj] = M
        self._dirty = True

    # --- collision prediction
    def _tracks_ahead(self, horizon, every=0.1):
        """Yield (robot name, arm points) along the rest of this step for the next `horizon` real seconds."""
        if not self.steps:
            return
        step = self.steps[self.si]
        frame_horizon = horizon * self.controller.speed_scale
        for name, frames in step["tracks"]:
            for ahead in np.arange(0.0, frame_horizon + 1e-9, every):
                k = min(int((self.t + ahead) / DT), len(frames) - 1)
                yield name, arm_points(self.robots[name], frames[k][0])

    def _obstacle_blocks(self, name, pts):
        if self.obstacle is None:
            return False
        centre = self.obstacle[1]
        half = OBSTACLE_SIZE / 2 + ARM_RADIUS + PREDICT_MARGIN
        return any(segment_hits_box(pts[i], pts[i + 1], centre, half) for i in range(len(pts) - 1))

    def _predict_collisions(self):
        if self.obstacle is None:
            return
        for name, pts in self._tracks_ahead(LOOKAHEAD):
            if self._obstacle_blocks(name, pts):
                self.controller.collision_predicted(
                    f"Object in the planned path of {name} - stopped before contact",
                    still_present=lambda: self.obstacle is not None)
                return

    # --- obstacle object
    def place_obstacle_in_path(self, seconds_ahead=3.0):
        """Drop the test object onto the planned path of the robot that is moving now."""
        if not self.steps:
            self.faults.append("No job is running, so there is no planned path. Start a job first, or place the object by x, y, z.")
            return False
        step = self.steps[self.si]
        for name, frames in step["tracks"]:
            k = min(int((self.t + seconds_ahead) / DT), len(frames) - 1)
            tip = arm_points(self.robots[name], frames[k][0])[-1]
            if tip[2] > C.TABLE_H + 0.03:
                self.place_obstacle_at(tip)
                return True
        return False

    def place_obstacle_at(self, centre):
        self.remove_obstacle()
        shape = Garage.box((OBSTACLE_SIZE,) * 3, tuple(centre), (0.9, 0.2, 0.9, 1))
        self.env.add(shape)
        self.obstacle = (shape, np.asarray(centre, float))
        self.controller._note(f"Test object placed at ({centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f})")
        self._dirty = True

    def remove_obstacle(self):
        if self.obstacle is not None:
            self.obstacle[0].T = HIDDEN
            self.obstacle = None
            self.controller._note("Test object removed")
            self._dirty = True

    # --- Swift
    def _step_swift(self, force=False):
        if self.env is None:
            return
        self._since_step += 1
        if not (self._dirty or force or self._since_step >= 12):
            return
        try:
            self.env.step(DT)
            self.swift_ok = True
        except Exception as exc:          # browser tab closed / reloaded
            if self.swift_ok:
                self.faults.append(f"Swift connection problem: {exc}")
            self.swift_ok = False
        self._dirty, self._since_step = False, 0

    def _on_stop(self, text):
        self.faults.append(text)
        del self.faults[:-30]

    # ------------------------------------------------------------ jogging (teach)
    def _can_jog(self):
        """Jogging needs the scene open, no job active, and no stop in force."""
        return self.env is not None and self.controller.jog_allowed

    def _check_pose(self, name, q):
        """None if q is allowed, else the reason it is not."""
        robot = self.robots[name]
        lo, hi = robot.qlim
        if np.any(q < lo - 1e-9) or np.any(q > hi + 1e-9):
            return "joint limit"
        set_collision_limits(**LIMITS[name])
        if arm_collision(robot, q):
            return "would hit the table / wall / itself"
        if self.obstacle is not None and self._obstacle_blocks(name, arm_points(robot, q)):
            return "test object in the way"
        return None

    def _apply_jog(self, name, q_new):
        why = self._check_pose(name, q_new)
        if why:
            self.faults.append(f"Jog refused ({name}): {why}")
            del self.faults[:-30]
            return False
        self._show(name, q_new)
        return True

    def jog_joint(self, name, index, delta_rad):
        if not self._can_jog():
            return False
        q = self.q[name].copy()
        q[index] += delta_rad
        return self._apply_jog(name, q)

    def jog_cartesian(self, name, axis, delta_m):
        """Move the tool centre point along world x (0), y (1) or z (2), keeping its orientation."""
        if not self._can_jog():
            return False
        robot, q = self.robots[name], self.q[name]
        J = numeric_jacobian(robot, q)
        twist = np.zeros(6)
        twist[axis] = delta_m
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(6), twist)
        biggest = float(np.max(np.abs(dq)))
        if biggest > JOG_STEP_MAX:
            dq *= JOG_STEP_MAX / biggest
        q_new = np.clip(q + dq, *robot.qlim)
        moved = (robot.fkine(q_new).t - robot.fkine(q).t)[axis]
        if abs(moved) < 0.3 * abs(delta_m) or np.sign(moved) != np.sign(delta_m):
            self.faults.append(f"Jog refused ({name}): near a singularity or joint limit")
            return False
        return self._apply_jog(name, q_new)

    def set_gripper(self, width):
        if self._can_jog():
            self.width["TransferBot"] = width
            self._show("TransferBot", self.q["TransferBot"], None, width)

    def go_home(self, name):
        """Jump-free return of one robot to its ready pose, only when idle and jogging is allowed."""
        if not self._can_jog():
            return False
        return self._apply_jog(name, np.array(self.robots[name].home_q, float))

    # ------------------------------------------------------------ status for the GUI
    def status(self):
        c = self.controller
        title, detail, fg, bg = c.banner()
        plc = self.plc_estop.poll()
        step = self.steps[self.si] if self.steps else None
        progress = 0.0
        if step is not None:
            total = sum(max((len(f) for _, f in s["tracks"]), default=0) for s in self.steps) * DT
            done = sum(max((len(f) for _, f in s["tracks"]), default=0) for s in self.steps[:self.si]) * DT + self.t
            progress = done / total if total else 0.0
        robots = {}
        for name, robot in self.robots.items():
            lo, hi = robot.qlim
            q = self.q[name]
            robots[name] = {"q": q.copy(), "tcp": robot.fkine(q).t.copy(),
                            "margin": np.minimum(q - lo, hi - q), "span": hi - lo}
        return {"state": title, "detail": detail, "fg": fg, "bg": bg, "jog_allowed": c.jog_allowed,
                "step": step["title"] if step else "", "progress": progress, "speed_scale": c.speed_scale,
                "zone": c.zone_level, "latched": dict(c.latched), "plc": plc, "sim_estop": self.sim_estop.pressed,
                "faults": list(self.faults), "log": list(c.log[-40:]), "robots": robots,
                "obstacle": self.obstacle is not None, "swift_ok": self.swift_ok,
                "job_active": c.job_active}
