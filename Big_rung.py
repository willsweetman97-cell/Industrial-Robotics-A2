import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import spatialgeometry as sg
from spatialmath import SE3

try:
    import Garage as G
except ImportError:
    import garage as G

# Compatibility: older Swift versions call shape._update_pyb(), which newer spatialgeometry removed.
if not hasattr(sg.Shape, "_update_pyb"):
    sg.Shape._update_pyb = lambda self: None

READY, RUNNING, STOPPED, CLEARED = "READY", "RUNNING", "STOPPED", "CLEARED"
SLOW = 0.3                                           # speed scale while the operator is near
CAUSE = {"gui_estop": "on-screen e-stop", "physical_estop": "physical e-stop (or PLC not responding)",
         "light_curtain": "light curtain broken", "obstacle": "obstacle in the robot's path"}
OBST_R, MARGIN, LOOK, AHEAD = 0.06, 0.08, 45, 100    # object radius, clearance (m); frames looked ahead / dropped ahead
BANNER = {READY: "#6c7a89", RUNNING: "#2e9e4f", STOPPED: "#c0392b", CLEARED: "#d68910"}


# ----------------------------------------------------------------------------- state machine
class Safety:
    def __init__(self):
        self.state, self.causes, self.operator_near, self.log = READY, set(), False, []
        self._lock = threading.RLock()               # the PLC thread and the GUI both use this

    def _note(self, text):
        self.log.append((time.strftime("%H:%M:%S"), text))

    def trip(self, cause):
        """Stop now. Safe to call repeatedly and from any thread."""
        with self._lock:
            if cause not in self.causes:
                self.causes.add(cause)
                self._note("STOP: " + CAUSE[cause])
            self.state = STOPPED

    def release(self, cause):
        """The cause has gone. Still stopped until Reset, then Start / Resume."""
        with self._lock:
            if cause in self.causes:
                self.causes.discard(cause)
                self._note(f"released: {CAUSE[cause]} (still stopped)")

    def reset(self):
        with self._lock:
            if self.state != STOPPED:
                return False, "Nothing to reset"
            if self.causes:
                return False, "Cannot reset, still active: " + ", ".join(CAUSE[c] for c in sorted(self.causes))
            self.state = CLEARED
            self._note("reset acknowledged - press Resume (or Abort)")
            return True, ""

    def start(self):
        """READY -> RUNNING (start) or CLEARED -> RUNNING (resume)."""
        with self._lock:
            if self.causes or self.state not in (READY, CLEARED):
                return False, f"Cannot start from {self.state}"
            self.state = RUNNING
            self._note("running")
            return True, ""

    def finish(self):
        with self._lock:
            if self.state == RUNNING:
                self.state = READY
                self._note("ready")

    def set_operator_near(self, near):
        with self._lock:
            if near != self.operator_near:
                self.operator_near = near
                self._note("operator near: speed limited to 30%" if near else "operator left: normal speed")

    @property
    def speed_scale(self):
        return 0.0 if self.state != RUNNING else (SLOW if self.operator_near else 1.0)


# ----------------------------------------------------------------------------- physical e-stop (PLC)
# ASSUMPTION (check with the lab): Modbus TCP, e-stop on a discrete input that reads 0 when pressed
# (normally-closed). For another protocol write a class with read_pressed(). Needs `pip install pymodbus`.
class FakeInput:
    """Stand-in for testing without hardware."""
    def __init__(self):
        self.pressed, self.fail = False, False

    def read_pressed(self):
        if self.fail:
            raise ConnectionError("simulated PLC failure")
        return self.pressed


class ModbusInput:
    def __init__(self, host, port=502, address=0, unit=1, pressed_when=0, client=None):
        self.host, self.port, self.address, self.unit, self.pressed_when = host, port, address, unit, pressed_when
        self._client = client

    def read_pressed(self):
        if self._client is None:
            from pymodbus.client import ModbusTcpClient
            self._client = ModbusTcpClient(self.host, port=self.port, timeout=0.5)
        if not self._client.connect():
            raise ConnectionError(f"no connection to {self.host}:{self.port}")
        for keyword in ("device_id", "slave", "unit"):        # the keyword differs between pymodbus versions
            try:
                reply = self._client.read_discrete_inputs(self.address, count=1, **{keyword: self.unit})
                break
            except TypeError:
                continue
        else:
            raise RuntimeError("unsupported pymodbus version")
        if reply.isError():
            raise IOError(str(reply))
        return bool(reply.bits[0]) == bool(self.pressed_when)


class PhysicalEstop:
    """Background thread: polls the PLC, sleeping between reads. No answer counts as pressed (fail safe)."""

    def __init__(self, safety, source, period=0.05):
        self.safety, self.source, self.period, self.status = safety, source, period, "starting"
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                pressed = self.source.read_pressed()
                self.status = "PRESSED" if pressed else "released"
            except Exception as exc:
                pressed, self.status = True, f"no response ({exc})"
            self.safety.trip("physical_estop") if pressed else self.safety.release("physical_estop")
            self._stop.wait(self.period)                      # sleeps, never a busy loop


# ----------------------------------------------------------------------------- equipment models
DARK, YELLOW = (0.2, 0.2, 0.22, 1), (1.0, 0.85, 0.0, 1)
LAMP_ON = {"green": (0.1, 0.9, 0.2, 1), "amber": (1.0, 0.7, 0.0, 1), "red": (1.0, 0.1, 0.1, 1)}
LAMP_OFF = {"green": (0.05, 0.25, 0.1, 1), "amber": (0.3, 0.22, 0.03, 1), "red": (0.3, 0.05, 0.05, 1)}
CURTAIN_Y, STRIP_Y = -0.85, -1.10                            # in front of the table (its front edge is y = -0.5)
OPERATOR_Y = {"away": -1.9, "near": -1.30, "inside": -0.70}
STACK, ESTOP = (2.05, 0.78), (1.95, -0.40)


class Equipment:
    """Light curtain, barriers, floor strip, stack light, e-stop station and an operator figure.

    State changes only MOVE shapes (a lit and an unlit lamp are two shapes, one hidden under the floor):
    older Swift versions freeze if a shape's colour is changed while running."""

    def __init__(self):
        H = G.TABLE_H
        self.static = [G.box((0.08, 0.08, 1.5), (sx * 2.3, CURTAIN_Y, 0.75), DARK) for sx in (-1, 1)]
        self.static += [G.box((0.03, 1.8, 0.9), (sx * 2.45, 0.0, 0.45), (1.0, 0.85, 0.0, 0.45)) for sx in (-1, 1)]
        self.static += [G.box((4.9, 0.12, 0.004), (0, STRIP_Y, 0.002), YELLOW),
                        G.cyl(0.015, 0.45, (*STACK, H + 0.225), DARK),
                        G.box((0.12, 0.12, 0.06), (*ESTOP, H + 0.03), YELLOW)]
        curtain = (4.6, 0.01, 0.55), (0, CURTAIN_Y, 1.15)
        self.pairs = {"curtain": (G.box(curtain[0], curtain[1], (0.1, 0.9, 0.2, 0.25)),      # normal
                                  G.box(curtain[0], curtain[1], (1.0, 0.1, 0.1, 0.4)),       # broken
                                  SE3(*curtain[1]))}
        for name, z in (("green", 0.31), ("amber", 0.36), ("red", 0.41)):
            centre = (*STACK, H + z)
            self.pairs[name] = (G.cyl(0.035, 0.045, centre, LAMP_OFF[name]), G.cyl(0.035, 0.045, centre, LAMP_ON[name]),
                                SE3(*centre))
        self.button = G.cyl(0.035, 0.04, (*ESTOP, H + 0.08), (0.9, 0.05, 0.05, 1))
        self.body = G.cyl(0.17, 0.9, (0, OPERATOR_Y["away"], 0.45), (0.3, 0.4, 0.6, 1))
        self.head = G._make(sg.Sphere, SE3(0, OPERATOR_Y["away"], 1.02), (0.85, 0.7, 0.55, 1), radius=0.11)
        self.blocker = G._make(sg.Sphere, SE3(0, 0, -5), (0.9, 0.1, 0.1, 1), radius=OBST_R)   # test object, hidden
        self._seen = {}
        for key in self.pairs:
            self._show(key, 0)                               # starting look: normal curtain, lamps off

    def shapes(self):
        pair_shapes = [shape for a, b, _ in self.pairs.values() for shape in (a, b)]
        return self.static + pair_shapes + [self.button, self.body, self.head, self.blocker]

    def _move(self, shape, pose):
        if self._seen.get(id(shape)) != tuple(pose.t):
            shape.T, self._seen[id(shape)] = pose.A, tuple(pose.t)

    def _show(self, key, which):
        """Show shape 0 or 1 of a pair at its place and hide the other under the floor."""
        shapes, home = self.pairs[key][:2], self.pairs[key][2]
        self._move(shapes[which], home)
        self._move(shapes[1 - which], SE3(0, 0, -5))

    def set_operator(self, where):
        self._move(self.body, SE3(0, OPERATOR_Y[where], 0.45))
        self._move(self.head, SE3(0, OPERATOR_Y[where], 1.02))

    def set_blocker(self, xyz):
        self._move(self.blocker, SE3(*(xyz if xyz is not None else (0, 0, -5))))

    def update(self, s):
        self._show("curtain", int("light_curtain" in s.causes))
        on = {"green": s.state == RUNNING, "red": s.state == STOPPED,
              "amber": s.state in (READY, CLEARED) or (s.state == RUNNING and s.operator_near)}
        for name, lit in on.items():
            self._show(name, int(lit))
        pressed = bool({"gui_estop", "physical_estop"} & s.causes)
        self._move(self.button, SE3(*ESTOP, G.TABLE_H + (0.065 if pressed else 0.08)))


def point_segment(p, a, b):
    """Distance from point p to segment a-b."""
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


# ----------------------------------------------------------------------------- guard: safety-aware playback
class Guard:
    def __init__(self, panel=True):
        self.safety, self.use_panel, self.root, self.env, self.equipment = Safety(), panel, None, None, None
        self.tracks = self.points = self.obstacle = self.plc = self.fake = None
        self.k, self.message, self.abort, self.logged, self.last_state = 0, "", False, -1, READY
        self.last_q, self.homes = {}, {}                     # each robot's last pose and ready pose
        self.clock, self.sleep = time.perf_counter, time.sleep   # tests swap these for a fake clock

    def attach(self, env):
        """Call once Swift is open: adds the equipment models and the control panel."""
        self.env, self.equipment = env, Equipment()
        for shape in self.equipment.shapes():
            env.add(shape)
        if self.use_panel:
            self._build_panel()

    # -- playback ------------------------------------------------------------
    def play(self, tracks, show, points, dt, speed=1.0, fps=25):
        """Safety-aware replacement for big_sim's play(). tracks = [(robot name, frames)],
        show(name, frame) draws one frame, points(name, q) gives the arm's joint positions.
        Returns True when finished, False if the run was aborted (robots then sit at their ready poses)."""
        self.tracks, self.points = tracks, points
        total = max(len(frames) for _, frames in tracks)
        sim, last, prev = 0.0, -1, self.clock()
        while last < total - 1:
            now = self.clock()
            wall, prev = now - prev, now
            self.pump()
            if self.abort and self.safety.state == CLEARED:
                return self._recover(show, points, dt, speed, fps)
            self.check_obstacle()
            sim += wall * self.safety.speed_scale * speed    # stopped = 0, operator near = 0.3, frozen clock
            self.k = min(int(sim / dt), total - 1)
            if self.k != last:
                for name, frames in tracks:
                    frame = frames[min(self.k, len(frames) - 1)]
                    show(name, frame)
                    self.last_q[name] = frame[0]
                last = self.k
            self.env.step(dt)
            self.sleep(1.0 / fps)                            # sleeps between frames; never a busy loop
        return True

    def _recover(self, show, points, dt, speed, fps):
        """After a reset: every robot returns to its ready pose, then the system is READY."""
        self.abort = False
        tracks = []
        for name, home in self.homes.items():
            q0, q1 = np.asarray(self.last_q.get(name, home), float), np.asarray(home, float)
            n = max(30, int(np.max(np.abs(q1 - q0)) / 0.9 / dt))
            t = np.linspace(0, 1, n)
            s = 10 * t**3 - 15 * t**4 + 6 * t**5
            tracks.append((name, [(q0 + e * (q1 - q0), {}, "returning to ready pose", None) for e in s]))
        self.safety.start()
        self.play(tracks, show, points, dt, speed, fps)
        self.safety.finish()
        return False

    def check_obstacle(self):
        """Look ahead along every robot's planned path; stop before an arm gets within the clearance."""
        if self.obstacle is None or "obstacle" in self.safety.causes or not self.tracks:
            return
        for name, frames in self.tracks:
            for j in range(0, LOOK + 1, 4):
                pts = self.points(name, frames[min(self.k + j, len(frames) - 1)][0])
                if min(point_segment(self.obstacle, pts[i], pts[i + 1]) for i in range(len(pts) - 1)) < OBST_R + MARGIN:
                    self.safety.trip("obstacle")
                    return

    def pump(self):
        """Per-frame housekeeping: panel events, equipment colours."""
        if self.safety.state != self.last_state:
            self.last_state, self.message = self.safety.state, ""
        if self.root is not None:
            try:
                self.root.update()
                self._refresh()
            except tk.TclError:                              # panel window closed: treat as a stop
                self.root = None
                self.safety.trip("gui_estop")
        if self.equipment:
            self.equipment.update(self.safety)

    # -- actions (panel buttons) ---------------------------------------------
    def gui_estop(self):
        self.safety.trip("gui_estop")

    def release_gui_estop(self):
        self.safety.release("gui_estop")

    def reset(self):
        self.message = self.safety.reset()[1]

    def start(self):
        self.message = self.safety.start()[1]

    def abort_home(self):
        if self.safety.state == CLEARED:
            self.abort = True
        else:
            self.message = "Reset the stop first"

    def toggle_operator(self):
        near = not self.safety.operator_near
        self.safety.set_operator_near(near)
        self.equipment.set_operator("near" if near else "away")

    def toggle_curtain(self):
        if "light_curtain" in self.safety.causes:
            self.safety.release("light_curtain")
            self.equipment.set_operator("near" if self.safety.operator_near else "away")
        else:
            self.safety.trip("light_curtain")
            self.equipment.set_operator("inside")

    def place_obstacle(self):
        """Drop the test object on the first moving robot's planned tool path, about 2 s ahead."""
        if not self.tracks:
            self.message = "Start the job first"
            return
        name, frames = self.tracks[0]
        self.obstacle = self.points(name, frames[min(self.k + AHEAD, len(frames) - 1)][0])[-1].copy()
        self.equipment.set_blocker(self.obstacle)

    def remove_obstacle(self):
        self.obstacle = None
        self.equipment.set_blocker(None)
        self.safety.release("obstacle")

    def use_plc(self, source):
        if self.plc:
            self.plc.stop()
        self.plc = PhysicalEstop(self.safety, source)

    def use_fake_plc(self):
        self.fake = FakeInput()
        self.use_plc(self.fake)

    # -- control panel ---------------------------------------------------------
    def _build_panel(self):
        r = self.root = tk.Tk()
        r.title("DoGoodBot safety panel")
        self.banner = tk.Label(r, font=("Arial", 18, "bold"), fg="white", width=34, pady=8)
        self.banner.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.info = tk.Label(r, justify="left", anchor="w")
        self.info.grid(row=1, column=0, columnspan=3, sticky="ew", padx=6)
        tk.Button(r, text="E-STOP  (space bar)", bg="#c0392b", fg="white", font=("Arial", 16, "bold"), height=2,
                  takefocus=0, command=self.gui_estop).grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=4)

        def button(text, command, row, col):
            b = ttk.Button(r, text=text, command=command, takefocus=0)
            b.grid(row=row, column=col, sticky="ew", padx=4, pady=2)
            return b

        button("Release on-screen e-stop", self.release_gui_estop, 3, 0)
        button("1. Reset", self.reset, 3, 1)
        self.go = button("2. Start", self.start, 3, 2)
        button("Abort and return to ready", self.abort_home, 4, 0)
        self.op = button("Operator approaches", self.toggle_operator, 4, 1)
        button("Break / clear light curtain", self.toggle_curtain, 4, 2)
        button("Place test object in path", self.place_obstacle, 5, 0)
        button("Remove test object", self.remove_obstacle, 5, 1)
        plc = ttk.LabelFrame(r, text="Physical e-stop (PLC, Modbus TCP)")
        plc.grid(row=6, column=0, columnspan=3, sticky="ew", padx=6, pady=4)
        self.host, self.port, self.addr = tk.StringVar(value="192.168.0.10"), tk.StringVar(value="502"), tk.StringVar(value="0")
        for col, (label, var) in enumerate((("host", self.host), ("port", self.port), ("input", self.addr))):
            ttk.Label(plc, text=label).grid(row=0, column=2 * col)
            ttk.Entry(plc, textvariable=var, width=14 if col == 0 else 5).grid(row=0, column=2 * col + 1)
        ttk.Button(plc, text="Connect PLC", takefocus=0, command=lambda: self.use_plc(
            ModbusInput(self.host.get(), int(self.port.get()), int(self.addr.get())))).grid(row=1, column=0, columnspan=2, pady=2)
        ttk.Button(plc, text="Fake input (testing only)", takefocus=0, command=self.use_fake_plc).grid(row=1, column=2, columnspan=2)
        ttk.Button(plc, text="Fake press", takefocus=0, command=lambda: self.fake and setattr(self.fake, "pressed", True)).grid(row=1, column=4)
        ttk.Button(plc, text="Fake release", takefocus=0, command=lambda: self.fake and setattr(self.fake, "pressed", False)).grid(row=1, column=5)
        self.plc_label = ttk.Label(plc, text="PLC: not connected")
        self.plc_label.grid(row=2, column=0, columnspan=6, sticky="w")
        self.log_box = tk.Listbox(r, height=8)
        self.log_box.grid(row=7, column=0, columnspan=3, sticky="ew", padx=6)
        tk.Label(r, text="Teaching prototype only - not a certified safety system.", fg="#555").grid(row=8, column=0, columnspan=3)
        r.bind("<space>", lambda e: None if isinstance(e.widget, ttk.Entry) else self.gui_estop())

    def _refresh(self):
        s = self.safety
        why = " - " + ", ".join(CAUSE[c] for c in sorted(s.causes)) if s.causes else ""
        self.banner.config(text=s.state + why, bg=BANNER[s.state])
        self.info.config(text=f"speed: {int(s.speed_scale * 100)}%\n{self.message or ('Press 2. Start' if s.state == READY else '')}")
        self.go.config(text="2. Resume" if s.state == CLEARED else "2. Start")
        self.op.config(text="Operator leaves" if s.operator_near else "Operator approaches")
        self.plc_label.config(text="PLC: " + (self.plc.status if self.plc else "not connected"))
        if self.logged != len(s.log):
            self.logged = len(s.log)
            self.log_box.delete(0, "end")
            for stamp, text in s.log[-8:]:
                self.log_box.insert("end", f"{stamp}  {text}")


# ----------------------------------------------------------------------------- run big_sim with safety
def main():
    import big_sim                                           # unchanged; we wrap two of its functions
    guard = Guard()
    launch, shown, rendered_labels = big_sim.launch_garage, {}, {}

    def launch_with_equipment(tools):
        env = launch(tools)
        guard.attach(env)
        return env

    def robot_of(actor):
        return actor.robot if hasattr(actor, "robot") else actor.arm.robot

    def play(env, actors, movers, tracks, rendered, speed=big_sim.PLAYBACK_SPEED):
        if not guard.homes:
            guard.homes = {name: robot_of(actor).home_q for name, actor in actors.items()}

        def show(name, frame):
            q, poses, label, width = frame
            if label != rendered_labels.get(name):
                print(f"  - {name}: {label}")
                rendered_labels[name] = label
            actors[name].show(q, width)
            for obj, M in poses.items():
                prev = rendered.get(obj)
                if prev is None or not np.array_equal(prev, M):
                    movers[obj].set_pose(SE3(M, check=False))
                    rendered[obj] = M

        return guard.play(tracks, show, lambda name, q: big_sim.arm_points(robot_of(actors[name]), q),
                          big_sim.DT, speed, big_sim.RENDER_FPS)

    big_sim.launch_garage, big_sim.play = launch_with_equipment, play
    big_sim.main()


if __name__ == "__main__":
    main()
