"""Operator GUI for the workcell: job control, safety, and an advanced teach / jog interface.

Author: [Your name]

Run:   python gui.py      (all project files sit in the same folder)

  Left   - system state banner, E-stop / Reset / Start-Resume / Pause / Abort, the physical PLC e-stop
           connection, and the simulated sensors (operator in zone, light curtain, test object)
  Middle - job set-up (hardware, job, plan, start) and the fault / event log
  Right  - teach / jog: pick a robot, jog each joint or the tool along x, y, z, see joint angles,
           distance to each limit and the tool position

Nothing in here loops or sleeps: Tk's timer calls Workcell.tick() every 40 ms.
Educational demo only - not a certified safety system.
"""
from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import big_sim as B
import chain_layout as C
from estop_input import ModbusEStop, NoPlcEStop
from workcell import Workcell

TICK_MS = 40
REPEAT_MS = 80              # jog buttons repeat this often while held
JOINT_STEPS_DEG = ["0.5", "1", "2", "5", "10"]
CART_STEPS_MM = ["2", "5", "10", "20", "50"]
HARDWARE = {"No nut and bolt": None, "Small nut + small bolt": ("small", "small"),
            "Small nut + large bolt": ("small", "large"), "Large nut + small bolt": ("large", "small"),
            "Large nut + large bolt": ("large", "large")}
DISCLAIMER = ("Educational demo - not a certified safety system. It does not replace the laboratory's "
              "approved risk controls or the hard-wired e-stop circuit.")


class App:
    def __init__(self, root, workcell):
        self.root, self.wc = root, workcell
        self.messages: queue.Queue = queue.Queue()      # from the planning thread
        self.planning = False
        self.launched = False
        self.repeat_id = None
        self.jog_widgets = []
        root.title("Workcell - teach / jog and safety")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<space>", lambda e: self.estop())
        self._build()
        root.after(TICK_MS, self.on_tick)

    # ------------------------------------------------------------ layout
    def _build(self):
        root = self.root
        self.banner = tk.Label(root, text="READY", font=("Segoe UI", 22, "bold"), pady=8)
        self.banner.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.detail = tk.Label(root, text="", font=("Segoe UI", 11), anchor="w", padx=8)
        self.detail.grid(row=1, column=0, columnspan=3, sticky="ew")
        left = ttk.Frame(root, padding=6)
        mid = ttk.Frame(root, padding=6)
        right = ttk.Frame(root, padding=6)
        left.grid(row=2, column=0, sticky="n")
        mid.grid(row=2, column=1, sticky="nsew")
        right.grid(row=2, column=2, sticky="n")
        root.columnconfigure(1, weight=1)
        tk.Label(root, text=DISCLAIMER, fg="#555", font=("Segoe UI", 8)).grid(row=3, column=0, columnspan=3)
        self._safety_panel(left)
        self._job_panel(mid)
        self._jog_panel(right)

    def _safety_panel(self, f):
        box = ttk.LabelFrame(f, text="Safety", padding=6)
        box.pack(fill="x")
        tk.Button(box, text="E-STOP  (space bar)", bg="#cf222e", fg="white", font=("Segoe UI", 14, "bold"),
                  command=self.estop).pack(fill="x", pady=2)
        ttk.Button(box, text="Release on-screen e-stop", command=self.wc.release_sim_estop).pack(fill="x", pady=1)
        ttk.Button(box, text="1. Reset / acknowledge", command=self.reset).pack(fill="x", pady=1)
        ttk.Button(box, text="2. Start / Resume", command=self.resume).pack(fill="x", pady=1)
        ttk.Button(box, text="Pause", command=self.wc.pause).pack(fill="x", pady=1)
        ttk.Button(box, text="Abort job and go to ready poses", command=self.abort).pack(fill="x", pady=1)

        plc = ttk.LabelFrame(f, text="Physical e-stop (PLC, Modbus TCP)", padding=6)
        plc.pack(fill="x", pady=6)
        self.plc_vars = {"host": tk.StringVar(value="192.168.0.10"), "port": tk.StringVar(value="502"),
                         "address": tk.StringVar(value="0"), "unit": tk.StringVar(value="1")}
        for row, (key, label) in enumerate((("host", "PLC address"), ("port", "Port"), ("address", "Input / coil no."),
                                            ("unit", "Unit id"))):
            ttk.Label(plc, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(plc, textvariable=self.plc_vars[key], width=14).grid(row=row, column=1)
        self.active_low = tk.BooleanVar(value=True)
        ttk.Checkbutton(plc, text="Input reads 0 when pressed (normally-closed contact)",
                        variable=self.active_low).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Button(plc, text="Connect", command=self.connect_plc).grid(row=5, column=0, sticky="ew")
        ttk.Button(plc, text="Disconnect", command=self.disconnect_plc).grid(row=5, column=1, sticky="ew")
        self.plc_label = tk.Label(plc, text="not connected", font=("Segoe UI", 10, "bold"))
        self.plc_label.grid(row=6, column=0, columnspan=2, sticky="ew", pady=3)

        sens = ttk.LabelFrame(f, text="Simulated sensors and test object", padding=6)
        sens.pack(fill="x")
        for text, level in (("Operator approaches (warning zone: slow)", 1), ("Operator breaks the light curtain", 2),
                            ("Operator leaves", 0)):
            ttk.Button(sens, text=text, command=lambda v=level: self.wc.post("zone", v)).pack(fill="x", pady=1)
        ttk.Button(sens, text="Place test object in the robot's path", command=lambda: self.wc.post("obstacle_path")
                   ).pack(fill="x", pady=(6, 1))
        ttk.Button(sens, text="Remove test object", command=lambda: self.wc.post("obstacle_remove")).pack(fill="x", pady=1)
        self.random_var = tk.BooleanVar()
        ttk.Checkbutton(sens, text="Random operator intrusions", variable=self.random_var,
                        command=self.toggle_random).pack(anchor="w")

    def _job_panel(self, f):
        setup = ttk.LabelFrame(f, text="Job", padding=6)
        setup.pack(fill="x")
        self.hw_var = tk.StringVar(value=list(HARDWARE)[0])
        self.job_var = tk.StringVar(value=list(C.JOBS)[0])
        ttk.Label(setup, text="Nut and bolt").grid(row=0, column=0, sticky="w")
        self.hw_box = ttk.Combobox(setup, textvariable=self.hw_var, values=list(HARDWARE), state="readonly", width=24)
        self.hw_box.grid(row=0, column=1, sticky="ew")
        ttk.Label(setup, text="Real robot job").grid(row=1, column=0, sticky="w")
        self.job_box = ttk.Combobox(setup, textvariable=self.job_var, values=list(C.JOBS), state="readonly", width=24)
        self.job_box.grid(row=1, column=1, sticky="ew")
        self.plan_btn = ttk.Button(setup, text="Plan job (opens Swift the first time)", command=self.plan)
        self.plan_btn.grid(row=2, column=0, columnspan=2, sticky="ew", pady=3)
        self.start_btn = ttk.Button(setup, text="Start job", command=self.start_job, state="disabled")
        self.start_btn.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.plan_label = ttk.Label(setup, text="Choose the job, then plan it.", wraplength=330)
        self.plan_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=3)
        self.step_label = ttk.Label(setup, text="", wraplength=330)
        self.step_label.grid(row=5, column=0, columnspan=2, sticky="w")
        self.progress = ttk.Progressbar(setup, maximum=1.0)
        self.progress.grid(row=6, column=0, columnspan=2, sticky="ew", pady=2)
        self.speed_label = ttk.Label(setup, text="")
        self.speed_label.grid(row=7, column=0, columnspan=2, sticky="w")

        faults = ttk.LabelFrame(f, text="Faults and events", padding=6)
        faults.pack(fill="both", expand=True, pady=6)
        self.fault_label = tk.Label(faults, text="No faults", fg="#1a7f37", anchor="w", justify="left",
                                    wraplength=330, font=("Segoe UI", 10, "bold"))
        self.fault_label.pack(fill="x")
        self.log = tk.Text(faults, height=16, width=48, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)

    def _jog_panel(self, f):
        box = ttk.LabelFrame(f, text="Teach / jog", padding=6)
        box.pack(fill="x")
        self.robot_var = tk.StringVar(value="UR3e")
        ttk.Label(box, text="Robot").grid(row=0, column=0, sticky="w")
        robot_box = ttk.Combobox(box, textvariable=self.robot_var, values=list(self.wc.robots), state="readonly", width=12)
        robot_box.grid(row=0, column=1, columnspan=2, sticky="w")
        self.jog_note = ttk.Label(box, text="", foreground="#a40", wraplength=300)
        self.jog_note.grid(row=1, column=0, columnspan=4, sticky="w")
        self.jstep = tk.StringVar(value="2")
        self.cstep = tk.StringVar(value="10")
        ttk.Label(box, text="Joint step (deg)").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Combobox(box, textvariable=self.jstep, values=JOINT_STEPS_DEG, width=6, state="readonly").grid(row=2, column=2)

        self.joint_rows = []
        for j in range(6):
            r = 3 + j
            name = ttk.Label(box, text=f"J{j + 1}", width=3)
            value = ttk.Label(box, text="0.0", width=8, font=("Consolas", 10))
            bar = ttk.Progressbar(box, maximum=1.0, length=70)
            minus = ttk.Button(box, text="-", width=3)
            plus = ttk.Button(box, text="+", width=3)
            name.grid(row=r, column=0)
            value.grid(row=r, column=1)
            bar.grid(row=r, column=2, padx=2)
            minus.grid(row=r, column=3)
            plus.grid(row=r, column=4)
            self._hold(minus, lambda j=j: self.jog_joint(j, -1))
            self._hold(plus, lambda j=j: self.jog_joint(j, +1))
            self.joint_rows.append((value, bar))
            self.jog_widgets += [minus, plus]

        ttk.Label(box, text="Tool step (mm)").grid(row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Combobox(box, textvariable=self.cstep, values=CART_STEPS_MM, width=6, state="readonly").grid(row=9, column=2, pady=(8, 0))
        for i, axis in enumerate("XYZ"):
            r = 10 + i
            ttk.Label(box, text=axis, width=3).grid(row=r, column=0)
            minus = ttk.Button(box, text=f"{axis} -", width=5)
            plus = ttk.Button(box, text=f"{axis} +", width=5)
            minus.grid(row=r, column=1)
            plus.grid(row=r, column=2)
            self._hold(minus, lambda a=i: self.jog_cartesian(a, -1))
            self._hold(plus, lambda a=i: self.jog_cartesian(a, +1))
            self.jog_widgets += [minus, plus]
        self.tcp_label = ttk.Label(box, text="tool x y z:", font=("Consolas", 10))
        self.tcp_label.grid(row=13, column=0, columnspan=5, sticky="w", pady=4)
        grip = ttk.Frame(box)
        grip.grid(row=14, column=0, columnspan=5, sticky="w")
        self.grip_open = ttk.Button(grip, text="Open TransferBot gripper", command=lambda: self.wc.set_gripper(B.OPEN_WIDTH))
        self.grip_close = ttk.Button(grip, text="Close", command=lambda: self.wc.set_gripper(B.CLOSED_WIDTH))
        self.grip_open.pack(side="left")
        self.grip_close.pack(side="left", padx=4)
        self.home_btn = ttk.Button(box, text="Send this robot to its ready pose", command=self.go_home)
        self.home_btn.grid(row=15, column=0, columnspan=5, sticky="ew", pady=4)
        self.jog_widgets += [self.grip_open, self.grip_close, self.home_btn]

    # ------------------------------------------------------------ hold-to-repeat buttons (timer based, no loop)
    def _hold(self, button, action):
        def press(_):
            self._repeat(action)

        def release(_):
            if self.repeat_id is not None:
                self.root.after_cancel(self.repeat_id)
                self.repeat_id = None
        button.bind("<ButtonPress-1>", press)
        button.bind("<ButtonRelease-1>", release)

    def _repeat(self, action):
        action()
        self.repeat_id = self.root.after(REPEAT_MS, lambda: self._repeat(action))

    # ------------------------------------------------------------ actions
    def jog_joint(self, j, sign):
        self.wc.jog_joint(self.robot_var.get(), j, sign * np.deg2rad(float(self.jstep.get())))

    def jog_cartesian(self, axis, sign):
        self.wc.jog_cartesian(self.robot_var.get(), axis, sign * float(self.cstep.get()) / 1000.0)

    def go_home(self):
        self.wc.go_home(self.robot_var.get())

    def estop(self):
        self.wc.press_sim_estop()

    def reset(self):
        self.wc.reset()

    def resume(self):
        ok, msg = self.wc.start_resume()
        if not ok:
            self.wc.faults.append(msg)

    def abort(self):
        ok, msg = self.wc.abort_job()
        if not ok:
            self.wc.faults.append(msg)

    def start_job(self):
        ok, msg = self.wc.start_job(self.job_var.get())
        if not ok:
            self.wc.faults.append(msg)

    def toggle_random(self):
        if self.random_var.get():
            self.wc.start_random_intrusions()
        else:
            self.wc.stop_random_intrusions()

    def connect_plc(self):
        self.disconnect_plc()
        try:
            self.wc.plc_estop = ModbusEStop(
                self.plc_vars["host"].get(), int(self.plc_vars["port"].get()), int(self.plc_vars["address"].get()),
                int(self.plc_vars["unit"].get()), active_low=self.active_low.get())
        except Exception as exc:
            self.wc.faults.append(f"PLC connection failed: {exc}")

    def disconnect_plc(self):
        close = getattr(self.wc.plc_estop, "close", None)
        if close:
            close()
        self.wc.plc_estop = NoPlcEStop()

    # --- planning runs in a worker thread so the window stays responsive (and the e-stop stays live)
    def plan(self):
        if self.planning:
            return
        if self.launched and not self.wc.controller.jog_allowed:
            self.wc.faults.append("Finish or abort the running job first")
            return
        self.planning = True
        self.plan_btn.state(["disabled"])
        job, hardware = self.job_var.get(), HARDWARE[self.hw_var.get()]
        first = not self.launched
        post = self.messages.put

        def work():
            try:
                if first:
                    sizes = None if hardware is None else {"nut": hardware[0], "bolt": hardware[1]}
                    if not self.wc.plan_hardware(sizes, post):
                        post(("done", "SupplyBot could not plan its motions"))
                        return
                error = self.wc.plan_job(job, post)
                post(("done", error))
            except Exception as exc:                      # show the problem instead of dying silently
                post(("done", f"{type(exc).__name__}: {exc}"))
        threading.Thread(target=work, name="planner", daemon=True).start()

    def _plan_finished(self, error):
        self.planning = False
        self.plan_btn.state(["!disabled"])
        if error:
            self.plan_label.config(text=f"Planning failed: {error}")
            return
        if not self.launched:
            self.wc.launch()                              # Swift must be started from this (main) thread
            self.launched = True
            self.hw_box.state(["disabled"])
        self.start_btn.state(["!disabled"])
        self.plan_label.config(text=f"'{self.job_var.get()}' is planned. Press Start job.")

    # ------------------------------------------------------------ the tick (timer driven)
    def on_tick(self):
        try:
            if self.launched:
                self.wc.tick()
            else:                                         # e-stop inputs are still watched before Swift is up
                self.wc._poll_inputs()
        except Exception as exc:
            self.wc.faults.append(f"Internal error: {type(exc).__name__}: {exc}")
        while True:
            try:
                kind, *rest = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "done":
                self._plan_finished(rest[0])
            else:
                self.plan_label.config(text=str(kind))
        try:
            self.refresh(self.wc.status())
        except Exception as exc:
            self.wc.faults.append(f"Display error: {type(exc).__name__}: {exc}")
        self.root.after(TICK_MS, self.on_tick)

    def refresh(self, s):
        self.banner.config(text=s["state"], fg=s["fg"], bg=s["bg"])
        self.detail.config(text=s["detail"] or " ")
        # PLC e-stop
        plc = s["plc"]
        if plc.text == "not connected":
            self.plc_label.config(text="PLC e-stop: not connected", bg="#d0d7de", fg="black")
        elif not plc.healthy:
            self.plc_label.config(text=f"PLC e-stop: {plc.text}", bg="#cf222e", fg="white")
        elif plc.pressed:
            self.plc_label.config(text="PLC e-stop: PRESSED", bg="#cf222e", fg="white")
        else:
            self.plc_label.config(text="PLC e-stop: released (healthy)", bg="#1a7f37", fg="white")
        # job
        self.step_label.config(text=s["step"])
        self.progress["value"] = s["progress"]
        self.speed_label.config(text=f"Speed: {s['speed_scale'] * 100:.0f}%" + ("   (safe speed - operator in warning zone)"
                                                                               if s["zone"] == 1 else ""))
        if not s["swift_ok"]:
            self.detail.config(text="Swift is not responding - keep its browser tab open")
        # faults
        faults = s["faults"][-4:]
        if faults:
            self.fault_label.config(text="\n".join(faults), fg="#cf222e")
        else:
            self.fault_label.config(text="No faults", fg="#1a7f37")
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        import time as _t
        for stamp, text in s["log"]:
            self.log.insert("end", f"{_t.strftime('%H:%M:%S', _t.localtime(stamp))}  {text}\n")
        self.log.see("end")
        self.log.config(state="disabled")
        # jog panel
        name = self.robot_var.get()
        info = s["robots"][name]
        for j, (value, bar) in enumerate(self.joint_rows):
            value.config(text=f"{np.rad2deg(info['q'][j]):7.1f}°")
            bar["value"] = float(np.clip(info["margin"][j] / (info["span"][j] / 2), 0, 1)) if info["span"][j] else 0
        x, y, z = info["tcp"]
        self.tcp_label.config(text=f"tool  x {x:6.3f}  y {y:6.3f}  z {z:6.3f} m")
        allowed = s["jog_allowed"] and self.launched
        for w in self.jog_widgets:
            w.state(["!disabled"] if allowed else ["disabled"])
        if allowed:
            self.jog_note.config(text="")
        elif not self.launched:
            self.jog_note.config(text="Jogging is available once the scene is open (plan a job first).")
        else:
            self.jog_note.config(text="Jogging is disabled: a job is active or the system is stopped.")

    def close(self):
        self.wc.stop_random_intrusions()
        self.disconnect_plc()
        self.root.destroy()
        sys.exit(0)


def main():
    if not B.check_files():
        return
    wc = Workcell()
    root = tk.Tk()
    App(root, wc)
    root.mainloop()


if __name__ == "__main__":
    main()
