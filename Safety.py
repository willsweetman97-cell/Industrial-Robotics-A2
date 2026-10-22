"""Safety state machine for the workcell (teaching/prototyping demo, NOT a certified safety system).

Author: Muhammad / William

The rules (all event driven - nothing here loops or waits):
  * An e-stop (simulated button OR the physical PLC mushroom button), a broken light curtain, or a
    predicted collision with an object in a planned path STOPS all motion at once.
  * The stop is LATCHED. Releasing the e-stop / clearing the zone / removing the object does NOT
    restart anything.
  * The operator must press Reset (acknowledge). Reset is refused while any cause is still present.
  * After Reset the system is STOPPED (safe, holding position). A separate Start / Resume press is
    needed to move again.
  * An operator in the outer warning zone only slows the robots (safe speed).
"""
from __future__ import annotations

import time
from enum import Enum


class State(Enum):
    READY = "READY"                      # idle, nothing running; teach / jog allowed
    RUNNING = "RUNNING"                  # job in progress
    PAUSED = "PAUSED"                    # operator pause (no reset needed to resume)
    PROTECTIVE_STOP = "PROTECTIVE STOP"  # light curtain / predicted collision; latched
    ESTOP = "EMERGENCY STOP"             # e-stop; latched
    STOPPED = "STOPPED - reset done"     # reset acknowledged; waiting for Start / Resume


BANNER = {  # state -> (text colour, background colour)
    State.READY: ("white", "#1f6feb"),
    State.RUNNING: ("white", "#1a7f37"),
    State.PAUSED: ("black", "#d4a72c"),
    State.PROTECTIVE_STOP: ("black", "#fb8500"),
    State.ESTOP: ("white", "#cf222e"),
    State.STOPPED: ("black", "#d4a72c"),
}

SLOW_SCALE = 0.3          # safe speed while somebody is in the warning zone


class SafetyController:
    def __init__(self):
        self.state = State.READY
        self.job_active = False           # a job is loaded and not finished/aborted
        self.zone_level = 0               # 0 clear, 1 operator in warning zone, 2 light curtain broken
        self.estop_inputs: dict[str, bool] = {}     # source name -> True while pressed / unhealthy
        self.latched: dict[str, str] = {}           # cause key -> text (needs Reset)
        self.conditions = {}              # cause key -> callable() -> True while the cause is still present
        self.log: list[tuple[float, str]] = []
        self.on_stop = None               # optional callback(cause text) when motion is halted

    # ------------------------------------------------------------ helpers
    def _note(self, text):
        self.log.append((time.time(), text))
        del self.log[:-200]

    def _latch(self, key, text, condition=None):
        new = key not in self.latched
        self.latched[key] = text
        if condition is not None:
            self.conditions[key] = condition
        self.state = State.ESTOP if any(k.startswith("estop") for k in self.latched) else State.PROTECTIVE_STOP
        if new:
            self._note(f"STOP: {text}")
            if self.on_stop:
                self.on_stop(text)

    def cause_present(self, key):
        if key.startswith("estop:"):
            return self.estop_inputs.get(key[6:], False)
        if key == "zone":
            return self.zone_level >= 2
        fn = self.conditions.get(key)
        return bool(fn()) if fn else False

    # ------------------------------------------------------------ inputs (call from any event handler)
    def set_estop(self, source, active, text=None):
        """Report an e-stop source (called every poll; only changes matter)."""
        was = self.estop_inputs.get(source, False)
        self.estop_inputs[source] = bool(active)
        if active and not was:
            self._latch(f"estop:{source}", text or f"E-stop pressed ({source})")
        elif was and not active:
            self._note(f"{source} e-stop released - system stays stopped until Reset, then Start")

    def set_zone(self, level):
        """0 clear, 1 operator in warning zone (slow), 2 light curtain broken (stop)."""
        if level == self.zone_level:
            return
        self.zone_level = level
        self._note({0: "Operator zone clear", 1: "Operator in warning zone - safe speed",
                    2: "Light curtain broken"}[level])
        if level >= 2:
            self._latch("zone", "Light curtain broken - operator in the robot zone")

    def collision_predicted(self, text, still_present):
        """A planned path is blocked. still_present() is asked again when Reset is pressed."""
        self._latch("obstacle", text, condition=still_present)

    # ------------------------------------------------------------ operator actions
    def reset(self):
        """Acknowledge the stop. Returns (ok, message)."""
        if not self.latched:
            return False, "Nothing to reset"
        present = [t for k, t in self.latched.items() if self.cause_present(k)]
        if present:
            return False, "Cannot reset yet - still present: " + "; ".join(present)
        self.latched.clear()
        self.conditions.clear()
        self.state = State.STOPPED if self.job_active else State.READY
        self._note("Reset acknowledged. Press Start / Resume to move again." if self.job_active
                   else "Reset acknowledged.")
        return True, "Reset done"

    def start_resume(self):
        """Separate start / resume action. Returns (ok, message)."""
        if self.latched:
            return False, "Reset the stop first"
        if self.state in (State.STOPPED, State.PAUSED) or (self.state is State.READY and self.job_active):
            self.state = State.RUNNING
            self._note("Running")
            return True, "Running"
        return False, f"Nothing to start (state: {self.state.value})"

    def begin_job(self):
        if self.latched or self.state in (State.ESTOP, State.PROTECTIVE_STOP):
            return False, "Reset the stop first"
        self.job_active = True
        self.state = State.RUNNING
        self._note("Job started")
        return True, "Job started"

    def pause(self):
        if self.state is State.RUNNING:
            self.state = State.PAUSED
            self._note("Paused by operator")

    def job_finished(self):
        self.job_active = False
        if not self.latched:
            self.state = State.READY
            self._note("Job finished")

    # ------------------------------------------------------------ queries
    @property
    def motion_enabled(self):
        return self.state is State.RUNNING

    @property
    def speed_scale(self):
        return SLOW_SCALE if self.zone_level == 1 else 1.0

    @property
    def jog_allowed(self):
        return self.state is State.READY and not self.job_active

    def banner(self):
        fg, bg = BANNER[self.state]
        detail = ""
        if self.latched:
            detail = "; ".join(self.latched.values())
        elif self.state is State.STOPPED:
            detail = "Press Start / Resume to continue"
        elif self.zone_level == 1 and self.state is State.RUNNING:
            detail = "Operator in warning zone - running at safe speed"
        return self.state.value, detail, fg, bg
