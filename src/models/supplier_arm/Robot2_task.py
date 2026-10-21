"""Robot 2 task helpers: pick tool from the green mark, hang it on the board.

Author: Jake Salamon

Poses come from layout.py (world frame, metres, origin at the blue mark).
All moves use quintic time scaling, so every segment starts and ends at rest with no
sudden changes in speed. Tune the speeds below.
"""
from __future__ import annotations

import numpy as np
from spatialmath import SE3

from layout import (GRASP_Z, GREEN_MARK, HANG_ORIENT, HANG_Z, RETURN_TOOL,
                    SLOT_X, TOOL_THICKNESS, WALL_Y)

VERSION = "v9 - v8 + per-robot link clearance"

DT = 0.02            # simulation step (s)
JOINT_SPEED = 0.6    # average joint speed for joint-space moves (rad/s)
CART_SPEED = 0.08    # average tool speed for straight-line RMRC moves (m/s)
LINK_CLEARANCE = 0.085  # min gap between non-neighbouring links (m); links are ~0.045 radius

# Heights/positions the arm must stay clear of. Defaults suit the bench scene (bench top at
# z = 0, board at WALL_Y); call set_collision_limits() for a different scene.
_LIMITS = {"floor_z": 0.03, "wall_y": WALL_Y - 0.05}


def set_collision_limits(floor_z=None, wall_y=None):
    """Set the work-surface height and wall position that arm_collision() checks."""
    if floor_z is not None:
        _LIMITS["floor_z"] = floor_z
    if wall_y is not None:
        _LIMITS["wall_y"] = wall_y


# Pick from the green mark with the gripper pointing down; place on the board with the
# gripper pointing at the wall and the tool hanging vertically. See layout.py.
TABLE_PICK = SE3(GREEN_MARK[0], GREEN_MARK[1], GRASP_Z) * SE3.Rx(np.pi)
RACK_PLACE = SE3(SLOT_X[RETURN_TOOL], WALL_Y - TOOL_THICKNESS, HANG_Z) * HANG_ORIENT
APPROACH_OFFSET = SE3(0, 0, -0.10)  # back off 10 cm along the tool z-axis


def approach(T: SE3) -> SE3:
    """Pose 10 cm before T along the tool's own approach axis."""
    return T * APPROACH_OFFSET


def quintic(n: int) -> np.ndarray:
    """Smooth 0 -> 1 profile with zero velocity and acceleration at both ends."""
    t = np.linspace(0, 1, n)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _pose_reached(robot, q, T: SE3, pos_tol: float = 1e-3, ang_tol: float = 0.02) -> bool:
    """True if joint vector q really puts the end effector at T (position AND orientation).

    The orientation test uses the rotation angle between the two poses. (The skew-vector
    error used elsewhere is blind to a 180 degree rotation, which can pass as "solved".)
    """
    Tq = robot.fkine(q).A
    if np.linalg.norm(Tq[:3, 3] - T.A[:3, 3]) > pos_tol:
        return False
    Rd = T.A[:3, :3] @ Tq[:3, :3].T
    return float(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1))) <= ang_tol


def _polish_ik(robot, T: SE3, guess):
    """Bounded least-squares IK from guess (backup for when ikine_LM fails). Returns q or None."""
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    lo, hi = robot.qlim
    goal = T.A

    def residual(q):
        Tq = robot.fkine(q).A
        return np.r_[Tq[:3, 3] - goal[:3, 3], 0.3 * (Tq[:3, :3] - goal[:3, :3]).ravel()]

    sol = least_squares(residual, np.clip(guess, lo + 1e-6, hi - 1e-6), bounds=(lo, hi))
    q = np.asarray(sol.x, dtype=float)
    return q if _pose_reached(robot, q, T) else None


def _ik_candidates(robot, T: SE3, q0, tries: int = 40, seed: int = 1):
    """All distinct IK solutions found, best first (close to q0, away from joint limits)."""
    q0 = np.asarray(q0, dtype=float)
    lo, hi = robot.qlim
    rng = np.random.default_rng(seed)
    seeds = [q0] + [rng.uniform(0.8 * lo, 0.8 * hi) for _ in range(tries)]
    found = []
    for n, guess in enumerate(seeds + [None] * 5 * tries):
        if guess is None:                      # hard pose: nothing found yet, so try many more starts
            if found:
                break
            guess = rng.uniform(0.9 * lo, 0.9 * hi)
        sol = robot.ikine_LM(T, q0=guess, joint_limits=True)
        q = np.asarray(sol.q, dtype=float) if sol.success else None
        if q is None or not _pose_reached(robot, q, T):
            q = _polish_ik(robot, T, guess)
            if q is None:
                continue
        margin = np.minimum(q - lo, hi - q).clip(0)
        cost = np.linalg.norm(q - q0) + 3 * np.sum(np.exp(-margin / 0.26))
        found.append((cost, q))
    found.sort(key=lambda c: c[0])
    unique = []
    for _, q in found:
        if not any(np.allclose(q, u, atol=0.02) for u in unique):
            unique.append(q)
    return unique


def solve_ik(robot, T: SE3, q0: np.ndarray, tries: int = 40, seed: int = 1):
    """Best joint-limited IK solution (or None). Prefers solutions near q0 and away from limits."""
    candidates = _ik_candidates(robot, T, q0, tries, seed)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------- collision checks
def arm_points(robot, q) -> np.ndarray:
    """Joint-frame origins from the base to the flange, then the tool tip (8 points for 6 DoF)."""
    T = robot.base.A
    pts = [T[:3, 3].copy()]
    for i, link in enumerate(robot.links):
        T = T @ link.A(q[i]).A
        pts.append(T[:3, 3].copy())
    pts.append(np.asarray(robot.fkine(q).t, dtype=float))
    return np.array(pts)


def _segment_distance(p1, q1, p2, q2) -> float:
    """Shortest distance between segments p1-q1 and p2-q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e < 1e-12:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0, 1) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def arm_collision(robot, q, clearance: float | None = None) -> bool:
    """True if the arm hits itself, the bench or the wall at joint vector q.

    Links are treated as capsules (line segments). Neighbouring links are skipped, as
    are links separated only by a very short link (they always touch at the joint).
    """
    if clearance is None:                   # slimmer arms (UR3e) can set robot.link_clearance
        clearance = getattr(robot, "link_clearance", LINK_CLEARANCE)
    pts = arm_points(robot, q)
    segs = [(pts[i], pts[i + 1], float(np.linalg.norm(pts[i + 1] - pts[i])))
            for i in range(len(pts) - 1)]
    real = [i for i, sg in enumerate(segs) if sg[2] > 1e-6]
    for a in range(len(real)):
        for b in range(a + 1, len(real)):
            i, j = real[a], real[b]
            if j - i < 2 or sum(segs[k][2] for k in range(i + 1, j)) < clearance:
                continue
            if _segment_distance(segs[i][0], segs[i][1], segs[j][0], segs[j][1]) < clearance:
                return True
    joints = pts[1:-1]                      # skip the base and the tool tip
    if np.any(joints[:, 2] < _LIMITS["floor_z"]):  # into the work surface
        return True
    if np.any(joints[:, 1] > _LIMITS["wall_y"]):   # into the wall
        return True
    return False


def path_collision_free(robot, traj, step: int = 2) -> bool:
    """True if no sampled configuration along traj is in collision."""
    return not any(arm_collision(robot, q) for q in traj[::step])


def plan_joint_move(robot, T: SE3, q0, speed: float = JOINT_SPEED):
    """Joint-space move to pose T that avoids self/bench/wall collisions.

    Tries the IK solutions best-first and uses the first whose goal AND path are
    collision-free. Returns (trajectory, goal) or (None, None) if T is unreachable.
    If every solution collides, returns the best one with a warning.
    """
    candidates = _ik_candidates(robot, T, q0)
    if not candidates:
        return None, None
    for q_goal in candidates:
        traj = joint_move(q0, q_goal, speed=speed)
        if not arm_collision(robot, q_goal) and path_collision_free(robot, traj):
            return traj, q_goal
    print("Warning: every IK solution for this move collides - check layout.py / DH values.")
    return joint_move(q0, candidates[0], speed=speed), candidates[0]


def joint_move(q_start, q_goal, speed: float = JOINT_SPEED, dt: float = DT) -> np.ndarray:
    """Smooth joint-space trajectory; duration is set from the largest joint change."""
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    n = max(30, int(np.max(np.abs(q_goal - q_start)) / speed / dt))
    return q_start + np.outer(quintic(n), q_goal - q_start)


def hold(q, seconds: float, dt: float = DT) -> np.ndarray:
    """Stand still at joint vector q for a short pause (e.g. while gripping)."""
    return np.tile(np.asarray(q, float), (max(1, int(seconds / dt)), 1))


def pose_error(T_now: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
    """Pose error T_now -> T_ref as (dx, dy, dz, rx, ry, rz) in the WORLD frame.

    Matches the world-frame Jacobian from robot.jacob0(). (spatialmath's tr2delta gives
    the error in the end-effector frame, which sends the arm the wrong way here.)
    """
    dp = T_ref[:3, 3] - T_now[:3, 3]
    Rd = T_ref[:3, :3] @ T_now[:3, :3].T
    dr = 0.5 * np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]])
    return np.r_[dp, dr]


def numeric_jacobian(robot, q, h: float = 1e-6) -> np.ndarray:
    """World-frame 6 x n Jacobian by finite differences of fkine.

    Built from the same fkine() the rest of the code uses, so base and tool offsets
    are always included and the frames always match pose_error().
    """
    q = np.asarray(q, dtype=float)
    T0 = robot.fkine(q).A
    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        dq = q.copy()
        dq[i] += h
        J[:, i] = pose_error(T0, robot.fkine(dq).A) / h
    return J


def rmrc_line(robot, T_goal: SE3, q0, speed: float = CART_SPEED, dt: float = DT,
              damping: float = 0.03, quiet: bool = False) -> np.ndarray:
    """Straight-line Cartesian move using resolved-rate motion control (RMRC).

    The reference pose follows a quintic profile between the current pose and T_goal.
    Each step turns the pose error into joint motion with a damped least-squares
    inverse, so it stays stable near singularities. Joints are clamped to their limits
    (and a warning printed) instead of aborting the move.
    Returns the joint trajectory (steps x n).
    """
    q = np.array(q0, dtype=float)
    lo, hi = robot.qlim
    T_start = robot.fkine(q)
    dist = float(np.linalg.norm(T_goal.t - T_start.t))
    n = max(30, int(dist / speed / dt))
    s = quintic(n)
    out = [q.copy()]
    limit_hits = 0
    for i in range(n - 1):
        T_ref = T_start.interp(T_goal, float(s[i + 1]))
        delta = pose_error(robot.fkine(q).A, T_ref.A)  # world-frame pose error (3 pos, 3 rot)
        J = numeric_jacobian(robot, q)
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), delta)
        q_new = np.clip(q + dq, lo, hi)
        if not np.allclose(q_new, q + dq):
            limit_hits += 1
        q = q_new
        out.append(q.copy())
    end_error = float(np.linalg.norm(T_goal.t - robot.fkine(q).t))
    if quiet:
        return np.array(out)
    if limit_hits or end_error > 0.002:
        print(f"RMRC warning: {limit_hits} joint-limit hit(s), end error {end_error * 1000:.1f} mm")
    out = np.array(out)
    if not path_collision_free(robot, out):
        print("RMRC warning: this straight-line move passes through a collision")
    return out
