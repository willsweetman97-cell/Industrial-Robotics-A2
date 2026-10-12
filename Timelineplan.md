# Project Timeline – Pick-and-Place DoGoodBot

41013 Industrial Robotics – Lab Assignment 2 (35% of subject)
Plan written: Friday 2 October 2026. Final demo/viva: Friday 23 October 2026.

> The Canvas specification and any later announcements override this plan if dates differ.
> Items marked **TBC** need confirming with a tutor or on the Subject schedule.

## 1. Key deadlines

| # | Deliverable | Type | Due | Weight | Status |
|---|-------------|------|-----|--------|--------|
| 1 | Proposal | Group | 21:00 Fri 11 Sep 2026 | 2/35 | Done |
| 2.1 | Safety documentation (signed Risk Assessment + SWMS PDF) | Group | **21:00 Fri 2 Oct 2026** | 1/35 | Due today |
| 2.2 | Git commits + Spark+ feedback | Individual | **21:00 Fri 2 Oct 2026** | 2/35 | Due today |
| 3 | 60-second promotion video (YouTube, Unlisted) | Group | 21:00 Fri 16 Oct 2026 | 5/35 | Planned |
| 4 | Final system demonstration / group viva (online) | Individual mark | Fri 23 Oct 2026 | 15/35 | Planned |
| 5 | Final video (~3 min) | Group | 23:59 Fri 23 Oct 2026 | 10/35 | Planned |
| – | Final individual code viva | Individual | **TBC** (see Final Code Viva page) | Separate | Book a slot |

## 2. Team and ownership

| Member | Personal robot model | Main areas of ownership |
|--------|---------------------|-------------------------|
| Will Sweetman | Robot 3 | *(fill in – e.g. working robot, RMRC, safety state machine)* |
| Muhammad Suhaib Khan | Robot 2 | *(fill in)* |
| Jake Salamon | Robot 3 | Robot that returns tools to shelf|

Real robot: UR3E. PLC bonus: **[attempting / not attempting]**. RGB-D bonus: **[attempting / not attempting]**.

## 3. Phased plan

### Phase 0 – Today (Fri 2 Oct): submit progress items
- [X] Safety documentation PDF signed by all members and submitted (one group submission)
- [X] Every member has regular, attributable commits on `main`/feature branches
- [X] Every member completes Spark+ feedback in their own words
- [ ] Merge this timeline into `main`

### Phase 1 – Foundations (Sat 3 – Sun 11 Oct)
Goal: a runnable skeleton that every member can build on.
- [ ] `src/main.py` entry point launches Swift with an empty workcell
- [ ] Agree the shared world frame and base poses for all robots, tools and surfaces (document in `docs/`)
- [ ] Each member: kinematic parameters sourced/justified for their personal robot; tutor sign-off on the model
- [ ] Each member: first working model in `src/models/` (DH parameters, joint limits, correct `fkine`)
- [ ] Scene v1: tool rack, workpiece, work surfaces and robot bases placed
- [ ] E-stop state machine designed (states: RUNNING, STOPPED/FAULT, RESET-ACKNOWLEDGED, READY, RESUME) – event/timer driven, no busy-wait loops
- [ ] Real robot: confirm availability, induction status, booking times (drop-ins TBC)
- [ ] Code standard and `docs/CONTRIBUTIONS.md` filled in

**Exit check (Sun 11 Oct):** three models appear in Swift in the shared scene and can move to a joint configuration.

### Phase 2 – Core capability (Mon 12 – Sun 18 Oct)
Goal: each subsystem works on its own.
- [ ] Higher-fidelity visuals for each personal model (geometry, colour, proportions better than Assignment 1)
- [ ] IK and joint-space / Cartesian trajectories for all three robots
- [ ] RMRC for the cutting stroke (singularity/manipulability handling)
- [ ] Pick-and-place sequence: select tool → hand over → deliver to working robot (multiple poses)
- [ ] Support for a final joint state supplied during marking
- [ ] GUI: joint jogging, Cartesian x/y/z jogging, selected-robot switching, state and fault display
- [ ] Safety v1: simulated GUI e-stop with deliberate reset then separate resume
- [ ] Safety v1: light curtain / unsafe-zone asynchronous trigger
- [ ] Safety v1: collision checking against an object placed in the planned path (stop first, then re-plan/avoid)
- [ ] Safety equipment models placed to match the risk assessment
- [ ] Real robot: first supervised, low-speed motion from Python
- [ ] Physical e-stop input wired/read in `src/hardware/` (if PLC e-stop available)
- [ ] Capture raw footage (screen recordings + real robot) as features land

**Exit check (Sun 18 Oct):** end-to-end sequence runs in simulation without safety features; each safety feature demonstrable individually.

### Phase 3 – Promo video, integration and hardening (Mon 12 – Fri 16 Oct, overlaps Phase 2)
Promo video runs in parallel so it does not block development.
- [ ] Mon 12 Oct: script and storyboard for the exactly-60-second video
- [ ] Wed 14 Oct: record footage from whatever is working; narration/captions
- [ ] Thu 15 Oct: edit, check duration is exactly 60 s, upload to UTS student YouTube as Unlisted
- [ ] **Fri 16 Oct, before 21:00:** submit URL on Canvas and post the same URL in the A2 Teams channel

### Phase 4 – Integration, freeze and rehearsal (Mon 19 – Thu 22 Oct)
- [ ] Mon 19 Oct: **feature freeze** – bug fixes only after this point
- [ ] Mon 19 – Tue 20 Oct: integrate GUI, motion, safety and real robot into one run; fix coordinate frame mismatches
- [ ] Tue 20 Oct: record real-robot evidence video (backup for anything that cannot run live online); include physical e-stop demo
- [ ] Wed 21 Oct: full dress rehearsal of the demo against the marking criteria (section 4); each member rehearses explaining their own model and code
- [ ] Wed 21 Oct: draft final video script; record voice-over and any extra footage
- [ ] Thu 22 Oct: second rehearsal, fix remaining issues, tag a release (`v1.0-demo`) on `main`
- [ ] Thu 22 Oct: confirm tools, internet, cameras and screen-share work for all members

### Phase 5 – Submission week finish (Fri 23 Oct)
- [ ] Online final demonstration / group viva (all members present, faces shown, each speaks and demonstrates their own model)
- [ ] Final video edited, exported and submitted by 23:59 (aim to submit by early evening, not at the deadline)

### Phase 6 – After submission
- [ ] Book/attend final individual code viva (**date TBC**); each member reviews code they claim as their own, including tests, bugs and design decisions

## 4. Demo readiness checklist (15 marks)

| Criterion | Marks | Evidence to have ready |
|-----------|-------|------------------------|
| Individual model + integrated task | 10 | Each member explains their model, IK, trajectories, RMRC, GUI, real robot; techniques chosen purposefully |
| Simulated GUI e-stop | 1 | Immediate stop, visible state, reset, then separate resume, safe recovery |
| Physical e-stop | 1 | Hardware input halts real system; controlled recovery |
| Simulated safety sensor | 1 | Light curtain / unsafe-zone signal changes trajectory or state |
| Collision response | 1 | Deliberately introduced obstacle detected before impact; stop or avoid |
| Safety equipment modelling | 1 | Strategically placed models matching the risk assessment |

## 5. Final video checklist (10 marks)

- [ ] Development and learning story solving a problem with a novel robotic solution (1.5)
- [ ] Real robot controlled from Python and integrated into the system (1)
- [ ] Professional presentation and coherent storytelling (2)
- [ ] Explains 41013 learning outcomes: modelling, planning/control, safety, user interaction (4)
- [ ] Explains sensing that adds capability or improves safety (1)
- [ ] Evidence-based future work (0.5)

## 6. Bonus decisions

Bonuses can only recover marks lost within Assignment 2 (cap stays at 35%). Decide by **Mon 12 Oct** so the work and safety approval fit the schedule.

- **PLC (up to 2):** needs a real PLC-driven sequence, an additional sensor that changes the sequence, a PLC/Python handshake with timeouts, and sensor/e-stop work included in the approved Risk Assessment and SWMS *before* energising the cell.
- **RGB-D (up to 2):** needs real RGB-D data that changes robot behaviour (calibration, mapping, recognition), not just a displayed stream.

Decision: **[PLC: yes/no] [RGB-D: yes/no]**

## 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Real robot booking or induction delays | Book early; keep a recorded backup of real-robot runs |
| Personal model not approved or inaccurate | Tutor sign-off in Phase 1; share sources in `docs/` |
| Frame mismatches between robots and objects | Single world-frame document; test transforms early |
| Integration late in the schedule | Weekly integration check; feature freeze on 19 Oct |
| Uneven commit history | Small, regular commits; follow the per-module author header |
| E-stop restarts unexpectedly | Test state machine against the reset-then-resume requirement before any hardware use |
| Online demo failure | Rehearse twice; have pre-recorded evidence ready |

## 8. Working agreements

- Commit at least every working session; branch per feature, merge via pull request into `main`
- Do not commit virtual environments, installed toolboxes, caches or third-party packages
- Weekly check-in to review the exit checks above and update the status boxes
- Only claim code you can explain in the viva
