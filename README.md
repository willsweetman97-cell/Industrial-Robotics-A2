# Industrial-Robotics-A2
This project delivers a fully autonomous workshop cell where four robot arms replace the human technician entirely. The system is fully state driven, with live GUI monitoring, safe zone logic, collision avoidance, and e-stop recovery throughout.

 [Project Title] – Pick-and-Place DoGoodBot

41013 Robotics – Lab Assignment 2

## Group members

| Name | Student number | Personal robot model | GitHub username |
|------|----------------|----------------------|-----------------|
| [Name 1] | [Number] | [Robot model] | [username] |
| [Name 2] | [Number] | [Robot model] | [username] |
| [Name 3] | [Number] | [Robot model] | [username] |

## Project description

SafeCo is investigating small robotic systems for homes, offices and workplaces.
This project is a three-robot tool-handling workcell:

1. **Selector robot** – identifies which tool the task needs (e.g. a saw) and picks it from the tool rack.
2. **Delivery robot** – takes the tool from the selector and delivers it to the working robot.
3. **Working robot** – uses the tool to perform the task (e.g. sawing a piece of wood).

The robots share one simulated workcell built in Swift (Python). A real robot
([real robot name]) reproduces [part of the workflow] and is controlled from Python.

## Key features

- Shared Swift simulation with consistent coordinate frames across robots, tools and work surfaces
- One parameter-based robot model created by each group member
- Inverse kinematics, joint/Cartesian trajectories, and RMRC for the cutting stroke
- Python GUI with joint jogging, Cartesian x/y/z jogging, and state/fault display
- Simulated GUI e-stop and physical e-stop input, with deliberate reset and separate resume
- Integrated safety: light curtain / unsafe-zone sensing, collision detection and avoidance, keep-out zones, safety equipment models

## Repository structure

```
src/
  models/     Robot arm models (one per student, plus any supporting robots)
  scene/      Workcell environment, tools, workpieces, safety equipment
  motion/     IK, trajectories, RMRC, collision checking
  gui/        Teach/jog interface
  safety/     E-stop state machine, light curtain / unsafe-zone logic
  hardware/   Real-robot and PLC/e-stop interfaces
docs/         Design notes, diagrams, safety documentation
tests/        Tests
```

## Setup

Requires Python 3.10+ and the 41013 `ir_support` package provided by the subject
(not included in this repository).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install `ir_support` as instructed on the 41013 Canvas page.

## Running

```bash
python -m src.main        # [update once the main entry point exists]
```

## Code standard

See [`docs/CODE_STANDARD.md`](docs/CODE_STANDARD.md).

## Contributions

Each module lists its author in its file header. See [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md).
