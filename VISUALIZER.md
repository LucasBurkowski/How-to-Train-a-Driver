# Field Visualizer

A Python GUI for real-time visualization of simulated robots on the FRC field when the simulation runs **without a WPILib connection**.

## Overview

The field visualizer uses **Matplotlib** to display:
- **Robot positions** on the field (color-coded by alliance: red/blue)
- **Robot orientations** with heading arrows
- **Game piece status** (diamond indicator ◆ when possessed)
- **Step counter** for tracking simulation progress
- **Game pieces** scattered on the field (gold circles)

## Features

### FieldVisualizer Class
A persistent matplotlib figure that can be updated frame-by-frame:

```python
from frc_rl.sim.field_visualizer import FieldVisualizer
from frc_rl.sim.robot_sim import RobotSimulator

# Create visualizer
viz = FieldVisualizer(field_length=16.54, field_width=8.21)

# Run simulation loop
for step in range(500):
    # ... update robot simulators ...
    states = [robot.state for robot in all_robots]
    viz.render(states, step=step, game_pieces=game_pieces)

viz.close()
```

### visualize_frame() Function
Quick one-shot visualization for debugging or inspection:

```python
from frc_rl.sim.field_visualizer import visualize_frame

# Display a single frame
visualize_frame(robot_states, step=0, game_pieces=game_pieces)
```

## Usage

### Demo Script

Run the included demo to see the visualizer in action:

```bash
python scripts/visualizer_demo.py
```

This creates 6 robots (3 red, 3 blue) moving around the field in a simple pattern.

### Integration with Your Simulation

**Step 1: Import the visualizer**

```python
from frc_rl.sim.field_visualizer import FieldVisualizer
```

**Step 2: Create a visualizer instance**

```python
viz = FieldVisualizer(title="My Simulation")
```

**Step 3: Render each frame**

Inside your simulation loop:

```python
for step in range(num_steps):
    # ... advance simulation ...
    viz.render(
        robot_states=[robot.state for robot in robots],
        step=step,
        game_pieces=game_piece_positions,
    )
```

**Step 4: Clean up**

```python
viz.close()
```

## Robot Display

Each robot is displayed as:
- **Circle**: Robot body (red or blue depending on alliance)
- **Arrow**: Heading direction (white arrow pointing forward)
- **Label**: Robot ID (e.g., "R0") + "◆" if holding a game piece
- **Outline**: Black border for clarity

## Field Layout

- **Field boundary**: Black rectangle (16.54 m × 8.21 m for Reefscape 2025)
- **Grid**: Light gray grid overlay
- **Game pieces**: Gold circles (typically at intake zones or scoring positions)

## Configuration

The visualizer can be customized:

```python
viz = FieldVisualizer(
    field_length=16.54,    # metres
    field_width=8.21,      # metres
    title="Custom Title"
)
```

Internal constants can be modified in `src/frc_rl/sim/field_visualizer.py`:
- `ROBOT_SIZE`: Visual size of robot circles (default: 0.42 m)
- `ALLIANCE_COLORS`: Hex colors for red/blue alliances
- `ROBOT_OUTLINE_WIDTH`: Border thickness
- `TEXT_SIZE`: Font size for labels

## Performance Notes

- Rendering is optimized for interactive use (~100-500 fps typical)
- Each frame clears and redraws all robot elements
- For very high-frequency simulation (>1000 Hz), consider rendering every Nth frame:

```python
if step % 10 == 0:  # Render every 10 steps
    viz.render(states, step=step)
```

## Headless / Server Environments

The visualizer requires a display server. For headless environments:

1. **Use a virtual display** (e.g., Xvfb on Linux):
   ```bash
   xvfb-run -a python scripts/train.py
   ```

2. **Or disable rendering** by not creating a visualizer (it's optional)

3. **Or save frames to disk** by extending the visualizer to use `fig.savefig()` instead of `plt.show()`

## Dependencies

The visualizer requires:
- `matplotlib` (already in project dependencies)
- `numpy` (already in project dependencies)

No additional packages needed!

## Examples

### Example 1: Simple Visualization

```python
from frc_rl.sim.robot_sim import RobotSimulator
from frc_rl.sim.field_visualizer import FieldVisualizer
import math

# Create robots
robots = [
    RobotSimulator(robot_id=0, alliance="red", start_x=2, start_y=2, start_theta=0),
    RobotSimulator(robot_id=3, alliance="blue", start_x=14.5, start_y=2, start_theta=math.pi),
]

# Create visualizer
viz = FieldVisualizer()

# Simulate
for step in range(100):
    for robot in robots:
        robot.step(0.3, 0, 0.1, 16.54, 8.21)
    
    viz.render([r.state for r in robots], step=step)

viz.close()
```

### Example 2: Batch Frame Visualization

```python
from frc_rl.sim.field_visualizer import visualize_frame

# Display a pre-recorded state
states = [...]  # List of RobotState objects
pieces = [(8.27, 4.1), (8.27, 2.0)]

visualize_frame(states, step=42, game_pieces=pieces)
plt.show()
```

## Future Enhancements

Potential additions:
- Scoring zone visualization (colored regions)
- Intake zone indicators
- Velocity vectors (displayed as arrows)
- Score/status panel (live score display)
- Recording to video file
- Replay functionality (load and playback recorded states)
- Interactive controls (pause, step, speed adjustment)
