# Control Loop Health Monitor

A diagnostic tool that screens industrial control loops for two of the
most common causes of poor performance: **valve stiction** and
**controller tuning issues** — using only the OP (controller output)
and PV (process variable) data that every DCS/PLC historian already
records.

## Background

Published control-engineering research shows that valve stiction alone
causes 20-30% of oscillating control loops in the process industry
(Jelali & Huang, 2010). Manually checking every loop in a plant with
hundreds of them is impractical — this tool automates the first-pass
screening so engineers know which loops to physically inspect first.

## How it works

1. **`generate_data.py`** — simulates realistic closed-loop behavior
   (a real PID controller reacting to error, with a physics-based
   two-parameter stiction model — stiction band `S` and slip jump `J`
   — sitting between the controller output and the actual valve
   position). Produces healthy, sticky-valve, and badly-tuned example
   loops.

2. **`detect.py`** — the diagnostic logic:
   - Flags a loop as healthy or showing a performance issue, based on
     oscillation amplitude during steady-state operation (periods
     right after a setpoint change are excluded, so normal
     setpoint-tracking isn't mistaken for a fault).
   - For flagged loops, gives a **pattern hint**: a jagged,
     stuck-then-jump PV trace suggests valve stiction; a smooth,
     sustained oscillation suggests a controller tuning issue. This
     mirrors published plant-engineering guidance and is presented as
     a guideline, not a certain diagnosis — the same way it's used in
     real practice.

3. **`app.py`** — a Streamlit dashboard: upload a CSV (or use the
   built-in sample data generator), see a prioritized table of loop
   health across the whole file, and drill into any loop's trend chart
   and OP-vs-PV cross-plot.

## Tech stack

Python, Pandas, NumPy, Streamlit, Matplotlib

## Running it locally

```bash
pip install -r requirements.txt
stream lit run app.py
