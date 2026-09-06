"""
generate_data.py

Simulates REAL closed-loop control behavior (not just canned curves):

    SP --> [PID controller] --> OP --> [valve, possibly sticky] -->
    actual valve position --> [process, first-order lag] --> PV --> back to controller

This is what makes the three scenarios below physically honest:

  1. Healthy loop   -> normal PID + healthy valve (OP follows PV cleanly)
  2. Sticky valve    -> normal PID + a valve with STICTION (stuck-then-jump)
                        sitting between OP and the real valve position.
                        The controller keeps "fighting" the stuck valve,
                        which is what produces a genuine, self-sustaining
                        limit-cycle oscillation -- the classic stiction
                        signature, not a hand-drawn wobble.
  3. Bad tuning     -> normal (healthy) valve, but the PID gains are set
                        too aggressive, so the controller itself overreacts
                        and creates a smooth, sustained oscillation even
                        though the valve is moving exactly as told.

Stiction model: two-parameter model used in control literature --
    S = stiction band  (how far OP must move before the valve breaks free)
    J = slip jump       (how far the valve jumps once it breaks free)
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)


def _apply_stiction_step(op_now, stem_prev, last_moved_op, S, J):
    """One step of the stiction model. Returns (new_stem, new_last_moved_op)."""
    delta = op_now - last_moved_op
    if abs(delta) < S:
        return stem_prev, last_moved_op          # stuck
    direction = np.sign(delta)
    new_stem = op_now + direction * J             # breaks free, overshoots by J
    return new_stem, op_now


def simulate_loop(tag, n=300, setpoint=50.0, tau=4.0, dt=1.0,
                   Kp=0.6, Ki=0.12, noise=0.08, dead_time=2,
                   stiction=False, S=5.0, J=4.0,
                   bad_tuning=False):
    """
    Runs a real step-by-step closed-loop simulation.

    dead_time matters more than it looks: a pure first-order lag process
    is very forgiving of aggressive PID gains (it just settles faster).
    Real plant loops oscillate from bad tuning because every real process
    has some TRANSPORT DELAY (dead time) -- e.g. it takes time for hot
    fluid to physically travel from the valve to the temperature sensor.
    Aggressive gains + dead time is what actually causes instability, so
    we model that delay explicitly here.
    """
    sp = np.full(n, setpoint, dtype=float)
    sp[int(n * 0.27):int(n * 0.53)] += 10   # a setpoint change partway through
    sp[int(n * 0.53):] -= 5

    if bad_tuning:
        # over-aggressive gains -> combined with dead time, causes sustained oscillation
        Kp, Ki = 2.2, 0.55

    op = np.zeros(n)
    pv = np.zeros(n)
    stem = np.zeros(n)          # actual valve position (may differ from OP if sticky)

    pv[0] = setpoint
    stem[0] = setpoint
    integral = 0.0
    last_moved_op = setpoint

    for i in range(1, n):
        error = sp[i] - pv[i - 1]
        integral += error * dt
        raw_op = Kp * error + Ki * integral
        op[i] = np.clip(raw_op, -20, 120)   # a real valve can't exceed 0-100% travel

        if stiction:
            stem[i], last_moved_op = _apply_stiction_step(op[i], stem[i - 1], last_moved_op, S, J)
        else:
            stem[i] = op[i]

        # process sees the valve position from `dead_time` steps ago
        delayed_stem = stem[max(0, i - dead_time)]

        # process: first-order lag responding to the (delayed) valve position
        alpha = dt / (tau + dt)
        pv[i] = pv[i - 1] + alpha * (delayed_stem - pv[i - 1]) + RNG.normal(0, noise)

    op += RNG.normal(0, noise * 0.4, n)   # small measurement noise on OP too

    return pd.DataFrame({
        "Timestamp": np.arange(n),
        "Loop_Tag": tag,
        "OP": np.round(op, 2),
        "PV": np.round(pv, 2),
        "SP": np.round(sp, 2),
    })


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    loops = [
        simulate_loop("TIC-101", setpoint=60),                                   # healthy
        simulate_loop("LIC-402", setpoint=40),                                   # healthy
        simulate_loop("FIC-204", setpoint=50, stiction=True, S=6.0, J=6.0),      # severe stiction
        simulate_loop("PIC-305", setpoint=30, stiction=True, S=3.0, J=3.0),      # mild stiction
        simulate_loop("FIC-118", setpoint=45, bad_tuning=True),                  # bad tuning, healthy valve
    ]

    combined = pd.concat(loops, ignore_index=True)
    combined.to_csv("data/plant_loops_sample.csv", index=False)

    for df in loops:
        tag = df["Loop_Tag"].iloc[0]
        df.to_csv(f"data/{tag}.csv", index=False)

    print("Generated data/plant_loops_sample.csv with loops:",
          combined["Loop_Tag"].unique().tolist())
