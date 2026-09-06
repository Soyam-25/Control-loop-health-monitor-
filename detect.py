"""
detect.py

Given OP/PV/SP data for a single control loop, decide whether it is:
  - Healthy
  - Performance issue detected -- with a pattern-based HINT at likely
    cause (valve stiction vs. controller tuning)

APPROACH (mirrors how a control engineer actually reasons about this,
and matches published guidance on loop diagnosis):

  Step 1: Is the loop oscillating at all during STEADY-STATE operation?
      We deliberately exclude the settling period right after a
      setpoint change -- normal setpoint tracking looks "oscillatory"
      for a few samples and shouldn't be confused with a real fault.
      Measured via how much PV wanders around its own rolling average.

  Step 2: If it IS oscillating, what does the oscillation shape look
      like -- JAGGED or SMOOTH?
      This is a real, published diagnostic guideline (Emerson/plant
      engineering literature): oscillation from a faulty valve tends
      to look like a square wave or sawtooth (sudden jumps interrupted
      by flat/stuck spells), while oscillation from poor tuning tends
      to look like a smooth, continuous sine wave. It's stated
      explicitly in the literature as a GUIDELINE, not a certain test
      -- so this tool reports it as a pattern hint, not a definitive
      verdict. The final call always belongs to the engineer who
      physically inspects the flagged loop.

      Measured with a "jerkiness-to-amplitude ratio": how big are the
      sudden sample-to-sample jumps in PV, relative to how much PV is
      swinging overall? A jagged, stuck-then-jump trace has a LOW
      ratio (long quiet spells then a big jump). A smooth sine-wave
      trace has a HIGH ratio (every step changes by a similar, modest
      amount).

Why not just fit a straight line to OP-vs-PV (the classic "cross-plot"
method)? Because any real process has some dead time (transport delay),
and dead time alone makes even a perfectly smooth, tuning-driven
oscillation trace a non-straight shape on that plot. So a simple
straight-line fit can't reliably tell stiction and tuning issues apart
by itself -- the jaggedness-based shape check is a cleaner, more direct
signature of an actually STUCK valve versus a continuously-moving one.
"""

import numpy as np
import pandas as pd


def _oscillation_amplitude(pv):
    """How much PV wanders around its own rolling average -- proxy for
    'is this loop oscillating at all'."""
    rolling_mean = pd.Series(pv).rolling(20, min_periods=1, center=True).mean()
    deviation = pv - rolling_mean.values
    return np.std(deviation)


def _jaggedness_ratio(pv):
    """
    Ratio of (average size of sudden jumps) to (overall swing amplitude).

    LOW ratio -> long quiet/stuck spells punctuated by sudden jumps
                 (jagged, square-wave-like)   -> valve stiction pattern
    HIGH ratio -> PV changes by a similar modest amount every step
                  (smooth, sine-wave-like)    -> tuning-issue pattern
    """
    d2 = np.abs(np.diff(pv, n=2))
    amplitude = np.std(pv)
    if amplitude < 1e-6:
        return 1.0
    return np.mean(d2) / amplitude


def _last_steady_block(sp, settle_samples=15):
    """
    Returns (start, end) indices of the LAST contiguous steady-state
    block -- the stretch of data after the final setpoint change has
    had time to settle. We use one contiguous block (not scattered
    steady-state points stitched together) because rolling-window
    metrics need a genuinely continuous time series; stitching
    non-adjacent samples together creates fake jumps at the seams.

    This also mirrors real practice: engineers pick a clean steady
    window of historian data to run loop diagnostics on, rather than
    mixing in transition periods.
    """
    sp = np.asarray(sp)
    changes = np.where(np.diff(sp) != 0)[0]
    last_change = changes[-1] if len(changes) else -1
    start = min(len(sp) - 1, last_change + settle_samples)
    return start, len(sp)


def classify_loop(df, jaggedness_threshold=0.20, osc_threshold=0.55):
    """
    df must have columns: OP, PV, SP
    Returns a dict with status, severity score, a pattern hint, and the
    raw metrics (so the dashboard can show WHY a loop was classified
    the way it was).
    """
    if "SP" in df.columns:
        start, end = _last_steady_block(df["SP"].values)
    else:
        start, end = 0, len(df)
    pv = df["PV"].values.astype(float)[start:end]

    osc = _oscillation_amplitude(pv)
    jag = _jaggedness_ratio(pv)

    if osc < osc_threshold:
        status = "Healthy"
        pattern_hint = "--"
        severity = round(max(0, (osc_threshold - osc)) * 2, 1)
    else:
        status = "Performance issue detected"
        severity = round(min(10, osc * 2), 1)
        if jag < jaggedness_threshold:
            pattern_hint = "Jagged / stuck-then-jump pattern -> valve stiction likely"
        else:
            pattern_hint = "Smooth, sustained oscillation -> controller tuning likely"

    return {
        "status": status,
        "severity": severity,
        "pattern_hint": pattern_hint,
        "oscillation": round(osc, 3),
        "jaggedness_ratio": round(jag, 3),
    }


def analyze_all_loops(df):
    """df must have columns: Loop_Tag, OP, PV, SP. Returns one row per loop tag."""
    results = []
    for tag, sub in df.groupby("Loop_Tag"):
        r = classify_loop(sub)
        r["Loop_Tag"] = tag
        results.append(r)

    out = pd.DataFrame(results)
    out = out[["Loop_Tag", "status", "severity", "pattern_hint", "oscillation", "jaggedness_ratio"]]
    out = out.sort_values("severity", ascending=False).reset_index(drop=True)
    return out
