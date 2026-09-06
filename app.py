"""
app.py

Control Loop Health Monitor -- Streamlit dashboard.

Upload a CSV of control loop data (Timestamp, Loop_Tag, OP, PV, SP) and
this tool flags which loops are healthy vs. showing a performance
problem, with a pattern-based hint at likely cause (valve stiction vs.
controller tuning). See detect.py for the diagnostic logic itself.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from detect import analyze_all_loops, classify_loop
from generate_data import simulate_loop

st.set_page_config(page_title="Control Loop Health Monitor", layout="wide")

st.title("Control loop health monitor")
st.caption(
    "Upload historian-style OP/PV/SP data to screen control loops for "
    "valve stiction and controller tuning issues."
)

with st.expander("What does this tool do? (read before uploading)"):
    st.markdown(
        """
        This tool looks for the two most common causes of poorly
        performing control loops in process plants:

        - **Valve stiction** -- the valve's stem sticks due to friction,
          then suddenly jumps once enough force builds up, causing a
          jagged, stuck-then-jump oscillation.
        - **Controller tuning issues** -- the controller gains are too
          aggressive for the process, causing a smooth, sustained
          oscillation even though the valve itself is healthy.

        It expects a CSV with these columns: `Timestamp, Loop_Tag, OP, PV, SP`.

        The pattern hint (jagged vs. smooth) is a **guideline**, the
        same way plant engineers use it in practice -- not a certain
        diagnosis. Always confirm with a physical inspection.
        """
    )

uploaded = st.file_uploader("Upload loop data (CSV)", type="csv")

if uploaded is None:
    st.info("Upload a CSV to get started, or try the sample data below.")
    if st.button("Load sample plant data"):
        loops = [
            simulate_loop("TIC-101", setpoint=60),
            simulate_loop("LIC-402", setpoint=40),
            simulate_loop("FIC-204", setpoint=50, stiction=True, S=6.0, J=6.0),
            simulate_loop("PIC-305", setpoint=30, stiction=True, S=3.0, J=3.0),
            simulate_loop("FIC-118", setpoint=45, bad_tuning=True),
        ]
        st.session_state["df"] = pd.concat(loops, ignore_index=True)
else:
    st.session_state["df"] = pd.read_csv(uploaded)

if "df" in st.session_state:
    df = st.session_state["df"]

    required_cols = {"Timestamp", "Loop_Tag", "OP", "PV"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        results = analyze_all_loops(df)

        st.subheader("Loop health summary")

        n_issues = (results["status"] != "Healthy").sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Loops analyzed", len(results))
        col2.metric("Issues detected", int(n_issues))
        col3.metric("Healthy", int(len(results) - n_issues))

        def _highlight(row):
            color = "" if row["status"] == "Healthy" else "background-color: #fdecea"
            return [color] * len(row)

        st.dataframe(
            results.style.apply(_highlight, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Inspect a loop")
        selected_tag = st.selectbox("Loop tag", results["Loop_Tag"].tolist())

        loop_df = df[df["Loop_Tag"] == selected_tag]
        loop_result = classify_loop(loop_df)

        st.markdown(
            f"**Status:** {loop_result['status']}  \n"
            f"**Severity score:** {loop_result['severity']} / 10  \n"
            f"**Pattern hint:** {loop_result['pattern_hint']}"
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        ax1.plot(loop_df["Timestamp"], loop_df["OP"], label="OP", linewidth=1.2)
        ax1.plot(loop_df["Timestamp"], loop_df["PV"], label="PV", linewidth=1.2)
        ax1.set_title(f"{selected_tag} -- trend")
        ax1.set_xlabel("Time")
        ax1.legend()

        ax2.scatter(loop_df["OP"], loop_df["PV"], s=8, alpha=0.6)
        ax2.set_title(f"{selected_tag} -- OP vs PV")
        ax2.set_xlabel("OP (controller output)")
        ax2.set_ylabel("PV (process variable)")

        st.pyplot(fig)

        st.caption(
            "Left: how OP and PV move over time. Right: the cross-plot -- "
            "a straight line means the valve is tracking OP cleanly; a "
            "jagged or looped shape signals a problem worth inspecting."
        )
