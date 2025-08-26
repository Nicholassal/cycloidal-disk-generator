# disk_generator.py
"""
Disk Generator — Cycloidal Disk Equation Builder (for SOLIDWORKS)
=================================================================
VS Code–ready Streamlit app.

What it does
------------
- Accepts R_p, e, r, N with hover help for MechEs.
- Generates **SolidWorks-ready** parametric expressions x(t) and y(t)
  (copy buttons copy *only* the right-hand expressions — no `x=`/`y=`).
- Plots the curve for t in [0, 2π] to visualize changes.
- Warns about common SolidWorks pitfalls (radians, singularities, parametric mode).

SolidWorks constraints respected (see SW Help):
- Equation Driven Curve uses **radians**.
- Use **Parametric** with t1=0, t2=2*pi (or 2*pi - 1e-6 if it complains).
- Inverse tangent is **atn** in SW equations.
- You can’t reference Global Variables directly inside the Equation Driven Curve
  (link a dimension to a GV if you need parameterization).
"""

from typing import Tuple
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# -------------------------------
# Formatting & math helpers
# -------------------------------
def _fmt_num(x: float) -> str:
    """Compact numeric format suitable for CAD strings (stable, short)."""
    return format(float(x), ".12g")

def make_sw_equations(R_p: float, e: float, r: float, N: int) -> Tuple[str, str]:
    """
    Build *only the right-hand expressions* for SolidWorks X(t), Y(t).
    Uses 'atn' for arctangent per SolidWorks function naming.
    """
    N_int = int(N)
    _N_ = 1 - N_int
    R_p_e_N = R_p / (e * N_int)

    Rp   = _fmt_num(R_p)
    ee   = _fmt_num(e)
    rr   = _fmt_num(r)
    Nn   = str(N_int)
    N_   = _fmt_num(_N_)
    RpeN = _fmt_num(R_p_e_N)

    x_expr = (
        f"({Rp}*cos(t)) - "
        f"({rr}*cos(t+atn(sin({N_}*t)/({RpeN}-cos({N_}*t))))) - "
        f"({ee}*cos({Nn}*t))"
    )
    y_expr = (
        f"(-{Rp}*sin(t)) + "
        f"({rr}*sin(t+atn(sin({N_}*t)/({RpeN}-cos({N_}*t))))) + "
        f"({ee}*sin({Nn}*t))"
    )
    return x_expr, y_expr

def sample_curve(R_p: float, e: float, r: float, N: int,
                 t1=0.0, t2=2*math.pi, samples=2000):
    """
    Evaluate X(t), Y(t) for plotting. Adds a small epsilon if the
    arctangent denominator approaches zero.
    """
    N_int = int(N)
    _N_ = 1 - N_int
    R_p_e_N = R_p / (e * N_int)

    t = np.linspace(t1, t2, samples)
    denom = (R_p_e_N - np.cos(_N_ * t))
    eps = 1e-9
    mask_bad = np.isclose(denom, 0.0, atol=1e-9)
    denom_safe = np.where(mask_bad, np.sign(denom) * eps, denom)

    phi = np.arctan(np.sin(_N_ * t) / denom_safe)

    X = R_p * np.cos(t) - r * np.cos(t + phi) - e * np.cos(N_int * t)
    Y = -R_p * np.sin(t) + r * np.sin(t + phi) + e * np.sin(N_int * t)

    diagnostics = {
        "has_singularity": bool(np.any(mask_bad)),
        "R_p_over_eN": R_p_e_N,
        "R_p_over_eN_in_unit_interval": (-1.0 <= R_p_e_N <= 1.0),
    }
    return X, Y, t, diagnostics

def validate_inputs(R_p: float, e: float, r: float, N: int):
    """
    Validate value ranges & likely SolidWorks failures.
    Returns (is_ok, messages:list[str]).
    """
    msgs = []
    ok = True
    if not (isinstance(N, (int, np.integer)) and N >= 2):
        ok = False; msgs.append("N must be an integer ≥ 2 (number of fixed pins).")
    if R_p <= 0 or e <= 0 or r <= 0:
        ok = False; msgs.append("All geometry parameters must be positive (R_p, e, r > 0).")
    if e * N == 0:
        ok = False; msgs.append("Denominator e*N must be non-zero.")
    if e * N != 0:
        RpeN = R_p / (e * N)
        if -1.0 <= RpeN <= 1.0:
            msgs.append(
                f"Warning: R_p/(e*N) = {RpeN:.6g} lies in [-1,1]. "
                "The `atn` denominator may hit zero for some t; SolidWorks may reject or break the curve."
            )
    return ok, msgs


# -------------------------------
# Streamlit UI (VS Code friendly)
# -------------------------------
st.set_page_config(page_title="Disk Generator", page_icon="🌀", layout="centered")
st.title("🌀 Disk Generator")
st.caption("Cycloidal Disk Equation Builder — SolidWorks Equation-Driven Curve (Parametric, radians)")

with st.form("inputs"):
    c1, c2, c3, c4 = st.columns(4)

    R_p = c1.number_input(
        "R_p",
        min_value=0.0, value=50.0, step=1.0, format="%.6f",
        help="Pitch circle radius of fixed pins (mm). If outer pin circle diameter is D, use R_p = D/2."
    )
    e = c2.number_input(
        "e",
        min_value=0.0, value=2.5, step=0.1, format="%.6f",
        help="Cycloid eccentricity (mm). Typically the cam/eccentric offset driving the disk."
    )
    r = c3.number_input(
        "r",
        min_value=0.0, value=2.0, step=0.1, format="%.6f",
        help="Fixed pin radius (mm). If pin diameter is d, r = d/2."
    )
    N = c4.number_input(
        "N",
        min_value=2, value=10, step=1,
        help="Number of fixed pins around the ring. Higher N increases reduction; mind lobe geometry."
    )

    st.markdown("_All trig functions use **radians** (SolidWorks Equation Driven Curve requirement)._")
    submit = st.form_submit_button("Generate equations")

if submit:
    ok, msgs = validate_inputs(R_p, e, r, N)
    for m in msgs:
        if m.lower().startswith("warning"):
            st.warning(m)
        else:
            st.error(m)

    if ok:
        x_expr, y_expr = make_sw_equations(R_p, e, r, int(N))
        st.success("Equations generated. Paste into SolidWorks (Parametric mode).")

        # Text boxes + precise copy buttons (copy RHS only)
        def copy_box(label: str, payload: str, key: str):
            st.text_area(label, payload, height=60, key=f"ta_{key}")
            st.markdown(
                f"""
                <div style="margin-top:-10px;margin-bottom:10px;">
                  <button onclick="navigator.clipboard.writeText(`{payload}`)">
                    Copy {label} (expression only)
                  </button>
                </div>
                """,
                unsafe_allow_html=True
            )

        copy_box("x(t) expression", x_expr, "x")
        copy_box("y(t) expression", y_expr, "y")

        # Plot preview
        X, Y, T, diag = sample_curve(R_p, e, r, int(N))
        if diag["has_singularity"]:
            st.warning(
                "Singularity detected in the `atn` argument for some t. "
                "In SolidWorks, consider adjusting parameters or using t2 = 2*pi - 1e-6."
            )

        st.subheader("Preview (t from 0 to 2π)")
        fig = plt.figure()
        plt.plot(X, Y)
        plt.axis("equal")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Cycloidal Disk")
        st.pyplot(fig, clear_figure=True)

with st.expander("SolidWorks tips & constraints"):
    st.markdown(
        """
        - Use **Parametric** mode in *Equation Driven Curve* and paste the two expressions.
        - Set **t1 = 0**, **t2 = 2*pi** (radians). If SW complains about closure, try `2*pi - 1e-6`.
        - SolidWorks uses **`atn`** for arctan in equations.
        - You cannot reference **Global Variables** directly inside the curve; drive a dimension with a GV instead.
        """
    )

st.divider()
st.caption("Built for VS Code + Streamlit. Mechanical-engineer friendly.")
