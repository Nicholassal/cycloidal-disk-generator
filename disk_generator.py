import math
from typing import Tuple
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# -------------------------------
# Math Helpers
# -------------------------------
def _fmt_num(x: float) -> str:
    return format(float(x), ".12g")

def make_sw_equations(R_p: float, e: float, r: float, N: int) -> Tuple[str, str]:
    N_int = int(N)
    _N_ = 1 - N_int
    R_p_e_N = R_p / (e * N_int)

    Rp = _fmt_num(R_p)
    ee = _fmt_num(e)
    rr = _fmt_num(r)
    Nn = str(N_int)
    N_ = _fmt_num(_N_)
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
                 t1=0.0, t2=2*math.pi, samples=1000):
    N_int = int(N)
    _N_ = 1 - N_int
    R_p_e_N = R_p / (e * N_int)

    t = np.linspace(t1, t2, samples)
    denom = (R_p_e_N - np.cos(_N_ * t))
    eps = 1e-9
    denom_safe = np.where(np.isclose(denom, 0.0, atol=1e-9), np.sign(denom) * eps, denom)
    phi = np.arctan(np.sin(_N_ * t) / denom_safe)

    X = R_p * np.cos(t) - r * np.cos(t + phi) - e * np.cos(N_int * t)
    Y = -R_p * np.sin(t) + r * np.sin(t + phi) + e * np.sin(N_int * t)

    diagnostics = {
        "has_singularity": bool(np.any(np.isclose(denom, 0.0, atol=1e-9))),
        "R_p_over_eN": R_p_e_N,
        "R_p_over_eN_in_unit_interval": (-1.0 <= R_p_e_N <= 1.0),
    }
    return X, Y, t, diagnostics

def validate_inputs(R_p, e, r, N):
    msgs = []
    ok = True
    if not (isinstance(N, (int, np.integer)) and N >= 2):
        ok = False; msgs.append("Number of pins must be an integer ≥ 2.")
    if R_p <= 0 or e <= 0 or r <= 0:
        ok = False; msgs.append("All geometry parameters must be positive.")
    if e * N == 0:
        ok = False; msgs.append("Eccentricity × Pin Count must be non-zero.")
    else:
        RpeN = R_p / (e * N)
        if -1.0 <= RpeN <= 1.0:
            msgs.append(f"Warning: R_p/(e×N) = {RpeN:.6g} ∈ [-1,1]. May cause singularities in SolidWorks.")
    return ok, msgs

# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Disk Generator", page_icon="🌀", layout="wide")

# CSS to fix padding issues and spacing
st.markdown("""
<style>
/* Top padding to avoid browser UI overlap */
.block-container {
    padding-top: 3.5rem;
}
/* Small margin between sections on left side */
.section-margin {
    margin-bottom: 1.5rem;
}
/* Copy button style */
.copy-btn {
    font-size: 14px;
    padding: 5px 10px;
    cursor: pointer;
    margin-left: 0.5rem;
}
/* Align buttons top-right inside code container */
.code-container {
    position: relative;
}
/* Place copy button top-right */
.copy-btn-container {
    position: absolute;
    top: 5px;
    right: 5px;
}
</style>
""", unsafe_allow_html=True)

st.title("🌀 Cycloidal Disk Generator")

# Left and right columns
left_col, right_col = st.columns([1.3, 1])

with left_col:
    # Configuration inputs
    with st.form("input_form"):
        R_p = st.number_input(
            "Pin Circle Radius (Rₚ, mm)",
            value=50.0, min_value=0.01, step=1.0,
            help="Pitch circle radius of fixed pins (mm). Use half of outer pin circle diameter."
        )
        e = st.number_input(
            "Eccentricity (e, mm)",
            value=2.5, min_value=0.0, step=0.1,
            help="Cycloid eccentricity (mm). Usually cam/eccentric offset driving the disk."
        )
        r = st.number_input(
            "Pin Radius (r, mm)",
            value=2.0, min_value=0.01, step=0.1,
            help="Fixed pin radius (mm). If pin diameter is d, r = d/2."
        )
        N = st.number_input(
            "Number of Pins (N)",
            value=10, min_value=2, step=1,
            help="Number of fixed pins around the ring. Higher N increases reduction."
        )
        generate = st.form_submit_button("Generate")

    if not generate:
        R_p, e, r, N = 50.0, 2.5, 2.0, 10

    ok, msgs = validate_inputs(R_p, e, r, N)
    if generate:
        for msg in msgs:
            (st.warning if msg.lower().startswith("warning") else st.error)(msg)

    if ok:
        x_expr, y_expr = make_sw_equations(R_p, e, r, N)
    else:
        x_expr = y_expr = ""

    # Expressions + copy buttons section
    st.markdown('<div class="section-margin">', unsafe_allow_html=True)
    st.subheader("SolidWorks Expressions")

    def copy_button_js(expr_id, btn_id):
        # Return JS + HTML for a button copying expr to clipboard
        return f"""
        <button class="copy-btn" id="{btn_id}" title="Copy expression to clipboard"
          onclick="
            const el = document.getElementById('{expr_id}');
            if (!navigator.clipboard) {{
              el.select();
              document.execCommand('copy');
            }} else {{
              navigator.clipboard.writeText(el.innerText);
            }}
            const btn = document.getElementById('{btn_id}');
            btn.innerText = 'Copied!';
            setTimeout(() => btn.innerText = 'Copy', 1500);
          }}">Copy</button>
        """

    # Show expressions with copy buttons aligned top-right
    for label, expr, key in [("x(t)", x_expr, "x"), ("y(t)", y_expr, "y")]:
        if expr:
            components.html(f"""
            <div class="code-container" style="position:relative; margin-bottom:0.75rem; border:1px solid #ddd; border-radius:5px; padding:10px; font-family: monospace; white-space: pre-wrap;">
                <div id="expr_{key}">{expr}</div>
                <div class="copy-btn-container">{copy_button_js(f"expr_{key}", "btn_" + key)}</div>
            </div>
            """, height=70)

    st.markdown('</div>', unsafe_allow_html=True)

    # SolidWorks tips expander
    with st.expander("ℹ️ SolidWorks Tips & Constraints", expanded=False):
        st.markdown("""
        - Use **Parametric** mode in *Equation Driven Curve* and paste the two expressions.
        - Set **t1 = 0**, **t2 = 2*pi** (radians). Use `2*pi - 1e-6` if SolidWorks complains about curve closure.
        - Use `atn()` instead of `atan()` in SW equations.
        - You **cannot** reference Global Variables inside the curve directly; link dimensions to GVs instead.
        """)

with right_col:
    st.subheader("Preview")

    if ok:
        X, Y, _, diag = sample_curve(R_p, e, r, N)

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(X, Y, linewidth=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle='--', linewidth=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0.05)
        st.pyplot(fig, clear_figure=True)

        if diag.get("has_singularity"):
            st.warning(
                "Singularity detected in `atn` argument. "
                "In SolidWorks, consider using `t2 = 2*pi - 1e-6`."
            )
    else:
        st.info("Please enter valid parameters and click Generate.")

st.caption("Built for VS Code + Streamlit • Mechanical Engineer Friendly • 🌀")
