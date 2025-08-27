import math
import html as html_lib
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

st.title("🌀 Cycloidal Disk Generator")

# Two-column layout
left_col, right_col = st.columns([1.3, 1])

with left_col:
    # Parameters title (as requested)
    st.subheader(" Parameters")

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

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)


    # If expressions exist, render them inside a single HTML block so CSS applies
    if x_expr and y_expr:
        # escape the expressions for safe HTML placement
        x_html = html_lib.escape(x_expr)
        y_html = html_lib.escape(y_expr)

        # Build an HTML snippet that contains all CSS + both equation cards + JS copy handlers.
        # Important: CSS/JS are inside this HTML so they affect the cards (components.html uses an iframe).
        html_snippet = """
        <style>
        /* Container */
        .eq-wrap { display:flex; flex-direction:column; gap:14px; width:100%; box-sizing:border-box; padding-bottom:6px; }

        /* REAL card */
        .eq-card {
            border: 1px solid rgba(209, 213, 219, 1);
            border-radius: 10px;
            background: #ffffff;
            overflow: hidden;
        }
        .eq-header {
            display:flex;
            align-items:center;
            justify-content:space-between;
            padding:10px 12px;
            background:#f8fafc;
            border-bottom:1px solid #e6eef6;
            font-weight:700;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Courier New", monospace;
            color:#0f172a;
        }
        .eq-body {
            padding:12px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Courier New", monospace;
            white-space: pre-wrap;
            word-break: break-word;
            max-height:180px;
            overflow:auto;
            font-size:13px;
            line-height:1.35;
        }

        /* copy button (square) */
        .copy-btn {
            width:36px;
            height:36px;
            border-radius:8px;
            border:1px solid #cbd5e1;
            background:#ffffff;
            display:inline-flex;
            align-items:center;
            justify-content:center;
            cursor:pointer;
            transition: transform 0.06s ease, background 0.12s ease;
        }
        .copy-btn:hover { background:#f8fafc; transform: translateY(-1px); }
        .copy-btn:active { transform: scale(0.98); }

        .copy-btn svg { width:18px; height:18px; fill:#0f172a; }
        </style>

        <div class="eq-wrap">
          <div class="eq-card">
            <div class="eq-header">
              <div>x(t)</div>
              <button id="copy_x" class="copy-btn" title="Copy x(t)">
                <!-- Clipboard icon -->
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v16c0 
                           1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 18H8V7h11v16z"/>
                </svg>
              </button>
            </div>
            <pre class="eq-body" id="expr_x">{X_EXPR}</pre>
          </div>

          <div class="eq-card">
            <div class="eq-header">
              <div>y(t)</div>
              <button id="copy_y" class="copy-btn" title="Copy y(t)">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v16c0 
                           1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 18H8V7h11v16z"/>
                </svg>
              </button>
            </div>
            <pre class="eq-body" id="expr_y">{Y_EXPR}</pre>
          </div>
        </div>

        <script>
        // copy function with fallback
        function _copyTextFrom(elId, btnId) {
            const el = document.getElementById(elId);
            if (!el) return;
            const text = el.innerText || el.textContent || "";
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(() => {
                    const b = document.getElementById(btnId);
                    if (b) { b.style.background = '#bbf7d0'; setTimeout(()=> b.style.background='', 700); }
                }).catch(() => { fallbackCopy(el, btnId); });
            } else {
                fallbackCopy(el, btnId);
            }

            function fallbackCopy(elem, btnId) {
                try {
                    const range = document.createRange();
                    range.selectNode(elem);
                    const sel = window.getSelection();
                    sel.removeAllRanges();
                    sel.addRange(range);
                    document.execCommand('copy');
                    sel.removeAllRanges();
                    const b = document.getElementById(btnId);
                    if (b) { b.style.background = '#bbf7d0'; setTimeout(()=> b.style.background='', 700); }
                } catch (err) {
                    // swallow
                }
            }
        }

        document.getElementById('copy_x').addEventListener('click', function(){ _copyTextFrom('expr_x', 'copy_x'); });
        document.getElementById('copy_y').addEventListener('click', function(){ _copyTextFrom('expr_y', 'copy_y'); });
        </script>
        """.replace("{X_EXPR}", x_html).replace("{Y_EXPR}", y_html)

        # Render the HTML with a height that's comfortably tall for two cards.
        # If your equations are very long, increase the height or adjust max-height in CSS above.
        components.html(html_snippet, height=320, scrolling=True)

    else:
        st.info("Enter valid parameters and click Generate to see SolidWorks expressions.")

    # Extra tips
    with st.expander("ℹ️ More SolidWorks Tips", expanded=False):
        st.markdown("""
        - Paste both expressions in Parametric mode.
        - t1 = 0, t2 = 2*pi (use `2*pi - 1e-6` if SolidWorks complains about closure).
        - Use `atn()` instead of `atan()` in SolidWorks equations.
        - You cannot reference Global Variables inside the curve directly; link dimensions to GVs instead.
        """)

with right_col:
    st.subheader("Preview")

    if ok:
        X, Y, _, diag = sample_curve(R_p, e, r, N)

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(X, Y, linewidth=0.9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle='--', linewidth=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0.05)
        st.pyplot(fig, clear_figure=True)

        if diag.get("has_singularity"):
            st.warning(
                "Singularity detected in `atn` argument. In SolidWorks, consider using `t2 = 2*pi - 1e-6`."
            )
    else:
        st.info("Please enter valid parameters and click Generate.")

st.caption("Built for VS Code + Streamlit • Mechanical Engineer Friendly • 🌀")
