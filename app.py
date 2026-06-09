"""Streamlit dashboard — Blanchard-Gali con modelos anidados.

Mueve los parámetros y mira las IRFs a un shock de petróleo (+10%). Un único
modelo maestro; los demás modelos quedan anidados según los parámetros.
Corre con:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from oilnk_model import Params, PRESETS, irf, optimal_irf, welfare_table
from oilnk_model.theme import OILNK, REGIME_COLORS, plotly_layout

st.set_page_config(page_title="Oil & NK — modelos anidados", layout="wide")

# ---- panels shown in the IRF grid ------------------------------------------
PANELS = [
    ("pi", "π  inflación core"),
    ("picpi", "π  headline (IPC)"),
    ("x", "x  brecha de bienestar"),
    ("y", "y  producto (PIB)"),
    ("ye", "yᵉ  producto potencial"),
    ("c", "c  consumo"),
    ("r_e", "rᵉ  tasa eficiente"),
    ("i", "i  tasa de política"),
]
POLICIES = {
    "optimo": "Óptima (Ramsey)",
    "core": "Taylor a core",
    "headline": "Taylor a headline",
}
SHORT = {"pi": "π core", "picpi": "π headline", "x": "brecha x", "y": "producto y",
         "ye": "potencial yᵉ", "c": "consumo c", "r_e": "rᵉ", "i": "tasa i"}
SHOCK_SIZE = {"oil": 0.10, "costpush": 0.01}
SHOCK_LABELS = {"oil": "Petróleo (+10%)", "costpush": "Cost-push puro (+1 pp)"}

DEFAULTS = dict(alpha=0.05, chi=0.06, gamma=0.8, theta=0.75, rho_s=0.9, rho_u=0.5,
                sigma=1.0, phi=1.0, phi_pi=1.5, phi_x=0.125, epsilon=6.0)
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def yvals(s):
    """IRF values in puntos %, with numerical zeros snapped to 0 (clean flat lines)."""
    v = s.value.values
    return np.where(np.abs(v) < 1e-12, 0.0, v) * 100


# ---- cached compute --------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute(p: Params, horizon: int):
    out = {}
    for sh in ("oil", "costpush"):
        sz = SHOCK_SIZE[sh]
        out[(sh, "core")] = irf(p.update(rule_cpi=0.0), horizon=horizon, shock=sz, which=sh)
        out[(sh, "headline")] = irf(p.update(rule_cpi=1.0), horizon=horizon, shock=sz, which=sh)
        out[(sh, "optimo")] = optimal_irf(p, horizon=horizon, shock=sz, which=sh)
    eu = out[("oil", "core")].attrs.get("eu", [1, 1])
    return out, eu


@st.cache_data(show_spinner=False)
def compute_welfare(p: Params, which: str):
    return welfare_table(p, which)


# ---- sidebar: parameters ---------------------------------------------------
st.sidebar.title("Parámetros")

preset = st.sidebar.selectbox("Preset (caso anidado)", ["Personalizado"] + list(PRESETS))
if st.sidebar.button("Aplicar preset", width="stretch") and preset != "Personalizado":
    for k, v in PRESETS[preset].items():
        st.session_state[k] = float(v)
    st.rerun()

st.sidebar.markdown("**Petróleo y tecnología**")
st.sidebar.slider("α — share de petróleo en producción", 0.0, 0.50, step=0.01, key="alpha")
st.sidebar.slider("χ — share de energía en el IPC (consumo)", 0.0, 0.40, step=0.01, key="chi")
st.sidebar.slider("ρₛ — persistencia del petróleo", 0.0, 0.99, step=0.01, key="rho_s")

st.sidebar.markdown("**Fricciones**")
st.sidebar.slider("γ — rigidez de salario real", 0.0, 0.99, step=0.01, key="gamma")
st.sidebar.slider("θ — rigidez de precios (Calvo)", 0.05, 0.95, step=0.01, key="theta")

st.sidebar.markdown("**Preferencias**")
st.sidebar.slider("σ — aversión al riesgo / 1/EIS", 0.5, 4.0, step=0.1, key="sigma")
st.sidebar.slider("φ — inversa de Frisch", 0.0, 5.0, step=0.1, key="phi")
st.sidebar.slider("ε — elasticidad entre variedades", 2.0, 11.0, step=0.5, key="epsilon")

st.sidebar.markdown("**Regla de Taylor**")
st.sidebar.slider("φπ — respuesta a inflación", 1.0, 3.0, step=0.05, key="phi_pi")
st.sidebar.slider("φx — respuesta a la brecha", 0.0, 1.0, step=0.025, key="phi_x")

st.sidebar.markdown("**Cost-push exógeno**")
st.sidebar.slider("ρᵤ — persistencia del cost-push", 0.0, 0.95, step=0.05, key="rho_u")

horizon = st.sidebar.slider("Horizonte (trimestres)", 8, 40, 16, step=1)

p = Params(**{k: float(st.session_state[k]) for k in DEFAULTS})

# ---- header ----------------------------------------------------------------
st.title("Petróleo y política monetaria — modelo Blanchard–Galí")
st.caption(
    "Un único modelo maestro. Al mover los parámetros se encienden, uno a uno, "
    "los distintos modelos. Elige el shock: petróleo (+10%) o cost-push puro (+1 pp)."
)

TOUR = """
Cada paso enciende **una característica más** del modelo. Usa el selector
**Preset** de la barra lateral (y luego *Aplicar preset*), o mueve los sliders
a mano, y observa los 8 paneles.

**1 · NK estándar (sin petróleo).** Preset *NK estándar (sin petróleo)*.
El shock de petróleo no hace nada: sin petróleo en la economía, su precio no
tiene efectos macro. *(α = 0, χ = 0)*

**2 · Petróleo en la producción.** Preset *Petróleo solo en producción*.
El potencial **yᵉ cae**: el petróleo es un shock de **oferta**. Pero π y x
siguen en 0 — se cumple la **divine coincidence**: la política acomoda la caída
del potencial, sin trade-off. *(α > 0, χ = 0, γ = 0)*

**3 · Petróleo también en el consumo.** Preset *Petróleo en producción y consumo*.
Aparece la brecha **headline vs. core** (la energía entra al IPC) y el **consumo
cae más que el PIB** (efecto ingreso). Sigue sin trade-off (π core y x en 0).
*(χ > 0, γ = 0)*

**4 · Rigidez de salario real.** Preset *Con rigidez de salario real*.
Ahora la cuña salarial **varía con el shock** → **cost-push** → π core sube y x
cae: aparece el **trade-off**. *(γ > 0)*

**5 · Persistencia: transitorio vs. permanente.** Mueve **ρₛ** entre bajo y alto
(preset *Shock casi permanente* para el extremo). Más persistente → respuestas
mayores y más largas, y la tasa eficiente **rᵉ más baja** (→ 0 si es casi
permanente). El **carácter** del shock no cambia: sigue siendo oferta con cost-push.

**6 · La política importa.** Compara los tres colores. La **óptima** ancla el
core (lo lleva incluso a negativo: compromiso) y minimiza la pérdida. Reaccionar
al **headline** sobre-ajusta y **agranda la brecha** (panel de bienestar ≈ 3×).

**7 · Petróleo vs. cost-push puro.** Selecciona **los dos shocks** para verlos lado
a lado. En **π** y la **brecha** se ven parecidos (ambos rompen la divine
coincidence). La diferencia: el **potencial yᵉ** sólo cae con el petróleo —el
cost-push lo deja plano— y el cost-push genera trade-off aun con γ = 0.
"""

with st.expander("📋 Cómo usarlo — recorrido paso a paso", expanded=True):
    st.markdown(TOUR)

# ---- nested-case indicator -------------------------------------------------
def badges(p: Params):
    out = []
    if p.alpha < 1e-9 and p.chi < 1e-9:
        out.append(("NK estándar — sin petróleo", "info"))
    elif p.chi < 1e-9:
        out.append(("Petróleo solo en producción · core = headline", "info"))
    else:
        out.append(("Petróleo en producción y consumo", "info"))
    if p.gamma < 1e-9:
        out.append(("γ = 0 ⇒ divine coincidence: π = x = 0", "success"))
    else:
        out.append((f"γ = {p.gamma:.2f} ⇒ cuña variable ⇒ trade-off", "warning"))
    if p.theta <= 0.10:
        out.append(("θ pequeño ⇒ precios casi flexibles", "info"))
    if p.rho_s >= 0.95:
        out.append(("ρₛ alto ⇒ shock casi permanente (rᵉ ≈ 0)", "info"))
    elif p.rho_s <= 0.3:
        out.append(("ρₛ bajo ⇒ shock transitorio (rᵉ alta)", "info"))
    if p.phi_pi <= 1.0:
        out.append(("φπ ≤ 1 ⇒ se viola el principio de Taylor (indeterminación)", "error"))
    return out

cols = st.columns(len(badges(p)))
for col, (txt, kind) in zip(cols, badges(p)):
    getattr(col, kind)(txt)

data, eu = compute(p, horizon)
if eu != [1, 1]:
    st.error("El modelo no tiene equilibrio único con estos parámetros "
             "(revisa φπ > 1 y las fricciones). Las IRFs pueden no ser válidas.")

c1, c2 = st.columns(2)
with c1:
    show = st.multiselect("Políticas a comparar", list(POLICIES.values()),
                          default=list(POLICIES.values()))
with c2:
    shock_sel = st.multiselect("Shock(s) — elige dos para comparar lado a lado",
                               list(SHOCK_LABELS.values()), default=[SHOCK_LABELS["oil"]])
active = [k for k, v in POLICIES.items() if v in show] or ["optimo"]
shocks = [k for k, v in SHOCK_LABELS.items() if v in shock_sel] or ["oil"]

# ---- IRF grid --------------------------------------------------------------
if len(shocks) == 1:
    sh = shocks[0]
    st.markdown(f"##### Respuestas a {SHOCK_LABELS[sh]} · desvíos en puntos %")
    fig = make_subplots(rows=4, cols=2, subplot_titles=[t for _, t in PANELS],
                        vertical_spacing=0.09, horizontal_spacing=0.08)
    for idx, (var, _t) in enumerate(PANELS):
        rr, cc = idx // 2 + 1, idx % 2 + 1
        for reg in active:
            s = data[(sh, reg)]
            s = s[s.variable == var].sort_values("period")
            fig.add_trace(go.Scatter(
                x=s.period, y=yvals(s), mode="lines", name=POLICIES[reg],
                legendgroup=reg, showlegend=(idx == 0),
                line=dict(color=REGIME_COLORS[reg], width=2.6)), row=rr, col=cc)
        fig.add_hline(y=0, line=dict(color=OILNK["muted"], width=1), row=rr, col=cc)
    fig.update_layout(**plotly_layout(height=860, margin=dict(l=55, r=20, t=80, b=45),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0)))
else:
    st.markdown("##### Petróleo vs. cost-push puro · desvíos en puntos %  "
                "*(misma forma en π y brecha; el potencial yᵉ sólo cae con el petróleo)*")
    fig = make_subplots(rows=8, cols=2, shared_xaxes=True,
                        column_titles=[SHOCK_LABELS[s] for s in shocks],
                        vertical_spacing=0.035, horizontal_spacing=0.11)
    for ri, (var, _t) in enumerate(PANELS):
        for ci, sh in enumerate(shocks):
            for reg in active:
                s = data[(sh, reg)]
                s = s[s.variable == var].sort_values("period")
                fig.add_trace(go.Scatter(
                    x=s.period, y=yvals(s), mode="lines", name=POLICIES[reg],
                    legendgroup=reg, showlegend=(ri == 0 and ci == 0),
                    line=dict(color=REGIME_COLORS[reg], width=2.2)), row=ri + 1, col=ci + 1)
            fig.add_hline(y=0, line=dict(color=OILNK["muted"], width=1), row=ri + 1, col=ci + 1)
        fig.update_yaxes(title_text=SHORT[var], row=ri + 1, col=1)
    fig.update_layout(**plotly_layout(height=1180, margin=dict(l=75, r=20, t=70, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0)))

fig.update_xaxes(showgrid=True)
st.plotly_chart(fig, width="stretch")

# ---- welfare panel ---------------------------------------------------------
st.subheader("Bienestar — pérdida relativa a la óptima")
st.caption("L = Var(π core) + ϑ·Var(x),  ϑ = κ/ε.  Óptima (Ramsey) = 1.  "
           "(Con cost-push puro no hay distinción core/headline.)")
for col, sh in zip(st.columns(len(shocks)), shocks):
    rel = compute_welfare(p, sh)["rel"]
    wfig = go.Figure()
    wfig.add_trace(go.Bar(
        x=[POLICIES[k] for k in ["optimo", "core", "headline"]],
        y=[rel["optimo"], rel["core"], rel["headline"]],
        marker_color=[REGIME_COLORS[k] for k in ["optimo", "core", "headline"]],
        text=[f"×{rel[k]:.2f}" for k in ["optimo", "core", "headline"]],
        textposition="outside",
    ))
    wfig.update_layout(**plotly_layout(
        height=340, showlegend=False, margin=dict(l=45, r=10, t=42, b=35),
        title=dict(text=SHOCK_LABELS[sh], font=dict(color=OILNK["slate"], size=14))))
    wfig.update_yaxes(title_text="pérdida relativa")
    col.plotly_chart(wfig, width="stretch")

with st.expander("Cómo leer los paneles"):
    st.markdown(
        "- **π core**: inflación de los precios pegajosos — la relevante para el bienestar.\n"
        "- **π headline (IPC)**: incluye el pass-through directo de la energía.\n"
        "- **x brecha**: producto menos potencial eficiente (el output gap de bienestar).\n"
        "- **yᵉ potencial**: cae con el petróleo y **no depende de la política** "
        "(la misma línea para las tres reglas).\n"
        "- **c consumo**: cae más que el PIB cuando χ > 0 (efecto ingreso).\n"
        "- **rᵉ tasa eficiente**: sube con shocks transitorios, → 0 con permanentes.\n"
        "- **i tasa de política**: cómo responde la TPM bajo cada regla."
    )
