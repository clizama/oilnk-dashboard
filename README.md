# Oil & NK — dashboard interactivo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nkoil-dashboard.streamlit.app/)

**App en vivo → https://nkoil-dashboard.streamlit.app/**

App interactiva para "jugar" con los parámetros de un modelo de **shocks de petróleo
y política monetaria** y ver las IRFs en vivo. Hay **un único modelo maestro**
(Blanchard–Galí estructural, petróleo en producción y consumo); los demás modelos
quedan **anidados** como casos particulares:

| Parámetros | Modelo |
|------------|--------|
| `alpha=0, chi=0` | NK estándar de 3 ecuaciones, sin petróleo |
| `chi=0, alpha>0` | petróleo solo en producción; core = headline |
| `gamma=0` | sin rigidez real ⇒ *divine coincidence* (π = x = 0) |
| `theta→0` | precios flexibles ⇒ dinero neutral |

Además del shock de petróleo, el modelo trae un **cost-push exógeno** (markup
shock, AR(1) con persistencia `rho_u`) para comparar el petróleo con un cost-push
"puro" del NK estándar. Seleccionando los dos shocks, el dashboard los muestra
lado a lado: se parecen en π y la brecha, pero **sólo el petróleo baja el
potencial** `yᵉ` (y el cost-push genera trade-off aun con `gamma = 0`).

## Estructura

```
.
├── oilnk_model/        # paquete del modelo
│   ├── params.py       # parámetros estructurales + presets (casos anidados)
│   ├── model.py        # modelo maestro en forma canónica (gensys)
│   ├── gensys.py       # solver de expectativas racionales (Sims 2002)
│   ├── optimal.py      # política óptima (Ramsey, compromiso)
│   ├── welfare.py      # pérdida de bienestar (varianzas vía Lyapunov)
│   └── theme.py        # paleta oilnk para Plotly
├── tests/              # validación contra los CSV de Dynare (vendored)
│   ├── test_validation.py
│   └── dynare_csv/     # IRFs de referencia exportadas desde los .mod
├── app.py              # UI Streamlit
└── requirements.txt
```

## Correr la app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sliders para todos los parámetros, IRFs a un shock de petróleo (+10%)
comparando las tres políticas (óptima / Taylor a core / Taylor a headline),
un indicador "¿en qué modelo estoy?" que detecta el caso anidado, presets y un
panel de bienestar (pérdida relativa a la óptima).

### Recorrido paso a paso

Cada paso enciende una característica más del modelo (usa el selector **Preset**
de la barra lateral o mueve los sliders):

1. **NK estándar (sin petróleo).** El shock de petróleo no hace nada: sin
   petróleo en la economía, su precio no tiene efectos macro. *(α = 0, χ = 0)*
2. **Petróleo en la producción.** El potencial **yᵉ cae** (shock de oferta),
   pero π y x siguen en 0: **divine coincidence**, sin trade-off. *(α > 0, γ = 0)*
3. **Petróleo también en el consumo.** Aparece **headline vs. core** y el
   **consumo cae más que el PIB** (efecto ingreso). Sigue sin trade-off. *(χ > 0)*
4. **Rigidez de salario real.** La cuña varía con el shock ⇒ **cost-push** ⇒ π
   sube y x cae: aparece el **trade-off**. *(γ > 0)*
5. **Persistencia.** Mueve **ρₛ**: más persistente ⇒ respuestas mayores y rᵉ más
   baja, pero el carácter del shock no cambia.
6. **La política importa.** La **óptima** ancla el core (compromiso); reaccionar
   al **headline** agranda la brecha (panel de bienestar ≈ 3×).
7. **Petróleo vs. cost-push puro.** Selecciona los dos shocks: se ven parecidos en
   π y la brecha, pero el **potencial yᵉ** sólo cae con el petróleo.

## Validación

```bash
pip install -r requirements.txt
pytest
```

La validación es la garantía de calidad: el solver Python reproduce las IRFs de
Dynare a precisión de máquina (`maxerr ~ 1e-16`). Los CSV de referencia en
`tests/dynare_csv/` se exportaron desde los `.mod` de Dynare, que siguen siendo la
fuente de verdad del modelo.

## Uso programático

```python
from oilnk_model import Params, irf
df = irf(Params(alpha=0.05, chi=0.06, gamma=0.8, rho_s=0.9), horizon=16)
# df: columnas period, variable, value (long format, como los CSV de Dynare)
```
