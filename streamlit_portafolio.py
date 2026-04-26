# streamlit_portafolio.py
# -------------------------------------------------------------
# Herramienta Streamlit para armar y balancear un portafolio
# con data de Yahoo Finance y optimización (mín var / máx Sharpe).
# Autor: ChatGPT (GPT-5 Thinking)
# -------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import math
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import streamlit as st

# ----------------------------
# Configuración de la página
# ----------------------------
st.set_page_config(
    page_title="Portafolio de Acciones — Yahoo Finance",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Portafolio de Acciones con Yahoo Finance")
st.caption("Arma tu portafolio, descarga precios y optimiza pesos (mínimo riesgo o máximo Sharpe).")

# ----------------------------
# Utilidades
# ----------------------------

@st.cache_data(show_spinner=False)
def descargar_precios(tickers, start, end, interval):
    """Descarga precios ajustados de Yahoo Finance."""
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    # Normalizar a un DataFrame con columnas de tickers (Close ajustado)
    if len(tickers) == 1:
        # yfinance devuelve Series si hay 1 ticker
        close = data["Close"].to_frame(tickers[0]).copy()
    else:
        # Cuando son varios, queda multiíndice: col nivel 0=ticker, nivel 1=campo
        close = pd.concat([data[t]["Close"].rename(t) for t in tickers], axis=1)
    # Limpiar columnas vacías y duplicados
    close = close.dropna(how="all", axis=1)
    close = close.sort_index()
    return close

def calcular_retorno_precios(precios, metodo_retorno):
    """Calcula retornos a partir de precios: log o simple."""
    if precios.empty:
        return pd.DataFrame()
    if metodo_retorno == "Logarítmico":
        rets = np.log(precios / precios.shift(1))
    else:
        rets = precios.pct_change()
    return rets.dropna(how="all")

def anualizar_params(mu, cov, freq):
    """Anualiza media y covarianza según frecuencia."""
    # Frecuencias aproximadas
    f_map = {"D": 252, "W": 52, "M": 12}
    k = f_map.get(freq, 252)
    mu_a = mu * k
    cov_a = cov * k
    return mu_a, cov_a, k

def port_stats(w, mu, cov, rf=0.0):
    """Devuelve rendimiento, volatilidad y Sharpe de un portafolio."""
    w = np.array(w).reshape(-1, 1)
    ret = float(w.T @ mu)
    vol = float(np.sqrt(w.T @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe

def optimizar_pesos(mu, cov, rf, metodo, bounds, shorting, target_ret=None):
    """
    Optimiza pesos con restricciones:
    - suma de pesos = 1
    - bounds por activo
    - método: 'Igualitario', 'Mín Var', 'Máx Sharpe', 'Riesgo Mín con retorno objetivo'
    """
    n = len(mu)
    if n == 0:
        return np.array([])

    # Configuración de cotas
    if shorting:
        lb, ub = bounds
        lb = min(lb, -1.0)
        ub = max(ub, 1.0)
        bnds = tuple((lb, ub) for _ in range(n))
    else:
        lb, ub = bounds
        lb = max(0.0, lb)
        ub = max(lb, ub)
        bnds = tuple((lb, ub) for _ in range(n))

    # Restricción suma de pesos
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    w0 = np.array([1.0 / n] * n)

    if metodo == "Igualitario":
        return w0

    if metodo == "Mín Var":
        def obj(w):
            return (w @ cov @ w)
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x

    if metodo == "Máx Sharpe":
        def neg_sharpe(w):
            r, v, s = port_stats(w, mu, cov, rf)
            return -s
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x

    if metodo == "Riesgo Mín con retorno objetivo":
        if target_ret is None:
            target_ret = float(mu.mean())
        def obj(w):
            return (w @ cov @ w)
        cons_rt = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: w @ mu - target_ret},
        )
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons_rt)
        if not res.success:
            # Fallback: mínimo var
            res = minimize(lambda w: (w @ cov @ w), w0, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x

    return w0

def construir_frontera(mu, cov, rf, bnds, shorting, npts=50):
    """Calcula puntos de la frontera eficiente (target returns equiespaciados)."""
    n = len(mu)
    if n == 0:
        return pd.DataFrame(columns=["ret", "vol", "sharpe"])
    mu_min, mu_max = float(mu.min()), float(mu.max())
    targets = np.linspace(mu_min, mu_max, npts)

    puntos = []
    for t in targets:
        w = optimizar_pesos(mu, cov, rf, "Riesgo Mín con retorno objetivo", bnds, shorting, target_ret=t)
        r, v, s = port_stats(w, mu, cov, rf)
        puntos.append((r, v, s))
    df = pd.DataFrame(puntos, columns=["ret", "vol", "sharpe"])
    return df

def fig_frontera(df, r_opt=None, v_opt=None, s_opt=None):
    """Grafica frontera eficiente y (opcional) el portafolio óptimo."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not df.empty:
        ax.plot(df["vol"], df["ret"], linewidth=2, marker="")
    if r_opt is not None and v_opt is not None:
        ax.scatter([v_opt], [r_opt], s=60)
    ax.set_xlabel("Volatilidad anual")
    ax.set_ylabel("Retorno anual")
    ax.set_title("Frontera eficiente")
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig

def descargar_csv(df: pd.DataFrame, filename: str) -> bytes:
    """Convierte un DataFrame a bytes CSV para descargar."""
    return df.to_csv(index=True).encode("utf-8")


# ----------------------------
# Sidebar — Parámetros
# ----------------------------
with st.sidebar:
    st.header("⚙️ Parámetros")

    tickers_str = st.text_area(
        "Tickers (separados por coma)",
        value="AAPL, MSFT, NVDA, AMZN, GOOG"
    )
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    col_dates = st.columns(2)
    with col_dates[0]:
        start_date = st.date_input("Fecha inicio", value=date.today() - timedelta(days=365*3))
    with col_dates[1]:
        end_date = st.date_input("Fecha fin", value=date.today())

    intervalo = st.selectbox(
        "Frecuencia de precios",
        options={"Diaria": "1d", "Semanal": "1wk", "Mensual": "1mo"},
        index=0
    )
    # Mapea lo que se muestra al usuario a códigos de yfinance
    freq_map = {"Diaria": "1d", "Semanal": "1wk", "Mensual": "1mo"}
    intervalo_yf = freq_map[intervalo]  # para descargar precios
    freq_code = {"1d": "D", "1wk": "W", "1mo": "M"}[intervalo_yf]  # para anualizar

    metodo_retorno = st.radio("Método de retornos", ["Simple", "Logarítmico"], index=0)

    st.divider()

    st.subheader("Optimización")
    metodo = st.selectbox("Método", ["Igualitario", "Mín Var", "Máx Sharpe", "Riesgo Mín con retorno objetivo"], index=2)
    rf = st.number_input("Tasa libre de riesgo (anual, p.ej. 0.04 = 4%)", value=0.02, step=0.005, format="%.4f")

    shorting = st.checkbox("Permitir 'shorting' (pesos negativos)", value=False)
    min_w = st.number_input("Peso mínimo por activo", value=0.0, step=0.05, format="%.2f")
    max_w = st.number_input("Peso máximo por activo", value=1.0, step=0.05, format="%.2f")

    target_ret = None
    if metodo == "Riesgo Mín con retorno objetivo":
        target_ret = st.number_input("Retorno objetivo (anual)", value=0.10, step=0.01, format="%.4f")

    st.divider()
    construir_frontier = st.checkbox("Calcular y mostrar frontera eficiente", value=True)

# ----------------------------
# Descarga de datos y cálculos
# ----------------------------
precios = descargar_precios(tickers, start_date, end_date + timedelta(days=1), intervalo_yf)

if precios.empty:
    st.warning("No se pudieron descargar precios. Verifica los tickers o el rango de fechas.")
    st.stop()

st.subheader("Precios ajustados")
st.dataframe(precios.tail(10))

# Retornos
ret_diarios = calcular_retorno_precios(precios, metodo_retorno)

# Media / Covarianza
mu = ret_diarios.mean().values.reshape(-1, 1)
cov = ret_diarios.cov().values

# Anualizar
mu_a, cov_a, k = anualizar_params(mu, cov, freq_code)

# Optimización
bounds = (min_w, max_w)
w_opt = optimizar_pesos(mu_a, cov_a, rf, metodo, bounds, shorting, target_ret=target_ret)
labels = list(precios.columns)

# Métricas del portafolio óptimo
ret_opt, vol_opt, sharpe_opt = port_stats(w_opt, mu_a, cov_a, rf)

st.subheader("Pesos óptimos")
tabla_pesos = pd.DataFrame({
    "Ticker": labels,
    "Peso": w_opt
}).set_index("Ticker").sort_values("Peso", ascending=False)
st.dataframe(tabla_pesos.style.format({"Peso": "{:.2%}"}))

colm = st.columns(3)
colm[0].metric("Retorno (anual)", f"{ret_opt:.2%}")
colm[1].metric("Volatilidad (anual)", f"{vol_opt:.2%}")
colm[2].metric("Sharpe", f"{sharpe_opt:.2f}")

# Composición acumulada del portafolio (backtest) con pesos fijos
st.subheader("Backtest del portafolio (pesos fijos)")
ret_port_diario = (ret_diarios.values @ w_opt.reshape(-1, 1)).flatten()
acum = (1 + ret_port_diario).cumprod()
serie_port = pd.Series(acum, index=ret_diarios.index, name="Portafolio")
serie_bench = precios[labels[0]] / precios[labels[0]].iloc[0]
serie_bench.name = labels[0]

fig_acum, ax_acum = plt.subplots(figsize=(8, 4.5))
ax_acum.plot(serie_port.index, serie_port.values, label="Portafolio")
ax_acum.plot(serie_bench.index, serie_bench.values, label=f"Benchmark: {labels[0]}")
ax_acum.set_title("Crecimiento de 1 unidad monetaria")
ax_acum.set_xlabel("Fecha")
ax_acum.set_ylabel("Índice de valor")
ax_acum.grid(True, linestyle="--", alpha=0.4)
ax_acum.legend()
st.pyplot(fig_acum, clear_figure=True)

# Frontera eficiente
if construir_frontier:
    st.subheader("Frontera eficiente")
    df_frontier = construir_frontera(mu_a.flatten(), cov_a, rf, bounds, shorting, npts=60)
    fig_front = fig_frontera(df_frontier, r_opt=ret_opt, v_opt=vol_opt, s_opt=sharpe_opt)
    st.pyplot(fig_front, clear_figure=True)

    with st.expander("Descargar frontera eficiente (CSV)"):
        st.download_button(
            "Descargar 'frontera.csv'",
            data=descargar_csv(df_frontier, "frontera.csv"),
            file_name="frontera.csv",
            mime="text/csv"
        )

# Exportables
with st.expander("Descargas"):
    st.download_button(
        "Descargar precios (CSV)",
        data=descargar_csv(precios, "precios.csv"),
        file_name="precios.csv",
        mime="text/csv"
    )
    st.download_button(
        "Descargar retornos (CSV)",
        data=descargar_csv(ret_diarios, "retornos.csv"),
        file_name="retornos.csv",
        mime="text/csv"
    )
    st.download_button(
        "Descargar pesos óptimos (CSV)",
        data=descargar_csv(tabla_pesos, "pesos_optimos.csv"),
        file_name="pesos_optimos.csv",
        mime="text/csv"
    )

st.info(
    "💡 Sugerencias: \n"
    "- Cambia el intervalo (diario/semanal/mensual) según tu horizonte.\n"
    "- Ajusta **peso mínimo/máximo** para concentrar o diversificar.\n"
    "- Usa **Riesgo mínimo con retorno objetivo** para fijar un target anual.\n"
    "- Activa *shorting* solo si entiendes el apalancamiento implícito."
)
