# streamlit_portafolio.py
# -------------------------------------------------------------
# Herramienta Streamlit para armar y balancear un portafolio
# con data de Yahoo Finance y optimización (mín var / máx Sharpe).
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
    if len(tickers) == 1:
        close = data["Close"].to_frame(tickers[0]).copy()
    else:
        # 🔧 FIX: evitar KeyError si falta algún ticker
        close = pd.concat(
            [data[t]["Close"].rename(t) for t in tickers if t in data.columns.get_level_values(0)],
            axis=1
        )
    close = close.dropna(how="all", axis=1)
    close = close.sort_index()
    return close

def calcular_retorno_precios(precios, metodo_retorno):
    if precios.empty:
        return pd.DataFrame()
    if metodo_retorno == "Logarítmico":
        rets = np.log(precios / precios.shift(1))
    else:
        rets = precios.pct_change()
    return rets.dropna(how="all")

def anualizar_params(mu, cov, freq):
    f_map = {"D": 252, "W": 52, "M": 12}
    k = f_map.get(freq, 252)
    return mu * k, cov * k, k

# 🔧 FIX: robustecer cálculo
def port_stats(w, mu, cov, rf=0.0):
    w = np.asarray(w, dtype=float).reshape(-1, 1)
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    cov = np.asarray(cov, dtype=float)

    ret = float(np.dot(w.T, mu).item())
    vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w)).item()))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe

def optimizar_pesos(mu, cov, rf, metodo, bounds, shorting, target_ret=None):
    n = len(mu)
    if n == 0:
        return np.array([])

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

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    w0 = np.array([1.0 / n] * n)

    if metodo == "Igualitario":
        return w0

    if metodo == "Mín Var":
        res = minimize(lambda w: (w @ cov @ w), w0, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x

    if metodo == "Máx Sharpe":
        res = minimize(lambda w: -port_stats(w, mu, cov, rf)[2], w0, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x

    if metodo == "Riesgo Mín con retorno objetivo":
        if target_ret is None:
            target_ret = float(mu.mean())
        cons_rt = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: np.dot(w, mu).item() - target_ret},
        )
        res = minimize(lambda w: (w @ cov @ w), w0, method="SLSQP", bounds=bnds, constraints=cons_rt)
        if not res.success:
            res = minimize(lambda w: (w @ cov @ w), w0, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x

    return w0

def construir_frontera(mu, cov, rf, bnds, shorting, npts=50):
    mu = np.asarray(mu, dtype=float).reshape(-1,1)  # 🔧 FIX
    targets = np.linspace(mu.min(), mu.max(), npts)

    puntos = []
    for t in targets:
        w = optimizar_pesos(mu, cov, rf, "Riesgo Mín con retorno objetivo", bnds, shorting, target_ret=t)
        r, v, s = port_stats(w, mu, cov, rf)
        puntos.append((r, v, s))
    return pd.DataFrame(puntos, columns=["ret", "vol", "sharpe"])

def fig_frontera(df, r_opt=None, v_opt=None, s_opt=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not df.empty:
        ax.plot(df["vol"], df["ret"], linewidth=2)
    if r_opt is not None and v_opt is not None:
        ax.scatter([v_opt], [r_opt], s=60)
    ax.set_xlabel("Volatilidad anual")
    ax.set_ylabel("Retorno anual")
    ax.set_title("Frontera eficiente")
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig

def descargar_csv(df: pd.DataFrame, filename: str) -> bytes:
    return df.to_csv(index=True).encode("utf-8")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:

    tickers_str = st.text_area("Tickers", "AAPL, MSFT, NVDA, AMZN, GOOG")
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    start_date = st.date_input("Fecha inicio", value=date.today() - timedelta(days=365*3))
    end_date = st.date_input("Fecha fin", value=date.today())

    intervalo = st.selectbox(
        "Frecuencia de precios",
        options={"Diaria": "1d", "Semanal": "1wk", "Mensual": "1mo"}
    )
    freq_map = {"Diaria": "1d", "Semanal": "1wk", "Mensual": "1mo"}
    intervalo_yf = freq_map[intervalo]
    freq_code = {"1d": "D", "1wk": "W", "1mo": "M"}[intervalo_yf]

    metodo_retorno = st.radio("Método de retornos", ["Simple", "Logarítmico"], index=0)

    metodo = st.selectbox("Método", ["Igualitario", "Mín Var", "Máx Sharpe", "Riesgo Mín con retorno objetivo"])
    rf = st.number_input("Rf", value=0.02)

    shorting = st.checkbox("Shorting")
    min_w = 0.0
    max_w = 1.0
    bounds = (min_w, max_w)

    target_ret = None
    if metodo == "Riesgo Mín con retorno objetivo":
        target_ret = st.number_input("Target", 0.1)

# ----------------------------
# RUN
# ----------------------------
precios = descargar_precios(tickers, start_date, end_date + timedelta(days=1), intervalo_yf)

if precios.empty:
    st.warning("No se pudieron descargar precios.")
    st.stop()

ret_diarios = calcular_retorno_precios(precios, metodo_retorno)

# 🔧 FIX CRÍTICO
ret_diarios = ret_diarios.dropna(axis=1)

if ret_diarios.shape[1] == 0:
    st.warning("No hay activos válidos.")
    st.stop()

mu = ret_diarios.mean().values.reshape(-1, 1)
cov = ret_diarios.cov().values

mu_a, cov_a, _ = anualizar_params(mu, cov, freq_code)

w_opt = optimizar_pesos(mu_a, cov_a, rf, metodo, bounds, shorting, target_ret=target_ret)
w_opt = np.asarray(w_opt, dtype=float)  # 🔧 FIX

labels = ret_diarios.columns.tolist()

ret_opt, vol_opt, sharpe_opt = port_stats(w_opt, mu_a, cov_a, rf)

st.subheader("Pesos óptimos")
tabla_pesos = pd.DataFrame({"Peso": w_opt}, index=labels)
st.dataframe(tabla_pesos.style.format({"Peso": "{:.2%}"}))

st.metric("Retorno", f"{ret_opt:.2%}")
st.metric("Vol", f"{vol_opt:.2%}")
st.metric("Sharpe", f"{sharpe_opt:.2f}")

# ----------------------------
# Frontera
# ----------------------------
df_frontier = construir_frontera(mu_a, cov_a, rf, bounds, shorting)

fig = fig_frontera(df_frontier, ret_opt, vol_opt, sharpe_opt)
st.pyplot(fig)
