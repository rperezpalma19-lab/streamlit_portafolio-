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

st.set_page_config(
    page_title="Portafolio de Acciones — Yahoo Finance",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Portafolio de Acciones con Yahoo Finance")

# =========================
# FIX 1: DESCARGA ROBUSTA
# =========================
@st.cache_data(show_spinner=False)
def descargar_precios(tickers, start, end, interval):

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

    if data.empty:
        return pd.DataFrame()

    # 1 ticker
    if len(tickers) == 1:
        try:
            return data["Close"].to_frame(tickers[0])
        except:
            return pd.DataFrame()

    # MultiIndex seguro
    if isinstance(data.columns, pd.MultiIndex):
        precios = []
        for t in tickers:
            if t in data.columns.get_level_values(0):
                try:
                    precios.append(data[t]["Close"].rename(t))
                except:
                    pass
        return pd.concat(precios, axis=1)

    # fallback
    if "Close" in data.columns:
        return data["Close"].to_frame(tickers[0])

    return pd.DataFrame()

# =========================
# RETORNOS
# =========================
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

# =========================
# FIX 2: PORT_STATS ROBUSTO
# =========================
def port_stats(w, mu, cov, rf=0.0):

    w = np.asarray(w, dtype=float).reshape(-1, 1)
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    cov = np.asarray(cov, dtype=float)

    ret = float(np.dot(w.T, mu))
    vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan

    return ret, vol, sharpe

# =========================
# OPTIMIZACIÓN
# =========================
def optimizar_pesos(mu, cov, rf, metodo, bounds, shorting, target_ret=None):

    mu = np.asarray(mu, dtype=float).reshape(-1,1)
    cov = np.asarray(cov, dtype=float)

    n = mu.shape[0]

    if shorting:
        lb, ub = bounds
        bnds = tuple((lb, ub) for _ in range(n))
    else:
        lb, ub = bounds
        bnds = tuple((max(0.0, lb), ub) for _ in range(n))

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    w0 = np.ones(n)/n

    if metodo == "Igualitario":
        return w0

    if metodo == "Mín Var":
        res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=bnds, constraints=cons)
        return np.asarray(res.x, dtype=float)

    if metodo == "Máx Sharpe":
        res = minimize(lambda w: -port_stats(w, mu, cov, rf)[2], w0,
                       method="SLSQP", bounds=bnds, constraints=cons)
        return np.asarray(res.x, dtype=float)

    if metodo == "Riesgo Mín con retorno objetivo":
        if target_ret is None:
            target_ret = float(mu.mean())

        cons_rt = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: np.dot(w, mu).item() - target_ret},
        )

        res = minimize(lambda w: w @ cov @ w, w0,
                       method="SLSQP", bounds=bnds, constraints=cons_rt)

        return np.asarray(res.x, dtype=float)

    return w0

# =========================
# FIX 3: SIN FLATTEN
# =========================
def construir_frontera(mu, cov, rf, bnds, shorting, npts=50):

    mu = np.asarray(mu, dtype=float).reshape(-1,1)

    targets = np.linspace(mu.min(), mu.max(), npts)

    puntos = []
    for t in targets:
        w = optimizar_pesos(mu, cov, rf,
                            "Riesgo Mín con retorno objetivo",
                            bnds, shorting, target_ret=t)
        r, v, s = port_stats(w, mu, cov, rf)
        puntos.append((r, v, s))

    return pd.DataFrame(puntos, columns=["ret", "vol", "sharpe"])

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    tickers = [t.strip().upper() for t in st.text_area(
        "Tickers", "AAPL, MSFT, NVDA").split(",") if t.strip()]

    start_date = st.date_input("Inicio", date.today()-timedelta(days=365*3))
    end_date = st.date_input("Fin", date.today())

    freq_map = {"Diaria": "1d", "Semanal": "1wk", "Mensual": "1mo"}
    freq_label = st.selectbox("Frecuencia", list(freq_map.keys()))
    intervalo_yf = freq_map[freq_label]
    freq_code = {"1d":"D","1wk":"W","1mo":"M"}[intervalo_yf]

    metodo = st.selectbox("Método",
        ["Igualitario","Mín Var","Máx Sharpe","Riesgo Mín con retorno objetivo"])

    rf = st.number_input("Rf", value=0.02)

    min_w = 0.0
    max_w = 1.0
    bounds = (min_w, max_w)

    shorting = st.checkbox("Shorting")

    target_ret = None
    if metodo == "Riesgo Mín con retorno objetivo":
        target_ret = st.number_input("Target", 0.1)

# =========================
# RUN
# =========================
precios = descargar_precios(tickers, start_date, end_date, intervalo_yf)

if precios.empty:
    st.error("Error descargando datos")
    st.stop()

ret = calcular_retorno_precios(precios, "Simple")

mu = ret.mean().values.reshape(-1,1)
cov = ret.cov().values

mu_a, cov_a, _ = anualizar_params(mu, cov, freq_code)

w = optimizar_pesos(mu_a, cov_a, rf, metodo, bounds, shorting, target_ret)

r,v,s = port_stats(w, mu_a, cov_a, rf)

st.write("Pesos:", w)
st.write("Ret:", r, "Vol:", v, "Sharpe:", s)

df_front = construir_frontera(mu_a, cov_a, rf, bounds, shorting)

fig, ax = plt.subplots()
ax.plot(df_front["vol"], df_front["ret"])
ax.scatter(v,r)
st.pyplot(fig)
