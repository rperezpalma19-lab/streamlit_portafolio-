# streamlit_portafolio.py
# -------------------------------------------------------------
# Portafolio de Acciones con Yahoo Finance + Optimización
# -------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta
from scipy.optimize import minimize
import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Portafolio", page_icon="📈", layout="wide")

st.title("📈 Portafolio de Acciones")
st.caption("Optimización: Mín Var / Máx Sharpe")

# =========================
# HELPERS
# =========================
def to_col(x):
    x = np.array(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x

# =========================
# DATA
# =========================
@st.cache_data
def descargar_precios(tickers, start, end, interval):
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if len(tickers) == 1:
        close = data["Close"].to_frame(tickers[0])
    else:
        close = pd.concat([data[t]["Close"] for t in tickers], axis=1)
        close.columns = tickers

    return close.dropna(how="all")

def calcular_retornos(precios):
    return precios.pct_change().dropna()

def anualizar(mu, cov, freq):
    f = {"D":252, "W":52, "M":12}[freq]
    return mu*f, cov*f

# =========================
# MÉTRICAS
# =========================
def port_stats(w, mu, cov, rf=0):
    w = to_col(w)
    mu = to_col(mu)
    cov = np.array(cov)

    if w.shape[0] != mu.shape[0]:
        raise ValueError(f"Dim mismatch: w{w.shape}, mu{mu.shape}")

    ret = float(w.T @ mu)
    vol = float(np.sqrt(w.T @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan

    return ret, vol, sharpe

# =========================
# OPTIMIZACIÓN
# =========================
def optimizar_pesos(mu, cov, rf, metodo, bounds, shorting, target_ret=None):

    mu = to_col(mu)
    n = mu.shape[0]

    if shorting:
        bnds = tuple((bounds[0], bounds[1]) for _ in range(n))
    else:
        bnds = tuple((max(0, bounds[0]), bounds[1]) for _ in range(n))

    cons = [{"type":"eq","fun":lambda w: np.sum(w)-1}]

    w0 = np.ones(n)/n

    if metodo == "Igualitario":
        return w0

    if metodo == "Mín Var":
        obj = lambda w: w @ cov @ w
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x

    if metodo == "Máx Sharpe":
        def obj(w):
            return -port_stats(w, mu, cov, rf)[2]
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x

    if metodo == "Riesgo Mín con retorno objetivo":
        if target_ret is None:
            target_ret = float(mu.mean())

        cons.append({"type":"eq","fun":lambda w: (w @ mu).item() - target_ret})

        obj = lambda w: w @ cov @ w
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x

    return w0

# =========================
# FRONTERA
# =========================
def frontera(mu, cov, rf, bounds, shorting):
    mu = to_col(mu)

    targets = np.linspace(mu.min(), mu.max(), 50)
    results = []

    for t in targets:
        w = optimizar_pesos(mu, cov, rf, "Riesgo Mín con retorno objetivo", bounds, shorting, t)
        r,v,s = port_stats(w, mu, cov, rf)
        results.append((r,v,s))

    return pd.DataFrame(results, columns=["ret","vol","sharpe"])

# =========================
# SIDEBAR (FIX CLAVE AQUÍ)
# =========================
with st.sidebar:

    tickers = st.text_input("Tickers", "AAPL,MSFT,NVDA").split(",")

    start = st.date_input("Inicio", date.today()-timedelta(days=365*3))
    end = st.date_input("Fin", date.today())

    freq_options = {"Diaria":"1d","Semanal":"1wk","Mensual":"1mo"}
    freq_label = st.selectbox("Frecuencia", list(freq_options.keys()))
    intervalo = freq_options[freq_label]

    freq_code = {"1d":"D","1wk":"W","1mo":"M"}[intervalo]

    metodo = st.selectbox("Método", ["Igualitario","Mín Var","Máx Sharpe","Riesgo Mín con retorno objetivo"])
    rf = st.number_input("Rf", value=0.02)

    bounds = (0.0,1.0)
    shorting = st.checkbox("Shorting")

    target = None
    if metodo == "Riesgo Mín con retorno objetivo":
        target = st.number_input("Target", 0.1)

# =========================
# RUN
# =========================
precios = descargar_precios(tickers, start, end, intervalo)

ret = calcular_retornos(precios)

mu = ret.mean().values
cov = ret.cov().values

mu, cov = anualizar(mu, cov, freq_code)

w = optimizar_pesos(mu, cov, rf, metodo, bounds, shorting, target)

r,v,s = port_stats(w, mu, cov, rf)

# =========================
# OUTPUT
# =========================
st.subheader("Pesos")
st.dataframe(pd.DataFrame({"Peso":w}, index=tickers))

c1,c2,c3 = st.columns(3)
c1.metric("Retorno", f"{r:.2%}")
c2.metric("Vol", f"{v:.2%}")
c3.metric("Sharpe", f"{s:.2f}")

# =========================
# FRONTERA
# =========================
df = frontera(mu, cov, rf, bounds, shorting)

fig, ax = plt.subplots()
ax.plot(df["vol"], df["ret"])
ax.scatter(v,r)
ax.set_xlabel("Volatilidad")
ax.set_ylabel("Retorno")
ax.set_title("Frontera eficiente")

st.pyplot(fig)
