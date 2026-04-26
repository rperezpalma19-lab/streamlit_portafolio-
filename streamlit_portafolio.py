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
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1) if x.ndim == 1 else x

# =========================
# DATA (ROBUSTO)
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
        group_by="ticker"
    )

    if data.empty:
        return pd.DataFrame()

    # 1 ticker
    if len(tickers) == 1:
        try:
            return data["Close"].to_frame(tickers[0])
        except:
            return pd.DataFrame()

    # MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        precios = []
        for t in tickers:
            if t in data.columns.get_level_values(0):
                try:
                    serie = data[t]["Close"].rename(t)
                    precios.append(serie)
                except:
                    pass
        if len(precios) == 0:
            return pd.DataFrame()
        return pd.concat(precios, axis=1)

    # Fallback
    if "Close" in data.columns:
        return data["Close"].to_frame(name=tickers[0])

    return pd.DataFrame()

def calcular_retornos(precios):
    return precios.pct_change().dropna()

def anualizar(mu, cov, freq):
    f = {"D":252, "W":52, "M":12}[freq]
    return mu*f, cov*f

# =========================
# MÉTRICAS (FIX DEFINITIVO)
# =========================
def port_stats(w, mu, cov, rf=0):

    w = np.asarray(w, dtype=float).reshape(-1, 1)
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    cov = np.asarray(cov, dtype=float)

    if w.shape[0] != mu.shape[0]:
        raise ValueError(f"Dim mismatch: w{w.shape}, mu{mu.shape}")

    ret = float(np.dot(w.T, mu))
    vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan

    return ret, vol, sharpe

# =========================
# OPTIMIZACIÓN
# =========================
def optimizar_pesos(mu, cov, rf, metodo, bounds, shorting, target_ret=None):

    mu = to_col(mu)
    cov = np.asarray(cov, dtype=float)
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
        return np.asarray(res.x, dtype=float)

    if metodo == "Máx Sharpe":
        obj = lambda w: -port_stats(w, mu, cov, rf)[2]
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons)
        return np.asarray(res.x, dtype=float)

    if metodo == "Riesgo Mín con retorno objetivo":
        if target_ret is None:
            target_ret = float(mu.mean())

        cons.append({"type":"eq","fun":lambda w: (np.dot(w, mu).item()) - target_ret})

        obj = lambda w: w @ cov @ w
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons)
        return np.asarray(res.x, dtype=float)

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
# SIDEBAR
# =========================
with st.sidebar:

    tickers = [
        t.strip().upper()
        for t in st.text_input("Tickers", "AAPL,MSFT,NVDA").split(",")
        if t.strip()
    ]

    start = st.date_input("Inicio", date.today()-timedelta(days=365*3))
    end = st.date_input("Fin", date.today())

    freq_options = {"Diaria":"1d","Semanal":"1wk","Mensual":"1mo"}
    freq_label = st.selectbox("Frecuencia", list(freq_options.keys()))
    intervalo = freq_options[freq_label]

    freq_code = {"1d":"D","1wk":"W","1mo":"M"}[intervalo]

    metodo = st.selectbox("Método", ["Igualitario","Mín Var","Máx Sharpe","Riesgo Mín con retorno objetivo"])
    rf = st.number_input("Tasa libre de riesgo", value=0.02)

    bounds = (0.0,1.0)
    shorting = st.checkbox("Permitir shorting")

    target = None
    if metodo == "Riesgo Mín con retorno objetivo":
        target = st.number_input("Retorno objetivo", 0.1)

# =========================
# RUN
# =========================
precios = descargar_precios(tickers, start, end, intervalo)

if precios.empty:
    st.error("No se pudieron descargar precios. Verifica los tickers.")
    st.stop()

ret = calcular_retornos(precios)

mu = ret.mean().values
cov = ret.cov().values

mu, cov = anualizar(mu, cov, freq_code)

w = optimizar_pesos(mu, cov, rf, metodo, bounds, shorting, target)

r,v,s = port_stats(w, mu, cov, rf)

# =========================
# OUTPUT
# =========================
st.subheader("Pesos óptimos")
st.dataframe(pd.DataFrame({"Peso":w}, index=precios.columns))

c1,c2,c3 = st.columns(3)
c1.metric("Retorno", f"{r:.2%}")
c2.metric("Volatilidad", f"{v:.2%}")
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
ax.grid(True)

st.pyplot(fig)
