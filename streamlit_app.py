"""
Streamlit Dashboard for Federated Learning
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import time
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Config ───────────────────────────────────────────────────────────────────

SERVER_URL = "http://localhost:5000"

st.set_page_config(
    page_title="FedAvg Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
  }
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0d1117 100%);
    color: #e2e8f0;
  }
  .metric-card {
    background: linear-gradient(135deg, #1e2433, #252d3d);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #60a5fa;
  }
  .metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .client-card {
    background: #1a2030;
    border: 1px solid #2d3748;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
  }
  .status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .status-idle    { background: #1e3a2f; color: #34d399; border: 1px solid #34d399; }
  .status-updated { background: #1e2a4a; color: #60a5fa; border: 1px solid #60a5fa; }
  .fedavg-formula {
    font-family: 'Space Mono', monospace;
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 16px;
    font-size: 0.85rem;
    color: #a78bfa;
    margin: 10px 0;
  }
  .stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 10px 20px;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(59,130,246,0.4);
  }
  div[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1f2937;
  }
  .round-indicator {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    color: #f59e0b;
    text-shadow: 0 0 30px rgba(245,158,11,0.5);
    text-align: center;
  }
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs):
    try:
        fn = getattr(requests, method)
        r  = fn(f"{SERVER_URL}{path}", timeout=30, **kwargs)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_status():
    return api("get", "/api/status")


def plotly_dark():
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Space Mono"),
        xaxis=dict(gridcolor="#1f2937", linecolor="#374151"),
        yaxis=dict(gridcolor="#1f2937", linecolor="#374151"),
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 FedAvg Control")
    st.markdown("---")

    st.markdown("### ⚙️ Round Parameters")
    num_clients = st.slider("Clients per Round", 2, 8, 3)
    local_epochs = st.slider("Local Epochs (E)", 1, 20, 5)
    fraction_fit = st.slider("Client Fraction (C)", 0.1, 1.0, 1.0, 0.1)

    st.markdown("---")
    st.markdown("### 🔒 Privacy")
    dp_enabled = st.toggle("Differential Privacy", value=False)
    if dp_enabled:
        epsilon   = st.slider("ε (privacy budget)", 0.1, 10.0, 1.0, 0.1)
        clip_norm = st.slider("Clip Norm", 0.1, 5.0, 1.0, 0.1)

    st.markdown("---")
    col_run, col_reset = st.columns(2)
    with col_run:
        run_round = st.button("▶ Run Round", use_container_width=True)
    with col_reset:
        reset_btn = st.button("↺ Reset", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Auto Training")
    auto_rounds = st.number_input("Auto Rounds", 1, 50, 5)
    auto_train  = st.button("🚀 Train All Rounds", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; color:#4b5563; font-family:monospace; line-height:1.6;">
    FedAvg • McMahan et al. 2017<br>
    CIFAR-10 Simulation<br>
    Non-IID data distribution
    </div>
    """, unsafe_allow_html=True)


# ─── Actions ──────────────────────────────────────────────────────────────────

if reset_btn:
    res = api("post", "/api/reset")
    if "error" not in res:
        st.success("✅ Server reset successfully")
        time.sleep(0.5)
        st.rerun()

if run_round:
    with st.spinner("🔄 Running federated round…"):
        res = api("post", "/api/simulate_round", json={
            "num_clients": num_clients,
            "epochs": local_epochs,
            "fraction": fraction_fit,
        })
    if "error" not in res:
        st.success(f"✅ Round {res['round']} complete | Loss: {res['avg_loss']:.4f} | Acc: {res['avg_accuracy']:.4f}")
        st.rerun()
    else:
        st.error(f"❌ {res['error']}")

if auto_train:
    progress_bar = st.progress(0, text="Initializing…")
    for i in range(auto_rounds):
        progress_bar.progress((i+1)/auto_rounds, text=f"Round {i+1}/{auto_rounds}…")
        res = api("post", "/api/simulate_round", json={
            "num_clients": num_clients,
            "epochs": local_epochs,
            "fraction": fraction_fit,
        })
        if "error" in res:
            st.error(f"Round {i+1} failed: {res['error']}")
            break
        time.sleep(0.2)
    progress_bar.progress(1.0, text="✅ Training complete!")
    st.rerun()


# ─── Main Layout ──────────────────────────────────────────────────────────────

status = get_status()

st.markdown("# Federated Image Classification")
st.markdown("<p style='color:#4b5563;font-family:monospace;font-size:0.85rem;margin-top:-10px;'>Decentralized Training via FedAvg Algorithm · CIFAR-10</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Formula banner
st.markdown("""
<div class="fedavg-formula">
  FedAvg:  w<sub>t+1</sub> = Σ<sub>k∈S</sub> (n<sub>k</sub> / N) · w<sup>k</sup><sub>t+1</sub>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  ClientUpdate: w ← w − η·∇L(w; B)
</div>
""", unsafe_allow_html=True)

# ── KPI Cards
history = status.get("training_history", [])
current_round = status.get("round", 0)
latest_loss   = history[-1]["avg_loss"]     if history else 0.0
latest_acc    = history[-1]["avg_accuracy"] if history else 0.0
best_acc      = max((h["avg_accuracy"] for h in history), default=0.0)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{current_round}</div>
        <div class="metric-label">Training Round</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{latest_loss:.4f}</div>
        <div class="metric-label">Latest Loss</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{latest_acc*100:.1f}%</div>
        <div class="metric-label">Latest Accuracy</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{best_acc*100:.1f}%</div>
        <div class="metric-label">Best Accuracy</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts
if history:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-title">Convergence — Loss</div>', unsafe_allow_html=True)
        rounds = [h["round"] for h in history]
        losses = [h["avg_loss"] for h in history]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds, y=losses, mode="lines+markers",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=6, color="#f59e0b"),
            fill="tozeroy",
            fillcolor="rgba(245,158,11,0.07)",
            name="Avg Loss"
        ))
        fig.update_layout(**plotly_dark(), height=280, margin=dict(l=20, r=20, t=20, b=20),
                          xaxis_title="Round", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">Convergence — Accuracy</div>', unsafe_allow_html=True)
        accs = [h["avg_accuracy"] for h in history]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=rounds, y=accs, mode="lines+markers",
            line=dict(color="#34d399", width=2),
            marker=dict(size=6, color="#34d399"),
            fill="tozeroy",
            fillcolor="rgba(52,211,153,0.07)",
            name="Avg Accuracy"
        ))
        fig2.update_layout(**plotly_dark(), height=280, margin=dict(l=20, r=20, t=20, b=20),
                           xaxis_title="Round", yaxis_title="Accuracy",
                           yaxis=dict(gridcolor="#1f2937", linecolor="#374151", range=[0, 1]))
        st.plotly_chart(fig2, use_container_width=True)

    # Clients per round bar
    st.markdown('<div class="section-title">Clients & Samples per Round</div>', unsafe_allow_html=True)
    clients_per_round = [h["num_clients"]   for h in history]
    samples_per_round = [h["total_samples"] for h in history]

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Bar(x=rounds, y=clients_per_round, name="Clients",
                           marker_color="#6366f1", opacity=0.8), secondary_y=False)
    fig3.add_trace(go.Scatter(x=rounds, y=samples_per_round, name="Samples",
                              line=dict(color="#f472b6", width=2),
                              marker=dict(size=5)), secondary_y=True)
    fig3.update_layout(**plotly_dark(), height=250, margin=dict(l=20, r=20, t=10, b=20))
    fig3.update_yaxes(title_text="Num Clients",  secondary_y=False, gridcolor="#1f2937")
    fig3.update_yaxes(title_text="Total Samples", secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("▶ Click **Run Round** in the sidebar to start federated training.")

# ── Prediction Simulator
st.markdown("---")
st.markdown("### 🔍 Global Model Inference Simulator")
col_pred_left, col_pred_right = st.columns([1, 2])

with col_pred_left:
    classes = status.get("classes", [])
    if st.button("🎲 Simulate Classification", use_container_width=True):
        pred = api("post", "/api/predict")
        if "error" not in pred:
            st.session_state["last_pred"] = pred
        else:
            st.error(pred["error"])

with col_pred_right:
    if "last_pred" in st.session_state:
        p = st.session_state["last_pred"]
        preds = p["all_predictions"]

        fig4 = go.Figure(go.Bar(
            x=[pp["confidence"] for pp in reversed(preds)],
            y=[pp["class"]      for pp in reversed(preds)],
            orientation="h",
            marker=dict(
                color=[pp["confidence"] for pp in reversed(preds)],
                colorscale=[[0, "#1e2433"], [1, "#3b82f6"]],
            ),
            text=[f'{pp["confidence"]:.1f}%' for pp in reversed(preds)],
            textposition="outside",
        ))
        fig4.update_layout(**plotly_dark(), height=320, margin=dict(l=10, r=40, t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

        st.success(f"**Top Prediction:** `{p['top_prediction'].upper()}` — {p['confidence']:.1f}% confidence (Round {p['model_round']})")

# ── Training history table
if history:
    st.markdown("---")
    st.markdown("### 📋 Training Log")
    import pandas as pd
    df = pd.DataFrame(history)[["round", "avg_loss", "avg_accuracy", "num_clients", "total_samples"]]
    df.columns = ["Round", "Avg Loss", "Avg Accuracy", "Clients", "Total Samples"]
    df["Avg Loss"]     = df["Avg Loss"].apply(lambda x: f"{x:.4f}")
    df["Avg Accuracy"] = df["Avg Accuracy"].apply(lambda x: f"{x:.2%}")
    st.dataframe(df, use_container_width=True, hide_index=True)

# ── Auto-refresh
st.markdown("---")
refresh = st.checkbox("🔄 Auto-refresh (5s)")
if refresh:
    time.sleep(5)
    st.rerun()
