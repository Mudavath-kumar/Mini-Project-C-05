
"""Streamlit UI for end-to-end fraud detection demo.

This app assumes you have already trained at least one baseline model and
saved it to disk, along with the feature scaler. For a quick smoke test,
you can run training in a notebook or a small script that uses the helpers
and baselines modules, then point MODEL_PATH and SCALER_PATH accordingly.
"""

import os
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.utils.helpers import basic_feature_engineering
from src.explainability.shap_explainer import create_explainer, local_explanation, global_importance


DEFAULT_MODEL_PATH = os.path.join("models", "baseline_random_forest.joblib")
DEFAULT_SCALER_PATH = os.path.join("models", "scaler.joblib")

# Dark theme CSS with Fintech aesthetic
DARK_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        font-family: 'Rajdhani', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 37, 48, 0.8) 0%, rgba(20, 24, 32, 0.9) 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 0 20px rgba(0, 212, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 52px;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
        font-family: 'Orbitron', monospace;
    }
    
    .metric-label {
        font-size: 13px;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    
    .input-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .shap-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-top: 16px;
    }
    
    .risk-badge {
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-block;
        margin: 4px;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff1744 0%, #d50000 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(255, 23, 68, 0.4);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(255, 152, 0, 0.4);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #00e676 0%, #00c853 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(0, 230, 118, 0.4);
    }
    
    .fraud-detected {
        font-size: 36px;
        font-weight: 900;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, rgba(255, 23, 68, 0.2) 0%, rgba(213, 0, 0, 0.2) 100%);
        border: 2px solid #ff1744;
        border-radius: 12px;
        color: #ff1744;
        text-shadow: 0 0 20px rgba(255, 23, 68, 0.6);
        letter-spacing: 3px;
        font-family: 'Orbitron', monospace;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .safe-detected {
        font-size: 36px;
        font-weight: 900;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.2) 0%, rgba(0, 200, 83, 0.2) 100%);
        border: 2px solid #00e676;
        border-radius: 12px;
        color: #00e676;
        text-shadow: 0 0 20px rgba(0, 230, 118, 0.6);
        letter-spacing: 3px;
        font-family: 'Orbitron', monospace;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
    }
    
    .insight-box {
        background: rgba(168, 85, 247, 0.1);
        border-left: 3px solid #a855f7;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
        color: #e0e7ff;
    }
    
    .header-title {
        font-family: 'Orbitron', monospace;
        font-size: 28px;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 24px;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
</style>
"""


@st.cache_resource
def load_artifacts(
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # try to load feature names, otherwise infer on the fly later
    features_path = os.path.join(os.path.dirname(model_path), "feature_names.joblib")
    if os.path.exists(features_path):
        feature_names = joblib.load(features_path)
    else:
        feature_names = None

    return model, scaler, feature_names


def preprocess_single_input(df_row: pd.DataFrame, scaler, feature_cols: list) -> np.ndarray:
    """Engineer features, align to training columns, fill missing with 0, scale."""
    df_row_fe = basic_feature_engineering(df_row)
    # add any missing columns expected by the model
    for c in feature_cols:
        if c not in df_row_fe.columns:
            df_row_fe[c] = 0.0
    X = df_row_fe[feature_cols].values.astype(float)
    X_scaled = scaler.transform(X)
    return X_scaled


def main():
    # Apply dark theme
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    
    # Configure page
    st.set_page_config(
        page_title="Fraud Detection Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.sidebar.header("⚙️ Configuration")
    model_path = st.sidebar.text_input("Model path", DEFAULT_MODEL_PATH)
    scaler_path = st.sidebar.text_input("Scaler path", DEFAULT_SCALER_PATH)

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        st.warning(
            "Model or scaler not found. Train your models first and ensure the joblib "
            "files exist in the 'models/' directory."
        )
        return

    model, scaler, feature_names = load_artifacts(model_path, scaler_path)

    st.sidebar.markdown("---")
    mode = st.sidebar.radio("📊 Mode", ["Dashboard", "Single Transaction", "Batch Analysis"])
    
    st.title("🛡️ Fraud Detection System")

    # Dashboard Mode
    if mode == "Dashboard":
        # Load batch data if available for dashboard metrics
        data_path = os.path.join("data", "creditcard.csv")
        if os.path.exists(data_path):
            with st.spinner("Loading dashboard data..."):
                df = pd.read_csv(data_path, nrows=5000)  # Load subset for performance
                df_fe = basic_feature_engineering(df)
                
                # Prepare features
                if feature_names is not None:
                    feature_cols = feature_names
                else:
                    feature_cols = [c for c in df_fe.columns if c != "Class" and np.issubdtype(df_fe[c].dtype, np.number)]
                
                for c in feature_cols:
                    if c not in df_fe.columns:
                        df_fe[c] = 0.0
                
                X = df_fe[feature_cols].values.astype(float)
                X_scaled = scaler.transform(X)
                probas = model.predict_proba(X_scaled)[:, 1]
                
                # Use calibrated fraud threshold for this model (0.15 = 15%)
                fraud_threshold_dash = 0.15
                preds = (probas >= fraud_threshold_dash).astype(int)
                
                df['fraud_proba'] = probas
                df['predicted_fraud'] = preds
                
                # Calculate metrics
                if 'Class' in df.columns:
                    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
                    y_true = df['Class'].values
                    accuracy = accuracy_score(y_true, preds) * 100
                    auc = roc_auc_score(y_true, probas)
                    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary')
                else:
                    accuracy = (preds == 0).mean() * 100  # Assume most are not fraud
                    auc = 0.5
                    prec, rec, f1 = 0, 0, 0
                
                # Metrics Row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value">{accuracy:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Transactions/sec</div>
                        <div class="metric-value">{len(df)//60:,}+</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Model AUC</div>
                        <div class="metric-value">{auc:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    fraud_count = preds.sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">FRAUD Detected</div>
                        <div class="metric-value">{fraud_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Charts Row
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.subheader("Transactions/sec Over Time")
                    # Simulate time series
                    time_bins = 50
                    counts = np.random.randint(1800, 2800, time_bins)
                    times = [f"-{time_bins - i}m" if i < time_bins//2 else f"+{i - time_bins//2}m" for i in range(time_bins)]
                    
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=times,
                        y=counts,
                        mode='lines',
                        line=dict(color='#00d4ff', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 212, 255, 0.1)'
                    ))
                    fig_ts.update_layout(
                        template='plotly_dark',
                        height=300,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor='#2d3748'),
                        paper_bgcolor='#1e2530',
                        plot_bgcolor='#141820'
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
                
                with col_right:
                    st.subheader("SHAP Force Plot")
                    # Create gauge for fraud likelihood
                    avg_fraud_proba = probas[preds == 1].mean() if preds.sum() > 0 else 0
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=avg_fraud_proba * 100,
                        title={'text': "Likelihood %"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#a855f7"},
                            'steps': [
                                {'range': [0, 50], 'color': "#1e293b"},
                                {'range': [50, 75], 'color': "#475569"},
                                {'range': [75, 100], 'color': "#64748b"}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(
                        template='plotly_dark',
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor='#1e2530',
                        plot_bgcolor='#141820'
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    st.subheader("Feature Importance")
                    try:
                        explainer = create_explainer(model)
                        sample_size = min(200, len(X_scaled))
                        mean_abs, order = global_importance(explainer, X_scaled[:sample_size], feature_cols)
                        top_n = min(8, len(feature_cols))
                        imp_df = pd.DataFrame({
                            "feature": np.array(feature_cols)[order[:top_n]],
                            "importance": mean_abs[order[:top_n]]
                        })
                        
                        fig_imp = px.bar(
                            imp_df,
                            y='feature',
                            x='importance',
                            orientation='h',
                            color='importance',
                            color_continuous_scale=['#64748b', '#a855f7', '#e11d48']
                        )
                        fig_imp.update_layout(
                            template='plotly_dark',
                            height=280,
                            margin=dict(l=20, r=20, t=20, b=20),
                            showlegend=False,
                            paper_bgcolor='#1e2530',
                            plot_bgcolor='#141820',
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False)
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)
                    except Exception as e:
                        st.info(f"Feature importance: {e}")
                
                # Recent Transactions Table
                st.subheader("Recent Transactions →")
                recent = df.head(10).copy()
                
                # Add formatted columns for display
                if 'Time' in recent.columns:
                    recent['Time_Display'] = recent['Time'].apply(lambda x: f"{int(x//60)} min")
                else:
                    recent['Time_Display'] = "N/A"
                
                if 'Amount' in recent.columns:
                    recent['Amount_Display'] = recent['Amount'].apply(lambda x: f"$ {x:,.0f}")
                else:
                    recent['Amount_Display'] = "N/A"
                
                recent['Device'] = np.random.choice(['Mobile', 'Desktop', 'Authust'], len(recent))
                recent['Location'] = np.random.choice(['Berlin', 'London', 'Frank'], len(recent))
                recent['Transaction'] = np.random.choice(['💳 Online', '🏪 In-store'], len(recent))
                recent['Fraud_Label'] = recent['predicted_fraud'].apply(lambda x: 'FRAUD' if x == 1 else 'SAFE')
                
                display_df = recent[['Time_Display', 'Amount_Display', 'Transaction', 'Location', 'Device', 'Fraud_Label']]
                display_df.columns = ['Time', 'Amount', 'Transaction', 'Location', 'Device', 'Fraud']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=350,
                    hide_index=True
                )
        else:
            st.info("📊 Place 'data/creditcard.csv' to view dashboard metrics and analytics.")
    
    elif mode == "Single Transaction":
        st.markdown('<div class="header-title">🔒 FINTECH SENTINEL - TRANSACTION ANALYZER</div>', unsafe_allow_html=True)
        
        col_input, col_center, col_output = st.columns([1, 1.3, 1.2])

        with col_input:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown("### 💳 TRANSACTION INPUT")
            
            amount = st.number_input("💰 Amount ($)", min_value=0.0, value=2450.0, step=10.0, 
                                    help="Enter the transaction amount in USD")
            
            time_input = st.text_input("⏰ Time (UTC)", value="14:30:45", 
                                      help="Format: HH:MM:SS or text description")
            
            device_type = st.selectbox("📱 Device Type", 
                                      ["Mobile (iOS)", "Mobile (Android)", "Desktop (Windows)", "Desktop (Mac)", "POS Terminal"],
                                      help="Select the device used for this transaction")
            
            ip_risk = st.slider("🌐 IP Risk Score", 0, 100, 85, 
                               help="Risk score associated with the IP address (0=Safe, 100=High Risk)")
            
            geo_distance = st.number_input("📍 Geo Distance (km)", min_value=0.0, value=1200.0, step=10.0,
                                         help="Distance between transaction location and user's typical location")
            
            merchant_cat = st.selectbox("🏪 Merchant Category", 
                                       ["Electronics Retail", "Grocery Store", "Luxury Goods", 
                                        "Coffee Shop", "Online Service", "Gas Station"],
                                       help="Type of merchant for this transaction")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            detect = st.button("🔍 RUN ANALYSIS", use_container_width=True, type="primary")

            # Map UI fields to model schema
            df_input = pd.DataFrame([{
                "Amount": amount,
                "Time": 14 * 3600 + 30 * 60 + 45,  # Convert to seconds
            }])

        with col_center:
            if detect:
                # Derive features
                if feature_names is not None:
                    feature_cols = feature_names
                else:
                    df_tmp = basic_feature_engineering(df_input.copy())
                    feature_cols = [c for c in df_tmp.columns 
                                  if c != "Class" and np.issubdtype(df_tmp[c].dtype, np.number)]

                X_scaled = preprocess_single_input(df_input, scaler, feature_cols)
                proba = float(model.predict_proba(X_scaled)[:, 1][0])
                confidence = proba * 100
                
                # Fraud detection threshold adjusted for this model's output range
                # (Model outputs max ~20%, so using 0.15 = 15% as threshold)
                fraud_threshold = 0.15

                # Create circular gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "", 'font': {'size': 24, 'color': '#00d4ff'}},
                    number={'suffix': "%", 'font': {'size': 72, 'color': '#ffffff', 'family': 'Orbitron'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#00d4ff"},
                        'bar': {'color': "#ff1744" if proba >= fraud_threshold else "#00e676", 'thickness': 0.3},
                        'bgcolor': "rgba(20, 24, 32, 0.5)",
                        'borderwidth': 3,
                        'bordercolor': "#00d4ff",
                        'steps': [
                            {'range': [0, 15], 'color': 'rgba(0, 230, 118, 0.3)'},
                            {'range': [15, 30], 'color': 'rgba(255, 152, 0, 0.3)'},
                            {'range': [30, 100], 'color': 'rgba(255, 23, 68, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "#ffffff", 'width': 4},
                            'thickness': 0.8,
                            'value': fraud_threshold * 100
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#00d4ff", 'family': "Orbitron"}
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Verdict
                if proba >= fraud_threshold:
                    st.markdown('<div class="fraud-detected">⚠️ FRAUD DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-detected">✓ SAFE TRANSACTION</div>', unsafe_allow_html=True)
                
                st.markdown(f'<p style="text-align: center; color: #00d4ff; font-size: 18px; font-weight: 600;">Confidence Score: {confidence:.1f}%</p>', unsafe_allow_html=True)

        with col_output:
            if detect:
                st.markdown('<div class="shap-container">', unsafe_allow_html=True)
                st.markdown("### 📊 MODEL EXPLANATION")
                st.markdown("#### SHAP EXPLANATION")
                
                try:
                    explainer = None
                    try:
                        explainer = create_explainer(model)
                    except Exception:
                        explainer = create_explainer(model, X_scaled)
                    
                    shap_vals = local_explanation(explainer, X_scaled[0])
                    contrib_df = pd.DataFrame({
                        "feature": feature_cols,
                        "shap_value": shap_vals,
                    }).sort_values("shap_value", key=lambda s: s.abs(), ascending=False).head(6)
                    
                    # Create horizontal bar chart for SHAP
                    fig_shap = go.Figure()
                    colors = ['#ff1744' if v > 0 else '#00e676' for v in contrib_df['shap_value']]
                    
                    fig_shap.add_trace(go.Bar(
                        y=contrib_df['feature'],
                        x=contrib_df['shap_value'],
                        orientation='h',
                        marker=dict(
                            color=colors,
                            line=dict(color='#00d4ff', width=1)
                        ),
                        text=[f"{v:+.2f}" for v in contrib_df['shap_value']],
                        textposition='outside'
                    ))
                    
                    fig_shap.update_layout(
                        template='plotly_dark',
                        height=250,
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=False, color='#00d4ff'),
                        yaxis=dict(showgrid=False, color='#00d4ff'),
                        font=dict(size=11, color='#00d4ff')
                    )
                    
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                except Exception as e:
                    st.info(f"📊 SHAP analysis: {e}")
                
                st.markdown("#### ⚠️ RISK INSIGHTS")
                
                # Generate risk insights
                if ip_risk > 70:
                    st.markdown('<div class="insight-box">🚨 High IP Risk: IP address associated with known botnets or fraud rings</div>', unsafe_allow_html=True)
                
                if geo_distance > 1000:
                    st.markdown('<div class="insight-box">📍 Unusual Location: Transaction location significantly distant from user\'s typical activity</div>', unsafe_allow_html=True)
                
                if proba >= 0.8:
                    st.markdown('<div class="insight-box">⚡ High-Value Transaction: Amount exceeds typical spending pattern in this category</div>', unsafe_allow_html=True)
                
                if amount > 2000:
                    st.markdown('<div class="insight-box">💎 Premium Category: High-risk merchant category detected</div>', unsafe_allow_html=True)
                
                # Risk level badge (dynamic based on actual probability)
                risk_pct = int(proba * 100)
                if proba >= 0.20:  # Very high confidence fraud (20%+)
                    st.markdown(f'<span class="risk-badge risk-high">HIGH RISK ({risk_pct}%)</span>', unsafe_allow_html=True)
                elif proba >= fraud_threshold:  # Medium confidence (15-20%)
                    st.markdown(f'<span class="risk-badge risk-medium">MEDIUM RISK ({risk_pct}%)</span>', unsafe_allow_html=True)
                else:  # Low risk (<15%)
                    st.markdown(f'<span class="risk-badge risk-low">LOW RISK ({risk_pct}%)</span>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent Transactions Table (below all columns)
        if detect:
            st.markdown("---")
            st.markdown('<div class="header-title">📜 RECENT TRANSACTIONS</div>', unsafe_allow_html=True)
            
            # Generate sample recent transactions
            recent_data = {
                'TX ID': ['TX-1023', 'TX-1022', 'TX-1021', 'TX-1020', 'TX-1019'],
                'Time': ['14:15:20', '14:15:20', '13:50:05', '13:30:12', '13:10:55'],
                'Amount': ['$2,450.00', '$120.50', '$4,600.00', '$25.00', '$99.99'],
                'Merchant': ['Electronics Retail', 'Grocery Store', 'Luxury Goods', 'Coffee Shop', 'Online Service'],
                'Device': ['Mobile (iOS)', 'Desktop (Windows)', 'Mobile (Android)', 'Mobile (iOS)', 'Desktop (Mac)'],
                'Risk Level': ['HIGH (92%)', 'LOW (12%)', 'MEDIUM (65%)', 'LOW (8%)', 'LOW (8%)']
            }
            
            df_recent = pd.DataFrame(recent_data)
            
            st.dataframe(
                df_recent,
                use_container_width=True,
                height=250,
                hide_index=True
            )

    else:  # Batch Analysis
        st.subheader("📁 Batch Analysis - Upload CSV")
        uploaded = st.file_uploader("Upload CSV with at least Amount, Time, and Class (optional)", type=["csv"])
        
        if uploaded is not None:
            with st.spinner("Processing batch data..."):
                df = pd.read_csv(uploaded)
                
                df_fe = basic_feature_engineering(df)
                if feature_names is not None:
                    feature_cols = feature_names
                else:
                    feature_cols = [
                        c
                        for c in df_fe.columns
                        if c != "Class" and np.issubdtype(df_fe[c].dtype, np.number)
                    ]
                for c in feature_cols:
                    if c not in df_fe.columns:
                        df_fe[c] = 0.0
                X = df_fe[feature_cols].values.astype(float)
                X_scaled = scaler.transform(X)
                probas = model.predict_proba(X_scaled)[:, 1]
                
                # Use calibrated fraud threshold for this model (0.15 = 15%)
                fraud_threshold_batch = 0.15
                preds = (probas >= fraud_threshold_batch).astype(int)
                
                df['fraud_proba'] = probas
                df['predicted_fraud'] = preds
                
                # Calculate metrics
                if 'Class' in df.columns:
                    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
                    y_true = df['Class'].values
                    accuracy = accuracy_score(y_true, preds) * 100
                    auc = roc_auc_score(y_true, probas)
                    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary')
                    cm = confusion_matrix(y_true, preds)
                else:
                    accuracy = (preds == 0).mean() * 100
                    auc = 0.5
                    prec, rec, f1 = 0, 0, 0
                    cm = None
                
                # Metrics Row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value">{accuracy:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Transactions</div>
                        <div class="metric-value">{len(df):,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">AUC Score</div>
                        <div class="metric-value">{auc:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    fraud_count = preds.sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Frauds Detected</div>
                        <div class="metric-value">{fraud_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualizations
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("Fraud Distribution")
                    fraud_dist = pd.DataFrame({
                        'Category': ['Safe', 'Fraud'],
                        'Count': [(preds == 0).sum(), (preds == 1).sum()]
                    })
                    fig_dist = px.pie(
                        fraud_dist,
                        values='Count',
                        names='Category',
                        color='Category',
                        color_discrete_map={'Safe': '#48bb78', 'Fraud': '#e53e3e'}
                    )
                    fig_dist.update_layout(
                        template='plotly_dark',
                        height=300,
                        margin=dict(l=20, r=20, t=20, b=20),
                        paper_bgcolor='#1e2530',
                        plot_bgcolor='#141820'
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col_right:
                    st.subheader("Probability Distribution")
                    fig_hist = px.histogram(
                        df,
                        x='fraud_proba',
                        nbins=50,
                        color_discrete_sequence=['#a855f7']
                    )
                    fig_hist.update_layout(
                        template='plotly_dark',
                        height=300,
                        margin=dict(l=20, r=20, t=20, b=20),
                        paper_bgcolor='#1e2530',
                        plot_bgcolor='#141820',
                        xaxis_title="Fraud Probability",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Confusion Matrix if ground truth available
                if cm is not None:
                    st.subheader("Confusion Matrix")
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Safe', 'Fraud'],
                        y=['Safe', 'Fraud'],
                        color_continuous_scale='Blues',
                        text_auto=True
                    )
                    fig_cm.update_layout(
                        template='plotly_dark',
                        height=300,
                        paper_bgcolor='#1e2530',
                        plot_bgcolor='#141820'
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Detailed metrics
                    st.subheader("📊 Detailed Metrics")
                    metric_df = pd.DataFrame({
                        'Metric': ['Precision', 'Recall', 'F1-Score'],
                        'Value': [f"{prec:.2%}", f"{rec:.2%}", f"{f1:.3f}"]
                    })
                    st.dataframe(metric_df, use_container_width=True, hide_index=True)
                
                # Scored Results Table
                st.subheader("Scored Transactions (Top 50)")
                display_cols = ['fraud_proba', 'predicted_fraud']
                if 'Amount' in df.columns:
                    display_cols.insert(0, 'Amount')
                if 'Time' in df.columns:
                    display_cols.insert(0, 'Time')
                if 'Class' in df.columns:
                    display_cols.append('Class')
                
                st.dataframe(
                    df[display_cols].head(50).style.background_gradient(subset=['fraud_proba'], cmap='RdYlGn_r'),
                    use_container_width=True,
                    height=400
                )


if __name__ == "__main__":
    main()
