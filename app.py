import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# --------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudGuard AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Professional Card" look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 50px;
        font-size: 18px;
        border-radius: 10px;
    }
    .reportview-container {
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 2. LOAD MODEL & SCALER
# --------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = load_model('final_fraud_model.keras')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_artifacts()

# --------------------------------------------------------------------------------
# 3. SIDEBAR - INPUT PARAMETERS
# --------------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2058/2058768.png", width=100)
st.sidebar.title("Configuration Panel")
st.sidebar.markdown("---")

def user_input_features():
    st.sidebar.subheader("1. Transaction Details")
    type_trans = st.sidebar.selectbox(
        "Type",
        ("CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"),
        index=1
    )
    step = st.sidebar.slider("Time Step (Hour of Day)", 0, 744, 1, help="Hour of the month (1-744)")
    
    st.sidebar.subheader("2. Monetary Details")
    amount = st.sidebar.number_input("Amount ($)", min_value=0.0, value=15000.0, step=100.0)
    
    with st.sidebar.expander("Origin Account Details", expanded=True):
        old_bal_org = st.number_input("Old Balance (Org)", min_value=0.0, value=50000.0)
        new_bal_org = st.number_input("New Balance (Org)", min_value=0.0, value=35000.0)
    
    with st.sidebar.expander("Destination Account Details"):
        old_bal_dest = st.number_input("Old Balance (Dest)", min_value=0.0, value=0.0)
        new_bal_dest = st.number_input("New Balance (Dest)", min_value=0.0, value=0.0)
    
    return type_trans, step, amount, old_bal_org, new_bal_org, old_bal_dest, new_bal_dest

type_trans, step, amount, old_bal_org, new_bal_org, old_bal_dest, new_bal_dest = user_input_features()

# --------------------------------------------------------------------------------
# 4. MAIN DASHBOARD UI
# --------------------------------------------------------------------------------

# Header Section
st.title("🛡️ FraudGuard AI System")
st.markdown("### Real-time Transaction Monitoring Dashboard")
st.markdown("---")

# Top Metrics Row (Transaction Overview)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Transaction Type", value=type_trans)
with col2:
    st.metric(label="Amount", value=f"${amount:,.2f}")
with col3:
    st.metric(label="Origin Balance Change", value=f"${old_bal_org - new_bal_org:,.2f}")
with col4:
    hour_val = step % 24
    st.metric(label="Time of Day", value=f"{hour_val}:00 hrs")

st.markdown("---")

# --------------------------------------------------------------------------------
# 5. PREDICTION LOGIC & VISUALIZATION
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# 5. PREDICTION LOGIC & VISUALIZATION (Updated for Dark Mode)
# --------------------------------------------------------------------------------

if st.button("🚀 RUN FRAUD ANALYSIS"):
    if model and scaler:
        with st.spinner('Analyzing patterns with Hybrid AI Model...'):
            
            # --- Feature Engineering ---
            errorBalanceOrig = new_bal_org + amount - old_bal_org
            errorBalanceDest = old_bal_dest + amount - new_bal_dest
            hour = step % 24
            
            type_CASH_OUT = 1 if type_trans == 'CASH_OUT' else 0
            type_DEBIT = 1 if type_trans == 'DEBIT' else 0
            type_PAYMENT = 1 if type_trans == 'PAYMENT' else 0
            type_TRANSFER = 1 if type_trans == 'TRANSFER' else 0
            
            features = np.array([[
                type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER,
                errorBalanceOrig, errorBalanceDest, hour
            ]])
            
            # Scaling & Predicting
            features_scaled = scaler.transform(features)
            prediction_prob = model.predict(features_scaled)
            fraud_prob = prediction_prob[0][0]
            risk_score = fraud_prob * 100

            # --- RESULT DASHBOARD ---
            
            # Layout: Left (Text/Status) | Right (Gauge Chart)
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                st.subheader("Analysis Report")
                
                # --- DARK MODE STYLING APPLIED HERE ---
                if fraud_prob > 0.5:
                    # FRAUD CASE: Dark Red Background + Bright Text
                    st.markdown(f"""
                        <div style="
                            background-color: #3b1111; 
                            color: #ffffff; 
                            padding: 20px; 
                            border-radius: 10px; 
                            border: 1px solid #ff4b4b;
                            box-shadow: 0 0 10px rgba(255, 75, 75, 0.3);">
                            <h3 style="color: #ff4b4b; margin-top: 0;">🚨 CRITICAL ALERT: FRAUD DETECTED</h3>
                            <p style="font-size: 16px;">This transaction has been flagged by the AI system.</p>
                            <hr style="border-color: #ff4b4b;">
                            <h4>Why this is flagged?</h4>
                            <ul style="color: #ffcccc;">
                                <li><strong>High Risk Score:</strong> <span style="color: #ff4b4b; font-size: 18px;">{risk_score:.2f}%</span></li>
                                <li><strong>Pattern Match:</strong> Matches known malicious signatures (ANN+LSTM).</li>
                                <li><strong>Anomaly:</strong> Irregular balance discrepancy detected.</li>
                            </ul>
                            <div style="background-color: #5c1818; padding: 10px; border-radius: 5px; margin-top: 15px;">
                                <strong>⚠️ Recommended Action:</strong> Block transaction immediately.
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # SAFE CASE: Dark Green Background + Bright Text
                    st.markdown(f"""
                        <div style="
                            background-color: #0d2e18; 
                            color: #ffffff; 
                            padding: 20px; 
                            border-radius: 10px; 
                            border: 1px solid #00cc96;
                            box-shadow: 0 0 10px rgba(0, 204, 150, 0.3);">
                            <h3 style="color: #00cc96; margin-top: 0;">✅ SAFE TRANSACTION</h3>
                            <p style="font-size: 16px;">This transaction appears normal.</p>
                            <hr style="border-color: #00cc96;">
                            <h4>Assessment</h4>
                            <ul style="color: #ccffdd;">
                                <li><strong>Low Risk Score:</strong> <span style="color: #00cc96; font-size: 18px;">{risk_score:.2f}%</span></li>
                                <li><strong>Behavior:</strong> Consistent with standard user activity.</li>
                            </ul>
                            <div style="background-color: #154525; padding: 10px; border-radius: 5px; margin-top: 15px;">
                                <strong>👍 Action:</strong> Process transaction normally.
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            with res_col2:
                # Gauge Chart (Dark Mode Optimized Colors)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_score,
                    title = {'text': "Fraud Risk Score", 'font': {'color': 'white'}}, # White Title
                    number = {'font': {'color': 'white'}}, # White Number
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#ff4b4b" if risk_score > 50 else "#00cc96"}, # Neon Red/Green
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "white",
                        'steps' : [
                            {'range': [0, 50], 'color': "#1a1a1a"},
                            {'range': [50, 80], 'color': "#333333"},
                            {'range': [80, 100], 'color': "#4d4d4d"}
                        ],
                        'threshold' : {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_score
                        }
                    }
                ))
                
                # Make Chart Background Transparent for Dark Mode
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white"},
                    height=300, 
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Model files not found. Please upload .keras and .pkl files.")