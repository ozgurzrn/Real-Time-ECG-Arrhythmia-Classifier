import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import sys
import os
import warnings

# --- 1. CONFIGURATION & WARNINGS ---
# Suppress all warnings to keep the dashboard clean
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add src to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import your local modules
from model.ensemble_model import SOTAEnsembleModel
from data.preprocess import denoise_signal
from scipy import signal as scipy_signal

# Page Config
st.set_page_config(
    page_title="ECG Arrhythmia Classifier",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CONSTANTS ---
MODELS_DIR = os.path.join(parent_dir, '..', 'models')
CLASSES = {0: 'Normal (N)', 1: 'Supraventricular (S)', 2: 'Ventricular (V)', 3: 'Fusion (F)', 4: 'Unknown (Q)'}
RISK_LEVELS = {
    0: "Low Risk",
    1: "Medium Risk (Consult Cardiologist)",
    2: "High Risk (Immediate Attention)",
    3: "Moderate Risk",
    4: "Unknown Risk"
}

# --- 3. HELPER FUNCTIONS ---
def get_available_models():
    """Scan models directory for .pth files."""
    if not os.path.exists(MODELS_DIR):
        return []
    return [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]

@st.cache_resource
def load_model_dynamic(model_filename):
    """Load the specific model selected by the user."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    model = SOTAEnsembleModel(num_classes=5)
    try:
        # weights_only=False suppresses the security warning for local files
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        print(f"Model Error: {e}")
        return None, device

def preprocess_input(data):
    """
    Preprocess raw ECG input: Denoise -> Resample -> Normalize.
    """
    # 1. Denoise
    clean = denoise_signal(data, fs=125)
    
    # 2. Resample to 187
    if len(clean) != 187:
        clean = scipy_signal.resample(clean, 187)
        
    # 3. Normalize
    norm = (clean - np.mean(clean)) / (np.std(clean) + 1e-6)
    
    return torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def load_csv_safe(file_obj):
    """
    Smart CSV Loader that handles headers automatically.
    """
    try:
        # 1. Try reading without header first
        df = pd.read_csv(file_obj, header=None)
        
        # 2. Check first row for strings (is it a header?)
        is_str = False
        try:
            float(df.iloc[0, 0])
        except (ValueError, TypeError):
            is_str = True
            
        if is_str:
            # Reload with header (skip first row)
            file_obj.seek(0)
            df = pd.read_csv(file_obj, header=0)
            
        # 3. Force Numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # 4. Drop NaN rows
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        return None

# --- 4. MAIN UI ---
st.title("💓 AI-Powered ECG Arrhythmia Classifier")
st.markdown("### Real-Time Analysis Dashboard")

# Sidebar
st.sidebar.header("Configuration")

# Model Selector
available_models = get_available_models()
if available_models:
    # Default to finetuned if available, else first
    default_idx = 0
    if 'sota_finetuned_russia.pth' in available_models:
        default_idx = available_models.index('sota_finetuned_russia.pth')
    
    selected_model_name = st.sidebar.selectbox("Select AI Model", available_models, index=default_idx)
else:
    st.error("No models found in 'models/' directory!")
    selected_model_name = None

uploaded_file = st.sidebar.file_uploader("Upload ECG CSV", type=['csv'])

if uploaded_file is not None:
    df = load_csv_safe(uploaded_file)
    
    if df is not None and df.shape[0] >= 1:
        row_idx = st.sidebar.slider("Select Heartbeat Index", 0, max(0, df.shape[0]-1), 0)
        
        # Extract signal
        raw_signal = df.iloc[row_idx, :].values
        # Remove any NaNs
        raw_signal = raw_signal[~np.isnan(raw_signal)]
        
        # UI Layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("ECG Signal Visualization")
            tensor_input = preprocess_input(raw_signal)
            processed_signal = tensor_input.squeeze().numpy()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=raw_signal, mode='lines', name='Raw Signal', line=dict(color='gray', width=1, dash='dot')))
            fig.add_trace(go.Scatter(y=processed_signal, mode='lines', name='Denoised', line=dict(color='#ff4b4b', width=2)))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Diagnosis")
            
            if selected_model_name:
                model, device = load_model_dynamic(selected_model_name)
                
                if model:
                    tensor_input = tensor_input.to(device)
                    with torch.no_grad():
                        outputs = model(tensor_input)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                        pred_idx = torch.argmax(outputs, dim=1).item()
                        
                    pred_label = CLASSES[pred_idx]
                    confidence = probs[pred_idx] * 100
                    risk = RISK_LEVELS[pred_idx]
                    
                    # Dynamic Coloring
                    risk_color = "#09ab3b" if pred_idx == 0 else "#ff4b4b"
                    
                    st.markdown(f"""
                    <div style="background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color};">
                        <h2 style="color: white; margin:0;">{pred_label}</h2>
                        <p style="color: gray; margin:0;">Confidence: {confidence:.1f}%</p>
                        <p style="color: {risk_color}; font-weight: bold; margin-top: 10px;">{risk}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    st.markdown("**Probability Distribution**")
                    prob_df = pd.DataFrame({'Class': list(CLASSES.values()), 'Probability': probs})
                    st.bar_chart(prob_df.set_index('Class'))
                else:
                    st.error(f"Failed to load model: {selected_model_name}")
            else:
                st.warning("Please verify models exist in the 'models' folder.")

    else:
        st.error("Could not read data from CSV.")
else:
    st.info("Waiting for file upload...")