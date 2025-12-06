import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import sys
import os
import warnings
from scipy import signal as scipy_signal

# --- 1. CONFIGURATION & WARNINGS ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import local modules
try:
    from model.ensemble_model import SOTAEnsembleModel
    from data.preprocess import denoise_signal
except (ImportError, ModuleNotFoundError):
    sys.path.append(os.path.join(parent_dir, 'model'))
    sys.path.append(os.path.join(parent_dir, 'data'))
    from model.ensemble_model import SOTAEnsembleModel
    from data.preprocess import denoise_signal

# Page Config
st.set_page_config(
    page_title="CardioAI - Clinical Report",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CONSTANTS ---
MODEL_PATH = os.path.join(parent_dir, '..', 'models', 'sota_finetuned_russia.pth')

CLINICAL_LABELS = {
    0: 'Normal Sinus Rhythm',
    1: 'Supraventricular Ectopy (PAC)',
    2: 'Ventricular Ectopy (PVC)',
    3: 'Fusion Beat',
    4: 'Unknown/Paced'
}

# --- 3. HELPER FUNCTIONS ---
def scroll_to_top():
    js = '''<script>window.scrollTo({ top: 0, behavior: 'smooth' });</script>'''
    components.html(js, height=0)

def scroll_to_bottom():
    js = '''<script>window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });</script>'''
    components.html(js, height=0)

@st.cache_resource
def load_clinical_model():
    """Load the Universal Fine-Tuned Model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SOTAEnsembleModel(num_classes=5)
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Critical Error: Universal Model not found at {MODEL_PATH}")
            return None, device
            
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, device

def preprocess_batch(data_batch):
    processed = []
    for sig in data_batch:
        clean = denoise_signal(sig, fs=125)
        if len(clean) != 187:
            clean = scipy_signal.resample(clean, 187)
        norm = (clean - np.mean(clean)) / (np.std(clean) + 1e-6)
        processed.append(norm)
    return torch.tensor(np.array(processed), dtype=torch.float32).unsqueeze(1)

def load_csv_safe(file_obj):
    try:
        df = pd.read_csv(file_obj, header=None)
        is_str = False
        try:
            float(df.iloc[0, 0])
        except (ValueError, TypeError):
            is_str = True
        
        if is_str:
            file_obj.seek(0)
            df = pd.read_csv(file_obj, header=0)
            
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"CSV Error: {e}")
        return None

# --- 4. MAIN UI ---
st.title("⚕️ CardioAI: Clinical Decision Support")
st.markdown("### Automated Holter Analysis Report")

# Sidebar Configuration
st.sidebar.header("Data Source Configuration")
source_mode = st.sidebar.radio("Input Method", ["Upload Patient File", "Load Demo Case"])

df = None
selected_demo_case = None

if source_mode == "Upload Patient File":
    uploaded_file = st.sidebar.file_uploader("Import Holter Record (CSV)", type=['csv'])
    if uploaded_file is not None:
        df = load_csv_safe(uploaded_file)
else:
    # Demo Mode
    selected_demo_case = st.sidebar.selectbox("Select Patient Case", [
        "Demo: Normal Patient",
        "Demo: Supraventricular (SVT)",
        "Demo: Ventricular (PVC)"
    ])
    
    demo_paths = {
        "Demo: Normal Patient": os.path.join(parent_dir, '..', 'data', 'portfolio_normal.csv'),
        "Demo: Supraventricular (SVT)": os.path.join(parent_dir, '..', 'data', 'portfolio_supra.csv'),
        "Demo: Ventricular (PVC)": os.path.join(parent_dir, '..', 'data', 'portfolio_pvc.csv')
    }
    
    target_path = demo_paths[selected_demo_case]
    
    if os.path.exists(target_path):
        try:
            df = pd.read_csv(target_path, header=None)
            st.sidebar.success(f"Loaded: {selected_demo_case}")
        except Exception as e:
            st.sidebar.error(f"Error loading demo: {e}")
    else:
        st.sidebar.warning("Demo data not found. Please run GET_PORTFOLIO_DATA.bat")

if df is not None:
    # Limit beats
    MAX_BEATS = 5000
    if len(df) > MAX_BEATS:
        st.toast(f"File too large. Analyzing first {MAX_BEATS} beats...", icon="⚠️")
        df = df.iloc[:MAX_BEATS]
        
    st.sidebar.info(f"Analyzing {len(df)} heartbeats...")
    
    # --- ANALYSIS PHASE ---
    # We assume 'model' logic is needed
    with st.spinner("Analyzing Heart Rhythm..."):
        model, device = load_clinical_model()
        
        if model:
            raw_signals = df.iloc[:, :187].values
            
            batch_size = 128
            all_preds = []
            all_probs = []
            
            # Progress bar
            progress_bar = st.sidebar.progress(0)
            
            for i in range(0, len(raw_signals), batch_size):
                batch = raw_signals[i : i+batch_size]
                tensor_batch = preprocess_batch(batch).to(device)
                
                with torch.no_grad():
                    outputs = model(tensor_batch)
                    probs = torch.softmax(outputs, dim=1)
                    # We will filter later, for now just get raw preds and probs
                    
                all_probs.extend(probs.cpu().numpy())
                progress_bar.progress(min((i + batch_size) / len(raw_signals), 1.0))
            
            progress_bar.empty()
            all_probs = np.array(all_probs)
            
            # --- HIGH PRECISION LOGIC ---
            # Thresholding: Any V or S beat with confidence < 0.90 is re-classified as Normal (0)
            # to reduce noise in Demo.
            CONFIDENCE_THRESHOLD = 0.90
            
            final_preds = []
            for i, p_vec in enumerate(all_probs):
                idx = np.argmax(p_vec)
                confidence = p_vec[idx]
                
                if idx in [1, 2]: # S or V
                     if confidence < CONFIDENCE_THRESHOLD:
                         final_preds.append(0) # Downgrade to Normal
                     else:
                         final_preds.append(idx)
                else:
                    final_preds.append(idx)
            
            final_preds = np.array(final_preds)
            
            # --- BURDEN CALCULATION ---
            total_beats = len(final_preds)
            counts = {k: 0 for k in range(5)}
            unique, u_counts = np.unique(final_preds, return_counts=True)
            for k, v in zip(unique, u_counts):
                counts[k] = v
            
            s_burden = (counts[1] / total_beats) * 100
            v_burden = (counts[2] / total_beats) * 100
            ectopic_burden = s_burden + v_burden
            
            # NEW THRESHOLD: > 3.0%
            if ectopic_burden > 3.0:
                status_text = "ARRHYTHMIA DETECTED"
                status_color = "#ff4b4b" # Red
                sub_text = f"Ectopic Burden: {ectopic_burden:.1f}% (>3% Threshold)"
            else:
                status_text = "SINUS RHYTHM"
                status_color = "#09ab3b" # Green
                sub_text = f"Normal Rhythm (Burden: {ectopic_burden:.1f}%)"
            
            # --- AUTO SCROLL LOGIC ---
            # Inject scroll only if Demo Mode
            if selected_demo_case:
                if "Ventricular" in selected_demo_case:
                     scroll_to_bottom()
                elif "Normal" in selected_demo_case:
                     scroll_to_top()
                     
            # Render Header Card
            st.markdown(f"""
            <div style="background-color: #1e1e1e; padding: 25px; border-radius: 15px; text-align: center; border-bottom: 5px solid {status_color}; margin-bottom: 20px;">
                <h1 style="color: {status_color}; margin:0; font-size: 3em;">{status_text}</h1>
                <p style="color: gray; font-size: 1.2em; margin-top: 10px;">{sub_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Arrhythmia Burden")
                normal_count = counts[0]
                ectopic_count = counts[1] + counts[2]
                other_count = counts[3] + counts[4]
                
                labels = ['Normal', 'Ectopic', 'Other']
                values = [normal_count, ectopic_count, other_count]
                colors = ['#09ab3b', '#ff4b4b', '#fabc3f']
                
                fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker=dict(colors=colors))])
                fig_pie.update_layout(template="plotly_dark", height=300, margin=dict(l=20,r=20,t=20,b=20))
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.markdown(f"**PACs**: {counts[1]} | **PVCs**: {counts[2]}")
                
            with col2:
                st.subheader("Evidence Viewer")
                st.caption("Most Significant Anomaly (High Confidence)")
                
                target_idx = -1
                best_conf = -1.0
                anomaly_type = "None"
                
                # Check V
                v_indices = np.where(final_preds == 2)[0]
                if len(v_indices) > 0:
                    v_probs = all_probs[v_indices, 2]
                    best_local = np.argmax(v_probs)
                    target_idx = v_indices[best_local]
                    best_conf = v_probs[best_local]
                    anomaly_type = CLINICAL_LABELS[2]
                elif len(np.where(final_preds == 1)[0]) > 0:
                     # Check S
                     s_indices = np.where(final_preds == 1)[0]
                     s_probs = all_probs[s_indices, 1]
                     best_local = np.argmax(s_probs)
                     target_idx = s_indices[best_local]
                     best_conf = s_probs[best_local]
                     anomaly_type = CLINICAL_LABELS[1]
                
                if target_idx != -1:
                    beat_signal = raw_signals[target_idx]
                    fig_sig = go.Figure()
                    fig_sig.add_trace(go.Scatter(y=beat_signal, mode='lines', name='ECG', line=dict(color='#ff4b4b', width=3)))
                    fig_sig.update_layout(title=f"Sample #{target_idx} | {anomaly_type} ({best_conf*100:.1f}%)", template="plotly_dark", height=300)
                    st.plotly_chart(fig_sig, use_container_width=True)
                else:
                    st.success("No significant anomalies detected.")
                    # Show normal beat
                    if counts[0] > 0:
                        n_idx = np.where(final_preds == 0)[0][0]
                        fig_sig = go.Figure()
                        fig_sig.add_trace(go.Scatter(y=raw_signals[n_idx], mode='lines', line=dict(color='#09ab3b', width=2)))
                        fig_sig.update_layout(title="Reference Normal Beat", template="plotly_dark", height=300)
                        st.plotly_chart(fig_sig, use_container_width=True)

else:
    st.error("No data loaded.")
