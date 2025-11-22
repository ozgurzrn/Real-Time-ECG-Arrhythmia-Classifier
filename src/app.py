import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
from scipy.signal import find_peaks
from model.model import ResNet1D
from utils.gradcam import GradCAM
from data.preprocess import denoise_signal
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="ECG Arrhythmia Classifier",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1D(num_classes=5)
    try:
        model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    except:
        st.warning("Model not found. Please train the model first.")
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Class Mapping
CLASSES = {0: 'N (Normal)', 1: 'S (Supraventricular)', 2: 'V (Ventricular)', 3: 'F (Fusion)', 4: 'Q (Unknown)'}
COLORS = {0: 'green', 1: 'yellow', 2: 'red', 3: 'orange', 4: 'gray'}

def process_and_predict(signal_data, fs=360):
    # Denoise
    clean_signal = denoise_signal(signal_data, fs)
    
    # Detect R-peaks (simple method for demo if no annotations)
    # In a real app, we might use a better detector like Pan-Tompkins
    peaks, _ = find_peaks(clean_signal, distance=fs*0.4, height=np.mean(clean_signal))
    
    beats = []
    beat_indices = []
    window_samples = int(fs * 0.6 / 2)
    
    for peak in peaks:
        if peak < window_samples or peak + window_samples > len(clean_signal):
            continue
            
        beat = clean_signal[peak - window_samples : peak + window_samples]
        # Normalize
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-6)
        beats.append(beat)
        beat_indices.append((peak - window_samples, peak + window_samples))
        
    if not beats:
        return [], [], clean_signal, peaks
        
    # Batch predict
    beat_tensor = torch.FloatTensor(np.array(beats)).unsqueeze(1).to(device)
    with torch.no_grad():
        outputs = model(beat_tensor)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confidences = torch.max(probs, dim=1).values.cpu().numpy()
        
    results = []
    for i, pred in enumerate(preds):
        results.append({
            'index': i,
            'prediction': CLASSES[pred],
            'confidence': confidences[i],
            'beat': beats[i],
            'range': beat_indices[i],
            'pred_idx': pred
        })
        
    return results, clean_signal, peaks

# Sidebar
st.sidebar.title("❤️ ECG Classifier")
st.sidebar.info("Upload an ECG recording to detect arrhythmias.")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Signal Column)", type=['csv', 'txt'])

# Main Content
st.title("Real-Time ECG Arrhythmia Detection")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] == 1:
            signal_data = df.iloc[:, 0].values
        else:
            # Assume second column is signal if multiple
            signal_data = df.iloc[:, 1].values
            
        st.success("File uploaded successfully!")
        
        # Process
        with st.spinner("Analyzing ECG signal..."):
            results, clean_signal, peaks = process_and_predict(signal_data)
        
        # Summary Statistics
        st.subheader("📊 Detection Summary")
        
        # Count each class
        from collections import Counter
        class_counts = Counter([r['pred_idx'] for r in results])
        total = len(results)
        
        # Create columns for metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Normal (N)", f"{class_counts.get(0, 0)}", f"{class_counts.get(0, 0)/total*100:.1f}%")
        with col2:
            st.metric("Supraventricular (S)", f"{class_counts.get(1, 0)}", f"{class_counts.get(1, 0)/total*100:.1f}%")
        with col3:
            st.metric("Ventricular (V)", f"{class_counts.get(2, 0)}", f"{class_counts.get(2, 0)/total*100:.1f}%")
        with col4:
            st.metric("Fusion (F)", f"{class_counts.get(3, 0)}", f"{class_counts.get(3, 0)/total*100:.1f}%")
        with col5:
            st.metric("Unknown (Q)", f"{class_counts.get(4, 0)}", f"{class_counts.get(4, 0)/total*100:.1f}%")
        
        # Arrhythmia Alert
        arrhythmia_count = class_counts.get(1, 0) + class_counts.get(2, 0) + class_counts.get(3, 0)
        arrhythmia_percentage = (arrhythmia_count / total) * 100
        
        if arrhythmia_percentage > 5:
            st.error(f"⚠️ **ARRHYTHMIA DETECTED**: {arrhythmia_count} abnormal beats ({arrhythmia_percentage:.1f}% of total)")
        elif arrhythmia_percentage > 0:
            st.warning(f"⚡ Minor arrhythmia detected: {arrhythmia_count} abnormal beats ({arrhythmia_percentage:.1f}% of total)")
        else:
            st.success("✅ **NORMAL RHYTHM**: No arrhythmias detected")
            
        # Visualization
        st.subheader("ECG Signal Analysis")
        
        # Limit display to first 10 seconds (3600 samples at 360 Hz)
        display_length = min(3600, len(clean_signal))
        display_signal = clean_signal[:display_length]
        
        # Filter results to only those in the display range
        display_results = [r for r in results if r['range'][1] <= display_length]
        
        st.info(f"Displaying first {display_length/360:.1f} seconds ({len(display_results)} beats). Total: {len(results)} beats classified.")
        
        # Create interactive plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=display_signal, mode='lines', name='ECG Signal', line=dict(color='black', width=1)))
        
        # Add colored regions for beats
        for res in display_results:
            start, end = res['range']
            pred_class = res['pred_idx']
            color = COLORS[pred_class]
            fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.2, layer="below", line_width=0,
                          annotation_text=res['prediction'][0], annotation_position="top left")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Beat Analysis
        st.subheader("Beat-by-Beat Explanation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_beat_idx = st.selectbox("Select Beat to Explain", range(len(results)), format_func=lambda x: f"Beat {x+1}: {results[x]['prediction']}")
            
        if results:
            selected_res = results[selected_beat_idx]
            beat_signal = selected_res['beat']
            
            # Grad-CAM
            grad_cam = GradCAM(model, model.layer4)
            beat_tensor = torch.FloatTensor(beat_signal).unsqueeze(0).unsqueeze(0).to(device)
            cam, _ = grad_cam(beat_tensor, selected_res['pred_idx'])
            
            with col2:
                # Plot Beat with Heatmap
                fig_beat = go.Figure()
                fig_beat.add_trace(go.Scatter(y=beat_signal, mode='lines', name='Beat Signal', line=dict(color='black')))
                
                # Overlay Heatmap (as scatter with color intensity)
                # Normalize cam for color mapping
                fig_beat.add_trace(go.Scatter(
                    y=beat_signal,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=cam,
                        colorscale='Jet',
                        showscale=True,
                        colorbar=dict(title="Importance")
                    ),
                    name='Grad-CAM'
                ))
                
                fig_beat.update_layout(title=f"Grad-CAM Explanation for Beat {selected_beat_idx+1} ({selected_res['prediction']})")
                st.plotly_chart(fig_beat, use_container_width=True)
                
                st.info(f"Confidence: {selected_res['confidence']:.2%}")
                st.write("The heatmap highlights the regions of the ECG beat that the model focused on to make this prediction.")
        
        # PDF Export
        st.markdown("---")
        if st.button("📄 Download PDF Report", type="primary"):
            try:
                from utils.pdf_report import generate_pdf_report
                pdf_filename = generate_pdf_report(results, clean_signal, display_results, display_signal)
                
                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(
                        label="💾 Click to Download",
                        data=pdf_file,
                        file_name=f"ecg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"Error generating PDF: {e}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file containing ECG signal data.")
    
    # Demo button
    if st.button("Load Demo Data"):
        # Generate synthetic data or load a sample if available
        t = np.linspace(0, 10, 3600)
        ecg = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 5 * t) # Dummy
        # In real scenario, load one of the downloaded files
        st.warning("Demo mode using synthetic data. Upload real data for accurate results.")
