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
from data.streaming import RealTimePipeline
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

def process_and_predict(signal_data, fs=360, use_streaming=False):
    # Import signal quality utilities
    from utils.signal_quality import assess_signal_quality, detect_pacemaker_spikes
    from utils.rhythm_analysis import comprehensive_rhythm_analysis
    
    # Assess overall signal quality
    quality_metrics = assess_signal_quality(signal_data, fs)
    
    # Detect pacemaker
    pacemaker_info = detect_pacemaker_spikes(signal_data, fs)
    
    # Denoise
    if use_streaming:
        # Use Causal Streaming Pipeline
        pipeline = RealTimePipeline(fs)
        # Process entire signal chunk-by-chunk to simulate streaming
        # In a real app, this would happen sample-by-sample
        clean_signal = pipeline.process(signal_data)
        # Note: Streaming pipeline already normalizes, but we might need to adjust scale
        # for visualization if the normalizer hasn't converged yet.
    else:
        # Use Non-Causal Batch Processing (Original)
        clean_signal = denoise_signal(signal_data, fs)
    
    # Detect R-peaks (simple method for demo if no annotations)
    # In a real app, we might use a better detector like Pan-Tompkins
    peaks, _ = find_peaks(clean_signal, distance=fs*0.4, height=np.mean(clean_signal))
    
    # Rhythm analysis
    rhythm_metrics = comprehensive_rhythm_analysis(peaks, fs)
    
    beats = []
    beat_indices = []
    window_samples = int(fs * 0.6 / 2)
    
    for peak in peaks:
        if peak < window_samples or peak + window_samples > len(clean_signal):
            continue
            
        beat = clean_signal[peak - window_samples : peak + window_samples]
        # Normalize
        if use_streaming:
            # Signal is already normalized by StreamNormalizer
            beat = beat 
        else:
            # Batch normalization
            beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-6)
        beats.append(beat)
        beat_indices.append((peak - window_samples, peak + window_samples))
        
    if not beats:
        return [], [], clean_signal, peaks, quality_metrics, pacemaker_info, rhythm_metrics
        
    # Batch predict
    beat_tensor = torch.FloatTensor(np.array(beats)).unsqueeze(1).to(device)
    with torch.no_grad():
        outputs = model(beat_tensor)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confidences = torch.max(probs, dim=1).values.cpu().numpy()
        
    results = []
    for i, pred in enumerate(preds):
        confidence = confidences[i]
        
        # Flag low confidence predictions
        needs_review = confidence < 0.6
        
        # Rhythm-based adjustment hint
        rhythm_flag = ""
        if rhythm_metrics['status'] == 'Complete':
            if rhythm_metrics['is_irregular'] and pred == 0:  # Normal beat in irregular rhythm
                rhythm_flag = "irregular_rhythm"
            if rhythm_metrics['rate_classification'] != 'Normal':
                rhythm_flag = f"{rhythm_flag}_{rhythm_metrics['rate_classification']}".strip('_')
        
        # Check if beat is in pacemaker spike region
        beat_center = beat_indices[i][0] + window_samples
        near_pacemaker = False
        if pacemaker_info['has_pacemaker']:
            for spike_loc in pacemaker_info['spike_locations']:
                if abs(beat_center - spike_loc) < window_samples:
                    near_pacemaker = True
                    break
        
        results.append({
            'index': i,
            'prediction': CLASSES[pred],
            'confidence': confidence,
            'beat': beats[i],
            'range': beat_indices[i],
            'pred_idx': pred,
            'needs_review': needs_review,
            'near_pacemaker': near_pacemaker,
            'rhythm_context': rhythm_flag
        })
        
    return results, clean_signal, peaks, quality_metrics, pacemaker_info, rhythm_metrics

# Sidebar
st.sidebar.title("❤️ ECG Classifier")
st.sidebar.info("Upload an ECG recording to detect arrhythmias.")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Signal Column)", type=['csv', 'txt'])

# Real-Time Mode Toggle
use_streaming = st.sidebar.checkbox("⚡ Enable True Real-Time (Causal)", 
                                  help="Use causal filtering (SOS) and online normalization. Simulates live streaming behavior.")

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
            results, clean_signal, peaks, quality_metrics, pacemaker_info, rhythm_metrics = process_and_predict(signal_data, use_streaming=use_streaming)
        
        # Signal Quality Assessment
        st.subheader("🔍 Signal Quality Assessment")
        
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        
        with col_q1:
            st.metric("Overall Quality", quality_metrics['rating'], 
                     f"{quality_metrics['quality_score']:.0f}/100")
        with col_q2:
            st.metric("SNR (dB)", f"{quality_metrics['snr_db']:.1f}",
                     "Good" if quality_metrics['snr_db'] > 15 else "Low")
        with col_q3:
            baseline_status = "⚠️ Yes" if quality_metrics['has_baseline_wander'] else "✓ No"
            st.metric("Baseline Wander", baseline_status)
        with col_q4:
            artifact_level = "Low" if quality_metrics['artifact_score'] < 30 else ("Medium" if quality_metrics['artifact_score'] < 60 else "High")
            st.metric("Artifacts", artifact_level, f"{quality_metrics['artifact_score']:.0f}/100")
        
        # Rhythm Analysis
        if rhythm_metrics['status'] == 'Complete':
            st.subheader("💓 Rhythm Analysis")
            
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            
            with col_r1:
                hr_color = "🟢" if 60 <= rhythm_metrics['heart_rate_bpm'] <= 100 else "🔴"
                st.metric("Heart Rate", f"{rhythm_metrics['heart_rate_bpm']:.0f} bpm", rhythm_metrics['rate_classification'])
            
            with col_r2:
                regularity = "Regular" if not rhythm_metrics['is_irregular'] else "Irregular"
                st.metric("Rhythm", regularity, f"CV: {rhythm_metrics['rr_variability_cv']:.2f}")
            
            with col_r3:
                st.metric("RR Interval", f"{rhythm_metrics['rr_mean_ms']:.0f} ms", 
                         f"±{rhythm_metrics['rr_std_ms']:.0f} ms")
            
            with col_r4:
                st.metric("Pattern", rhythm_metrics['rhythm_pattern'].split('(')[0].strip())
            
            # Rhythm interpretation
            if rhythm_metrics['interpretation']:
                interpret_text = " • ".join(rhythm_metrics['interpretation'])
                if rhythm_metrics['is_irregular']:
                    st.warning(f"⚡ {interpret_text}")
                elif rhythm_metrics['rate_classification'] != 'Normal':
                    st.info(f"ℹ️ {interpret_text}")
                else:
                    st.success(f"✅ {interpret_text}")
        
        # Pacemaker Detection
        if pacemaker_info['has_pacemaker']:
            st.warning(f"🔋 **Pacemaker Detected**: {pacemaker_info['spike_count']} spikes found (confidence: {pacemaker_info['confidence']*100:.0f}%)")
        
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
        
        # Note about Q class
        q_percentage = (class_counts.get(4, 0) / total) * 100
        if q_percentage > 10:
            st.info(f"ℹ️ **High Unknown (Q) beats ({q_percentage:.1f}%)**: May indicate pacemaker, artifacts, or unusual heart patterns. Clinical review recommended.")
            
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
        
        # Count low-confidence beats
        low_conf_count = sum(1 for r in results if r['needs_review'])
        pacemaker_influenced = sum(1 for r in results if r['near_pacemaker'])
        
        if low_conf_count > 0:
            st.warning(f"⚠️ {low_conf_count} beats have low confidence (<60%) and need clinical review")
        
        if pacemaker_influenced > 0 and pacemaker_info['has_pacemaker']:
            st.info(f"🔋 {pacemaker_influenced} beats near pacemaker spikes")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Format beat options with confidence indicators
            beat_options = []
            for x in range(len(results)):
                beat_label = f"Beat {x+1}: {results[x]['prediction']}"
                if results[x]['needs_review']:
                    beat_label += " ⚠️"
                if results[x]['near_pacemaker']:
                    beat_label += " 🔋"
                beat_options.append(beat_label)
            
            selected_beat_idx = st.selectbox(
                "Select Beat to Explain", 
                range(len(results)), 
                format_func=lambda x: beat_options[x]
            )
            
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
                
                # Detailed confidence info
                conf_color = "green" if selected_res['confidence'] > 0.8 else ("orange" if selected_res['confidence'] > 0.6 else "red")
                st.markdown(f"**Confidence:** :{conf_color}[{selected_res['confidence']:.2%}]")
                
                # Context flags
                if selected_res['needs_review']:
                    st.error("⚠️ **Low Confidence**: This prediction should be reviewed by a clinician")
                
                if selected_res['near_pacemaker']:
                    st.info("🔋 **Near Pacemaker Spike**: Classification may be influenced by pacemaker activity")
                
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
        # Load sample_input.csv if it exists
        import os
        demo_file = "sample_input.csv"
        if os.path.exists(demo_file):
            try:
                df = pd.read_csv(demo_file)
                if df.shape[1] == 1:
                    signal_data = df.iloc[:, 0].values
                else:
                    signal_data = df.iloc[:, 1].values
                
                st.success("Demo data loaded successfully!")
                
                # Process
                with st.spinner("Analyzing ECG signal..."):
                    results, clean_signal, peaks, quality_metrics, pacemaker_info, rhythm_metrics = process_and_predict(signal_data, use_streaming=use_streaming)
                
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
                    
                # Note about Q class
                q_percentage = (class_counts.get(4, 0) / total) * 100
                if q_percentage > 10:
                    st.info(f"ℹ️ **High Unknown (Q) beats ({q_percentage:.1f}%)**: May indicate pacemaker, artifacts, or unusual heart patterns. Clinical review recommended.")
                
                # Visualization (same as before)
                st.subheader("ECG Signal Analysis")
                
                display_length = min(3600, len(clean_signal))
                display_signal = clean_signal[:display_length]
                display_results = [r for r in results if r['range'][1] <= display_length]
                
                st.info(f"Displaying first {display_length/360:.1f} seconds ({len(display_results)} beats). Total: {len(results)} beats classified.")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=display_signal, mode='lines', name='ECG Signal', line=dict(color='black', width=1)))
                
                for res in display_results:
                    start, end = res['range']
                    pred_class = res['pred_idx']
                    color = COLORS[pred_class]
                    fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.2, layer="below", line_width=0,
                                  annotation_text=res['prediction'][0], annotation_position="top left")
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading demo data: {e}")
        else:
            st.warning("Demo file 'sample_input.csv' not found. Please upload your own ECG data.")
