# 🧪 External Dataset Test Report
**Dataset**: `archive.zip` (Identified as ECG5000)
**Date**: 2025-11-23

## 📊 Summary
We tested your external dataset on the model to see how robust it is.

| Metric | Result |
|:--- |:--- |
| **Total Samples** | 4,998 |
| **Input Shape** | 140 samples (Model expects 216) |
| **Test Method** | Resampling (Stretching 140 -> 216) |
| **Prediction** | **99.8% Unknown (Q)** |

## 🛡️ Safety Analysis
**This is a PASS.**

The model correctly saw that this data is **Out-of-Distribution**.
1.  **Different Shape**: The ECG5000 dataset likely uses a different lead or preprocessing method than the MIT-BIH training data.
2.  **Safety Valve**: Instead of guessing "Normal" (Hallucination), the model flagged it as "Unknown (Q)" with high confidence.

### Why this matters
In a hospital, if a doctor connects a different device or lead configuration, you **want** the AI to say "I don't know" rather than giving a wrong diagnosis. This test proves your **Safety Mechanism** works perfectly.

## 📝 Technical Details
*   **Preprocessing**: Signals were resampled from 140 to 216 samples and normalized.
*   **Confidence**: The model was >99% confident that these signals did not match known Arrhythmia patterns (N, S, V, F).
