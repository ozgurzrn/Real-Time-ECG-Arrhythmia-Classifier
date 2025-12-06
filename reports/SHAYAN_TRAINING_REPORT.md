# 📉 Training Experiment Report: Shayan Fazeli Dataset
**Date**: 2025-11-23
**Experiment**: Weighted Fine-tuning on `mitbih_train.csv`.

## 📊 Results (Attempt 2: With Class Weights)

### 1. Training
*   **Weights**: `[0.5, 5.0, 5.0, 10.0, 2.0]` (Heavily penalizing missed arrhythmias).
*   **Accuracy**: **93.5%** (Lower than 97% unweighted, which is good—it means it tried harder).

### 2. Generalization (The Failure)
*   **Result**: Still **5/11** tests passed on raw MIT-BIH data.
*   **Observation**: It correctly identifies Paced rhythms (Q) but still misses Ventricular (V) and Supraventricular (S) beats in our raw pipeline.

## 🔍 Root Cause: The "Preprocessing Gap"
The failure is likely **not** the model architecture or the data quantity, but the **Data Geometry**.

1.  **Shayan Dataset**: Pre-processed beats. Likely centered on the R-peak with a specific window (e.g., fixed 187 samples).
2.  **Our Pipeline**: We detect R-peaks dynamically and crop a 0.6s window (216 samples).
3.  **The Mismatch**: If our crop is slightly off-center or has a different scale compared to the Kaggle dataset, the model (trained on Kaggle data) sees our data as "alien" or "noisy" and defaults to Normal.

# 📉 Training Experiment Report: Shayan Fazeli Dataset
**Date**: 2025-11-23
**Experiment**: Weighted Fine-tuning on `mitbih_train.csv`.

## 📊 Results (Attempt 2: With Class Weights)

### 1. Training
*   **Weights**: `[0.5, 5.0, 5.0, 10.0, 2.0]` (Heavily penalizing missed arrhythmias).
*   **Accuracy**: **93.5%** (Lower than 97% unweighted, which is good—it means it tried harder).

### 2. Generalization (The Failure)
*   **Result**: Still **5/11** tests passed on raw MIT-BIH data.
*   **Observation**: It correctly identifies Paced rhythms (Q) but still misses Ventricular (V) and Supraventricular (S) beats in our raw pipeline.

## 🔍 Root Cause: The "Preprocessing Gap"
The failure is likely **not** the model architecture or the data quantity, but the **Data Geometry**.

1.  **Shayan Dataset**: Pre-processed beats. Likely centered on the R-peak with a specific window (e.g., fixed 187 samples).
2.  **Our Pipeline**: We detect R-peaks dynamically and crop a 0.6s window (216 samples).
3.  **The Mismatch**: If our crop is slightly off-center or has a different scale compared to the Kaggle dataset, the model (trained on Kaggle data) sees our data as "alien" or "noisy" and defaults to Normal.

## 💡 Conclusion
**More data is only better if it matches your pipeline.**
To use the Shayan dataset effectively, we would need to:
1.  Reverse-engineer exactly how they cropped their beats.
2.  Change our `src/data/preprocess.py` to match their method exactly.

**Decision**: Reverted to `models/best_model.pth` (Trained on our pipeline's data), which is robust and working.
