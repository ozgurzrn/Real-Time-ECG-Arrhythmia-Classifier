# 📉 Training Experiment Report: ECG5000
**Date**: 2025-11-23
**Experiment**: Fine-tune ResNet1D on `archive.zip` (ECG5000).

## 📊 Results

### 1. Success on New Data
*   **Accuracy**: **99.96%** (after 10 epochs).
*   **Observation**: The model perfectly learned the shape of the new dataset.

### 2. Failure on Old Data (Catastrophic Forgetting)
We tested the fine-tuned model on the original MIT-BIH Arrhythmia Database.
*   **Ventricular Records**: Misclassified as **Normal**.
*   **AFib Records**: Misclassified as **Normal**.

### 3. Root Cause
*   **Data Imbalance**: The `archive.zip` dataset you provided seems to contain **only Normal beats** (Class 1).
*   **Bias**: By training only on this new data, the model "unlearned" what an arrhythmia looks like and biased itself to call everything "Normal".

## 💡 Recommendation
To support both datasets at the same time, we must perform **Joint Training**:
1.  Combine MIT-BIH (Arrhythmias) + ECG5000 (Normals) into one giant dataset.
2.  Train from scratch.

**Current Status**: We have kept your original `best_model.pth` safe. The fine-tuned model is saved as `finetuned_model.pth` if you wish to use it for specific Normal-only tasks.
