# 🛡️ Clinical Robustness Report: The "Universal Model" Quest
**Date**: 2025-11-23
**Objective**: Train a model on external data (Shayan/Kaggle) that works on our raw clinical pipeline.

## 🧪 Experiments Summary

| Experiment | Strategy | Validation Acc (Shayan) | Generalization (MIT-BIH) | Key Finding |
| :--- | :--- | :--- | :--- | :--- |
| **1. Baseline** | Fine-tune (No Weights) | **97.2%** | 5/11 Passed | **Lazy Model**: Guessed "Normal" for everything. Missed all arrhythmias. |
| **2. Weighted** | Class Weights | 93.5% | 5/11 Passed | **Better Balance**: Caught Paced (Q) rhythms, but still missed V/S. |
| **3. Robust** | Weights + Augmentation | 90.2% | 5/11 Passed | **Trade-off**: Caught complex V-beats (207/208) but confused Paced (Q) with V. |

## 🧠 Deep Analysis

### 1. The "Preprocessing Gap" (The Silent Killer)
The biggest problem was not the model, but the **Data Geometry**.
*   **Kaggle Data**: Pre-cropped, centered, fixed window.
*   **Our Pipeline**: Dynamic R-peak detection, 0.6s crop.
*   **Result**: The model learned to recognize "Kaggle-shaped" beats. When fed "Our-shaped" beats, it hesitated.

### 2. Augmentation: A Double-Edged Sword
Adding noise and shifting (Experiment 3) helped the model learn **invariant features** for Ventricular beats (it finally caught the complex Bigeminy cases!).
*   **But**: It blurred the fine detail needed to distinguish **Pacemaker Spikes** from noise, causing it to misclassify Paced rhythms as Ventricular.

## 💡 Clinical Conclusion
To build a truly "Universal" clinical model, we cannot rely on pre-processed datasets like Kaggle's `mitbih_train.csv`.
**We must:**
1.  Download the **Raw Signal Data** (not CSVs of beats).
2.  Process it through **Our Own Pipeline** (filtering, peak detection, cropping).
3.  Train on *that* data.

**Status**: Reverted to `models/best_model.pth` (Trained on our pipeline). It remains the most reliable for this specific application.
