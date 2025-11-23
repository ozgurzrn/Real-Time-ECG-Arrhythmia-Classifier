# 🎓 Interview Prep Guide: ECG Arrhythmia Classifier

**Goal**: Help you explain this project confidently in a job interview.

---

## 1. The "Elevator Pitch" (30 Seconds)

"I built a production-grade deep learning system to detect heart arrhythmias in real-time. I used a **1D ResNet** architecture trained on the MIT-BIH database.

The most interesting part was the **validation strategy**. I discovered that standard random splits led to data leakage (99% accuracy but poor generalization). I implemented a **patient-level split**, which dropped accuracy to 70% but gave a true measure of real-world performance. I also added **Grad-CAM** for explainability so doctors can trust the model."

---

## 2. Key Technical Decisions (The "Why")

### Q: Why ResNet1D? Why not LSTM or Transformer?
**A**:
*   **Efficiency**: CNNs (ResNet) are much faster for inference than RNNs/LSTMs.
*   **Feature Extraction**: ResNets are excellent at finding local patterns (like the shape of a QRS complex).
*   **Vanishing Gradients**: The "Residual Connections" (skip connections) allow training deep networks without gradient problems.

### Q: Why 1D Convolutions?
**A**: ECG is a time-series signal (1D), not an image (2D). 1D Convolutions slide over the time axis to detect patterns like "sharp spike" or "wide wave".

### Q: How did you handle Class Imbalance?
**A**:
*   **Problem**: 90% of beats are Normal. The model gets lazy and predicts "Normal" for everything.
*   **Solution 1**: **SMOTE** (Synthetic Minority Over-sampling Technique) to generate fake minority samples during training.
*   **Solution 2**: **Class Weights** in the Loss Function (penalize mistakes on arrhythmias 5x more than mistakes on normal beats).

---

## 3. The "Hero Story": Patient Split Discovery 🌟

**This is your strongest talking point.**

*   **The Trap**: Most tutorials split data randomly: `train_test_split(beats)`.
*   **The Issue**: Patient A has a unique heartbeat shape. If Patient A's beats are in both Train and Test, the model memorizes "Patient A" instead of learning "Arrhythmia".
*   **Your Discovery**:
    *   Random Split Test Accuracy: **99%** (Illusion)
    *   Validation on New Patient: **45%** (Reality)
*   **Your Fix**: Split by **Patient ID**. Train on Patients 100-219, Test on Patients 220-234.
*   **Result**: 70% accuracy on unseen patients. Lower number, but **honest and real**.

---

## 4. Code Walkthrough (Where things live)

### `src/model/model.py` (The Brain)
*   **`ResNet1D` class**: Defines the neural network.
*   **`ResidualBlock`**: The building block. It has two conv layers and a "shortcut" that adds the input to the output (`x + out`).

### `src/data/preprocess.py` (The Pipeline)
*   **`denoise_signal`**: Uses a bandpass filter (0.5-50Hz) to remove muscle noise and gravity baseline wander.
*   **`segment_beats`**: Finds R-peaks and cuts a 0.6-second window around them.

### `src/utils/gradcam.py` (The Explanation)
*   **How it works**: It looks at the last convolutional layer. It finds which "pixels" (time points) had the highest activation for the predicted class.
*   **Result**: It highlights the QRS complex or T-wave to show *what* the model looked at.

---

## 5. Common Interview Questions

**Q: How would you deploy this?**
**A**: "I've already containerized it with **Docker**. For production, I'd deploy the Docker container to AWS ECS or Kubernetes. I'd wrap the model in a **FastAPI** endpoint for low-latency requests."

**Q: How do you handle noise?**
**A**: "I implemented a **Signal Quality Index (SQI)**. If the signal is too noisy (SNR < 10dB), the system rejects it and asks the user to re-record, rather than making a bad guess."

**Q: What would you improve next?**
**A**: "I'd train on a larger dataset like **PTB-XL** (18k patients) to improve generalization. The current model struggles with rare fusion beats because the MIT-BIH dataset is small."

**Q: Your model only sees 0.6 seconds. How can it detect AFib?**
**A**: "It doesn't! The Deep Learning model is a **Beat Classifier** (it looks at the shape of a single beat). I built a separate **Rhythm Analysis Module** that calculates RR-intervals over 10 seconds to detect AFib (irregularly irregular rhythm). It's a hybrid system."

**Q: You use `filtfilt`. That's not real-time!**
**A**: "Correct. `filtfilt` is non-causal (looks ahead). I designed this as a **Buffered Real-Time** system (like a bedside monitor that updates every few seconds) where we have the full buffer. For a pacemaker, I would switch to `sosfilt` (causal filtering) to avoid latency."

**Q: 70% accuracy on new patients is low.**
**A**: "It's not low; it's **real**. Most papers claim 99% because they mix patient data (data leakage). I chose to be honest about the domain shift problem. To improve it, I would use **Domain Adaptation** or train on a larger dataset like PTB-XL."

---

## 6. Cheat Sheet for You

*   **ResNet**: Residual Network (skips layers to learn better).
*   **Grad-CAM**: Gradient-weighted Class Activation Mapping (heatmaps).
*   **SMOTE**: Making fake data to balance classes.
*   **Inference**: The act of making a prediction.
*   **Epoch**: One full pass through the training data.
*   **Loss Function**: Cross Entropy (measures how wrong the model is).

**Study this guide, and you will sound like a Senior Engineer.**
