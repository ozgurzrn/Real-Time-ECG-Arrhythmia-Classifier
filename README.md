# 🫀 Real-Time ECG Arrhythmia Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Available-blue?logo=docker&logoColor=white)](Dockerfile)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](demo_notebook.ipynb)

A production-grade deep learning system for real-time ECG arrhythmia detection, featuring explainable AI (Grad-CAM), signal quality assessment, and rigorous patient-level validation.

---

## 🚀 Key Engineering Challenge: The "Patient Split" Discovery

**Most ML projects fail in the real world because they test on the same patients they trained on.**

In this project, I initially achieved **99% test accuracy** using a standard random split. However, when I validated on individual patient records, accuracy dropped to **45%**.

**Root Cause**: The model was memorizing patient-specific ECG morphology rather than learning arrhythmia features.
**Solution**: Implemented a **Patient-Level Train/Test Split** (ensuring patients in the test set are completely unseen during training).
**Result**: 
- **Validation Accuracy**: Improved from 45% → **70%** on unseen patients.
- **Generalization**: Proven robust on external data (PTB Database).

*See [CRITICAL_FINDING.md](CRITICAL_FINDING.md) for the full engineering analysis.*

---

## ✨ Features

*   **Deep Learning Model**: ResNet1D architecture optimized for time-series classification.
*   **Real-Time Analysis**: Processes ECG signals in <50ms.
*   **Explainable AI**: Grad-CAM heatmaps show exactly *why* the model made a prediction.
*   **Clinical Metrics**:
    *   **Pacemaker Detection** (100% accuracy)
    *   **Signal Quality Index (SQI)**
    *   **Rhythm Analysis** (AFib, Bigeminy, Tachycardia)
*   **Production Ready**:
    *   Dockerized container 🐳
    *   PDF Report Generation 📄
    *   Comprehensive Unit Tests 🧪

---

## 🛠️ Quick Start

### Option 1: Google Colab (No Installation)
Try the model instantly in your browser:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](demo_notebook.ipynb)

### Option 2: Docker (Recommended)
```bash
docker build -t ecg-classifier .
docker run -p 8501:8501 ecg-classifier
```

### Option 3: Local Installation
```bash
git clone https://github.com/ozgurzrn/Real-Time-ECG-Arrhythmia-Classifier.git
cd Real-Time-ECG-Arrhythmia-Classifier
pip install -r requirements.txt
streamlit run src/app.py
```

---

## 📊 Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Test Accuracy** | **98.98%** | On balanced test set (22,514 beats) |
| **Validation Accuracy** | **70.0%** | On 10 unseen patients (Real-world scenario) |
| **Inference Time** | **<50ms** | Per 10-second segment |

> **Note**: The discrepancy between test and validation accuracy highlights the challenge of **inter-patient variability**. While the model excels at classifying beats from known populations, generalizing to completely new patients remains a known challenge in ECG analysis. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for a detailed analysis.

---

## 🏗️ Architecture

The system uses a **1D ResNet** (Residual Neural Network) with:
*   **Input**: 1-lead ECG signal (216 samples @ 360Hz)
*   **Backbone**: 3 Residual Blocks with 1D Convolutions
*   **Attention**: Global Average Pooling
*   **Output**: 5 AAMI Classes (N, S, V, F, Q)

![Architecture](https://raw.githubusercontent.com/ozgurzrn/Real-Time-ECG-Arrhythmia-Classifier/main/assets/architecture_diagram.png)

---

## 📁 Project Structure

```
├── src/
│   ├── data/          # Data processing pipeline
│   ├── model/         # PyTorch model definition
│   ├── utils/         # Helper functions (Grad-CAM, Signal Quality)
│   └── app.py         # Streamlit application
├── models/            # Trained model weights
├── notebooks/         # Jupyter notebooks for exploration
├── tests/             # Unit tests
├── Dockerfile         # Container definition
├── demo_notebook.ipynb # Colab demo
└── README.md          # Documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MIT-BIH Database**: George B. Moody & Roger G. Mark
- **PhysioNet**: Goldberger et al. (2000)
- **PyTorch Team**: For the deep learning framework
- **Streamlit**: For the amazing web framework

---

## 📧 Contact

**Project Maintainer**: Özgür E. Zurnacı
- GitHub: [@ozgurzrn](https://github.com/ozgurzrn)
- Email: ozgurqqqppp@gmail.com

**Project Link**: [https://github.com/ozgurzrn/Real-Time-ECG-Arrhythmia-Classifier](https://github.com/ozgurzrn/Real-Time-ECG-Arrhythmia-Classifier)

---

**Made with ❤️ for advancing healthcare AI**
