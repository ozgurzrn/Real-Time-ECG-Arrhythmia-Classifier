# 🫀 Real-Time ECG Arrhythmia Classifier

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A deep learning-based application for classifying ECG arrhythmias with **Explainable AI** (Grad-CAM) visualizations. Built with PyTorch and deployed via Streamlit.

---

## ✨ Features

- **🎯 High Accuracy**: 97.87% test accuracy with ResNet1D architecture
- **🔍 Explainable AI**: Grad-CAM heatmaps show which ECG regions influenced predictions
- **📊 Real-Time Analysis**: Instant classification of 5 AAMI arrhythmia types:
  - **N**: Normal
  - **S**: Supraventricular  
  - **V**: Ventricular
  - **F**: Fusion
  - **Q**: Unknown
- **📄 PDF Reports**: Export professional clinical reports
- **🎨 Interactive Dashboard**: User-friendly Streamlit interface
- **⚡ GPU Accelerated**: CUDA support for fast inference

---

## 🚀 Demo

### Quick Start
```bash
# Clone the repository
git clone https://github.com/ozgurzrn/Real-Time-ECG-Arrhythmia-Classifier.git
cd Real-Time-ECG-Arrhythmia-Classifier

# Install dependencies (Python 3.11 recommended)
pip install -r requirements.txt

# Launch the dashboard
streamlit run src/app.py
```

### Using the App
1. **Upload** a CSV file containing ECG signal data
2. **View** automatic arrhythmia detection summary
3. **Explore** interactive ECG visualization with beat classifications
4. **Analyze** individual beats with Grad-CAM explanations
5. **Export** professional PDF reports

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 97.87% |
| **Precision (Weighted)** | 0.98 |
| **Recall (Weighted)** | 0.98 |
| **F1-Score (Weighted)** | 0.98 |

### Class-Specific Performance
- **Normal (N)**: F1 = 0.99 (Precision: 1.00, Recall: 0.98)
- **Ventricular (V)**: F1 = 0.95 (Precision: 0.94, Recall: 0.97)
- **Unknown (Q)**: F1 = 0.97 (Precision: 0.95, Recall: 0.99)
- **Supraventricular (S)**: F1 = 0.84 (Precision: 0.76, Recall: 0.93)
- **Fusion (F)**: F1 = 0.82 (Precision: 0.77, Recall: 0.88)

---

## 🏗️ Architecture

### Model: ResNet1D
- **Input**: Time-series ECG beat (187 samples)
- **Architecture**: 
  - 1D Convolutional layers with residual connections
  - 4 residual blocks (32 → 64 → 128 → 256 filters)
  - Global Average Pooling
  - Fully connected output layer (5 classes)
- **Training**: 
  - SMOTE for class balancing
  - Adam optimizer
  - Cross-entropy loss

### Data Pipeline
```
MIT-BIH Database 
    ↓
Bandpass Filter (0.5-50 Hz)
    ↓
R-Peak Detection
    ↓
Beat Segmentation (±0.3s window)
    ↓
Normalization
    ↓
ResNet1D Classifier
    ↓
Grad-CAM Explanation
```

---

## 📁 Project Structure

```
Real-Time ECG Arrhythmia Classifier/
├── data/
│   ├── raw/              # MIT-BIH database files
│   └── processed/        # Preprocessed .npy files
├── models/               # Trained model checkpoints
├── src/
│   ├── data/            # Data loading & preprocessing
│   │   ├── download_data.py
│   │   ├── preprocess.py
│   │   └── make_dataset.py
│   ├── model/           # Model architecture & training
│   │   ├── model.py
│   │   └── train.py
│   ├── utils/           # Helper utilities
│   │   ├── gradcam.py
│   │   └── pdf_report.py
│   └── app.py           # Streamlit dashboard
├── notebooks/           # Exploration notebooks
├── requirements.txt     # Dependencies
├── LICENSE             # MIT License
└── README.md
```

---

## 🔧 Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (optional, but recommended)

### Step-by-Step

1. **Clone the repository**
```bash
git clone https://github.com/ozgurzrn/Real-Time-ECG-Arrhythmia-Classifier.git
cd Real-Time-ECG-Arrhythmia-Classifier
```

2. **Create virtual environment** (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download & preprocess data**
```bash
python src/data/download_data.py
python src/data/make_dataset.py
```

5. **Train the model** (optional - pre-trained model included)
```bash
python src/model/train.py
```

6. **Launch the dashboard**
```bash
streamlit run src/app.py
```

---

## 📖 Usage

### Command Line Interface

**Download Data:**
```bash
python src/data/download_data.py
```

**Preprocess Data:**
```bash
python src/data/make_dataset.py
```

**Train Model:**
```bash
python src/model/train.py
```

**Run Dashboard:**
```bash
streamlit run src/app.py
```

### Python API

```python
from model.model import ResNet1D
from utils.gradcam import GradCAM
import torch

# Load model
model = ResNet1D(num_classes=5)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

# Make prediction
beat = torch.randn(1, 187)  # Your preprocessed ECG beat
output = model(beat)
prediction = torch.argmax(output, dim=1).item()

# Generate Grad-CAM
grad_cam = GradCAM(model, model.layer4)
heatmap, class_idx = grad_cam(beat)
```

---

## 🎓 Dataset

**MIT-BIH Arrhythmia Database**
- Source: [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- 48 half-hour ECG recordings
- 360 Hz sampling rate
- ~110,000 labeled beats
- 5 AAMI arrhythmia classes

**Citation:**
```
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. 
IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
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

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

**Made with ❤️ for advancing healthcare AI**
