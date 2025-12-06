# Real-Time ECG Arrhythmia Classifier

## Overview
This repository contains a framework for real-time arrhythmia detection using 12-lead and single-lead ECG data. The system addresses the challenge of domain shift in biomedical signals—specifically, the performance degradation observed when models trained on standard datasets (e.g., MIT-BIH at 125Hz) are deployed on hardware with different sampling rates or signal characteristics (e.g., INCART at 257Hz).

The project implements a domain adaptation pipeline that fine-tunes a hybrid ResNet1D-BiLSTM architecture, achieving robust generalization across diverse patient populations. It includes an inference dashboard for visualizing model attention and burden analysis.

## Methodology
The core architecture utilizes an ensemble approach to capture both morphological and temporal features of the ECG signal:
* **Spatial Feature Extraction:** ResNet1D blocks process the raw signal to identify morphological anomalies (e.g., wide QRS complexes).
* **Temporal Modelling:** Bidirectional LSTM layers analyze the sequence of heartbeats to detect rhythm-based anomalies.
* **Attention Mechanism:** A soft attention layer highlights specific time-steps contributing to the classification decision.

## Performance
The model was evaluated using a rigorous cross-dataset validation protocol.

| Metric | Score | Dataset | Notes |
| :--- | :--- | :--- | :--- |
| Accuracy | 95.5% | INCART (St. Petersburg) | Post-adaptation |
| Recall (V-Beat) | 0.96 | INCART (St. Petersburg) | Sensitivity to Ventricular Ectopy |
| Precision | 0.94 | INCART (St. Petersburg) | |

## Usage

### 1. Environment Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Acquisition

The `Setup_Test_Data.bat` script handles the ingestion of validation subsets from PhysioNet (NSRDB, SVDB, INCART). It performs automatic resampling to 125Hz.

### 3. Inference

The dashboard allows for the analysis of Holter records. It includes a pre-loaded demonstration mode for rapid validation.

```bash
Run_Clinical_Demo.bat
```

## Risks and Limitations

  * **Domain Dependence:** While domain adaptation improves performance, the model may still exhibit reduced accuracy on datasets with significantly different noise profiles or lead configurations not present in the training set.
  * **False Positives:** High-sensitivity modes may result in over-flagging of artifacts as ectopic beats.
  * **Clinical Validation:** This tool is intended for research and retrospective analysis. It has not been cleared for use as a primary diagnostic tool in a clinical setting.

## License

MIT
