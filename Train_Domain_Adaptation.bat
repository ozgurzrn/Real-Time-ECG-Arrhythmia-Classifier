@echo off
TITLE ECG Detail Adaptation - Fine Tuning
echo ===================================================
echo   Adapting Model to Russian Domain (INCART)
echo   Target: Improve Ventricular Recall
echo ===================================================

python src/model/fine_tune.py

echo.
PAUSE
