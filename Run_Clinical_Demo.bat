@echo off
TITLE CardioAI - Clinical Mode (Doctor)
echo ===================================================
echo   CardioAI - Clinical Decision Support System
echo   Mode: Universal Model (US/Russia Fine-Tuned)
echo ===================================================

streamlit run src/ui/clinical_dashboard.py

echo.
PAUSE
