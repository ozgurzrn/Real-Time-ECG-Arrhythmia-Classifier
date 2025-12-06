@echo off
TITLE Portfolio Data Ingestion
echo ===================================================
echo   Downloading Portfolio Stress Test Suite
echo   Sources: NSRDB (Normal), SVDB (Supra), INCART (Ventricular)
echo ===================================================

python src/data/fetch_portfolio_suite.py

echo.
PAUSE
