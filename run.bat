@echo off
echo ========================================
echo  Soil Detection System - Setup
echo ========================================
echo.

echo Installing required packages...
pip install flask pandas numpy scikit-learn joblib

echo.
echo ========================================
echo  Starting Flask Application...
echo ========================================
echo.

python app.py

pause
