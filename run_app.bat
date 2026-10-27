@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo VAA FlowVision CrowdFlow V6 Complete
echo Detection + Tracking + Fusion + Flow + Forecast + Anomaly
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
  echo [LOI] Khong tim thay Python trong PATH.
  echo Hay cai Python 3.10 tro len va chon "Add Python to PATH".
  pause
  exit /b 1
)

python -c "import streamlit,pandas,numpy,plotly,cv2,scipy,ultralytics" >nul 2>&1
if errorlevel 1 (
  echo [CAI DAT] Dang cai cac thu vien can thiet. Lan dau co the can Internet...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [LOI] Cai thu vien that bai.
    pause
    exit /b 1
  )
)

echo [KIEM TRA] Chay validation V6...
python validate_v6.py
if errorlevel 1 (
  echo [LOI] Validation V6 that bai. Hay xem thong bao phia tren.
  pause
  exit /b 1
)

echo [BAT DAU] Mo ung dung tai http://localhost:8501
python -m streamlit run app.py
pause
