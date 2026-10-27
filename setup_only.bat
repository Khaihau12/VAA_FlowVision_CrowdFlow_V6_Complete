@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo Dang cai cac thu vien cho VAA FlowVision CrowdFlow V6...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Cai thu vien that bai.
) else (
  echo Cai thu vien thanh cong.
)
pause
