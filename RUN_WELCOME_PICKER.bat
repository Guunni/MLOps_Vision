@echo off
setlocal
cd /d "%~dp0"
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0render_welcome_picker.ps1"
echo.
if errorlevel 1 (
  echo [ERROR] 작업이 실패했거나 취소되었습니다.
) else (
  echo [OK] 작업 완료.
)
pause
