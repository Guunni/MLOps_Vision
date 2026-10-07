@echo off
chcp 65001 >nul
setlocal
pushd "%~dp0"
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0render_welcome_ab.ps1"
set "exitcode=%ERRORLEVEL%"
echo.
if not "%exitcode%"=="0" (
  echo Rendering failed. Please check the PowerShell message above.
) else (
  echo Rendering completed. Check the Disney video folder.
)
pause
popd
exit /b %exitcode%
