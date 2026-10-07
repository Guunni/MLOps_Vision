@echo off
setlocal
cd /d "%~dp0"
set "PS=C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" set "PS=powershell.exe"
"%PS%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0render_welcome_picker.ps1"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" goto ok
echo FAILED. Error code: %RC%
pause
exit /b %RC%
:ok
echo DONE.
pause
exit /b 0
