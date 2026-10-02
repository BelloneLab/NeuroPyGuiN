@echo off
setlocal
rem Double-click from a checkout, or download this file alone to bootstrap GitHub.
set "NPG_INSTALL_SCRIPT=%~dp0install\install.ps1"
if exist "%NPG_INSTALL_SCRIPT%" goto install
set "NPG_INSTALL_SCRIPT=%TEMP%\NeuroPyGuiN-install-%RANDOM%-%RANDOM%.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/BelloneLab/NeuroPyGuiN/main/install/install.ps1' -OutFile $env:NPG_INSTALL_SCRIPT"
if errorlevel 1 goto failed
:install
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%NPG_INSTALL_SCRIPT%" %*
if errorlevel 1 goto failed
echo Installation complete.
pause
exit /b 0
:failed
echo Installation stopped. Review the error above and re-run to retry.
pause
exit /b 1
