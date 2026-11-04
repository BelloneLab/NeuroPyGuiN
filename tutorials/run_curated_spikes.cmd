@echo off
setlocal
rem Use the same Python environment as NeuroPyGuiN, not an unrelated Python.
set "TUTORIAL_PYTHON=%USERPROFILE%\anaconda3\envs\neuropyguin\python.exe"
if defined NEUROPYGUIN_PYTHON set "TUTORIAL_PYTHON=%NEUROPYGUIN_PYTHON%"
if not exist "%TUTORIAL_PYTHON%" (
    echo Python environment not found: "%TUTORIAL_PYTHON%"
    echo Set NEUROPYGUIN_PYTHON to your neuropyguin python.exe, or run the script in your IDE.
    pause
    exit /b 1
)
"%TUTORIAL_PYTHON%" "%~dp0curated_spikes_tutorial.py"
set "TUTORIAL_EXIT=%ERRORLEVEL%"
pause
exit /b %TUTORIAL_EXIT%
