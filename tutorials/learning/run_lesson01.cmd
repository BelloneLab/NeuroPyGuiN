@echo off
setlocal
rem Runs lesson 1 with the neuropyguin environment. Edit DATASET_PATH in the script first.
set "TUTORIAL_PYTHON=%USERPROFILE%\anaconda3\envs\neuropyguin\python.exe"
if defined NEUROPYGUIN_PYTHON set "TUTORIAL_PYTHON=%NEUROPYGUIN_PYTHON%"
if not exist "%TUTORIAL_PYTHON%" (
    echo Python environment not found: "%TUTORIAL_PYTHON%"
    pause
    exit /b 1
)
"%TUTORIAL_PYTHON%" "%~dp0lesson01_inventory.py"
set "TUTORIAL_EXIT=%ERRORLEVEL%"
pause
exit /b %TUTORIAL_EXIT%
