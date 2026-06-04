@echo off
title Transcribe App - Setup
echo.
echo ================================================
echo   Transcribe App  Setup  (5-10 min)
echo ================================================
echo.
pause

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [1/3] Installing Python 3.12...
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 ( echo ERROR: Python install failed & pause & exit /b 1 )
) else ( echo [1/3] Python 3.12 OK )

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [2/3] Installing FFmpeg...
    winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
    if errorlevel 1 ( echo ERROR: FFmpeg install failed & pause & exit /b 1 )
) else ( echo [2/3] FFmpeg OK )

echo [3/3] Installing whisperx...
py -3.12 -m pip install --upgrade pip >nul 2>&1
py -3.12 -m pip install whisperx
if errorlevel 1 ( echo ERROR: whisperx install failed & pause & exit /b 1 )

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo GPU detected - installing CUDA PyTorch...
    py -3.12 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
)

:: py -3.12 ?????? python.exe ?????????where py????????????????
for /f "tokens=*" %%P in ('py -3.12 -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%P"
if "%PYTHON_EXE%"=="" ( echo WARNING: python.exe not found & goto done )

echo Creating desktop shortcut...
powershell -Command "$py='%PYTHON_EXE%'; $p='%~dp0_app\transcribe_app.py'; $w='%~dp0_app'; $a=[char]34+$p+[char]34; $d=[Environment]::GetFolderPath('Desktop'); $s=(New-Object -ComObject WScript.Shell).CreateShortcut($d+'\transcribe.lnk'); $s.TargetPath=$py; $s.Arguments=$a; $s.WorkingDirectory=$w; $s.IconLocation='shell32.dll,23'; $s.Save()"

:done
echo.
echo ================================================
echo   Setup complete!
echo   Double-click [transcribe] on your desktop.
echo ================================================
echo.
pause
exit /b 0