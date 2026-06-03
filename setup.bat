@echo off
title Transcription App - First Time Setup

:: Check Python 3.12
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 not found. Installing...
    winget install Python.Python.3.12 --source winget --silent
    if errorlevel 1 (
        echo Failed to install Python. Please install manually from https://python.org
        pause
        exit /b 1
    )
)

:: Check ffmpeg
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo Installing ffmpeg...
    winget install Gyan.FFmpeg --source winget --silent
)

:: Install whisperx
echo Installing whisperx...
py -3.12 -m pip install whisperx
if errorlevel 1 (
    echo whisperx install failed. Check your internet connection.
    pause
    exit /b 1
)

:: Check NVIDIA GPU
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo NVIDIA GPU detected. Installing GPU support...
    py -3.12 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
)

:: Desktop shortcut
powershell -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\transcribe.lnk');$s.TargetPath='py';$s.Arguments='-3.12 \"%~dp0transcribe_app.py\"';$s.WorkingDirectory='%~dp0';$s.IconLocation='shell32.dll,23';$s.Save()"

echo.
echo Setup complete. Desktop shortcut created.
echo Launch the app from your desktop.
echo.
pause
