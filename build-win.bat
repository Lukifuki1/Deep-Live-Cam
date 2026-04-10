@echo off
REM Build script for Deep-Live-Cam Windows executable
REM Requires Python 3.11+ and all requirements installed

echo ========================================
echo Building Deep-Live-Cam for Windows...
echo ========================================

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Install PyInstaller
echo Installing PyInstaller...
pip install pyinstaller

REM Create models directory placeholder (models are downloaded at runtime)
if not exist models\nul mkdir models

REM Build the executable
echo Building executable with PyInstaller...
pyinstaller deep_live_cam.spec --clean --noconfirm

echo ========================================
echo Build complete!
echo Executable is in dist\DeepLiveCam.exe
echo ========================================

pause