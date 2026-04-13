@echo off
:: ============================================================
:: start_phone_camera.bat
:: Sets up ADB port forwarding from the Android phone's
:: IP Webcam stream and launches the detector.
::
:: One-time phone setup:
::   1. Enable USB debugging:
::      Settings > About phone > tap "Build number" 7 times
::      Settings > Developer options > USB debugging ON
::   2. Install "IP Webcam" from the Play Store
::   3. Open IP Webcam > tap "Start server"
::   4. Connect phone via USB and allow USB debugging when prompted
::
:: One-time PC setup:
::   1. Download ADB platform-tools (no Android Studio needed):
::      https://developer.android.com/tools/releases/platform-tools
::   2. Extract to C:\platform-tools
::   3. Add C:\platform-tools to your system PATH
:: ============================================================

set PHONE_PORT=8080
set VENV=.venv\Scripts\activate.bat

echo Checking ADB...
where adb >nul 2>&1
if errorlevel 1 (
    echo ERROR: adb not found. Download platform-tools from:
    echo   https://developer.android.com/tools/releases/platform-tools
    pause
    exit /b 1
)

echo Checking connected devices...
adb devices
echo.

echo Forwarding port %PHONE_PORT% over USB...
adb forward tcp:%PHONE_PORT% tcp:%PHONE_PORT%
if errorlevel 1 (
    echo ERROR: ADB forward failed. Make sure:
    echo   - Phone is connected via USB
    echo   - USB debugging is enabled
    echo   - IP Webcam app is running on the phone
    pause
    exit /b 1
)

echo Port forwarding active: localhost:%PHONE_PORT% -> phone:%PHONE_PORT%
echo.

echo Activating virtual environment...
call %VENV%

echo Starting detector...
python detect.py --source http://localhost:%PHONE_PORT%/video

echo.
echo Cleaning up ADB forward...
adb forward --remove tcp:%PHONE_PORT%
