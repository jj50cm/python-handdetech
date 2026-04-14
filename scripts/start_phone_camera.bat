@echo off
:: ============================================================
:: start_phone_camera.bat
:: Launches the phone detector using IP Webcam over Wi-Fi
:: or ADB USB port forwarding.
::
:: One-time phone setup:
::   1. Install "IP Webcam" from the Play Store
::   2. Open IP Webcam > tap "Start server"
::   3. Note the IP address shown on the phone screen
::
:: Usage:
::   start_phone_camera.bat              <- Wi-Fi using PHONE_IP below
::   start_phone_camera.bat usb          <- ADB USB forwarding
:: ============================================================

set PHONE_IP=192.168.0.161
set PHONE_PORT=8080
set VENV=.venv\Scripts\activate.bat
set ADB=adb

if /I "%1"=="usb" goto usb_mode

:wifi_mode
set STREAM_URL=http://%PHONE_IP%:%PHONE_PORT%/video
echo Wi-Fi mode: connecting to %STREAM_URL%
echo Make sure IP Webcam is running on your phone.
echo.
goto launch

:usb_mode
echo USB mode: setting up ADB port forwarding...
%ADB% devices
echo.
%ADB% forward tcp:%PHONE_PORT% tcp:%PHONE_PORT%
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
set STREAM_URL=http://localhost:%PHONE_PORT%/video

:launch
echo Activating virtual environment...
call %VENV%

echo Starting detector...
python detect.py --source %STREAM_URL%

if /I "%1"=="usb" (
    echo.
    echo Cleaning up ADB forward...
    %ADB% forward --remove tcp:%PHONE_PORT%
)
