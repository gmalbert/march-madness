@echo off
REM Start Opening Line Capture in background
REM This runs the capture script continuously in a minimized window

title Opening Line Capture
echo.
echo ========================================
echo   Opening Line Capture - Starting
echo ========================================
echo.
echo This window will minimize and run in the background.
echo The script will check for new opening lines every 2 hours.
echo.
echo To stop: Close this window or press Ctrl+C
echo.
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the capture script
python auto_capture_opening_lines.py

pause
