@echo off
setlocal DisableDelayedExpansion
rem Double-click once after installing 64-bit Python 3.12 for Windows.
pushd "%~dp0"
if errorlevel 1 goto directory_error
set "PYTHONUTF8=1"
set "ORION_PYTHON=py -3.12"
%ORION_PYTHON% -c "import sys, platform, struct; sys.exit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64', 'x86_64') else 1)" >nul 2>&1
if not errorlevel 1 goto python_found
set "ORION_PYTHON=python"
%ORION_PYTHON% -c "import sys, platform, struct; sys.exit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64', 'x86_64') else 1)" >nul 2>&1
if not errorlevel 1 goto python_found
echo Python 3.12 ^(64-bit x86-64^) was not found.
echo Install it from https://www.python.org/downloads/windows/
echo Include the Python launcher or add Python to PATH, then run this file again.
goto failed

:python_found
if exist ".venv-gui\Scripts\python.exe" goto check_environment
if exist ".venv-gui" (
    echo The .venv-gui folder is not a Windows Python environment.
    echo Rename that folder, then run setup_windows.bat again.
    goto failed
)
echo Creating the Windows desktop environment...
%ORION_PYTHON% -m venv ".venv-gui"
if errorlevel 1 goto failed

:check_environment
".venv-gui\Scripts\python.exe" -c "import sys, platform, struct; sys.exit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64', 'x86_64') else 1)"
if errorlevel 1 (
    echo The existing .venv-gui environment needs 64-bit x86-64 Python 3.12.
    echo Rename that folder, then run setup_windows.bat again.
    goto failed
)
echo Installing analysis and desktop dependencies. Internet access is required.
".venv-gui\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv-gui\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 goto failed
".venv-gui\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
echo.
echo Setup complete. Double-click launch_gui.bat to open scikitORION.
popd
if not "%ORION_NO_PAUSE%"=="1" pause
exit /b 0

:failed
echo.
echo Setup did not complete. Review the message above, then try again.
popd
if not "%ORION_NO_PAUSE%"=="1" pause
exit /b 1

:directory_error
echo Could not open the scikitORION folder.
if not "%ORION_NO_PAUSE%"=="1" pause
exit /b 1
