@echo off
setlocal DisableDelayedExpansion
rem Launch from this folder even when double-clicked or called from elsewhere.
pushd "%~dp0"
if errorlevel 1 goto directory_error
set "PYTHONUTF8=1"
if not exist ".venv-gui\Scripts\python.exe" (
    echo Run setup_windows.bat first to install the Windows desktop environment.
    goto failed
)
".venv-gui\Scripts\python.exe" -X utf8 "orion_gui.py" %*
set "ORION_EXIT_CODE=%ERRORLEVEL%"
if not "%ORION_EXIT_CODE%"=="0" (
    echo.
    echo scikitORION could not finish. Review the error above.
    echo If dependencies are missing, run setup_windows.bat again.
    if not "%ORION_NO_PAUSE%"=="1" pause
)
popd
exit /b %ORION_EXIT_CODE%

:failed
popd
if not "%ORION_NO_PAUSE%"=="1" pause
exit /b 1

:directory_error
echo Could not open the scikitORION folder.
if not "%ORION_NO_PAUSE%"=="1" pause
exit /b 1
