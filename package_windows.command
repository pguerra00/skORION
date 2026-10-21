#!/bin/zsh
# Double-click on macOS to rebuild the clean Windows transfer ZIP.
orion_directory="${0:A:h}"
cd "$orion_directory" || exit 1

if [[ -x "$orion_directory/.venv-gui/bin/python" ]]; then
    orion_python="$orion_directory/.venv-gui/bin/python"
elif [[ -x "$orion_directory/.venv/bin/python" ]]; then
    orion_python="$orion_directory/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    orion_python="$(command -v python3)"
else
    print 'Python 3 was not found. Install Python, then run this command again.'
    if [[ -t 0 ]]; then read '?Press Return to close.'; fi
    exit 1
fi

"$orion_python" "$orion_directory/package_windows.py"
orion_exit_code=$?
if (( orion_exit_code != 0 )); then
    print '\nThe Windows ZIP could not be created. Review the error above.'
    if [[ -t 0 ]]; then read '?Press Return to close.'; fi
    exit "$orion_exit_code"
fi

print '\nCopy dist/scikitORION-windows.zip to your Windows computer.'
print 'Extract it, run setup_windows.bat once, then use launch_gui.bat.'
if [[ -t 1 ]]; then
    /usr/bin/open -R "$orion_directory/dist/scikitORION-windows.zip"
fi
