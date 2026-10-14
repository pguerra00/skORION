#!/bin/zsh
# Double-click on macOS after installing requirements.txt in .venv-gui.
orion_directory="${0:A:h}"
cd "$orion_directory" || exit 1
if [[ -x "$orion_directory/.venv-gui/bin/python" ]]; then
    "$orion_directory/.venv-gui/bin/python" "$orion_directory/orion_gui.py" "$@"
else
    print 'Set up the desktop environment first:'
    print '  python3.12 -m venv .venv-gui'
    print '  .venv-gui/bin/python -m pip install -r requirements.txt'
    print 'Then double-click launch_gui.command again.'
    read '?Press Return to close.'
    exit 1
fi
