"""Native desktop identity and the approved orbital-diamond app icon."""
from pathlib import Path
import sys
import warnings

from PySide6.QtGui import QIcon


APP_NAME = 'scikitORION'
WINDOWS_APP_ID = 'scikitORION.Desktop'
ICON_DIRECTORY = Path(__file__).resolve().parent / 'assets' / 'icons'


def prepare_taskbar():
    """Give Windows its own taskbar group before QApplication creates any UI."""
    if sys.platform != 'win32':
        return
    import ctypes

    set_app_id = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
    set_app_id.argtypes = [ctypes.c_wchar_p]
    set_app_id.restype = ctypes.c_long
    result = set_app_id(WINDOWS_APP_ID)
    if result < 0:
        warnings.warn(f'Windows could not set the scikitORION taskbar identity '
                      f'(HRESULT 0x{result & 0xFFFFFFFF:08X}).', RuntimeWarning)


def apply_application_identity(app):
    """Set Qt's application icon (including the macOS Dock) before showing UI."""
    # ICO supplies the small native sizes; PNG supplies up to 1024 px for the
    # macOS Dock and high-DPI displays. Paths never depend on the launch folder.
    icon = QIcon(str(ICON_DIRECTORY / 'scikitorion.ico'))
    icon.addFile(str(ICON_DIRECTORY / 'scikitorion.png'))
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setWindowIcon(icon)
    return icon
