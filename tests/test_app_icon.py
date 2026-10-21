"""Desktop icon resources, Qt integration, and native Windows identity."""
import ctypes
import importlib.util
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None
if QT_AVAILABLE:
    os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'offscreen')
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QIcon, QImage
    from PySide6.QtWidgets import QApplication, QDialog
    from gui_identity import APP_NAME, ICON_DIRECTORY, WINDOWS_APP_ID, prepare_taskbar
    from orion_gui import OrionWindow


class IconBundleTests(unittest.TestCase):
    def test_windows_icon_includes_taskbar_sizes_in_portable_bundle(self):
        from package_windows import build_bundle

        with tempfile.TemporaryDirectory() as directory:
            bundle, _ = build_bundle(Path(directory) / 'app.zip')
            with ZipFile(bundle) as archive:
                for filename in ('gui_identity.py', 'assets/icons/scikitorion.ico',
                                 'assets/icons/scikitorion.png'):
                    self.assertEqual(archive.read('scikitORION/' + filename),
                                     (ROOT / filename).read_bytes())
                ico = archive.read('scikitORION/assets/icons/scikitorion.ico')
        reserved, kind, count = struct.unpack_from('<HHH', ico)
        self.assertEqual((reserved, kind), (0, 1))
        sizes = set()
        for index in range(count):
            width, height = struct.unpack_from('<BB', ico, 6 + 16 * index)
            self.assertEqual(width, height)
            sizes.add(width or 256)
        self.assertEqual(sizes, {16, 20, 24, 32, 40, 48, 64, 128, 256})


@unittest.skipUnless(QT_AVAILABLE, 'Install requirements.txt to test desktop icons')
class DesktopIconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_app_window_and_dialog_use_the_icon_from_another_launch_folder(self):
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                window = OrionWindow()
                self.addCleanup(window.close)
                dialog = QDialog(window)
                self.addCleanup(dialog.close)
                self.assertEqual(self.app.applicationName(), APP_NAME)
                self.assertEqual(self.app.applicationDisplayName(), APP_NAME)
                icon = self.app.windowIcon()
                self.assertFalse(icon.isNull())
                self.assertEqual(window.windowIcon().cacheKey(), icon.cacheKey())
                self.assertEqual(dialog.windowIcon().cacheKey(), icon.cacheKey())
                for size in (16, 24, 32, 48, 256, 512, 1024):
                    with self.subTest(size=size):
                        pixmap = icon.pixmap(QSize(size, size), 1.0)
                        self.assertFalse(pixmap.isNull())
                        self.assertEqual(pixmap.size(), QSize(size, size))
                source = QImage(str(ICON_DIRECTORY / 'scikitorion.png'))
                self.assertEqual(source.size(), QSize(1024, 1024))
                self.assertEqual(source.pixelColor(0, 0).alpha(), 0)
                self.assertGreaterEqual(source.pixelColor(512, 512).alpha(), 250)
                # QPixmap premultiplies alpha; allow its one-step rounding.
                rendered = icon.pixmap(QSize(1024, 1024), 1.0).toImage().pixelColor(512, 512)
                for actual, expected in zip(rendered.getRgb(), source.pixelColor(512, 512).getRgb()):
                    self.assertLessEqual(abs(actual - expected), 1)
            finally:
                os.chdir(previous_directory)

    def test_each_platform_resource_decodes_at_native_sizes(self):
        for filename, sizes in (
            ('scikitorion.ico', (16, 20, 24, 32, 40, 48, 64, 128, 256)),
            ('scikitorion.png', (1024,)),
        ):
            icon = QIcon(str(ICON_DIRECTORY / filename))
            for size in sizes:
                with self.subTest(filename=filename, size=size):
                    self.assertIn(QSize(size, size), icon.availableSizes())
                    self.assertEqual(icon.pixmap(QSize(size, size), 1.0).size(), QSize(size, size))

    def test_windows_identity_and_non_windows_guard(self):
        native = Mock()
        native.shell32.SetCurrentProcessExplicitAppUserModelID.return_value = 0
        with patch.object(sys, 'platform', 'win32'), patch.object(ctypes, 'windll', native, create=True):
            prepare_taskbar()
        native.shell32.SetCurrentProcessExplicitAppUserModelID.assert_called_once_with(WINDOWS_APP_ID)
        native.reset_mock()
        with patch.object(sys, 'platform', 'darwin'), patch.object(ctypes, 'windll', native, create=True):
            prepare_taskbar()
        native.shell32.SetCurrentProcessExplicitAppUserModelID.assert_not_called()

    @unittest.skipUnless(sys.platform == 'win32', 'Native Windows shell check')
    def test_windows_shell_receives_the_app_identity(self):
        prepare_taskbar()
        get_app_id = ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID
        get_app_id.argtypes = [ctypes.POINTER(ctypes.c_wchar_p)]
        get_app_id.restype = ctypes.c_long
        value = ctypes.c_wchar_p()
        self.assertEqual(get_app_id(ctypes.byref(value)), 0)
        try:
            self.assertEqual(value.value, WINDOWS_APP_ID)
        finally:
            release = ctypes.windll.ole32.CoTaskMemFree
            release.argtypes = [ctypes.c_void_p]
            release.restype = None
            release(ctypes.cast(value, ctypes.c_void_p))


if __name__ == '__main__':
    unittest.main()
