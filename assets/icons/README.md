# Desktop application icon

`scikitorion.png` and `scikitorion.ico` use the approved **Orbital Diamond**
concept from `../branding/concepts/02-orbital-diamond.png`, with its original
colors and transparent corners preserved.

- PNG: 1024 × 1024 pixels for the macOS Dock and high-DPI screens.
- ICO: 16, 20, 24, 32, 40, 48, 64, 128, and 256 pixel images for Windows.

The PNG is a Lanczos downsample of the approved source. The ICO images are
exported directly from that source using Pillow's ICO writer. No image-generation
step is needed to launch the application.

`gui_identity.py` loads both resources into a Qt `QIcon`, sets the application
icon and name, and supplies the main window icon before it is shown. On Windows,
the startup routine also sets `scikitORION.Desktop` as the explicit process
AppUserModelID before creating `QApplication`. The existing launchers on macOS
and Windows both use this startup routine. Keep this directory with the scripts
when copying the application; the Windows transfer ZIP includes these files.

This sets the running application's Dock/taskbar icon. Finder/Explorer icons
for the source launch scripts and the Python interpreter are controlled by the OS.

References: [Qt 6.10 application icons](https://doc.qt.io/qt-6.10/appicon.html)
and [Microsoft process AppUserModelID](https://learn.microsoft.com/en-us/windows/win32/api/shobjidl_core/nf-shobjidl_core-setcurrentprocessexplicitappusermodelid).
