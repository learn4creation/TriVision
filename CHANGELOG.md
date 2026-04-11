# Changelog

All notable changes to this project will be documented in this file.

## [3.0.0] - 2026-04-11

### Added
- **CVIPtools-Style Help Window:** Replaced the flat documentation viewer with a professional two-panel Help window — collapsible Contents tree on the left, rich styled HTML content on the right — mirroring the classic CVIPtools documentation interface. Accessible via **Help → Documentation**.
- **Comprehensive Documentation (`docs.md`):** Wrote full in-application documentation covering every feature: interface layout, all 112+ algorithm categories (Filtering, Edge Detection, Morphology, Segmentation, Feature Extraction, Color Processing, Frequency Domain, Compression, Restoration, Transforms, and TriVision Fusion), Pipeline Builder, Batch Processor, Webcam, Quality Metrics, A/B Compare, Keyboard Shortcuts, Plugin SDK, Settings, Themes, FAQ, and Troubleshooting.
- **Settings Dialog (`File → Settings`):** Added a persistent user settings system. A new `SettingsManager` singleton loads and saves `~/.trivision/trivision_settings.json`. The Settings dialog allows the user to configure the **default recording directory** with a live path display and folder-picker browse button.
- **Recording Save Popup:** When a Webcam recording stops (via Stop Record button or camera stop), a styled notification popup appears showing the saved **file name** and **folder location**, with "Show Details" for the full path.
- **`SettingsManager` Class:** Singleton settings manager that persists user preferences to `~/.trivision/trivision_settings.json`. Automatically creates the directory and file on first launch.
- **Docker Support:** Added `Dockerfile`, `Dockerfile.web`, `docker-compose.yml`, and `.dockerignore` for running TriVision as a containerised desktop app (via X11 forwarding) or headless web UI.
- **Windows Installer (`installer.iss`):** Added Inno Setup script to package the Nuitka-compiled executable into a professional `TriVision_Setup.exe` one-click installer with Start Menu and optional desktop shortcut.
- **Build Script (`build.bat`):** Automated build pipeline using Nuitka to compile Python to a standalone Windows `.exe` with all assets bundled.
- **App Icon & Assets (`assets/`):** Added `logo.png` and `logo.ico` for application window icon and installer icon.

### Changed
- **Recording Directory:** Default recording path is now read from `SettingsManager` (configurable via Settings dialog) instead of being hardcoded to `~/Videos/TriVision/`.
- **`_toggle_record` Method:** Now stores the saved file path and triggers the save popup on stop; improved error message when video writer fails to open.
- **File → Settings Menu:** Rebuilt the File menu to include **Settings…** entry before Exit, cleanly separated by a menu separator.
- **Help Menu Wiring:** `_show_docs()` now opens the new `HelpWindow` instead of the old flat `DocumentationWindow`.
- **`.gitignore`:** Expanded to cover `user_installers/`, all `*.exe` binaries, `data/outputs/`, `data/uploads/`, `main.build/`, `main.dist/`, Jupyter checkpoints, and `docker-compose.override.yml`.

### Removed
- **`DocumentationWindow` class:** Replaced entirely by the new two-panel `HelpWindow`.

---

## [1.0.4] - 2026-04-09

### Added
- **Asynchronous Camera Engine:** Live algorithms now execute on a background `ThreadPoolExecutor`. The physical camera I/O feed now natively maintains a stable 30 FPS regardless of algorithm computation latency, eliminating video freezing.
- **Auto-Maximized Launching:** Handlers upgraded to automatically force the `MainWindow` state to detect and lock to the absolute resolution bounds of the host monitor.

### Changed
- **Deployment Pipeline Upgrade:** `build.bat` overhauled to compile the Python environment completely to C++ binaries utilizing Nuitka (bypassing PyInstaller). Final output is cleanly sandboxed into the `dist\TriVision` folder.
- **Inno Setup Script Pathing:** `installer.iss` routing updated to target the Nuitka sandbox explicitly, streamlining the final `TriVision_Setup.exe` generation.
- **Strict UI CSS Constraints:** Re-architected the `WebcamTab` button grid using a standardized padding matrix and a hard-stop `min-width: 100px` to permanently fix dynamic text cutoff issues (e.g., characters vanishing off-bounds) during button state-swaps. 
- **Developer Tracking Restraint:** Refined local code documentation and variable logic text strings to meet academic presentation standards.

### Fixed
- **Zombie Uninstallation Bug:** Hard-wired the primary Qt `closeEvent` hook to trigger `os._exit(0)`. This completely destroys orphaned concurrent algorithm threads directly from the CPU, finally fixing the bug where the application refused to cleanly uninstall.
- **Hardware Lockup Error:** Fixed the underlying crash where `Stop` requests on heavy algorithms prevented `cv2.VideoCapture` from executing `release()`, eliminating the persistent physical webcam LED glow bug.
- **QSplitter Disappearing Thresholds:** Initialized `setChildrenCollapsible(False)` to prevent the user from accidentally swiping the side-panel layouts to a `0px` invisible width configuration.
- **Console Telemetry Spam:** Injected critical `os.environ` bypasses to locally silence noisy `MS Sans Serif` fallback errors and violently stabilize the `OpenCV` streaming connection exclusively through Microsoft `DSHOW`.
