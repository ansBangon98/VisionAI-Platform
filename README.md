# Vision-Analytics

Computer vision analytics project using Python, PySide6, and GStreamer.

## Requirements

- Ubuntu Linux
- Python `3.10.12`
- Webcam or compatible video input device, if you want to run the camera tests

Run all commands from the project root:

```bash
cd Vision-Analytics
```

## Ubuntu System Dependencies

Update package metadata first:

```bash
sudo apt update
```

Install GObject Introspection and build dependencies:

```bash
sudo apt install -y \
    build-essential \
    pkg-config \
    libffi-dev \
    libcairo2-dev \
    libgirepository-2.0-dev \
    gobject-introspection
```

Install the GStreamer introspection files:

```bash
sudo apt install -y \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0
```

Install the main GStreamer plugins:

```bash
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav
```

Install the common Qt XCB dependency required by PySide6 on Ubuntu:

```bash
sudo apt install -y libxcb-cursor0
```

Ubuntu provides the GStreamer introspection package and X11 video-sink plugins separately. On the tested Ubuntu setup, the GStreamer packages report version `1.28.2` with `gst-launch-1.0`.

## Python Environment

Confirm that Python is the expected version:

```bash
python --version
```

Expected:

```text
Python 3.10.12
```

If `python` points to a different version, use your Python `3.10.12` executable when creating the virtual environment.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade Python packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the project requirements:

```bash
python -m pip install -r requirements.txt
```

Use `python -m pip` instead of plain `pip` so packages are installed into the currently active Python environment.

## Verify GStreamer

Confirm that `gi` is loaded from the project virtual environment:

```bash
python -c "import gi; print(gi.__file__)"
```

The path should point inside this project, for example:

```text
.../Vision-Analytics/.venv/lib/python3.10/site-packages/gi/__init__.py
```

Test the GStreamer Python binding:

```bash
python -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; Gst.init(None); print(Gst.version_string())"
```

Expected output:

```text
GStreamer 1.28.2
```

Test the GStreamer video API:

```bash
python -c "import gi; gi.require_version('Gst', '1.0'); gi.require_version('GstVideo', '1.0'); from gi.repository import Gst, GstVideo; Gst.init(None); print('GStreamer and GstVideo are working')"
```

Expected output:

```text
GStreamer and GstVideo are working
```

## Verify GStreamer Without Python

Check that video output works:

```bash
gst-launch-1.0 videotestsrc ! videoconvert ! ximagesink
```

You should see a moving test-pattern window. Press `Ctrl+C` to stop it.

Check that `ximagesink` is installed:

```bash
gst-inspect-1.0 ximagesink
```

If this displays plugin information, the video sink is installed correctly.

## Run Project Tests

Run the basic Python-only GStreamer test:

```bash
python test/basic_gstreamer.py
```

Check available camera devices:

```bash
ls -l /dev/video*
```

Example output:

```text
crw-rw----+ 1 root video 81, 0 Aug  4 21:26 /dev/video0
crw-rw----+ 1 root video 81, 1 Aug  4 21:26 /dev/video1
crw-rw----+ 1 root video 81, 2 Aug  4 18:56 /dev/video2
crw-rw----+ 1 root video 81, 3 Aug  4 18:56 /dev/video3
```

Run the webcam test:

```bash
python core/camera/gstreamer.py
```

If your USB camera needs MJPEG input, run:

```bash
python core/camera/gstreamer.py --usb-format mjpeg
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'gi'`

Make sure the virtual environment is active:

```bash
source .venv/bin/activate
which python
python --version
python -m pip --version
```

Then reinstall the Python bindings:

```bash
python -m pip install pycairo
python -m pip install PyGObject
```

### `Dependency 'girepository-2.0' is required but not found`

Install the missing Ubuntu package:

```bash
sudo apt install -y libgirepository-2.0-dev
```

Then retry:

```bash
python -m pip install --no-cache-dir PyGObject
```

### `ValueError: Namespace Gst not available`

Install the GStreamer introspection packages:

```bash
sudo apt install -y \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0
```

### `no element "ximagesink"`

Install the X11 video sink plugin:

```bash
sudo apt install -y gstreamer1.0-x
```

Verify it:

```bash
gst-inspect-1.0 ximagesink
```

### Qt XCB Plugin Error

Install the Qt XCB dependency:

```bash
sudo apt install -y libxcb-cursor0
```

Then run the app explicitly with XCB:

```bash
QT_QPA_PLATFORM=xcb python app.py
```

### Accidentally Using System Python

Check which Python and `gi` module are being used:

```bash
which python
python --version
python -c "import sys; print(sys.executable)"
python -c "import gi; print(gi.__file__)"
```

All Python paths should point inside:

```text
Vision-Analytics/.venv/
```

The most important final verification command is:

```bash
source .venv/bin/activate
python -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; Gst.init(None); print(Gst.version_string()); print(gi.__file__)"
```

It should report `GStreamer 1.28.2` and a `gi` path inside the Python `3.10` virtual environment.
