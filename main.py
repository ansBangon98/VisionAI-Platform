
import os
import sys
import fcntl
import platform


from ui.analytics_demo import main

APP_DIR_NAME = "VisionAnalytics"

currplatform = platform.system().lower()
_IS_WINDOWS = currplatform == "windows"

# HELPERS FUNCTION
def get_base_dir() -> str:
    """Return the root data directory, creating it if needed."""
    if _IS_WINDOWS:
        base = os.path.join(os.environ.get("PROGRAMDATA", "C:\\ProgramData"), APP_DIR_NAME)
    else:
        # Prefer /var/lib when running as a service; fallback to ~/.local/share
        if os.geteuid() == 0:          # type: ignore[attr-defined]
            base = f"/var/lib/{APP_DIR_NAME}"
        else:
            base = os.path.join(os.path.expanduser("~"), ".local", "share", APP_DIR_NAME)

    os.makedirs(base, exist_ok=True)
    return base

# Single-instance guard using fcntl file lock
_lock_fp = None

def _acquire_single_instance_lock() -> None:
    global _lock_fp
    try:
        base = get_base_dir()
        os.makedirs(base, exist_ok=True)
        lock_path = os.path.join(base, f"{APP_DIR_NAME}.lock")
        _lock_fp = open(lock_path, "a+")
        try:
            fcntl.flock(_lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                _lock_fp.close()
            except Exception:
                pass
            sys.exit(0)
        else:
            # Record PID for observability; lock releases automatically on exit
            try:
                _lock_fp.seek(0)
                _lock_fp.truncate()
                _lock_fp.write(str(os.getpid()))
                _lock_fp.flush()
            except Exception:
                pass
    except Exception as e:
        pass

if __name__ == "__main__":
    _acquire_single_instance_lock()
    main()
