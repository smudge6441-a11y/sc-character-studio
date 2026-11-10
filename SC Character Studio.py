"""SC Character Studio by valkyrie_41.

    python "SC Character Studio.py"            open the desktop app (closing the window quits)
    python "SC Character Studio.py" --serve    use it in a web browser at http://localhost:8752 instead

The window is Microsoft Edge WebView2 (built into Windows 10 and 11) showing the app's local
server, which runs in the background. The packaged .exe is built from this file
(tools/build_release.py).
"""
import os
import socket
import sys
import threading

from studio import paths

paths.migrate_old_layout()
if sys.stdout is None or sys.stderr is None:
    # windowed .exe has no console: keep a small log in its data folder instead (useful for bug reports)
    os.makedirs(paths.HOME, exist_ok=True)
    _log = open(paths.LOG, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log

from studio import server  # noqa: E402

TITLE = "SC Character Studio by valkyrie_41"


def _free_port(preferred):
    """The usual port if it's free, else any free one (e.g. a second copy is already open)."""
    for port in (preferred, 0):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            return s.getsockname()[1]
        except OSError:
            continue
        finally:
            s.close()
    raise RuntimeError("no free port")


def _fatal(msg):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, msg, TITLE, 0x10)
    except Exception:
        print(msg)
    sys.exit(1)


def main():
    if "--serve" in sys.argv:
        return server.main()
    try:
        import webview
    except ImportError:
        _fatal("The window component (pywebview) is missing. Run:  pip install pywebview\n"
               "or start it in your browser with:  python \"SC Character Studio.py\" --serve")
    port = _free_port(server.PORT)
    srv = server.make_server(port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.create_window(TITLE, "http://127.0.0.1:%d/" % port, width=1400, height=900,
                          min_size=(900, 600), background_color="#0d1117", text_select=True)
    try:
        # keep the window's own storage (remembered choices) with the app's data, not a temp profile
        webview.start(gui="edgechromium", private_mode=False, storage_path=os.path.join(paths.CACHE, "window"))
    except Exception as e:  # e.g. the WebView2 runtime is missing on an old Windows 10
        _fatal("Couldn't open the app window (%s).\n\nInstall Microsoft Edge WebView2 Runtime from "
               "https://developer.microsoft.com/microsoft-edge/webview2/ and try again." % e)
    srv.shutdown()


if __name__ == "__main__":
    main()
