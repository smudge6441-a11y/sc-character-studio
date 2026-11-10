"""Build the standalone Windows app for sharing.

    python tools/build_release.py

Makes dist/SC-Character-Studio.exe (a desktop app: Python, the libraries and the page bundled
inside, shown in its own window; no install needed) and dist/SC-Character-Studio.zip
(the .exe + README + LICENSE), ready to upload as a GitHub release.
Needs:  pip install pyinstaller pywebview zstandard numpy pillow

Nothing personal goes in: your character library, settings.json and the data/ cache stay out
(each PC rebuilds its cache from its own game install). The .exe keeps everything it writes in a
StudioData folder next to itself.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project folder
DIST = os.path.join(HERE, "dist")
NAME = "SC-Character-Studio"


def main():
    work = tempfile.mkdtemp(prefix="scstudio-build-")
    try:
        subprocess.check_call([
            sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed", "--name", NAME,
            "--icon", os.path.join(HERE, "tools", "app.ico"),
            "--add-data", os.path.join(HERE, "web") + os.pathsep + "web",
            *[a for m in ("server", "heads", "facefit", "items", "hairmatch") for a in ("--hidden-import", "studio." + m)],
            "--collect-all", "webview",
            "--exclude-module", "tkinter", "--exclude-module", "matplotlib",
            "--workpath", os.path.join(work, "build"), "--specpath", work, "--distpath", DIST,
            os.path.join(HERE, "SC Character Studio.py"),
        ])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    exe = os.path.join(DIST, NAME + ".exe")
    out = os.path.join(DIST, NAME + ".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(exe, "SC Character Studio/%s.exe" % NAME)
        z.write(os.path.join(HERE, "README.md"), "SC Character Studio/README.md")
        z.write(os.path.join(HERE, "LICENSE"), "SC Character Studio/LICENSE")
    print("built %s (%.1f MB) and %s" % (exe, os.path.getsize(exe) / 1e6, out))


if __name__ == "__main__":
    main()
