"""Where things live.

RES   read-only app files (the web/ page): inside the .exe when packaged, else the project folder.
HOME  everything the app writes: the character library, caches, settings and the log.
      Packaged .exe: one "StudioData" folder next to the .exe. From source: the project folder.
"""
import os
import shutil
import sys

FROZEN = getattr(sys, "frozen", False)
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if FROZEN:
    EXE_DIR = os.path.dirname(sys.executable)
    RES = getattr(sys, "_MEIPASS", PROJECT)
    HOME = os.path.join(EXE_DIR, "StudioData")
else:
    RES = HOME = PROJECT

WEB = os.path.join(RES, "web")
LIBRARY = os.path.join(HOME, "characters")
CACHE = os.path.join(HOME, "data")
SETTINGS = os.path.join(HOME, "settings.json")
LOG = os.path.join(HOME, "SC-Character-Studio.log")


def migrate_old_layout():
    """v1.0/v1.1 kept characters/, data/ and settings.json loose next to the .exe: move them in."""
    if not FROZEN:
        return
    os.makedirs(HOME, exist_ok=True)
    for name in ("characters", "data", "settings.json", "SC-Character-Studio.log"):
        old, new = os.path.join(EXE_DIR, name), os.path.join(HOME, name)
        if os.path.exists(old) and not os.path.exists(new):
            try:
                shutil.move(old, new)
            except OSError:
                pass
