#!/usr/bin/env python3

# Import tkinter_fix to apply the ScreenChanged patch at startup
# The module's apply_patch() is called automatically on import
import tkinter_fix  # noqa: F401 - needed for side effects

from modules import core

if __name__ == '__main__':
    core.run()
