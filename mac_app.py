"""Launcher for the packaged macOS app bundle.

Runs Step-Away with the live preview window on by default so people who
never opened a terminal still get the visible HUD and quit button.
"""
import sys

sys.argv = ["Step Away", "--preview"]

import main

sys.exit(main.cli())
