#!/bin/zsh
# Build the distributable macOS app bundle into dist/Step Away.app.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
"$VENV" -m PyInstaller \
  --noconfirm --clean \
  --windowed \
  --name "Step Away" \
  --collect-all mediapipe \
  --collect-all cv2 \
  --hidden-import objc \
  --hidden-import Foundation \
  --hidden-import AppKit \
  mac_app.py

echo "Bundle ready: dist/Step Away.app"
