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

APP="dist/Step Away.app"
PLIST="$APP/Contents/Info.plist"

# Privacy usage descriptions are mandatory: without them macOS aborts the
# process the moment the camera is touched instead of showing a prompt.
PB=/usr/libexec/PlistBuddy
"$PB" -c "Add :NSCameraUsageDescription string 'Step-Away uses the camera to tell when you are at your desk, locks your screen when you leave, and nudges you to stretch and rest your eyes. Video never leaves your Mac.'" "$PLIST" 2>/dev/null || \
  "$PB" -c "Set :NSCameraUsageDescription string 'Step-Away uses the camera to tell when you are at your desk, locks your screen when you leave, and nudges you to stretch and rest your eyes. Video never leaves your Mac.'" "$PLIST"
"$PB" -c "Add :NSAppleEventsUsageDescription string 'Step-Away sends the standard lock-screen keyboard shortcut when it detects you have stepped away.'" "$PLIST" 2>/dev/null || \
  "$PB" -c "Set :NSAppleEventsUsageDescription string 'Step-Away sends the standard lock-screen keyboard shortcut when it detects you have stepped away.'" "$PLIST"

# Editing the plist invalidates the signature, so reseal the bundle.
codesign --force --deep --sign - "$APP"

echo "Bundle ready: $APP"
