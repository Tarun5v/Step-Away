"""Step-Away: a webcam-powered presence guard for your Mac.

Watches for you at the desk, locks the machine when you step away,
and nudges you to stretch after long focus sessions.
"""

import subprocess
import sys
import threading
import time

import cv2
import mediapipe as mp


CAMERA_INDEX = 0
CHECK_INTERVAL_SECONDS = 3
ABSENCE_LOCK_SECONDS = 10
ABSENCE_WARNING_SECONDS = 5


class PresenceMonitor(threading.Thread):
    """Background thread that keeps track of whether anyone sits at the desk."""

    def __init__(self, camera_index: int = CAMERA_INDEX):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self._state_lock = threading.Lock()
        self._face_present = False
        self._running = True
        self.error = None

    @property
    def face_present(self) -> bool:
        with self._state_lock:
            return self._face_present

    def stop(self) -> None:
        self._running = False

    def _open_camera(self):
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open camera {self.camera_index}. "
                "Make sure it exists and is not in use by another app."
            )
        return capture

    def _detect_face(self, detector, frame) -> bool:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = detector.process(image)
        return bool(result.detections)

    def run(self) -> None:
        try:
            capture = self._open_camera()
        except RuntimeError as error:
            self.error = str(error)
            self._running = False
            return

        with mp.solutions.face_detection.FaceDetector(
            model_selection=0, min_detection_confidence=0.5
        ) as detector:
            failed_reads = 0
            while self._running:
                grabbed, frame = capture.read()
                if not grabbed:
                    failed_reads += 1
                    if failed_reads >= 5:
                        self.error = (
                            "Camera stopped delivering frames. "
                            "Check macOS camera permission for your terminal app."
                        )
                        break
                    time.sleep(CHECK_INTERVAL_SECONDS)
                    continue

                failed_reads = 0
                present = self._detect_face(detector, frame)
                changed = present != self.face_present
                with self._state_lock:
                    self._face_present = present

                if changed:
                    state = "arrived" if present else "left"
                    stamp = time.strftime("%H:%M:%S")
                    print(f"[{stamp}] You {state} the desk.")

                time.sleep(CHECK_INTERVAL_SECONDS)

        capture.release()


def lock_screen() -> bool:
    """Lock the Mac with the Cmd+Ctrl+Q shortcut via AppleScript."""
    script = (
        'tell application "System Events" to '
        'keystroke "q" using {command down, control down}'
    )
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:
        print(
            "Could not lock the screen automatically. Grant Accessibility "
            "permission to your terminal app in System Settings > Privacy & Security.",
        )
        return False
    return True


class StepAwayApp:
    """Ties presence monitoring into security and wellness behaviours."""

    def __init__(self):
        self.monitor = PresenceMonitor()
        self.absent_since = None
        self.locked = False
        self.warned = False

    def _handle_presence(self) -> None:
        now = time.time()

        if self.monitor.face_present:
            if self.locked:
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] Welcome back! Security guard re-armed.")
            self.absent_since = None
            self.locked = False
            self.warned = False
            return

        if self.absent_since is None:
            self.absent_since = now
            return

        absence = now - self.absent_since
        if not self.locked:
            if absence >= ABSENCE_LOCK_SECONDS:
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] No one at the desk - locking the screen.")
                lock_screen()
                self.locked = True
            elif absence >= ABSENCE_WARNING_SECONDS and not self.warned:
                remaining = round(ABSENCE_LOCK_SECONDS - absence)
                print(f"Desk empty - locking in {remaining}s. Hop back in view to cancel.")
                self.warned = True

    def run(self) -> int:
        print("Step-Away is starting up. Press Ctrl+C to quit.")
        self.monitor.start()

        try:
            while self.monitor.is_alive():
                self._handle_presence()
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        self.monitor.stop()
        self.monitor.join(timeout=CHECK_INTERVAL_SECONDS * 2)

        if self.monitor.error:
            print(f"Stopped: {self.monitor.error}")
            return 1

        print("Step-Away closed. See you soon!")
        return 0


if __name__ == "__main__":
    sys.exit(StepAwayApp().run())
